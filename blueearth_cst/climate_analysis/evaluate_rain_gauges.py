"""Sample native precipitation cells and retain observed-only station evidence."""

import json
from importlib.metadata import version
from pathlib import Path

import geopandas as gpd
import hydromt  # noqa: F401 -- registers the raster accessor
import numpy as np
import pandas as pd
import xarray as xr
from pyproj import Geod

from blueearth_cst.climate_analysis.diagnostic_outputs import rain_gauge_outputs
from blueearth_cst.climate_analysis.rain_gauges import (
    matched_tables,
    read_precip,
    read_stations,
)
from blueearth_cst.shared.forcing_data import validate_canonical_units
from blueearth_cst.shared.provenance import file_sha256, toolbox_identity


def evaluate(
    settings: dict,
    sources: dict[str, str],
    region_path: str,
    root: str,
    window: dict,
    comparison: dict,
    m0: int,
) -> None:
    """Write checked station samples, common masks, tables and input provenance."""
    root = Path(root)
    for folder in ("tables", "_engine"):
        (root / folder).mkdir(parents=True, exist_ok=True)
    stations = read_stations(settings["stations_path"])
    observed = read_precip(settings["precip_path"], settings["time_label"])
    basin = gpd.read_file(region_path).to_crs(4326).geometry.union_all()
    points = gpd.GeoDataFrame(
        stations.copy(), geometry=gpd.points_from_xy(stations.x, stations.y), crs=4326
    )
    # Comparison years are reporting-year START labels; extraction window is calendar years.
    start = max(
        pd.Timestamp(year=window["start"], month=1, day=1),
        pd.Timestamp(year=comparison["start"], month=m0, day=1),
    )
    end = min(
        pd.Timestamp(year=window["end"] + 1, month=1, day=1),
        pd.Timestamp(year=comparison["end"] + 1, month=m0, day=1),
    )
    if start >= end:
        raise ValueError("Rain-gauge comparison and extraction windows do not overlap")
    axis = pd.date_range(start + pd.Timedelta(days=1), end, freq="D", name="time")
    exclusions = []
    for station in observed.columns.difference(stations.index):
        exclusions.append(
            {"id": station, "source": "observed", "reason": "unknown_station_id"}
        )
    for station in stations.index.difference(observed.columns):
        exclusions.append(
            {"id": station, "source": "observed", "reason": "no_observation_column"}
        )
    accepted = []
    for station, point in points.geometry.items():
        if not basin.covers(point):
            exclusions.append(
                {"id": station, "source": "all", "reason": "outside_basin"}
            )
        elif station in observed:
            accepted.append(station)
    labels = ["observed", *sources]
    values = np.full((len(axis), len(accepted), len(labels)), np.nan)
    for j, station in enumerate(accepted):
        values[:, j, 0] = observed[station].reindex(axis)
    cell_rows = []
    for source_index, (source, path) in enumerate(sources.items(), 1):
        with xr.open_dataset(path) as ds:
            field = ds.precip
            times = pd.DatetimeIndex(field.time.values)
            if (
                times.has_duplicates
                or times.hasnans
                or not times.equals(times.normalize())
            ):
                raise ValueError(
                    f"{source}: expected unique midnight interval-end timestamps"
                )
            # Reviewed historical stores may retain HydroMT's daily-depth mm label.
            # All explicit rate labels use the same vocabulary as forcing admission.
            unit_check = ds[["precip"]].copy(deep=False)
            unit_check.precip.attrs = dict(field.attrs)
            if unit_check.precip.attrs.get("units") == "mm":
                unit_check.precip.attrs["units"] = "mm/day"
            validate_canonical_units(unit_check, ["precip"], source=source)
            if field.raster.crs is None:
                raise ValueError(f"{source}: source raster has no CRS")
            native = points.loc[accepted].to_crs(field.raster.crs)
            rows, cols = field.raster.rowcol(
                native.geometry.x.values,
                native.geometry.y.values,
                mask_outside=True,
                nodata=-1,
            )
            if not accepted:
                continue
            sampled = field.raster.sample(native)
            for j, station in enumerate(accepted):
                if rows[j] < 0 or cols[j] < 0:
                    exclusions.append(
                        {
                            "id": station,
                            "source": source,
                            "reason": "outside_source_grid",
                        }
                    )
                    continue
                series = sampled.isel(index=j).to_series()
                nodata = field.raster.nodata
                if nodata is not None:
                    series = series.mask(series == nodata)
                series = series.where(np.isfinite(series) & (series >= 0))
                values[:, j, source_index] = series.reindex(axis)
                cx = float(field[field.raster.x_dim][cols[j]])
                cy = float(field[field.raster.y_dim][rows[j]])
                cell = (
                    gpd.GeoSeries(gpd.points_from_xy([cx], [cy]), crs=field.raster.crs)
                    .to_crs(4326)
                    .iloc[0]
                )
                distance = Geod(ellps="WGS84").inv(
                    stations.loc[station, "x"],
                    stations.loc[station, "y"],
                    cell.x,
                    cell.y,
                )[2]
                cell_rows.append(
                    {
                        **stations.loc[station].to_dict(),
                        "source": source,
                        "cell_x": cx,
                        "cell_y": cy,
                        "cell_crs": str(field.raster.crs),
                        "distance_m": distance,
                    }
                )
                if not np.isfinite(values[:, j, source_index]).any():
                    exclusions.append(
                        {
                            "id": station,
                            "source": source,
                            "reason": "sampled_cell_nodata_in_period",
                        }
                    )
    valid = np.isfinite(values) & (values >= 0)
    daily_ds = xr.Dataset(
        {
            "precip": (("time", "station", "source"), values),
            "valid": (("time", "station", "source"), valid.astype("int8")),
            "joint_valid": (("time", "station"), valid.all(axis=2).astype("int8")),
        },
        coords={"time": axis, "station": accepted, "source": labels},
    )
    daily_ds.precip.attrs.update(units="mm/day", time_label="interval_end")
    daily_ds.attrs["calendar_attribution"] = "interval_end minus one day; midnight UTC"
    daily_ds.to_netcdf(root / "_engine/station_daily.nc")
    collected = {name: [] for name in ("coverage", "agreement", "monthly", "annual")}
    for j, station in enumerate(accepted):
        tables = matched_tables(
            pd.DataFrame(values[:, j, :], index=axis, columns=labels), m0
        )
        for name, table in tables.items():
            table.insert(0, "id", station)
            collected[name].append(table)
    # Header-only products remain readable when no station can be compared.
    if not accepted:
        empty_tables = matched_tables(
            pd.DataFrame(np.nan, index=axis, columns=labels), m0
        )
        for table in empty_tables.values():
            table.insert(0, "id", "")
    for name, frames in collected.items():
        table = (
            pd.concat(frames, ignore_index=True)
            if frames
            else empty_tables[name].iloc[:0]
        )
        table.to_csv(root / f"tables/{name}.csv", index=False)
    station_table = (
        pd.DataFrame(cell_rows)
        if cell_rows
        else stations.reset_index(drop=True).assign(
            source="", cell_x=np.nan, cell_y=np.nan, cell_crs="", distance_m=np.nan
        )
    )
    # Retain excluded metadata too, including tokens needed by the explicit plot inventory.
    missing = stations.loc[~stations.index.isin(station_table.id)].reset_index(
        drop=True
    )
    station_table = pd.concat([station_table, missing], ignore_index=True)
    station_table.to_csv(root / "tables/stations.csv", index=False)
    pd.DataFrame(exclusions, columns=["id", "source", "reason"]).to_csv(
        root / "tables/exclusions.csv", index=False
    )
    record = {
        "schema_version": 1,
        "settings": settings,
        "window": window,
        "comparison_period": comparison,
        "water_year_start_month": m0,
        "time_label": "interval_end",
        "calendar_attribution": "end minus one day",
        "observation_label_shift_days": int(settings["time_label"] == "interval_start"),
        "sampling": "HydroMT raster.sample(wdw=0); no interpolation or nearest-valid search",
        "gap_policy": "observed only; common valid dates across all sources",
        "input_sha256": {
            str(path): file_sha256(path)
            for path in [
                *sources.values(),
                region_path,
                settings["stations_path"],
                settings["precip_path"],
            ]
        },
        "sources": sources,
        "software": toolbox_identity(),
        "library_versions": {
            name: version(name) for name in ("hydromt", "numpy", "pandas", "xarray")
        },
        "implementation_sha256": {
            name: file_sha256(Path(__file__).with_name(name))
            for name in (
                "rain_gauges.py",
                "evaluate_rain_gauges.py",
                "diagnostic_outputs.py",
            )
        },
        "expected_interval_end_period": [axis[0].isoformat(), axis[-1].isoformat()],
        "exclusions": exclusions,
        "comparable_stations": int(valid.all(axis=2).any(axis=0).sum()),
        "output_base": "rain_gauges root",
        "outputs": [
            Path(path).relative_to(root).as_posix()
            for group in rain_gauge_outputs(root, stations.token).values()
            for path in group.values()
        ],
    }
    (root / "_engine/evaluation.json").write_text(
        json.dumps(record, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    sm = globals()["snakemake"]
    from blueearth_cst.shared.snake_utils import tee_to_log

    with tee_to_log(sm.log[0]):
        evaluate(
            dict(sm.params.settings),
            dict(zip(sm.params.sources, sm.input.climate)),
            sm.input.region,
            sm.params.root,
            dict(sm.params.window),
            dict(sm.params.comparison),
            sm.params.m0,
        )
