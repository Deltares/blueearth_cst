"""Native-cell comparisons at declared spatial locations and observation sites."""

import json
from pathlib import Path

import geopandas as gpd
import hydromt  # noqa: F401 -- registers the raster accessor
import numpy as np
import pandas as pd
import xarray as xr

from blueearth_cst.climate_analysis.diagnostic_settings import carries_temp
from blueearth_cst.climate_analysis.diagnostic_tables import verified_temperature
from blueearth_cst.climate_analysis.diagnostics import reporting_year
from blueearth_cst.climate_analysis.rain_gauges import (
    matched_tables,
    read_precip,
    read_stations,
    read_temperature,
)
from blueearth_cst.shared.forcing_data import validate_canonical_units
from blueearth_cst.shared.forcing_recipe import normalize_forcing_recipe
from blueearth_cst.shared.provenance import file_sha256, toolbox_identity


def location_index(
    locations_path: str, subbasins_path: str, gauges: dict | None
) -> pd.DataFrame:
    """Keep stable spatial IDs, and map gauges only to unambiguous coincident sites.

    Rain gauges use their original coordinates, never river-snapped coordinates.
    Unmatched gauges retain their literal IDs through a separate station token.
    """
    locations = gpd.read_file(locations_path).to_crs(4326)
    subbasins = gpd.read_file(subbasins_path).to_crs(4326)
    rows = []
    for row in locations.itertuples():
        rows.append(
            {
                "id": f"location_{int(row.wflow_id)}",
                "name": row.station_name,
                "kind": row.location_role,
                "x": row.original_x,
                "y": row.original_y,
                "location_code": row.location_code,
                "spatial_location_id": row.location_id,
                "wflow_id": row.wflow_id,
                "subbasin_id": row.subbasin_id,
                "gauge_id": "",
            }
        )
    registry = pd.DataFrame(
        rows,
        columns=[
            "id",
            "name",
            "kind",
            "x",
            "y",
            "location_code",
            "spatial_location_id",
            "wflow_id",
            "subbasin_id",
            "gauge_id",
        ],
    )
    if registry.id.duplicated().any():
        raise ValueError("Spatial comparison location IDs must be unique")
    if gauges:
        for row in read_stations(gauges["stations_path"]).itertuples():
            matches = registry.index[
                np.isclose(registry.x.astype(float), row.x, rtol=0, atol=1e-8)
                & np.isclose(registry.y.astype(float), row.y, rtol=0, atol=1e-8)
                & registry.gauge_id.eq("")
            ]
            if len(matches) == 1:
                registry.loc[matches[0], "gauge_id"] = row.id
                registry.loc[matches[0], ["x", "y"]] = [row.x, row.y]
                continue
            point = gpd.points_from_xy([row.x], [row.y])[0]
            containing = subbasins.loc[subbasins.geometry.covers(point), "subbasin_id"]
            registry.loc[len(registry)] = {
                "id": f"station_{row.token}",
                "name": getattr(row, "name", "") or row.id,
                "kind": "observation",
                "x": row.x,
                "y": row.y,
                "location_code": "",
                "spatial_location_id": "",
                "wflow_id": "",
                "subbasin_id": containing.iloc[0] if len(containing) == 1 else "",
                "gauge_id": row.id,
            }
    return registry.sort_values("id").reset_index(drop=True)


def temperature_tables(daily: pd.DataFrame, m0: int) -> dict[str, pd.DataFrame]:
    """Complete matched monthly and reporting-year mean temperatures, without infill."""
    joint = np.isfinite(daily).all(axis=1)
    days = daily.index - pd.Timedelta(days=1)
    result = {}
    for name, groups in (
        ("monthly", days.to_period("M")),
        ("annual", reporting_year(days, m0)),
    ):
        rows = []
        for group in pd.unique(groups):
            mask = groups == group
            expected = (
                group.days_in_month
                if name == "monthly"
                else (
                    pd.Timestamp(year=int(group) + 1, month=m0, day=1)
                    - pd.Timestamp(year=int(group), month=m0, day=1)
                ).days
            )
            complete = mask.sum() == expected and joint[mask].all()
            for source in daily:
                rows.append(
                    {
                        "period": str(group),
                        "calendar_month": group.month if name == "monthly" else 0,
                        "source": source,
                        "expected_days": expected,
                        "paired_days": int(joint[mask].sum()),
                        "mean_degC": daily.loc[mask, source].mean()
                        if complete
                        else np.nan,
                        "reason": "" if complete else "incomplete_period",
                    }
                )
        result[name] = pd.DataFrame(rows)
    return result


def monthly_agreement(table: pd.DataFrame, value: str) -> pd.DataFrame:
    """Metrics for exactly the complete paired months shown in agreement panels."""
    wide = table.pivot(index="period", columns="source", values=value)
    rows = []
    for source in wide.columns.drop("observed"):
        paired = wide[["observed", source]].dropna()
        errors = paired[source] - paired.observed
        enough = (
            len(paired) >= 2
            and paired.observed.nunique() > 1
            and paired[source].nunique() > 1
        )
        rows.append(
            {
                "source": source,
                "paired_months": len(paired),
                "pearson_r": paired.observed.corr(paired[source]) if enough else np.nan,
                "bias": errors.mean(),
                "rmse": np.sqrt((errors**2).mean()),
                "reason": "" if enough else "too_few_pairs_or_constant",
            }
        )
    return pd.DataFrame(rows)


def sample_source(
    path: str,
    registry: pd.DataFrame,
    axis: pd.DatetimeIndex,
    source: str,
    data_sources=None,
    *,
    recipe=None,
) -> tuple[dict, list, list]:
    """Sample containing native cells; record outside-grid and nodata exclusions."""
    points = gpd.GeoDataFrame(
        registry.copy(), geometry=gpd.points_from_xy(registry.x, registry.y), crs=4326
    )
    samples, cells, exclusions = {}, [], []
    with xr.open_dataset(path) as ds:
        times = pd.DatetimeIndex(ds.time.values)
        if times.has_duplicates or times.hasnans or not times.equals(times.normalize()):
            raise ValueError(
                f"{source}: expected unique midnight interval-end timestamps"
            )
        for variable in ("precip", "temp"):
            if variable == "temp" and not carries_temp(source):
                continue
            if variable not in ds:
                if variable == "precip":
                    raise ValueError(f"{source}: missing precipitation")
                continue
            checked = ds[[variable]].copy(deep=False)
            checked[variable].attrs = dict(ds[variable].attrs)
            if variable == "precip" and checked[variable].attrs.get("units") == "mm":
                checked[variable].attrs["units"] = "mm/day"
            if variable == "precip":
                validate_canonical_units(checked, [variable], source=source)
            field = ds[variable]
            if field.raster.crs is None:
                raise ValueError(f"{source}.{variable}: source raster has no CRS")
            native = points.to_crs(field.raster.crs)
            rows, cols = field.raster.rowcol(
                native.geometry.x.values,
                native.geometry.y.values,
                mask_outside=True,
                nodata=-1,
            )
            values = np.full((len(axis), len(registry)), np.nan)
            eligible = np.flatnonzero(
                (rows >= 0) & (cols >= 0) & registry.inside_basin.to_numpy()
            )
            sampled = (
                field.raster.sample(native.iloc[eligible]) if len(eligible) else None
            )
            for i, position in enumerate(eligible):
                note = ""
                series = sampled.isel(index=i).to_series()
                if field.raster.nodata is not None:
                    series = series.mask(series == field.raster.nodata)
                series = series.where(np.isfinite(series))
                if variable == "precip":
                    series = series.where(series >= 0)
                else:
                    verified, note = verified_temperature(
                        series,
                        field.attrs,
                        ds.attrs,
                        source,
                        data_sources,
                        recipe=recipe,
                    )
                    if not verified:
                        series[:] = np.nan
                    if note and not verified:
                        exclusions.append(
                            {
                                "id": registry.iloc[position].id,
                                "source": source,
                                "variable": variable,
                                "reason": note,
                            }
                        )
                values[:, position] = series.reindex(axis)
                cells.append(
                    {
                        "id": registry.iloc[position].id,
                        "source": source,
                        "variable": variable,
                        "cell_x": float(field[field.raster.x_dim][cols[position]]),
                        "cell_y": float(field[field.raster.y_dim][rows[position]]),
                        "cell_crs": str(field.raster.crs),
                        "data_source": recipe.temp_pet_source
                        if variable == "temp" and recipe
                        else source,
                        "units_note": note,
                    }
                )
            for position, location in registry.iterrows():
                reason = (
                    "outside_basin"
                    if not location.inside_basin
                    else "outside_source_grid"
                    if rows[position] < 0 or cols[position] < 0
                    else "sampled_cell_nodata_in_period"
                    if not np.isfinite(values[:, position]).any()
                    else ""
                )
                if reason:
                    exclusions.append(
                        {
                            "id": location.id,
                            "source": source,
                            "variable": variable,
                            "reason": reason,
                        }
                    )
            samples[variable] = values
    return samples, cells, exclusions


def evaluate(
    root: str | Path,
    sources: dict[str, str],
    locations_path: str,
    subbasins_path: str,
    region_path: str,
    window: dict,
    comparison: dict,
    m0: int,
    gauges: dict | None = None,
    data_sources=None,
    forcing: dict | None = None,
) -> None:
    """Retain all locations, matched summaries, native samples and provenance."""
    root = Path(root)
    engine = root / "_engine/locations"
    engine.mkdir(parents=True, exist_ok=True)
    registry = location_index(locations_path, subbasins_path, gauges)
    if registry.empty:
        raise ValueError(
            "Location comparison requires at least one spatial location or gauge"
        )
    basin = gpd.read_file(region_path).to_crs(4326).geometry.union_all()
    registry["inside_basin"] = [
        basin.covers(point) for point in gpd.points_from_xy(registry.x, registry.y)
    ]
    start = max(
        pd.Timestamp(year=window["start"], month=1, day=1),
        pd.Timestamp(year=comparison["start"], month=m0, day=1),
    )
    end = min(
        pd.Timestamp(year=window["end"] + 1, month=1, day=1),
        pd.Timestamp(year=comparison["end"] + 1, month=m0, day=1),
    )
    if start >= end:
        raise ValueError("Location comparison and extraction windows do not overlap")
    axis = pd.date_range(start + pd.Timedelta(days=1), end, name="time")
    observations = {}
    if gauges:
        observations["precip"] = read_precip(
            gauges["precip_path"], gauges["time_label"]
        )
        if "temp_path" in gauges:
            observations["temp"] = read_temperature(
                gauges["temp_path"], gauges["time_label"]
            )
    labels = [*(["observed"] if observations else []), *sources]
    values = {
        v: np.full((len(axis), len(registry), len(labels)), np.nan)
        for v in ("precip", "temp")
    }
    exclusions, cells, temperature_sources = [], [], []
    for variable in values:
        observed = observations.get(variable, pd.DataFrame())
        registry[f"observed_{variable}"] = registry.gauge_id.isin(observed.columns)
        for position, row in registry.iterrows():
            if row[f"observed_{variable}"] and row.inside_basin:
                values[variable][:, position, 0] = observed[row.gauge_id].reindex(axis)
            elif row.gauge_id and variable in observations:
                exclusions.append(
                    {
                        "id": row.id,
                        "source": "observed",
                        "variable": variable,
                        "reason": "outside_basin"
                        if not row.inside_basin
                        else "no_observation_column",
                    }
                )
        for station in observed.columns.difference(registry.gauge_id):
            exclusions.append(
                {
                    "id": station,
                    "source": "observed",
                    "variable": variable,
                    "reason": "unknown_station_id",
                }
            )
    recipes = {
        source: normalize_forcing_recipe(source, (forcing or {}).get(source))
        for source in sources
    }
    for source, path in sources.items():
        samples, native_cells, rejected = sample_source(
            path, registry, axis, source, data_sources, recipe=recipes[source]
        )
        cells.extend(native_cells)
        exclusions.extend(rejected)
        for variable, data in samples.items():
            values[variable][:, :, labels.index(source)] = data
        if "temp" in samples:
            temperature_sources.append(source)
    ds = xr.Dataset(
        {v: (("time", "location", "source"), data) for v, data in values.items()},
        coords={"time": axis, "location": registry.id.to_list(), "source": labels},
    )
    ds.precip.attrs.update(units="mm/day", time_label="interval_end")
    ds.temp.attrs.update(units="degC", time_label="interval_end")
    ds.to_netcdf(engine / "daily.nc")
    for position, row in registry.iterrows():
        target = root / "tables/locations" / row.id
        target.mkdir(parents=True, exist_ok=True)
        precip_labels = [*(["observed"] if row.observed_precip else []), *sources]
        frame = ds.precip.sel(location=row.id, source=precip_labels).to_pandas()
        precip_tables = matched_tables(frame, m0, observed=bool(row.observed_precip))
        registry.loc[position, "common_precip_days"] = int(
            np.isfinite(frame).all(axis=1).sum()
        )
        registry.loc[position, "observed_precip_days"] = (
            int(frame.observed.notna().sum()) if row.observed_precip else 0
        )
        for name, table in precip_tables.items():
            table.to_csv(target / f"{name}.csv", index=False)
        if row.observed_precip:
            monthly_agreement(
                precip_tables["monthly"].rename(columns={"month": "period"}), "total_mm"
            ).to_csv(target / "monthly_agreement.csv", index=False)
        temp_labels = [
            *(["observed"] if row.observed_temp else []),
            *temperature_sources,
        ]
        if temp_labels:
            frame = ds.temp.sel(location=row.id, source=temp_labels).to_pandas()
            temp_tables = temperature_tables(frame, m0)
            for name, table in temp_tables.items():
                table.to_csv(target / f"temp_{name}.csv", index=False)
            if row.observed_temp:
                monthly_agreement(temp_tables["monthly"], "mean_degC").to_csv(
                    target / "temp_monthly_agreement.csv", index=False
                )
        pd.DataFrame(
            [cell for cell in cells if cell["id"] == row.id],
            columns=[
                "id",
                "source",
                "variable",
                "cell_x",
                "cell_y",
                "cell_crs",
                "data_source",
                "units_note",
            ],
        ).to_csv(target / "native_cells.csv", index=False)
        pd.DataFrame(
            [item for item in exclusions if item["id"] == row.id],
            columns=["id", "source", "variable", "reason"],
        ).to_csv(target / "exclusions.csv", index=False)
    registry.to_csv(root / "locations.csv", index=False)
    pd.DataFrame(exclusions, columns=["id", "source", "variable", "reason"]).to_csv(
        engine / "exclusions.csv", index=False
    )
    inputs = [*sources.values(), locations_path, subbasins_path, region_path]
    inputs.extend(
        [data_sources] if isinstance(data_sources, str) else data_sources or []
    )
    if gauges:
        inputs.extend(
            gauges[key]
            for key in ("stations_path", "precip_path", "temp_path")
            if key in gauges
        )
    record = {
        "schema_version": 1,
        "sources": list(sources),
        "temperature_sources": temperature_sources,
        "temperature_labels": {
            source: recipes[source].temp_pet_source for source in temperature_sources
        },
        "forcing_recipes": {
            source: recipe.to_dict() for source, recipe in recipes.items()
        },
        "water_year_start_month": m0,
        "comparison_period": comparison,
        "window": window,
        "gauges": gauges,
        "input_sha256": {str(path): file_sha256(path) for path in inputs},
        "sampling": "HydroMT raster.sample(wdw=0); native containing cell at original coordinates",
        "calendar_attribution": "midnight UTC interval end minus one day",
        "gap_policy": "no infill; complete matched months and reporting years within each location and variable",
        "observation_mapping": "unique coincident original coordinates within 1e-8 degrees; otherwise separate station ID",
        "software": toolbox_identity(),
        "implementation_sha256": {
            name: file_sha256(Path(__file__).with_name(name))
            for name in ("location_comparison.py", "rain_gauges.py")
        },
        "exclusions": exclusions,
    }
    (engine / "evaluation.json").write_text(
        json.dumps(record, indent=2) + "\n", encoding="utf-8"
    )


if "snakemake" in globals():
    from blueearth_cst.shared.snake_utils import tee_to_log

    sm = globals()["snakemake"]
    with tee_to_log(sm.log[0]):
        evaluate(
            sm.params.root,
            dict(zip(sm.params.sources, sm.input.climate, strict=True)),
            sm.input.locations,
            sm.input.subbasins,
            sm.input.region,
            dict(sm.params.window),
            dict(sm.params.comparison),
            int(sm.params.m0),
            sm.params.gauges,
            sm.params.data_sources,
            dict(sm.params.forcing),
        )
