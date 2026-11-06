"""WF1 observed water-balance adapter; no simulation output is consumed."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from hydromt.gis import flw

from blueearth_cst.model.budyko import (
    budyko_settings,
    discharge_depth,
    matched_water_balance,
    read_daily_discharge,
)
from blueearth_cst.model.observation_validation import validate_observation_station_ids
from blueearth_cst.spatial.delineation import full_catchment
from blueearth_cst.spatial.products import read_hydrography_seam


def _sha256(path: str) -> str:
    """Hash a retained input with bounded memory and a closed handle."""
    with open(path, "rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def evaluate(
    hydrography_path: str,
    registry_path: str,
    forcing_path: str,
    recipe_path: str,
    observations_path: str,
    root: str,
    settings: dict,
    start: str,
    end: str,
    water_year_start: int,
) -> None:
    """Prepare complete matched totals on original snapped upstream supports.

    Prepared model forcing is sampled by nearest model cell at each original
    hydrography cell centre. Hydrography cell areas weight both meteorological
    means and discharge-to-depth conversion, preserving one contributing support.
    Missing spatial coverage makes the corresponding daily mean unavailable.
    """
    declaration = budyko_settings(settings, observations_path)
    if declaration is None:
        raise ValueError("Budyko evaluation requires an enabled declaration")
    registry = validate_observation_station_ids(observations_path, registry_path)
    observed = read_daily_discharge(observations_path, declaration["time_label"])
    summaries, annuals, stations, daily_frames = [], [], [], []
    with (
        read_hydrography_seam(hydrography_path) as maps,
        xr.open_dataset(forcing_path) as forcing,
    ):
        for variable in ("precip", "pet"):
            unit = forcing[variable].attrs.get(
                "unit", forcing[variable].attrs.get("units")
            )
            if unit not in ("mm", "mm/day", "mm d**-1", "mm d-1"):
                raise ValueError(
                    f"prepared forcing {variable} must be daily mm, got {unit!r}"
                )
        network = flw.flwdir_from_da(maps["flow_direction"], ftype="d8")
        spatial_dims = maps.raster.dims
        forcing_index = pd.DatetimeIndex(forcing.time.values)
        if forcing_index.has_duplicates or not forcing_index.equals(
            forcing_index.normalize()
        ):
            raise ValueError("prepared forcing must have unique midnight interval ends")
        for station in registry.itertuples():
            key = str(station.wflow_id)
            if key not in observed:
                continue
            outlet = np.ravel_multi_index(
                (int(station.snapped_row), int(station.snapped_col)), network.shape
            )
            mask = full_catchment(network, int(outlet))
            area = maps["cell_area"].where(mask)
            cell_values = area.values[mask]
            if not np.isfinite(cell_values).all() or (cell_values <= 0).any():
                raise ValueError(f"station {key}: invalid contributing cell areas")
            area_m2 = float(cell_values.sum()) * 1e6
            fields = {}
            for variable in ("precip", "pet"):
                field = forcing[variable].raster.reproject_like(maps, method="nearest")
                valid = field.where(mask).notnull().where(mask, True).all(spatial_dims)
                fields[variable] = (
                    (
                        (field.where(mask) * area).sum(spatial_dims, skipna=True)
                        / float(cell_values.sum())
                    )
                    .where(valid)
                    .to_pandas()
                )
            fields["runoff_mm"] = discharge_depth(observed[key], area_m2)
            daily = pd.DataFrame(fields)
            annual, summary = matched_water_balance(daily, start, end, water_year_start)
            annual.insert(0, "wflow_id", station.wflow_id)
            annuals.append(annual)
            summaries.append(
                {
                    "wflow_id": station.wflow_id,
                    "station_name": station.station_name,
                    **summary,
                }
            )
            stations.append(
                {
                    "wflow_id": station.wflow_id,
                    "station_name": station.station_name,
                    "area_m2": area_m2,
                    "snapped_row": station.snapped_row,
                    "snapped_col": station.snapped_col,
                    "cell_count": int(mask.sum()),
                    "mask_sha256": hashlib.sha256(mask.tobytes()).hexdigest(),
                }
            )
            daily.index.name = "time"
            daily.insert(0, "wflow_id", station.wflow_id)
            daily_frames.append(daily.reset_index())
    if not stations:
        raise ValueError("Budyko requires at least one observed station")
    destination = Path(root)
    (destination / "tables").mkdir(parents=True, exist_ok=True)
    (destination / "_engine").mkdir(parents=True, exist_ok=True)
    pd.DataFrame(stations).to_csv(destination / "tables/stations.csv", index=False)
    pd.concat(annuals, ignore_index=True).to_csv(
        destination / "tables/annual.csv", index=False
    )
    pd.DataFrame(summaries).to_csv(destination / "tables/summary.csv", index=False)
    pd.concat(daily_frames, ignore_index=True).to_csv(
        destination / "_engine/daily.csv", index=False
    )
    paths = {
        "observations": observations_path,
        "registry": registry_path,
        "forcing": forcing_path,
        "hydrography": hydrography_path,
        "forcing_recipe": recipe_path,
    }
    record = {
        "observation_support": {
            "declaration": declaration,
            "status": "user_declared",
            "timezone": "UTC",
            "label_shift_days": int(declaration["time_label"] == "interval_start"),
        },
        "files": {
            name: {
                "path": str(path),
                "sha256": _sha256(path),
            }
            for name, path in paths.items()
        },
        "window": {"start": start, "end_exclusive": end},
        "water_year_start": water_year_start,
        "spatial_method": "nearest model forcing cell at hydrography cell centres, weighted by hydrography cell_area",
        "temporal_method": "complete matched water years; ratios of totals; no infilling or spinup exclusion",
    }
    (destination / "_engine/methods.json").write_text(
        json.dumps(record, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    sm = globals()["snakemake"]
    from blueearth_cst.model.plot_budyko import render
    from blueearth_cst.shared.snake_utils import tee_to_log

    with tee_to_log(sm.log[0]):
        evaluate(
            **{
                key: str(value)
                for key, value in sm.input.items()
                if key != "model_final"
            },
            root=sm.params.root,
            settings=sm.params.settings,
            start=sm.params.start,
            end=sm.params.end,
            water_year_start=sm.params.water_year_start,
        )
        render(sm.params.root, annotation=sm.params.annotation)
