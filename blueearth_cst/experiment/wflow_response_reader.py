"""Terminate Wflow column names and model metadata at the neutral response seam."""

import csv
import tomllib
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from blueearth_cst.experiment.response_series import ResponseSeries, validate_responses
from blueearth_cst.shared.native_locations import native_location_labels


@dataclass(frozen=True)
class NativeRunArtifacts:
    """Explicit native output association; neither path encodes run identity."""

    csv_path: Path
    toml_path: Path
    temporal_path: Path | None = None


@dataclass(frozen=True)
class ResponseRequest:
    """Canonical variables to retain at all their declared native locations."""

    variables: tuple[str, ...]


# Wflow 1.0.2 src/standard_name.jl. Depth fluxes are per MODEL timestep;
# retaining `mm dt-1` avoids silently treating every possible model step as a day.
# Headers are the current CST setup_gauges_and_outputs binding. A matching
# header alone is insufficient: open_responses also checks the TOML parameter.
NATIVE_VARIABLES = {
    "q": ("Q", "river_water__volume_flow_rate", "m3 s-1"),
    "precip": ("p", "atmosphere_water__precipitation_volume_flux", "mm dt-1"),
    "pet": (
        "pet",
        "land_surface_water__potential_evaporation_volume_flux",
        "mm dt-1",
    ),
    "aet": ("aet", "land_surface__evapotranspiration_volume_flux", "mm dt-1"),
    "gwr": (
        "gwr",
        "soil_water_saturated_zone_top__net_recharge_volume_flux",
        "mm dt-1",
    ),
    "overland_flow": ("qof", "land_surface_water__volume_flow_rate", "m3 s-1"),
    "snow": ("swe", "snowpack_liquid_water__depth", "mm"),
    "temp": ("temp", "atmosphere_air__temperature", "degC"),
}


def open_responses(
    run_id: str,
    native_artifacts: NativeRunArtifacts,
    response_request: ResponseRequest,
) -> tuple[ResponseSeries, ...]:
    """Read requested native responses without conversions or path parsing.

    The TOML supplies calendar/timestep and verifies each physical parameter.
    Native CSV order supplies location ordinals. Wflow's run_timestep! advances
    the clock, updates the model and then writes at that clock time, hence the
    explicit interval-end binding. No spin-up trimming is introduced here.
    """
    requested = response_request.variables
    if not run_id or not requested or len(set(requested)) != len(requested):
        raise ValueError("response request needs a run id and unique variables")
    unknown = set(requested) - NATIVE_VARIABLES.keys()
    if unknown:
        raise ValueError(
            f"run_id={run_id}: unknown response variables {sorted(unknown)}"
        )
    with native_artifacts.toml_path.open("rb") as stream:
        config = tomllib.load(stream)
    time_config = config["time"]
    calendar = time_config["calendar"]
    step = timedelta(seconds=time_config["timestepsecs"])
    columns = config["output"]["csv"]["column"]
    with native_artifacts.csv_path.open(encoding="utf-8", newline="") as stream:
        header = next(csv.reader(stream))
    if len(set(header)) != len(header):
        raise ValueError(f"run_id={run_id}: duplicate native CSV columns")
    # Every mapped column is labelled by its wflow_id, and same-cell duplicates
    # (a gauge on the outlet cell) dropped, BEFORE ordinals are assigned, so
    # every reader agrees with the planned request.
    static = config.get("input", {}).get("path_static")
    static_path = (
        native_artifacts.toml_path.parent / config.get("dir_input", ".") / static
        if static
        else None
    )
    labels = (
        native_location_labels(columns, header, static_path)
        if static_path is not None and static_path.is_file()
        else None
    )
    frame = pd.read_csv(native_artifacts.csv_path, index_col=0, parse_dates=True)
    expected_first = pd.Timestamp(time_config["starttime"]) + step
    expected_last = pd.Timestamp(time_config["endtime"])
    if (
        frame.empty
        or frame.index[0] != expected_first
        or frame.index[-1] != expected_last
    ):
        raise ValueError(
            f"run_id={run_id}: response coverage must be {expected_first} through {expected_last}"
        )
    result = []
    for variable in requested:
        native_header, parameter, units = NATIVE_VARIABLES[variable]
        declarations = [item for item in columns if item.get("header") == native_header]
        if not declarations or any(
            item.get("parameter") != parameter for item in declarations
        ):
            raise ValueError(
                f"run_id={run_id}: {variable} native parameter must be {parameter}"
            )
        if variable in ("pet", "temp") and any(
            item.get("map") != "subcatchment" or item.get("reducer") != "mean"
            for item in declarations
        ):
            raise ValueError(
                f"run_id={run_id}: {variable} native output must be a subcatchment mean"
            )
        prefix = native_header + "_"
        locations = [
            (column, column[len(prefix) :] if labels is None else labels[column])
            for column in frame.columns
            if column.startswith(prefix) and (labels is None or column in labels)
        ]
        if not locations or any(not location for _, location in locations):
            raise ValueError(f"run_id={run_id}: missing requested response {variable}")
        for ordinal, (column, location) in enumerate(locations):
            values = frame[column].to_numpy(dtype="float64")
            result.append(
                ResponseSeries(
                    run_id=run_id,
                    variable=variable,
                    location_id=location,
                    location_ordinal=ordinal,
                    time=frame.index,
                    calendar=calendar,
                    timestep=step,
                    time_label="interval_end",
                    units=units,
                    values=values,
                    missing=np.isnan(values),
                )
            )
    responses = tuple(result)
    validate_responses(responses, evaluated_runs={run_id})
    return responses
