"""Verified, rounded scalar response tables; native CSVs are conversion inputs."""

import csv
import json
import os
import tempfile
import tomllib
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from blueearth_cst.experiment.response_series import ResponseSeries, validate_responses
from blueearth_cst.experiment.wflow_response_reader import (
    NATIVE_VARIABLES,
    NativeRunArtifacts,
    ResponseRequest,
    open_responses,
)
from blueearth_cst.shared.csv_output import normalize_csv_output
from blueearth_cst.shared.provenance import file_sha256
from blueearth_cst.shared.tidy_wflow_table import _format_value

SCHEMA = "response-tables/1"


def read_table_record(record_path: str | Path) -> dict:
    """Verify every declared table, including variables not selected by a consumer."""
    record_path = Path(record_path)
    root = record_path.parent.parent.resolve()
    record = json.loads(record_path.read_text(encoding="utf-8"))
    if record.get("schema_version") != SCHEMA:
        raise ValueError(
            "Unsupported response tables; regenerate using the current format"
        )
    normalize_csv_output({"csv": {"decimal_places": record["decimal_places"]}})
    variables = record.get("variables")
    if not isinstance(variables, dict) or not variables:
        raise ValueError("Response tables require variable metadata")
    for variable, item in variables.items():
        if (
            variable not in NATIVE_VARIABLES
            or item["units"] != NATIVE_VARIABLES[variable][2]
        ):
            raise ValueError(f"Invalid response variable or units: {variable}")
        path = root / item["path"]
        if path.is_symlink() or not path.resolve().is_relative_to(root):
            raise ValueError("Response table escapes its output directory")
        if file_sha256(path) != item["sha256"]:
            raise ValueError(f"Response table checksum mismatch: {variable}")
        locations = item["locations"]
        if not locations or len(set(locations)) != len(locations):
            raise ValueError(f"Response table locations must be unique: {variable}")
    return record


def open_table_responses(
    run_id: str, record_path: str | Path, request: ResponseRequest
) -> tuple[ResponseSeries, ...]:
    """Read the retained tables as neutral responses without a model or raw CSV."""
    record_path = Path(record_path)
    record = read_table_record(record_path)
    if not request.variables or len(set(request.variables)) != len(request.variables):
        raise ValueError("Response request requires unique variables")
    missing = set(request.variables) - set(record["variables"])
    if missing:
        raise ValueError(f"Missing retained response variables: {sorted(missing)}")
    temporal = record["temporal"]
    step = timedelta(seconds=temporal["timestep_seconds"])
    result = []
    for variable in request.variables:
        item = record["variables"][variable]
        path = record_path.parent.parent / item["path"]
        with path.open(encoding="utf-8", newline="") as stream:
            header = next(csv.reader(stream))
        if len(set(header)) != len(header) or set(header) != {
            "time",
            *item["locations"],
        }:
            raise ValueError(
                f"Response table columns disagree with metadata: {variable}"
            )
        frame = pd.read_csv(
            path, index_col=0, parse_dates=True, float_precision="round_trip"
        )
        if file_sha256(path) != item["sha256"]:
            raise ValueError(f"Response table changed during reading: {variable}")
        if (
            frame.empty
            or str(frame.index[0]) != temporal["start"]
            or str(frame.index[-1]) != temporal["end"]
        ):
            raise ValueError(f"Response table time coverage differs: {variable}")
        for ordinal, location in enumerate(item["locations"]):
            values = frame[location].to_numpy(dtype="float64")
            result.append(
                ResponseSeries(
                    run_id=run_id,
                    variable=variable,
                    location_id=location,
                    location_ordinal=ordinal,
                    time=frame.index,
                    calendar=temporal["calendar"],
                    timestep=step,
                    time_label=temporal["time_label"],
                    units=item["units"],
                    values=values,
                    missing=np.isnan(values),
                )
            )
    responses = tuple(result)
    validate_responses(responses, evaluated_runs={run_id})
    return responses


def publish_response_tables(
    native: NativeRunArtifacts,
    directory: str | Path,
    *,
    variables: tuple[str, ...] | None = None,
    decimal_places: int = 5,
    record_path: str | Path | None = None,
    prefix: str = "output",
) -> Path:
    """Stage rounded variables, publish their record last, then verify the result.

    Conversion never deletes native inputs. The workflow removes its temporary
    native output only after this function and all declared consumers succeed.
    """
    normalize_csv_output({"csv": {"decimal_places": decimal_places}})
    directory = Path(directory)
    record_path = (
        Path(record_path)
        if record_path is not None
        else directory / "_engine/response_tables.json"
    )
    root = record_path.parent.parent.resolve()
    if directory.resolve() != root or Path(prefix).name != prefix or not prefix:
        raise ValueError("Response tables and record must share one output root")
    before = {
        "csv_sha256": file_sha256(native.csv_path),
        "toml_sha256": file_sha256(native.toml_path),
    }
    with native.toml_path.open("rb") as stream:
        config = tomllib.load(stream)
    if variables is None:
        headers = {item["header"] for item in config["output"]["csv"]["column"]}
        supported = {item[0] for item in NATIVE_VARIABLES.values()}
        if headers - supported:
            raise ValueError(
                f"Unsupported scalar output headers: {sorted(headers - supported)}"
            )
        variables = tuple(
            key for key, item in NATIVE_VARIABLES.items() if item[0] in headers
        )
    responses = open_responses("conversion", native, ResponseRequest(variables))
    first = responses[0]
    if first.timestep != timedelta(days=1) or not first.time.equals(
        first.time.normalize()
    ):
        raise ValueError("Date-only response tables require daily midnight timestamps")
    temporal = {
        "calendar": first.calendar,
        "timestep_seconds": int(first.timestep.total_seconds()),
        "time_label": first.time_label,
        "start": str(first.time[0]),
        "end": str(first.time[-1]),
    }
    directory.mkdir(parents=True, exist_ok=True)
    record_path.parent.mkdir(parents=True, exist_ok=True)
    staged = {}
    metadata = {}
    try:
        for variable in variables:
            series = [item for item in responses if item.variable == variable]
            if any(not item.time.equals(first.time) for item in series):
                raise ValueError("Scalar response time axes differ")
            frame = pd.DataFrame(
                {
                    item.location_id: [
                        _format_value(value, decimal_places) for value in item.values
                    ]
                    for item in sorted(series, key=lambda item: int(item.location_id))
                }
            )
            frame.insert(0, "time", first.time.strftime("%Y-%m-%d"))
            target = directory / f"{prefix}_{NATIVE_VARIABLES[variable][0].lower()}.csv"
            if target.resolve() in {
                native.csv_path.resolve(),
                native.toml_path.resolve(),
            }:
                raise ValueError("Response export must preserve its native inputs")
            descriptor, path = tempfile.mkstemp(
                prefix=".response-", suffix=".csv", dir=directory
            )
            staged[target] = Path(path)
            with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as stream:
                frame.to_csv(stream, index=False)
            metadata[variable] = {
                "path": target.relative_to(directory).as_posix(),
                "sha256": file_sha256(path),
                "units": series[0].units,
                "locations": [item.location_id for item in series],
                "parameter": NATIVE_VARIABLES[variable][1],
                "spatial_support": "point" if variable == "q" else "subcatchment_mean",
            }
        record = {
            "schema_version": SCHEMA,
            "decimal_places": decimal_places,
            "temporal": temporal,
            "variables": metadata,
            "native_input": before,
        }
        descriptor, path = tempfile.mkstemp(
            prefix=".response-", suffix=".json", dir=record_path.parent
        )
        staged[record_path] = Path(path)
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(record, stream, indent=2, allow_nan=False)
            stream.write("\n")
        after = {
            "csv_sha256": file_sha256(native.csv_path),
            "toml_sha256": file_sha256(native.toml_path),
        }
        if before != after:
            raise ValueError("Native inputs changed during response conversion")
        for target, path in staged.items():
            os.replace(path, target)
        verified = open_table_responses(
            "conversion", record_path, ResponseRequest(variables)
        )
        if [item.key for item in verified] != [item.key for item in responses]:
            raise ValueError("Converted response identities differ")
        for original, retained in zip(responses, verified, strict=True):
            if not original.time.equals(retained.time) or not np.array_equal(
                original.missing, retained.missing
            ):
                raise ValueError("Converted response coverage or missing values differ")
            valid = ~original.missing
            error = np.abs(original.values[valid] - retained.values[valid])
            tolerance = 0.5 * 10.0**-decimal_places + np.spacing(
                np.abs(original.values[valid])
            )
            if np.any(error > tolerance):
                raise ValueError(
                    "Converted response values exceed selected rounding tolerance"
                )
    finally:
        for path in staged.values():
            path.unlink(missing_ok=True)
    return record_path
