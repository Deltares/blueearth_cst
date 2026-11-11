"""Complete retained compact response coverage without native scalar inputs."""

import tomllib
from pathlib import Path

import pandas as pd

from blueearth_cst.experiment.content_identity import (
    content_sha256,
    read_canonical_json,
)
from blueearth_cst.experiment.simulation_record import atomic_record
from blueearth_cst.experiment.wflow_response_reader import ResponseRequest
from blueearth_cst.shared.provenance import file_sha256
from blueearth_cst.shared.response_tables import open_table_responses, read_table_record


class MissingResponseRequirement(ValueError):
    """Retained response tables do not satisfy the frozen complete request."""


class ResponseReaderUnavailable(ValueError):
    """The compact reader revision differs from the frozen request."""


def _reader_changed(experiment_root, recorded):
    """The refusal for a frozen experiment read with a different reader.

    Not an installation fault, though the class name says "unavailable".
    Reader source changes require a new experiment.
    """
    return ResponseReaderUnavailable(
        f"experiment {Path(experiment_root).name!r} was simulated with response "
        f"reader {str(recorded.get('revision'))[:12]}, and this toolbox's reader "
        f"is {response_reader_revision()[:12]}, so its retained responses cannot "
        "be re-read. Set a new experiment_name to simulate again, or move this "
        "experiment aside."
    )


def response_reader_revision():
    """Fingerprint the compact reader and neutral validation implementation."""
    directory = Path(__file__).parent
    return content_sha256(
        [
            {"path": name, "sha256": file_sha256(directory / name)}
            for name in (
                "response_series.py",
                "../shared/response_tables.py",
                "../shared/csv_output.py",
            )
        ]
    )


def make_response_request(run_ids, variables, *, decimal_places=5):
    """Freeze evaluated runs and model-declared ordered locations before execution.

    Each variable declaration supplies variable, locations, units, calendar,
    timestep, time_label, start, end and missing_value. Location order comes from
    the native simulator's output plan, never from completed CSV contents.
    """
    request = {
        "schema_version": "response-request/3",
        "reader": {"name": "wflow-tables", "revision": response_reader_revision()},
        "decimal_places": decimal_places,
        "run_ids": list(run_ids),
        "variables": variables,
        "expected_series": [
            [run, item["variable"], location]
            for run in run_ids
            for item in variables
            for location in item["locations"]
        ],
    }
    validate_response_request(request)
    return request


def validate_response_request(request):
    """Require exact independently declared keys and complete physical metadata."""
    if (
        set(request)
        != {
            "schema_version",
            "reader",
            "decimal_places",
            "run_ids",
            "variables",
            "expected_series",
        }
        or request["schema_version"] != "response-request/3"
    ):
        raise MissingResponseRequirement(
            "unsupported response request; create a new experiment using response-request/3"
        )
    from blueearth_cst.shared.csv_output import normalize_csv_output

    normalize_csv_output({"csv": {"decimal_places": request["decimal_places"]}})
    _v2_exact(request["reader"], {"name", "revision"}, "response reader")
    if request["reader"]["name"] != "wflow-tables":
        raise MissingResponseRequirement(
            "retained responses require the wflow-tables reader"
        )
    runs = request["run_ids"]
    if (
        not runs
        or any(
            not isinstance(run, str)
            or not (
                run.startswith("run_") and run[4:].isascii() and run[4:].isdecimal()
            )
            for run in runs
        )
        or runs != sorted(set(runs))
    ):
        raise MissingResponseRequirement(
            "response request requires unique ordered prefixed run ids (legacy numeric IDs require a new simulation)"
        )
    variables = request["variables"]
    if not variables:
        raise MissingResponseRequirement("response request has no variables")
    names = []
    for item in variables:
        if set(item) != {
            "variable",
            "locations",
            "units",
            "calendar",
            "timestep",
            "time_label",
            "start",
            "end",
            "missing_value",
        }:
            raise MissingResponseRequirement("response variable metadata is incomplete")
        names.append(item["variable"])
        locations = item["locations"]
        if (
            not locations
            or len(set(locations)) != len(locations)
            or any(not isinstance(value, str) or not value for value in locations)
        ):
            raise MissingResponseRequirement(
                "response locations must be unique ordered text"
            )
        if any(
            not isinstance(item[key], str) or not item[key]
            for key in (
                "variable",
                "units",
                "calendar",
                "timestep",
                "time_label",
                "start",
                "end",
            )
        ):
            raise MissingResponseRequirement(
                "response physical metadata must be explicit"
            )
    if len(set(names)) != len(names):
        raise MissingResponseRequirement("duplicate requested variables")
    expected = [
        [run, item["variable"], location]
        for run in runs
        for item in variables
        for location in item["locations"]
    ]
    if request["expected_series"] != expected:
        raise MissingResponseRequirement(
            "expected_series differs from frozen runs and locations"
        )


def _v2_exact(value, fields, name):
    if type(value) is not dict or set(value) != set(fields):
        raise MissingResponseRequirement(f"{name} has unclassified fields")
    return value


def build_response_inventory_v2(
    experiment_root,
    table_records,
    temporal_preparation,
    *,
    grid_tomls=None,
    _grid_metadata=None,
):
    """Verify compact responses against the independently frozen complete request."""
    from blueearth_cst.experiment.simulation_record import read_simulation_intent_v2
    from blueearth_cst.shared.workflow_config_snapshot import file_reference

    root = Path(experiment_root).resolve()
    intent = read_simulation_intent_v2(root)
    request = intent["documents"]["response_request"]
    validate_response_request(request)
    if request["reader"] != {
        "name": "wflow-tables",
        "revision": response_reader_revision(),
    }:
        raise _reader_changed(root, request["reader"])
    if set(table_records) != set(request["run_ids"]):
        raise MissingResponseRequirement(
            "table runs differ from frozen response request"
        )
    _v2_exact(
        temporal_preparation,
        {
            "source_calendar",
            "prepared_forcing_calendar",
            "response_calendar",
            "operations",
            "prepared_start",
            "prepared_end",
            "response_start",
            "response_end",
            "time_label",
        },
        "temporal preparation",
    )
    temporal_digest = content_sha256(temporal_preparation)
    suffix = ["clip_to_configured_window"]
    warmup = "prepend_warmup_years" in temporal_preparation["operations"]
    if warmup:
        suffix.append("prepend_warmup_years")
    suffix.append("refresh_toml_endpoints")
    chains = [suffix]
    if temporal_preparation["source_calendar"] == "noleap":
        chains = [
            [conversion, *suffix]
            for conversion in ("cftime_to_datetime64", "hydromt_reader_to_datetime64")
        ]
    if (
        temporal_preparation["operations"] not in chains
        or temporal_preparation["prepared_forcing_calendar"] != "proleptic_gregorian"
    ):
        raise MissingResponseRequirement("unsupported temporal preparation chain")
    artifacts, series, identity_artifacts = [], [], []
    declarations = {item["variable"]: item for item in request["variables"]}
    for run in request["run_ids"]:
        path = Path(table_records[run]).resolve(strict=True)
        if not path.is_relative_to(root):
            raise MissingResponseRequirement(
                f"run {run}: table record escapes experiment"
            )
        record = read_table_record(path)
        first_native = pd.Timestamp(
            temporal_preparation["prepared_start"]
        ) + pd.Timedelta(seconds=record["temporal"]["timestep_seconds"])
        response_start = pd.Timestamp(temporal_preparation["response_start"])
        if (
            response_start <= first_native if warmup else response_start != first_native
        ) or pd.Timestamp(temporal_preparation["prepared_end"]) != pd.Timestamp(
            temporal_preparation["response_end"]
        ):
            raise MissingResponseRequirement(
                "temporal preparation differs from response clock"
            )
        if record["decimal_places"] != request["decimal_places"]:
            raise MissingResponseRequirement(
                f"run {run}: CSV precision differs from frozen request"
            )
        if set(record["variables"]) != set(declarations):
            raise MissingResponseRequirement(
                f"run {run}: retained variables differ from frozen request"
            )
        reference = file_reference(path, "experiment_root", root)
        index = len(artifacts)
        artifacts.append({"run_id": run, "file": reference})
        identity_artifacts.append(
            {
                "run_id": run,
                "sha256": reference["sha256"],
                "size_bytes": reference["size_bytes"],
            }
        )
        values = open_table_responses(run, path, ResponseRequest(tuple(declarations)))
        if {item.key for item in values} != {
            tuple(key) for key in request["expected_series"] if key[0] == run
        }:
            raise MissingResponseRequirement(
                f"run {run}: response keys differ from frozen request"
            )
        for item in values:
            seconds = item.timestep.total_seconds()
            observed = {
                "units": item.units,
                "calendar": item.calendar,
                "timestep": "P1D" if seconds == 86400 else f"PT{seconds:g}S",
                "time_label": item.time_label,
                "start": str(item.time[0]),
                "end": str(item.time[-1]),
                "missing_value": "NaN",
            }
            declared = declarations[item.variable]
            if any(declared[key] != value for key, value in observed.items()):
                raise MissingResponseRequirement(
                    f"{item.key}: physical metadata differs from frozen request"
                )
            if declared["locations"][item.location_ordinal] != item.location_id:
                raise MissingResponseRequirement(
                    f"{item.key}: native location order differs"
                )
            if (
                observed["calendar"] != temporal_preparation["response_calendar"]
                or pd.Timestamp(observed["start"])
                != pd.Timestamp(temporal_preparation["response_start"])
                or pd.Timestamp(observed["end"])
                != pd.Timestamp(temporal_preparation["response_end"])
                or observed["time_label"] != temporal_preparation["time_label"]
            ):
                raise MissingResponseRequirement(
                    f"run {run}: temporal preparation differs from retained responses"
                )
            series.append(
                {
                    "run_id": run,
                    "variable": item.variable,
                    "location_id": item.location_id,
                    "location_ordinal": item.location_ordinal,
                    "artifact": index,
                    **observed,
                    "table_selector": {
                        "variable": item.variable,
                        "temporal_preparation_sha256": temporal_digest,
                    },
                }
            )
    series.sort(
        key=lambda item: (item["run_id"], item["variable"], item["location_ordinal"])
    )
    grids, grid_identity = [], []
    grid_request = intent["documents"]["settings"].get("netcdf_grid")
    if grid_request is not None:
        from blueearth_cst.shared.grid_output import configured_grid_path, validate_grid

        if grid_tomls is None or set(grid_tomls) != set(request["run_ids"]):
            raise MissingResponseRequirement(
                "grid TOMLs differ from frozen response runs"
            )
        for run in request["run_ids"]:
            toml = Path(grid_tomls[run]).resolve(strict=True)
            path = configured_grid_path(toml)
            if (
                path is None
                or not toml.is_relative_to(root)
                or not path.resolve(strict=True).is_relative_to(root)
            ):
                raise MissingResponseRequirement(
                    f"run {run}: grid path escapes experiment"
                )
            with toml.open("rb") as handle:
                config = tomllib.load(handle)
            if {
                key: value
                for key, value in config["output"]["netcdf_grid"].items()
                if key != "path"
            } != grid_request:
                raise MissingResponseRequirement(
                    f"run {run}: grid request differs from frozen intent"
                )
            saved = (
                None
                if _grid_metadata is None
                else _grid_metadata[run]["validation_contract"]
            )
            metadata = validate_grid(path, toml, expected_contract=saved)
            reference = file_reference(path, "experiment_root", root)
            grids.append(
                {
                    "run_id": run,
                    "file": reference,
                    "toml": file_reference(toml, "experiment_root", root),
                    "metadata": metadata,
                }
            )
            grid_identity.append(
                {
                    "run_id": run,
                    "sha256": reference["sha256"],
                    "size_bytes": reference["size_bytes"],
                    "metadata": metadata,
                }
            )
    elif grid_tomls:
        raise MissingResponseRequirement("unexpected retained grids")
    identity = {
        "schema_version": "response-inventory-identity/3",
        "simulation_id": intent["simulation_id"],
        "collection_id": intent["collection"]["collection_id"],
        "collection_revision": intent["collection"]["collection_revision"],
        "model_digest": intent["documents"]["model_reference"]["digest"],
        "simulator": intent["simulator"],
        "settings_sha256": intent["identity_digests"]["settings"],
        "response_request_sha256": intent["identity_digests"]["response_request"],
        "temporal_preparation": temporal_preparation,
        "artifacts": identity_artifacts,
        "series": series,
        "grids": grid_identity,
    }
    inventory = {
        key: identity[key]
        for key in (
            "simulation_id",
            "collection_id",
            "collection_revision",
            "model_digest",
            "simulator",
            "settings_sha256",
        )
    }
    inventory.update(
        {
            "schema_version": "response-inventory/5",
            "response_request": {
                "schema_version": "section-reference/1",
                "document": file_reference(
                    root / "_engine/simulation_intent.json", "experiment_root", root
                ),
                "section": "/documents/response_request",
                "section_sha256": content_sha256(request),
            },
            "temporal_preparation": temporal_preparation,
            "temporal_preparation_sha256": temporal_digest,
            "artifacts": artifacts,
            "series": series,
            "grids": grids,
            "identity_projection": identity,
            "response_identity_sha256": content_sha256(identity),
        }
    )
    inventory["response_inventory_sha256"] = content_sha256(inventory)
    return inventory


def read_response_inventory_v2(experiment_root):
    """Verify compact retained evidence without opening a native scalar CSV."""
    from blueearth_cst.shared.workflow_config_snapshot import resolve_file_reference

    root = Path(experiment_root).resolve()
    path = root / "_engine/response_inventory.json"
    try:
        stored = read_canonical_json(path)
        if stored.get("schema_version") != "response-inventory/5":
            raise MissingResponseRequirement(
                "unsupported native response inventory; create a new experiment using compact response tables"
            )
        if stored["response_inventory_sha256"] != content_sha256(
            {
                key: value
                for key, value in stored.items()
                if key != "response_inventory_sha256"
            }
        ):
            raise MissingResponseRequirement(
                "response inventory digest mismatch before table opening"
            )
        ref = stored["response_request"]
        intent_path = resolve_file_reference(ref["document"], {"experiment_root": root})
        if intent_path != root / "_engine/simulation_intent.json":
            raise MissingResponseRequirement(
                "response request points to another intent"
            )
        records = {}
        for artifact in stored["artifacts"]:
            _v2_exact(artifact, {"run_id", "file"}, "table artifact")
            if artifact["run_id"] in records:
                raise MissingResponseRequirement("duplicate table run")
            records[artifact["run_id"]] = resolve_file_reference(
                artifact["file"], {"experiment_root": root}
            )
        tomls, metadata = {}, {}
        for grid in stored["grids"]:
            _v2_exact(grid, {"run_id", "file", "toml", "metadata"}, "grid artifact")
            resolve_file_reference(grid["file"], {"experiment_root": root})
            tomls[grid["run_id"]] = resolve_file_reference(
                grid["toml"], {"experiment_root": root}
            )
            metadata[grid["run_id"]] = grid["metadata"]
        observed = build_response_inventory_v2(
            root,
            records,
            stored["temporal_preparation"],
            grid_tomls=tomls or None,
            _grid_metadata=metadata or None,
        )
        if stored != observed:
            raise MissingResponseRequirement(
                "response inventory differs from compact table evidence"
            )
        return observed
    except (OSError, KeyError, TypeError, ValueError) as exc:
        if isinstance(exc, ResponseReaderUnavailable):
            raise
        raise MissingResponseRequirement(f"retained responses {path}: {exc}") from exc


def publish_response_inventory_v2(
    experiment_root, table_records, temporal_preparation, *, grid_tomls=None
):
    """Publish verified compact response inventory before simulation readiness."""
    root = Path(experiment_root).resolve()
    path = root / "_engine/response_inventory.json"
    if path.resolve() != path:
        raise MissingResponseRequirement("response publication path is aliased")
    inventory = build_response_inventory_v2(
        root, table_records, temporal_preparation, grid_tomls=grid_tomls
    )
    if path.exists():
        if read_response_inventory_v2(root) != inventory:
            raise MissingResponseRequirement(
                "immutable response inventory already differs"
            )
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        atomic_record(path, inventory)
    return inventory
