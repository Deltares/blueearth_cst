"""Complete retained native response coverage without a live model or generation."""

import hashlib
import tomllib
from pathlib import Path

import pandas as pd

from blueearth_cst.experiment.content_identity import (
    content_sha256,
    read_canonical_json,
)
from blueearth_cst.experiment.simulation_record import atomic_record
from blueearth_cst.experiment.wflow_response_reader import (
    NativeRunArtifacts,
    ResponseRequest,
    open_responses,
)
from blueearth_cst.shared.provenance import file_sha256


class MissingResponseRequirement(ValueError):
    """Retained native responses do not satisfy the frozen complete request."""


class ResponseReaderUnavailable(ValueError):
    """The native reader revision needed by retained state is not installed."""


def _reader_changed(experiment_root, recorded):
    """The refusal for a frozen experiment read with a different reader.

    Not an installation fault, though the class name says "unavailable".
    Reader source changes require a new experiment, apart from the explicitly
    reviewed vocabulary-only revision accepted when reopening retained state.
    """
    return ResponseReaderUnavailable(
        f"experiment {Path(experiment_root).name!r} was simulated with response "
        f"reader {str(recorded.get('revision'))[:12]}, and this toolbox's reader "
        f"is {response_reader_revision()[:12]}, so its retained responses cannot "
        "be re-read. Set a new experiment_name to simulate again, or move this "
        "experiment aside."
    )


def response_reader_revision():
    """Fingerprint the actual native reader and neutral validation implementation."""
    directory = Path(__file__).parent
    return content_sha256(
        [
            {"path": name, "sha256": file_sha256(directory / name)}
            for name in (
                "response_series.py",
                "wflow_response_reader.py",
                # The labels the reader keys every series by.
                "../shared/native_locations.py",
            )
        ]
    )


def _terminology_reader_matches(recorded, current_revision):
    """Accept only the reviewed pre-indicator docstring revision for reads.

    Pin the complete neutral implementation, retain the other source hashes,
    and account for the LF/CRLF bytes used by previous installations. Neither
    saved records nor the current writer's revision are changed.
    """
    if set(recorded) != {"name", "revision"} or recorded["name"] != "wflow-csv":
        return False
    directory = Path(__file__).parent
    source = (directory / "response_series.py").read_bytes().replace(b"\r\n", b"\n")
    if hashlib.sha256(source).hexdigest() != (
        "991cdf04d7191c77ff3c1fadbcd3800b0e8dc47be1e5e24c4ba7c22848cf37bb"
    ):
        return False
    rows = [
        {"path": name, "sha256": file_sha256(directory / name)}
        for name in (
            "response_series.py",
            "wflow_response_reader.py",
            "../shared/native_locations.py",
        )
    ]
    if content_sha256(rows) != current_revision:
        return False
    for historical in (
        "148f4f51059adbaa902f0b53b7e4cbed65e9758384fa3c9f2bab67faab2788e9",
        "6902702887ad3f454b99c57ccddce26e0ed703f2a12631e269c1486d9874def3",
    ):
        if recorded["revision"] == content_sha256(
            [{**rows[0], "sha256": historical}, *rows[1:]]
        ):
            return True
    return False


def make_response_request(run_ids, variables):
    """Freeze evaluated runs and model-declared ordered locations before execution.

    Each variable declaration supplies variable, locations, units, calendar,
    timestep, time_label, start, end and missing_value. Location order comes from
    the native simulator's output plan, never from completed CSV contents.
    """
    request = {
        "schema_version": "response-request/2",
        "reader": {"name": "wflow-csv", "revision": response_reader_revision()},
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
        != {"schema_version", "reader", "run_ids", "variables", "expected_series"}
        or request["schema_version"] != "response-request/2"
    ):
        raise MissingResponseRequirement("invalid response-request/2 fields")
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


def _validate_temporal(temporal, config):
    required = {
        "source_calendar",
        "prepared_forcing_calendar",
        "response_calendar",
        "operations",
        "prepared_start",
        "prepared_end",
        "response_start",
        "response_end",
        "time_label",
    }
    if set(temporal) != required:
        raise MissingResponseRequirement("temporal preparation evidence is incomplete")
    suffix = ["clip_to_configured_window", "refresh_toml_endpoints"]
    expected_operations = [suffix]
    if temporal["source_calendar"] == "noleap":
        expected_operations = [
            [conversion, *suffix]
            for conversion in ("cftime_to_datetime64", "hydromt_reader_to_datetime64")
        ]
    if (
        temporal["operations"] not in expected_operations
        or temporal["prepared_forcing_calendar"] != "proleptic_gregorian"
    ):
        raise MissingResponseRequirement("unsupported temporal preparation chain")
    clock = config["time"]
    if (
        pd.Timestamp(temporal["prepared_start"]) != pd.Timestamp(clock["starttime"])
        or pd.Timestamp(temporal["prepared_end"]) != pd.Timestamp(clock["endtime"])
        or temporal["response_calendar"] != clock["calendar"]
        or pd.Timestamp(temporal["response_start"])
        != pd.Timestamp(clock["starttime"])
        + pd.Timedelta(seconds=clock["timestepsecs"])
        or pd.Timestamp(temporal["response_end"]) != pd.Timestamp(clock["endtime"])
        or temporal["time_label"] != "interval_end"
    ):
        raise MissingResponseRequirement(
            "temporal preparation differs from native simulation clock"
        )


def _v2_exact(value, fields, name):
    if type(value) is not dict or set(value) != set(fields):
        raise MissingResponseRequirement(f"{name} has unclassified fields")
    return value


def _v2_native_semantics(config, variable):
    """Project only the TOML fields consumed by the native response reader."""
    from blueearth_cst.experiment.wflow_response_reader import NATIVE_VARIABLES

    clock = config["time"]
    required_clock_fields = {"calendar", "starttime", "endtime", "timestepsecs"}
    allowed_clock_fields = required_clock_fields | {"time_units"}
    if (
        type(clock) is not dict
        or not required_clock_fields <= set(clock) <= allowed_clock_fields
    ):
        raise MissingResponseRequirement("native clock has unclassified fields")
    if type(clock["timestepsecs"]) is not int or clock["timestepsecs"] <= 0:
        raise MissingResponseRequirement("native timestep must be positive")
    if "time_units" in clock and (
        not isinstance(clock["time_units"], str) or not clock["time_units"]
    ):
        raise MissingResponseRequirement("native time units must be explicit")
    normalized_clock = {
        "calendar": clock["calendar"],
        "starttime": str(pd.Timestamp(clock["starttime"])),
        "endtime": str(pd.Timestamp(clock["endtime"])),
        "timestepsecs": clock["timestepsecs"],
    }
    header, parameter, _ = NATIVE_VARIABLES[variable]
    columns = config["output"]["csv"]["column"]
    if type(columns) is not list:
        raise MissingResponseRequirement("native output declarations are invalid")
    declarations = []
    for item in columns:
        if item.get("header") != header:
            continue
        required_declaration_fields = {"header", "parameter"}
        allowed_declaration_fields = required_declaration_fields | {"map", "reducer"}
        if (
            type(item) is not dict
            or not required_declaration_fields
            <= set(item)
            <= allowed_declaration_fields
        ):
            raise MissingResponseRequirement(
                "native output declaration has unclassified fields"
            )
        if any(
            not isinstance(item[field], str) or not item[field]
            for field in {"map", "reducer"} & set(item)
        ):
            raise MissingResponseRequirement(
                "native output routing metadata is invalid"
            )
        declarations.append({"header": item["header"], "parameter": item["parameter"]})
    if not declarations or any(item["parameter"] != parameter for item in declarations):
        raise MissingResponseRequirement("native parameter differs from reader mapping")
    return {
        "parameter": parameter,
        "output_declarations": declarations,
        "clock": normalized_clock,
    }


def build_response_inventory_v2(
    experiment_root, native_runs, temporal_preparation, *, _retained=False
):
    """Check every requested native run and build the acyclic v2 inventory."""
    from blueearth_cst.experiment.simulation_record import read_simulation_intent_v2
    from blueearth_cst.experiment.wflow_response_reader import NATIVE_VARIABLES
    from blueearth_cst.shared.workflow_config_snapshot import file_reference

    root = Path(experiment_root).resolve()
    intent = read_simulation_intent_v2(root)
    request = intent["documents"]["response_request"]
    validate_response_request(request)
    current_reader = {
        "name": "wflow-csv",
        "revision": response_reader_revision(),
    }
    if request["reader"] != current_reader and not (
        _retained
        and _terminology_reader_matches(request["reader"], current_reader["revision"])
    ):
        raise _reader_changed(experiment_root, request["reader"])
    if set(native_runs) != set(request["run_ids"]):
        raise MissingResponseRequirement(
            "native runs differ from frozen response request"
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
    temporal_ref = {
        "schema_version": "local-section-reference/1",
        "section": "/temporal_preparation",
        "section_sha256": temporal_digest,
    }
    inventory_artifacts = []
    identity_artifacts = []
    inventory_series = []
    identity_series = []
    metadata = {item["variable"]: item for item in request["variables"]}
    for run in request["run_ids"]:
        native = native_runs[run]
        if native.temporal_path is None and not _retained:
            raise MissingResponseRequirement(f"run {run}: missing temporal handoff")
        paths = (native.csv_path, native.toml_path)
        if native.temporal_path is not None:
            paths += (native.temporal_path,)
        for artifact_path in paths:
            if not Path(artifact_path).resolve(strict=True).is_relative_to(root):
                raise MissingResponseRequirement(
                    f"run {run}: native artifact escapes experiment"
                )
        if native.temporal_path is not None:
            temporal = read_canonical_json(Path(native.temporal_path))
            if temporal != temporal_preparation:
                raise MissingResponseRequirement(
                    f"run {run}: inconsistent temporal preparation"
                )
        with Path(native.toml_path).open("rb") as handle:
            config = tomllib.load(handle)
        _validate_temporal(temporal_preparation, config)
        csv_ref = file_reference(native.csv_path, "experiment_root", root)
        toml_ref = file_reference(native.toml_path, "experiment_root", root)
        index = len(inventory_artifacts)
        inventory_artifacts.append({"run_id": run, "file": csv_ref})
        identity_artifacts.append(
            {
                "run_id": run,
                "sha256": csv_ref["sha256"],
                "size_bytes": csv_ref["size_bytes"],
            }
        )
        values = open_responses(
            run,
            native,
            ResponseRequest(tuple(item["variable"] for item in request["variables"])),
        )
        actual_keys = {item.key for item in values}
        expected_keys = {
            tuple(key) for key in request["expected_series"] if key[0] == run
        }
        if actual_keys != expected_keys:
            raise MissingResponseRequirement(
                f"run {run}: response keys missing={sorted(expected_keys - actual_keys)} "
                f"extra={sorted(actual_keys - expected_keys)}"
            )
        for item in values:
            declared = metadata[item.variable]
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
            for key, value in observed.items():
                if declared[key] != value:
                    raise MissingResponseRequirement(
                        f"{item.key}: {key} expected={declared[key]} observed={value}"
                    )
            if declared["locations"][item.location_ordinal] != item.location_id:
                raise MissingResponseRequirement(
                    f"{item.key}: native location order differs"
                )
            header = NATIVE_VARIABLES[item.variable][0]
            column = f"{header}_{item.location_id}"
            semantic_selector = {
                "column": column,
                **_v2_native_semantics(config, item.variable),
                "temporal_preparation_sha256": temporal_digest,
            }
            scientific = {
                "run_id": run,
                "variable": item.variable,
                "location_id": item.location_id,
                "location_ordinal": item.location_ordinal,
                "artifact": index,
                **observed,
                "native_selector": semantic_selector,
            }
            identity_series.append(scientific)
            inventory_series.append(
                {
                    **scientific,
                    "native_selector": {
                        "column": column,
                        "toml": toml_ref,
                        "temporal": temporal_ref,
                    },
                }
            )

    def order(item):
        return (
            item["run_id"],
            item["variable"],
            item["location_ordinal"],
            item["location_id"],
        )

    inventory_series.sort(key=order)
    identity_series.sort(key=order)
    intent_ref = file_reference(
        root / "_engine/simulation_intent.json", "experiment_root", root
    )
    response_ref = {
        "schema_version": "section-reference/1",
        "document": intent_ref,
        "section": "/documents/response_request",
        "section_sha256": content_sha256(request),
    }
    identity = {
        "schema_version": "response-inventory-identity/1",
        "simulation_id": intent["simulation_id"],
        "collection_id": intent["collection"]["collection_id"],
        "collection_revision": intent["collection"]["collection_revision"],
        "model_digest": intent["documents"]["model_reference"]["digest"],
        "simulator": intent["simulator"],
        "settings_sha256": intent["identity_digests"]["settings"],
        "response_request_sha256": intent["identity_digests"]["response_request"],
        "temporal_preparation": temporal_preparation,
        "artifacts": identity_artifacts,
        "series": identity_series,
    }
    inventory = {
        "schema_version": "response-inventory/3",
        "simulation_id": intent["simulation_id"],
        "collection_id": identity["collection_id"],
        "collection_revision": identity["collection_revision"],
        "model_digest": identity["model_digest"],
        "simulator": intent["simulator"],
        "settings_sha256": identity["settings_sha256"],
        "response_request": response_ref,
        "temporal_preparation": temporal_preparation,
        "temporal_preparation_sha256": temporal_digest,
        "artifacts": inventory_artifacts,
        "series": inventory_series,
        "identity_projection": identity,
        "response_identity_sha256": content_sha256(identity),
    }
    inventory["response_inventory_sha256"] = content_sha256(inventory)
    return inventory


def read_response_inventory_v2(experiment_root):
    """Validate the enclosing digest before following any local selector reference."""
    from blueearth_cst.shared.workflow_config_snapshot import resolve_file_reference

    root = Path(experiment_root).resolve()
    path = root / "_engine/response_inventory.json"
    try:
        stored = read_canonical_json(path)
        _v2_exact(
            stored,
            {
                "schema_version",
                "simulation_id",
                "collection_id",
                "collection_revision",
                "model_digest",
                "simulator",
                "settings_sha256",
                "response_request",
                "temporal_preparation",
                "temporal_preparation_sha256",
                "artifacts",
                "series",
                "identity_projection",
                "response_identity_sha256",
                "response_inventory_sha256",
            },
            "response inventory",
        )
        if stored["schema_version"] != "response-inventory/3":
            raise MissingResponseRequirement("unsupported response inventory version")
        digest = content_sha256(
            {
                key: value
                for key, value in stored.items()
                if key != "response_inventory_sha256"
            }
        )
        if stored["response_inventory_sha256"] != digest:
            raise MissingResponseRequirement(
                "response inventory digest mismatch before native opening"
            )
        temporal_digest = content_sha256(stored["temporal_preparation"])
        if stored["temporal_preparation_sha256"] != temporal_digest:
            raise MissingResponseRequirement("common temporal digest differs")
        response_ref = _v2_exact(
            stored["response_request"],
            {"schema_version", "document", "section", "section_sha256"},
            "response request section",
        )
        if (
            response_ref["schema_version"] != "section-reference/1"
            or response_ref["section"] != "/documents/response_request"
            or response_ref["document"]["path_base"] != "experiment_root"
            or response_ref["document"]["path"] != "_engine/simulation_intent.json"
        ):
            raise MissingResponseRequirement("response request section is invalid")
        intent_path = resolve_file_reference(
            response_ref["document"], {"experiment_root": root}
        )
        intent = read_canonical_json(intent_path)
        if response_ref["section_sha256"] != content_sha256(
            intent["documents"]["response_request"]
        ):
            raise MissingResponseRequirement("response request section digest differs")
        native_runs = {}
        for index, artifact in enumerate(stored["artifacts"]):
            _v2_exact(artifact, {"run_id", "file"}, "native artifact")
            matching = [item for item in stored["series"] if item["artifact"] == index]
            if not matching or artifact["run_id"] in native_runs:
                raise MissingResponseRequirement(
                    "ambiguous native artifact association"
                )
            tomls = []
            for item in matching:
                selector = _v2_exact(
                    item["native_selector"],
                    {"column", "toml", "temporal"},
                    "native selector",
                )
                local = _v2_exact(
                    selector["temporal"],
                    {"schema_version", "section", "section_sha256"},
                    "local temporal section",
                )
                if local != {
                    "schema_version": "local-section-reference/1",
                    "section": "/temporal_preparation",
                    "section_sha256": temporal_digest,
                }:
                    raise MissingResponseRequirement("local temporal section differs")
                tomls.append(selector["toml"])
            if any(item != tomls[0] for item in tomls):
                raise MissingResponseRequirement("ambiguous native TOML association")
            if (
                artifact["file"]["path_base"] != "experiment_root"
                or tomls[0]["path_base"] != "experiment_root"
            ):
                raise MissingResponseRequirement("native reference has wrong root")
            native_runs[artifact["run_id"]] = NativeRunArtifacts(
                resolve_file_reference(artifact["file"], {"experiment_root": root}),
                resolve_file_reference(tomls[0], {"experiment_root": root}),
                None,
            )
        # Temporary handoff files are deliberately unnecessary after publication.
        observed = _build_response_inventory_v2_retained(
            root, native_runs, stored["temporal_preparation"]
        )
        if stored != observed:
            raise MissingResponseRequirement(
                "response inventory differs from native evidence"
            )
        return observed
    except (OSError, KeyError, TypeError, ValueError) as exc:
        if isinstance(exc, ResponseReaderUnavailable):
            raise
        raise MissingResponseRequirement(f"retained responses {path}: {exc}") from exc


def _build_response_inventory_v2_retained(root, native_runs, temporal_preparation):
    """Rebuild through checked native files without requiring temporary JSONs."""
    return build_response_inventory_v2(
        root, native_runs, temporal_preparation, _retained=True
    )


def publish_response_inventory_v2(experiment_root, native_runs, temporal_preparation):
    """Publish the immutable v2 inventory; simulation readiness follows separately."""
    root = Path(experiment_root).resolve()
    path = root / "_engine/response_inventory.json"
    if path.resolve() != path:
        raise MissingResponseRequirement("response publication path is aliased")
    inventory = build_response_inventory_v2(root, native_runs, temporal_preparation)
    if path.exists():
        if read_response_inventory_v2(root) != inventory:
            raise MissingResponseRequirement(
                "immutable response inventory already differs"
            )
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        atomic_record(path, inventory)
    return inventory
