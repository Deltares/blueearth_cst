"""Freeze simulation inputs independently of indicator selection and completion."""

import hashlib
import os
import re
from pathlib import Path

from blueearth_cst.experiment.content_identity import (
    atomic_record,
    canonical_json_bytes,
    confined_path,
    content_sha256,
    read_canonical_json,
)


class SimulationFrozenError(ValueError):
    """An experiment cannot reuse changed immutable simulation inputs."""


class IndicatorsOnlySimulationUnavailable(ValueError):
    """The selected experiment has no complete retained simulation."""


def _simulation_path(root, relative):
    path = confined_path(root, relative)
    if path != root / relative:
        raise SimulationFrozenError(f"simulation record path is aliased: {relative}")
    return path


_V2_DOCUMENTS = frozenset(
    {
        "model_reference",
        "settings",
        "environment",
        "simulator_adapter_code",
        "response_request",
        "preparation",
    }
)
_V2_PROJECTED = frozenset({"settings", "response_request", "preparation"})


def _v2_fields(value, expected, name):
    if type(value) is not dict or set(value) != set(expected):
        raise ValueError(f"{name} requires exactly {sorted(expected)}")
    return value


def _v2_digest(value, name):
    if type(value) is not str or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError(f"{name} requires full lowercase SHA-256")
    return value


def _v2_model_reference(document):
    from blueearth_cst.shared.model_digest import (
        DIGEST_VERSION,
        model_digest_from_entries,
    )

    _v2_fields(
        document,
        {"digest_version", "model_path", "model_toml", "digest", "inputs"},
        "model_reference",
    )
    if document["digest_version"] != DIGEST_VERSION:
        raise ValueError("model reference digest version differs")
    if type(document["inputs"]) is not dict or not document["inputs"]:
        raise ValueError("model reference inputs are empty")
    for name, digest in document["inputs"].items():
        if type(name) is not str or not name or "\\" in name or name.startswith("/"):
            raise ValueError("model input path is invalid")
        if digest != "<absent>":
            _v2_digest(digest, f"model input {name}")
    observed = model_digest_from_entries(sorted(document["inputs"].items()))
    if document["digest"] != observed:
        raise ValueError("model reference digest differs from input hashes")
    return observed


def _v2_simulation_id(intent):
    return content_sha256(
        {
            "schema_version": "simulation-identity/2",
            "collection_id": intent["collection"]["collection_id"],
            "collection_revision": intent["collection"]["collection_revision"],
            "run_ids": intent["collection"]["run_ids"],
            "model_digest": intent["documents"]["model_reference"]["digest"],
            "simulator_name_and_revision": intent["simulator"],
            "simulator_adapter_code_sha256": intent["document_digests"][
                "simulator_adapter_code"
            ],
            "settings_sha256": intent["identity_digests"]["settings"],
            "simulation_response_request_sha256": intent["identity_digests"][
                "response_request"
            ],
            "environment_sha256": intent["document_digests"]["environment"],
            "preparation_sha256": intent["identity_digests"]["preparation"],
        }
    )


def simulation_intent_v2(
    experiment_name, collection, simulator, documents, preparation_identity
):
    """Build the closed immutable simulation intent from checked resolved inputs."""
    _v2_fields(
        collection,
        {
            "resolution_mode",
            "manifest",
            "collection_id",
            "collection_revision",
            "run_ids",
        },
        "simulation collection",
    )
    if collection["resolution_mode"] not in {"project-generation", "explicit-manifest"}:
        raise ValueError("invalid collection resolution mode")
    for field in ("collection_id", "collection_revision"):
        _v2_digest(collection[field], field)
    runs = collection["run_ids"]
    if (
        type(runs) is not list
        or not runs
        or any(
            type(run) is not str
            or not (
                run.startswith("run_") and run[4:].isascii() and run[4:].isdecimal()
            )
            for run in runs
        )
        or runs != sorted(set(runs))
    ):
        raise ValueError(
            "simulation run IDs must be unique and ordered prefixed IDs (legacy numeric IDs require a new simulation)"
        )
    _v2_fields(simulator, {"name", "revision"}, "simulator")
    if simulator["name"] != "wflow":
        raise ValueError("unsupported simulator")
    _v2_digest(simulator["revision"], "simulator revision")
    _v2_fields(documents, _V2_DOCUMENTS, "simulation documents")
    _v2_model_reference(documents["model_reference"])
    settings = documents["settings"]
    from blueearth_cst.experiment.simulation_runs import include_unperturbed

    include_unperturbed(settings)
    _v2_fields(
        {key: value for key, value in settings.items() if key != "include_unperturbed"},
        {"simulation_window"},
        "settings",
    )
    window = _v2_fields(settings["simulation_window"], {"start", "end"}, "window")
    if (
        type(window["start"]) is not int
        or type(window["end"]) is not int
        or window["start"] < 1
        or window["end"] < window["start"]
    ):
        raise ValueError("invalid simulation window")
    from blueearth_cst.experiment.response_inventory import validate_response_request

    validate_response_request(documents["response_request"])
    if documents["response_request"]["run_ids"] != runs:
        raise ValueError("response request run IDs differ from selected collection")
    _v2_fields(
        preparation_identity,
        {
            "schema_version",
            "ancillary",
            "catalog",
            "forcing_elevation",
            "generated_forcing_reader",
            "pet_method",
        },
        "preparation identity",
    )
    if preparation_identity["schema_version"] != "forcing-preparation-identity/2":
        raise ValueError("unsupported preparation identity version")
    projections = {
        "settings": documents["settings"],
        "response_request": documents["response_request"],
        "preparation": preparation_identity,
    }
    intent = {
        "schema_version": "simulation-intent/2",
        "canonicalization_id": "collection-canon/1",
        "simulation_id": "",
        "experiment_name": experiment_name,
        "collection": collection,
        "simulator": simulator,
        "documents": documents,
        "document_digests": {
            key: content_sha256(documents[key]) for key in sorted(_V2_DOCUMENTS)
        },
        "identity_projections": projections,
        "identity_digests": {
            key: content_sha256(projections[key]) for key in sorted(_V2_PROJECTED)
        },
    }
    intent["simulation_id"] = _v2_simulation_id(intent)
    return intent


def read_simulation_intent_v2(experiment_root):
    """Check the retained v2 intent, its collection and preparation closure."""
    from blueearth_cst.experiment.scenario_collection_v2 import read_collection_v2
    from blueearth_cst.experiment.wf4_ancillary_descriptor import (
        preparation_identity_v2,
    )
    from blueearth_cst.shared.workflow_config_snapshot import resolve_file_reference

    root = Path(experiment_root).resolve()
    project = root.parent.parent
    path = _simulation_path(root, "_engine/simulation_intent.json")
    intent = read_canonical_json(path)
    _v2_fields(
        intent,
        {
            "schema_version",
            "canonicalization_id",
            "simulation_id",
            "experiment_name",
            "collection",
            "simulator",
            "documents",
            "document_digests",
            "identity_projections",
            "identity_digests",
        },
        "simulation intent",
    )
    if (
        intent["schema_version"] != "simulation-intent/2"
        or intent["canonicalization_id"] != "collection-canon/1"
        or intent["experiment_name"] != root.name
    ):
        raise SimulationFrozenError("unsupported or misplaced simulation intent")
    marker_path = resolve_file_reference(
        intent["collection"]["manifest"], {"project_root": project}
    )
    if intent["collection"]["manifest"]["path_base"] != "project_root":
        raise SimulationFrozenError("collection marker must bind project root")
    marker = read_collection_v2(marker_path)
    from blueearth_cst.experiment.simulation_runs import (
        include_unperturbed,
        selected_run_ids,
    )

    if (
        marker["collection_id"] != intent["collection"]["collection_id"]
        or marker["collection_revision"] != intent["collection"]["collection_revision"]
        or selected_run_ids(
            marker,
            marker_path,
            include=include_unperturbed(intent["documents"]["settings"]),
        )
        != intent["collection"]["run_ids"]
    ):
        raise SimulationFrozenError("selected collection differs from intent")
    projected = preparation_identity_v2(
        intent["documents"]["preparation"],
        project_root=project,
        experiment_root=root,
    )
    expected = simulation_intent_v2(
        root.name,
        intent["collection"],
        intent["simulator"],
        intent["documents"],
        projected,
    )
    if canonical_json_bytes(expected) != canonical_json_bytes(intent):
        raise SimulationFrozenError("retained simulation intent differs")
    return intent


def capture_simulation_sources_v2(config_path, project_root, invocation_id):
    """Capture exact WF4 source bytes before Snakemake parses its configfile."""
    from blueearth_cst.shared.config_composition import capture_configuration_sources
    from blueearth_cst.shared.workflow_config_snapshot import file_reference

    if (
        type(invocation_id) is not str
        or re.fullmatch(r"[0-9a-f]{32}", invocation_id) is None
    ):
        raise ValueError("WF4 source capture requires an invocation ID")
    root = Path(project_root).resolve()
    directory = _simulation_path(
        root, f"config/_engine/invocations/{invocation_id}/config"
    )
    sources = capture_configuration_sources(
        config_path,
        "simulate_system",
        ("project", "basin", "climate", "model", "workflows.simulate_system"),
    )
    inventory = []
    for source in sources:
        if not re.fullmatch(r"[a-z0-9_]+", source.id):
            raise ValueError("WF4 captured source has an invalid ID")
        target = _simulation_path(
            root,
            f"config/_engine/invocations/{invocation_id}/config/sources/{source.id}/{source.original_path.name}",
        )
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            if target.read_bytes() != source.data:
                raise SimulationFrozenError("WF4 source capture already differs")
        else:
            with target.open("xb") as handle:
                handle.write(source.data)
                handle.flush()
                os.fsync(handle.fileno())
        inventory.append(
            {
                "id": source.id,
                "role": source.role,
                "original_path": str(source.original_path),
                "capture": file_reference(target, "project_root", root),
            }
        )
    atomic_record(directory / "capture.json", {"sources": inventory})
    return directory / "capture.json"


def read_simulation_sources_v2(project_root, invocation_id):
    """Recover preparse WF4 buffers and refuse any live source drift."""
    from blueearth_cst.shared.workflow_config_snapshot import (
        CapturedSource,
        resolve_file_reference,
    )

    root = Path(project_root).resolve()
    manifest = read_canonical_json(
        _simulation_path(
            root, f"config/_engine/invocations/{invocation_id}/config/capture.json"
        )
    )
    _v2_fields(manifest, {"sources"}, "WF4 source capture")
    sources = []
    for item in manifest["sources"]:
        _v2_fields(
            item, {"id", "role", "original_path", "capture"}, "WF4 captured source"
        )
        path = resolve_file_reference(item["capture"], {"project_root": root})
        data = path.read_bytes()
        original = Path(item["original_path"]).resolve(strict=True)
        if (
            original.read_bytes() != data
            or hashlib.sha256(data).hexdigest() != item["capture"]["sha256"]
        ):
            raise SimulationFrozenError(
                "WF4 config source changed after preparse capture"
            )
        sources.append(CapturedSource(item["id"], item["role"], original, data))
    if not sources or sources[0].id != "project":
        raise SimulationFrozenError("WF4 source capture omits project")
    return tuple(sources)


def _assert_model_reference_current(retained, live, model_root):
    """Refuse model drift while naming the changed runtime input."""
    if retained == live:
        return
    from blueearth_cst.experiment.check_model_reference import compare_reference

    differences = compare_reference(retained, model_root)
    detail = "; ".join(differences) if differences else "reference metadata changed"
    raise SimulationFrozenError(
        f"retained model reference differs from live model: {detail}. "
        "Use a new experiment name."
    )


def live_simulation_inputs_v2(
    experiment_root,
    *,
    project_root,
    model_root,
    collection_path,
    collection,
    resolution_mode,
    preparation,
    simulation_window,
    julia_command,
    header_path,
    include_unperturbed=True,
):
    """Resolve all current scientific WF4 inputs into an immutable intent."""
    import tomllib

    from blueearth_cst.experiment.content_identity import stage_environment
    from blueearth_cst.experiment.forcing_window import forcing_window
    from blueearth_cst.experiment.simulation_code import simulation_code_inventory
    from blueearth_cst.experiment.simulator_adapter import plan_native_response_request
    from blueearth_cst.experiment.wf4_ancillary_descriptor import (
        preparation_identity_v2,
    )
    from blueearth_cst.experiment.write_model_reference import build_model_reference
    from blueearth_cst.shared.workflow_config_snapshot import file_reference

    root = Path(experiment_root).resolve()
    project = Path(project_root).resolve()
    model = Path(model_root).resolve()
    reference = build_model_reference(model, project)
    retained = _simulation_path(root, "_engine/model_reference.yml")
    import yaml

    retained_reference = yaml.safe_load(retained.read_bytes())
    _assert_model_reference_current(retained_reference, reference, model)
    with (model / reference["model_toml"]).open("rb") as handle:
        model_config = tomllib.load(handle)
    first, last = forcing_window(simulation_window["start"], simulation_window["end"])
    from blueearth_cst.experiment.simulation_runs import selected_run_ids

    run_ids = selected_run_ids(collection, collection_path, include=include_unperturbed)
    request = plan_native_response_request(
        model / reference["model_toml"],
        run_ids,
        julia_command=julia_command,
        header_path=header_path,
        first_time=first,
        last_time=last,
        calendar="standard",
        timestep_seconds=model_config["time"]["timestepsecs"],
    )
    repo = Path(__file__).resolve().parents[2]
    code = simulation_code_inventory(repo)
    environment = stage_environment(
        ["hydromt-wflow", "xarray", "netCDF4", "pyproj"],
        julia_manifest=repo / "Manifest.toml",
        julia_command=julia_command,
    )
    documents = {
        "model_reference": reference,
        "settings": {"simulation_window": dict(simulation_window)},
        "environment": environment,
        "simulator_adapter_code": code,
        "response_request": request,
        "preparation": preparation,
    }
    if not include_unperturbed:
        documents["settings"]["include_unperturbed"] = False
    selection = {
        "resolution_mode": resolution_mode,
        "manifest": file_reference(collection_path, "project_root", project),
        "collection_id": collection["collection_id"],
        "collection_revision": collection["collection_revision"],
        "run_ids": run_ids,
    }
    simulator = {
        "name": "wflow",
        "revision": content_sha256(
            {"julia:Wflow": environment["packages"]["julia:Wflow"]}
        ),
    }
    identity = preparation_identity_v2(
        preparation, project_root=project, experiment_root=root
    )
    return simulation_intent_v2(root.name, selection, simulator, documents, identity)


def freeze_simulation_v2(experiment_root, intent, *, invocation_id, command):
    """Publish intent and exact creator archive before native WF4 preparation."""
    from blueearth_cst.shared.config_composition import compose_captured_config
    from blueearth_cst.shared.provenance import (
        environment_file_hashes,
        toolbox_identity,
    )
    from blueearth_cst.shared.snake_utils import ADVANCED_SETTINGS
    from blueearth_cst.shared.workflow_config_snapshot import (
        file_reference,
        publish_archive,
        read_archive,
        run_record_document,
    )

    root = Path(experiment_root).resolve()
    project = root.parent.parent
    path = _simulation_path(root, "_engine/simulation_intent.json")
    reserved = (root / "_engine/reservation.json").exists()
    recipe_path = root / "_engine/simulation-freeze.json"
    if reserved and recipe_path.exists():
        from blueearth_cst.experiment.reservation_freeze import finish_freeze_owned

        recipe = read_canonical_json(recipe_path)
        if canonical_json_bytes(recipe["intent"]) != canonical_json_bytes(intent):
            raise SimulationFrozenError(
                "experiment freeze recipe differs; use a new experiment"
            )
        return finish_freeze_owned(root)
    if path.exists():
        retained = read_simulation_intent_v2(root)
        if canonical_json_bytes(retained) != canonical_json_bytes(intent):
            raise SimulationFrozenError("experiment intent differs; use a new name")
        archive = read_archive(
            project, intent["simulation_id"][:12], f"experiments/{root.name}/config"
        )
        if archive["owner"] != {"kind": "experiment", "id": intent["simulation_id"]}:
            raise SimulationFrozenError("creator archive owner differs")
        return retained
    elif (root / "config/run_record.yml").exists():
        raise SimulationFrozenError("unpaired experiment creator archive exists")
    if (root / "hydrology/wflow/output").exists() and any(
        (root / "hydrology/wflow/output").glob("run_*.csv")
    ):
        raise SimulationFrozenError("unrecorded native WF4 output exists")
    sources = read_simulation_sources_v2(project, invocation_id)
    projection = ("project", "basin", "climate", "model", "workflows.simulate_system")
    loaded, _ = compose_captured_config(
        sources[0], sources, "simulate_system", projection
    )
    if Path(loaded["project"]["project_dir"]).resolve() != project:
        raise SimulationFrozenError("captured WF4 project root differs")
    if not reserved and not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        atomic_record(path, intent)
    intent_bytes = canonical_json_bytes(intent)
    intent_reference = {
        "schema_version": "artifact-reference/1",
        "path_base": "project_root",
        "path": path.relative_to(project).as_posix(),
        "sha256": hashlib.sha256(intent_bytes).hexdigest(),
        "size_bytes": len(intent_bytes),
    }
    generated = [
        {
            "role": "simulation_intent",
            "file": intent_reference,
        },
        {
            "role": "model_reference",
            "file": file_reference(
                root / "_engine/model_reference.yml", "project_root", project
            ),
        },
        {
            "role": "forcing_elevation_catalog",
            "file": intent["documents"]["preparation"]["catalog"],
        },
    ]
    record, payloads = run_record_document(
        workflow="simulate_system",
        invocation_id=invocation_id,
        owner_kind="experiment",
        owner_id=intent["simulation_id"],
        loaded_config=loaded,
        advanced_settings=ADVANCED_SETTINGS,
        projection=projection,
        toolbox=toolbox_identity(),
        environment=environment_file_hashes(),
        invocation={
            "entry_point": "scripts/simulate_system.py",
            "command": [str(argument) for argument in command],
            "targets": ["all"],
            "working_directory": str(Path.cwd()),
            "overrides": {},
        },
        sources=sources,
        referenced_inputs=[],
        generated_inputs=generated,
    )
    if reserved:
        from blueearth_cst.experiment.reservation_freeze import (
            finish_freeze_owned,
            prepare_freeze_owned,
        )

        prepare_freeze_owned(root, intent, record, payloads)
        return finish_freeze_owned(root, recover=True)
    publish_archive(
        project,
        intent["simulation_id"][:12],
        f"experiments/{root.name}/config",
        record,
        payloads,
    )
    return read_simulation_intent_v2(root)


def read_simulation_v2(experiment_root):
    """Require checked response inventory and creator archive for readiness."""
    from blueearth_cst.experiment.response_inventory import read_response_inventory_v2
    from blueearth_cst.shared.workflow_config_snapshot import (
        resolve_file_reference,
        validate_archive,
    )

    root = Path(experiment_root).resolve()
    project = root.parent.parent
    marker_path = _simulation_path(root, "_engine/simulation.json")
    marker = read_canonical_json(marker_path)
    _v2_fields(
        marker,
        {
            "schema_version",
            "canonicalization_id",
            "status",
            "simulation_id",
            "intent",
            "archive",
            "response_inventory",
        },
        "simulation readiness",
    )
    if (
        marker["schema_version"] != "simulation/3"
        or marker["canonicalization_id"] != "collection-canon/1"
        or marker["status"] != "ready"
    ):
        raise IndicatorsOnlySimulationUnavailable("unsupported simulation marker")
    intent_path = resolve_file_reference(marker["intent"], {"project_root": project})
    if intent_path != _simulation_path(root, "_engine/simulation_intent.json"):
        raise IndicatorsOnlySimulationUnavailable(
            "simulation marker points to another intent"
        )
    intent = read_simulation_intent_v2(root)
    if marker["simulation_id"] != intent["simulation_id"]:
        raise IndicatorsOnlySimulationUnavailable("simulation marker identity differs")
    archive_path = resolve_file_reference(marker["archive"], {"project_root": project})
    if archive_path != _simulation_path(root, "config/run_record.yml"):
        raise IndicatorsOnlySimulationUnavailable(
            "simulation marker points to another archive"
        )
    archive = validate_archive(archive_path.parent)
    if archive["owner"] != {"kind": "experiment", "id": intent["simulation_id"]}:
        raise IndicatorsOnlySimulationUnavailable("simulation creator archive differs")
    inventory_path = resolve_file_reference(
        marker["response_inventory"], {"project_root": project}
    )
    if inventory_path != _simulation_path(root, "_engine/response_inventory.json"):
        raise IndicatorsOnlySimulationUnavailable(
            "simulation marker points to another inventory"
        )
    inventory = read_response_inventory_v2(root)
    if inventory["simulation_id"] != intent["simulation_id"]:
        raise IndicatorsOnlySimulationUnavailable("response inventory identity differs")
    return marker


def publish_simulation_v2(experiment_root, inventory):
    """Publish readiness only after every native response has been checked."""
    from blueearth_cst.experiment.response_inventory import read_response_inventory_v2
    from blueearth_cst.shared.workflow_config_snapshot import (
        file_reference,
        validate_archive,
    )

    root = Path(experiment_root).resolve()
    project = root.parent.parent
    intent = read_simulation_intent_v2(root)
    if read_response_inventory_v2(root) != inventory:
        raise SimulationFrozenError("response inventory differs before readiness")
    archive_path = _simulation_path(root, "config/run_record.yml")
    archive = validate_archive(archive_path.parent)
    if archive["owner"] != {"kind": "experiment", "id": intent["simulation_id"]}:
        raise SimulationFrozenError("creator archive owner differs")
    marker = {
        "schema_version": "simulation/3",
        "canonicalization_id": "collection-canon/1",
        "status": "ready",
        "simulation_id": intent["simulation_id"],
        "intent": file_reference(
            root / "_engine/simulation_intent.json", "project_root", project
        ),
        "archive": file_reference(archive_path, "project_root", project),
        "response_inventory": file_reference(
            root / "_engine/response_inventory.json", "project_root", project
        ),
    }
    path = _simulation_path(root, "_engine/simulation.json")
    if path.exists():
        if read_simulation_v2(root) != marker:
            raise SimulationFrozenError("ready simulation marker differs")
        return marker
    atomic_record(path, marker)
    return read_simulation_v2(root)
