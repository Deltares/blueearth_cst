"""Verified redo recipe for the reservation-owned simulation/archive pair."""

import hashlib
import stat
from pathlib import Path

import yaml

from blueearth_cst.experiment.content_identity import (
    atomic_record,
    canonical_json_bytes,
    confined_path,
    read_canonical_json,
)
from blueearth_cst.experiment.reservation import (
    bind_reservation_owned,
    read_reservation,
)
from blueearth_cst.shared.workflow_config_snapshot import (
    file_reference,
    publish_archive,
    read_archive,
    resolve_file_reference,
    validate_archive,
)


def _plain_path(project: Path, relative: str) -> Path:
    path = project / relative
    for part in (path, *path.parents):
        if part.exists() or part.is_symlink():
            attributes = getattr(part.lstat(), "st_file_attributes", 0)
            if part.is_symlink() or attributes & stat.FILE_ATTRIBUTE_REPARSE_POINT:
                raise ValueError(f"freeze path is redirected: {part}")
    return confined_path(project, relative)


def _write_exact(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(data)
        stream.flush()
        import os

        os.fsync(stream.fileno())


def prepare_freeze_owned(
    root: Path, intent: dict, record: dict, payloads: dict
) -> None:
    """Store complete exact creator evidence before publishing either paired record."""
    root = Path(root).resolve()
    project = root.parent.parent
    reservation = read_reservation(project, root.name)
    if reservation["resolution"] is None:
        raise ValueError("simulation freeze requires a bound ready collection")
    payload_root = _plain_path(
        project, f"experiments/{root.name}/_engine/freeze-payloads"
    )
    if payload_root.exists():
        raise ValueError(
            "partial freeze payloads exist without a completed recipe; preserve for diagnosis"
        )
    payload_root.mkdir(parents=True)
    references = []
    for relative, data in sorted(payloads.items()):
        path = _plain_path(
            project, f"experiments/{root.name}/_engine/freeze-payloads/{relative}"
        )
        _write_exact(path, data)
        references.append(
            {
                "archive_path": relative,
                "file": file_reference(path, "project_root", project),
            }
        )
    _write_exact(
        payload_root / "run_record.yml",
        yaml.safe_dump(record, sort_keys=False).encode("utf-8"),
    )
    checked = validate_archive(payload_root)
    if canonical_json_bytes(checked) != canonical_json_bytes(record):
        raise ValueError("prepared archive differs from freeze recipe")
    recipe = {
        "schema_version": "experiment-simulation-freeze/1",
        "reservation_id": reservation["reservation_id"],
        "simulation_id": intent["simulation_id"],
        "intent": intent,
        "archive_record": record,
        "payloads": references,
    }
    _validate_recipe(root, recipe)
    atomic_record(root / "_engine/simulation-freeze.json", recipe)


def _validate_recipe(root: Path, recipe: dict) -> tuple[dict, dict]:
    from blueearth_cst.experiment.check_model_reference import compare_reference
    from blueearth_cst.experiment.simulation_record import simulation_intent_v2
    from blueearth_cst.experiment.wf4_ancillary_descriptor import (
        preparation_identity_v2,
    )

    project = root.parent.parent
    if set(recipe) != {
        "schema_version",
        "reservation_id",
        "simulation_id",
        "intent",
        "archive_record",
        "payloads",
    }:
        raise ValueError("invalid simulation freeze recipe fields")
    if recipe["schema_version"] != "experiment-simulation-freeze/1":
        raise ValueError("unsupported simulation freeze recipe")
    reservation = read_reservation(project, root.name)
    if recipe["reservation_id"] != reservation["reservation_id"]:
        raise ValueError("simulation freeze reservation owner differs")
    supplied = recipe["intent"]
    rebuilt = simulation_intent_v2(
        root.name,
        supplied["collection"],
        supplied["simulator"],
        supplied["documents"],
        supplied["identity_projections"]["preparation"],
    )
    if canonical_json_bytes(rebuilt) != canonical_json_bytes(supplied):
        raise ValueError(
            "simulation freeze intent is invalid or names another experiment"
        )
    preparation = preparation_identity_v2(
        supplied["documents"]["preparation"],
        project_root=project,
        experiment_root=root,
    )
    if preparation != supplied["identity_projections"]["preparation"]:
        raise ValueError("simulation freeze preparation closure differs")
    differences = compare_reference(
        supplied["documents"]["model_reference"],
        project / "models/hydrology/wflow",
    )
    if differences:
        raise ValueError("simulation freeze model differs: " + "; ".join(differences))
    if recipe["simulation_id"] != recipe["intent"]["simulation_id"]:
        raise ValueError("simulation freeze intent identity differs")
    record = recipe["archive_record"]
    if record["owner"] != {"kind": "experiment", "id": recipe["simulation_id"]}:
        raise ValueError("simulation freeze archive owner differs")
    collection = recipe["intent"]["collection"]
    resolution = reservation["resolution"]
    if resolution is None or any(
        collection[key] != resolution[key]
        for key in ("collection_id", "collection_revision", "run_ids")
    ):
        raise ValueError("simulation freeze collection differs from reservation")
    payloads = {}
    for item in recipe["payloads"]:
        if (
            set(item) != {"archive_path", "file"}
            or item["file"]["path_base"] != "project_root"
        ):
            raise ValueError("invalid simulation freeze payload")
        relative = item["archive_path"]
        expected = _plain_path(
            project, f"experiments/{root.name}/_engine/freeze-payloads/{relative}"
        )
        actual = resolve_file_reference(item["file"], {"project_root": project})
        if actual != expected or relative in payloads:
            raise ValueError("simulation freeze payload is redirected or duplicated")
        payloads[relative] = actual.read_bytes()
    expected_paths = {entry["archived_path"] for entry in record["source_files"]} | {
        "sources/SOURCES.md"
        if record["archive_schema_version"] == "source-archive/1"
        else "SOURCES.md"
    }
    if set(payloads) != expected_paths:
        raise ValueError("simulation freeze payload inventory differs")
    for entry in record["source_files"]:
        data = payloads[entry["archived_path"]]
        if (
            len(data) != entry["size_bytes"]
            or hashlib.sha256(data).hexdigest() != entry["sha256"]
        ):
            raise ValueError("simulation freeze archive source bytes differ")
    checked = validate_archive(root / "_engine/freeze-payloads")
    if canonical_json_bytes(checked) != canonical_json_bytes(record):
        raise ValueError("simulation freeze prepared archive differs")
    intent_bytes = canonical_json_bytes(recipe["intent"])
    expected_intent = {
        "schema_version": "artifact-reference/1",
        "path_base": "project_root",
        "path": f"experiments/{root.name}/_engine/simulation_intent.json",
        "sha256": hashlib.sha256(intent_bytes).hexdigest(),
        "size_bytes": len(intent_bytes),
    }
    inputs = {item["role"]: item["file"] for item in record["generated_inputs"]}
    if inputs.get("simulation_intent") != expected_intent:
        raise ValueError("simulation freeze archive intent reference differs")
    return recipe["intent"], payloads


def finish_freeze_owned(root: Path, *, recover: bool = False) -> dict:
    """Finish only the retained recipe; ordinary incomplete retries require recovery."""
    from blueearth_cst.experiment.simulation_record import read_simulation_intent_v2

    root = Path(root).resolve()
    project = root.parent.parent
    _plain_path(project, f"experiments/{root.name}/config")
    _plain_path(project, f"experiments/{root.name}/_engine/simulation-freeze.json")
    recipe = read_canonical_json(root / "_engine/simulation-freeze.json")
    intent, payloads = _validate_recipe(root, recipe)
    path = root / "_engine/simulation_intent.json"
    archive_path = root / "config/run_record.yml"
    if path.exists() and canonical_json_bytes(
        read_simulation_intent_v2(root)
    ) != canonical_json_bytes(intent):
        raise ValueError("simulation intent differs from retained freeze recipe")
    if not path.exists() or not archive_path.exists():
        if not recover:
            raise ValueError("interrupted simulation freeze; use --recover-reservation")
        if any((root / "hydrology/wflow/output").glob("run_*")):
            raise ValueError(
                "native responses exist before complete simulation freeze; preserve for diagnosis"
            )
        config_directory = root / "config"
        if (
            not archive_path.exists()
            and config_directory.exists()
            and any(config_directory.iterdir())
        ):
            raise ValueError("unknown creator archive occupant; preserve for diagnosis")
        if not path.exists():
            atomic_record(path, intent)
        if not archive_path.exists():
            publish_archive(
                project,
                intent["simulation_id"][:12],
                f"experiments/{root.name}/config",
                recipe["archive_record"],
                payloads,
            )
    archive = read_archive(
        project, intent["simulation_id"][:12], f"experiments/{root.name}/config"
    )
    if canonical_json_bytes(archive) != canonical_json_bytes(recipe["archive_record"]):
        raise ValueError("published creator archive differs from freeze recipe")
    binding = {
        "simulation_id": intent["simulation_id"],
        "intent": file_reference(path, "project_root", project),
        "creator_archive": file_reference(archive_path, "project_root", project),
    }
    bind_reservation_owned(project, root.name, simulation=binding)
    return read_simulation_intent_v2(root)
