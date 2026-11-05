"""Reservation-owned simulation freeze uses verified retained creator evidence."""

import copy
import hashlib
import shutil
import uuid
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import pytest

from blueearth_cst.experiment import reservation, reservation_freeze
from blueearth_cst.experiment.content_identity import atomic_record, read_canonical_json
from blueearth_cst.experiment.scenario_collection_v2 import read_collection_v2
from blueearth_cst.experiment.simulation_record import (
    capture_simulation_sources_v2,
    freeze_simulation_v2,
    read_simulation_intent_v2,
    simulation_intent_v2,
)
from blueearth_cst.experiment.wf4_ancillary_descriptor import preparation_identity_v2
from blueearth_cst.shared.workflow_config_snapshot import (
    archive_lock,
    file_reference,
    validate_archive,
)
from tests import _v2_experiment as helper


@dataclass(frozen=True)
class PendingFreeze:
    project: Path
    root: Path
    intent: dict
    invocation: str
    legacy_root: Path

    @property
    def recipe(self):
        return self.root / "_engine/simulation-freeze.json"

    @property
    def intent_path(self):
        return self.root / "_engine/simulation_intent.json"

    @property
    def archive_path(self):
        return self.root / "config/run_record.yml"


@pytest.fixture(scope="module")
def built(tmp_path_factory: pytest.TempPathFactory) -> Iterator[helper.V2Experiment]:
    """Build retained evidence once and ensure test mutations never reach it."""
    fixture = helper.build_v2_experiment(tmp_path_factory.mktemp("freeze-base"))
    before = inventory(fixture.config.parent)
    yield fixture
    assert inventory(fixture.config.parent) == before


@pytest.fixture
def pending(built: helper.V2Experiment, tmp_path: Path) -> PendingFreeze:
    """Copy retained evidence; admit and prepare a private reservation for each case."""
    fixture = helper.copy_v2_experiment(built, tmp_path)
    project = fixture.project
    marker = read_collection_v2(fixture.collection_marker)
    manifest = file_reference(fixture.collection_marker, "project_root", project)
    request = {
        "schema_version": "experiment-request/1",
        "generation": None,
        "selection": {"mode": "explicit-manifest", "manifest": manifest},
        "input_catalogs": [],
        "simulation": {
            "climate_source": "era5",
            "model_root": "models/hydrology/wflow",
            "outvars": ["gwr"],
            "model_request": None,
        },
    }
    sources = [
        reservation.CapturedSource(role, path, path.read_bytes())
        for role, path in (
            ("project", fixture.config),
            ("simulate_system", tmp_path / "simulate_system.yml"),
        )
    ]
    resolution = {
        "resolution_mode": "explicit-manifest",
        "manifest": manifest,
        "collection_id": marker["collection_id"],
        "collection_revision": marker["collection_revision"],
        "run_ids": [item["run_id"] for item in marker["series"]],
    }
    with archive_lock(project, "project-execution"):
        selected = reservation.admit_experiment_owned(
            project,
            request=request,
            sources=sources,
            creator_invocation_id=uuid.uuid4().hex,
        )
        reservation.bind_reservation_owned(
            project, selected.name, resolution=resolution
        )
    root = project / "experiments" / selected.name
    shutil.copyfile(
        fixture.root / "_engine/model_reference.yml",
        root / "_engine/model_reference.yml",
    )
    retained = read_simulation_intent_v2(fixture.root)
    documents = copy.deepcopy(retained["documents"])
    documents["preparation"] = helper._preparation(
        project, root, tmp_path / "reserved-preparation"
    )
    identity = preparation_identity_v2(
        documents["preparation"], project_root=project, experiment_root=root
    )
    intent = simulation_intent_v2(
        root.name, resolution, retained["simulator"], documents, identity
    )
    invocation = uuid.uuid4().hex
    capture_simulation_sources_v2(fixture.config, project, invocation)
    return PendingFreeze(project, root, intent, invocation, fixture.root)


def inventory(root):
    """Keep every diagnostic file; lock contents are inaccessible under Windows ownership."""
    return {
        path.relative_to(root).as_posix(): (
            ("link", str(path.readlink()))
            if path.is_symlink()
            else (
                "directory"
                if path.is_dir()
                else (
                    ("lock", path.stat().st_size)
                    if path.suffix == ".lock"
                    else hashlib.sha256(path.read_bytes()).hexdigest()
                )
            )
        )
        for path in root.rglob("*")
    }


def freeze(pending):
    with archive_lock(pending.project, "project-execution"):
        return freeze_simulation_v2(
            pending.root,
            pending.intent,
            invocation_id=pending.invocation,
            command=["synthetic-freeze-fixture"],
        )


def recover(pending):
    with archive_lock(pending.project, "project-execution"):
        return reservation_freeze.finish_freeze_owned(pending.root, recover=True)


def assert_bound_pair(pending):
    assert read_simulation_intent_v2(pending.root) == pending.intent
    archive = validate_archive(pending.archive_path.parent)
    assert archive["owner"] == {
        "kind": "experiment",
        "id": pending.intent["simulation_id"],
    }
    bound = reservation.read_reservation(pending.project, pending.root.name)
    assert bound["state"] == "simulation_bound"
    assert bound["revision"] == 2
    assert bound["simulation"]["intent"] == file_reference(
        pending.intent_path, "project_root", pending.project
    )
    assert bound["simulation"]["creator_archive"] == file_reference(
        pending.archive_path, "project_root", pending.project
    )
    assert archive == read_canonical_json(pending.recipe)["archive_record"]
    return bound


def interrupt(pending, monkeypatch, boundary):
    """Fault actual publication, retaining all writes preceding the selected boundary."""
    with monkeypatch.context() as patch:
        if boundary in {"before recipe", "before intent"}:
            original = reservation_freeze.atomic_record
            fault_path = (
                pending.recipe if boundary == "before recipe" else pending.intent_path
            )

            def fault(path, document):
                if Path(path) == fault_path:
                    raise RuntimeError(f"injected {boundary}")
                return original(path, document)

            patch.setattr(reservation_freeze, "atomic_record", fault)
        elif boundary == "before binding":

            def fault(*args, **kwargs):
                raise RuntimeError("injected before binding")

            patch.setattr(reservation_freeze, "bind_reservation_owned", fault)
        else:
            original = reservation_freeze.publish_archive

            def fault(*args, **kwargs):
                if boundary == "after intent":
                    raise RuntimeError("injected after intent")
                return original(*args, **kwargs, failure_after=boundary)

            patch.setattr(reservation_freeze, "publish_archive", fault)
        with pytest.raises(RuntimeError, match="injected"):
            freeze(pending)


@pytest.mark.parametrize("legacy_archive", [False, True])
def test_initial_freeze_publishes_checked_pair_and_repeat_is_byte_noop(
    pending, monkeypatch, legacy_archive
):
    from blueearth_cst.shared import workflow_config_snapshot as snapshot
    from tests.test_source_archive import _legacy_archive

    writer = snapshot.run_record_document
    with monkeypatch.context() as patch:
        if legacy_archive:
            patch.setattr(
                snapshot,
                "run_record_document",
                lambda **kwargs: _legacy_archive(*writer(**kwargs)),
            )
        assert freeze(pending) == pending.intent
    assert_bound_pair(pending)
    before = inventory(pending.project)
    assert freeze(pending) == pending.intent
    assert recover(pending) == pending.intent
    assert recover(pending) == pending.intent
    assert inventory(pending.project) == before


@pytest.mark.parametrize(
    "boundary",
    [
        "before intent",
        "after intent",
        "prepared",
        "old_detached",
        "new_installed",
        "before binding",
    ],
)
def test_interrupted_verified_freeze_recovers_twice(pending, monkeypatch, boundary):
    interrupt(pending, monkeypatch, boundary)
    assert pending.recipe.exists()
    assert pending.intent_path.exists() == (boundary != "before intent")
    assert pending.archive_path.exists() == (
        boundary in {"new_installed", "before binding"}
    )
    assert (
        reservation.read_reservation(pending.project, pending.root.name)["state"]
        == "resolved"
    )
    if boundary not in {"new_installed", "before binding"}:
        before = inventory(pending.project)
        with pytest.raises(ValueError, match="recover-reservation"):
            freeze(pending)
        assert inventory(pending.project) == before
    assert recover(pending) == pending.intent
    assert_bound_pair(pending)
    before = inventory(pending.project)
    assert recover(pending) == pending.intent
    assert inventory(pending.project) == before


def test_partial_payloads_without_recipe_refuse_and_remain_unchanged(
    pending, monkeypatch
):
    interrupt(pending, monkeypatch, "before recipe")
    assert not pending.recipe.exists()
    assert not pending.intent_path.exists()
    assert not pending.archive_path.exists()
    before = inventory(pending.project)
    with pytest.raises(ValueError, match="partial freeze payloads"):
        freeze(pending)
    with pytest.raises(OSError):
        recover(pending)
    assert inventory(pending.project) == before


@pytest.mark.parametrize(
    "damage",
    [
        "owner",
        "identity",
        "unknown field",
        "payload bytes",
        "archive bytes",
        "payload redirected",
        "duplicate payload",
    ],
)
def test_corrupted_recipe_or_payload_refuses_without_mutation(
    pending, monkeypatch, damage
):
    interrupt(pending, monkeypatch, "before intent")
    recipe = read_canonical_json(pending.recipe)
    if damage == "owner":
        recipe["reservation_id"] = "0" * 32
    elif damage == "identity":
        recipe["simulation_id"] = "0" * 64
    elif damage == "unknown field":
        recipe["unknown"] = True
    elif damage == "payload bytes":
        relative = recipe["payloads"][0]["file"]["path"]
        (pending.project / relative).write_bytes(b"corrupted creator source")
    elif damage == "archive bytes":
        (pending.root / "_engine/freeze-payloads/run_record.yml").write_bytes(
            b"invalid: archive\n"
        )
    elif damage == "payload redirected":
        original = pending.project / recipe["payloads"][0]["file"]["path"]
        redirected = pending.project / "foreign-source.yml"
        redirected.write_bytes(original.read_bytes())
        recipe["payloads"][0]["file"] = file_reference(
            redirected, "project_root", pending.project
        )
    else:
        recipe["payloads"].append(copy.deepcopy(recipe["payloads"][0]))
    if damage not in {"payload bytes", "archive bytes"}:
        atomic_record(pending.recipe, recipe, replace=True)
    before = inventory(pending.project)
    with pytest.raises((ValueError, OSError)):
        recover(pending)
    assert inventory(pending.project) == before
    assert not pending.intent_path.exists()
    assert not pending.archive_path.exists()


@pytest.mark.parametrize(
    "occupant", ["archive", "archive before intent", "intent", "native response"]
)
def test_unknown_occupants_preserve_interruption_evidence(
    pending, monkeypatch, occupant
):
    interrupt(
        pending,
        monkeypatch,
        "before intent" if occupant == "archive before intent" else "after intent",
    )
    if occupant in {"archive", "archive before intent"}:
        target = pending.root / "config/foreign.yml"
    elif occupant == "intent":
        target = pending.intent_path
    else:
        target = pending.root / "hydrology/wflow/output/run_unknown.csv"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(b"unowned bytes\n")
    before = inventory(pending.project)
    with pytest.raises((ValueError, OSError)):
        recover(pending)
    assert inventory(pending.project) == before
    assert not pending.archive_path.exists()
    if occupant == "archive before intent":
        assert not pending.intent_path.exists()


def test_legacy_partial_pair_keeps_existing_refusal_and_no_recipe(pending):
    root = pending.legacy_root
    intent = read_simulation_intent_v2(root)
    (root / "config/run_record.yml").unlink()
    before = inventory(pending.project)
    with (
        archive_lock(pending.project, "project-execution"),
        pytest.raises((ValueError, OSError)),
    ):
        freeze_simulation_v2(
            root, intent, invocation_id=pending.invocation, command=["fixture"]
        )
    assert inventory(pending.project) == before
    assert not (root / "_engine/simulation-freeze.json").exists()
    assert not (root / "_engine/reservation.json").exists()


def test_current_model_drift_refuses_retained_recipe_without_mutation(
    pending, monkeypatch
):
    interrupt(pending, monkeypatch, "before intent")
    model = pending.project / "models/hydrology/wflow/wflow_sbm.toml"
    model.write_bytes(
        model.read_bytes() + b"\n# model changed after freeze preparation\n"
    )
    before = inventory(pending.project)
    with pytest.raises(ValueError, match="model differs"):
        recover(pending)
    assert inventory(pending.project) == before
    assert not pending.intent_path.exists()
    assert not pending.archive_path.exists()


def test_payload_link_with_identical_bytes_refuses_without_following_alias(
    pending, monkeypatch
):
    interrupt(pending, monkeypatch, "before intent")
    recipe = read_canonical_json(pending.recipe)
    payload = pending.project / recipe["payloads"][0]["file"]["path"]
    foreign = pending.project / "foreign-source.yml"
    original = payload.read_bytes()
    foreign.write_bytes(original)
    payload.unlink()
    payload.symlink_to(foreign)
    before = inventory(pending.project)
    with pytest.raises(ValueError, match="redirected"):
        recover(pending)
    assert inventory(pending.project) == before
    assert payload.is_symlink()
    assert foreign.read_bytes() == original
    assert not pending.intent_path.exists()
    assert not pending.archive_path.exists()
