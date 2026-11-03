"""Real early admission guards precede producer calls and retain one durable owner."""

import copy
import hashlib
import logging
import shutil
import subprocess
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path

import pytest
import yaml

from blueearth_cst.experiment import reservation, reservation_context
from blueearth_cst.experiment import reservation_request as request_module
from blueearth_cst.experiment.reservation_request import capture_request
from blueearth_cst.experiment.response_inventory import publish_response_inventory_v2
from blueearth_cst.experiment.scenario_collection_v2 import read_collection_v2
from blueearth_cst.experiment.simulation_record import (
    capture_simulation_sources_v2,
    freeze_simulation_v2,
    publish_simulation_v2,
    read_simulation_intent_v2,
    simulation_intent_v2,
)
from blueearth_cst.experiment.wf4_ancillary_descriptor import preparation_identity_v2
from blueearth_cst.shared.workflow_config_snapshot import archive_lock
from scripts import generate_scenarios, run_workflows
from tests import _v2_experiment as helper

REPOSITORY = Path(__file__).resolve().parents[1]


@pytest.fixture
def catalog_console(monkeypatch):
    """Exercise the real HydroMT console logger, independent of pytest's setup."""
    logger = logging.getLogger("hydromt.data_catalog.data_catalog")
    monkeypatch.setattr(logger, "handlers", [])
    monkeypatch.setattr(logger, "level", logging.INFO)
    monkeypatch.setattr(logger, "propagate", False)
    yield logger


def write_yaml(path, document):
    path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")


@dataclass(frozen=True)
class Case:
    project: Path
    config: Path
    simulation: Path
    generation: Path
    catalog: Path
    legacy_root: Path | None = None
    marker: Path | None = None


def configuration(base, *, retained=None):
    project = retained.project if retained else base / "project"
    catalog = base / "tiny-catalog.yml"
    write_yaml(
        catalog,
        {
            "elevation": {
                "data_type": "RasterDataset",
                "uri": "elevation.tif",
                "driver": "rasterio",
                "data_adapter": {"unit_mult": {"elevtn": 1}},
            }
        },
    )
    generation = base / "generate_scenarios.yml"
    generated = yaml.safe_load(
        (
            REPOSITORY / "test_case/project_config_rapid_generate_scenarios.yml"
        ).read_bytes()
    )
    generated["seed"] = "auto"
    generated["n_realizations"] = 1
    write_yaml(generation, generated)
    simulation = base / "simulate_system.yml"
    settings = {"operation": "simulate-and-indicators"}
    if retained:
        settings["scenario_collection"] = {
            "manifest_path": str(retained.collection_marker)
        }
    write_yaml(simulation, settings)
    write_yaml(
        base / "build_model.yml",
        {"simulation_window": {"start": 2000, "end": 2016}},
    )
    workflows = {
        name: {"enabled": False} for name in ("analyze_climate", "analyze_projections")
    }
    workflows.update(
        build_model={"enabled": False, "config_path": "build_model.yml"},
        generate_scenarios={
            "enabled": retained is None,
            "config_path": generation.name,
        },
        simulate_system={"enabled": True, "config_path": simulation.name},
    )
    config = base / "project.yml"
    write_yaml(
        config,
        {
            "schema_version": 2,
            "project": {"project_dir": str(project), "catalog": [str(catalog)]},
            "basin": {"region": "{'subbasin': [9.666, 0.4476]}", "resolution": 0.00833},
            "climate": {"selected": "era5", "window": {"start": 2000, "end": 2016}},
            "model": {"outvars": ["gwr"]},
            "workflows": workflows,
        },
    )
    return Case(
        project,
        config,
        simulation,
        generation,
        catalog,
        retained.root if retained else None,
        retained.collection_marker if retained else None,
    )


@pytest.fixture
def pending(tmp_path):
    base = tmp_path / "pending"
    base.mkdir()
    return configuration(base)


@pytest.fixture
def retained(tmp_path):
    base = tmp_path / "retained"
    base.mkdir()
    fixture = helper.build_v2_experiment(base)
    return configuration(base, retained=fixture)


def admit(case, **options):
    with archive_lock(case.project, "project-execution"):
        return reservation_context.admit_context_owned(
            case.config,
            REPOSITORY,
            case.project,
            uuid.uuid4().hex,
            execute_generation=case.marker is None,
            **options,
        )


def durable_inventory(case):
    """Invocation histories may grow on failed attempts; experiment evidence may not."""
    return {
        path.relative_to(case.project).as_posix(): (
            "directory"
            if path.is_dir()
            else hashlib.sha256(path.read_bytes()).hexdigest()
        )
        for prefix in ("experiments",)
        for path in (case.project / prefix).rglob("*")
    }


def bind(case, context, *, freeze=False):
    with archive_lock(case.project, "project-execution"):
        context.bind_selection()
        if not freeze:
            return
        root = case.project / "experiments" / context.name
        original = read_simulation_intent_v2(case.legacy_root)
        shutil.copyfile(
            case.legacy_root / "_engine/model_reference.yml",
            root / "_engine/model_reference.yml",
        )
        documents = copy.deepcopy(original["documents"])
        documents["preparation"] = helper._preparation(
            case.project, root, case.config.parent / "new-preparation"
        )
        projection = preparation_identity_v2(
            documents["preparation"], project_root=case.project, experiment_root=root
        )
        resolution = reservation.read_reservation(case.project, context.name)[
            "resolution"
        ]
        from blueearth_cst.experiment.response_inventory import make_response_request

        documents["response_request"] = make_response_request(
            resolution["run_ids"], documents["response_request"]["variables"]
        )
        if not yaml.safe_load(case.simulation.read_bytes()).get(
            "include_unperturbed", True
        ):
            documents["settings"]["include_unperturbed"] = False
        intent = simulation_intent_v2(
            context.name, resolution, original["simulator"], documents, projection
        )
        invocation = uuid.uuid4().hex
        capture_simulation_sources_v2(case.config, case.project, invocation)
        freeze_simulation_v2(
            root, intent, invocation_id=invocation, command=["fixture"]
        )


def test_unperturbed_switch_binds_only_perturbed_runs_and_refuses_changes(retained):
    settings = yaml.safe_load(retained.simulation.read_bytes())
    settings["include_unperturbed"] = False
    write_yaml(retained.simulation, settings)
    context = admit(retained)
    bind(retained, context, freeze=True)
    record = reservation.read_reservation(retained.project, context.name)
    assert record["resolution"]["run_ids"] == ["run_2"]
    assert record["simulation"] is not None
    assert admit(retained).action == "continue"
    settings["include_unperturbed"] = True
    write_yaml(retained.simulation, settings)
    before = durable_inventory(retained)
    with pytest.raises(ValueError, match="include_unperturbed"):
        admit(retained)
    assert durable_inventory(retained) == before


def reject_producers(monkeypatch):
    calls = []

    def producer(*args, **kwargs):
        calls.append((args, kwargs))
        pytest.fail("producer boundary reached before scientific admission refusal")

    monkeypatch.setattr(generate_scenarios, "run_owned", producer)
    monkeypatch.setattr(run_workflows, "run_project_child", producer)
    from blueearth_cst.shared import workflow_archive_launch

    monkeypatch.setattr(workflow_archive_launch, "prepare_workflow", producer)
    return calls


def test_admission_can_return_selection_without_a_console_announcement(pending, capsys):
    context = admit(pending, announce=False)
    assert context.action == "create"
    assert "Selected experiment:" not in capsys.readouterr().out
    context = admit(pending)
    assert context.action == "continue"
    assert (
        f"Selected experiment: {context.name}; action: continue"
        in capsys.readouterr().out
    )


def test_initial_pending_continue_new_and_exact_pin_use_durable_ids(pending):
    first = admit(pending)
    assert first.name == "stress_test_01"
    assert first.reservation_id
    assert first.environment() == {
        "CST_EXPERIMENT_NAME": first.name,
        "CST_EXPERIMENT_RESERVATION_ID": first.reservation_id,
    }
    assert (
        reservation.read_reservation(pending.project, first.name)["state"] == "pending"
    )
    for namespace in ("scenarios", "models", "data"):
        assert not (pending.project / namespace).exists()
    before = durable_inventory(pending)
    repeated = admit(pending)
    assert (repeated.name, repeated.reservation_id) == (
        first.name,
        first.reservation_id,
    )
    assert durable_inventory(pending) == before
    second = admit(pending, new_experiment=True)
    assert second.name == "stress_test_02"
    assert second.reservation_id != first.reservation_id
    settings = yaml.safe_load(pending.simulation.read_bytes())
    settings["experiment_name"] = "stress_test_01"
    write_yaml(pending.simulation, settings)
    pinned = admit(pending)
    assert pinned.reservation_id == first.reservation_id
    before = durable_inventory(pending)
    with pytest.raises(ValueError, match="numbered pin"):
        admit(pending, new_experiment=True)
    assert durable_inventory(pending) == before


def test_new_flag_with_disabled_simulation_refuses_before_producers(
    pending, monkeypatch
):
    config = yaml.safe_load(pending.config.read_bytes())
    config["workflows"]["simulate_system"]["enabled"] = False
    write_yaml(pending.config, config)
    monkeypatch.setattr(
        generate_scenarios,
        "run_owned",
        lambda *args, **kwargs: pytest.fail("producer started for invalid flags"),
    )
    with pytest.raises(ValueError, match="require enabled simulate_system"):
        run_workflows.run(str(pending.config), 1, [], new_experiment=True)
    assert not (pending.project / "experiments").exists()
    assert not (pending.project / "experiments/_engine").exists()


def test_public_runner_reserves_before_first_producer(
    pending, monkeypatch, capfd, catalog_console
):
    # Bind in the test call phase: pytest replaces streams after fixture setup.
    catalog_console.addHandler(logging.StreamHandler(sys.stderr))
    observed = []
    catalog_reads = []
    original_catalog_value = request_module._catalog_value

    def catalog_value(path):
        catalog_reads.append(path)
        return original_catalog_value(path)

    monkeypatch.setattr(request_module, "_catalog_value", catalog_value)

    def first_producer(*args, **kwargs):
        context = kwargs["reservation_context"]
        reservation_record = reservation.read_reservation(pending.project, context.name)
        assert reservation_record["reservation_id"] == context.reservation_id
        assert reservation_record["state"] == "pending"
        assert context.name == "stress_test_01"
        assert not (pending.project / "scenarios").exists()
        observed.append(context.reservation_id)
        return 17  # Stop at the observed boundary without executing scientific work.

    monkeypatch.setattr(generate_scenarios, "run_owned", first_producer)
    monkeypatch.setattr(
        run_workflows,
        "run_project_child",
        lambda *args, **kwargs: pytest.fail("later producer reached"),
    )
    assert (
        run_workflows.run(str(pending.config), 1, [], project_dir=pending.project) == 17
    )
    assert len(observed) == 1
    assert len(catalog_reads) >= 3  # Admission, its recheck, and the first handoff.
    captured = capfd.readouterr()
    output = captured.out + captured.err
    assert "run_workflows" in output
    assert "Parsing data catalog" not in output
    assert "Logging error" not in output
    assert (
        reservation.read_reservation(pending.project, "stress_test_01")[
            "reservation_id"
        ]
        == observed[0]
    )


@pytest.mark.process_isolation
@pytest.mark.parametrize("state", ["pending", "simulation_bound"])
def test_admission_and_recheck_do_not_import_hydromt(pending, retained, state):
    """Block direct and transitive imports in a clean interpreter."""
    program = """
import importlib.abc
import sys
import uuid
from pathlib import Path

class BlockHydroMT(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'hydromt' or fullname.startswith('hydromt.'):
            raise AssertionError('admission imported HydroMT')

sys.meta_path.insert(0, BlockHydroMT())
from blueearth_cst.experiment.reservation_context import admit_context_owned
from blueearth_cst.shared.workflow_config_snapshot import archive_lock

config, root = map(Path, sys.argv[1:3])
generation = sys.argv[3] == 'pending'
with archive_lock(root, 'project-execution'):
    context = admit_context_owned(
        config, Path.cwd(), root, uuid.uuid4().hex,
        execute_generation=generation, announce=False,
    )
    context.recheck()
assert 'hydromt' not in sys.modules
assert context.request['schema_version'] == 'experiment-request/2'
print('admission checks complete')
"""
    case = pending if state == "pending" else retained
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            program,
            str(case.config),
            str(case.project),
            state,
        ],
        cwd=REPOSITORY,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.strip() == "admission checks complete"
    assert result.stderr == ""


@pytest.mark.parametrize("state", ["pending", "resolved", "simulation_bound"])
@pytest.mark.parametrize("change", ["catalog", "science"])
def test_changed_request_refuses_public_runner_before_producers(
    pending, retained, monkeypatch, state, change
):
    case = pending if state == "pending" else retained
    context = admit(case)
    if state != "pending":
        bind(case, context, freeze=state == "simulation_bound")
    if change == "catalog":
        document = yaml.safe_load(case.catalog.read_bytes())
        document["elevation"]["data_adapter"]["unit_mult"]["elevtn"] = 1000
        write_yaml(case.catalog, document)
    else:
        document = yaml.safe_load(case.config.read_bytes())
        document["model"]["outvars"] = ["actual evapotranspiration"]
        write_yaml(case.config, document)
    before = durable_inventory(case)
    calls = reject_producers(monkeypatch)
    with pytest.raises(ValueError):
        run_workflows.run(str(case.config), 1, [], project_dir=case.project)
    assert not calls
    assert durable_inventory(case) == before
    retained_record = reservation.read_reservation(case.project, context.name)
    assert retained_record["reservation_id"] == context.reservation_id
    assert retained_record["state"] == state
    if state == "pending" and change == "science":
        fresh = admit(case, new_experiment=True)
        assert fresh.name == "stress_test_02"
        assert fresh.reservation_id != context.reservation_id
        assert (
            reservation.read_reservation(case.project, context.name) == retained_record
        )
        assert (
            reservation.read_reservation(case.project, fresh.name)["request"]
            != retained_record["request"]
        )


@pytest.mark.parametrize("owner", ["legacy", "reserved"])
def test_indicators_only_uses_retained_readiness_without_new_input_reads(
    retained, monkeypatch, owner
):
    name = helper.EXPERIMENT
    reservation_id = None
    if owner == "reserved":
        context = admit(retained)
        bind(retained, context, freeze=True)
        name, reservation_id = context.name, context.reservation_id
        root = retained.project / "experiments" / name
        source_runs, target_runs = (
            helper.native_runs(retained.legacy_root),
            helper.native_runs(root),
        )
        for run_id, target in target_runs.items():
            source = source_runs[run_id]
            for field in ("csv_path", "toml_path", "temporal_path"):
                target_path = getattr(target, field)
                target_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(getattr(source, field), target_path)
        with archive_lock(retained.project, "project-execution"):
            response_inventory = publish_response_inventory_v2(
                root, target_runs, helper.default_responses().temporal
            )
            publish_simulation_v2(root, response_inventory)
    settings = {
        "experiment_name": name,
        "operation": "indicators-only",
        "indicators": ["gwr"],
    }
    write_yaml(retained.simulation, settings)
    retained.generation.unlink()
    (retained.config.parent / "build_model.yml").unlink()
    retained.catalog.unlink()
    (retained.project / "models/hydrology/wflow/wflow_sbm.toml").unlink()
    monkeypatch.setattr(
        reservation_context,
        "capture_request",
        lambda *args, **kwargs: pytest.fail("indicators-only captured a new Request"),
    )
    monkeypatch.setattr(
        reservation_context,
        "resolve_selected_collection",
        lambda *args, **kwargs: pytest.fail("indicators-only read generation inputs"),
    )
    before = durable_inventory(retained)
    with pytest.raises(ValueError, match="indicators-only"):
        admit(retained, new_experiment=True)
    assert durable_inventory(retained) == before
    context = admit(retained)
    assert context.name == name
    assert context.reservation_id == reservation_id
    assert context.request is None
    context.recheck()
    assert durable_inventory(retained) == before
    # A dry run selects the same retained experiment without a request to compare.
    preview = admit(retained, preview=True)
    assert (preview.name, preview.request) == (name, None)
    assert durable_inventory(retained) == before
    if owner == "legacy":
        assert not (retained.project / "experiments/_engine/allocation.json").exists()


def test_changed_catalog_during_invocation_recheck_refuses_and_preserves_owner(pending):
    context = admit(pending)
    document = yaml.safe_load(pending.catalog.read_bytes())
    document["elevation"]["uri"] = "different-elevation.tif"
    write_yaml(pending.catalog, document)
    before = durable_inventory(pending)
    with pytest.raises(ValueError):
        context.recheck()
    assert durable_inventory(pending) == before
    assert (
        reservation.read_reservation(pending.project, context.name)["reservation_id"]
        == context.reservation_id
    )


@pytest.mark.parametrize(
    "mode", ["same retained collection", "different collection", "active generation"]
)
def test_bound_selector_mode_policy_uses_exact_retained_identity(
    retained, monkeypatch, mode
):
    """Policy unit: real Requests/readers; only already-validated durable binding is stubbed."""
    original, _ = capture_request(
        retained.config, REPOSITORY, execute_generation=True, execute_model=True
    )
    original["selection"] = {"mode": "project-generation", "manifest": None}
    requested, _ = capture_request(retained.config, REPOSITORY)
    reservation.validate_request(original, retained.project)
    reservation.validate_request(requested, retained.project)
    marker = read_collection_v2(retained.marker)
    resolution = {
        "collection_id": marker["collection_id"],
        "collection_revision": marker["collection_revision"],
        "run_ids": [entry["run_id"] for entry in marker["series"]],
    }
    if mode == "different collection":
        resolution["collection_id"] = "0" * 64
    monkeypatch.setattr(
        reservation_context,
        "read_reservation",
        lambda *args: {"resolution": resolution},
    )
    context = reservation_context.ReservationContext(
        retained.project,
        retained.config,
        REPOSITORY,
        "stress_test_01",
        uuid.uuid4().hex,
        mode == "active generation",
        False,
        original,
    )
    before_original, before_requested = (
        copy.deepcopy(original),
        copy.deepcopy(requested),
    )
    before = durable_inventory(retained)
    if mode == "active generation":
        monkeypatch.setattr(
            reservation_context,
            "resolve_selected_collection",
            lambda *args: pytest.fail("changed science reached collection lookup"),
        )
    if mode == "same retained collection":
        context.compare(original, requested)
    else:
        with pytest.raises(ValueError):
            context.compare(original, requested)
    assert original == before_original
    assert requested == before_requested
    assert durable_inventory(retained) == before


@pytest.mark.parametrize(
    "change",
    [
        "missing pointer",
        "changed science",
        "different collection",
        "standalone",
        "stale pointer",
    ],
)
def test_generation_can_restore_pointer_only_for_unchanged_bound_collection(
    retained, monkeypatch, change
):
    from blueearth_cst.experiment.collection_resolution import (
        GeneratedCollectionStale,
        GeneratedCollectionUnavailable,
    )
    from blueearth_cst.shared.workflow_config_snapshot import file_reference

    original, _ = capture_request(retained.config, REPOSITORY, execute_generation=True)
    original["selection"] = {"mode": "project-generation", "manifest": None}
    requested = copy.deepcopy(original)
    marker = read_collection_v2(retained.marker)
    resolution = {
        "manifest": file_reference(retained.marker, "project_root", retained.project),
        "collection_id": marker["collection_id"],
        "collection_revision": marker["collection_revision"],
        "run_ids": [entry["run_id"] for entry in marker["series"]],
    }
    if change == "changed science":
        requested["generation"]["n_realizations"] += 1
    if change == "different collection":
        resolution["collection_id"] = "0" * 64
    monkeypatch.setattr(
        reservation_context,
        "read_reservation",
        lambda *args: {"resolution": resolution},
    )

    def missing(*args):
        if change == "changed science":
            pytest.fail("changed science reached collection lookup")
        error = (
            GeneratedCollectionStale
            if change == "stale pointer"
            else GeneratedCollectionUnavailable
        )
        raise error("project-generation request pointer missing or stale")

    monkeypatch.setattr(reservation_context, "resolve_selected_collection", missing)
    context = reservation_context.ReservationContext(
        retained.project,
        retained.config,
        REPOSITORY,
        "stress_test_01",
        uuid.uuid4().hex,
        change != "standalone",
        False,
        original,
    )
    before = durable_inventory(retained)
    if change == "missing pointer":
        context.compare(original, requested)
    else:
        with pytest.raises(ValueError):
            context.compare(original, requested)
    assert durable_inventory(retained) == before
