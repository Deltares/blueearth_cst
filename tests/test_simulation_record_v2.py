"""Simulation v2 freezes exact sources and the closed scientific intent."""

import pytest
import yaml

from blueearth_cst.experiment.content_identity import content_sha256
from blueearth_cst.experiment.response_inventory import make_response_request
from blueearth_cst.experiment.simulation_record import (
    SimulationFrozenError,
    _assert_model_reference_current,
    capture_simulation_sources_v2,
    read_simulation_sources_v2,
    simulation_intent_v2,
)
from blueearth_cst.experiment.write_model_reference import build_model_reference
from blueearth_cst.shared.model_digest import model_digest_from_entries


def test_preparse_wf4_sources_refuse_comment_mutation(tmp_path):
    config = tmp_path / "project_config.yml"
    workflow = tmp_path / "simulate_system.yml"
    workflow.write_text("operation: simulate-and-indicators\n", encoding="utf-8")
    config.write_text(
        yaml.safe_dump(
            {
                "project": {"project_dir": str(tmp_path / "project")},
                "workflows": {
                    "simulate_system": {
                        "enabled": True,
                        "config_path": workflow.name,
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    project = tmp_path / "project"
    invocation = "a" * 32
    capture_simulation_sources_v2(config, project, invocation)
    sources = read_simulation_sources_v2(project, invocation)
    assert [source.id for source in sources] == ["project", "workflow_simulate_system"]
    workflow.write_text(workflow.read_text() + "# revised\n", encoding="utf-8")
    with pytest.raises(SimulationFrozenError, match="changed after preparse"):
        read_simulation_sources_v2(project, invocation)


def test_live_model_freeze_names_the_changed_runtime_input(tmp_path):
    """WF4's parse-time guard must identify the model file that drifted."""
    model = tmp_path / "models" / "hydrology" / "wflow"
    (model / "forcing").mkdir(parents=True)
    (model / "forcing" / "inmaps_historical.nc").write_bytes(b"before")
    (model / "wflow_sbm.toml").write_text(
        '[input]\npath_forcing = "forcing/inmaps_historical.nc"\n',
        encoding="utf-8",
    )
    retained = build_model_reference(model, tmp_path)
    (model / "forcing" / "inmaps_historical.nc").write_bytes(b"after")
    live = build_model_reference(model, tmp_path)

    with pytest.raises(
        SimulationFrozenError,
        match=r"forcing/inmaps_historical\.nc.*new experiment",
    ):
        _assert_model_reference_current(retained, live, model)


def test_simulation_intent_binds_prep_and_selected_runs():
    entries = {"wflow_sbm.toml": "a" * 64, "output.csv": "<absent>"}
    reference = {
        "digest_version": 1,
        "model_path": "models/hydrology/wflow",
        "model_toml": "wflow_sbm.toml",
        "digest": model_digest_from_entries(sorted(entries.items())),
        "inputs": entries,
    }
    request = make_response_request(
        ["run_01"],
        [
            {
                "variable": "q",
                "locations": ["outlet"],
                "units": "m3/s",
                "calendar": "standard",
                "timestep": "P1D",
                "time_label": "interval_end",
                "start": "2046-01-02 00:00:00",
                "end": "2054-12-31 00:00:00",
                "missing_value": "NaN",
            }
        ],
    )
    documents = {
        "model_reference": reference,
        "settings": {"simulation_window": {"start": 2046, "end": 2054}},
        "environment": {"packages": {}},
        "simulator_adapter_code": [],
        "response_request": request,
        "preparation": {"schema_version": "forcing-preparation/2"},
    }
    identity = {
        "schema_version": "forcing-preparation-identity/2",
        "ancillary": [],
        "catalog": {},
        "forcing_elevation": {},
        "generated_forcing_reader": {},
        "pet_method": "debruin",
    }
    selection = {
        "resolution_mode": "explicit-manifest",
        "manifest": {"path": "scenarios/_engine/collections/a/collection.json"},
        "collection_id": "a" * 64,
        "collection_revision": "b" * 64,
        "run_ids": ["run_01"],
    }
    intent = simulation_intent_v2(
        "experiment",
        selection,
        {"name": "wflow", "revision": "c" * 64},
        documents,
        identity,
    )
    assert intent["identity_digests"]["preparation"] == content_sha256(identity)
    identity["pet_method"] = "makkink"
    changed = simulation_intent_v2(
        "experiment",
        selection,
        {"name": "wflow", "revision": "c" * 64},
        documents,
        identity,
    )
    assert changed["simulation_id"] != intent["simulation_id"]


# --- ported from the retired v1 simulation-record tests (t2610041227) --------
#
# Every case below starts from a real frozen v2 experiment
# (`tests/_v2_experiment.py`); none stubs a reader.


@pytest.fixture(scope="module")
def _frozen_once(tmp_path_factory):
    from tests._v2_experiment import build_v2_experiment

    return build_v2_experiment(tmp_path_factory.mktemp("frozen"), publish=False)


@pytest.fixture
def frozen(_frozen_once, tmp_path):
    from tests._v2_experiment import copy_v2_experiment

    return copy_v2_experiment(_frozen_once, tmp_path)


def _files(root):
    return {
        p: (p.read_bytes(), p.stat().st_mtime_ns)
        for p in root.rglob("*")
        if p.is_file()
    }


def test_a_repeated_freeze_reuses_the_intent_and_rewrites_nothing(frozen):
    from blueearth_cst.experiment.simulation_record import (
        freeze_simulation_v2,
        read_simulation_intent_v2,
    )

    root = frozen.root
    intent = read_simulation_intent_v2(root)
    before = _files(frozen.project)
    assert (
        freeze_simulation_v2(
            root, intent, invocation_id="e" * 32, command=["simulate_system"]
        )
        == intent
    )
    assert before == _files(frozen.project)


@pytest.mark.parametrize("document", ["settings", "environment"])
def test_a_changed_intent_refuses_the_retained_experiment(frozen, document):
    from copy import deepcopy

    from blueearth_cst.experiment.simulation_record import (
        freeze_simulation_v2,
        read_simulation_intent_v2,
    )
    from blueearth_cst.experiment.wf4_ancillary_descriptor import (
        preparation_identity_v2,
    )

    root = frozen.root
    retained = read_simulation_intent_v2(root)
    documents = deepcopy(retained["documents"])
    if document == "settings":
        documents["settings"]["simulation_window"]["end"] += 1
    else:
        documents["environment"]["packages"]["extra"] = "changed"
    changed = simulation_intent_v2(
        root.name,
        retained["collection"],
        retained["simulator"],
        documents,
        preparation_identity_v2(
            documents["preparation"], project_root=frozen.project, experiment_root=root
        ),
    )
    assert changed["simulation_id"] != retained["simulation_id"]
    with pytest.raises(SimulationFrozenError, match="intent differs"):
        freeze_simulation_v2(
            root, changed, invocation_id="e" * 32, command=["simulate_system"]
        )
    assert read_simulation_intent_v2(root) == retained


def test_a_tampered_retained_intent_refuses_without_the_live_model(frozen):
    import shutil

    from blueearth_cst.experiment.content_identity import (
        canonical_json_bytes,
        read_canonical_json,
    )
    from blueearth_cst.experiment.simulation_record import read_simulation_intent_v2

    root = frozen.root
    shutil.rmtree(frozen.project / "models")
    path = root / "_engine/simulation_intent.json"
    intent = read_canonical_json(path)
    intent["documents"]["settings"]["simulation_window"]["end"] = 2099
    path.write_bytes(canonical_json_bytes(intent))
    with pytest.raises(SimulationFrozenError, match="intent differs"):
        read_simulation_intent_v2(root)


def test_readiness_refuses_an_inventory_that_was_not_published(frozen):
    from blueearth_cst.experiment.response_inventory import (
        MissingResponseRequirement,
    )
    from blueearth_cst.experiment.simulation_record import publish_simulation_v2

    root = frozen.root
    forged = {"simulation_id": "0" * 64, "response_inventory_sha256": "0" * 64}
    with pytest.raises((MissingResponseRequirement, FileNotFoundError)):
        publish_simulation_v2(root, forged)
    assert not (root / "_engine/simulation.json").exists()


def test_the_experiment_name_does_not_enter_the_simulation_identity(frozen):
    from blueearth_cst.experiment.simulation_record import read_simulation_intent_v2
    from blueearth_cst.experiment.wf4_ancillary_descriptor import (
        preparation_identity_v2,
    )

    root = frozen.root
    retained = read_simulation_intent_v2(root)
    renamed = simulation_intent_v2(
        "another_experiment",
        retained["collection"],
        retained["simulator"],
        retained["documents"],
        preparation_identity_v2(
            retained["documents"]["preparation"],
            project_root=frozen.project,
            experiment_root=root,
        ),
    )
    assert renamed["experiment_name"] == "another_experiment"
    assert renamed["simulation_id"] == retained["simulation_id"]


@pytest.mark.parametrize("wflow_available", [True, False])
@pytest.mark.parametrize("include", [False, True])
def test_the_live_input_planner_applies_the_prepared_clock_once(
    frozen, monkeypatch, wflow_available, include
):
    """The production WF4 planner, with only Julia and the stage probe replaced."""
    import subprocess
    from pathlib import Path
    from types import SimpleNamespace

    from blueearth_cst.experiment import content_identity
    from blueearth_cst.experiment.scenario_collection_v2 import read_collection_v2
    from blueearth_cst.experiment.simulation_record import (
        live_simulation_inputs_v2,
        read_simulation_intent_v2,
    )

    root, project = frozen.root, frozen.project
    model = project / "models/hydrology/wflow"
    toml = model / "wflow_sbm.toml"
    original = toml.read_bytes()
    monkeypatch.setattr(
        content_identity,
        "stage_environment",
        lambda *args, **kwargs: {
            "packages": {"julia:Wflow": "1.0.2:fixture"} if wflow_available else {},
            "locks": {},
        },
    )
    calls = []

    def native_headers(command, *, check):
        assert check is True
        assert command[-2] == str(toml.resolve())
        Path(command[-1]).write_text("Q_9\nQ_2\ngwr_9\n", encoding="utf-8")
        calls.append(command)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(subprocess, "run", native_headers)
    retained = read_simulation_intent_v2(root)
    kwargs = {
        "project_root": project,
        "model_root": model,
        "collection_path": frozen.collection_marker,
        "collection": read_collection_v2(frozen.collection_marker),
        "resolution_mode": "explicit-manifest",
        "preparation": retained["documents"]["preparation"],
        "simulation_window": {"start": 2046, "end": 2054},
        "julia_command": ["fixture-julia"],
        "header_path": project / "headers.txt",
        "include_unperturbed": include,
    }
    if not wflow_available:
        with pytest.raises(KeyError, match="julia:Wflow"):
            live_simulation_inputs_v2(root, **kwargs)
    else:
        intent = live_simulation_inputs_v2(root, **kwargs)
        request = intent["documents"]["response_request"]
        q = next(item for item in request["variables"] if item["variable"] == "q")
        assert q["start"] == "2046-01-02 00:00:00"
        assert q["end"] == "2054-12-31 00:00:00"
        assert q["calendar"] == "standard"
        assert q["locations"] == ["9", "2"]
        assert request["run_ids"] == (["run_1", "run_2"] if include else ["run_2"])
    assert len(calls) == 1
    assert toml.read_bytes() == original
