"""Current compact experiments reuse only identical frozen inputs."""

from copy import deepcopy
from pathlib import Path

import pytest

from blueearth_cst.experiment.simulation_record import (
    SimulationFrozenError,
    read_simulation_intent_v2,
    simulation_intent_v2,
)
from blueearth_cst.experiment.simulation_reuse import check_simulation_reuse
from tests._v2_experiment import build_v2_experiment


@pytest.fixture(scope="module")
def completed(tmp_path_factory):
    return build_v2_experiment(tmp_path_factory.mktemp("completed-reuse"))


@pytest.fixture
def observed_completed(completed, monkeypatch):
    from blueearth_cst.experiment import content_identity, simulation_code
    from blueearth_cst.experiment.scenario_collection_v2 import read_collection_v2

    retained = read_simulation_intent_v2(completed.root)
    code = deepcopy(retained["documents"]["simulator_adapter_code"])
    environment = deepcopy(retained["documents"]["environment"])
    observations = []

    def observe_code(repo, entries, *, excluded):
        observations.append(tuple(entries))
        return deepcopy(code)

    monkeypatch.setattr(simulation_code, "repository_code_inventory", observe_code)
    monkeypatch.setattr(
        content_identity, "stage_environment", lambda *a, **kw: environment
    )
    kwargs = {
        "project_root": completed.project,
        "model_root": completed.project / "models/hydrology/wflow",
        "collection_path": completed.collection_marker,
        "collection": read_collection_v2(completed.collection_marker),
        "resolution_mode": retained["collection"]["resolution_mode"],
        "preparation": deepcopy(retained["documents"]["preparation"]),
        "simulation_window": deepcopy(
            retained["documents"]["settings"]["simulation_window"]
        ),
        "julia_command": ["julia"],
    }
    return retained, kwargs, code, environment, observations


def test_completed_recheck_matches_full_planning_without_starting_julia(
    completed, observed_completed, monkeypatch, tmp_path
):
    from blueearth_cst.experiment import simulator_adapter
    from blueearth_cst.experiment.simulation_record import live_simulation_inputs_v2
    from blueearth_cst.experiment.simulation_reuse import (
        check_completed_simulation_inputs,
    )

    retained, kwargs, _, _, observations = observed_completed
    monkeypatch.setattr(
        simulator_adapter,
        "plan_native_response_request",
        lambda *a, **kw: deepcopy(retained["documents"]["response_request"]),
    )
    full = live_simulation_inputs_v2(
        completed.root, **kwargs, header_path=tmp_path / "unused-headers.txt"
    )
    assert full["simulation_id"] == retained["simulation_id"]

    def refuse_julia(*args, **kwargs):
        raise AssertionError("a completed repeat must not reconstruct native headers")

    monkeypatch.setattr(simulator_adapter, "plan_native_response_request", refuse_julia)
    before = (completed.root / "_engine/simulation_intent.json").read_bytes()
    check_completed_simulation_inputs(completed.root, **kwargs)
    assert observations[0] == observations[1]
    assert (completed.root / "_engine/simulation_intent.json").read_bytes() == before


@pytest.mark.parametrize(
    "changed",
    [
        "model",
        "code",
        "environment",
        "settings",
        "precision",
        "preparation",
        "collection",
    ],
)
def test_completed_recheck_observes_changed_inputs_every_time(
    completed, observed_completed, changed
):
    from blueearth_cst.experiment.simulation_reuse import (
        check_completed_simulation_inputs,
    )

    _, kwargs, code, environment, _ = observed_completed
    check_completed_simulation_inputs(completed.root, **kwargs)
    if changed == "model":
        model = Path(kwargs["model_root"]) / "wflow_sbm.toml"
        original = model.read_bytes()
        model.write_bytes(original + b"\n# changed model bytes\n")
    elif changed == "code":
        code[0]["sha256"] = "d" * 64
    elif changed == "environment":
        environment["packages"]["julia:Wflow"] = "changed"
    elif changed == "settings":
        kwargs["simulation_window"]["end"] += 1
    elif changed == "precision":
        kwargs["csv_output"] = {"decimal_places": 3}
    elif changed == "preparation":
        kwargs["preparation"]["pet_method"] = "changed"
    else:
        kwargs["collection"]["collection_revision"] = "d" * 64
    try:
        with pytest.raises(SimulationFrozenError):
            check_completed_simulation_inputs(completed.root, **kwargs)
    finally:
        if changed == "model":
            model.write_bytes(original)


def _rebuild(intent):
    return simulation_intent_v2(
        intent["experiment_name"],
        intent["collection"],
        intent["simulator"],
        intent["documents"],
        intent["identity_projections"]["preparation"],
    )


@pytest.fixture(scope="module")
def retained(tmp_path_factory):
    experiment = build_v2_experiment(tmp_path_factory.mktemp("reuse"), publish=False)
    intent = read_simulation_intent_v2(experiment.root)
    intent["documents"]["simulator_adapter_code"] = [
        {"path": "blueearth_cst/experiment/simulator_adapter.py", "sha256": "a" * 64},
        {"path": "blueearth_cst/shared/execution_outputs.py", "sha256": "b" * 64},
    ]
    return _rebuild(intent)


def test_identical_repeat_reuses_the_frozen_simulation(retained):
    check_simulation_reuse(retained, deepcopy(retained))


def test_any_changed_source_inventory_requires_new_experiment(retained):
    current = deepcopy(retained)
    current["documents"]["simulator_adapter_code"][0]["sha256"] = "f" * 64
    with pytest.raises(SimulationFrozenError):
        check_simulation_reuse(retained, _rebuild(current))
