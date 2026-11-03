"""Execution diagnostics cannot invalidate otherwise identical frozen responses."""

import json
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
    "changed", ["model", "code", "environment", "settings", "preparation", "collection"]
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


def _archive_upgrade(retained):
    from blueearth_cst.experiment.simulation_reuse import (
        _ARCHIVE_AFTER,
        _ARCHIVE_BEFORE,
        _ARCHIVE_LAUNCH,
    )

    previous = deepcopy(retained)
    previous["documents"]["simulator_adapter_code"].append(
        {"path": _ARCHIVE_LAUNCH, "sha256": _ARCHIVE_BEFORE}
    )
    current = deepcopy(retained)
    current["documents"]["simulator_adapter_code"].extend(
        {"path": path, "sha256": checksum} for path, checksum in _ARCHIVE_AFTER.items()
    )
    return _rebuild(previous), _rebuild(current)


def test_audited_archive_update_reuses_original_simulation_without_rewriting_it(
    retained,
):
    previous, current = _archive_upgrade(retained)
    before = deepcopy((previous, current))
    check_simulation_reuse(previous, current)
    assert (previous, current) == before


@pytest.mark.parametrize(
    "change",
    [
        "launcher",
        "helper",
        "old_launcher",
        "code",
        "settings",
        "added_code",
        "duplicate_code",
    ],
)
@pytest.mark.parametrize("portable_source", [False, True])
def test_archive_bridge_refuses_any_unaudited_or_scientific_change(
    retained, change, portable_source
):
    previous, current = _archive_upgrade(retained)
    if portable_source:
        from blueearth_cst.experiment.simulation_reuse import _ARCHIVE_PORTABLE

        previous = deepcopy(current)
        for item in previous["documents"]["simulator_adapter_code"]:
            if item["path"] in _ARCHIVE_PORTABLE:
                item["sha256"] = _ARCHIVE_PORTABLE[item["path"]]
        previous = _rebuild(previous)
    code = current["documents"]["simulator_adapter_code"]
    if change == "launcher":
        code[-2]["sha256"] = "f" * 64
    elif change == "helper":
        code[-1]["sha256"] = "f" * 64
    elif change == "old_launcher":
        previous["documents"]["simulator_adapter_code"][-1]["sha256"] = "f" * 64
        previous = _rebuild(previous)
    elif change == "code":
        code[0]["sha256"] = "f" * 64
    elif change == "settings":
        current["documents"]["settings"]["simulation_window"]["end"] += 1
    elif change == "duplicate_code":
        previous["documents"]["simulator_adapter_code"].append(
            deepcopy(previous["documents"]["simulator_adapter_code"][-1])
        )
        previous = _rebuild(previous)
    else:
        code.append({"path": "blueearth_cst/shared/unaudited.py", "sha256": "f" * 64})
    with pytest.raises(SimulationFrozenError):
        check_simulation_reuse(previous, _rebuild(current))


@pytest.mark.parametrize("target_name", ["_ARCHIVE_PRODUCT", "_ARCHIVE_PORTABLE"])
def test_previous_archive_bridge_remains_valid(retained, target_name):
    from blueearth_cst.experiment import simulation_reuse

    target = getattr(simulation_reuse, target_name)
    previous, current = _archive_upgrade(retained)
    for item in current["documents"]["simulator_adapter_code"]:
        if item["path"] in target:
            item["sha256"] = target[item["path"]]
    check_simulation_reuse(previous, _rebuild(current))


def test_portable_archive_upgrade_reuses_without_rewriting_intents(retained):
    from blueearth_cst.experiment.simulation_reuse import _ARCHIVE_PORTABLE

    _, current = _archive_upgrade(retained)
    previous = deepcopy(current)
    for item in previous["documents"]["simulator_adapter_code"]:
        if item["path"] in _ARCHIVE_PORTABLE:
            item["sha256"] = _ARCHIVE_PORTABLE[item["path"]]
    previous = _rebuild(previous)
    before = deepcopy((previous, current))
    check_simulation_reuse(previous, current)
    assert (previous, current) == before


def test_archive_bridge_matches_the_audited_checkout():
    import hashlib

    from blueearth_cst.experiment.simulation_reuse import _ARCHIVE_AFTER

    root = Path(__file__).resolve().parents[1]
    for path, expected in _ARCHIVE_AFTER.items():
        observed = hashlib.sha256(
            (root / path).read_bytes().replace(b"\r\n", b"\n")
        ).hexdigest()
        assert observed == expected, "Archive compatibility bridge requires a new audit"


def test_diagnostic_edit_reuses_original_identity_without_mutating_intents(retained):
    current = deepcopy(retained)
    current["documents"]["simulator_adapter_code"][1]["sha256"] = "c" * 64
    current = _rebuild(current)
    assert current["simulation_id"] != retained["simulation_id"]
    before = deepcopy((retained, current))
    check_simulation_reuse(retained, current)
    assert (retained, current) == before


def test_removing_settings_summary_reuses_the_previous_archive_version(retained):
    from blueearth_cst.experiment.simulation_reuse import _ARCHIVE_PREVIOUS

    _, current = _archive_upgrade(retained)
    previous = deepcopy(retained)
    previous["documents"]["simulator_adapter_code"].extend(
        {"path": path, "sha256": checksum}
        for path, checksum in _ARCHIVE_PREVIOUS.items()
    )
    check_simulation_reuse(_rebuild(previous), current)


@pytest.mark.parametrize(
    "component",
    [
        "code",
        "settings",
        "environment",
        "preparation",
        "collection",
        "simulator",
        "model",
        "response",
    ],
)
def test_diagnostic_edit_cannot_hide_changed_scientific_inputs(retained, component):
    current = deepcopy(retained)
    current["documents"]["simulator_adapter_code"][1]["sha256"] = "c" * 64
    if component == "code":
        current["documents"]["simulator_adapter_code"][0]["sha256"] = "d" * 64
    elif component == "settings":
        current["documents"]["settings"]["simulation_window"]["end"] += 1
    elif component == "environment":
        current["documents"]["environment"]["packages"]["changed"] = "1"
    elif component == "preparation":
        current["identity_projections"]["preparation"]["pet_method"] = "changed"
    elif component == "collection":
        current["collection"]["collection_revision"] = "d" * 64
    elif component == "model":
        from blueearth_cst.shared.model_digest import model_digest_from_entries

        model = current["documents"]["model_reference"]
        model["inputs"][model["model_toml"]] = "d" * 64
        model["digest"] = model_digest_from_entries(sorted(model["inputs"].items()))
    elif component == "response":
        current["documents"]["response_request"]["variables"][0]["units"] = "m"
    else:
        current["simulator"]["revision"] = "d" * 64
    with pytest.raises(SimulationFrozenError, match="simulation inputs changed"):
        check_simulation_reuse(retained, _rebuild(current))


def _historical_code():
    """Exact normalized inventory captured before the source-boundary refactor."""
    return json.loads(
        (Path(__file__).parent / "data/simulation-code-before-boundary.json").read_text(
            encoding="utf-8"
        )
    )


def _pre_temperature_code():
    """Normalized inventory captured from 76a2a874, before native temperature."""
    return json.loads(
        (
            Path(__file__).parent / "data/simulation-code-before-temperature.json"
        ).read_text(encoding="utf-8")
    )


def _pre_release_tag_code():
    """Normalized inventory captured from 094bab3a^, before the release-tag helper."""
    return json.loads(
        (
            Path(__file__).parent / "data/simulation-code-before-release-tag.json"
        ).read_text(encoding="utf-8")
    )


def test_boundary_bridge_matches_audited_sources():
    from blueearth_cst.experiment.content_identity import content_sha256
    from blueearth_cst.experiment.simulation_code import simulation_code_inventory
    from blueearth_cst.experiment.simulation_reuse import (
        _BOUNDARY_AFTER,
        _BOUNDARY_BEFORE,
        _BOUNDARY_PREVIOUS,
        _BOUNDARY_TEMPERATURE,
    )

    assert content_sha256(_historical_code()) == _BOUNDARY_BEFORE
    assert content_sha256(_pre_temperature_code()) == _BOUNDARY_PREVIOUS
    assert content_sha256(_pre_release_tag_code()) == _BOUNDARY_TEMPERATURE
    assert (
        content_sha256(simulation_code_inventory(Path(__file__).resolve().parents[1]))
        == _BOUNDARY_AFTER
    ), "Scientific source changes require a new compatibility audit"


@pytest.mark.parametrize(
    ("path", "scientific"),
    [
        ("plotting/export.py", False),
        ("plotting/prepared_maps.py", False),
        ("indicators/sets/default/calculations.py", False),
        ("shared/snake_utils.py", False),
        ("shared/run_log_core.py", False),
        ("shared/workflow_config_snapshot.py", False),
        ("experiment/generation_plan.py", False),
        ("experiment/downscale_climate_forcing.py", True),
        ("experiment/forcing_window.py", True),
        ("experiment/simulator_adapter.py", True),
        ("experiment/run_wflow_batch.jl", True),
        ("shared/wflow_progress.jl", True),
        ("climate_analysis/prepare_climate_data_catalog.py", True),
        ("experiment/wf4_ancillary_descriptor.py", True),
        ("experiment/simulation_code.py", True),
    ],
)
def test_source_boundary_drives_actual_reuse_guard(
    retained, monkeypatch, path, scientific
):
    from blueearth_cst.experiment.simulation_code import simulation_code_inventory

    repo = Path(__file__).resolve().parents[1]
    previous = deepcopy(retained)
    previous["documents"]["simulator_adapter_code"] = simulation_code_inventory(repo)
    previous = _rebuild(previous)
    current = deepcopy(previous)
    read_bytes = Path.read_bytes
    changed = (repo / "blueearth_cst" / path).resolve()
    assert changed.is_file()

    def edited_bytes(source):
        content = read_bytes(source)
        return (
            content + b"\n# source edit\n" if source.resolve() == changed else content
        )

    monkeypatch.setattr(Path, "read_bytes", edited_bytes)
    current["documents"]["simulator_adapter_code"] = simulation_code_inventory(repo)
    current = _rebuild(current)
    if scientific:
        assert current["simulation_id"] != previous["simulation_id"]
        with pytest.raises(SimulationFrozenError):
            check_simulation_reuse(previous, current)
    else:
        assert current["simulation_id"] == previous["simulation_id"]
        check_simulation_reuse(previous, current)


@pytest.mark.parametrize(
    "change",
    [None, "old_code", "new_code", "duplicate", "settings", "environment", "response"],
)
@pytest.mark.parametrize(
    "transition", ["boundary", "temperature", "previous", "release_tag"]
)
def test_boundary_bridge_accepts_only_the_audited_transition(
    retained, change, transition
):
    from blueearth_cst.experiment.simulation_code import simulation_code_inventory

    previous = deepcopy(retained)
    previous["documents"]["simulator_adapter_code"] = {
        "temperature": _pre_temperature_code,
        "release_tag": _pre_release_tag_code,
    }.get(transition, _historical_code)()
    current = deepcopy(retained)
    current["documents"]["simulator_adapter_code"] = simulation_code_inventory(
        Path(__file__).resolve().parents[1]
    )
    if transition == "previous":
        current["documents"]["simulator_adapter_code"] = _pre_temperature_code()
    if change == "old_code":
        previous["documents"]["simulator_adapter_code"][0]["sha256"] = "f" * 64
    elif change == "new_code":
        current["documents"]["simulator_adapter_code"][0]["sha256"] = "f" * 64
    elif change == "duplicate":
        current["documents"]["simulator_adapter_code"].append(
            deepcopy(current["documents"]["simulator_adapter_code"][0])
        )
    elif change == "settings":
        current["documents"]["settings"]["simulation_window"]["end"] += 1
    elif change == "environment":
        current["documents"]["environment"]["packages"]["julia:Wflow"] = "changed"
    elif change == "response":
        current["documents"]["response_request"]["variables"][0]["units"] = "changed"
    previous, current = _rebuild(previous), _rebuild(current)
    before = deepcopy((previous, current))
    if change:
        with pytest.raises(SimulationFrozenError):
            check_simulation_reuse(previous, current)
    else:
        check_simulation_reuse(previous, current)
    assert (previous, current) == before


@pytest.mark.parametrize(
    "inventory", [_historical_code, _pre_temperature_code, _pre_release_tag_code]
)
def test_completed_legacy_inventory_reuses_records_without_rewriting_bytes(
    tmp_path, monkeypatch, inventory
):
    from blueearth_cst.experiment import content_identity, simulation_record
    from blueearth_cst.experiment.scenario_collection_v2 import read_collection_v2
    from blueearth_cst.experiment.simulation_reuse import (
        check_completed_simulation_inputs,
    )

    original = simulation_record.simulation_intent_v2

    def historical_intent(name, collection, simulator, documents, preparation):
        documents = {**documents, "simulator_adapter_code": inventory()}
        return original(name, collection, simulator, documents, preparation)

    # Build supported retained records using the captured pre-change inventory.
    # This is a migration fixture, not a historical production model run.
    with monkeypatch.context() as patch:
        patch.setattr(simulation_record, "simulation_intent_v2", historical_intent)
        experiment = build_v2_experiment(tmp_path)
    retained = read_simulation_intent_v2(experiment.root)
    monkeypatch.setattr(
        content_identity,
        "stage_environment",
        lambda *a, **kw: deepcopy(retained["documents"]["environment"]),
    )
    before = {
        path: path.read_bytes() for path in experiment.root.rglob("*") if path.is_file()
    }
    kwargs = dict(
        project_root=experiment.project,
        model_root=experiment.project / "models/hydrology/wflow",
        collection_path=experiment.collection_marker,
        collection=read_collection_v2(experiment.collection_marker),
        resolution_mode=retained["collection"]["resolution_mode"],
        preparation=retained["documents"]["preparation"],
        simulation_window=retained["documents"]["settings"]["simulation_window"],
        julia_command=["unused"],
    )
    check_completed_simulation_inputs(experiment.root, **kwargs)
    # Temperature support in the reader cannot authorize changed model outputs.
    model = Path(kwargs["model_root"]) / "wflow_sbm.toml"
    model.write_text(
        model.read_text() + '\n[[output.csv.column]]\nheader="temp"\n'
        'parameter="atmosphere_air__temperature"\n'
        'map="subcatchment"\nreducer="mean"\n',
        encoding="utf-8",
    )
    with pytest.raises(SimulationFrozenError):
        check_completed_simulation_inputs(experiment.root, **kwargs)
    assert {
        path: path.read_bytes() for path in experiment.root.rglob("*") if path.is_file()
    } == before
