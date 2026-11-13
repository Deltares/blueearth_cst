"""WF4 can omit unperturbed members while retaining collection provenance."""

from types import SimpleNamespace

import pytest

from blueearth_cst.experiment import simulation_runner
from blueearth_cst.experiment.simulation_runs import (
    include_unperturbed,
    selected_run_ids,
)


@pytest.mark.parametrize("value", [None, "false", 0, 1, []])
def test_switch_refuses_non_booleans(value):
    with pytest.raises(ValueError, match="include_unperturbed must be true or false"):
        include_unperturbed({"include_unperturbed": value})


@pytest.mark.parametrize("include", [False, True])
def test_selection_keeps_ids_order_and_collection_files(tmp_path, include):
    from blueearth_cst.shared.workflow_config_snapshot import file_reference

    marker = tmp_path / "scenarios/_engine/collections/fixture/collection.json"
    marker.parent.mkdir(parents=True)
    lookup = marker.with_name("scenario_run_lookup.csv")
    data = "run_id,st_id\nrun_01,\nrun_02,1\nrun_03,2\nrun_04,\nrun_05,1\nrun_06,2\n"
    lookup.write_text(data, encoding="utf-8")
    collection = {
        "series": [{"run_id": f"run_{i:02d}"} for i in range(1, 7)],
        "scenario_run_lookup": file_reference(lookup, "project_root", tmp_path),
    }
    expected = [
        f"run_{i:02d}" for i in ([1, 2, 3, 4, 5, 6] if include else [2, 3, 5, 6])
    ]
    assert selected_run_ids(collection, marker, include=include) == expected
    assert len(collection["series"]) == 6
    assert lookup.read_text(encoding="utf-8") == data
    assert include_unperturbed({}) is True


def test_reference_indicators_refused_before_simulation(monkeypatch):
    indicator = SimpleNamespace(
        indicator="q_wet", declaration=SimpleNamespace(reference="unperturbed_runs")
    )
    monkeypatch.setattr(
        "blueearth_cst.experiment.indicator_sets.selected_sets",
        lambda *args, **kwargs: [
            SimpleNamespace(set_id="seasonal", indicators=[indicator])
        ],
    )
    with pytest.raises(
        ValueError, match=r"seasonal:q_wet.*operations: \[simulations\]"
    ):
        simulation_runner.simulation_indicator_sets(
            {}, {"include_unperturbed": False}, SimpleNamespace(legacy=True)
        )


def test_initialization_defaults_and_validation():
    from blueearth_cst.experiment.simulation_runs import initialization

    assert initialization({}) == {"state": "auto", "warmup_years": 1}
    assert initialization({"initialization": None}) == {
        "state": "auto",
        "warmup_years": 1,
    }
    assert initialization({"initialization": {"state": "cold", "warmup_years": 0}}) == {
        "state": "cold",
        "warmup_years": 0,
    }
    for bad in (
        {"state": "warm"},
        {"warmup_years": -1},
        {"warmup_years": True},
        {"warmup_years": 1.0},
        {"spinup": 1},
        [],
    ):
        with pytest.raises(ValueError, match="initialization"):
            initialization({"initialization": bad})


def _model(tmp_path, *, cold=True, variables=True, state=True):
    model = tmp_path / "models/hydrology/wflow"
    path = model / "historical/default/run/outstate/outstates.nc"
    path.parent.mkdir(parents=True)
    if state:
        path.write_bytes(b"historical end state")
    config = {"model": {"cold_start__flag": cold}}
    if variables:
        config["state"] = {"variables": {"soil_temp": "soil_temp"}}
    return model, config


@pytest.mark.parametrize(
    "state,cold,source",
    [
        ("auto", True, "historical"),
        ("auto", False, "model"),
        ("cold", True, "cold"),
        ("historical", True, "historical"),
    ],
)
def test_explicit_state_wins_over_the_default(tmp_path, state, cold, source):
    import hashlib

    from blueearth_cst.experiment.simulation_runs import (
        resolve_initialization,
        validate_frozen_initialization,
    )

    model, config = _model(tmp_path, cold=cold)
    resolved = resolve_initialization(
        model, config, {"state": state, "warmup_years": 1}, tmp_path
    )
    validate_frozen_initialization(resolved)
    assert (resolved["state"], resolved["source"], resolved["warmup_years"]) == (
        state,
        source,
        1,
    )
    if source == "historical":
        assert resolved["state_file"]["path"] == (
            "models/hydrology/wflow/historical/default/run/outstate/outstates.nc"
        )
        assert (
            resolved["state_file"]["sha256"]
            == hashlib.sha256(b"historical end state").hexdigest()
        )
    else:
        assert "state_file" not in resolved


@pytest.mark.parametrize("state", ["historical", "cold"])
def test_conflicting_explicit_states_refuse(tmp_path, state):
    from blueearth_cst.experiment.simulation_runs import resolve_initialization

    model, config = _model(tmp_path, cold=False)
    with pytest.raises(ValueError, match="conflicts with the warm state"):
        resolve_initialization(
            model, config, {"state": state, "warmup_years": 1}, tmp_path
        )


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"state": False}, "historical end state is missing"),
        ({"variables": False}, "state.variables"),
    ],
)
def test_historical_state_requires_wf1_outputs(tmp_path, kwargs, match):
    from blueearth_cst.experiment.simulation_runs import resolve_initialization

    model, config = _model(tmp_path, **kwargs)
    with pytest.raises(ValueError, match=match):
        resolve_initialization(
            model, config, {"state": "auto", "warmup_years": 1}, tmp_path
        )


def test_legacy_intents_keep_the_model_state_without_warmup():
    from blueearth_cst.experiment.simulation_runs import frozen_initialization

    assert frozen_initialization({"simulation_window": {}}) == {
        "state": "model",
        "source": "model",
        "warmup_years": 0,
    }


def test_named_recipe_binds_its_own_end_state(tmp_path):
    from blueearth_cst.experiment.simulation_runs import resolve_initialization
    from blueearth_cst.shared.named_forcings import HistoricalLayout

    model, config = _model(tmp_path)
    named = model / HistoricalLayout("chirps").end_state
    named.parent.mkdir(parents=True)
    named.write_bytes(b"chirps end state")
    resolved = resolve_initialization(
        model,
        config,
        {"state": "historical", "warmup_years": 0},
        tmp_path,
        HistoricalLayout("chirps").end_state,
    )
    assert resolved["state_file"]["path"] == (
        "models/hydrology/wflow/historical/chirps/run/outstate/outstates.nc"
    )
    with pytest.raises(ValueError, match=r"historical[\\/]era5[\\/]run"):
        resolve_initialization(
            model,
            config,
            {"state": "historical", "warmup_years": 0},
            tmp_path,
            HistoricalLayout("era5").end_state,
        )
