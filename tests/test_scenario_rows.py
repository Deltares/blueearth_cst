"""P1 configured completeness, namespace and ancestry falsifiers."""

from dataclasses import replace

import pytest

from blueearth_cst.experiment.scenario_rows import (
    EmptyScenarioSetError,
    ScenarioRow,
    UnitNamespaceCapacityError,
    enumerate_stochastic,
    stochastic_rows,
    validate_forest,
    validate_stochastic,
)


def test_rows_are_parse_time_pure(monkeypatch):
    """Enumeration needs neither a generated table nor a resolved auto seed."""

    def forbidden(*args, **kwargs):
        pytest.fail("enumeration attempted file I/O")

    monkeypatch.setattr("builtins.open", forbidden)
    spec = {
        "n_realizations": 2,
        "climate_perturbations": {"temp": {"n_levels": 2}, "precip": {"n_levels": 3}},
        "seed": "auto",
    }
    rows = enumerate_stochastic(spec, unit_id_capacity=100)
    assert len(rows) == 14
    assert [row.run_id for row in rows] == [f"run_{i:03}" for i in range(1, 15)]
    assert rows[0].as_record() == {
        "run_id": "run_001",
        "derived_from": "",
        "evaluated": "true",
        "scenario_type": "stochastic",
        "rlz": "1",
        "st_id": "",
    }
    assert rows[7].derived_from == ""
    assert all(row.derived_from == "run_008" for row in rows[8:])
    assert all(row.evaluated for row in rows)
    assert rows == enumerate_stochastic({**spec, "seed": 123}, unit_id_capacity=100)
    validate_stochastic(rows, n_realizations=2, st_num=6, unit_id_capacity=100)


@pytest.mark.parametrize(
    "defect",
    [
        "missing_cell",
        "missing_design",
        "cross_draw",
        "chain",
        "no_edge",
        "unevaluated",
        "wrong_payload",
    ],
)
def test_pairing_and_configured_completeness_refuse(defect):
    rows = list(stochastic_rows(2, 2, unit_id_capacity=10))
    if defect == "missing_cell":
        rows.pop()
    elif defect == "missing_design":
        rows = [row for row in rows if dict(row.payload)["st_id"] != "2"]
    elif defect == "cross_draw":
        rows[1] = replace(rows[1], derived_from="04")
    elif defect == "chain":
        rows[2] = replace(rows[2], derived_from="02")
    elif defect == "no_edge":
        rows[1] = replace(rows[1], derived_from="")
    elif defect == "unevaluated":
        rows[0] = replace(rows[0], evaluated=False)
    else:
        rows[1] = replace(rows[1], payload=(("rlz", "2"), ("st_id", "1")))
    with pytest.raises(ValueError):
        validate_stochastic(rows, n_realizations=2, st_num=2, unit_id_capacity=10)


@pytest.mark.parametrize("parents", [("1", ""), ("2", "1"), ("9", "")])
def test_forest_refuses_self_cycle_and_missing_ancestor(parents):
    rows = tuple(
        ScenarioRow(f"run_{i}", f"run_{p}" if p else "", True, "fixture", ())
        for i, p in enumerate(parents, 1)
    )
    with pytest.raises(ValueError, match="ancestr|ancestor"):
        validate_forest(rows, unit_id_capacity=2)


def test_empty_edges_are_valid_for_independent_fixture():
    rows = (
        ScenarioRow("run_1", "", True, "fixture", ()),
        ScenarioRow("run_2", "", True, "fixture", ()),
    )
    validate_forest(rows, unit_id_capacity=2)


def test_empty_set_retired_toggle_and_capacity_refuse():
    with pytest.raises(EmptyScenarioSetError):
        validate_forest((), unit_id_capacity=1)
    with pytest.raises(EmptyScenarioSetError):
        enumerate_stochastic({"n_realizations": 0}, unit_id_capacity=1)
    with pytest.raises(ValueError, match="retired"):
        enumerate_stochastic({"run_historical": False}, unit_id_capacity=1)
    with pytest.raises(
        UnitNamespaceCapacityError,
        match="bundles=not-evaluated.*replacement unit_id_capacity=14",
    ):
        stochastic_rows(2, 6, unit_id_capacity=13)
    for value in [True, "2", 1.5, 0, -1]:
        with pytest.raises(ValueError):
            stochastic_rows(2, 6, unit_id_capacity=value)
