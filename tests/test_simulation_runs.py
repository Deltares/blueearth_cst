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
