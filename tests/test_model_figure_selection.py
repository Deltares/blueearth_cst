"""Location figures are optional while observation-based scores remain available."""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from blueearth_cst.model import plot_results


@pytest.mark.parametrize("enabled", [False, True])
def test_station_plot_switch_preserves_performance_table(
    tmp_path, monkeypatch, enabled
):
    time = pd.date_range("2000-01-01", "2002-12-31")
    discharge = xr.DataArray(
        np.ones((len(time), 1)),
        dims=("time", "index"),
        coords={"time": time, "index": [101]},
    )
    model = SimpleNamespace(
        output_csv=SimpleNamespace(read=lambda: None, data={"Q_outlets": discharge}),
        geoms=SimpleNamespace(data={}),
    )
    monkeypatch.setattr(plot_results, "WflowSbmModel", lambda **kwargs: model)
    discharge = discharge.assign_coords(station_name=("index", ["Outlet"]))
    monkeypatch.setattr(
        plot_results, "evaluation_arrays", lambda *args: (discharge, xr.Dataset())
    )
    observations = tmp_path / "observations.csv"
    observations.touch()
    registry = tmp_path / "locations.csv"
    registry.touch()
    monkeypatch.setattr(
        plot_results, "validate_observation_station_ids", lambda *a: None
    )
    monkeypatch.setattr(
        plot_results,
        "open_timeseries_from_table",
        lambda *a, **kw: discharge.assign_coords(index=[1010]).rename(
            {"index": "wflow_id"}
        ),
    )
    monkeypatch.setattr(
        plot_results,
        "resolve_stations",
        lambda *a: {101: SimpleNamespace(wflow_id=1010)},
    )
    scores = pd.DataFrame(
        {"1010": [0.75]},
        index=pd.MultiIndex.from_tuples(
            [("KGE", "daily")], names=["metric", "time_type"]
        ),
    )
    monkeypatch.setattr(plot_results, "compute_indicators", lambda **kw: scores.copy())
    rendered = []
    monkeypatch.setattr(
        plot_results, "plot_station_evaluation", lambda **kw: rendered.append(kw)
    )
    evaluation = tmp_path / "evaluation"
    previous = evaluation / "plots/stations/previous.png"
    previous.parent.mkdir(parents=True)
    previous.write_bytes(b"retained")
    plot_results.analyse_wflow_historical(
        project_dir=tmp_path,
        observations_fn=observations,
        location_registry=registry,
        response_record=tmp_path / "response_tables.json",
        evaluation_dir=evaluation,
        station_plots=enabled,
    )
    actual = pd.read_csv(
        evaluation / "tables" / "performance_indicators.csv", index_col=[0, 1]
    )
    pd.testing.assert_frame_equal(actual, scores)
    assert len(rendered) == int(enabled)
    assert previous.read_bytes() == b"retained"


@pytest.mark.workflow_contract
def test_csv_precision_is_wired_to_authoritative_export(tmp_path):
    from pathlib import Path

    from blueearth_cst.shared.config_composition import load_composed_config
    from tests.conftest import parse_workflow, write_config

    config = load_composed_config(
        Path(__file__).with_name("project_config_fixture.yml")
    )
    config["workflows"]["build_model"]["output"] = {"csv": {"decimal_places": 8}}
    workflow = parse_workflow("build_model.smk", write_config(tmp_path, config))
    export = workflow.get_rule("export_simulation_tables")
    evaluation = workflow.get_rule("plot_model_evaluation")
    assert export.params.decimal_places == 8
    assert export.output.response_record in evaluation.input
    assert set(export.output.tables) <= set(evaluation.input)
    native = workflow.get_rule("run_historical_simulation")
    assert str(native.output.csv_path) in native.temp_output
