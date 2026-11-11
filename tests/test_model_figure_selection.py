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
        evaluation_dir=evaluation,
        station_plots=enabled,
    )
    actual = pd.read_csv(evaluation / "performance_indicators.csv", index_col=[0, 1])
    pd.testing.assert_frame_equal(actual, scores)
    assert len(rendered) == int(enabled)
    assert previous.read_bytes() == b"retained"
