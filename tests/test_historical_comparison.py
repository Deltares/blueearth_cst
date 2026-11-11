"""Independent arithmetic and boundary checks for retained-run comparisons."""

import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from blueearth_cst.model.historical_comparison import (
    complete_reporting_years,
    evaluation_dates,
    prepare_comparison,
    read_comparison,
    read_observations,
)
from blueearth_cst.shared.provenance import file_sha256


def runs_for(index, a=None, b=None):
    """Unit-level decoded records matching historical-run/1 reader outputs."""
    stations = pd.DataFrame(
        {
            "wflow_id": ["1010", "1020"],
            "station_name": ["Upper", "Outlet"],
            "subbasin_id": [101, 102],
        }
    )
    record = dict(
        model_sha256="fixed",
        forcing_sha256="forcing",
        recipe={"precip_source": "era5"},
        time={
            "starttime": str(index[0] - pd.Timedelta(days=1)),
            "endtime": str(index[-1]),
            "timestepsecs": 86400,
        },
        units="m3 s-1",
        time_label="interval_end",
    )
    a = np.arange(1, len(index) + 1, dtype=float) if a is None else a
    b = np.asarray(a) * 2 if b is None else b
    return {
        "a": (
            record,
            pd.DataFrame({"1010": a, "1020": np.zeros(len(index))}, index=index),
            stations,
        ),
        "b": (
            copy.deepcopy(record),
            pd.DataFrame({"1020": np.zeros(len(index)), "1010": b}, index=index),
            stations.iloc[::-1].copy(),
        ),
    }


def test_common_support_independent_daily_monthly_arithmetic():
    index = pd.date_range("2001-01-29", periods=6)
    runs = runs_for(index, [2, 4, np.nan, 8, 10, 12], [1, -1, 3, 4, 5, 6])
    obs = pd.DataFrame({"1010": [1, 2, 3, 4, np.nan, 6]}, index=index)
    metadata, tables = prepare_comparison(
        runs, index, water_year_start="Oct", observations=obs
    )
    support = tables["support"].query('wflow_id == "1010"')
    assert support.simulated_common.tolist() == [True, False, False, True, True, True]
    assert support.observation_paired.tolist() == [
        True,
        False,
        False,
        True,
        False,
        True,
    ]
    assert support.water_year.tolist() == [2000] * 6
    scores = (
        tables["performance"]
        .query('wflow_id == "1010" and run == "a"')
        .set_index(["metric", "scale"])
    )
    # Independently selected pairs are sim=[2,8,12], obs=[1,4,6].
    assert scores.loc[("MSE", "daily"), "value"] == pytest.approx(53 / 3)
    assert scores.loc[("RMSE", "daily"), "value"] == pytest.approx(np.sqrt(53 / 3))
    assert scores.loc[("Pbias", "daily"), "value"] == pytest.approx(100)
    assert scores.loc[("NSE", "daily"), "value"] == pytest.approx(1 - 53 / (38 / 3))
    # Incomplete months: January pair [2,1], February means [10,5].
    assert scores.loc[("MSE", "monthly"), "value"] == pytest.approx(13)
    assert scores.loc[("KGE", "monthly"), "value"] == pytest.approx(1 - np.sqrt(2))
    assert scores.loc[("VE", "daily"), "value"] == pytest.approx(0)
    assert metadata["requested_start"] == "2001-01-29"
    monthly = tables["monthly"].query(
        'wflow_id == "1010" and series == "a" and support == "observation_paired"'
    )
    assert monthly.days.tolist() == [1, 2]
    assert monthly.discharge.tolist() == [2, 10]
    fdc = tables["flow_duration"].query(
        'wflow_id == "1010" and series == "a" and support == "observation_paired"'
    )
    assert fdc.discharge.tolist() == [12, 8, 2]
    assert fdc.exceedance.tolist() == [25, 50, 75]
    missing = tables["performance"].query('wflow_id == "1020"')
    assert missing.value.isna().all()
    assert set(missing.reason) == {"observations_unavailable"}


def test_reordered_stations_have_identical_results():
    index = pd.date_range("2000-01-01", periods=4)
    runs = runs_for(index)
    _, expected = prepare_comparison(runs, index)
    record, frame, stations = runs["b"]
    runs["b"] = record, frame.iloc[:, ::-1], stations.iloc[::-1]
    _, actual = prepare_comparison(runs, index)
    for key in actual:
        pd.testing.assert_frame_equal(actual[key], expected[key])


@pytest.mark.parametrize("n, excluded", [(365, 0), (366, 366), (400, 366)])
def test_calendar_year_warmup_preserves_leap_boundary(n, excluded):
    index = pd.date_range("2000-01-02", periods=n)
    metadata, tables = prepare_comparison(runs_for(index), index)
    assert tables["coverage"].warmup_days.tolist() == [excluded, excluded]
    assert metadata["warmup_exclusive_end"] == (
        "2000-01-02" if n == 365 else "2001-01-02"
    )
    assert set(tables["daily"].support) == {"simulated_common"}


@pytest.mark.parametrize("flow", [0.0, 5.0])
def test_zero_constant_flows_keep_errors_but_explain_undefined_skill(flow):
    index = pd.date_range("2000-01-01", periods=3)
    runs = runs_for(index, [flow] * 3, [flow] * 3)
    obs = pd.DataFrame({"1010": [flow] * 3}, index=index)
    _, tables = prepare_comparison(runs, index, observations=obs)
    scores = (
        tables["performance"]
        .query('wflow_id == "1010" and run == "a" and scale == "daily"')
        .set_index("metric")
    )
    assert scores.loc["RMSE", "value"] == 0
    assert np.isnan(scores.loc["KGE", "value"])
    assert scores.loc["KGE", "reason"]
    fdc = tables["flow_duration"].query(
        'wflow_id == "1010" and support == "observation_paired"'
    )
    assert set(fdc.discharge) == {flow}


@pytest.mark.parametrize("damage", ["model", "time", "station", "period"])
def test_incompatible_runs_and_unavailable_period_reject(damage):
    index = pd.date_range("2000-01-01", periods=3)
    runs = runs_for(index)
    record, _, stations = runs["b"]
    if damage == "model":
        record["model_sha256"] = "changed"
    elif damage == "time":
        record["time"]["timestepsecs"] = 3600
    elif damage == "station":
        stations.loc[stations.wflow_id.eq("1010"), "subbasin_id"] = 999
    else:
        index = pd.date_range("1999-12-31", periods=4)
    with pytest.raises(ValueError, match="incompatible|unavailable"):
        prepare_comparison(runs, index)


def test_explicit_window_and_observation_schema(tmp_path):
    dates = evaluation_dates({"start": 2000, "end": 2000})
    assert len(dates) == 366
    assert dates[0] == pd.Timestamp("2000-01-01")
    assert dates[-1] == pd.Timestamp("2000-12-31")
    dates = evaluation_dates({"start": 1999, "end": 2000}, "Oct")
    assert len(dates) == 731
    assert dates[0] == pd.Timestamp("1999-10-01")
    assert dates[-1] == pd.Timestamp("2001-09-30")
    for window in (
        {"start": "2000-01-01", "end": "2000-12-31"},
        {"start": True, "end": 2000},
        {"start": 2001, "end": 2000},
    ):
        with pytest.raises(ValueError, match="integers|ordered"):
            evaluation_dates(window)
    path = tmp_path / "obs.csv"
    path.write_text("time;01010\n2000-01-01;1\n2000-01-03;2\n")
    assert read_observations(path, {"1010", "1020"}).columns.tolist() == ["1010"]
    path.write_text("time;1010;01010\n2000-01-01;1;2\n")
    with pytest.raises(ValueError, match="duplicate"):
        read_observations(path, {"1010"})


def write_tables(root: Path, metadata: dict, tables: dict) -> None:
    (root / "tables").mkdir(parents=True)
    for key, table in tables.items():
        table.to_csv(root / "tables" / f"{key}.csv", index=False, float_format="%.17g")
    metadata["table_hashes"] = {
        p.name: file_sha256(p) for p in (root / "tables").iterdir()
    }
    (root / "comparison.json").write_text(json.dumps(metadata))


def test_public_comparison_expands_year_window_without_trimming(tmp_path, monkeypatch):
    import yaml

    from blueearth_cst.model import historical_comparison as comparison

    runs = runs_for(pd.date_range("1999-01-01", "2002-12-31"))
    for name in runs:
        directory = tmp_path / name
        directory.mkdir()
        (directory / "complete.json").write_text("retained fixture")
    monkeypatch.setattr(
        comparison, "run_directory", lambda project, name: project / name
    )
    monkeypatch.setattr(
        comparison, "read_historical_run", lambda project, name: runs[name]
    )
    config = tmp_path / "comparison.yml"
    settings = {
        "project_dir": ".",
        "runs": ["a", "b"],
        "output_dir": "result",
        "evaluation_window": {"start": 2000, "end": 2001},
        "water_year_start": "Oct",
    }
    config.write_text(yaml.safe_dump(settings))
    metadata, tables = read_comparison(comparison.compare(config))
    assert metadata["requested_start"] == "2000-10-01"
    assert metadata["requested_end"] == "2002-09-30"
    assert metadata["warmup_exclusive_end"] == "2000-01-01"
    assert len(tables["support"]) == 730 * 2
    settings["output_dir"] = "unavailable"
    settings["evaluation_window"]["end"] = 2002
    config.write_text(yaml.safe_dump(settings))
    with pytest.raises(ValueError, match="unavailable"):
        comparison.compare(config)
    assert not (tmp_path / "unavailable").exists()


def test_retained_tables_render_without_source_runs_and_damage_refuses(
    tmp_path, monkeypatch
):
    import matplotlib.dates as mdates

    from blueearth_cst.model import plot_historical_comparison as plotting
    from blueearth_cst.model.plot_historical_comparison import render

    index = pd.date_range("2000-01-01", periods=3)
    runs = runs_for(index)
    metadata, tables = prepare_comparison(
        runs, index, observations=pd.DataFrame(index=index)
    )
    root = tmp_path / "comparison"
    write_tables(root, metadata, tables)
    # There is no mutable model, observation file or retained-run directory here.
    destination = tmp_path / "rendered"
    exporter = plotting.export_figure

    def inspect_empty_domains(fig, path, **kwargs):
        if "observation_paired" in path.name:
            assert len(fig.axes[0].get_yticks()) == 0
            if "hydrograph" in path.name:
                assert fig.axes[0].get_xlim() == pytest.approx(
                    mdates.date2num([index[0], index[-1]])
                )
            elif "fdc" in path.name:
                assert fig.axes[0].get_xlim() == (0, 100)
        exporter(fig, path, **kwargs)

    monkeypatch.setattr(plotting, "export_figure", inspect_empty_domains)
    render(root, destination)
    assert len(list((destination / "plots").glob("*.png"))) == 12
    assert (destination / "_engine/plots/figure_titles.json").is_file()
    with (root / "tables/daily.csv").open("a") as stream:
        stream.write("changed")
    with pytest.raises(ValueError, match="damaged"):
        read_comparison(root)


@pytest.mark.parametrize(
    ("first", "last", "start_month", "expected"),
    [
        # Interval-end responses: a run started on 1 January has no response
        # for that day, so its first calendar year is incomplete.
        ("2010-01-02", "2016-12-31", "Jan", {"start": 2011, "end": 2016}),
        ("2010-01-01", "2016-12-31", "Jan", {"start": 2010, "end": 2016}),
        ("2010-01-02", "2016-12-30", "Jan", {"start": 2011, "end": 2015}),
        # Water years are labelled by their start year.
        ("2010-01-02", "2016-12-31", "Oct", {"start": 2010, "end": 2015}),
    ],
)
def test_complete_reporting_years_stay_inside_the_responses(
    first, last, start_month, expected
):
    window = complete_reporting_years(
        pd.Timestamp(first), pd.Timestamp(last), start_month
    )
    assert window == expected
    dates = evaluation_dates(window, start_month)
    assert dates[0] >= pd.Timestamp(first) and dates[-1] <= pd.Timestamp(last)


def test_complete_reporting_years_refuse_a_window_without_one():
    with pytest.raises(ValueError, match="no complete reporting year"):
        complete_reporting_years(pd.Timestamp("2010-01-02"), pd.Timestamp("2010-12-31"))
