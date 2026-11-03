"""ERFA numerical conventions and independent checker discrimination."""

from dataclasses import replace
from datetime import timedelta

import numpy as np
import pandas as pd
import pytest

from blueearth_cst.indicators import checking, loader
from blueearth_cst.indicators.contract import IndicatorResult
from blueearth_cst.indicators.series import ResponseSeries
from blueearth_cst.indicators.sets.eflow_erfa.calculations import annual_metrics


def _series(daily):
    return ResponseSeries(
        "r",
        "q",
        "g",
        0,
        daily.index,
        "standard",
        timedelta(days=1),
        "interval_end",
        "m3 s-1",
        daily.to_numpy(),
        daily.isna().to_numpy(),
    )


def test_water_year_boundary_windows_changes_and_runs():
    daily = pd.Series(5.0, index=pd.date_range("2000-10-01", "2002-09-30"))
    daily.loc["2001-09-29":"2001-10-01"] = 20
    annual, coverage = annual_metrics(daily, "YE-SEP")
    assert list(annual.index) == [2001, 2002]
    assert coverage.retained.all()
    for metric, expected in {
        "amax3d": [15, 20],
        "wyday_amax1d": [364, 1],
        "num_high_pulses": [1, 1],
        "avg_dur_high_pulses": [2, 1],
        "num_rises": [1, 0],
        "num_falls": [0, 1],
        "mean_rate_rise": [15, np.nan],
        "mean_rate_fall": [np.nan, 15],
        "num_zero_seq": [0, 0],
        "avg_dur_zero_seq": [0, 0],
        "time_main_zero_seq": [np.nan, np.nan],
    }.items():
        np.testing.assert_allclose(annual[metric], expected, atol=1e-12, equal_nan=True)


def test_zero_run_missing_breaks_and_earliest_ties():
    daily = pd.Series(5.0, index=pd.date_range("2000-10-01", "2001-09-30"))
    daily.iloc[99:103] = 0
    daily.iloc[199:203] = 0
    annual, _ = annual_metrics(daily, "YE-SEP")
    expected = {
        "num_low_pulses": 2,
        "avg_dur_low_pulses": 4,
        "num_zero_seq": 2,
        "avg_dur_zero_seq": 4,
        "time_main_zero_seq": 100,
        "amin7d": 15 / 7,
    }
    for metric, value in expected.items():
        assert annual.loc[2001, metric] == pytest.approx(value)
    daily.iloc[100] = np.nan
    annual, _ = annual_metrics(daily, "YE-SEP")
    assert annual.loc[2001, "num_zero_seq"] == 3
    assert annual.loc[2001, "time_main_zero_seq"] == 200
    assert annual.loc[2001, "avg_dur_zero_seq"] == pytest.approx(7 / 3)


def test_leap_month_completeness_and_partial_year_evidence():
    daily = pd.Series(5.0, index=pd.date_range("1999-10-01", "2001-12-31"))
    daily.loc["2000-02-01":"2000-02-06"] = np.nan
    annual, evidence = annual_metrics(daily, "YE-SEP")
    assert list(annual.index) == [2000, 2001]
    assert list(evidence.expected_days) == [366, 365, 365]
    assert list(evidence.retained) == [True, True, False]
    assert np.isnan(annual.loc[2000, "february_flow"])
    assert annual.loc[2001, "february_flow"] == 5
    with pytest.raises(ValueError, match="no retained years"):
        annual_metrics(daily * np.nan, "YE-SEP")


@pytest.mark.parametrize("anchor", ["YE-DEC", "YE-SEP", "YE-MAR", "YS-OCT"])
def test_all_70_indicators_against_independent_checker(anchor, monkeypatch):
    from blueearth_cst.indicators.sets.eflow_erfa import calculations

    time = pd.date_range("1999-01-01", "2004-12-31")
    position = np.arange(len(time))
    daily = pd.Series((position % 73).astype(float), index=time)
    daily.iloc[100:110] = 0
    daily.iloc[200:207] = np.nan
    series = [_series(daily)]
    declarations = loader.declarations("eflow-erfa")
    calls = []
    original = calculations.annual_metrics

    def counted(*args):
        calls.append(args)
        return original(*args)

    monkeypatch.setattr(calculations, "annual_metrics", counted)
    prepared = {}
    assert len(declarations) == 70
    assert len({item.name for item in declarations}) == 70
    for indicator in declarations:
        result = loader.evaluate(indicator, series, anchor=anchor, _prepared=prepared)
        checked = checking.check(indicator, series, result, anchor=anchor)
        assert checked["numerical_checks"] == 1
        assert result.diagnostics[0].name == "erfa_year_coverage"
        assert set(result.diagnostics[0].data.location_id) == {"g"}
    assert len(calls) == 1, "one annual table must supply all 70 summaries"
    indicator = declarations[0]
    result = loader.evaluate(indicator, series, anchor=anchor)
    with pytest.raises(ValueError, match="numerical mismatch"):
        checking.check(
            indicator, series, IndicatorResult(result.values + 1), anchor=anchor
        )


def test_prepared_erfa_tables_are_scoped_to_response_objects_and_anchor():
    daily = pd.Series(
        np.arange(1096, dtype=float) % 73,
        index=pd.date_range("2000-01-01", periods=1096),
    )
    indicator = loader.declarations("eflow-erfa")[0]
    prepared = {}
    original = replace(_series(daily), values=daily.to_numpy(copy=True))
    for series, anchor in (
        ([original], "YE-DEC"),
        ([_series(daily + 50)], "YE-DEC"),
        ([original], "YE-SEP"),
    ):
        actual = loader.evaluate(indicator, series, anchor=anchor, _prepared=prepared)
        checking.check(indicator, series, actual, anchor=anchor)
    # A later public call must observe even an in-place edit to an earlier array.
    series[0].values[:] += 100
    actual = loader.evaluate(indicator, series, anchor=anchor)
    checking.check(indicator, series, actual, anchor=anchor)


def test_linear_iqr_missing_metric_and_response_refusals():
    daily = pd.Series(
        np.repeat([1.0, 3.0, 9.0], [366, 365, 365]),
        index=pd.date_range("2000-01-01", "2002-12-31"),
    )
    declarations = {item.name: item for item in loader.declarations("eflow-erfa")}
    for suffix, expected in (("med", 3), ("iqr", 4)):
        item = declarations[f"q_amax1d_{suffix}"]
        result = loader.evaluate(item, [_series(daily)], anchor="YE-DEC")
        assert result.values["g"] == expected
        item = declarations[f"q_time_main_zero_seq_{suffix}"]
        assert np.isnan(
            loader.evaluate(item, [_series(daily)], anchor="YE-DEC").values["g"]
        )
    item = declarations["q_amax1d_med"]
    with pytest.raises(ValueError, match="timestep"):
        loader.evaluate(item, [_series(daily.drop(daily.index[10]))], anchor="YE-DEC")
    broken = _series(daily)
    with pytest.raises(ValueError, match="infinite"):
        loader.evaluate(
            item, [replace(broken, values=np.full(len(daily), np.inf))], anchor="YE-DEC"
        )
