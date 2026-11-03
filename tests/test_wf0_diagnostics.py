import numpy as np
import pandas as pd
import pytest

from blueearth_cst.climate_analysis import diagnostics as dg


def daily(start: str, end: str, value: float = 0.0) -> pd.Series:
    idx = pd.date_range(start, end, freq="D")
    return pd.Series(value, index=idx, dtype=float)


def test_monthly_percentiles_use_calendar_month_reference_and_midrank_ties():
    monthly = pd.DataFrame(
        {
            "calendar_month": [1] * 7 + [2, 2, 3],
            "reporting_year": [
                2000,
                2001,
                2002,
                2003,
                2004,
                2005,
                2006,
                2000,
                2001,
                2005,
            ],
            "total": [0, 0, 10, 20, 5, 30, np.nan, 100, 100, 10],
        }
    )
    percentiles, support = dg.monthly_reference_diagnostics(monthly, (2000, 2003))
    np.testing.assert_allclose(
        percentiles,
        [25, 25, 62.5, 87.5, 50, 100, np.nan, 50, 50, np.nan],
        equal_nan=True,
    )
    assert support[0] == {"calendar_month": 1, "start": 2000, "end": 2003, "count": 4}
    assert support[1]["count"] == 2
    assert support[2] == {"calendar_month": 3, "start": None, "end": None, "count": 0}


def test_wet_days_and_sdii_use_threshold():
    p = daily("2001-01-01", "2001-12-31")
    p.iloc[[0, 1, 2]] = [0.9, 1.0, 5.0]
    ws = dg.wet_day_stats(p, 1, 1.0)
    assert ws.loc[2001, "wet_days"] == 2
    assert ws.loc[2001, "sdii"] == pytest.approx(3.0)
    assert ws.loc[2001, "total"] == pytest.approx(6.9)


def test_rx5day_does_not_span_reporting_year_boundary():
    p = daily("2000-01-01", "2001-12-31")
    p["2000-12-29":"2001-01-02"] = 10.0  # 3 days in 2000, 2 in 2001
    ex = dg.extremes(p, 1, 1.0)
    assert ex.loc[2000, "Rx5day"] == pytest.approx(30.0)
    assert ex.loc[2001, "Rx5day"] == pytest.approx(20.0)


def test_water_year_labels_by_start_year():
    p = daily("2000-10-01", "2002-09-30", 1.0)
    assert list(dg.full_years(p, 10)) == [2000, 2001]
    assert list(dg.coverage(p, 10).index[:3]) == [10, 11, 12]


def test_missing_day_invalidates_year_and_month():
    p = daily("2000-01-01", "2001-12-31", 2.0)
    p["2001-03-05"] = np.nan
    assert list(dg.valid_years(p, 1)) == [2000]
    assert np.isnan(dg.annual(p, 1).loc[2001])
    assert np.isnan(dg.monthly_totals(p)["2001-03-01"])


def test_dry_spells_drop_censored_runs():
    # dry 3 at start, dry 4 before a gap, dry 1 after it and dry 5 at the end are censored; dry 2 is complete
    v = [0, 0, 0, 5, 0, 0, 5, 0, 0, 0, 0, np.nan, 0, 5, 0, 0, 0, 0, 0]
    p = pd.Series(v, index=pd.date_range("2000-01-01", periods=len(v)), dtype=float)
    assert sorted(dg.dry_spells(p, 1.0)) == [2]


def test_accumulation_timing_halfway():
    p = daily("2001-01-01", "2001-12-31")
    p["2001-01-10"] = 50.0
    p["2001-07-01"] = 50.0
    t = dg.accumulation_timing(p, 1)
    assert (
        t.loc[2001, 0.25] == 10 and t.loc[2001, 0.5] == 10 and t.loc[2001, 0.75] == 182
    )


def test_spi_reference_moments():
    rng = np.random.default_rng(1)
    idx = pd.date_range("1961-01-01", "2020-12-31", freq="D")
    p = pd.Series(rng.gamma(0.6, 8.0, idx.size), index=idx)
    for k in dg.SPI_SCALES:
        s = dg.spi(p, k, (1961, 2020)).dropna()
        assert abs(s.mean()) < 0.1 and abs(s.std() - 1) < 0.1


def test_tolerance_keeps_year_and_infills_totals():
    p = daily("2001-01-01", "2001-12-31", 2.0)
    p["2001-03-05"] = np.nan
    p["2001-03-06"] = np.nan
    tol = dg.Completeness(max_month=3, max_year=15)
    assert list(dg.valid_years(p, 1)) == []
    assert list(dg.valid_years(p, 1, tol)) == [2001]
    assert dg.annual(p, 1, tol=tol).loc[2001] == pytest.approx(
        730.0
    )  # gaps filled with March mean
    ws = dg.wet_day_stats(p, 1, 1.0, tol)
    assert ws.loc[2001, "wet_days"] == pytest.approx(
        365.0
    )  # 363 observed, scaled to the year
    assert ws.loc[2001, "sdii"] == pytest.approx(2.0)
    assert dg.coverage_class(p, 1, tol).loc[3, 2001] == 1


def test_tolerance_limits_month_and_year():
    p = daily("2001-01-01", "2001-12-31", 2.0)
    p["2001-03-01":"2001-03-04"] = np.nan  # 4 > 3 per month
    assert list(dg.valid_years(p, 1, dg.Completeness(3, 15))) == []
    q = daily("2001-01-01", "2001-12-31", 2.0)
    for m in range(1, 7):
        q[f"2001-{m:02d}-01" : f"2001-{m:02d}-03"] = (
            np.nan
        )  # 3 per month, 18 > 15 per year
    assert list(dg.valid_years(q, 1, dg.Completeness(3, 15))) == []
    assert list(dg.valid_years(q, 1, dg.Completeness(3, 20))) == [2001]


def test_extremes_ignore_infilled_days():
    p = daily("2001-01-01", "2001-12-31", 0.0)
    p["2001-06-10"] = 40.0
    p["2001-06-11"] = np.nan  # infilling would add rain; extremes must not see it
    ex = dg.extremes(p, 1, 1.0, dg.Completeness(3, 15))
    assert ex.loc[2001, "Rx1day"] == pytest.approx(40.0)
    assert np.isnan(dg.extremes(p, 1, 1.0).loc[2001, "Rx1day"])


def test_spi_pearson3_reference_moments_and_checks():
    rng = np.random.default_rng(3)
    idx = pd.date_range("1961-01-01", "2020-12-31", freq="D")
    p = pd.Series(rng.gamma(0.6, 8.0, idx.size), index=idx)
    s = dg.spi(p, 3, (1961, 2020), dist="pearson3").dropna()
    assert abs(s.mean()) < 0.1 and abs(s.std() - 1) < 0.1
    chk = dg.spi_fit_checks(p, 3, (1961, 2020))
    assert (
        list(chk.index) == list(range(1, 13)) and (chk["n"] >= 59).all()
    )  # SPI-3 Jan-Feb 1961 lack history
    assert (chk["zeros"] == 0).all() and chk["sw_gamma"].between(0, 1).all()
    with pytest.raises(ValueError):
        dg.spi(p, 3, (1961, 2020), dist="weibull")


def test_drought_event_table_mckee_rule():
    v = [0.5, -0.4, -1.2, -0.3, 0.2, -0.5, -0.6, 0.1, -1.5, -0.2, np.nan, -2.0, -0.1]
    s = pd.Series(v, index=pd.date_range("2000-01-01", periods=len(v), freq="MS"))
    ev = dg.drought_event_table(s)
    # run 1 (months 2-4) qualifies; run 2 never reaches -1; run 3 ends at a gap; run 4 starts after it and ends the record
    assert list(ev["duration"]) == [3, 2, 2]
    assert ev.loc[0, "severity"] == pytest.approx(1.9) and not ev.loc[0, "censored"]
    assert list(ev["censored"]) == [False, True, True]
    assert ev.loc[0, "start"] == pd.Timestamp("2000-02-01") and ev.loc[
        0, "intensity"
    ] == pytest.approx(-1.2)


def test_sen_trend_uncorrected_matches_scipy():
    from scipy import stats

    rng = np.random.default_rng(5)
    x, yrs = rng.normal(size=40), np.arange(1980, 2020)
    r = dg.sen_trend(pd.Series(x, index=yrs), correction="none")
    sc = stats.theilslopes(x, yrs, 0.95)
    assert r["slope"] / 10 == pytest.approx(sc.slope)
    assert r["lo"] / 10 == pytest.approx(sc.low_slope, abs=2e-4)
    assert r["hi"] / 10 == pytest.approx(sc.high_slope, abs=2e-4)


def test_yue_wang_controls_false_positives_under_ar1():
    rng = np.random.default_rng(42)
    n, phi, reps = 51, 0.5, 200
    yrs = np.arange(n)
    hits = {"none": 0, "yue_wang": 0}
    for _ in range(reps):
        e = rng.normal(size=n + 50)
        y = np.zeros(n + 50)
        for k in range(1, n + 50):
            y[k] = phi * y[k - 1] + e[k]
        s = pd.Series(y[50:], index=yrs)
        for c in hits:
            hits[c] += dg.sen_trend(s, correction=c)["p"] < 0.05
    assert hits["none"] / reps > 0.15  # independence assumption fails badly
    assert hits["yue_wang"] / reps < 0.10  # correction brings it near nominal


def test_start_year_sensitivity_and_pt_anomalies():
    yrs = np.arange(1981, 2021)
    s = pd.Series(np.arange(40.0), index=yrs)
    sens = dg.start_year_sensitivity(s, min_years=15)
    assert sens.index[0] == 1981 and sens.index[-1] == 2006  # last start keeps 15 years
    assert np.allclose(sens["slope"], 10.0)  # 1 unit/year
    idx = pd.date_range("1991-01-01", "2000-12-31", freq="D")
    p = pd.Series(2.0, index=idx)
    t = pd.Series(25.0, index=idx)
    p["1995-06-01":"1995-08-31"] = 1.0  # JJA 1995 half as wet
    t["1995-06-01":"1995-08-31"] = 26.0
    pt = dg.pt_anomalies(p, t, (1991, 2000))
    row = pt[(pt["season"] == "JJA") & (pt["year"] == 1995)].iloc[0]
    assert row["p_anom_pct"] < -40 and row["t_anom"] == pytest.approx(0.9)
    assert not (
        (pt["season"] == "DJF") & (pt["year"] == 1991)
    ).any()  # Dec 1990 missing


def test_break_tests_find_a_shift_and_not_noise():
    rng = np.random.default_rng(1)
    yrs = np.arange(1990, 2021)
    noise = pd.Series(rng.normal(size=31), index=yrs)
    shifted = noise + np.where(yrs > 2005, 2.0, 0.0)
    for test in (dg.pettitt, dg.snht):
        hit = test(shifted)
        assert hit["p"] < 0.01 and 2003 <= hit["break_after"] <= 2007
        assert test(noise)["p"] > 0.05
