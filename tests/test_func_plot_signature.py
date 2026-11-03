# -*- coding: utf-8 -*-
"""Numerical tests for exported discharge performance indicators.

Figure rendering, layout and plotting-only checks are temporary review probes
outside the test suite. CSV arithmetic, bias conventions and paired daily/monthly
calculations remain covered here.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from blueearth_cst.shared.func_plot_signature import (
    compute_indicators,
    rsquared,
)

#: A CONSTANT observation is the only way to state a bias or an error exactly,
#: and it is degenerate for the variance-normalised indicators: ``skills`` divides
#: by ``nanstd(qobs) == 0`` and reports NSE/KGE as inf/nan. The tests that use a
#: flat observation assert on RMSE, Pbias and VE, never on those — so the
#: warning is a property of the fixture, not a finding. Scoped per test rather
#: than module-wide, so a real divide-by-zero elsewhere still surfaces.
_FLAT_OBS_DIVIDE = pytest.mark.filterwarnings(
    "ignore:(divide by zero|invalid value) encountered in scalar divide:RuntimeWarning"
)


def _daily(values, start="2001-01-01"):
    time = pd.date_range(start, periods=len(values), freq="D")
    return xr.DataArray(
        np.asarray(values, dtype="float64"), dims="time", coords={"time": time}
    )


# ---------------------------------------------------------------------------
# rsquared
# ---------------------------------------------------------------------------


def test_a_perfect_line_scores_one():
    x = np.arange(50, dtype="float64")
    assert rsquared(x, 3.0 * x + 7.0) == pytest.approx(1.0)


def test_the_score_is_the_SQUARE_so_an_inverse_relation_also_scores_one():
    """The trap: R^2 cannot tell a perfect fit from a perfect ANTI-fit.

    Anything using this to decide "is the simulation good" needs the sign from
    elsewhere. Pinned so the property is visible rather than surprising.
    """
    x = np.arange(50, dtype="float64")
    assert rsquared(x, -3.0 * x + 7.0) == pytest.approx(1.0)


def test_pure_noise_against_a_ramp_scores_near_zero():
    rng = np.random.default_rng(0)
    x = np.arange(500, dtype="float64")
    assert rsquared(x, rng.normal(size=500)) < 0.05


# ---------------------------------------------------------------------------
# compute_indicators — the table shape
# ---------------------------------------------------------------------------


@pytest.fixture
def two_years():
    """Two years of daily flow with a seasonal cycle."""
    return _daily(10.0 + 5.0 * np.sin(np.arange(730) / 30.0))


def test_the_table_is_seven_indicators_by_two_time_types(two_years):
    df = compute_indicators(two_years, two_years, "S1")

    assert df.index.names == ["indicators", "time_type"]
    assert list(df.index.get_level_values("indicators").unique()) == [
        "KGE",
        "NSE",
        "NSElog",
        "RMSE",
        "MSE",
        "Pbias",
        "VE",
    ]
    assert list(df.index.get_level_values("time_type").unique()) == [
        "daily",
        "monthly",
    ]
    assert df.shape == (14, 1)


def test_the_single_column_is_named_after_the_station(two_years):
    """The column name is how ``performance_indicators.csv`` identifies its station.

    A default of ``"station"`` would collide the moment a basin has more than
    one, so the caller's name must reach the frame unmodified.
    """
    assert compute_indicators(two_years, two_years, "outlet_101").columns.tolist() == [
        "outlet_101"
    ]
    assert compute_indicators(two_years, two_years).columns.tolist() == ["station"]


# ---------------------------------------------------------------------------
# compute_indicators — the arithmetic
# ---------------------------------------------------------------------------


def test_a_simulation_identical_to_the_observation_scores_perfectly(two_years):
    """The identities every one of the seven indicators must satisfy at qsim == qobs.

    Efficiencies go to 1, errors to 0. This is the single check that catches an
    argument-order swap in any of the fourteen ``skills`` calls, since every
    indicator here is symmetric ONLY at a perfect fit.
    """
    df = compute_indicators(two_years, two_years, "S")[["S"]]

    for indicator in ("KGE", "NSE", "NSElog", "VE"):
        assert df.loc[indicator, "S"].tolist() == pytest.approx([1.0, 1.0])
    for indicator in ("RMSE", "MSE", "Pbias"):
        assert df.loc[indicator, "S"].tolist() == pytest.approx([0.0, 0.0])


@_FLAT_OBS_DIVIDE
def test_pbias_is_positive_when_the_simulation_over_predicts():
    """Sign convention, and it is the easy one to invert.

    A simulation 10% above the observation reports ``+10``. Reported the other
    way round, a wet bias would read as a dry one in every evaluation sheet.
    """
    obs = _daily(np.full(730, 10.0))
    sim = _daily(np.full(730, 11.0))

    df = compute_indicators(sim, obs, "S")
    assert df.loc[("Pbias", "daily"), "S"] == pytest.approx(10.0)
    assert df.loc[("Pbias", "monthly"), "S"] == pytest.approx(10.0)


@_FLAT_OBS_DIVIDE
def test_the_monthly_row_forgives_what_the_daily_row_punishes():
    """Why the table carries two time types rather than one.

    A simulation that alternates +5/-5 about a flat observation has the right
    monthly mean and the wrong daily value everywhere. If the monthly row were
    computed from the same series as the daily one — a resample dropped or
    applied to only one operand — the two would agree and the distinction the
    table exists to draw would be gone.
    """
    obs = _daily(np.full(730, 10.0))
    sim = _daily(10.0 + 5.0 * (-1.0) ** np.arange(730))

    df = compute_indicators(sim, obs, "S")
    daily, monthly = df.loc[("RMSE", "daily"), "S"], df.loc[("RMSE", "monthly"), "S"]

    assert daily == pytest.approx(5.0)
    assert monthly < 1.0
    assert df.loc[("VE", "daily"), "S"] < df.loc[("VE", "monthly"), "S"]


@_FLAT_OBS_DIVIDE
def test_both_operands_are_resampled_not_just_the_simulation():
    """A one-sided resample is invisible at a perfect fit and wrong everywhere else.

    Observation constant, simulation seasonal: if ``qobs`` were compared at daily
    resolution against a monthly ``qsim`` (or vice versa), xarray would align on
    time and the monthly row would collapse to a handful of points. Pinned by
    the count the monthly row is built from — 24 months over two years.
    """
    obs = _daily(np.full(730, 10.0))
    sim = _daily(10.0 + 5.0 * np.sin(np.arange(730) / 30.0))

    monthly_rmse = compute_indicators(sim, obs, "S").loc[("RMSE", "monthly"), "S"]
    expected = float(
        np.sqrt(
            ((sim.resample(time="ME").mean("time") - 10.0) ** 2).mean("time").values
        )
    )
    assert monthly_rmse == pytest.approx(expected, rel=1e-6)


def test_indicators_use_concurrent_nonnegative_finite_days_before_monthly_means():
    sim = _daily([99, 2, 4, 6, np.nan, 10, 0, 14], start="2001-01-30")
    obs = _daily([1, 2, 3, 4, -1, 0, 7, np.inf], start="2001-01-31").drop_sel(
        time="2001-02-02"
    )
    result = compute_indicators(sim, obs, "S")
    # Four pairs: obs[1,2,0,7],sim[2,4,0,14]. January has one pair;
    # February's three pairs average3/6. Independent daily/monthly MSE oracles
    # distinguish pairing-before-resampling from each series' available means.
    assert result.loc[("MSE", "daily"), "S"] == pytest.approx(13.5)
    assert result.loc[("RMSE", "daily"), "S"] == pytest.approx(np.sqrt(13.5))
    assert result.loc[("NSE", "daily"), "S"] == pytest.approx(-25 / 29)
    assert result.loc[("KGE", "daily"), "S"] == pytest.approx(1 - np.sqrt(2))
    assert result.loc[("MSE", "monthly"), "S"] == pytest.approx(5.0)
    assert result.loc[("NSE", "monthly"), "S"] == pytest.approx(-4.0)
    assert result.loc[("Pbias", "daily"), "S"] == pytest.approx(100.0)
    assert result.loc[("VE", "daily"), "S"] == pytest.approx(0.0)
    log_obs = np.log(np.array([1.0, 2.0, 0.0, 7.0]) + 1e-6)
    log_sim = np.log(np.array([2.0, 4.0, 0.0, 14.0]) + 1e-6)
    expected_log = (
        1 - ((log_sim - log_obs) ** 2).sum() / ((log_obs - log_obs.mean()) ** 2).sum()
    )
    assert result.loc[("NSElog", "daily"), "S"] == pytest.approx(expected_log)
    support = result.attrs["comparison_support"]
    assert support["paired_days"] == 4
    assert support["paired_months"] == 2
    assert support["observed"]["omitted_dates"] == 1


def test_indicators_handle_zero_denominators_and_no_concurrent_pairs():
    zero = _daily(np.zeros(45))
    result = compute_indicators(zero, zero, "S")
    for indicator in ("RMSE", "MSE"):
        assert result.loc[indicator, "S"].tolist() == pytest.approx([0.0, 0.0])
    for indicator in ("KGE", "NSE", "NSElog", "Pbias", "VE"):
        assert result.loc[indicator, "S"].isna().all()
    disjoint = compute_indicators(zero, _daily(np.ones(5), start="2002-01-01"), "S")
    assert disjoint["S"].isna().all()
    assert disjoint.attrs["comparison_support"]["paired_days"] == 0
    invalid = compute_indicators(
        _daily([1.0, np.inf, -1.0, 4.0]), _daily([1.0, 2.0, 3.0, 4.0]), "S"
    )
    assert invalid.loc[("MSE", "daily"), "S"] == pytest.approx(0.0)
    assert invalid.loc[("NSE", "daily"), "S"] == pytest.approx(1.0)
    assert invalid.attrs["comparison_support"]["paired_days"] == 2
