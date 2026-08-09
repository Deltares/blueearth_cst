"""Unit tests for blueearth_cst/shared/metrics_definition.py — pure-pandas streamflow metrics.

Testing pattern (M02c convention; see dev/milestones/phase-1/m02c/test-coverage-design.md):
- One test file per source module: tests/test_<module>.py
- Heavy deps (hydromt, xarray, geopandas) stubbed via sys.modules.setdefault
  at the top of the file. See tests/test_stage_data.py for the canonical
  example. This file uses no stubs because the tested functions are pure
  pandas/numpy.
- Inline fixtures per file. Promote to conftest.py only if reused in 3+ files.
- No Snakemake script injection — call functions directly.
"""

from __future__ import annotations


import numpy as np
import pandas as pd
import pytest

from blueearth_cst.shared import metrics_definition  # noqa: E402


def _daily_df(values, start="2020-01-01"):
    """Build a 1-column daily-indexed DataFrame for metric inputs."""
    idx = pd.date_range(start=start, periods=len(values), freq="D")
    return pd.DataFrame({"q": values}, index=idx)


def _three_year_index():
    """Three full calendar years of daily timestamps (2020-01-01 → 2022-12-31)."""
    return pd.date_range("2020-01-01", "2022-12-31", freq="D")


def test_Q7d_total_returns_rolling_seven_day_mean():
    df = _daily_df(list(range(20)))
    result = metrics_definition.Q7d_total(df)

    # First 6 rows are NaN (insufficient window); row 7 onward equals the
    # trailing 7-day mean of the input.
    assert result["q"].iloc[:6].isna().all()
    assert result["q"].iloc[6] == pytest.approx(np.mean(range(7)))
    assert result["q"].iloc[19] == pytest.approx(np.mean(range(13, 20)))


def test_Q7d_min_constant_series_returns_constant():
    idx = _three_year_index()
    df = pd.DataFrame({"q": np.full(len(idx), 100.0)}, index=idx)

    result = metrics_definition.Q7d_min(df)

    # Series of length 1 (one entry per column).
    assert result.iloc[0] == pytest.approx(100.0)


def test_Q7d_maxyear_averages_yearly_rolling_maxes():
    # 2020 = constant 50, 2021 = constant 100, 2022 = constant 150.
    # Rolling 7-day mean is constant within each year (after the first 6
    # days), so yearly max == yearly value. Mean of [50, 100, 150] == 100.
    parts = []
    for year, value in [("2020", 50.0), ("2021", 100.0), ("2022", 150.0)]:
        idx = pd.date_range(f"{year}-01-01", f"{year}-12-31", freq="D")
        parts.append(pd.DataFrame({"q": np.full(len(idx), value)}, index=idx))
    df = pd.concat(parts)

    result = metrics_definition.Q7d_maxyear(df)

    assert result.iloc[0] == pytest.approx(100.0)


def test_BFI_constant_series_returns_one():
    idx = _three_year_index()
    df = pd.DataFrame({"q": np.full(len(idx), 100.0)}, index=idx)

    result = metrics_definition.BFI(df)

    # Q7d_min == annmean for a constant series → BFI == 1.0.
    assert result.iloc[0] == pytest.approx(1.0)


def test_wetmonth_mean_finds_january_when_january_is_wet():
    # Construct a series where January is the only month with high values;
    # all other months are 0. wetmonth_mean should detect January as the
    # wet month and return January's annual mean (which equals the high
    # value, since the rest of the month doesn't dilute it).
    idx = _three_year_index()
    values = np.zeros(len(idx))
    jan_mask = idx.month == 1
    values[jan_mask] = 50.0
    df = pd.DataFrame({"q": values}, index=idx)

    result = metrics_definition.wetmonth_mean(df)

    # January mean per year == 50.0; mean across years == 50.0.
    assert result.iloc[0] == pytest.approx(50.0)


def test_drymonth_mean_finds_august_when_august_is_dry():
    # Mirror of wetmonth_mean: all months at 100.0 except August at 0.
    # August has the lowest monthlysum, so drymonth_mean returns August's
    # annual mean == 0.0.
    idx = _three_year_index()
    values = np.full(len(idx), 100.0)
    aug_mask = idx.month == 8
    values[aug_mask] = 0.0
    df = pd.DataFrame({"q": values}, index=idx)

    result = metrics_definition.drymonth_mean(df)

    assert result.iloc[0] == pytest.approx(0.0)
