"""Complete-hydrological-year bounds (design §5.4 A1), after the 2026-07-30 fix.

Falsifiers L1-L5 of `dev/r08/2026-07-30_wf2-hydyear-fix-falsifier.md`. The
October case is the one to watch: it was ALREADY correct before the fix, so a
naive `+1` would have broken it.
"""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from blueearth_cst.projections.get_change_climate_proj import hydrological_year_bounds


def _series(first, last):
    """Monthly series spanning `first`..`last` inclusive, e.g. '1985-01','2014-12'."""
    t = pd.date_range(first, last, freq="MS")
    return xr.DataArray(
        np.arange(len(t), dtype="float64"), dims="time", coords={"time": t}
    ).to_dataset(name="v")


# --- L1: the two cases A1 names ------------------------------------------------


def test_L1_january_start_over_30_calendar_years_gives_30_hydrological_years():
    start, end, n = hydrological_year_bounds(_series("1985-01", "2014-12"), "Jan")
    assert (str(start.date()), str(end.date())) == ("1985-01-01", "2014-12-01")
    assert n == 30


def test_L1_october_start_over_the_same_span_gives_29_and_was_already_correct():
    """A naive `+1` fix would have broken this. The 2014-10 year is genuinely
    incomplete -- the data stops 2014-12."""
    start, end, n = hydrological_year_bounds(_series("1985-01", "2014-12"), "Oct")
    assert (str(start.date()), str(end.date())) == ("1985-10-01", "2014-09-01")
    assert n == 29


# --- L2: the seed gains exactly one year --------------------------------------


def test_L2_the_seed_window_gains_the_year_the_config_asked_for():
    """[1990, 2010] slices 1990-01..2010-12: 21 complete January-start years."""
    start, end, n = hydrological_year_bounds(_series("1990-01", "2010-12"), "Jan")
    assert (str(start.date()), str(end.date())) == ("1990-01-01", "2010-12-01")
    assert n == 21


# --- L3/L4: partial years at either end are still dropped ---------------------


def test_L3_a_mid_year_end_still_drops_the_trailing_partial_year():
    """The property the old code got right; the fix must preserve it."""
    start, end, n = hydrological_year_bounds(_series("1985-01", "2014-06"), "Jan")
    assert str(end.date()) == "2013-12-01"
    assert n == 29


def test_L4_a_late_start_drops_the_leading_partial_year():
    """The 1985 hydrological year began before the data did."""
    start, end, n = hydrological_year_bounds(_series("1985-03", "2014-12"), "Jan")
    assert str(start.date()) == "1986-01-01"
    assert n == 29


def test_L3_L4_both_ends_partial():
    start, end, n = hydrological_year_bounds(_series("1985-03", "2014-06"), "Jan")
    assert (str(start.date()), str(end.date())) == ("1986-01-01", "2013-12-01")
    assert n == 28


# --- L5: no complete year fails loudly ----------------------------------------


def test_L5_a_span_shorter_than_one_year_raises():
    """An empty reference propagates as an empty denominator into every relative
    change factor."""
    with pytest.raises(ValueError, match="no complete hydrological year"):
        hydrological_year_bounds(_series("1985-03", "1985-11"), "Jan")


def test_L5_exactly_one_complete_year_is_fine():
    start, end, n = hydrological_year_bounds(_series("1985-01", "1985-12"), "Jan")
    assert n == 1 and (str(start.date()), str(end.date())) == ("1985-01-01", "1985-12-01")


# --- start months across the year ---------------------------------------------


@pytest.mark.parametrize(
    "month,expected_n", [("Jan", 30), ("Feb", 29), ("Jul", 29), ("Oct", 29), ("Dec", 29)]
)
def test_only_a_january_start_yields_30_over_1985_2014(month, expected_n):
    """A1's exact claim: 30 only for January, 29 for every other start month."""
    _, _, n = hydrological_year_bounds(_series("1985-01", "2014-12"), month)
    assert n == expected_n
