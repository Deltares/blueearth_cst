"""Month-length weighting tests for step 5b (design §5.6).

Each test names the falsifier it discharges from
``dev/milestones/r08/2026-07-30_wf2-5b-falsifier.md``, written before any 5b code.
"""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from blueearth_cst.projections.calendar_weights import (
    CalendarError,
    assert_weightable,
    days_in_month,
    month_length_weights,
)
from blueearth_cst.projections.series_identity import CALENDAR_UNKNOWN


def _monthly(year_start, n_months):
    """A datetime64 monthly axis — deliberately Gregorian, like a real series.

    Anchored on the 1st, not mid-month: ``freq="MS"`` snaps a mid-month start
    FORWARD to the next month start, so ``"2001-01-15"`` silently begins the axis
    in February and shifts every assertion by one month. Caught by G1 failing with
    a February of 31 days.
    """
    times = pd.date_range(f"{year_start}-01-01", periods=n_months, freq="MS")
    assert times[0].month == 1, "helper must start in January or the indices lie"
    return xr.DataArray(
        np.arange(n_months, dtype="float64"), dims="time", coords={"time": times}
    )


# --- G1: lengths come from the CALENDAR, never from the axis -------------------


def test_G1_noleap_february_is_28_even_across_leap_years():
    """The trap A3 exists to prevent, re-entered by using the axis.

    `time.dt.days_in_month` on this datetime64 axis returns 29 for Feb 2000 and
    Feb 2004. A noleap model has no such day.
    """
    da = _monthly(2000, 60)  # 2000-2004, includes two Gregorian leap years
    w = month_length_weights(da, "noleap")
    februaries = w[1::12]
    assert set(februaries) == {28.0}, (
        f"noleap February must always be 28, got {set(februaries)}"
    )

    # And prove the axis WOULD have lied, so this test cannot pass vacuously.
    axis_lengths = da["time"].dt.days_in_month.values[1::12]
    assert 29 in set(axis_lengths), "axis should contain a Gregorian leap February"


def test_G1_gregorian_february_is_29_in_a_leap_year():
    """Same code path must still be right where Gregorian IS the model calendar."""
    da = _monthly(2000, 24)
    w = month_length_weights(da, "proleptic_gregorian")
    assert w[1] == 29.0  # Feb 2000, a leap year
    assert w[13] == 28.0  # Feb 2001


# --- G2: 360_day must make 5b a no-op -----------------------------------------


def test_G2_360_day_weights_are_uniform():
    """5b's analogue of 5a's strict-generalization claim."""
    da = _monthly(2001, 36)
    w = month_length_weights(da, "360_day")
    assert set(w) == {30.0}


def test_G2_360_day_weighted_mean_equals_the_unweighted_mean():
    values = np.array([2.0, 5.0, 11.0, 3.0, 7.0, 1.0, 9.0, 4.0, 6.0, 8.0, 10.0, 12.0])
    w = month_length_weights(_monthly(2001, 12), "360_day")
    np.testing.assert_allclose(np.average(values, weights=w), values.mean(), rtol=1e-15)


# --- G3: on noleap it must NOT be a no-op -------------------------------------


def test_G3_noleap_weighting_changes_the_annual_mean():
    values = np.array([2.0, 5.0, 11.0, 3.0, 7.0, 1.0, 9.0, 4.0, 6.0, 8.0, 10.0, 12.0])
    w = month_length_weights(_monthly(2001, 12), "noleap")
    assert set(w) != {w[0]}, "noleap month lengths are not uniform"
    assert not np.isclose(np.average(values, weights=w), values.mean(), rtol=1e-9)


# --- G4: analytic agreement, not merely "different" ---------------------------


def test_G4_weighted_annual_mean_matches_the_hand_computed_value():
    """G3 only proves something moved; this proves it moved to the right place."""
    values = np.arange(1.0, 13.0)
    lengths = np.array(
        [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31], dtype="float64"
    )
    w = month_length_weights(_monthly(2001, 12), "noleap")
    np.testing.assert_array_equal(w, lengths)
    np.testing.assert_allclose(
        np.average(values, weights=w),
        (values * lengths).sum() / lengths.sum(),
        rtol=1e-15,
    )


@pytest.mark.parametrize(
    "calendar,expected",
    [
        ("noleap", 365),
        ("365_day", 365),
        ("all_leap", 366),
        ("366_day", 366),
        ("360_day", 360),
    ],
)
def test_G4_year_lengths_are_exact_for_every_fixed_calendar(calendar, expected):
    assert sum(days_in_month(2001, m, calendar) for m in range(1, 13)) == expected


# --- G5: an unweightable calendar must RAISE ----------------------------------


def test_G5_unknown_calendar_raises_naming_the_source():
    """A3's sentinel must be refused, not silently weighted with Gregorian lengths."""
    with pytest.raises(CalendarError, match="unknown"):
        assert_weightable(CALENDAR_UNKNOWN, source="cmip6_X_ssp245_r1i1p1f1")
    with pytest.raises(CalendarError, match="cmip6_X_ssp245_r1i1p1f1"):
        assert_weightable(CALENDAR_UNKNOWN, source="cmip6_X_ssp245_r1i1p1f1")


def test_G5_empty_calendar_raises():
    """ "" is what the pre-schema-3 code wrote; it must not be weightable."""
    for value in ("", None, "   "):
        with pytest.raises(CalendarError, match="unknown"):
            assert_weightable(value)


def test_G5_unrecognised_calendar_raises_rather_than_approximating():
    with pytest.raises(CalendarError, match="not weightable"):
        assert_weightable("mayan_long_count")


def test_G5_month_length_weights_refuses_before_computing_anything():
    with pytest.raises(CalendarError):
        month_length_weights(_monthly(2001, 12), CALENDAR_UNKNOWN)


# --- G6: the PURPOSE -- cross-calendar comparability --------------------------


def test_G6_weighting_brings_two_calendars_closer_for_the_same_climate():
    """The claim none of the mechanics test: this is what 5b is FOR.

    Same underlying climate — a seasonal cycle sampled as monthly means — on a
    360_day model and a noleap one. Unweighted, their annual means differ purely
    because noleap months are unequal. Weighting must shrink that procedural gap.
    """
    month = np.arange(12)
    climate = 20.0 + 8.0 * np.sin(2 * np.pi * (month - 3) / 12)  # seasonal cycle

    wnl = month_length_weights(_monthly(2001, 12), "noleap")
    naive = climate.mean()

    # The 360_day weighted mean IS the naive mean (G2), so "do the two calendars
    # agree after weighting?" reduces to "does noleap weighting correct toward the
    # duration-weighted truth?"
    assert not np.isclose(np.average(climate, weights=wnl), naive, rtol=1e-9), (
        "noleap weighting must correct the naive mean"
    )

    # Direction check, which is what makes the correction meaningful rather than
    # merely present: February is the SHORTEST month, so an anomaly confined to it
    # must count for less than 1/12 of the year.
    spiky = np.full(12, 10.0)
    spiky[1] = 100.0  # February
    assert np.average(spiky, weights=wnl) < spiky.mean(), (
        "a short February must carry LESS than 1/12 of the annual mean"
    )
    # ...and the mirror: an anomaly in a 31-day month must count for MORE.
    spiky_jan = np.full(12, 10.0)
    spiky_jan[0] = 100.0  # January, 31 days
    assert np.average(spiky_jan, weights=wnl) > spiky_jan.mean()
