"""Unit tests for the parse-time historical-window guard (snake_utils).

Layer A of the window-adequacy check: what the config REQUESTS. Layer B --
what the staged source actually covers -- is tested in
``tests/test_extract_historical_climate.py``.
"""

import pytest

from blueearth_cst.shared.snake_utils import (
    MIN_HISTORICAL_YEARS,
    historical_window_days,
    meets_min_historical_years,
    validate_historical_window,
)


def _window(start, end):
    """R14 `C-70`: the window is a pair of INCLUSIVE YEARS, not ISO timestamps.

    The retype is absorbed in ``historical_window_bounds`` (the one parser), so
    every case below asserts the same property it always did on the new
    spelling: `{start: 2000, end: 2016}` spans 2000-01-01 to 2016-12-31.
    """
    return {"start": start, "end": end}


# --- historical_window_days ------------------------------------------------


def test_span_of_a_full_year():
    assert historical_window_days(_window(2000, 2000)) == 365


def test_year_endpoints_are_accepted():
    """The spelling every shipped config uses since R14, and it is INCLUSIVE.

    2000..2020 is 2000-01-01 to 2020-12-31, the same span the v1 ISO pair
    spelled out -- which is what makes the retype value-preserving for the
    configs the toolbox ships.
    """
    assert historical_window_days(_window(2000, 2020)) == 7670


def test_missing_key_names_the_key():
    with pytest.raises(ValueError, match="missing 'end'"):
        historical_window_days({"start": 2000})


def test_unparseable_endpoint_names_the_key_and_value():
    with pytest.raises(ValueError, match=r"climate.window.start"):
        historical_window_days(_window("not-a-year", 2001))


def test_non_mapping_is_rejected():
    with pytest.raises(ValueError, match="must be a mapping"):
        historical_window_days(["2000-01-01", "2001-01-01"])


# --- meets_min_historical_years (calendar arithmetic) ----------------------


def test_exactly_sixteen_calendar_years_meets_the_floor():
    """Inclusive: the same date 16 years later passes, one day earlier does
    not. Calendar arithmetic, so this holds regardless of leap-day count."""
    from datetime import datetime as dt

    assert meets_min_historical_years(dt(2000, 1, 1), dt(2016, 1, 1))
    assert not meets_min_historical_years(dt(2000, 1, 1), dt(2015, 12, 31))


def test_leap_day_start_shifts_to_the_same_date():
    """The ordinary leap case needs no clamp: 16 is a multiple of 4, so a leap
    year plus 16 is another leap year and 29 Feb survives the shift."""
    from datetime import datetime as dt

    assert meets_min_historical_years(dt(2000, 2, 29), dt(2016, 2, 29))
    assert not meets_min_historical_years(dt(2000, 2, 29), dt(2016, 2, 28))


def test_leap_day_clamps_across_a_non_leap_century():
    """The one case where the clamp fires: 2084 + 16 = 2100, which the
    century rule makes a non-leap year, so 29 Feb has to become 28 Feb rather
    than raise ValueError."""
    from datetime import datetime as dt

    assert meets_min_historical_years(dt(2084, 2, 29), dt(2100, 2, 28))
    assert not meets_min_historical_years(dt(2084, 2, 29), dt(2100, 2, 27))


# --- validate_historical_window --------------------------------------------


def test_exactly_the_floor_is_accepted():
    assert validate_historical_window(_window(2000, 2016)) > 0


def test_one_day_under_the_floor_is_rejected():
    with pytest.raises(ValueError, match=f"{MIN_HISTORICAL_YEARS}-year minimum"):
        validate_historical_window(_window(2000, 2015))


def test_the_shipped_window_passes():
    assert validate_historical_window(_window(2000, 2020)) == 7670


def test_rejection_names_the_window_its_length_the_floor_and_the_cause():
    """The message has to be actionable on its own -- it is what replaces a
    MissingOutputException nine rules into the DAG, or a weathergenr crash a
    whole workflow away."""
    with pytest.raises(ValueError) as excinfo:
        validate_historical_window(_window(2000, 2006))
    message = str(excinfo.value)
    assert "2000-01-01" in message and "2006-12-31" in message
    # 7, not 6: the endpoints are INCLUSIVE years since R14 (`C-70`), so
    # 2000..2006 spans seven of them. The v1 ISO pair 2000-01-01..2006-01-01
    # spanned six, and the number moving here is the retype being honest rather
    # than a defect -- the same span in days, counted from a different endpoint.
    assert "spans 7 years" in message
    assert str(MIN_HISTORICAL_YEARS) in message
    assert "weathergenr" in message


def test_reversed_window_is_rejected_and_says_so():
    """A negative span is under the floor, but 'below the minimum' alone would
    send the reader looking for missing data rather than a swapped pair."""
    with pytest.raises(ValueError) as excinfo:
        validate_historical_window(_window(2020, 2000))
    assert "BEFORE starttime" in str(excinfo.value)


def test_a_ten_year_window_is_rejected_not_merely_warned():
    """The floor is UNIFIED (owner ruling 2026-08-01): WF1 no longer accepts a
    record WF3 would reject. An earlier revision let this through with a
    warning."""
    with pytest.raises(ValueError):
        validate_historical_window(_window(2000, 2010))
