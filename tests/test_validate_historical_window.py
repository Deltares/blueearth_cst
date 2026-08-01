"""Unit tests for the parse-time historical-window guard (snake_utils).

Layer A of the window-adequacy check: what the config REQUESTS. Layer B --
what the staged source actually covers -- is tested in
``tests/test_extract_historical_climate.py``.
"""

import pytest

from blueearth_cst.shared.snake_utils import (
    MIN_HISTORICAL_DAYS,
    WEATHERGEN_MIN_YEARS,
    historical_window_days,
    validate_historical_window,
)


def _window(start, end):
    return {"starttime": start, "endtime": end}


# --- historical_window_days ------------------------------------------------

def test_span_of_a_full_year():
    assert historical_window_days(_window("2000-01-01", "2001-01-01")) == 366


def test_iso_datetime_endpoints_are_accepted():
    """The spelling every shipped config uses."""
    days = historical_window_days(
        _window("2000-01-01T00:00:00", "2020-12-31T00:00:00")
    )
    assert days == 7670


def test_missing_key_names_the_key():
    with pytest.raises(ValueError, match="missing 'endtime'"):
        historical_window_days({"starttime": "2000-01-01"})


def test_unparseable_endpoint_names_the_key_and_value():
    with pytest.raises(ValueError, match=r"historical_window.starttime"):
        historical_window_days(_window("not-a-date", "2001-01-01"))


def test_non_mapping_is_rejected():
    with pytest.raises(ValueError, match="must be a mapping"):
        historical_window_days(["2000-01-01", "2001-01-01"])


# --- validate_historical_window --------------------------------------------

def test_a_year_exactly_is_accepted():
    """The floor is inclusive: 365 days passes, 364 does not."""
    assert validate_historical_window(_window("2001-01-01", "2002-01-01")) == 365


def test_one_day_under_the_floor_is_rejected():
    with pytest.raises(ValueError) as excinfo:
        validate_historical_window(_window("2001-01-01", "2001-12-31"))
    assert "364 days" in str(excinfo.value)


def test_the_shipped_window_passes():
    assert validate_historical_window(
        _window("2000-01-01T00:00:00", "2020-12-31T00:00:00")
    ) == 7670


def test_rejection_names_the_window_the_floor_and_the_cause():
    """The message has to be actionable on its own -- it is what replaces a
    MissingOutputException nine rules into the DAG."""
    with pytest.raises(ValueError) as excinfo:
        validate_historical_window(_window("2000-01-01", "2000-06-01"))
    message = str(excinfo.value)
    assert "2000-01-01" in message and "2000-06-01" in message
    assert str(MIN_HISTORICAL_DAYS) in message
    assert "plot_results" in message


def test_reversed_window_is_rejected_and_says_so():
    """A negative span is under the floor, but 'below the minimum' alone would
    send the reader looking for missing data rather than a swapped pair."""
    with pytest.raises(ValueError) as excinfo:
        validate_historical_window(_window("2020-01-01", "2000-01-01"))
    assert "BEFORE starttime" in str(excinfo.value)


def test_the_advisory_floor_is_not_enforced_here():
    """WEATHERGEN_MIN_YEARS is advisory: WF1 alone on a 10-year record is
    legitimate, so the parse-time guard must not reject it."""
    ten_years = _window("2000-01-01", "2010-01-01")
    assert validate_historical_window(ten_years) > 0
    assert WEATHERGEN_MIN_YEARS == 16
