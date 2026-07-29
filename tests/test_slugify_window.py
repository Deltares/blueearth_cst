"""Unit tests for blueearth_cst.shared.snake_utils.slugify_window.

Renders a historical-climate window to the dataset-store key component
YYYYMMDD_YYYYMMDD (design §4/§4c/§4d), enforcing the day-resolution invariant:
a nonzero time-of-day fails loud rather than colliding onto a shared day key.
"""
from __future__ import annotations


import pytest

from blueearth_cst.shared.snake_utils import slugify_window  # noqa: E402


def test_iso_midnight_window_renders_compact():
    assert (
        slugify_window("2000-01-01T00:00:00", "2020-12-31T00:00:00")
        == "20000101_20201231"
    )


def test_date_only_window_renders_compact():
    assert slugify_window("2015-01-01", "2020-12-31") == "20150101_20201231"


def test_seed_config_window():
    # The tracked test config window.
    assert (
        slugify_window("2015-01-01T00:00:00", "2020-12-31T00:00:00")
        == "20150101_20201231"
    )


def test_space_separated_datetime_accepted():
    assert (
        slugify_window("2000-01-01 00:00:00", "2001-01-01 00:00:00")
        == "20000101_20010101"
    )


@pytest.mark.parametrize(
    "start,end",
    [
        ("2000-01-01T12:00:00", "2020-12-31T00:00:00"),
        ("2000-01-01T00:00:00", "2020-12-31T06:30:00"),
        ("2000-01-01T00:00:01", "2020-12-31T00:00:00"),
    ],
)
def test_nonzero_time_of_day_raises(start, end):
    # §4c day-resolution invariant: sub-day components must fail loud.
    with pytest.raises(ValueError, match="time-of-day"):
        slugify_window(start, end)


@pytest.mark.parametrize("start,end", [("not-a-date", "2020-12-31"), ("2020-13-40", "2020-12-31")])
def test_unparseable_date_raises(start, end):
    with pytest.raises(ValueError):
        slugify_window(start, end)


def test_key_is_deterministic():
    a = slugify_window("2000-01-01T00:00:00", "2020-12-31T00:00:00")
    b = slugify_window("2000-01-01T00:00:00", "2020-12-31T00:00:00")
    assert a == b
