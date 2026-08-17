"""Unit tests for ``blueearth_cst/shared/climate_window.py``.

The module that made ``shared.historical_window`` a CEILING rather than a demand
(2026-08-16). Three layers now guard the same fact and this is the middle one:

* ``tests/test_validate_historical_window.py`` — what the config REQUESTS.
* **here** — the requested/delivered comparison itself, as pure functions.
* ``tests/test_extract_historical_climate.py`` — the extraction driving them.
"""

from __future__ import annotations

import types

import numpy as np
import pandas as pd
import pytest

from blueearth_cst.shared.climate_window import (
    NARROWING_TOLERANCE,
    intersect_bounds,
    report_coverage,
    resolve_coverage,
    time_axis_bounds,
)
from blueearth_cst.shared.snake_utils import MIN_HISTORICAL_YEARS


def _ds(start, end, freq="D"):
    """A stand-in carrying only what ``time_axis_bounds`` reads."""
    values = pd.date_range(start, end, freq=freq).values
    return types.SimpleNamespace(time=types.SimpleNamespace(values=values))


def _coverage(requested, delivered, source="era5"):
    return resolve_coverage(
        (pd.Timestamp(delivered[0]), pd.Timestamp(delivered[1])),
        requested[0],
        requested[1],
        source,
    )


# --- time_axis_bounds --------------------------------------------------------


def test_bounds_come_from_the_axis_extremes_not_its_ends():
    """min/max, not ``[0]``/``[-1]``: a store need not be time-sorted."""
    values = pd.to_datetime(["2005-06-01", "2000-01-01", "2010-12-31"]).values
    ds = types.SimpleNamespace(time=types.SimpleNamespace(values=values))
    start, end = time_axis_bounds(ds)
    assert start == pd.Timestamp("2000-01-01")
    assert end == pd.Timestamp("2010-12-31")


def test_a_single_timestep_is_a_zero_length_span_not_a_failure():
    start, end = time_axis_bounds(_ds("2000-01-01", "2000-01-01"))
    assert start == end


@pytest.mark.parametrize(
    "ds",
    [
        types.SimpleNamespace(),  # no time coord at all
        types.SimpleNamespace(time=types.SimpleNamespace(values=np.array([]))),
        types.SimpleNamespace(time=types.SimpleNamespace(values=np.array(["a", "b"]))),
    ],
)
def test_an_unreadable_axis_returns_None_rather_than_guessing(ds):
    """Skip-rather-than-guess: a caller that cannot introspect the axis must not
    invent a window to check against."""
    assert time_axis_bounds(ds) is None


# --- intersect_bounds --------------------------------------------------------


def test_the_overlap_of_two_records_is_the_later_start_and_the_earlier_end():
    first = (pd.Timestamp("1981-01-01"), pd.Timestamp("2020-12-31"))
    second = (pd.Timestamp("1990-01-01"), pd.Timestamp("2015-12-31"))
    assert intersect_bounds(first, second) == (
        pd.Timestamp("1990-01-01"),
        pd.Timestamp("2015-12-31"),
    )


def test_records_that_miss_each_other_have_no_overlap():
    first = (pd.Timestamp("1981-01-01"), pd.Timestamp("1990-12-31"))
    second = (pd.Timestamp("2000-01-01"), pd.Timestamp("2010-12-31"))
    assert intersect_bounds(first, second) is None


def test_a_single_shared_instant_still_counts_as_an_overlap():
    """The boundary case, spelled out: ``start > end`` is the miss, not ``==``."""
    moment = pd.Timestamp("2000-01-01")
    assert intersect_bounds((moment, moment), (moment, moment)) == (moment, moment)


def test_an_unreadable_side_makes_the_overlap_unreadable():
    assert intersect_bounds(None, (pd.Timestamp("2000-01-01"),) * 2) is None


# --- WindowCoverage ----------------------------------------------------------


def test_a_source_that_covers_the_request_is_not_narrowed():
    coverage = _coverage(("2000-01-01", "2020-12-31"), ("1990-01-01", "2025-12-31"))
    assert not coverage.is_narrowed
    assert coverage.meets_floor


def test_a_few_late_days_are_not_a_narrowing():
    """A daily product whose first file starts three weeks into January is
    normal. Flagging it would train the reader to ignore the line that matters.
    """
    late = pd.Timestamp("2000-01-01") + NARROWING_TOLERANCE - pd.Timedelta(days=1)
    coverage = _coverage(("2000-01-01", "2020-12-31"), (late, "2020-12-31"))
    assert not coverage.is_narrowed


def test_missing_years_off_either_end_is_a_narrowing():
    assert _coverage(
        ("1980-01-01", "2020-12-31"), ("1995-01-01", "2020-12-31")
    ).is_narrowed
    assert _coverage(
        ("1980-01-01", "2020-12-31"), ("1980-01-01", "2005-12-31")
    ).is_narrowed


def test_the_floor_has_no_tolerance():
    """A floor with a tolerance is not a floor. One day under is under."""
    start = pd.Timestamp("2000-01-01")
    exactly = start.replace(year=start.year + MIN_HISTORICAL_YEARS)
    assert _coverage(("2000-01-01", "2030-12-31"), (start, exactly)).meets_floor
    assert not _coverage(
        ("2000-01-01", "2030-12-31"), (start, exactly - pd.Timedelta(days=1))
    ).meets_floor


def test_the_description_carries_both_windows_and_the_length():
    text = _coverage(
        ("2000-01-01", "2020-12-31"), ("2005-01-01", "2020-12-31")
    ).describe()
    assert "requested 2000-01-01..2020-12-31" in text
    assert "delivered 2005-01-01..2020-12-31" in text
    assert "years" in text


# --- report_coverage ---------------------------------------------------------


def test_the_delivered_span_is_logged_even_when_nothing_is_wrong(capsys):
    """A line that appears only on trouble cannot be checked for absence."""
    report_coverage(
        _coverage(("2000-01-01", "2020-12-31"), ("1990-01-01", "2025-12-31")),
        enforce_min_years=True,
        where="unused",
    )
    out = capsys.readouterr().out
    assert "era5: requested 2000-01-01..2020-12-31" in out
    assert "WARNING" not in out


def test_a_narrowing_is_reported_and_never_raises(capsys):
    report_coverage(
        _coverage(("1980-01-01", "2020-12-31"), ("2000-01-01", "2020-12-31")),
        enforce_min_years=True,
        where="unused",
    )
    out = capsys.readouterr().out
    assert "does not cover the full shared.historical_window" in out
    assert "widest range it holds" in out


def test_below_the_floor_raises_when_the_caller_enforces_it():
    with pytest.raises(ValueError) as excinfo:
        report_coverage(
            _coverage(("2000-01-01", "2020-12-31"), ("2010-01-01", "2015-12-31")),
            enforce_min_years=True,
            where="DO THIS INSTEAD",
        )
    message = str(excinfo.value)
    assert f"{MIN_HISTORICAL_YEARS}-year minimum" in message
    assert "weathergenr" in message
    assert message.endswith("DO THIS INSTEAD")


def test_below_the_floor_only_warns_when_the_caller_does_not(capsys):
    """The candidate-source case. Same text, different consequence -- one string
    for both so the two cannot describe the same record differently."""
    coverage = _coverage(("2000-01-01", "2020-12-31"), ("2010-01-01", "2015-12-31"))
    result = report_coverage(
        coverage, enforce_min_years=False, where="KNOW THIS INSTEAD"
    )
    out = capsys.readouterr().out
    assert result is coverage
    assert f"{MIN_HISTORICAL_YEARS}-year minimum" in out
    assert "KNOW THIS INSTEAD" in out
    assert "WARNING" in out


def test_nothing_to_report_is_not_an_error(capsys):
    assert report_coverage(None, enforce_min_years=True, where="unused") is None
    assert capsys.readouterr().out == ""


# --- the consumer side: a store on disk ---------------------------------------


def _store_nc(path, start, end):
    """A minimal store: one cell, a daily time axis, nothing else asserted."""
    import xarray as xr

    time = pd.date_range(start, end, freq="D")
    ds = xr.Dataset(
        {"precip": (("time", "latitude", "longitude"), np.zeros((time.size, 1, 1)))},
        coords={"time": time, "latitude": [0.0], "longitude": [0.0]},
    )
    ds.to_netcdf(path)
    return path


def test_store_bounds_are_read_off_the_file(tmp_path):
    from blueearth_cst.shared.climate_window import store_time_bounds

    nc = _store_nc(tmp_path / "era5.nc", "2000-01-01", "2016-12-31")
    assert store_time_bounds(nc) == (
        pd.Timestamp("2000-01-01"),
        pd.Timestamp("2016-12-31"),
    )


def test_a_long_enough_store_passes_the_consumer_floor(tmp_path):
    from blueearth_cst.shared.climate_window import require_min_years, store_time_bounds

    nc = _store_nc(tmp_path / "era5.nc", "1990-01-01", "2016-12-31")
    assert require_min_years(store_time_bounds(nc), "era5", nc, where="ctx") is None


def test_a_short_store_is_refused_at_the_consumer(tmp_path):
    """The one case the wf0 relaxation opens: a candidate store promoted to
    primary without re-extraction, which the params rerun-trigger misses when the
    reading checkout has no `.snakemake` record of the wf0 run."""
    from blueearth_cst.shared.climate_window import require_min_years, store_time_bounds

    nc = _store_nc(tmp_path / "chirps.nc", "2005-01-01", "2012-12-31")
    with pytest.raises(ValueError) as excinfo:
        require_min_years(
            store_time_bounds(nc), "chirps", nc, where="SAYS WHO READS IT"
        )
    message = str(excinfo.value)
    assert f"{MIN_HISTORICAL_YEARS}-year minimum" in message
    assert "comparison candidate" in message
    assert message.endswith("SAYS WHO READS IT")


def test_an_unreadable_store_axis_does_not_raise():
    from blueearth_cst.shared.climate_window import require_min_years

    assert require_min_years(None, "era5", "nowhere.nc", where="ctx") is None
