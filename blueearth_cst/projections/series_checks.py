"""Validated monthly-series loading and calendar agreement for WF2 consumers."""

import os

import xarray as xr

from blueearth_cst.projections import series_identity
from blueearth_cst.projections.calendar_weights import (
    CalendarError,
    assert_weightable,
    same_month_lengths,
)


def load_checked_series(path, components, region_fp, source, series_cache=None):
    """Load and close a series once per identity-keyed, job-local cache.

    Return the loaded dataset and its expected digest. A shared path with a
    different expected identity is checked again, rather than reused silently.
    """
    series_cache = {} if series_cache is None else series_cache
    digest = series_identity.series_digest(dict(components), region_fp)
    key = (os.path.abspath(str(path)), digest)
    if key not in series_cache:
        with xr.open_dataset(path) as dataset:
            series_identity.assert_series_identity(
                path, digest, source, observed_attrs=dataset.attrs
            )
            series_cache[key] = dataset.load()
    return series_cache[key], digest


def assert_calendar_agreement(historical, scenario, source):
    """Return the historical calendar after checking comparable month lengths."""
    calendar = str(historical.attrs.get("cst_calendar", "") or "")
    clim_calendar = str(scenario.attrs.get("cst_calendar", "") or "")
    earliest = min(
        (t.year, t.month)
        for t in (historical.indexes["time"][0], scenario.indexes["time"][0])
    )
    if not same_month_lengths(calendar, clim_calendar, earliest):
        raise CalendarError(
            f"{source}: reference and scenario series carry "
            f"different calendars ({calendar!r} vs {clim_calendar!r}). Their annual "
            "aggregates are not comparable."
        )
    assert_weightable(calendar, source=source)
    assert_weightable(clim_calendar, source=source)
    return calendar
