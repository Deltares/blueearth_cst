"""Tests for recovering the model calendar from the store (A3, 5b prerequisite).

Background: our generated catalog requests ``preprocess: harmonise_dims``, whose
time branch converts a ``CFTimeIndex`` to a ``DatetimeIndex``. After that a
``noleap`` model is indistinguishable from ``proleptic_gregorian`` — and was in
fact being written out AS ``proleptic_gregorian``, with ``cst_calendar`` empty.
See ``dev/milestones/r08/2026-07-30_wf2-5b-calendar-blocker.md``.

These cover the pure parser. The network read that feeds it is exercised by the
workflow, not here.
"""

import pytest

from blueearth_cst.projections.series_identity import (
    CALENDAR_UNKNOWN,
    SCHEMA_VERSION,
    parse_store_calendar,
)


def _zmetadata(calendar):
    """The shape gs://cmip6 actually serves: consolidated metadata, keyed paths."""
    attrs = {"units": "hours since 1850-01-16 12:00:00.000000"}
    if calendar is not None:
        attrs["calendar"] = calendar
    return {"metadata": {"time/.zattrs": attrs, ".zattrs": {}}, "zarr_consolidated_format": 1}


@pytest.mark.parametrize(
    "calendar", ["noleap", "360_day", "proleptic_gregorian", "standard", "julian"]
)
def test_every_calendar_family_the_catalog_exposes_is_read_verbatim(calendar):
    """The fixture's 3 models are examples; the catalog spans 289.

    Reading verbatim rather than mapping onto a known set matters because the
    consumer must be able to REFUSE a calendar it cannot weight, which it can only
    do if the name reaches it intact.
    """
    assert parse_store_calendar(_zmetadata(calendar)) == calendar


def test_missing_calendar_yields_the_unknown_sentinel_not_an_empty_string():
    """An empty string is what the broken pre-3 schema wrote.

    "" is indistinguishable from "nobody looked". The sentinel is distinguishable,
    and that is the whole point: a step weighting by month length must refuse it.
    """
    assert parse_store_calendar(_zmetadata(None)) == CALENDAR_UNKNOWN
    assert parse_store_calendar({}) == CALENDAR_UNKNOWN
    assert parse_store_calendar({"metadata": {}}) == CALENDAR_UNKNOWN


def test_blank_calendar_is_also_unknown():
    assert parse_store_calendar(_zmetadata("   ")) == CALENDAR_UNKNOWN


def test_accepts_metadata_passed_either_wrapped_or_bare():
    """`.zmetadata` nests under "metadata"; a caller holding that dict is valid too."""
    wrapped = _zmetadata("noleap")
    assert parse_store_calendar(wrapped) == "noleap"
    assert parse_store_calendar(wrapped["metadata"]) == "noleap"


def test_schema_version_was_bumped_to_invalidate_the_false_calendars():
    """The version bump IS the invalidation lever.

    The calendar cannot be a digest component -- reading it needs the store, and
    DAG build is deliberately network-free -- so `cache_hit`'s schema check is what
    forces slices written with the false calendar to be re-fetched. If this ever
    reverts to "2", every one of those slices silently becomes valid again.
    """
    assert SCHEMA_VERSION == "4"
