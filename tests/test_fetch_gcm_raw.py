# -*- coding: utf-8 -*-
"""Unit tests for ``blueearth_cst/projections/fetch_gcm_raw.py``.

The module had NO coverage until 2026-08-12
(`dev/reviews/2026-08-11_test-suite-bloat-assessment.md` §4) and could not have
any: 336 lines with **zero functions**, the whole body inside
``if "snakemake" in globals():``. Its decisions were checked by running the
pipeline or not at all. The same commit lifts the pure ones out, by the argument
`[R7-22]` already made for ``downscale_climate_forcing.py``.

**What this file does NOT cover, deliberately.** The remote read itself — the
``DataCatalog``, ``get_rasterdataset``, ``.load()``, the store-calendar fetch —
is still ~150 inline lines and stays that way. It is exercised by
``--run-integration`` and by real runs, and faking a hydromt catalog well enough
to be evidence costs more than it proves. So `fetch_gcm_raw.py` moves from *no
coverage* to *its decision logic is pinned*, which is not the same as covered.

What IS here is every branch that decides something, and two of them are the
reason this mattered: ``check_time_axis``'s empty-window guard is described in
the source as **"invisible to the fixture gate, whose three models all cover
their windows"**, and ``raw_slice_attrs`` is the seam the reduce stage reads
back — the thing that lets it make zero remote calls.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from blueearth_cst.projections import series_identity
from blueearth_cst.projections.fetch_gcm_raw import (
    bbox_index_slice,
    calendar_pin,
    calendar_store_uri,
    check_time_axis,
    clip_to_bbox,
    hns_switch_row,
    is_irregular_grid_error,
    raw_slice_attrs,
    resolve_entry_name,
    stale_units,
)

# ---------------------------------------------------------------------------
# hns_switch_row — a slow run must explain itself
# ---------------------------------------------------------------------------


def test_the_expected_switch_value_logs_at_INFO_with_no_warning_tail():
    row, level = hns_switch_row("false")
    assert level == "INFO"
    assert "slower" not in row
    assert repr("false") in row


@pytest.mark.parametrize("value", ["true", "True", "1", ""])
def test_any_other_value_warns_and_says_what_it_will_cost(value):
    """The defect this row exists for.

    An inherited ``"true"`` turned a 58 s job into a 14-minute one with nothing
    in the log to say why. The row is reported rather than enforced — the
    module's ``setdefault`` fixes only the UNSET case, deliberately, so a
    deliberate export survives — which is exactly why the log has to carry it.
    """
    row, level = hns_switch_row(value)
    assert level == "WARNING"
    assert "~14x slower remote opens" in row
    assert repr(value) in row


# ---------------------------------------------------------------------------
# resolve_entry_name — the catalog's own grammar, not string surgery
# ---------------------------------------------------------------------------


def test_a_placeholder_entry_is_formatted_with_the_member():
    assert resolve_entry_name("cmip6_{member}_ssp245", "r1i1p1f1") == (
        "cmip6_r1i1p1f1_ssp245"
    )


def test_an_entry_without_a_placeholder_takes_the_member_as_a_SUFFIX():
    assert resolve_entry_name("cmip6_ssp245", "r1i1p1f1") == "cmip6_ssp245_r1i1p1f1"


def test_every_placeholder_occurrence_is_filled():
    """`str.format` replaces all of them; asserted so the branch cannot become
    a one-shot `replace` that leaves a second `{member}` literal in the name."""
    assert resolve_entry_name("{member}/x_{member}", "r2") == "r2/x_r2"


# ---------------------------------------------------------------------------
# stale_units — S8-08(a), repairing a slice cached before the units fix
# ---------------------------------------------------------------------------


def _ds(**units):
    """A dataset whose variables carry the given ``units`` (None = no attr)."""
    data = {}
    for name, unit in units.items():
        da = xr.DataArray([1.0, 2.0], dims="time")
        if unit is not None:
            da.attrs["units"] = unit
        data[name] = da
    return xr.Dataset(data)


def test_a_slice_already_carrying_the_right_units_is_not_stale():
    """The fast path: a repair is paid once per stale file, never repeatedly."""
    assert (
        stale_units(_ds(pr="mm/day", tas="degC"), {"pr": "mm/day", "tas": "degC"}) == {}
    )


def test_only_the_disagreeing_variable_is_reported():
    stale = stale_units(
        _ds(pr="kg m-2 s-1", tas="degC"), {"pr": "mm/day", "tas": "degC"}
    )
    assert stale == {"pr": "mm/day"}


def test_a_variable_the_slice_does_not_carry_is_absent_not_stale():
    """A configured variable missing from the slice is a different problem.

    Reporting it here would make the repair write an attribute onto a variable
    that does not exist, which raises rather than repairs.
    """
    assert stale_units(_ds(pr="mm/day"), {"pr": "mm/day", "tas": "degC"}) == {}


def test_a_variable_with_no_units_attribute_at_all_counts_as_stale():
    """`.attrs.get("units")` is None, which never equals the configured value."""
    assert stale_units(_ds(pr=None), {"pr": "mm/day"}) == {"pr": "mm/day"}


# ---------------------------------------------------------------------------
# check_time_axis — the two failures every downstream check would pass
# ---------------------------------------------------------------------------


def _index(*years):
    return pd.DatetimeIndex([f"{y}-01-01" for y in years])


def test_a_clean_axis_raises_nothing():
    idx = _index(2000, 2001, 2002)
    assert check_time_axis("src", idx, idx, ("2000-01-01", "2002-12-31")) is None


def test_a_duplicated_axis_names_the_count_and_the_fix():
    """D8: the catalog URI globs `{grid_label}/{version}` and ~6% of pinned
    stores match more than one. Two concatenated stores halve the effective
    record while looking fine."""
    idx = _index(2000, 2001, 2001, 2002, 2002)
    with pytest.raises(RuntimeError) as excinfo:
        check_time_axis("cmip6_x", idx, idx, ("2000-01-01", "2002-12-31"))

    message = str(excinfo.value)
    assert "cmip6_x" in message
    assert "2 duplicate step(s)" in message
    assert "Pin the version" in message


def test_an_empty_window_names_what_the_driver_actually_returned():
    """The failure the fixture gate structurally cannot see.

    Without this the attrs block dies on `index[0]` with a bare IndexError
    naming neither the source nor the window. Reporting the DRIVER's coverage is
    what separates 'the store is short' from 'the window is wrong' — the driver
    index is the axis before `.sel()` narrowed it to nothing.
    """
    with pytest.raises(RuntimeError) as excinfo:
        check_time_axis(
            "cmip6_y",
            _index(),
            _index(1850, 2014),
            ("2050-01-01", "2080-12-31"),
        )

    message = str(excinfo.value)
    assert "cmip6_y" in message
    assert "2050-01-01..2080-12-31" in message
    assert "1850" in message and "2014" in message
    assert "cannot produce a raw slice" in message


@pytest.mark.parametrize("driver", [None, "empty"])
def test_an_empty_window_with_nothing_from_the_driver_says_so(driver):
    """The branch a naive extraction drops: `driver_index` may be None or empty.

    `f"{driver_index[0]}"` on either would raise inside the error handler, which
    would replace a named failure with an anonymous one.
    """
    with pytest.raises(RuntimeError, match="no steps at all"):
        check_time_axis(
            "cmip6_z",
            _index(),
            None if driver is None else _index(),
            ("2050-01-01", "2080-12-31"),
        )


def test_duplicates_are_reported_BEFORE_emptiness():
    """Order matters: an axis cannot be both, but the guards are independent and
    a future edit could make the empty test shadow the duplicate one."""
    idx = _index(2000, 2000)
    with pytest.raises(RuntimeError, match="duplicate"):
        check_time_axis("src", idx, idx, ("2000-01-01", "2000-12-31"))


def test_an_absent_time_axis_is_not_an_error():
    """`data.indexes.get("time")` returns None for a dataset with no time dim,
    and that is a different question than an empty or ambiguous one."""
    assert check_time_axis("src", None, None, ("a", "b")) is None


# ---------------------------------------------------------------------------
# calendar_pin — ask a store that provably exists
# ---------------------------------------------------------------------------


def test_a_certified_variable_is_preferred_over_a_best_effort_one():
    """The crawl proved pr/tas present; any other name is best-effort (A3), so
    its store may not exist and the calendar read would fail on it."""
    assert calendar_pin({"hurs": ["v1"], "tas": ["v1"], "pr": ["v1"]}) == "tas"


def test_tas_wins_over_pr_when_both_are_pinned():
    assert calendar_pin({"pr": ["v1"], "tas": ["v1"]}) == "tas"


def test_pr_is_taken_when_tas_is_not_pinned():
    assert calendar_pin({"pr": ["v1"], "hurs": ["v1"]}) == "pr"


def test_a_member_pinning_only_a_best_effort_variable_still_yields_one():
    assert calendar_pin({"hurs": ["v1"]}) == "hurs"


def test_a_member_pinning_nothing_yields_the_empty_string():
    """Which is what makes the caller record CALENDAR_UNKNOWN rather than guess."""
    assert calendar_pin({}) == ""


# ---------------------------------------------------------------------------
# calendar_store_uri — address one store, list no bucket
# ---------------------------------------------------------------------------

SUFFIX = series_identity.STORE_GLOB_SUFFIX


def test_no_pinned_variable_means_no_store_to_ask():
    assert calendar_store_uri("gs://cmip6/{member}/{variable}", "r1", "", {}) == ""


def test_an_already_pinned_template_is_just_formatted():
    assert (
        calendar_store_uri(
            "gs://cmip6/x/{member}/{variable}/gn/v1", "r1i1p1f1", "tas", {"tas": ["v1"]}
        )
        == "gs://cmip6/x/r1i1p1f1/tas/gn/v1"
    )


def test_a_globbed_template_is_resolved_to_the_pinned_location():
    """Spending the D12 pin instead of listing the bucket: open 49.9 s pinned vs
    60.0 s globbed. What it removes is hydromt's resolver overhead on a wildcard
    URI, not a slow network listing — and it makes the store deterministic."""
    uri = calendar_store_uri(
        f"gs://cmip6/x/{{member}}/{{variable}}{SUFFIX}",
        "r1i1p1f1",
        "tas",
        {"tas": ["gn/v20190101", "gn/v20200101"]},
    )
    assert uri == "gs://cmip6/x/r1i1p1f1/tas/gn/v20200101"


def test_the_LAST_match_wins_when_a_variable_pins_several():
    uri = calendar_store_uri(
        f"gs://s/{{variable}}{SUFFIX}", "r1", "pr", {"pr": ["a", "b", "c"]}
    )
    assert uri.endswith("/c")


@pytest.mark.parametrize("pins", [{"tas": []}, {"pr": ["v1"]}])
def test_a_globbed_template_the_pins_cannot_resolve_yields_nothing(pins):
    """Empty match list, or a pin for a different variable. Either way there is
    no single location to address, so the caller records CALENDAR_UNKNOWN
    instead of reading a wildcard URI as if it were a store."""
    assert calendar_store_uri(f"gs://s/{{variable}}{SUFFIX}", "r1", "tas", pins) == ""


# ---------------------------------------------------------------------------
# raw_slice_attrs — the seam the reduce stage reads back
# ---------------------------------------------------------------------------


@pytest.fixture
def components():
    return {
        "catalog_entry": "cmip6_ssp245",
        "pins": {"r1i1p1f1": {"tas": ["gn/v1"], "pr": ["gn/v1"]}},
        "entry_identity": {"r1i1p1f1": {"metadata": {"crs": 4326}}},
    }


def _attrs(components, **overrides):
    kwargs = dict(
        member="r1i1p1f1",
        expected_raw_digest="abcdef0123456789",
        acquisition_window=("1950-01-01", "2014-12-31"),
        first="1950-01-01 00:00:00",
        last="2014-12-01 00:00:00",
        store_calendar="noleap",
        bbox=[9.0, 0.25, 9.5, 0.75],
        region_fp="regionfp",
        buffer=2,
    )
    kwargs.update(overrides)
    return raw_slice_attrs(components, **kwargs)


def test_a_raw_slice_never_claims_a_REDUCED_identity(components):
    """The deliberate absence, and the load-bearing one.

    `cst_series_digest` and `cst_reducer_module_hash` belong to a reduced
    series. Stamping either here would let the reduce stage's
    `assert_raw_identity` accept a pre-reduction file as post-reduction — and
    `cst_raw_digest` EXCLUDING the reducer hash is precisely what makes a
    formula edit re-read local disk instead of the network.
    """
    attrs = _attrs(components)
    assert "cst_series_digest" not in attrs
    assert "cst_reducer_module_hash" not in attrs
    assert attrs["cst_raw_digest"] == "abcdef0123456789"


def test_the_schema_version_is_the_one_the_reduce_stage_checks(components):
    assert _attrs(components)["cst_schema_version"] == series_identity.SCHEMA_VERSION


def test_the_calendar_comes_from_the_argument_not_from_the_time_axis(components):
    """Our catalog requests `preprocess: harmonise_dims`, whose time branch
    converts a CFTimeIndex away — after which a noleap model is
    indistinguishable from proleptic_gregorian. The store is the only place that
    still knows, so this field must be whatever the caller read from it."""
    assert _attrs(components, store_calendar="360_day")["cst_calendar"] == "360_day"
    assert (
        _attrs(components, store_calendar=series_identity.CALENDAR_UNKNOWN)[
            "cst_calendar"
        ]
        == series_identity.CALENDAR_UNKNOWN
    )


def test_the_window_is_recorded_as_a_two_part_string(components):
    assert _attrs(components)["cst_acquisition_window"] == "1950-01-01 / 2014-12-31"


def test_bounds_are_written_at_nine_significant_digits(components):
    """Enough that a re-derived bbox compares equal, few enough that float noise
    in the last bits does not invalidate a cache."""
    attrs = _attrs(components, bbox=[9.666666666666666, 0.4476, -1.5, 2.0])
    assert attrs["cst_region_bounds"] == "9.66666667, 0.4476, -1.5, 2"


def test_the_per_variable_provenance_is_serialized_deterministically(components):
    """`sort_keys` — the attrs ride in a netCDF the digest chain reads back, so
    dict ordering must not make two identical slices compare different."""
    attrs = _attrs(components)
    assert attrs["cst_source_paths"] == json.dumps(components["pins"], sort_keys=True)
    assert json.loads(attrs["cst_source_paths"])["r1i1p1f1"]["tas"] == ["gn/v1"]


def test_the_crs_is_read_out_of_the_members_own_entry_identity(components):
    assert _attrs(components)["cst_crs"] == "4326"


@pytest.mark.parametrize(
    "broken",
    [
        {},
        {"entry_identity": {}},
        {"entry_identity": {"r1i1p1f1": {}}},
        {"entry_identity": {"r1i1p1f1": {"metadata": None}}},
        {"entry_identity": None},
    ],
)
def test_a_missing_or_null_entry_identity_gives_an_empty_crs_not_a_crash(broken):
    """Every level of that lookup is optional in practice, and the `or {}` on
    `metadata` is there because the key can be present and null. A raw slice
    with no recorded CRS is a lesser problem than a fetch job that dies while
    stamping attributes on data it already paid to download."""
    attrs = _attrs({"catalog_entry": "e", "pins": {}, **broken})
    assert attrs["cst_crs"] == ""


def test_the_member_and_buffer_ride_along_unmodified(components):
    attrs = _attrs(components, buffer=0)
    assert attrs["cst_members"] == "r1i1p1f1"
    assert attrs["cst_buffer_degrees"] == 0


def test_every_value_survives_a_netcdf_round_trip(components):
    """The attrs are written to disk and read back by `assert_raw_identity`, so
    a type netCDF cannot store would fail at write time in a real run — after
    the download. Checked here instead.
    """
    ds = xr.Dataset({"pr": xr.DataArray(np.zeros(2), dims="time")})
    ds.attrs.update(_attrs(components))
    round_tripped = xr.Dataset.from_dict(ds.to_dict())
    assert round_tripped.attrs == ds.attrs


# ---------------------------------------------------------------------------
# is_irregular_grid_error — the branch's trigger
# ---------------------------------------------------------------------------


def test_hydromts_own_refusal_is_recognised():
    """The exact message `hydromt/gis/raster.py:443` raises."""
    assert is_irregular_grid_error(
        ValueError("The 'raster' accessor only applies to regular grids")
    )


def test_any_other_ValueError_is_not_the_grid_case():
    """Keyed on the MESSAGE: ValueError is far too common a type to branch on.

    A read that failed for an unrelated reason must propagate, not be retried
    through a path that assumes the grid was the problem.
    """
    assert not is_irregular_grid_error(ValueError("No valid spatial coords found"))
    assert not is_irregular_grid_error(KeyError("cmip6_X_historical_r1i1p1f1"))


# ---------------------------------------------------------------------------
# bbox_index_slice — the same cells hydromt takes, without needing regularity
#
# The parity tests below are the load-bearing ones: they pin the cells-in-index-
# space reading of `buffer` against hydromt ITSELF rather than against an
# argument, so a regular and an irregular slice of one region stay comparable.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("buffer", [0, 1, 2])
@pytest.mark.parametrize(
    "bbox",
    [
        (10.0, 45.0, 12.0, 47.0),
        (-3.25, -1.5, 2.75, 4.5),  # straddles both origins
        (0.1, 0.1, 0.2, 0.2),  # smaller than one cell
    ],
)
def test_a_uniform_grid_selects_exactly_the_cells_hydromt_would(bbox, buffer):
    """The reading of `buffer` is hydromt's, pinned against hydromt.

    `buffer` is CELLS, not degrees, however the config key is spelled. Asserting
    that in prose would be an argument; asserting it against `.raster.clip_bbox`
    on a grid hydromt accepts is evidence.
    """
    # Imported HERE, not at module scope: it registers the `.raster` accessor
    # this test compares against, and costs ~4 s that no other test in this file
    # should pay (the module's own note on why hydromt is a late import).
    import hydromt  # noqa: F401

    lat = np.arange(60.0, -60.1, -2.5)  # N->S, as `harmonise_dims` leaves it
    lon = np.arange(-180.0, 180.0, 2.5)
    ds = xr.Dataset(
        {"temp": (("lat", "lon"), np.zeros((lat.size, lon.size), dtype="float32"))},
        coords={"lat": lat, "lon": lon},
    )
    ds.raster.set_crs(4326)
    west, south, east, north = bbox

    expected = ds.raster.clip_bbox(bbox, buffer=buffer)
    got = ds.isel(
        {
            "lat": bbox_index_slice(lat, south, north, buffer, "latitude"),
            "lon": bbox_index_slice(lon, west, east, buffer, "longitude"),
        }
    )

    assert np.array_equal(got["lat"].values, expected["lat"].values)
    assert np.array_equal(got["lon"].values, expected["lon"].values)


def test_a_descending_axis_is_not_read_backwards():
    """The trap a label-based `sel(lat=slice(south, north))` falls into.

    `harmonise_dims` orients latitude N->S, so a label slice in south-to-north
    order returns NOTHING — and an empty spatial selection reduces to NaN rather
    than raising, several rules downstream.
    """
    lat = np.arange(60.0, -60.1, -2.5)
    ascending = bbox_index_slice(lat[::-1], 44.0, 47.0, 0, "latitude")
    descending = bbox_index_slice(lat, 44.0, 47.0, 0, "latitude")
    assert lat[descending].size > 0
    assert sorted(lat[descending]) == sorted(lat[::-1][ascending])


def test_a_gaussian_axis_is_clipped_where_the_region_actually_is():
    """The whole point: no regularity is required, and the cells are the right ones.

    A T42-like Gaussian latitude axis, whose spacing varies far above hydromt's
    5e-4 tolerance.
    """
    lat = np.array(
        [12.7788, 10.0022, 7.2255, 4.4489, 1.6722, -1.1045, -3.8811, -6.6578]
    )
    picked = lat[bbox_index_slice(lat, 1.0, 8.0, 0, "latitude")]
    assert set(np.round(picked, 4)) == {7.2255, 4.4489, 1.6722}


def test_a_length_one_axis_keeps_its_single_cell():
    """The ordinary small-basin path at Amon resolution, not an edge case.

    `midpoint_edges` refuses a length-1 axis (it has no midpoints), so this must
    be answered before the edge ladder is built.
    """
    assert bbox_index_slice(np.array([5.0]), 4.0, 6.0, 1, "latitude") == slice(0, 1)


def test_a_region_leaving_the_grids_span_is_refused_not_clamped():
    """The one case hydromt handles differently, and loudly rather than silently.

    On a grid spanning 360 degrees hydromt shifts longitudes across the 180
    meridian first; that helper reads `.raster.bounds`, so it cannot run on an
    irregular grid. Clamping instead would drop the wrapped part of the region.
    """
    lon = np.arange(-177.1875, 180.1, 2.8125)
    with pytest.raises(RuntimeError, match="180 meridian"):
        bbox_index_slice(lon, -179.9, -178.0, 0, "longitude", source="cmip6_X")


def test_a_non_monotonic_axis_is_refused_by_the_shared_geometry_check():
    """Reuses `grid_weights.check_axis`, so one definition of 'orderable' holds."""
    with pytest.raises(ValueError, match="not strictly monotonic"):
        bbox_index_slice(np.array([1.0, 3.0, 2.0]), 1.0, 3.0, 0, "latitude")


# ---------------------------------------------------------------------------
# clip_to_bbox — the branch's clip, on both dims at once
# ---------------------------------------------------------------------------


def _gaussian_ds():
    lat = np.array([12.7788, 10.0022, 7.2255, 4.4489, 1.6722, -1.1045, -3.8811])
    lon = np.arange(0.0, 20.0, 2.8125) - 10.0
    values = np.arange(lat.size * lon.size, dtype="float32").reshape(lat.size, lon.size)
    return xr.Dataset(
        {"temp": (("lat", "lon"), values)}, coords={"lat": lat, "lon": lon}
    )


def test_both_dimensions_are_clipped_and_the_values_ride_along():
    ds = _gaussian_ds()
    clipped = clip_to_bbox(ds, (-4.0, 1.0, 4.0, 8.0), 0, "lat", "lon")
    assert clipped.sizes["lat"] < ds.sizes["lat"]
    assert clipped.sizes["lon"] < ds.sizes["lon"]
    # the same cells, not merely the same shape
    assert np.array_equal(
        clipped["temp"].values,
        ds.sel(lat=clipped["lat"], lon=clipped["lon"])["temp"].values,
    )


def test_the_buffer_widens_the_selection_by_whole_cells():
    ds = _gaussian_ds()
    tight = clip_to_bbox(ds, (-4.0, 1.0, 4.0, 8.0), 0, "lat", "lon")
    buffered = clip_to_bbox(ds, (-4.0, 1.0, 4.0, 8.0), 1, "lat", "lon")
    assert buffered.sizes["lat"] == tight.sizes["lat"] + 2
    assert buffered.sizes["lon"] == tight.sizes["lon"] + 2


def test_an_empty_selection_is_refused_rather_than_returned():
    """A slice with an empty spatial dim reduces to NaN instead of failing.

    Unreachable at WF2's `buffer=1`, which always widens to at least two cells —
    which is why it is a guard rather than a fallback: it reports instead of
    inventing a selection hydromt has no equivalent for.
    """
    ds = _gaussian_ds()
    # A degenerate bbox sitting exactly on a cell edge, with no buffer to widen it.
    edge = float((ds["lat"].values[2] + ds["lat"].values[3]) / 2)
    with pytest.raises(RuntimeError, match="selected no cells"):
        clip_to_bbox(ds, (0.0, edge, 1.0, edge), 0, "lat", "lon", source="cmip6_X")
