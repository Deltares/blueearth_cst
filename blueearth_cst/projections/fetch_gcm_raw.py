"""Stage A1 — acquire one raw slice of a remote GCM store (design revision 6).

The **only** module in WF2 that opens the remote store. Stage A used to fetch and
reduce in one job, so anything invalidating the *reduction* re-triggered the
*download*: an edit to a formula cost nine remote reads. This split makes a
reduction edit re-read local disk instead.

Why the split is worth its second cache, measured on 2026-07-30
(`dev/milestones/r08/2026-07-30_wf2-fetch-reduce-benchmark.md`):

* opening one source (catalog URI glob resolution + store metadata): **1142 s**
* transferring its data (``.load()``): **19 s**
* the reduction arithmetic: **0.2 s**
* the raw slice on disk: **0.07 MB**

So the dominant cost is the *open*, which means the reduce stage must make **zero**
remote calls — not merely avoid the transfer. It therefore reads this file and
checks the digest recorded on it, which already encodes the store-index pins,
rather than reopening the store to re-verify them.
`series_identity.assert_raw_identity` is the check that makes that safe.

Contract with the reduce stage:

* this job writes ``cst_raw_digest`` = :func:`series_identity.raw_digest`, which
  excludes the reducer hash — that exclusion is what makes a formula edit free;
* the write is atomic, so a killed job cannot leave a valid-looking short file;
* the reduce job refuses a slice whose digest, schema, variables, time axis or
  recorded window disagree with what it expects.

Invoked from ``analyze_projections.smk`` via ``script:``; reads
``snakemake.input/output/params``, never ``sys.argv``.
"""
# NOTE: no `from __future__ import annotations` here. Snakemake's `script:`
# directive prepends its own preamble to a copy of this file, so a __future__
# import lands mid-file and raises SyntaxError at job start. A --dry-run cannot
# catch it (it never executes a script body) -- the other `script:` modules in this
# repo omit it for the same reason.

import contextlib
import json
import os

# MUST precede any import that transitively imports gcsfs (hydromt does). gcsfs
# >= 2026.4 turns on an experimental Extended filesystem BY DEFAULT and reads this
# switch at import time. That filesystem probes the bucket storage layout through an
# authenticated control-plane RPC on every operation; public CMIP6 reads have no
# credentials, so the probe fails, is deliberately never cached
# (`extended_gcsfs.py:155` "Dont cache UNKNOWN type"), and repeats per call --
# including inside the `/*/*` glob resolution. Measured on the same source, one
# process each: 57.7 s with the switch off versus >836 s (killed) with it on, a
# lower bound of 14x, with 266 fallback warnings and climbing.
# get_stats_climate_proj.py sets the same variable; as WF2's only remote-opening
# module this one must too, or the whole fetch stage sits on the slow side of a
# one-line switch.
os.environ.setdefault("GCSFS_EXPERIMENTAL_ZB_HNS_SUPPORT", "false")

import numpy as np

from blueearth_cst.projections import grid_weights, series_identity
from blueearth_cst.shared.snake_utils import log_row, tee_to_log

# ---------------------------------------------------------------------------
# The decisions, lifted out of the `script:` body so they can be tested.
#
# Everything below is PURE -- no network, no filesystem, no hydromt. The
# remaining inline body is the remote read itself, which is exercised only by
# `--run-integration`. Extracted 2026-08-12 by the same argument `[R7-22]`
# made for `downscale_climate_forcing.py`: a decision that lives only inside
# `if "snakemake" in globals():` is invisible to every unit test, so it is
# checked by running the pipeline or not at all.
# ---------------------------------------------------------------------------


def hns_switch_row(value):
    """``(message, level)`` reporting the gcsfs extended-filesystem switch.

    WARNING when the effective value is not the one this module needs, so a slow
    run explains itself: an inherited ``"true"`` turns a 58 s job into a
    14-minute one, and before this there was nothing in the log to say why.
    Reported rather than enforced -- see the `setdefault` note at the top.
    """
    return (
        f"gcsfs extended-filesystem switch = {value!r}"
        + ("" if value == "false" else "  <-- expect ~14x slower remote opens"),
        "INFO" if value == "false" else "WARNING",
    )


#: hydromt's own phrase for the refusal this module routes around
#: (``hydromt/gis/raster.py:441`` -- ``np.allclose(dys, dys[0], atol=5e-4)``).
#: Matched on the MESSAGE rather than on ``ValueError``, which is far too common
#: a type to key on; ``dev/scripts/stage_cmip6.py`` classifies the same phrase.
IRREGULAR_GRID_PHRASE = "only applies to regular grids"


def is_irregular_grid_error(error):
    """True when hydromt refused a read because the grid is not evenly spaced.

    27 of 67 CMIP6 models publish ``Amon`` on a GAUSSIAN grid, whose latitudes
    are Legendre roots and vary by ~1% -- against hydromt's 5e-4 tolerance, so
    no Gaussian grid can ever pass. The data is well formed; the accessor's
    precondition is simply tighter than the grid type allows
    (`dev/reference/workflows/wf2-cmip6-store-readability.md`
    section 1, which carries the measured table).
    """
    return IRREGULAR_GRID_PHRASE in str(error)


#: The phrase a multi-version refusal carries, so `dev/scripts/stage_cmip6.py`
#: can bucket it without matching an exception type. Same contract as
#: :data:`IRREGULAR_GRID_PHRASE` and for the same reason: the type here is
#: whatever xarray raised while combining, which is far too general to key on.
AMBIGUOUS_VERSION_PHRASE = "more than one published version"


def ambiguous_pins(uri, pins_for_member):
    """The per-variable version lists when they cannot name ONE store, else ``{}``.

    DELEGATES to :func:`series_identity.pinned_uri` rather than restating its
    rule, so the two cannot drift apart -- and they would have: since the owner
    ruling of 2026-08-19 the newest version wins, which resolves 182 of the 221
    combinations this used to report. A second copy of the logic here would
    still be naming versions for sources that now pin cleanly.

    Two situations reach ``pinned_uri``'s ``None`` without being ambiguity, and
    are screened out first:

    * the catalog URI does not end in the glob suffix -- it already names one
      store, so nothing is being chosen;
    * the index recorded no pins for this member -- the glob is all there is,
      and we cannot report versions we never crawled. The bucket may still hold
      several and that read can still fail; the honest boundary is that we do
      not claim to know.

    What remains is the 39 combinations where ``pr``'s newest and ``tas``'s
    newest are different locations and one URI cannot express both, plus any
    member pinning more than one grid label. Returns the pins verbatim so the
    caller can name every version, which is the whole point: the operator's next
    action is to pick one, and they cannot without seeing the list.
    """
    if not str(uri).endswith(series_identity.STORE_GLOB_SUFFIX):
        return {}
    if not pins_for_member:
        return {}
    if series_identity.pinned_uri(uri, pins_for_member) is not None:
        return {}
    return {name: list(paths) for name, paths in pins_for_member.items()}


def ambiguous_versions_phrase(ambiguous):
    """``pr: gn/v1, gn/v2; tas: gn/v1, gn/v2`` -- every version, per variable.

    Sorted by variable so two runs of the same source read identically; the
    version order inside a variable is the index's own, which is the order the
    crawl found them and therefore the order the glob would match.
    """
    return "; ".join(
        f"{name}: {', '.join(paths) or 'none'}"
        for name, paths in sorted(ambiguous.items())
    )


@contextlib.contextmanager
def explaining_ambiguous_versions(entry, ambiguous):
    """Re-raise a failed read with the versions the glob opened named in it.

    The store index records more than one published version for ~9% of member
    combinations (221 of 2426, across 46 of 289 entries). Those fall to the
    globbed URI, which matches every version, so ``open_mfdataset`` is handed
    two stores per variable -- and what comes back is ``MergeError: conflicting
    values for variable 'pr' on objects to be combined``, which names neither
    the source, nor the versions, nor anything the operator can act on.

    **Wraps the read rather than pre-empting it, and that is deliberate.**
    Refusing up front would be simpler, but two versions that differ only in
    metadata merge cleanly and produce a correct slice today -- so a pre-emptive
    refusal would break sources that currently work, in the name of a better
    error message. This only changes what a FAILURE says.

    The message states the ambiguity and the original error side by side
    without claiming one caused the other. A source can be both ambiguous and,
    say, published on a grid the irregular-grid branch cannot read either, and
    asserting a cause we have not established would send the operator after the
    wrong thing.

    A no-op when ``ambiguous`` is empty, so the caller needs no conditional.
    """
    try:
        yield
    except Exception as exc:
        if not ambiguous:
            raise
        raise RuntimeError(
            f"{entry}: the store index records {AMBIGUOUS_VERSION_PHRASE}, so the "
            f"catalog glob opened all of them ({ambiguous_versions_phrase(ambiguous)}). "
            f"The read then failed with {type(exc).__name__}: {exc}. Pin one version "
            "to read this source deterministically -- see "
            "dev/reference/workflows/wf2-cmip6-store-readability.md section 2."
        ) from exc


#: The name our own preprocessor is registered under. NOT `harmonise_dims`:
#: replacing hydromt's entry would change every catalog in the repo that asks
#: for it by name (WF3's downscaling, `prepare_climate_data_catalog`), which is
#: far more than this defect is about.
WIDE_TIME_PREPROCESS = "harmonise_dims_wide_time"


def seconds_resolution_index(index):
    """``index`` at second resolution, or ``None`` when that would truncate.

    A datetime64 index is returned coarsened to seconds only when the round trip
    back to its own unit is exact. CMIP6 ``Amon`` midpoints land on whole
    seconds -- the stored offsets are whole or half days -- so in practice this
    always succeeds on the data this pipeline reads; the check is here because
    the function cannot see the table it was handed, and silently dropping
    sub-second precision to make a merge work would be the wrong trade.

    Returns ``None`` for a cftime (object) index too: that one is widened by
    ``to_datetimeindex(time_unit="s")`` instead, which is a conversion rather
    than a cast.
    """
    dtype = getattr(index, "dtype", None)
    if dtype is None or not str(dtype).startswith("datetime64"):
        return None
    coarse = index.astype("datetime64[s]")
    if not (coarse.astype(dtype) == index).all():
        return None
    return coarse


def harmonise_dims_wide_time(ds):
    """``harmonise_dims`` with every time axis met at SECOND resolution.

    Two ways a CMIP6 read overflows `datetime64[ns]`, whose ceiling is
    ``2262-04-11T23:47:16``. Both are reached by stores published as EXTENSIONS
    TO 2300 under an `sspNNN` experiment id, and both arrive as

        Cannot cast ... to unit='ns' without overflow
        OutOfBoundsDatetime: Out of bounds nanosecond timestamp: 2262-04-16

    **One axis, decoded too narrow.** A non-standard calendar (`noleap`,
    `360_day`) decodes to a `CFTimeIndex`, and hydromt's `harmonise_dims` ends
    by converting it with `CFTimeIndex.to_datetimeindex()` and no `time_unit`,
    which defaults to nanoseconds. Converting it ourselves at second resolution
    first leaves hydromt's own `dtype == "O"` test False, so its conversion
    becomes a no-op and none of its lon/lat harmonisation is reimplemented here
    -- that function is called verbatim for everything it does well.
    `CCCma/CanESM5 ssp585` is this case.

    **Two axes, meeting at different widths.** This preprocessor runs per FILE
    inside `open_mfdataset`, and the catalog reads `pr` and `tas` from separate
    stores. A model may extend one variable to 2300 and stop the other at 2100
    -- `IPSL/IPSL-CM6A-LR ssp585` publishes exactly that under ONE version
    (`gr/v20190903`): `pr` runs to 2300 and falls back to cftime, `tas` runs to
    2100 and decodes to `datetime64[ns]`. Widening only the cftime side left the
    two at `[s]` and `[ns]`, and merging them aligns their indexes through
    `pandas.Index.union`, which adopts the FINER unit and overflows on the
    longer axis. The same union is why two published VERSIONS of different
    spans fail (`CSIRO-ARCCSS/ACCESS-CM2 ssp585`).

    So every axis is met at seconds, not only the cftime ones: a merge can only
    be safe if both sides are already wide. :func:`seconds_resolution_index`
    refuses to coarsen an axis that would lose precision, so this cannot quietly
    truncate a sub-second time coordinate to make a union succeed.

    The axis is cast back to nanoseconds in :func:`fetch_raw_slice`, AFTER the
    acquisition window has dropped the extension years -- so a slice that
    already staged is written exactly as before, and this changes only which
    sources can be read at all.

    xarray already warns that `to_datetimeindex`'s default becomes microseconds
    in a future release, which would raise the ceiling to year 294 247. That
    would fix the first case and NOT the second: two axes at `[us]` and `[ns]`
    still meet at `[ns]`, and a 2300 axis still does not fit there.
    """
    from hydromt.data_catalog.drivers.preprocessing import harmonise_dims

    index = ds.indexes["time"]
    if index.dtype == "O":
        ds = ds.assign_coords(time=index.to_datetimeindex(time_unit="s"))
    else:
        widened = seconds_resolution_index(index)
        if widened is not None:
            ds = ds.assign_coords(time=widened)
    return harmonise_dims(ds)


def register_wide_time_preprocess():
    """Add :func:`harmonise_dims_wide_time` to hydromt's preprocessor registry.

    `preprocess:` is resolved BY NAME out of
    ``hydromt.data_catalog.drivers.preprocessing.PREPROCESSORS``; the driver's
    options model takes a string, never a callable, so a registry write is the
    only way to reach it. This adds an entry to a public module-level dict -- it
    does not patch a vendored source file, which AGENTS.md forbids.

    Idempotent, and never displaces an existing name.
    """
    from hydromt.data_catalog.drivers import preprocessing

    preprocessing.PREPROCESSORS.setdefault(
        WIDE_TIME_PREPROCESS, harmonise_dims_wide_time
    )
    return WIDE_TIME_PREPROCESS


#: Bounds variables the generated catalog does not name. It drops the CF
#: abbreviation (`time_bnds`, `lat_bnds`, `lon_bnds`) and the `bnds` dimension,
#: which is what most of CMIP6 publishes -- but the spelling is a modelling
#: centre's choice, and IPSL writes `time_bounds`. A bounds variable that
#: survives the read is not merely noise: `pr` and `tas` come from SEPARATE
#: stores, so when their spans differ their bounds differ too, and the merge
#: dies with `MergeError: conflicting values for variable 'time_bounds'`.
#: Observed on `IPSL/IPSL-CM6A-LR ssp585`, whose `pr` runs to 2300 and `tas` to
#: 2100.
#:
#: Added to the spec rather than to the catalog for the same reason the
#: preprocess is: `cmip6_data.yml` is generated with no offline mode, so
#: changing what it writes means re-crawling `gs://cmip6` and re-stamping
#: `crawled_on` on it and the store index together.
EXTRA_DROP_VARIABLES = ("time_bounds", "lat_bounds", "lon_bounds")


def with_read_overrides(entry_spec):
    """A copy of ``entry_spec`` carrying the read settings the catalog lacks.

    Two of them, both narrow and both about what the driver hands back rather
    than about which store is read:

    * ``driver.options.preprocess`` becomes :data:`WIDE_TIME_PREPROCESS`, so a
      time axis running past 2262 can be decoded at all
      (:func:`harmonise_dims_wide_time`);
    * ``driver.options.drop_variables`` gains :data:`EXTRA_DROP_VARIABLES`,
      so a bounds variable the catalog did not know to name cannot break the
      merge of two variables read from separate stores.

    Overridden on the SPEC rather than in the catalog because
    ``config/catalogs/cmip6_data.yml`` is generated and has no offline mode:
    changing the names it writes means re-crawling `gs://cmip6`, which
    re-stamps `crawled_on` on the catalog and the store index together and
    re-pins every version. That entangles a read fix with every digest in WF2.

    Copies each level it touches, so the catalog entry this was read from is
    left alone -- a shallow `dict()` would share the nested `driver` mapping and
    mutate it for every later caller in the process. Existing drops are kept in
    their own order and nothing is added twice, so a catalog that grows one of
    these names later does not start listing it in duplicate.
    """
    spec = dict(entry_spec)
    driver = dict(spec.get("driver") or {})
    options = dict(driver.get("options") or {})
    options["preprocess"] = WIDE_TIME_PREPROCESS
    dropped = list(options.get("drop_variables") or [])
    dropped += [name for name in EXTRA_DROP_VARIABLES if name not in dropped]
    options["drop_variables"] = dropped
    driver["options"] = options
    spec["driver"] = driver
    return spec


def pin_tail(template_uri, pin_uri):
    """What `pinned_uri` substituted for the catalog URI's trailing glob.

    The generated catalog ends every URI `/{variable}/*/*` and
    `series_identity.pinned_uri` replaces that `/*/*` with the one
    `<grid_label>/<version>` the store index recorded -- so the pinned URI and
    the entry's own template differ by EXACTLY that string, and everything
    ahead of it (activity, institution, model, experiment, member, table) is
    already on the `fetching <entry>` row printed one line earlier.

    Naming the tail rather than the URI takes the row from ~130 characters to
    ~56 and drops nothing a reader could act on: the pin IS the fact the row
    exists to report. The full URI keeps two durable copies -- hydromt's echo
    of it in the log part (muted on the console, never dropped), and
    `cst_source_paths` stamped on the slice itself.

    Falls back to the whole URI whenever the two do not have that relationship,
    so a catalog whose URIs are not DRS-shaped still says where it read from.
    """
    suffix = series_identity.STORE_GLOB_SUFFIX
    if not template_uri.endswith(suffix):
        return pin_uri
    head = template_uri[: -len(suffix)]
    if not pin_uri.startswith(head + "/"):
        return pin_uri
    return pin_uri[len(head) + 1 :]


def bbox_index_slice(centers, low, high, buffer, axis="axis", source=""):
    """The index range ``.raster.clip_bbox`` would take, without needing regularity.

    hydromt maps the bbox through the grid's affine transform and rounds to whole
    cells (``gis/raster.py:1265-1271``). An affine transform is exactly what an
    irregular axis does not have -- but the transform is only ever used to ask
    *which cell edge is nearest this coordinate*, and midpoint edges answer that
    on any ordered 1-D axis. So this interpolates the coordinate's position
    within :func:`grid_weights.midpoint_edges` instead, which on a uniformly
    spaced axis is the same linear map hydromt inverts, and therefore selects the
    same cells (pinned by ``test_fetch_gcm_raw`` against ``clip_bbox`` itself).

    Two properties are inherited from hydromt rather than chosen here:

    * ``buffer`` is in CELLS, not degrees. hydromt has always read it as
      "resolution multiplicity" (``data_catalog.py:1370``), so a regular and an
      irregular slice of the same region must use the same reading or the two
      would differ silently inside one ensemble. The config key said
      ``buffer_degrees`` until t2608182238, which is the misnomer this reading
      had to work around; it is now ``buffer_cells`` and says what it spends.
    * the direction of the axis is irrelevant. ``harmonise_dims`` orients
      latitude N->S, so a label-based ``slice(south, north)`` would return
      NOTHING -- and nothing downstream catches an empty spatial selection,
      since ``check_time_axis`` is about time. Working in index space off the
      edge ladder is order-independent by construction.

    Raises when the bbox leaves the axis's own span, which is the one case
    hydromt handles differently: on a grid spanning 360 degrees it shifts
    longitudes across the 180 meridian first (``_meridian_offset``), and that
    helper reads ``.raster.bounds`` -- so it cannot run here. Refused loudly
    rather than clamped, which would drop the part of the region that wrapped.
    """
    arr = grid_weights.check_axis(centers, axis, source)
    if arr.size == 1:
        return slice(0, 1)
    edges = grid_weights.midpoint_edges(arr)
    span_low, span_high = min(edges[0], edges[-1]), max(edges[0], edges[-1])
    if low < span_low or high > span_high:
        where = f" ({source})" if source else ""
        raise RuntimeError(
            f"{axis}{where}: the region {low:.6g}..{high:.6g} leaves the grid's "
            f"own span {span_low:.6g}..{span_high:.6g}. On a regular grid hydromt "
            "would first shift the axis across the 180 meridian; that shift reads "
            "the raster accessor and so cannot run on an irregular grid. Refused "
            "rather than clipped to the edge, which would silently drop the part "
            "of the region that wrapped."
        )
    ladder = np.arange(edges.size, dtype="float64")
    ascending = edges[-1] > edges[0]
    positions = np.interp(
        [low, high],
        edges if ascending else edges[::-1],
        ladder if ascending else ladder[::-1],
    )
    first = max(int(np.round(positions.min() - buffer)), 0)
    last = max(int(np.round(positions.max() + buffer)), 0)
    return slice(first, last)


def clip_to_bbox(data, bbox, buffer, y_dim, x_dim, source=""):
    """Clip an unclipped dataset to ``bbox``, the way hydromt would have.

    The irregular-grid branch: ``get_rasterdataset`` is asked for the store
    WITHOUT a bbox, so hydromt never reaches ``_slice_spatial_dimensions`` -- the
    only step in its read path that needs an evenly spaced grid -- and the rename,
    unit conversion, CRS and nodata handling all still run. The bbox is then
    applied here, on an axis whose spacing nothing requires to be uniform.

    Refuses an empty selection instead of returning it. With ``buffer >= 1`` the
    index range is at least two cells wide, so this cannot trigger on WF2's
    configuration; it exists because an empty spatial selection produces a slice
    of all-NaN rather than an error, and would surface as a broken change factor
    several rules later.
    """
    west, south, east, north = bbox
    selection = {
        y_dim: bbox_index_slice(
            data[y_dim].values, south, north, buffer, "latitude", source
        ),
        x_dim: bbox_index_slice(
            data[x_dim].values, west, east, buffer, "longitude", source
        ),
    }
    clipped = data.isel(selection)
    empty = [dim for dim in (y_dim, x_dim) if clipped.sizes[dim] == 0]
    if empty:
        where = f" ({source})" if source else ""
        raise RuntimeError(
            f"clipping to {bbox}{where} selected no cells along {empty}. A raw "
            "slice with an empty spatial dimension reduces to NaN rather than "
            "failing, so it is refused here."
        )
    return clipped


def resolve_entry_name(catalog_entry, member):
    """The catalog's own name for one member's source.

    The generated catalog expands placeholders at generation time, so the member
    is part of the entry NAME (``get_stats_climate_proj.py:236``). Use the
    catalog's own grammar rather than string surgery.
    """
    return (
        catalog_entry.format(member=member)
        if "{member}" in catalog_entry
        else f"{catalog_entry}_{member}"
    )


def stale_units(dataset, variable_units):
    """Variables whose recorded ``units`` disagree with the configured ones.

    S8-08(a): a slice cached BEFORE the units fix still claims the
    pre-conversion units. Only variables PRESENT in the dataset are reported --
    a configured variable the slice does not carry is not stale, it is absent.
    """
    return {
        name: units
        for name, units in variable_units.items()
        if name in dataset and dataset[name].attrs.get("units") != units
    }


def check_time_axis(entry, index, driver_index, acquisition_window):
    """Raise if the selected time axis is ambiguous or empty.

    Two failure modes, both of which every check downstream would pass:

    * **duplicates (D8)** -- the catalog URI globs ``{grid_label}/{version}`` and
      ~6% of pinned stores match more than one. Two concatenated stores give a
      duplicated time axis, which halves the effective record while looking
      fine.
    * **an empty window** -- ``.load()`` succeeds, the duplicate test is
      trivially true (0 == 0), and the attrs block then dies on ``index[0]``
      with a bare ``IndexError`` naming neither the source nor the window.
      Reachable on real input -- a historical run starting after 1950, or a
      truncated ssp store -- and **invisible to the fixture gate**, whose three
      models all cover their windows.

    Called BEFORE ``.load()``: coordinates are read at open, so this costs
    nothing lazily and an ambiguous or empty source fails without first
    transferring every selected chunk (~19 s on the benchmark source).

    ``driver_index`` is the axis as the DRIVER returned it, before ``.sel()``
    narrowed it -- reporting both is what tells "the store is short" apart from
    "the window is wrong".
    """
    if index is not None and len(index) != len(set(index)):
        raise RuntimeError(
            f"{entry}: time axis has {len(index) - len(set(index))} duplicate "
            "step(s), so the catalog glob matched more than one store. Pin the "
            "version in the catalog rather than reading an ambiguous source."
        )
    if index is not None and len(index) == 0:
        covered = (
            f"{driver_index[0]}..{driver_index[-1]}"
            if driver_index is not None and len(driver_index)
            else "no steps at all"
        )
        raise RuntimeError(
            f"{entry}: no time steps inside the acquisition window "
            f"{acquisition_window[0]}..{acquisition_window[1]} (the driver "
            f"returned {covered}). This store does not cover the window this "
            "experiment acquires, so it cannot produce a raw slice."
        )


def calendar_pin(pins_for_member):
    """Which variable's store to ask for the model's true calendar.

    Prefer a CERTIFIED variable: the crawl proved ``pr``/``tas`` present, and
    any other name is best-effort (A3), so its store may not exist. Falls back
    to whatever the member does pin, and to ``""`` when it pins nothing.
    """
    return next(
        (v for v in ("tas", "pr") if v in pins_for_member),
        next(iter(pins_for_member), ""),
    )


def calendar_store_uri(template, member, calendar_var, pins_for_member):
    """Address ONE store directly, so the calendar read lists no bucket.

    Returns ``""`` when there is nothing to ask -- no pinned variable, or a
    globbed URI the pins cannot resolve to a single location -- and the caller
    then records :data:`series_identity.CALENDAR_UNKNOWN` rather than guessing.
    """
    if not calendar_var:
        return ""
    store_uri = template.format(member=member, variable=calendar_var)
    if store_uri.endswith(series_identity.STORE_GLOB_SUFFIX):
        matches = pins_for_member.get(calendar_var) or []
        store_uri = (
            store_uri[: -len(series_identity.STORE_GLOB_SUFFIX)] + "/" + matches[-1]
            if matches
            else ""
        )
    return store_uri


def raw_slice_attrs(
    components,
    member,
    expected_raw_digest,
    acquisition_window,
    first,
    last,
    store_calendar,
    bbox,
    region_fp,
    buffer,
):
    """The ``cst_*`` block stamped on a raw slice -- the seam with the reduce stage.

    ``series_identity.assert_raw_identity`` reads these back, which is what lets
    the reduce stage make ZERO remote calls. One key here is DIAGNOSTIC rather
    than part of the seam -- ``cst_entry_identity_digest`` exists so a digest
    mismatch can name the component that moved -- and two keys are deliberately
    ABSENT:
    ``cst_series_digest`` and ``cst_reducer_module_hash``. A raw slice is
    pre-reduction and must not claim an identity that implies arithmetic was
    applied -- and ``cst_raw_digest`` excluding the reducer hash is exactly what
    makes a formula edit free.
    """
    entry_meta = (components.get("entry_identity") or {}).get(member, {})
    return {
        "cst_schema_version": series_identity.SCHEMA_VERSION,
        "cst_raw_digest": expected_raw_digest,
        "cst_catalog_entry": components.get("catalog_entry", ""),
        "cst_acquisition_window": " / ".join(acquisition_window),
        "cst_time_first": first,
        "cst_time_last": last,
        # From the STORE, not from the index -- the index no longer knows.
        "cst_calendar": store_calendar,
        "cst_region_bounds": ", ".join(f"{b:.9g}" for b in bbox),
        "cst_region_fingerprint": region_fp,
        "cst_buffer_cells": buffer,
        "cst_members": member,
        "cst_source_paths": json.dumps(components.get("pins", {}), sort_keys=True),
        "cst_crs": str((entry_meta.get("metadata", {}) or {}).get("crs", "")),
        # Diagnostic, not identity. See `series_identity.entry_identity_digest`
        # for why this is the one component worth stamping separately and why
        # adding it owes no SCHEMA_VERSION bump.
        "cst_entry_identity_digest": series_identity.entry_identity_digest(
            components, member
        ),
    }


def fetch_raw_slice(
    *,
    region_path,
    raw_nc_out,
    catalog_path,
    catalog_entry,
    member,
    variables,
    variable_units,
    buffer,
    acquisition_window,
    components,
):
    """Fetch ONE raw slice of the remote store, or return early on a cache hit.

    Extracted from the `if "snakemake"` block on 2026-08-18 so a second caller
    could exist without a second implementation. `dev/scripts/stage_cmip6.py`
    stages slices outside a project with it; the Snakemake adapter below is the
    only other caller. Everything the two share -- the buffer and time
    semantics, the D8 duplicate-time check, the D12 pin, the units adapter, the
    calendar read and the atomic write -- lives here precisely so neither can
    drift from the other.

    Writes `raw_nc_out` atomically, stamped with `cst_raw_digest`, and returns
    without touching the network when a valid slice is already there.

    Args:
        region_path: polygon whose bounds clip the store. Part of the cache
            identity via `series_identity.region_fingerprint`, so a different
            polygon is a different slice.
        raw_nc_out: destination netCDF.
        catalog_path: hydromt data catalog (the generated cmip6 one).
        catalog_entry: entry name, `{member}` placeholder still in it.
        member: the ensemble member to resolve that placeholder with.
        variables: variable names to read.
        variable_units: name -> units string stamped after the adapter's
            conversion (S8-08(a)).
        buffer: degrees added around the polygon bounds.
        acquisition_window: (start, end) as the driver understands them.
        components: raw digest components -- `series_identity.raw_components`,
            carrying NO reducer hash. A caller that adds one makes a formula
            edit re-download, which is what the stage-A split exists to avoid.
    """
    # Report the switch above rather than enforce it. `setdefault` is correct:
    # it fixes the UNSET case, which is what the 57.7 s vs >836 s benchmark
    # actually measured, while leaving a deliberate export intact -- the opt-in
    # a future HNS-backed catalog would need. What was missing is legibility:
    # an inherited "true" turned a 58 s job into a 14-minute one with nothing in
    # the log to say why. One row, WARNING when the effective value is not the
    # one this module needs, so a slow run explains itself.
    _row, _level = hns_switch_row(
        os.environ.get("GCSFS_EXPERIMENTAL_ZB_HNS_SUPPORT", "")
    )
    log_row(_row, module="fetch", level=_level)

    # S8-08(a): see get_stats_climate_proj.py. The adapter converts the values
    # and leaves the `units` attribute describing the pre-conversion quantity.
    # NOTE: raw_digest_components carries NO reducer hash. See
    # series_identity.raw_components — the Snakefile must keep it that way, or a
    # formula edit re-downloads and the split buys nothing.

    region_fp = series_identity.region_fingerprint(region_path)
    expected_raw_digest = series_identity.raw_digest(components, region_fp)

    # --- revalidation before touching the network (D9's argument, one layer up)
    if series_identity.cache_hit(
        [raw_nc_out], expected_raw_digest, digest_attr="cst_raw_digest"
    ):
        # `raw cache_hit <entry> (<path>)` said one identity three times: the
        # file's basename IS the resolved entry name with `/` sanitized to `_`
        # (`series_identity.series_key`), so the name and the path restated each
        # other, and `cache_hit` restated the outcome in implementation
        # vocabulary. The basename alone carries all of it -- and it is the
        # spelling the `wrote raw` row already uses, so the two outcomes of this
        # rule now read as a pair.
        #
        # "already staged" rather than "already fetched": the file being present
        # is not what earns this row. `cache_hit` compares the DIGEST, so a
        # slice on disk whose recipe has moved is not skipped -- it re-fetches
        # and reports through the `fetching` row below.
        log_row(
            f"already staged, skipping {os.path.basename(raw_nc_out)}",
            module="fetch",
        )
        # S8-08(a): a slice cached BEFORE the units fix still claims the
        # pre-conversion units. Repair it in place rather than leaving the two
        # tiers disagreeing -- `scalar/` is stamped on every reduce, so
        # skipping this would make `raw/` the only artifact still lying about
        # its own values, and only on projects old enough to have a cache.
        #
        # Safe against the identity: `units` is a VARIABLE attribute and the
        # digest covers neither it nor the values. Paid once per stale file;
        # a slice already carrying the right units takes the fast path.
        import xarray as _xr

        with _xr.open_dataset(raw_nc_out) as _cached:
            stale = stale_units(_cached, variable_units)
            repaired = _cached.load() if stale else None
        if stale:
            for name, units in stale.items():
                repaired[name].attrs["units"] = units
            series_identity.write_netcdf_atomic(repaired, raw_nc_out)
            repaired.close()
            log_row(
                f"repaired stale units on the cached slice: {sorted(stale)}",
                module="fetch",
            )
        os.utime(raw_nc_out, None)
        return

    # The digest is deliberately NOT echoed here (nor on the cache-hit row
    # above): it is stamped on the slice as `cst_raw_digest`, so the durable
    # copy is the file's own, and a 12-char hex prefix identifies nothing a
    # reader can act on. What identifies the job is the RESOLVED entry name,
    # which is one field rather than the two `entry=<...{member}...>
    # member=<m>` spelled the placeholder and its value separately.
    #
    # The window is gone from the announcement. `acquisition_window` takes
    # exactly TWO values across the whole catalog -- one for `historical` and
    # one for every `sspNNN` -- so it is a property of the experiment the row
    # already names, restated on all 161 slices of a staging run. It keeps the
    # copies that matter: `cst_acquisition_window` on the slice, and
    # `check_time_axis`, which prints it in full at the one moment it is the
    # thing that is wrong. Both facts are already pinned --
    # `test_series_identity.py::test_acquisition_window_is_fixed_per_experiment_class`
    # for the two values, `test_fetch_gcm_raw.py` for the stamped attribute --
    # so a change making the window per-model turns those red, next to this.
    log_row(f"fetching {resolve_entry_name(catalog_entry, member)}", module="fetch")

    os.makedirs(os.path.dirname(raw_nc_out) or ".", exist_ok=True)

    # --- everything below the cache exit is MISS-ONLY work -----------------
    # hydromt is imported here, not with the module: it is used first at the
    # DataCatalog below, and a cache hit is the common case. Fresh-process
    # measurement, external review 2026-07-31: geopandas + xarray 2.7-3.0 s /
    # 118 MiB RSS versus geopandas + hydromt + xarray 6.7-7.4 s / 311 MiB, so a
    # cached job stops paying ~4 s and ~193 MiB peak (against the 9.98-10.24 s /
    # ~327 MiB the nine cached fetch jobs cost in wf2_benchmarks.md).
    # The `.raster` accessor hydromt registers is used only for the dimension
    # NAMES on the irregular-grid branch below -- this module reads through
    # DataCatalog, unlike get_stats_climate_proj.py. That branch exists because
    # the accessor's REGULARITY precondition is what refuses a Gaussian grid.
    # It stays INSIDE the tee: `tee_to_log` repoints library handlers bound
    # before entry, so an import that lands after entry must keep landing after
    # entry, or hydromt's StreamHandler binds to the real stdout and bypasses
    # the log file.
    import geopandas as gpd
    import hydromt

    # The region is read here too, not before the cache check: `bbox` is
    # consumed only by the read below and by the attrs at the end, both on this
    # path, so a cache hit now opens the polygon once (inside
    # `region_fingerprint`) instead of twice. Deliberately NOT folded into
    # `region_fingerprint` -- that function is a cache-identity contract
    # (design D9), not a place to hang a bounds helper.
    geom = gpd.read_file(region_path)
    bbox = list(geom.geometry.bounds.values[0])

    # The generated catalog expands placeholders at generation time, so the
    # member is part of the entry NAME (get_stats_climate_proj.py:236). Use the
    # catalog's own grammar rather than string surgery.
    entry = resolve_entry_name(catalog_entry, member)

    # --- spend the D12 pin instead of listing the bucket ------------------
    # The URI ends /{variable}/*/* , so resolving it lists the store to expand
    # {grid_label}/{version}. The index already records that location, so
    # substitute it and address the store directly.
    # Worth ~10 s per source: open 49.9 s pinned vs 60.0 s globbed, 3 samples
    # per arm, non-overlapping (benchmark note 3.2). But gcsfs answers the same
    # patterns in 0.41 s, so what this removes is hydromt's resolver overhead on
    # a wildcard URI, NOT a slow network listing -- describe it that way. The
    # second reason to keep it is determinism: one known store rather than a
    # pattern whose match set can change under the job.
    # Falls back to the globbed catalog whenever the pins cannot name one
    # location (per-variable divergence, or >1 match, which is D8's ambiguity and
    # must stay globbed so the duplicate-time assertion still fires).
    entry_spec = series_identity.load_catalog_entry(catalog_path, catalog_entry)
    # Both branches below register ONE entry from a spec of our own rather than
    # loading the whole catalog, which is what lets the preprocess override
    # reach either of them. The glob branch used `DataCatalog(data_libs=...)`
    # until 2026-08-19; from_dict expands `placeholders:` the same way (the
    # pinned branch has always relied on that to resolve the member), and it
    # stops parsing 289 entries to read one.
    register_wide_time_preprocess()
    read_spec = with_read_overrides(entry_spec)
    pins_for_member = (components.get("pins") or {}).get(member, {})
    pin_uri = series_identity.pinned_uri(
        str(entry_spec.get("uri", "")), pins_for_member
    )
    # Empty whenever `pinned_uri` succeeded, so the pinned branch carries no
    # explanation it does not need.
    ambiguous = ambiguous_pins(str(entry_spec.get("uri", "")), pins_for_member)
    if pin_uri is None:
        # The GLOB is named in full, and this is the branch where that
        # matters. The console mutes hydromt's `data_source - Reading <entry>
        # from <uri>` echo, which repeats the URI at ~175 characters; the
        # pinned branch above can afford to print only the pin because the
        # entry it hangs off is a fixed template the previous row already
        # named, while a glob is precisely the case where WHICH URI was used
        # is not derivable -- it is the pattern whose match set could change.
        log_row(
            f"no single pin; keeping the URI glob: {entry_spec.get('uri', '')}",
            module="fetch",
        )
        if ambiguous:
            # Said BEFORE the read, not only when one fails. The glob is about
            # to open every version, and on the runs where that happens to
            # merge cleanly this is the only notice that the source was chosen
            # for the operator rather than by them.
            log_row(
                f"more than one version on the store "
                f"({ambiguous_versions_phrase(ambiguous)}): {entry}",
                module="fetch",
                level="WARNING",
            )
        data_catalog = hydromt.DataCatalog()
        data_catalog.from_dict({catalog_entry: read_spec})
    else:
        pinned_spec = dict(read_spec)
        pinned_spec["uri"] = pin_uri
        # Registered through hydromt's own dict schema -- same driver, adapter and
        # metadata, only the URI narrowed. from_dict rather than a YAML round-trip:
        # hydromt 1.3's to_yml drops driver.options.preprocess (see
        # prepare_climate_data_catalog.py).
        data_catalog = hydromt.DataCatalog()
        data_catalog.from_dict({catalog_entry: pinned_spec})
        # The TAIL, not the URI: everything ahead of the pin is the entry, and
        # the row above names the entry. See `pin_tail` for where the full URI
        # still lives.
        log_row(
            f"pinned {pin_tail(str(entry_spec.get('uri', '')), pin_uri)}, "
            "no bucket listing",
            module="fetch",
        )
    # --- the read, with the irregular-grid branch behind it ----------------
    # 27 of 67 CMIP6 models publish Amon on a Gaussian grid, whose latitudes are
    # Legendre roots and so vary by ~1% -- an order above hydromt's 5e-4
    # regularity tolerance. Those models raised here and were SILENTLY ABSENT
    # from every WF2 ensemble: CanESM5, all five EC-Earth3 variants,
    # MPI-ESM1-2-HR/LR, CNRM-CM6-1/ESM2-1, MIROC6, MRI-ESM2-0, BCC-CSM2-MR
    # (measured 2026-08-18; the table is in section 1 of
    # dev/reference/workflows/wf2-cmip6-store-readability.md).
    #
    # `_slice_spatial_dimensions` is the ONLY step in hydromt's read path that
    # needs an evenly spaced grid, and it runs only when a bbox is passed. So the
    # branch asks for the same store WITHOUT one -- rename, unit conversion, CRS
    # and nodata all still hydromt's -- and applies the bbox with `clip_to_bbox`,
    # which needs the axes ordered but not evenly spaced.
    #
    # Tried in this order, rather than probing the grid first, because the probe
    # would have to guess which variable's store to read and could then DISAGREE
    # with hydromt's own verdict. The regular path is therefore untouched, down
    # to the cell selection, so no cached slice invalidates and the digest does
    # not move; the models this rescues never had a cached slice to begin with.
    # The price is a second open on the irregular path (~20 s pinned, against a
    # first open that has already paid the store metadata), on models that
    # previously produced nothing at all.
    # Only a FAILED read is rewritten, and only when the pins were ambiguous;
    # a source whose two published versions differ merely in metadata merges
    # cleanly and still stages. The duplicate-axis face of the same ambiguity
    # is caught by `check_time_axis` below, which carries its own message.
    with explaining_ambiguous_versions(entry, ambiguous):
        try:
            data = data_catalog.get_rasterdataset(
                entry,
                bbox=bbox,
                buffer=buffer,
                time_range=acquisition_window,
                variables=variables,
            )
        except ValueError as exc:
            if not is_irregular_grid_error(exc):
                raise
            # The WHY -- Gaussian latitudes against hydromt's 5e-4 regularity
            # tolerance -- is in `is_irregular_grid_error` and in
            # dev/reference/workflows/wf2-cmip6-store-readability.md,
            # section 1, which carries the measured table of which models. On the
            # console this row has one job: say that this source took the fallback
            # read, and which source. The WARNING level is what makes it findable;
            # 175 characters of explanation on every one of ~27 affected models is
            # not.
            log_row(
                f"irregular grid, applying the bbox directly: {entry}",
                module="fetch",
                level="WARNING",
            )
            try:
                data = data_catalog.get_rasterdataset(
                    entry,
                    time_range=acquisition_window,
                    variables=variables,
                )
                data = clip_to_bbox(
                    data,
                    bbox,
                    buffer,
                    data.raster.y_dim,
                    data.raster.x_dim,
                    source=entry,
                )
            except Exception as fallback_exc:
                # hydromt's own errors name neither the entry nor the member, and
                # this one is now two reads deep -- say which series died.
                raise RuntimeError(
                    f"{entry}: the irregular-grid read path failed after hydromt "
                    f"refused the grid ({type(fallback_exc).__name__}: {fallback_exc})"
                ) from fallback_exc
    # Kept for the empty-window error below: `time_range` is applied by the
    # driver and `.sel` narrows it again, so "the driver returned 1850..2014"
    # is the diagnostic that tells the two apart.
    driver_index = data.indexes.get("time")
    # cmip6/cmip5 cftime calendars are not always honoured by time_range alone.
    data = data.sel(time=slice(*acquisition_window))
    # Back to nanoseconds now the extension years are gone. `harmonise_dims_
    # wide_time` decodes at second resolution so a store running to 2300 can be
    # opened at all; every acquisition window ends in 2100, so by here the axis
    # fits `datetime64[ns]` again and a slice that already staged is written
    # byte-for-byte as before. Guarded on the dtype rather than applied blindly:
    # a store that decoded straight to datetime64 never went through the
    # widening, and `astype` on an already-ns axis is a no-op worth not doing.
    time_index = data.indexes.get("time")
    if time_index is not None and getattr(time_index, "dtype", None) == "datetime64[s]":
        data = data.assign_coords(time=time_index.astype("datetime64[ns]"))

    # D8: the catalog URI globs {grid_label}/{version} and ~6% of pinned stores
    # match more than one. Two concatenated stores give a duplicated time axis,
    # which halves the effective record while looking fine.
    # Checked BEFORE `.load()`: coordinates are read at open, so this costs
    # nothing lazily, and an ambiguous source now fails without first
    # transferring every selected chunk (~19 s on the benchmark source). Kept
    # AFTER `.sel()` so duplicates outside the acquisition window stay out of it.
    index = data.indexes.get("time")
    check_time_axis(entry, index, driver_index, acquisition_window)

    # Eager, and not only for speed: a lazy slice written by to_netcdf reads from
    # dask's thread pool and deadlocks on the HDF5 lock (measured, commit
    # bf1f4a5). After bbox/time slicing this is well under a megabyte.
    data = data.load()

    # --- the model's TRUE calendar, read from the store ---------------------
    # `index` is a DatetimeIndex by now and has no `.calendar`: our catalog
    # requests `preprocess: harmonise_dims`, whose time branch converts a
    # CFTimeIndex away (hydromt .../drivers/preprocessing.py:66). Reading
    # `.calendar` off it recorded "" while the file was written asserting
    # `proleptic_gregorian` -- false for every noleap/360_day model. So ask the
    # store, which is the only place that still knows.
    # One consolidated-metadata read, ~0.3 s; see the blocker note.
    pins_for_member = (components.get("pins") or {}).get(member, {})
    # Prefer a CERTIFIED variable: the crawl proved pr/tas present, and any
    # other name is best-effort (A3), so its store may not exist.
    calendar_var = calendar_pin(pins_for_member)
    store_uri = calendar_store_uri(
        pin_uri or str(entry_spec.get("uri", "")),
        member,
        calendar_var,
        pins_for_member,
    )
    store_calendar = (
        series_identity.read_store_calendar(store_uri)
        if store_uri
        else series_identity.CALENDAR_UNKNOWN
    )
    log_row(
        f"store calendar={store_calendar} ({calendar_var or 'no pin'})",
        module="fetch",
    )

    first, last = (str(index[0]), str(index[-1])) if index is not None else ("", "")
    for _name, _units in variable_units.items():
        if _name in data:
            data[_name].attrs["units"] = _units

    data.attrs.update(
        raw_slice_attrs(
            components,
            member,
            expected_raw_digest,
            acquisition_window,
            first,
            last,
            store_calendar,
            bbox,
            region_fp,
            buffer,
        )
    )
    # ...and drop the inherited attrs that describe ONE source file. This
    # slice merges pr and tas, so a single `variable_id` is wrong whichever
    # way the merge resolved it; `cst_source_paths` above carries the real
    # per-variable provenance (R9 P2 F4).
    series_identity.drop_inherited_single_source_attrs(data)

    series_identity.write_netcdf_atomic(data, raw_nc_out)
    log_row(
        f"wrote raw {os.path.basename(raw_nc_out)} "
        f"({os.path.getsize(raw_nc_out) / 1e6:.2f} MB, {len(index) if index is not None else 0} steps)",
        module="fetch",
    )
    data.close()


# ---------------------------------------------------------------------------
# Snakemake adapter. `if "snakemake" in globals():` is invisible to every unit
# test, so it is checked by running the pipeline or not at all -- which is the
# reason it now holds nothing but the unpacking.
# ---------------------------------------------------------------------------

if "snakemake" in globals():
    sm = globals()["snakemake"]

    with tee_to_log(sm.log[0]):
        fetch_raw_slice(
            region_path=sm.input.region_path,
            raw_nc_out=str(sm.output.raw_nc),
            catalog_path=sm.params.catalog_path,
            catalog_entry=sm.params.catalog_entry,
            member=sm.params.member,
            variables=list(sm.params.variables),
            # S8-08(a): see get_stats_climate_proj.py. The adapter converts the
            # values and leaves `units` describing the pre-conversion quantity.
            variable_units=dict(sm.params.variable_units),
            buffer=sm.params.buffer_cells,
            acquisition_window=tuple(sm.params.acquisition_window),
            # NOTE: carries NO reducer hash. See series_identity.raw_components --
            # the Snakefile must keep it that way, or a formula edit re-downloads
            # and the split buys nothing.
            components=sm.params.raw_digest_components,
        )
