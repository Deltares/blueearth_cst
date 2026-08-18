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

from blueearth_cst.projections import series_identity
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
    the reduce stage make ZERO remote calls. Two keys are deliberately ABSENT:
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
        "cst_buffer_degrees": buffer,
        "cst_members": member,
        "cst_source_paths": json.dumps(components.get("pins", {}), sort_keys=True),
        "cst_crs": str((entry_meta.get("metadata", {}) or {}).get("crs", "")),
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
        log_row(
            f"raw cache_hit entry={catalog_entry} member={member} ({raw_nc_out})",
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
    # reader can act on. What identifies the job is the entry and member.
    log_row(
        f"fetching entry={catalog_entry} "
        f"member={member} window={acquisition_window[0]}..{acquisition_window[1]}",
        module="fetch",
    )

    os.makedirs(os.path.dirname(raw_nc_out) or ".", exist_ok=True)

    # --- everything below the cache exit is MISS-ONLY work -----------------
    # hydromt is imported here, not with the module: it is used first at the
    # DataCatalog below, and a cache hit is the common case. Fresh-process
    # measurement, external review 2026-07-31: geopandas + xarray 2.7-3.0 s /
    # 118 MiB RSS versus geopandas + hydromt + xarray 6.7-7.4 s / 311 MiB, so a
    # cached job stops paying ~4 s and ~193 MiB peak (against the 9.98-10.24 s /
    # ~327 MiB the nine cached fetch jobs cost in wf2_benchmarks.md).
    # The `.raster` accessor hydromt registers is not used here -- this module
    # reads through DataCatalog, unlike get_stats_climate_proj.py.
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
    pin_uri = series_identity.pinned_uri(
        str(entry_spec.get("uri", "")),
        (components.get("pins") or {}).get(member, {}),
    )
    if pin_uri is None:
        # The URI goes on OUR row in both branches. The console mutes
        # hydromt's `data_source - Reading <entry> from <uri>` echo, which
        # repeats this URI at ~175 characters (4 such rows per WF2 run);
        # that mute is only information-preserving because the glob case
        # names its URI here rather than relying on the echo.
        log_row(
            f"no single pin; keeping the URI glob: {entry_spec.get('uri', '')}",
            module="fetch",
        )
        data_catalog = hydromt.DataCatalog(data_libs=catalog_path)
    else:
        pinned_spec = dict(entry_spec)
        pinned_spec["uri"] = pin_uri
        # Registered through hydromt's own dict schema -- same driver, adapter and
        # metadata, only the URI narrowed. from_dict rather than a YAML round-trip:
        # hydromt 1.3's to_yml drops driver.options.preprocess (see
        # prepare_climate_data_catalog.py).
        data_catalog = hydromt.DataCatalog()
        data_catalog.from_dict({catalog_entry: pinned_spec})
        log_row(f"pinned URI (no bucket listing): {pin_uri}", module="fetch")
    data = data_catalog.get_rasterdataset(
        entry,
        bbox=bbox,
        buffer=buffer,
        time_range=acquisition_window,
        variables=variables,
    )
    # Kept for the empty-window error below: `time_range` is applied by the
    # driver and `.sel` narrows it again, so "the driver returned 1850..2014"
    # is the diagnostic that tells the two apart.
    driver_index = data.indexes.get("time")
    # cmip6/cmip5 cftime calendars are not always honoured by time_range alone.
    data = data.sel(time=slice(*acquisition_window))

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
            buffer=sm.params.buffer_degrees,
            acquisition_window=tuple(sm.params.acquisition_window),
            # NOTE: carries NO reducer hash. See series_identity.raw_components --
            # the Snakefile must keep it that way, or a formula edit re-downloads
            # and the split buys nothing.
            components=sm.params.raw_digest_components,
        )
