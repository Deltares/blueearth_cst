"""Stage A1 — acquire one raw slice of a remote GCM store (design revision 6).

The **only** module in WF2 that opens the remote store. Stage A used to fetch and
reduce in one job, so anything invalidating the *reduction* re-triggered the
*download*: an edit to a formula cost nine remote reads. This split makes a
reduction edit re-read local disk instead.

Why the split is worth its second cache, measured on 2026-07-30
(`dev/working/2026-07-30_wf2-fetch-reduce-benchmark.md`):

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

Invoked from ``Snakefile_climate_projections`` via ``script:``; reads
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

if "snakemake" in globals():
    sm = globals()["snakemake"]

    with tee_to_log(sm.log[0]):
        import geopandas as gpd
        import hydromt

        region_path = sm.input.region_path
        raw_nc_out = str(sm.output.raw_nc)
        catalog_path = sm.params.catalog_path
        catalog_entry = sm.params.catalog_entry
        member = sm.params.member
        variables = list(sm.params.variables)
        buffer = sm.params.buffer_degrees
        acquisition_window = tuple(sm.params.acquisition_window)
        # NOTE: raw_digest_components carries NO reducer hash. See
        # series_identity.raw_components — the Snakefile must keep it that way, or a
        # formula edit re-downloads and the split buys nothing.
        components = sm.params.raw_digest_components

        geom = gpd.read_file(region_path)
        bbox = list(geom.geometry.bounds.values[0])
        region_fp = series_identity.region_fingerprint(region_path)
        expected_raw_digest = series_identity.raw_digest(components, region_fp)

        # --- revalidation before touching the network (D9's argument, one layer up)
        if series_identity.cache_hit([raw_nc_out], expected_raw_digest, digest_attr="cst_raw_digest"):
            log_row(
                f"raw cache_hit digest={expected_raw_digest[:12]} ({raw_nc_out})",
                module="fetch",
            )
            os.utime(raw_nc_out, None)
            raise SystemExit(0)

        log_row(
            f"fetching raw digest={expected_raw_digest[:12]} entry={catalog_entry} "
            f"member={member} window={acquisition_window[0]}..{acquisition_window[1]}",
            module="fetch",
        )

        os.makedirs(os.path.dirname(raw_nc_out) or ".", exist_ok=True)

        # The generated catalog expands placeholders at generation time, so the
        # member is part of the entry NAME (get_stats_climate_proj.py:236). Use the
        # catalog's own grammar rather than string surgery.
        entry = (
            catalog_entry.format(member=member)
            if "{member}" in catalog_entry
            else f"{catalog_entry}_{member}"
        )

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
            log_row("no single pin for this source; keeping the URI glob", module="fetch")
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
        # cmip6/cmip5 cftime calendars are not always honoured by time_range alone.
        data = data.sel(time=slice(*acquisition_window))
        # Eager, and not only for speed: a lazy slice written by to_netcdf reads from
        # dask's thread pool and deadlocks on the HDF5 lock (measured, commit
        # bf1f4a5). After bbox/time slicing this is well under a megabyte.
        data = data.load()

        # D8: the catalog URI globs {grid_label}/{version} and ~6% of pinned stores
        # match more than one. Two concatenated stores give a duplicated time axis,
        # which halves the effective record while looking fine.
        index = data.indexes.get("time")
        if index is not None and len(index) != len(set(index)):
            raise RuntimeError(
                f"{entry}: time axis has {len(index) - len(set(index))} duplicate "
                "step(s), so the catalog glob matched more than one store. Pin the "
                "version in the catalog rather than reading an ambiguous source."
            )

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
        calendar_var = next(
            (v for v in ("tas", "pr") if v in pins_for_member),
            next(iter(pins_for_member), ""),
        )
        store_uri = ""
        if calendar_var:
            template = pin_uri or str(entry_spec.get("uri", ""))
            store_uri = template.format(member=member, variable=calendar_var)
            if store_uri.endswith(series_identity.STORE_GLOB_SUFFIX):
                matches = pins_for_member.get(calendar_var) or []
                store_uri = (
                    store_uri[: -len(series_identity.STORE_GLOB_SUFFIX)]
                    + "/"
                    + matches[-1]
                    if matches
                    else ""
                )
        store_calendar = (
            series_identity.read_store_calendar(store_uri)
            if store_uri
            else series_identity.CALENDAR_UNKNOWN
        )
        log_row(f"store calendar={store_calendar} ({calendar_var or 'no pin'})", module="fetch")

        entry_meta = (components.get("entry_identity") or {}).get(member, {})
        first, last = (str(index[0]), str(index[-1])) if index is not None else ("", "")
        data.attrs.update(
            {
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
                "cst_source_paths": json.dumps(
                    components.get("pins", {}), sort_keys=True
                ),
                "cst_crs": str((entry_meta.get("metadata", {}) or {}).get("crs", "")),
                # Deliberately absent: cst_series_digest and
                # cst_reducer_module_hash. A raw slice is pre-reduction and must not
                # claim an identity that implies arithmetic was applied.
            }
        )

        series_identity.write_netcdf_atomic(data, raw_nc_out)
        log_row(
            f"wrote raw {os.path.basename(raw_nc_out)} "
            f"({os.path.getsize(raw_nc_out) / 1e6:.2f} MB, {len(index) if index is not None else 0} steps)",
            module="fetch",
        )
        data.close()
