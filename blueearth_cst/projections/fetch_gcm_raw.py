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
        data_catalog = hydromt.DataCatalog(data_libs=catalog_path)

        # The generated catalog expands placeholders at generation time, so the
        # member is part of the entry NAME (get_stats_climate_proj.py:236). Use the
        # catalog's own grammar rather than string surgery.
        entry = (
            catalog_entry.format(member=member)
            if "{member}" in catalog_entry
            else f"{catalog_entry}_{member}"
        )
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
                "cst_calendar": str(getattr(index, "calendar", "") or ""),
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
