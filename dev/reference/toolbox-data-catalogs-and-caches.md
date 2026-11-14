# Data catalogs, staging and caches

Last updated: 2026-11-14

Use this reference when changing a tracked data catalog, the tools that generate or stage catalog data, or
the project caches that hold extracted historical climate and CMIP6 series. The
[data catalog guide](../../docs/technical-manual/data-catalogs-and-access.md) explains catalog selection to
users, and the [forcing-unit reference](toolbox-forcing-units.md) owns forcing admission. The local
staging recipe lives in
[environment maintenance](../../dev/scripts/environment-notes.md#local-data-staging-p-drive-bypass).

## Tracked catalogs

Register every dataset in a catalog under [`config/catalogs/`](../../config/catalogs) and pass it to
HydroMT through `project.catalog` (`-d`); Snakefiles and modules do not hardcode data paths. The header of
each file states its origin. Later catalogs override same-named entries, so list order matters.

- **`deltares_data_pdrive.yml`:** HydroMT's `deltares_data` v1.2.0 catalog, vendored verbatim with the
  P-drive root; every tracked seed uses it, including the Linux seed. Change it only by re-vendoring.
- **`deltares_data.yml`:** the same datasets under a local archive root, for users with a local copy.
- **`deltares_data_linux.yml`:** an older-schema catalog with a Linux root; no tracked seed uses it.
- **`deltares_data_staged.yml`:** generated overlay of the staged entries; see [Staging](#staging).
- **`deltares_era5_daily_zarr.yml`:** ERA5 daily Zarr override loaded after the base catalog; a development
  aid outside the user-facing catalog inventory. The rapid seed lists the P-drive catalog, the staged
  overlay and this override, in that order.
- **`cmip6_data.yml`, `cmip6_store_index.json`:** generated together by
  [`generate_cmip6_catalog.py`](../../dev/scripts/generate_cmip6_catalog.py) from one live `gs://cmip6`
  crawl. The catalog has one entry per model and scenario with that model's existing members; the index
  records each source's grid label and version. Both carry the same `crawled_on`, which a consumer checks.
  Never hand-edit them; [`test_generate_cmip6_catalog.py`](../../tests/test_generate_cmip6_catalog.py)
  covers the generator.
- **`archive/`:** catalogs used only by archived configurations; not live.

The [CMIP6 source reference](wf2-cmip6-source-readability.md) owns CMIP6 grid and version handling.

Experiment reservations record each catalog's parsed YAML and resolved root
([`catalog_identity.py`](../../blueearth_cst/experiment/catalog_identity.py)): explicit `meta.root`, else the
first existing `meta.roots` candidate, else the catalog's directory. A newly available preferred root can
therefore refuse an existing experiment without any YAML change.

## Staging

Staging tools are repository maintenance under `dev/scripts/`; no run depends on them.

- [`stage_data.py`](../../dev/scripts/stage_data.py) mirrors a bounding-box subset of a remote HydroMT root
  to a local root with the same relative paths, configured by `stage_data.yml`. It skips files already
  present and removes a failed partial output.
- [`write_staged_catalog.py`](../../dev/scripts/write_staged_catalog.py) regenerates
  `deltares_data_staged.yml` from the vendored catalog and `stage_data.yml`: staged entries, with the local
  mirror as the first root and the P drive as fallback. Run it after changing either input;
  [`test_staged_catalog.py`](../../tests/test_staged_catalog.py) fails when the committed overlay is stale.
- [`stage_cmip6.py`](../../dev/scripts/stage_cmip6.py) writes CMIP6 slices for one region through WF2's own
  `fetch_gcm_raw.fetch_raw_slice`, stamped with the same raw digest. Placed in a project's WF2 `raw/`
  directory, they are cache hits only when the region polygon matches the project's.

## Project caches

Two caches persist inside a project and are reused across workflows and runs. Neither can silently feed
stale content into a product, so pruning them is disk hygiene and an explicit owner action.

**Historical climate store** — `data/climate/historical/<source>_<start>_<end>/`.
`climate_store_rule` in [`wf3_science.py`](../../blueearth_cst/shared/wf3_science.py) is the single producer
contract, declared by WF1 and WF3 and generated per candidate source by WF0, so whichever workflow runs
first extracts and the others reuse the store. The directory name is a cache key from source and window;
changing either creates a new store and strands the old one. Inputs are exactly the catalog file and the
shared region: editing either reruns extraction, while data changed behind an unchanged catalog entry needs
`--forcerun`. [`test_climate_store_contract.py`](../../tests/test_climate_store_contract.py) and
[`test_climate_store_freshness.py`](../../tests/test_climate_store_freshness.py) pin these rules.

**WF2 projection caches** — `data/climate/projections/<clim_project>/`. `raw/` holds remote slices from
[`fetch_gcm_raw.py`](../../blueearth_cst/projections/fetch_gcm_raw.py), the only WF2 module that opens the
remote store. The series files are reduced from those slices and carry a digest from
[`series_identity.py`](../../blueearth_cst/projections/series_identity.py) that covers the region polygon by
content and the reducer by `kernel_hash`. `analyze_projections.smk` lists the reducer root functions;
`kernel_hash` follows their calls into `blueearth_cst`, hashes behaviour rather than file bytes (comments
and docstrings do not count) and folds in `pixi.lock`. The raw digest excludes the reducer, so a reduction
change re-reads local slices without downloading. Consumers recompute expected digests and refuse a
mismatch ([`test_series_identity.py`](../../tests/test_series_identity.py)).

**Reuse across projects** — [`data_reuse.py`](../../blueearth_cst/shared/data_reuse.py) lets a project take
both caches from sibling projects named by `project.reuse_data_from` or `CST_REUSE_DATA_FROM`; the
[configuration reference](../../docs/technical-manual/configuration.md#data-reuse) states the user contract.
Every new historical store writes `_engine/extraction_record.json`, whose `MATCH_FIELDS` must equal the
request's; `write_reused_store` in the extraction script slices a covering store with a fresh extraction's
encoding, and [`test_data_reuse.py`](../../tests/test_data_reuse.py) pins equal `netcdf_content_sha256`. Keep
the sources out of rule `params` and the record out of the netCDF: either would change rerun decisions or
the WF3 seed.
Both pruners report until `--delete` is given:
[`prune_climate_store.py`](../../dev/scripts/prune_climate_store.py) lists stores the current configuration
no longer selects, and [`prune_series_cache.py`](../../dev/scripts/prune_series_cache.py) lists WF2 series
it no longer resolves. The [repository layout](dev-repository-layout.md#task-temporary-material) describes
the other report-only cleanup helpers.
