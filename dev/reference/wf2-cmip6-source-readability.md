# CMIP6 source readability

Last updated: 2026-10-30

Use this reference when changing WF2 fetches or diagnosing rejected source combinations.
The generated `config/catalogs/cmip6_store_index.json` is the current source inventory.
Historical crawl counts and staging outcomes are retained separately under R08.

For source-quality policy, see the
[WF2 technical manual](../../docs/technical-manual/workflow-analyze-projections.md).
CIESM source measurements and exclusion
checks (private-checkout-only citation: `../records/validation/ciesm-source-quality-2026-10-30.md`) retain the
evidence for rejecting that model before acquisition.

## 1. Grid geometry

HydroMT's regular raster accessor requires uniformly spaced coordinates. Ordered one-dimensional
latitude/longitude axes can still be readable when their spacing is irregular.
`blueearth_cst/projections/fetch_gcm_raw.py::fetch_raw_slice` handles the recognized regular-grid
failure by reopening the same source without a bbox and selecting the bbox in index space.

Keep rename, unit conversion, CRS and nodata handling with HydroMT. The fallback is not a general
unstructured-mesh reader, and successful coordinate probing does not establish successful data staging.
Time coverage and missing data remain separate admission conditions.

## 2. Published-version multiplicity

`blueearth_cst/projections/series_identity.py::selected_pins` selects the newest indexed version per
variable on one common grid. Shared versions retain the single HydroMT catalog read through `pinned_uri`.
Differing versions use separate HydroMT reads with the same adapters, bounds and acquisition window.
The split path requires equal dimension coordinates, CRS and source calendars before an exact merge;
it rejects duplicate or empty time axes. Variable-specific scalar metadata, such as temperature height,
need not exist on precipitation. No regridding or temporal interpolation reconciles incompatible sources.

Competing grid labels require an explicit common-grid choice in a supplied catalog and index; neither the
preflight nor fetcher chooses a regridding by lexical order. Missing pins retain the unindexed fallback.
On the split path every requested variable needs an indexed pin, including non-certified variables.

Version pinning is per member because members can have different publication histories.
Keep the selection in the store-index consumer rather than inventing one catalog-wide version.
The DAG preflight and fetch path must use the same selection policy.

Raw files retain the index inventory in `cst_source_paths` and record the actual selected paths in
`cst_selected_source_paths`; split reads also report each selected URI. Existing shared-pin cache identities
remain unchanged. Only differing-version sources gain the `per_variable_sources` digest component,
invalidating their old raw slices and derived series without a global schema bump or rebuild.

## Verification and evidence

`dev/scripts/probe_cmip6_grids.py` examines coordinate geometry. `dev/scripts/stage_cmip6.py` exercises
readability through actual staging. `tests/test_series_identity.py` owns version-selection checks.
Inspect fetched coverage and diagnostics before interpreting absent ensemble members scientifically.

The per-variable pinning verification (private-checkout-only citation:
`../records/validation/cmip6-per-variable-pinning-2026-10-22.md`) retains live availability, independent value
checks, cache compatibility and single-sample before/after timings for TaiESM1 and FIO-ESM-2-0. Those
measurements establish the tested cases, not a general speedup.

Readability measurements (private-checkout-only citation:
`../records/milestones/r08/cmip6-readability-measurements.md`) and the historical inventory
(private-checkout-only citation: `../records/milestones/r08/wf2-cmip6-store-inventory.md`) describe particular
crawls and verification runs. Their model counts are not current acceptance limits or availability claims.
