# CMIP6 source readability

Last updated: 2026-11-05

Use this reference when changing WF2 acquisition or diagnosing rejected CMIP6 source combinations. The
generated [`cmip6_store_index.json`](../../config/catalogs/cmip6_store_index.json) inventories observed
sources; it does not guarantee that a source can be staged. The
[WF2 technical manual](../../docs/technical-manual/workflow-analyze-projections.md) owns source-quality
exclusions, including `THU/CIESM`, and the interpretation of missing coverage.

## 1. Grid geometry

HydroMT's regular raster accessor requires uniformly spaced coordinates. Ordered one-dimensional latitude
and longitude axes can remain readable when spacing is irregular. In
[`fetch_raw_slice`](../../blueearth_cst/projections/fetch_gcm_raw.py), only HydroMT's recognized regular-grid
failure triggers a second read of the same source without a bounding box, followed by index-space clipping to
the requested bounds. The normal bounded read remains the first attempt. HydroMT still handles renaming,
unit conversion, CRS, and nodata in both paths.

The fallback requires ordered one-dimensional axes; it is not an unstructured-mesh reader. A successful
coordinate probe does not prove that data can be staged. Missing values and time coverage have separate
admission checks. Use [`probe_cmip6_grids.py`](../../dev/scripts/probe_cmip6_grids.py) to inspect geometry and
[`stage_cmip6.py`](../../dev/scripts/stage_cmip6.py) to exercise actual staging.

## 2. Published-version multiplicity

The catalog URI can match multiple grid labels and published versions. For each member,
[`selected_pins`](../../blueearth_cst/projections/series_identity.py) selects the newest indexed version of
each variable on **one common grid**. Competing grid labels require an explicit choice in the supplied catalog
and index; lexical order must not silently choose a regridding. Missing pins retain the unindexed catalog
read. DAG preflight and fetch must apply the same selection policy, because publication histories differ by
member.

When all variables select the same grid and version, `pinned_uri` keeps a single HydroMT catalog read.
When versions differ on the common grid, the fetcher reads each requested variable separately through HydroMT,
using the same adapters, bounds, and acquisition window. Every requested variable, including a non-certified
one, then needs an indexed pin. The split path rejects unknown or unequal source calendars, unequal CRS,
dimension sizes or dimension coordinates, and empty or duplicate time axes. Variable-specific scalar
coordinates, such as temperature height, need not exist on precipitation. The merge uses exact coordinates;
it neither regrids nor interpolates time to reconcile sources.

The raw slice records the index inventory in `cst_source_paths` and the paths actually selected in
`cst_selected_source_paths`; split reads also log each selected URI. Index pins enter the series identity.
Shared-pin cache identities keep their existing recipe. Only differing-version selections add
`per_variable_sources` to the digest components, invalidating affected raw slices and derived series without
a global schema change. Keep actual-source provenance distinct from the full index inventory.

## Verification

[`test_series_identity.py`](../../tests/test_series_identity.py) covers pin selection and cache identity;
[`test_fetch_gcm_raw.py`](../../tests/test_fetch_gcm_raw.py) covers geometry-trigger recognition, exact merge
conditions, and time-axis checks. Inspect staged coverage and diagnostics before interpreting an absent
ensemble member as a scientific result. These checks describe the implemented contract; they do not establish
current remote-store availability or a general performance gain.
