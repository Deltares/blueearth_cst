# Phase 1 report — engine-neutral spatial foundation

## Result

Phase 1 is complete and paused at master Gate 1. The no-wildcard
`prepare_spatial_maps` target writes the versioned
`blueearth-cst-spatial-v1` contract without scheduling or importing Wflow. A
separate adapter proof established that the pinned HydroMT-Wflow version can
consume this contract through public APIs without delineating from the original
global hydrography again.

## Written contract

The rule declares and validates these files under `{project_dir}/spatial/`:

- `spatial_maps.nc`
- `geoms/basins.geojson`
- `geoms/subbasins.geojson`
- `geoms/catchments.geojson`
- `geoms/rivers.geojson`
- `geoms/locations.geojson`
- `location_registry.csv`
- `spatial_catalog.yml`
- `spatial_report.yml`

`spatial_maps.nc` contains ArcGIS D8 `flow_direction`, cell-count
`flow_accumulation`, `upstream_area`, `cell_area`, `river_mask`, `river_order`,
`elevation`, `slope`, `basin_id`, `subbasin_id`, and namespaced land-cover,
monthly LAI, and soil variables. Every raster has CRS, nodata, units, source,
and resolution metadata. Vectors are valid EPSG:4326 geometries. The location
registry carries the basin, subbasin, location, station, Wflow, role, original
coordinate, and snapped-coordinate join fields required by the brief.

The generated `spatial_catalog.yml` exposes the raster dataset, registry, and
five vector layers through relative URIs. Every entry was reopened through
HydroMT during post-write validation; the catalog contains no Wflow model
configuration.

## Gabon integration result

The clean integration run used the local Deltares catalog and the configured
Gabon outlet, with no gauge-point file. The resolved path was therefore the
automatic fallback:

| Property | Result |
|---|---|
| Parent basins | 1 (`basin_id` 1) |
| Delineation method | automatic |
| Automatic subbasins | 20 (configured global ceiling: 20) |
| Gauge-derived subbasins | 0 |
| Subbasin IDs | 101–120 |
| Locations / Wflow IDs | 20 / 101–120 |
| Grid shape | 16 × 24 |
| CRS | EPSG:4326 |
| Resolution | 0.0083333333333° |
| Bounds | 9.65833333316084, 0.349999999932635, 9.858333333160658, 0.483333333266074 |

Post-write checks accepted only ArcGIS D8 codes
`{0,1,2,4,8,16,32,64,128}`, verified non-decreasing downstream flow
accumulation, exact raster/vector/registry ID agreement, non-overlapping
incremental subbasins, and containment of every incremental unit by its full
contributing catchment.

## Result delta

The P1 grid shape, CRS, resolution, and bounds exactly match the existing Gabon
Wflow `staticmaps.nc`. Differences are intentional and belong at the model
adapter boundary:

- P1 stores model-neutral ArcGIS D8; Wflow stores LDD.
- P1 stores raw or analysis-ready thematic layers; Wflow derives its
  physics-specific parameter maps and constants later.
- P1 assigns deterministic basin/subbasin/location identities and writes them
  explicitly in maps, vectors, and the registry.

No unexplained domain or alignment delta remains. The integration run itself
found and fixed decoded nodata handling, excess buffered cells, a final
one-cell grid-alignment border, and invalid self-touching polygon output before
this report was written.

## Gate 1 adapter proof

The proof instantiated `WflowSbmModel` using only the generated P1 catalog and
products. A project-owned adapter used public APIs to:

1. convert P1 ArcGIS D8 to Wflow LDD;
2. map P1 flow, subbasin, upstream-area, stream-order, cell-area, elevation,
   slope, and river-mask layers through `staticmaps.set`;
3. map P1 geometries through `geoms.set`;
4. call `setup_config`, `set_flwdir`, `setup_gauges`, `setup_outlets`, and
   `write`;
5. reopen the written Wflow model and compare its grid and IDs with P1.

The reopened model retained the exact P1 grid, subcatchment IDs 101–120, gauge
map IDs 101–120 (`gauges_locations`), and outlet ID 101. It wrote
`staticmaps.nc`, `wflow_sbm.toml`, and `staticgeoms/region.geojson`.

This selects the project-owned adapter route for Phase 2. Calling
`setup_basemaps` is rejected because the pinned implementation would delineate
again from a hydrography source and would not demonstrate consumption of P1.

## Validation record

- Focused spatial tests: 45 passed.
- Rule/DAG and workflow CLI tests: 15 passed.
- Full repository suite: 1,004 passed, 31 skipped, 1 expected xfail.
- Targeted dry-run: exactly one `prepare_spatial_maps` job and no Wflow edge.
- Workflow 1 dry-run: passed with 17 jobs, including the independent P1 rule.
- Real spatial target: succeeded and wrote all nine declared products.
- Public-API adapter proof: succeeded, wrote and reopened the Wflow triplet.

Successful real runs may display the repository's known benign empty
`Error in sys.excepthook` messages during Windows GDAL/rasterio interpreter
shutdown; the process exits zero and all products validate.

## Gate decision

Approve Phase 2 only if the adapter route and the intentional result delta
above are accepted. Phase 2 must consume the declared P1 files, preserve their
grid and identities, derive Wflow-only parameters/constants downstream, and
must not rerun basin delineation from the original hydrography source.

## Phase 2 correction

Gate 1 was approved. The first full Wflow runtime check then showed that the
automatic area partition could select subbasin outlets below the configured
`river_mask` threshold: all twenty IDs were spatially valid, but fifteen were
inactive for Wflow river discharge. Automatic outlet selection is now bounded
by both `automatic_subbasins.max_count` and the P1 river mask. Re-running the
same integration case therefore produces five subbasins/primary locations
(IDs 101–105), all on active river cells. This narrows the earlier automatic
fallback contract without changing gauge-driven partitions.
