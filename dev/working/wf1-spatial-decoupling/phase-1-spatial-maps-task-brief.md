# Task Brief — P1 engine-neutral spatial maps

### Context

- Canonical ruleset: repository-root `AGENTS.md`. Program-level constraints and gates are in [the master brief](master-task-brief.md).
- Today rule 1.03 `create_model` calls `hydromt build wflow_sbm`; domain delineation, hydrography, rivers, land use, LAI, soils, Wflow parameters, and TOML creation are therefore coupled.
- `shared.basin.region`, `resolution`, `hydrography`, and `basin_index` already define the model-free domain used by the shared climate store. Preserve these as the authoritative parent-basin inputs.
- The configured basin specification may resolve to one or several parent basins. Gauge points, when provided, control subbasin delineation; automatic river-network partitioning is the fallback.
- The automatic fallback has a hard global default ceiling of 20 generated subbasins. Gauge-derived subbasins are user-directed and are never silently pruned by this ceiling.

### Goal

Add a targetable Snakemake rule named `prepare_spatial_maps` that creates analysis-ready, model-neutral basin, hydrography, river, land-cover, vegetation, and soil products without creating or importing a Wflow model.

### Non-goals

- No `WflowSbmModel`, `hydromt build wflow_sbm`, `wflow_sbm.toml`, forcing, states, run outputs, calibration, or Wflow constants.
- No Wflow-specific land-use parameter tables, soil pedotransfer parameters, routing parameters, or engine-standard variable aliases.
- No redesign of Workflow 2/3 climate calculations or plot-output enumeration.
- No silent compatibility fallback that reads gauge points from two conflicting config keys.

### Allowed scope

**Permitted:** `Snakefile_model_creation`; a new `blueearth_cst/spatial/**` package; `blueearth_cst/shared/snake_utils.py` only for genuinely shared parsing/validation; `config/workflows/**`; observation templates under `config/templates/observations/**`; relevant `tests/**`; `dev/workflows/model_creation.md`; this brief directory.

**Approval-gated:** catalog edits under `config/catalogs/**`; changes to Workflow 2/3; `dev/baseline/manifest.json`; changing an existing public output path rather than adding the new `spatial/` contract.

**Forbidden / generated:** `.pixi/**`, `pixi.lock`, Julia manifests, vendored packages/docs, `test_case/**` outputs, and external production project directories.

### Required changes (checklist)

1. Extend the sectioned config without adding an explicit strategy switch:

   ```yaml
   shared:
     basin:
       region: "{'subbasin': [9.666, 0.4476], 'uparea': 100}"
       resolution: 0.00833
       hydrography: merit_hydro_ihu
       basin_index: merit_hydro_index
       gauge_points: null
       automatic_subbasins:
         max_count: 20
       spatial_sources:
         rivers: rivers_lin2019_v1
         lulc: vito
         lai: modis_lai
         soil: soilgrids
   ```

   Keep optional defaults compatible with the shipped catalog/template. Validate types and positive bounds at parse time. Migrate `workflows.model_creation.output_locations` to `shared.basin.gauge_points` atomically with all consumers; either provide a documented one-release compatibility path with conflict detection or make the breaking migration explicit. Do not silently prefer one when both are populated.

2. Resolve delineation independently for each parent basin:

   - Preserve separate parent features when `region` resolves several basins; otherwise create one parent basin.
   - Snap configured points to the river network using one documented tolerance.
   - Use distinct internal gauge points as subbasin control locations. An outlet-only point creates no extra subdivision; when no usable internal gauge remains for a parent basin, use automatic partitioning for that basin.
   - Reject duplicate points that snap to the same control cell; aliases are out of scope.
   - For automatic partitioning, derive candidates from the river network and deterministically coarsen the partition until the total number of automatically generated subbasins is `<= max_count`. Allocate at least one unit per fallback parent basin; fail if the number of such parents alone exceeds the ceiling.
   - Write both full contributing `catchments` for control points, which may overlap or nest, and non-overlapping incremental `subbasins`.

3. Implement the identity contract before writing spatial layers:

   - Parent basin: integer `basin_id`, code `B001`, explicit name or fallback `basin_001`.
   - Subbasin: `subbasin_id = basin_id * 100 + local_subbasin_number`, code `B001-S01`; number downstream-to-upstream, breaking branch ties by decreasing upstream area and then snapped grid row/column. Validate the local number range and positive-int32 compatibility.
   - Primary outlet/control location: `wflow_id = subbasin_id`, code `B001-S01-L01`.
   - Additional non-controlling locations, if supported by the input, use `wflow_id = 1_000_000 + subbasin_id * 100 + local_location_number`; validate global uniqueness and positive-int32 range.
   - Gauge-derived subbasin names inherit the supplied `station_name`; automatic names use `auto_<NN>`. Names are labels, never join keys.
   - Treat input `wflow_id` as optional. Generate it when absent; if supplied, require exact agreement with the resolved registry. The current single-basin Gabon IDs 101–104 are within the scheme; preserve their exact station assignments only when the resolved hierarchy agrees, otherwise emit an explicit migration crosswalk.

4. Produce an explicit spatial contract under `{project_dir}/spatial/`:

   - `spatial_maps.nc`: DEM/elevation, slope, flow direction in a documented model-neutral encoding, upstream area/flow accumulation, `basin_id`, `subbasin_id`, river mask/order, raw or analysis-ready LULC, LAI/vegetation, and soil layers with CRS, units, nodata, source, and resolution metadata.
   - `geoms/basins.geojson`, `geoms/subbasins.geojson`, `geoms/catchments.geojson`, `geoms/rivers.geojson`, and `geoms/locations.geojson`.
   - `location_registry.csv` containing at least `basin_id`, `basin_code`, `basin_name`, `subbasin_id`, `subbasin_code`, `subbasin_name`, `location_id`, `location_code`, `station_name`, `wflow_id`, `location_role`, original coordinates, and snapped coordinates.
   - A generated HydroMT-compatible `spatial_catalog.yml` that exposes the neutral products to model adapters without embedding Wflow configuration.

5. Wire `prepare_spatial_maps` with explicit file outputs, log, benchmark, tracked catalog/config inputs, and the gauge file as an `input:` when configured. A no-wildcard rule invocation must support spatial-only execution. Do not hide the contract behind a directory-only sentinel.

6. Update the clean config template, observation template/readme, Workflow 1 contract, and tests. Record source-grid/resampling choices and the catalog freshness boundary.

### Commit plan

| Subject | Paths | Invariant preserved |
|---|---|---|
| `feat(spatial): define the neutral config and identity contract` | config templates, validation helpers, unit tests, contract docs | New keys and all readers agree atomically; no ambiguous dual gauge source |
| `feat(spatial): build the engine-neutral spatial products` | `blueearth_cst/spatial/**`, geospatial tests | Same inputs produce the same grid, partition, IDs, and registry independent of input row order |
| `feat(wf1): add the prepare_spatial_maps rule` | `Snakefile_model_creation`, DAG tests, workflow contract | Spatial-only target is independently runnable and contains no Wflow edge |

### Progress

- [x] Reconciled and landed the overlapping `feat/wf1-improvements` branch.
- [x] Define and validate the neutral config and deterministic identity contract.
- [x] Build the engine-neutral spatial products.
- [x] Add and validate the `prepare_spatial_maps` rule.
- [x] Produce the adapter proof and pause at master Gate 1.

### Validation

1. **Per edit — narrow:** unit tests for config parsing, point snapping, duplicate rejection, per-basin fallback selection, automatic ceiling, deterministic ordering, identity generation, and registry/schema validation.
2. **Per rule edit — DAG:** `pixi run pytest tests/test_cli.py`; then `pixi run snakemake prepare_spatial_maps -c 1 -s Snakefile_model_creation --configfile tests/snake_config_model_test.yml --dry-run`.
3. **Behavioural tests:** shuffle gauge input rows and assert identical IDs and geospatial products; test single basin, multiple parent basins, internal gauges, outlet-only gauge, no gauges, and an automatic candidate network exceeding 20.
4. **Geospatial integration — once after the rule stabilizes:** run the spatial target in a clean dedicated project. Check CRS, bounds, resolution, nodata, units, flow-direction validity, monotonic downstream accumulation, raster/vector ID agreement, non-overlapping incremental subbasins, and contributing-catchment containment.
5. **Full gate — phase end:** `pixi run pytest tests/` plus Workflow 1 dry-run.

Falsifiers:

- Spatial independence is disproved if the target schedules `build_wflow_model`, creates `hydrology_model/`, writes a TOML, or imports `hydromt_wflow`.
- The automatic ceiling is disproved by any fallback result with more than 20 automatically generated subbasins.
- Determinism is disproved if row-order changes alter IDs, codes, or geometry for identical inputs.
- Model neutrality is disproved by Wflow parameter constants, Wflow-specific LDD names, or pedotransfer-derived Wflow parameter maps in `spatial_maps.nc`.

### Acceptance criteria

- `prepare_spatial_maps` completes with Wflow absent and creates every declared spatial output.
- Basin/subbasin/location IDs are deterministic, unique, positive-int32-compatible where Wflow will consume them, and traceable through `location_registry.csv`.
- Gauge information takes precedence per parent basin; automatic partitioning is the fallback and never exceeds its default global ceiling of 20.
- Spatial rasters and vectors agree on grid, CRS, domain, and IDs; required metadata is present.
- The generated catalog passes the applicable HydroMT validation and contains no Wflow model configuration.
- Roll back or stop at Gate 1 if the output cannot be exposed through a documented HydroMT-compatible source contract without Wflow-specific contamination.

### Output requirements

- Code, tests, config examples, and updated Workflow 1 contract.
- A short phase report listing output schemas, resolved delineation path per parent basin, automatic/gauge subbasin counts, ID ranges, commands run, and any data/resampling delta.
- **Results delta:** describe geometry, grid, or source differences from the current Wflow-produced basemaps; unexplained differences block P2.

### Task constraints

- Use `xarray`/`rioxarray`, `geopandas`, HydroMT, and flow-direction utilities already present in the locked environment; add no dependency without approval.
- Use catalog entry names rather than hardcoded source paths.
- Use explicit CRS transformations and preserve original plus snapped coordinates.
- Keep raw/analysis-ready thematic layers distinct from model parameter maps.
- Stop at master Gate 1 after P1 and the adapter proof; do not begin the Wflow implementation automatically.
