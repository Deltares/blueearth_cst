# Workflow: model_creation

Contract for `Snakefile_model_creation` (workflow 1). Format per
`dev/milestones/r01/modularity-contracts-design.md` §4. Records current behavior
and is grounded in `Snakefile_model_creation`, the templates under
`config/templates/`, and the rule-called modules under `blueearth_cst/model/`
and `blueearth_cst/spatial/`.

## Owned config keys (`workflows.model_creation.*`)

- `wflow_outvars` — Wflow output variables to emit (default `['river discharge']`).
- `model_build_config` — path to the hydromt build config (default `{static_dir}/wflow_build_model.yml`).
- `waterbodies_config` — path to the reservoirs/lakes/glaciers update config (default `{static_dir}/wflow_update_waterbodies.yml`).
- `observations_timeseries` — optional observed-discharge file for `plot_results`. Default `None`.

## Reads from `shared`

- `shared.basin.region`, `shared.basin.resolution` — basin delineation + model resolution.
- `shared.basin.gauge_points` — optional canonical gauge/control-point file.
  The former `workflows.model_creation.output_locations` key is accepted for
  one compatibility release; conflicting populated values fail at parse time.
- `shared.basin.automatic_subbasins.max_count` — global automatic-fallback
  ceiling (default 20; valid range 1–99).
- `shared.basin.gauge_snap_tolerance_m` — point-to-river snapping tolerance
  (default 10,000 m).
- `shared.basin.river_uparea_km2` — analysis river threshold (default 32 km²).
- `shared.basin.spatial_sources.{rivers,lulc,lai,soil}` — catalog entries for
  the model-neutral thematic products.
- `shared.historical_window.starttime`, `shared.historical_window.endtime` — forcing time range.
- `shared.clim_historical` — historical climate source (e.g. `era5`).

## Reads from `project`

- `project.project_dir` — output root (`basin_dir = {project_dir}/hydrology_model`).
- `project.static_dir` — location of the build/update config templates.
- `project.data_sources` — hydromt data-catalog YAML (passed to `hydromt build/update -d`).

## Rule 1.17: engine-neutral spatial foundation

`prepare_spatial_maps` is a no-wildcard target and can be requested directly:

```powershell
snakemake prepare_spatial_maps -c 1 -s Snakefile_model_creation --configfile <config.yml>
```

It resolves every parent feature independently, snaps configured gauge/control
points to the analysis river network, and uses internal controls where present.
An outlet-only or absent control set selects deterministic automatic
partitioning for that parent. The configured `automatic_subbasins.max_count`
is one global ceiling shared between fallback parents. Incremental subbasins
do not overlap; the separate catchment layer contains each control point's full
contributing area and therefore may overlap or nest.

The analysis grid comes from `shared.basin.hydrography` at the requested
resolution (native or an integer upscale). Flow direction is ArcGIS D8, not a
Wflow LDD map. Elevation and slope use average resampling; LULC uses nearest;
LAI and soil variables use average. Each raster records its catalog source,
resampling where applicable, resolution, units, nodata, and CRS.

The rule's freshness boundary is the workflow config, the catalog YAML file(s),
and the optional gauge file, all declared as Snakemake inputs. A change to a
data file hidden behind an unchanged catalog URI is not detectable by
Snakemake; touch/update the catalog or force this rule when refreshing such a
source. The generated catalog uses relative URIs, so the complete `spatial/`
directory is portable as a unit.

At the current P1 gate this rule is independent of `create_model`: targeting it
does not schedule Wflow or create `hydrology_model/`. The accepted P2 adapter
will add the downstream dependency without moving Wflow constants or derived
parameter maps into this product.

The Gate 1 adapter proof selected a project-owned adapter over another
`setup_basemaps` call. In the pinned `hydromt_wflow` version,
`setup_basemaps` delineates from a hydrography source and would therefore
repeat P1. The proof instead reads only `spatial_catalog.yml`, converts the
neutral ArcGIS D8 map to Wflow LDD at the adapter boundary, loads the neutral
base layers through the public `staticmaps.set`/`geoms.set` component APIs,
and then uses public `setup_config`, `set_flwdir`, `setup_gauges`,
`setup_outlets`, and `write` methods. A write/reopen check preserved the P1
grid, subbasin IDs 101–120, and location IDs 101–120 and produced the standard
Wflow `staticmaps.nc`, `wflow_sbm.toml`, and `staticgeoms/region.geojson`
triplet. Phase 2 must implement and test this route rather than read the
original global hydrography again.

## Input contract (external data — catalog sources required in `data_sources`)

- **Build** (`wflow_build_model.yml`): `merit_hydro_ihu`, `merit_hydro_index`
  (basemaps + rivers), `rivers_lin2019_v1` (river geometry), `vito` (LULC),
  `modis_lai` (LAI), `soilgrids` (soil).
- **Waterbodies** (`wflow_update_waterbodies.yml`): `hydro_reservoirs` (GRanD),
  `jrc` (reservoir timeseries), `hydro_lakes` (HydroLAKES), `rgi` (glaciers).
  Any source may be legitimately absent for a basin — the
  `add_reservoirs_lakes_glaciers` rule catches per-method `NoDataException`.
- **Forcing**: `shared.clim_historical` source (e.g. `era5`) over the
  historical window.

## Output contract (by role — not all are `rule all` targets)

**Direct `rule all` targets** (named statically by this workflow's `rule all`):
- `{basin_dir}/evaluation/plots/hydro_wflow_1.png` (the run)
- `{basin_dir}/plots/basin_area.png` (the model)
- `{basin_dir}/forcing/plots/precip.png` (model inputs)
- `{project_dir}/climate_historical/<key>/plots/source_{precip,temp,pet}.png`
  (R07 B4 — source-grid figures from the shared store; produced with no model)
- `{project_dir}/config/runs/snake_config_model_creation.yml` (verbatim snake-config snapshot)
- `{project_dir}/spatial/spatial_catalog.yml` (representative target for the
  complete rule-1.17 spatial product)

*R07 retired the project-level `plots/` tree: figures now attach to what they
DEPICT (P1), so they sit beside the subtree whose artifacts they show.*

**Downstream-contract artifacts** (produced by intermediate rules; consumed by
workflows 2/3; not in this `rule all`):
- `{basin_dir}/staticmaps.nc`
- `{basin_dir}/staticgeoms/region.geojson`
- `{basin_dir}/staticgeoms/outlets.geojson`
- `{basin_dir}/wflow_sbm.toml`
- `{project_dir}/climate_historical/wflow_data/inmaps_historical.nc`
- `{basin_dir}/run_default/output.csv`

**Spatial-foundation contract** (`blueearth-cst-spatial-v1`):

- `{project_dir}/spatial/spatial_maps.nc`
- `{project_dir}/spatial/geoms/{basins,subbasins,catchments,rivers,locations}.geojson`
- `{project_dir}/spatial/location_registry.csv`
- `{project_dir}/spatial/spatial_catalog.yml`
- `{project_dir}/spatial/spatial_report.yml`

The raster, vector, and registry identifiers are relational: basin IDs are
`1..N`; subbasin IDs use `basin_id * 100 + local_number`; each primary location
inherits its subbasin ID as `wflow_id`. The generated `spatial_catalog.yml`
exposes every artifact through HydroMT without containing Wflow configuration.

**Side-effect artifacts** (bookkeeping / traceability; no downstream reader):
- `{basin_dir}/staticgeoms/reservoirs_lakes_glaciers.txt` — waterbodies sentinel.
- `{basin_dir}/staticgeoms/outlet_index.csv` — position→subcatchment-ID map (R3 §4).
- `{project_dir}/logs/_parts/1.NN_{rule}.log`, `{project_dir}/benchmarks/_parts/1.NN_{rule}.tsv`
  (per-rule logs AND benchmarks live under `_parts/`; `gather_logs` (1.16) merges
  the logs into one `logs/wf1_model_creation.log` via
  `blueearth_cst/shared/merge_logs.py` and then **deletes** the parts, and
  `gather_benchmarks` (1.14) merges the benchmarks into one
  `benchmarks/wf1_benchmarks.md` (Markdown table, `rule` column + `TOTAL` row)
  via `merge_benchmarks.py`. All three workflows follow this scheme — WF2 2.07,
  WF3 3.13.)
  — ephemeral run artifacts (R3 §6); not manifest targets, not committed. The
  `1.NN_` prefix is the `W.NN` rule-numbering scheme (naming.md §9). The
  spatial rule uses `1.17_prepare_spatial_maps` during the P1 transition.

## Downstream consumers

- **Workflow 2** (`Snakefile_climate_projections`) reads
  `staticgeoms/region.geojson` (as an `ancient(...)` input to
  `monthly_stats_hist`/`_fut`).
- **Workflow 3** (`Snakefile_climate_experiment`) reads the built model,
  its `wflow_sbm.toml`, and the forcing layout.

## Outlet-naming convention (R3 §4 decision)

Outlet stations use the **positional `wflow_{1..N}`** convention (not the
basin-derived subcatchment IDs hydromt_wflow 1.x assigns). The real
subcatchment IDs are preserved in `staticgeoms/outlet_index.csv`
(`station_name`, `subcatchment_id`, `x`, `y`) — emitted on every run — and
surfaced in plot titles as a human aid. Rationale: static `rule all` /
manifest paths must be basin-independent (see design §4). The CSV column
`Q_outlets` is upstream hydromt_wflow vocabulary, kept as-is.

## `wflow_outvars` output set (known discrepancy — documented, not fixed in R3)

- Canonical `config/snake_config_model_test.yml`: `['river discharge']` — the
  minimal set (outlet Q only).
- Pytest fixture `tests/snake_config_model_test.yml`: all six mapped variables
  (`river discharge`, `precipitation`, `overland flow`,
  `actual evapotranspiration`, `groundwater recharge`, `snow`).

The two seed configs carry different output sets. Enabling the complete plot
suite (climate panels in `plot_results.py`) would require the fuller set but
**moves the baseline**, so it is a followup, not an R3 change (design §7.3).

## `wflow_outvars` → CSDMS mapping (`WFLOW_VARS`, `setup_gauges_and_outputs.py`)

Semantic name → Wflow.jl 1.x CSDMS Standard Name → reporting unit. Units are
the conventional Wflow 1.x output units; the header/param/unit pairings are
confirmed in the R3 §7.2 gauges audit (commit 7).

| Semantic name              | CSDMS name                                               | Unit      |
| -------------------------- | -------------------------------------------------------- | --------- |
| river discharge            | `river_water__volume_flow_rate`                          | m³ s⁻¹    |
| precipitation              | `atmosphere_water__precipitation_volume_flux`            | mm Δt⁻¹   |
| overland flow              | `land_surface_water__volume_flow_rate`                   | m³ s⁻¹    |
| actual evapotranspiration  | `land_surface__evapotranspiration_volume_flux`           | mm Δt⁻¹   |
| groundwater recharge       | `soil_water_saturated_zone_top__net_recharge_volume_flux`| mm Δt⁻¹   |
| snow                       | `snowpack_liquid_water__depth`                           | mm        |

`river discharge` is always emitted at outlets (`setup_outlets`, header `Q`);
`precipitation` is added at gauges when `output_locations` is set (header `P`);
remaining entries become basin-average timeseries (`{name}_basavg`, mean
reducer over `subcatchment`).
