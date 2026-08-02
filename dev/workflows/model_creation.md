# Workflow: model_creation

Contract for `Snakefile_model_creation` (workflow 1). Format per
`dev/milestones/r01/modularity-contracts-design.md` §4. Records **current** behavior
(R3 opening act) — R3 is behavior-preserving, so this doc is the baseline
the R3 code commits are checked against, not a description of intended
change. Grounded in `Snakefile_model_creation`, `config/wflow_build_model.yml`,
`config/wflow_update_waterbodies.yml`, and `src/setup_gauges_and_outputs.py`.

## Owned config keys (`workflows.model_creation.*`)

- `wflow_outvars` — Wflow output variables to emit (default `['river discharge']`).
- `model_build_config` — path to the hydromt build config (default `{static_dir}/wflow_build_model.yml`).
- `waterbodies_config` — path to the reservoirs/lakes/glaciers update config (default `{static_dir}/wflow_update_waterbodies.yml`).
- `output_locations` — optional gauge-locations file; when set, adds `setup_gauges` (Q, P at gauges). Default `None`.
- `observations_timeseries` — optional observed-discharge file for `plot_results`. Default `None`.

## Reads from `shared`

- `shared.basin.region`, `shared.basin.resolution` — basin delineation + model resolution.
- `shared.historical_window.starttime`, `shared.historical_window.endtime` — forcing time range.
- `shared.clim_historical` — historical climate source (e.g. `era5`).

## Reads from `project`

- `project.project_dir` — output root (`basin_dir = {project_dir}/hydrology_model`).
- `project.static_dir` — location of the build/update config templates.
- `project.data_sources` — hydromt data-catalog YAML (passed to `hydromt build/update -d`).

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
  `1.NN_` prefix is the `W.NN` rule-numbering scheme (naming.md §9).

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
