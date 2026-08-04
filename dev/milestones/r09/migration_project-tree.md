# Migration — project tree (R9)

Status: **DRAFT — blocked.** One finding invalidates a settled-framing decision of
`project-tree-design.md` v4 and needs an owner ruling before the map can be
completed. The map below is recorded as far as it is sound.

Date: 2026-08-04

Purpose: the old → new path map mandated by `naming.md` §7, and the artifact
inventory the design names as a precondition for a task brief. Consumed three
ways — by `semantic_tree_diff` to gate the migration, as the internal rename
record, and by the task brief.

## Method and sources of truth

| Source | Standing |
| --- | --- |
| `output:` declarations in the three Snakefiles | **Authoritative** for what the current code emits |
| Undeclared engine artifacts (hydromt, Wflow.jl, weathergenr) | Authoritative only when observed; not visible to `--dry-run` |
| `test_case/test_local` materialized tree (240 files) | **Cross-check only — NOT an inventory.** See below |
| `build_r07_path_map` in `semantic_tree_diff.py` | Template for the map's regex form; describes pre-R7 → R7, not R7 → R9 |

**The fixture is a mixed-era tree and must not be used as the inventory.** It
carries at least three eras at once: WF2 log parts from rules the v2 rework
merged away (`2.03_monthly_change/`, `monthly_stats_fut/`, `monthly_stats_hist/`),
both the batched and the pre-batch WF3 wflow log shapes
(`3.10_run_wflow/batch_N.log` *and* `3.10_run_wflow/rlz_<r>_cst_<c>.log`),
unmerged WF3 log parts sitting directly in `logs/` rather than under `_parts/`,
and **no `spatial/` subtree at all** — the ADR 0003 spatial artifacts postdate it.
Anything mapped from the fixture alone would bake those orphans into the contract.

---

## Finding 1 — BLOCKING: the project's `config/` is generated, not editable

The design's settled framing states *"`config/project.yml` is the editable
project source of truth"* and the tree labels `config/` **"editable project
source"**. Neither holds today.

`<project_dir>/config/` is written **in its entirety** by rule 1.01
`snapshot_config` → `blueearth_cst/model/copy_config_files.py`, which creates
`config/catalogs/`, `config/templates/`, and `config/observations/`
(`copy_config_files.py:252-261`) plus `config/runs/` and the immutable
digest-keyed bundle at `config/runs/model_creation/<digest>/`. These are
**provenance snapshots of inputs that live outside `project_dir`**: the editable
source of truth is the `snake_config_*.yml` passed via `--configfile`, which
lives in the toolbox, and the catalogs/templates/observations it references by
absolute path.

There is no `project.yml` anywhere, and nothing writes one.

**The collision.** `config/catalogs/` and `config/templates/` are the *same
paths* with *opposite semantics* in the two documents: v4 labels them per-basin
editable overrides; `copy_config_files.py` writes generated snapshots there.

**The ruling that unblocks the map is one question: where does the generated
config snapshot live under the v4 tree?** The snapshot is a real, sizeable
artifact set — `config/runs/snake_config_<workflow>.yml`,
`config/runs/model_creation/<digest>/`, `config/catalogs/`, `config/templates/`,
`config/observations/` — and the v4 tree has **no provenance root** to receive
it. The six roots are `config/`, `data/`, `models/`, `experiments/`, `logs/`,
`benchmarks/`, and under P4 none of them is a home for per-run generated
provenance at project scope.

Candidate homes, for the owner to rule between:

- **(A)** keep it under `config/` and drop the "editable project source" label
  from the subdirectories that are generated, distinguishing editable from
  snapshot *inside* `config/`;
- **(B)** give the snapshot its own project-scope root beside `logs/` and
  `benchmarks/`, which are already the project's generated run-record pair; or
- **(C)** treat it as a run record and place it under `logs/`, applying P7.

Every `config/` row in the map below is withheld pending this ruling.

**Two related notes, neither of which blocks the map:**

1. **The Q6 ruling is wrong in its conclusion, not just its reasoning.** I ruled
   that toolbox catalogs are "referenced, never copied". They are referenced as
   inputs **and copied as provenance**. v4's tree now carries the comment
   *"toolbox catalogs are referenced, not copied"* — that line is false in the
   accepted design of record and is corrected in v5.
2. **`config/project.yml` is new capability, not a relocation.** Nothing writes
   one; the source of truth is the `--configfile` in the toolbox. Introducing it
   moves config ownership from toolbox to project, touching `run_workflows.py`,
   the `--configfile` contract, and `suggest_experiment_name.py`. This is a
   **scope note on R9**, not a precondition for the map: it is a settled-framing
   decision and is not reopened here.

## Finding 2 — the spatial subtree is in flux

`dev/working/wf1-spatial-decoupling/` — P1 complete and gated; **P2 implemented,
Gate 2 review pending**. P2 changes `Snakefile_model_creation` and the model
build. The WF1 half of this map is therefore provisional and must be re-derived
after Gate 2 closes.

## Finding 3 — three tree-shape mismatches with v4

Not blocking, but the v4 tree is drawn from intent rather than from the emitted
set, and three parts do not correspond:

| v4 draws | Code emits | Issue |
| --- | --- | --- |
| `data/climate/historical/era5/` | `climate_historical/era5_<start>_<end>/` | The store is keyed by dataset **and window**. v4's "no window-ID directory" is a code change, not a move. The key must stay experiment-invariant either way. |
| `cmip6/timeseries/` | `cmip6/raw/` **and** `cmip6/scalar/` | Two directories, one target. Needs a split ruling or a rename of both. |
| `cmip6/change_factors/` | `cmip6/summary/cmip6_change_factors_{annual,monthly}.csv` | v4 draws a directory; the code writes two files inside `summary/`. |

---

## Path map — sound portions

`<P>` = `project_dir`. Withheld rows are marked. Ordered by destination root.

### → `models/hydrology/wflow/`

| Old | New |
| --- | --- |
| `<P>/hydrology_model/staticmaps.nc` | `models/hydrology/wflow/staticmaps.nc` |
| `<P>/hydrology_model/wflow_sbm.toml` | `models/hydrology/wflow/wflow_sbm.toml` |
| `<P>/hydrology_model/hydromt.log` | `models/hydrology/wflow/hydromt.log` |
| `<P>/hydrology_model/hydromt_data.yml` | `models/hydrology/wflow/hydromt_data.yml` |
| `<P>/hydrology_model/staticgeoms/*` | `models/hydrology/wflow/staticgeoms/*` |
| `<P>/hydrology_model/forcing/inmaps_historical.nc` | `models/hydrology/wflow/forcing/inmaps_historical.nc` |
| `<P>/hydrology_model/forcing/plots/*.png` | `models/hydrology/wflow/forcing/plots/*.png` |
| `<P>/hydrology_model/run_default/*` | `models/hydrology/wflow/run_default/*` |
| `<P>/hydrology_model/evaluation/*` | `models/hydrology/wflow/evaluation/*` |
| `<P>/hydrology_model/plots/basin_area.{png,pdf}` | `models/hydrology/wflow/plots/basin_area.{png,pdf}` |
| `<P>/hydrology_model/.model_built` | `models/hydrology/wflow/.model_built` |
| `<P>/hydrology_model/.outputs_configured` | `models/hydrology/wflow/.outputs_configured` |
| `<P>/config/generated/wflow_build_model_run.yml` | `models/hydrology/wflow/config/build_model.yml` |
| `<P>/config/generated/wflow_build_forcing_historical.yml` | `models/hydrology/wflow/config/build_historical_forcing.yml` |

The last two are the only `config/` rows not withheld: the design already routes
generated build YAML to the model root, which Finding 1 does not disturb.

### → `data/` — PROVISIONAL, see Finding 2

| Old | New |
| --- | --- |
| `<P>/spatial/spatial_maps.nc` | `data/spatial/spatial_maps.nc` |
| `<P>/spatial/spatial_catalog.yml` | `data/spatial/spatial_catalog.yml` |
| `<P>/spatial/spatial_report.yml` | `data/spatial/spatial_report.yml` |
| `<P>/spatial/location_registry.csv` | `data/spatial/location_registry.csv` |
| `<P>/spatial/geoms/{basins,catchments,locations,rivers,subbasins}.geojson` | `data/spatial/geoms/…` |
| `<P>/climate_historical/<store_key>/extract_historical.nc` | `data/climate/historical/era5/extract_historical.nc` † |
| `<P>/climate_historical/<store_key>/store_region.geojson` | `data/climate/historical/era5/store_region.geojson` † |
| `<P>/climate_historical/<store_key>/plots/source_*.png` | `data/climate/historical/era5/plots/source_*.png` † |
| `<P>/climate_historical/<store_key>/.guard_ok` | `data/climate/historical/era5/.guard_ok` † |
| `<P>/climate_projections/cmip6/raw/{series_key}.nc` | unresolved — Finding 3 |
| `<P>/climate_projections/cmip6/scalar/{series_key}.nc` | unresolved — Finding 3 |
| `<P>/climate_projections/cmip6/summary/*` | `data/climate/projections/cmip6/summary/*` |
| `<P>/climate_projections/cmip6/plots/*.png` | `data/climate/projections/cmip6/plots/*.png` |
| `<P>/climate_projections/cmip6/report.md` | **unplaced** — see gaps |

† depends on the window-key ruling in Finding 3.

`{series_key}` embeds verbatim CMIP model IDs (`NOAA-GFDL_GFDL-ESM4`) — tier-1,
never normalized by the naming rule.

### → `experiments/<id>/`

| Old | New |
| --- | --- |
| `weather_generator/output/rlz_<r>_cst_<c>.nc` | `climate/weathergenr/series/rlz_<r>_cst_<c>.nc` |
| `weather_generator/config/weathergen_config.yml` | `climate/weathergenr/config/weathergen_config.yml` |
| `weather_generator/_work/*` | `climate/weathergenr/_work/*` |
| `weather_generator/plots/*.png` | `climate/weathergenr/plots/*.png` |
| `weather_generator/output/{sim_dates,resampled_dates}.csv` | **unplaced** — see gaps |
| `hydrology_runs/rlz_<r>/config/cst_<c>.toml` | `hydrology/wflow/config/rlz_<r>_cst_<c>.toml` |
| `hydrology_runs/rlz_<r>/forcing/inmaps_cst_<c>.nc` | `hydrology/wflow/forcing/inmaps_rlz_<r>_cst_<c>.nc` |
| `hydrology_runs/rlz_<r>/output/cst_<c>.csv` | `hydrology/wflow/output/rlz_<r>_cst_<c>.csv` |
| `hydrology_runs/rlz_<r>/output/outstates_cst_<c>.nc` | `hydrology/wflow/output/outstates_rlz_<r>_cst_<c>.nc` |
| `hydrology_runs/rlz_<r>/config/log.txt` | **unplaced** — Wflow driver log, see gaps |
| `indicators/Qstats.csv` | `results/q_indicators.csv` |
| `indicators/basin.csv` | `results/basin_indicators.csv` |
| `indicators/RT_*.csv` | **deleted** — not migrated (v2 decision 3) |
| `data_catalog_climate_experiment.yml` | **unplaced** — generated catalog at the experiment root |
| `.project_consistency_ok` | `.project_consistency_ok` (unchanged) |
| `logs/*`, `benchmarks/*` | `logs/*`, `benchmarks/*` (unchanged) |
| `config/snake_config_climate_experiment.yml` | **withheld** — Finding 1 |
| `config/catalogs/*`, `config/*.yml` | **withheld** — Finding 1 |

### → project root

| Old | New |
| --- | --- |
| `<P>/logs/wf{1,2}_*.log` | unchanged |
| `<P>/logs/_parts/**` | unchanged (transient) |
| `<P>/benchmarks/wf{1,2}_benchmarks.md` | unchanged |
| `<P>/benchmarks/_parts/**` | unchanged (transient) |
| *(new)* | `logs/dag/`, `experiments/<id>/logs/dag/` — v4 decision Q4 |

---

## Artifacts the v4 design does not place

Each needs a home before the task brief:

1. `climate_projections/cmip6/report.md` — a WF2 human-readable summary.
2. `weather_generator/output/{sim_dates,resampled_dates}.csv` — generator
   products that are not per-member series; `series/` or `_work/`?
3. `experiments/<id>/data_catalog_climate_experiment.yml` — a generated catalog.
   The Q6 ruling says generated catalogs travel with their producer, which
   implies `climate/weathergenr/` or `climate/`, not the experiment root.
4. `hydrology_runs/rlz_<r>/config/log.txt` — written by the Wflow driver, not by
   a Snakemake `log:`. Undeclared. With the `rlz_<r>/` level gone it has no
   obvious home and would collide across members if left in `config/`.
5. `hydrology_model/{.model_built,.outputs_configured}` — build sentinels. The
   design places guard sentinels but not build sentinels.
6. `spatial/spatial_report.yml`, `location_registry.csv` — post-date the design;
   `data/spatial/` is drawn as `region.geojson` + `gauges.geojson` + `...`, and
   neither of those filenames exists. The real set is above.

## Orphans in the fixture — do NOT map

Present on disk, produced by no current rule. They must be pruned before any
reference snapshot, or the gate will compare against them:

- `logs/_parts/2.03_monthly_change/**`, `2.0N_monthly_stats_{fut,hist}/**`
- `experiments/experiment/logs/3.*.log` at the top level (pre-merge shape)
- `experiments/experiment/logs/3.10_run_wflow/rlz_<r>_cst_<c>.log` (pre-batch shape)
- `experiments/experiment/config/deltares_data.yml` (superseded by `config/catalogs/`)

`dev/scripts/prune_series_cache.py` covers the WF2 series class only; the log-part
orphans are not covered by it.

## Next steps

1. **Rule Finding 1** — where the generated config snapshot lives. This gates
   everything under `config/`.
2. Rule Finding 3's three mismatches (window key, `raw`/`scalar`, `change_factors`).
3. Place the six unplaced artifact classes.
4. Re-derive the WF1/spatial rows after Gate 2 closes (Finding 2).
5. Materialize a **clean** fixture from current code — the existing one cannot
   validate this map — and diff it against the completed map.
6. Encode the map as regex rules alongside `build_r07_path_map`.
