# Migration — project tree (R9)

Status: **COMPLETE** (2026-08-04). All three findings resolved and every artifact
class placed. The map is ready to be encoded as regex rules and to feed the task
brief.

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

## Finding 1 — RULED: the project's `config/` is generated, not editable

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

**RULED 2026-08-04 — option (A): the snapshot stays under `config/`**, with
editable and generated subtrees distinguished inside it. Recorded in
`project-tree-design.md` v6, which also restates P4 from *separate* to
*distinguishable*. The question was: where does the generated config snapshot
live under the v4 tree? The snapshot is a real, sizeable
artifact set — `config/runs/snake_config_<workflow>.yml`,
`config/runs/model_creation/<digest>/`, `config/catalogs/`, `config/templates/`,
`config/observations/` — and the v4 tree has **no provenance root** to receive
it. The six roots are `config/`, `data/`, `models/`, `experiments/`, `logs/`,
`benchmarks/`, and under P4 none of them is a home for per-run generated
provenance at project scope.

Options considered:

- **(A)** keep it under `config/`, distinguishing editable from snapshot inside
  it — **adopted**;
- **(B)** its own project-scope root beside `logs/` and `benchmarks/`;
- **(C)** under `logs/`, applying P7.

**Why (A).** `config/runs/snake_config_model_creation.yml` is a declared `input:`
of WF3's rule 3.00b drift guard (`Snakefile_climate_experiment:210, 290`), so the
snapshot is a consumed cross-workflow contract artifact, not an archive. (A) is
the only option that leaves that path untouched. (B) a `provenance/` root would
break it, add a seventh root, and leave project and experiment scopes disagreeing
about where the same artifact class lives. (C) `logs/` is disqualified outright:
it is what a user deletes to reclaim space, and its parts are merged-then-deleted
by design, while this bundle is immutable, retained, and read by a downstream
workflow.

The `config/` rows are mapped below. They are almost all **identity** — which is
the point: the cheapest correct answer was the one that moved nothing.

**Two related notes, neither of which blocks the map:**

1. **The Q6 ruling is wrong in its conclusion, not just its reasoning.** I ruled
   that toolbox catalogs are "referenced, never copied". They are referenced as
   inputs **and copied as provenance**. v4's tree now carries the comment
   *"toolbox catalogs are referenced, not copied"* — that line was false in the
   accepted design of record; corrected at v5 and re-ruled at v6.
2. **`config/project.yml` is new capability, not a relocation.** Nothing writes
   one; the source of truth is the `--configfile` in the toolbox. Introducing it
   moves config ownership from toolbox to project, touching `run_workflows.py`,
   the `--configfile` contract, and `suggest_experiment_name.py`. This is a
   **scope note on R9**, not a precondition for the map: it is a settled-framing
   decision and is not reopened here.

## Finding 2 — WITHDRAWN: the spatial subtree is settled

Recorded as a blocker on the strength of `master-task-brief.md`'s phase index,
which read *"P2 — implemented; Gate 2 review pending"*. **That line was stale.**
Gates 2 and 3 were both approved and `feat/wf1-spatial-decoupling` was merged to
`main` on 2026-08-02 (`29ccde9`); `ad9702d` closed the gates in
`phase-2-report.md` and `DEVLOG.md` but never updated the index. The branch is
deleted and rules 1.02 `prepare_spatial_maps` and 1.03 `build_wflow_model` are on
`main`.

The WF1 rows in this map were derived from `main`, so they already reflect the
landed work and are **final, not provisional**. The nine P1 products the report
names correspond exactly to the `data/spatial/` rows below: five vector layers
under `geoms/`, plus `location_registry.csv`, `spatial_catalog.yml`,
`spatial_maps.nc`, and `spatial_report.yml`.

The index line has been corrected.

## Finding 3 — RULED: three tree shapes did not match what the code emits

All three resolved **toward the code** at design v7, so none costs an
implementation change. Every one turned out to encode a prior decision, which is
what design principle P9 now generalises.

| v4 drew | Code emits | Ruling |
| --- | --- | --- |
| `data/climate/historical/<source>_<window>/` | `climate_historical/<source>_<window>/` | **Key kept** — it is a cache key (P3-1 §4), not multi-window support. Framing reworded. New obligation: prune orphaned store dirs. |
| `cmip6/timeseries/` | `cmip6/raw/` **and** `cmip6/scalar/` | **Both kept.** Two tiers of one identity; `scalar/` over `series/` is R8 ruling S8-03; `prune_series_cache.py` is keyed to the grammar. |
| `cmip6/change_factors/` | two files in `cmip6/summary/` | **Kept in `summary/`.** A directory for two files violates P5. |

---

## Path map — sound portions

`<P>` = `project_dir`. Ordered by destination root.

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
| `<P>/hydrology_model/.model_built` | `models/hydrology/wflow/.model_built` (sentinel rule) |
| `<P>/hydrology_model/.outputs_configured` | `models/hydrology/wflow/.outputs_configured` |
| `<P>/config/generated/wflow_build_model_run.yml` | `models/hydrology/wflow/config/build_model.yml` |
| `<P>/config/generated/wflow_build_forcing_historical.yml` | `models/hydrology/wflow/config/build_historical_forcing.yml` |

The last two are also listed under `config/` below, since that is where they come
from; the design routes generated build YAML to the model root.

### → `data/`

| Old | New |
| --- | --- |
| `<P>/spatial/spatial_maps.nc` | `data/spatial/spatial_maps.nc` |
| `<P>/spatial/spatial_catalog.yml` | `data/spatial/spatial_catalog.yml` |
| `<P>/spatial/spatial_report.yml` | `data/spatial/spatial_report.yml` |
| `<P>/spatial/location_registry.csv` | `data/spatial/location_registry.csv` |
| `<P>/spatial/geoms/{basins,catchments,locations,rivers,subbasins}.geojson` | `data/spatial/geoms/…` |
| `<P>/climate_historical/<store_key>/extract_historical.nc` | `data/climate/historical/<source>_<window>/extract_historical.nc` † |
| `<P>/climate_historical/<store_key>/store_region.geojson` | `data/climate/historical/<source>_<window>/store_region.geojson` † |
| `<P>/climate_historical/<store_key>/plots/source_*.png` | `data/climate/historical/<source>_<window>/plots/source_*.png` † |
| `<P>/climate_historical/<store_key>/.guard_ok` | `data/climate/historical/<source>_<window>/.guard_ok` † |
| `<P>/climate_projections/cmip6/raw/{series_key}.nc` | `data/climate/projections/cmip6/raw/{series_key}.nc` |
| `<P>/climate_projections/cmip6/scalar/{series_key}.nc` | `data/climate/projections/cmip6/scalar/{series_key}.nc` |
| `<P>/climate_projections/cmip6/summary/*` | `data/climate/projections/cmip6/summary/*` |
| `<P>/climate_projections/cmip6/plots/*.png` | `data/climate/projections/cmip6/plots/*.png` |
| `<P>/climate_projections/cmip6/report.md` | `data/climate/projections/cmip6/report.md` |

† the store key is retained; see Finding 3.

`{series_key}` embeds verbatim CMIP model IDs (`NOAA-GFDL_GFDL-ESM4`) — tier-1,
never normalized by the naming rule.

### → `experiments/<id>/`

| Old | New |
| --- | --- |
| `weather_generator/output/rlz_<r>_cst_<c>.nc` | `climate/weathergenr/output/rlz_<r>_cst_<c>.nc` |
| `weather_generator/config/weathergen_config.yml` | `climate/weathergenr/config/weathergen_config.yml` |
| `weather_generator/_work/*` | `climate/weathergenr/_work/*` |
| `weather_generator/plots/*.png` | `climate/weathergenr/plots/*.png` |
| `weather_generator/output/{sim_dates,resampled_dates}.csv` | `climate/weathergenr/output/…` (identity) |
| `hydrology_runs/rlz_<r>/config/cst_<c>.toml` | `hydrology/wflow/config/rlz_<r>_cst_<c>.toml` |
| `hydrology_runs/rlz_<r>/forcing/inmaps_cst_<c>.nc` | `hydrology/wflow/forcing/inmaps_rlz_<r>_cst_<c>.nc` |
| `hydrology_runs/rlz_<r>/output/cst_<c>.csv` | `hydrology/wflow/output/rlz_<r>_cst_<c>.csv` |
| `hydrology_runs/rlz_<r>/output/outstates_cst_<c>.nc` | `hydrology/wflow/output/outstates_rlz_<r>_cst_<c>.nc` |
| `hydrology_runs/rlz_<r>/config/log.txt` | `hydrology/wflow/output/rlz_<r>_cst_<c>.log` — **requires a code change**, see below |
| `indicators/Qstats.csv` | `results/q_indicators.csv` |
| `indicators/basin.csv` | `results/basin_indicators.csv` |
| `indicators/RT_*.csv` | **deleted** — not migrated (v2 decision 3) |
| `data_catalog_climate_experiment.yml` | `config/catalogs/data_catalog_climate_experiment.yml` (v6 ruling) |
| `.project_consistency_ok` | `.project_consistency_ok` (unchanged) |
| `logs/*`, `benchmarks/*` | `logs/*`, `benchmarks/*` (unchanged) |
| `config/snake_config_climate_experiment.yml` | unchanged (identity) |
| `config/catalogs/*` | unchanged (identity) |
| `data_catalog_climate_experiment.yml` | `config/catalogs/data_catalog_climate_experiment.yml` — resolved by the v6 ruling |

### → `config/` — RULED (A), identity except where noted

| Old | New |
| --- | --- |
| `<P>/config/runs/snake_config_model_creation.yml` | unchanged — **contract path**, declared input of WF3 rule 3.00b |
| `<P>/config/runs/snake_config_climate_projections.yml` | unchanged — contract path |
| `<P>/config/runs/model_creation/<digest>/**` | unchanged |
| `<P>/config/catalogs/*.yml` | unchanged |
| `<P>/config/templates/*.yml` | unchanged — snapshots, **not** the editable inputs v4 drew |
| `<P>/config/observations/*` | unchanged |
| `<P>/config/generated/wflow_build_model_run.yml` | `models/hydrology/wflow/config/build_model.yml` |
| `<P>/config/generated/wflow_build_forcing_historical.yml` | `models/hydrology/wflow/config/build_historical_forcing.yml` |
| *(not built)* | `config/project.yml` — new capability, see scope note |

### → project root

| Old | New |
| --- | --- |
| `<P>/logs/wf{1,2}_*.log` | unchanged |
| `<P>/logs/_parts/**` | unchanged (transient) |
| `<P>/benchmarks/wf{1,2}_benchmarks.md` | unchanged |
| `<P>/benchmarks/_parts/**` | unchanged (transient) |
| *(new)* | `logs/dag/`, `experiments/<id>/logs/dag/` — v4 decision Q4 |

---

## Placement of the previously unplaced classes

All closed at design v8 (and v6/v7 for two of them). One is not a pure move:

| Artifact | Home | Cost |
| --- | --- | --- |
| `data_catalog_climate_experiment.yml` | `experiments/<id>/config/catalogs/` | none (v6 ruling) |
| `cmip6/report.md` | `data/climate/projections/cmip6/report.md` | none (v7) |
| `sim_dates.csv`, `resampled_dates.csv` | `climate/weathergenr/output/` | none — `series/` renamed `output/` |
| `.model_built`, `.outputs_configured` | model root | none — sentinel rule generalised |
| `spatial/*` | `data/spatial/**` | none — final; Finding 2 withdrawn |
| Wflow's `log.txt` | `hydrology/wflow/output/rlz_<r>_cst_<c>.log` | **one-line code change** |

**Scope of the mandated rename record.** `naming.md` §7 requires an internal
rename note for `rule all` output filenames. Across this whole map that is
**exactly two files** — `Qstats.csv` → `q_indicators.csv` and `basin.csv` →
`basin_indicators.csv`. Everything else moving here is either a directory
relocation or a non-`rule all` artifact, and `series/` → `output/` is a directory
rename, so §7 does not extend to it. Stated so the implementer does not re-derive
the scope.

**The `log.txt` row is a defect, not a move.** Wflow's `[logging] path_log`
defaults to `log.txt` beside the TOML, so removing the `rlz_<r>/` level puts every
member's log at one path — and rule 3.10 batches members concurrently, making it a
race rather than an overwrite. Set `path_log` per member from the existing
layout-derived pointers, **in the same commit that removes the directory level**.
Falsifier 15 in the design is the concurrency check.

## Rule rename carried by R9

`export_wflow_results` → `derive_wflow_indicators` (rule 3.11). R9 renames its
outputs, so the old name is falsified by this milestone; the other nine rule
renames are R10. Path effect is confined to transient parts:

| Old | New |
| --- | --- |
| `logs/_parts/3.11_export_wflow_results.log` | `logs/_parts/3.11_derive_wflow_indicators.log` |
| `benchmarks/_parts/3.11_export_wflow_results.tsv` | `benchmarks/_parts/3.11_derive_wflow_indicators.tsv` |

Both are merged and deleted every run and are not baseline-pinned, so no durable
path or value changes. `LOG_RULES` must be updated in the same edit — an unlisted
label is not an error, it is a silently missing log section plus orphaned parts.

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

1. ~~Rule Finding 1~~ **done 2026-08-04** — option (A), design v6.
2. ~~Rule Finding 3's three mismatches~~ **done 2026-08-04** — design v7.
3. ~~Place the unplaced artifact classes~~ **done 2026-08-04** — design v8.
4. ~~Re-derive the WF1/spatial rows after Gate 2 closes~~ **not required** —
   Finding 2 withdrawn 2026-08-04; the rows were already derived from the landed
   work.
5. Materialize a **clean** fixture from current code — the existing one cannot
   validate this map — and diff it against the completed map.
6. Encode the map as regex rules alongside `build_r07_path_map`.
