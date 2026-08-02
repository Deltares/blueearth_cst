# Migration — P3-1 experiment structure (old → new wf3 output layout)

> Landed by the `p31:` commit series on `task/p31-experiment-structure`
> (design: `dev/milestones/p31/experiment-structure-design.md`, ACCEPTED 2026-07-23).
> Scope: paths move, values identical — no computational path was edited.
> Milestone evidence: `dev/milestones/p31/baseline_diffs.md`.

## 1. Old → new output-path map

All paths are project-root-relative (relative to `project_dir`).
`<name>` = `experiment_name` (validated: `^[a-z0-9][a-z0-9_]*$`, ≤64 chars,
no Windows reserved names); `<key>` = `<clim_source>_<YYYYMMDD>_<YYYYMMDD>`
(dataset + historical window, e.g. `era5_20000101_20201231`).

The five §6a content-bearing classes, plus the run dir and the keyed store:

| Class | Old | New |
| --- | --- | --- |
| Results CSVs | `climate_<name>/model_results/` (`Qstats.csv`, `basin.csv`, `RT_*.csv`) | `experiments/<name>/model_results/` |
| wf3 config snapshot | `config/snake_config_climate_experiment.yml` | `experiments/<name>/config/snake_config_climate_experiment.yml` |
| Run-dir tomls | `hydrology_model/run_climate_<name>/wflow_sbm_rlz_*_cst_*.toml` | `experiments/<name>/model_runs/` |
| Run-dir output CSVs / outstates | `hydrology_model/run_climate_<name>/output_rlz_*_cst_*.csv`, `outstates_*.nc` | `experiments/<name>/model_runs/` |
| Keyed extraction store | `climate_historical/raw_data/extract_historical.nc` (+ `<src>_orography.nc` sidecar) | `climate_historical/<key>/extract_historical.nc` (+ sidecar) |
| Experiment subtree (stress_test/, realization_*/, weathergen_config.yml, data_catalog_climate_experiment.yml) | `climate_<name>/…` | `experiments/<name>/…` |
| wf3 logs / benchmarks | `logs/`, `benchmarks/` (project-level, shared with wf1/wf2) | `experiments/<name>/logs/`, `experiments/<name>/benchmarks/` (incl. `benchmarks/_parts/` + `wf3_benchmarks.md`) |

Unchanged by design: `hydrology_model/` (pure wf1, incl. `run_default/`),
`climate_historical/wflow_data/inmaps_historical.nc` (wf1 forcing — C1 pin,
consumed by `setup_time_horizon.py`), `climate_projections/<clim_project>/`
(wf2), the wf1/wf2 config snapshots under project-level `config/`.

In-toml pointer strings legitimately changed value (same targets, new
depths; hydromt re-relativizes): `input.path_static` → 
`../../../hydrology_model/staticmaps.nc`, `state.path_input` →
`../../../hydrology_model/instate/instates.nc`, `input.path_forcing` →
`../realization_N/inmaps_….nc`. The milestone diff compares these in a
project-root-relative namespace (path-aware toml comparator,
`dev/scripts/semantic_tree_diff.py`), not as raw strings.

## 2. MISSING/EXTRA allowlist (§6a step 3 review rule)

Every entry carries a written justification; an entry not listed here fails
the milestone gate.

| Entry | Kind | Justification |
| --- | --- | --- |
| `experiments/<name>/.project_consistency_ok` | EXTRA | Per-experiment drift-guard sentinel, new in P3-1 (commit 1). A gate artifact, no scientific content; no pre-P3-1 counterpart. |
| `climate_historical/<key>/.guard_ok` | EXTRA | Key-level guard artifact (v4 ext2-1), new in P3-1. Same class as above. |
| `experiments/<name>/config/<data_catalog>.yml` (run-specific, via `--allow`; here `deltares_data.yml`) | EXTRA | De-collided per-experiment copy of the shared data catalog: the OLD layout's wf3 `copy_config` overwrote wf1's project-level copy — the exact collision P3-1 fixes. Verified byte-identical to the project-level copy at the 2026-07-23 gate. |

Final allowlist confirmed against the commit-6 milestone diff (2026-07-23):
`CLEAN: 83 files compared, 0 failed, 0 missing, 0 extra, 3 allowlisted` —
exactly the three entries above, no others. (The post-e2e re-diff additionally
shows the wf2 overlay outputs as EXTRA — present only because the e2e ran
wf2, which the §6a reference capture deliberately omits; see
`dev/milestones/p31/baseline_diffs.md`.)

Nothing is MISSING-by-design: there is no wf3 plots producer, so the reserved
`experiments/<name>/plots/` class has no files on either side.

## 3. Design erratum — gate 2c(ii) `--unlock` (Gate-1 endorsed)

Design §7 gate 2c(ii) expected `snakemake --unlock` to succeed with the wf1
config snapshot absent. On the pinned Snakemake 9.6.2 this is unsatisfiable
as written: `Workflow.unlock()` (`site-packages/snakemake/workflow.py:917`)
builds the DAG before releasing locks, so `--unlock` fails with
`MissingInputException` on ANY missing leaf input — a pre-existing,
guard-independent behavior (probe: with the guard inputs present and the
pre-existing `ancient(region.geojson)` absent, `--unlock` fails identically).
The endorsed resolution keeps the load-bearing §3b mechanism exactly as
designed; the pinned contract is **no degradation beyond baseline**:
`--unlock` with the snapshot absent fails only with the clean rule-level
`MissingInputException` (no traceback), and succeeds with every leaf input
present (the real recoverable-lock scenario — a crashed run implies the
snapshot existed). Pinned by
`tests/test_guard_invalidation.py::test_2c_fresh_project_missing_wf1_snapshot`.

## 4. Baseline machinery

- `dev/scripts/check_baseline.py`: wf3 TARGETS repointed (results →
  `experiments/<name>/model_results/`; config snapshot →
  `experiments/<name>/config/`). Only ever pointed at the tracked seed config
  (`config/workflows/snake_config_model_test.yml`).
- `dev/scripts/semantic_tree_diff.py`: P3-1 path map (directory-prefix
  rewrite rules), MISSING/EXTRA-empty-modulo-allowlist gate contract, and the
  4-step project-root-relative path-aware toml comparator; commit 5b extends
  the same cross-root treatment to YAML string leaves (`<PROJECT_ROOT>`
  normalization of path-prefixed leaves, ref-side path map — the config
  snapshots' `project_dir`, weathergen output paths, and catalog `uri`s
  differ only by each tree's own root in a cross-root gate) and excludes
  run-log FILES (`*.log`, `log.txt`) as non-content-bearing.
- `dev/baseline/manifest.json`: wf3 slice re-recorded value-identically at
  the new paths (commit 6; see `dev/milestones/p31/baseline_diffs.md`).
