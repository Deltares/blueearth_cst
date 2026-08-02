# Phase B report — commits 2–4

## Starting state

- Branch `task/p31-experiment-structure`, HEAD `5e7bd05` (commit 1, drift guard) on top of `main` `7d1be71`.
- Worktree: pre-existing unstaged deletion of `dev/working/design-runs/p31-experiment-structure/intake.md` (untouched); untracked `dev/milestones/p31/phase-a-report.md` + `dev/milestones/p31/phase-b-report.md` (untouched/this).
- Frozen reference tree: `examples/test_local_pre_p31/` (153 files) — READ-ONLY.
- Sentinel currently at `{exp_dir}/.project_consistency_ok` where `exp_dir = {project_dir}/climate_{experiment}` (moves automatically with commit-2 exp_dir redefinition).
- Guard artifact currently at `climate_historical/raw_data/.guard_ok` (moves in commit 4).
- Key files reviewed: `Snakefile_climate_experiment`, `snake_utils.py` (has `file_digest_or_absent`, `get_config`, `stress_test_grid`), `downscale_climate_forcing.py`, `prepare_climate_data_catalog.py`, `extract_historical_climate.py:143` (sidecar producer, derives from dirname(fn_out) — auto-carries).
- Existing tests: `test_prepare_climate_data_catalog.py::test_chirps_source_adds_orography_entry` encodes the OLD `../../` lookup — must update in commit 4.


## Commit 2 — validate_experiment_name + exp_dir move

**Implemented (design §2/§2a/§2b, §10.2):**
- `snake_utils.py`: `validate_experiment_name(name, project_dir)` — grammar `^[a-z0-9][a-z0-9_]*$`, ≤64 chars, Windows-reserved-name check (case- and extension-insensitive, checked BEFORE the grammar so lowercase `con`/`nul`/`com1` are rejected), containment assertion (direct child of `<project_dir>/experiments`). Uppercase rejected (not lowercased).
- Snakefile: import + call `experiment = validate_experiment_name(experiment, project_dir)` at line 41 (after read, before exp_dir). `exp_dir = f"{project_dir}/experiments/{experiment}"` (line 76). Manual project_dir-rooted moves: `rule all` snake_config + benchmarks targets → exp_dir; guard rule 3.00b log/benchmark → exp_dir; copy_config output → exp_dir/config/; all 10 `log:` + 12 `benchmark:` prefixes → exp_dir (incl. gather output line 389 + `parts_dir` param line 391). Run-dir paths left at `basin_dir/run_climate_{experiment}` (commit 3). wf2 snapshot path stays project-level (correct). guard_ok stays `climate_historical/raw_data/` (commit 4).
- `tests/test_validate_experiment_name.py`: full §2b matrix (24 cases).
- Fixups to commit-1/pre-existing tests for the new layout: `test_guard_invalidation.py:88` sentinel → `experiments/<name>/`; `test_workflow_climate_experiment.py:114` results → `experiments/<name>/`.

Commit hash: `3c77f81`

## Commit 3 — run-dir move + toml rewrite

**Implemented (design §5/§5a, §10.3):**
- Snakefile: 5 run-dir paths `{basin_dir}/run_climate_{experiment}/` → `{exp_dir}/model_runs/` (rules 3.09 toml, 3.10 toml_path/csv_file/state_file, 3.11 rlz_csv_fns).
- `downscale_climate_forcing.py`: two `../`-relative toml literals → absolute under `model_root`: `input.path_static = str(Path(model_root,"staticmaps.nc").resolve())`, `state.path_input = str(Path(model_root,"instate","instates.nc").resolve())`. hydromt's `config.write` re-relativizes on write. Left `state.path_output`, `input.path_forcing`, `output.csv.path` unchanged. No `dir_input` (A6 rejected).

**Gate 3 (execution smoke, scratch `proj_gate3`, full wf3 run, exit 0, 14/14 steps):**
- On-disk toml (`experiments/expa/model_runs/wflow_sbm_rlz_1_cst_1.toml`): `path_static = "../../../hydrology_model/staticmaps.nc"` (hydromt re-relativized the absolute path to the correct relative pointer from the new depth), `path_input = "../../../hydrology_model/instate/instates.nc"`, `path_forcing = "../realization_1/inmaps_rlz_1_cst_1.nc"` (auto-recomputed). Wflow ran and produced `output_rlz_1_cst_1.csv` (7670 rows) — confirms it read the relocated staticmaps via the re-relativized pointer.
- **Purity assertion: PASS** — `diff` of `hydrology_model/` file list+sizes pre/post the wf3 smoke shows ZERO changes. The `mode="r+"` session flushed nothing back into `hydrology_model/`. **No `mode` switch needed** (downscale_climate_forcing.py:43 left at `r+`).
- Gate 7 (gather smoke) captured en passant: `experiments/expa/benchmarks/wf3_benchmarks.md` non-empty, 12 per-rule rows from the moved `parts_dir`.

Commit hash: `6e2431c`

## Commit 4 — keyed store + chirps fix + guard-artifact move

**Implemented (design §4/§4a–§4d, §10.4):**
- `snake_utils.py`: `slugify_window(start, end) -> "YYYYMMDD_YYYYMMDD"` with the §4c day-resolution assertion (nonzero HH:MM:SS raises; unparseable date raises). Signature chosen minimal (two positional endpoints) — design was silent; recorded under Deviations.
- Snakefile: `store_key = f"{clim_source}_{slugify_window(hist_starttime, hist_endtime)}"`, `store_dir = {project_dir}/climate_historical/{store_key}`. Moved: extraction output → `{store_dir}/extract_historical.nc`; `generate_weather_realization` consumer input → same (ancient); guard `guard_ok` output → `{store_dir}/.guard_ok`; `extract_climate_grid` guard_ok ancient input → same. Added `oro_path = f"{store_dir}/{clim_source}_orography.nc"` param to rule 3.08.
- `prepare_climate_data_catalog.py`: added `oro_path` kwarg; chirps/chirps_global branch now uses the passed absolute path (raises if None), replacing the `../..` reconstruction. Wired from `__main__` via `sm.params.oro_path`.
- Tests: `tests/test_slugify_window.py` (8 cases incl. day-resolution failures); `test_prepare_climate_data_catalog.py` — rewrote the chirps orography test to the §4a form (deeper realization dir + passed `oro_path`, asserts uri==sidecar and exists), added `test_chirps_source_requires_oro_path` and `test_era5_source_ignores_oro_path`, fixed the chirps metadata test to pass `oro_path`.

Commit hash: `27a2a54`

## Gate results (per commit)

| Commit | Gate | Result |
| --- | --- | --- |
| 2 | `pytest tests/test_cli.py` | PASS (4/4) |
| 2 | gate 9 validation matrix (`test_validate_experiment_name.py`) | PASS (24 cases) |
| 2 | gate 1 ext — `experiment_name: "../evil"` fails at parse | PASS (parse error line 41 names `'../evil'` + grammar, exit 1) |
| 2 | clean `--dry-run` all three Snakefiles | PASS (all exit 0) |
| 3 | `pytest tests/test_cli.py` | PASS |
| 3 | clean `--dry-run` wf3 | PASS |
| 3 | gate 3 execution smoke (run-dir + toml re-relativization) | PASS — Wflow read relocated staticmaps via `path_static="../../../hydrology_model/staticmaps.nc"`, produced run CSV (7670 rows) |
| 3 | gate 3 purity assertion | PASS — `hydrology_model/` unchanged pre/post (no `mode` switch) |
| 3 | gate 7 gather smoke (captured here) | PASS — `wf3_benchmarks.md` non-empty, 12 rows |
| 4 | `slugify_window` unit tests | PASS (`test_slugify_window.py`, 8 cases) |
| 4 | §4a orography-lookup unit tests | PASS (`test_prepare_climate_data_catalog.py` — rewritten + 2 new) |
| 4 | clean `--dry-run` wf3 | PASS |
| 4 | gate 4 (reuse + guard interaction + alternation) | PASS (evidence below) |
| 4 | gate 5 (collision) | PASS (evidence below) |
| 4 | gate 8 (chirps path) | **unit-test minimum** — chirps data not staged locally (only era5 under `C:/data/wflow_global/hydromt/meteo/`); §4a lookup unit tests are the design's mandatory minimum |
| all | full default `pytest tests/` | PASS (287 passed, 3 skipped, 1 xfailed) |

## Gate-4 / gate-5 evidence (reuse, alternation, collisions)

Fresh scratch `proj_gate4` (seeded from frozen tree, wf1 snapshot `project_dir` rewritten to match; guarded sections byte-identical A/B/C). A: window 2000-01-01..2020-12-31; B: same window, `experiment_name=expb` only; C: window 2001-01-01..2020-12-31 (unguarded `historical_window`, guard still passes). All `--configfile` at the same project_dir. Dry-run reasons captured with `--workflow-profile none`.

- **A (first run, key `era5_20000101_20201231`):** full wf3 exit 0, 14/14 steps. Extraction landed at `climate_historical/era5_20000101_20201231/extract_historical.nc`; `.guard_ok` in the same key dir.
- **B (same key, skip):** B `--dry-run` schedules 13 jobs; guard runs (reason verbatim: `Missing output files: .../experiments/expb/.project_consistency_ok`), but **`extract_climate_grid` is ABSENT from B's schedule (0 occurrences)** and there is **NO "set of input files has changed" reason (0 occurrences)** — the v4 input-set-invariance fix works. B full run: 13/13 steps (one fewer than A — the skipped extraction), exit 0, `extract_climate_grid` never executed. `extract_historical.nc` mtime **unchanged** (reused); `.guard_ok` mtime **advanced** (B's guard rewrote it).
- **C (new key, re-extract):** C `--dry-run` schedules `extract_climate_grid` (new key, 5 occurrences). C full run: 14/14 steps, exit 0, extraction landed at the NEW key dir `era5_20010101_20201231/` — only after guard(C) passed (the `K'/.guard_ok` dependency edge).
- **Alternation (A re-invoked after B):** A `--dry-run` → `Nothing to be done (all requested files are present and up to date).` — **zero** "Params have changed" / guard re-run (the guard's experiment-invariant params on the shared `.guard_ok` cause no thrash). Rollback-trigger NOT tripped.
- **Gate 5 (collision) — the six live design classes:** with expa/expb/expc coexisting, the **five per-experiment classes** (config snapshot, logs, benchmarks, model_runs, model_results) show **0 cross-experiment file collisions** — all present per experiment, all under disjoint `experiments/<name>/` roots. The **sixth class, historical-climate reference, passes as REUSE, not collision** — shared by design: expa+expb → same key dir (one extraction reused), expc → own key; **2 `extract_historical.nc` files for 3 experiments** is the reuse proof. (plots is the reserved 7th, no wf3 producer.)

## Deviations & open questions

1. **`slugify_window` signature (design-silent).** Chose minimal `slugify_window(start, end) -> "<YYYYMMDD>_<YYYYMMDD>"` (two positional ISO endpoints). Key string is exactly `<clim_source>_<YYYYMMDD>_<YYYYMMDD>` as design §4/§4d specify. Day-resolution assertion raises `ValueError` on any nonzero HH:MM:SS or unparseable date.
2. **Gate 8 = unit-test minimum (design-sanctioned).** chirps/chirps_global source data is not staged locally (catalog points at `C:\data\wflow_global\hydromt\meteo\chirps_*`, absent; only era5 present). Per §4a the mandatory minimum when a full chirps stage is unavailable is the unit test on the rewritten lookup — implemented (`test_chirps_source_adds_orography_entry` asserts uri==passed sidecar + exists; `test_chirps_source_requires_oro_path`; `test_era5_source_ignores_oro_path`). Full chirps smoke deferred (no data).
3. **Purity: no `mode` switch.** downscale_climate_forcing.py:43 left at `mode="r+"`. The gate-3 check is a `find -printf '%p\t%s'` diff of `hydrology_model/` pre/post — it proves **no new or size-changed files** (exactly the gate's "gained no wf3-written files" requirement, which PASSES). Note: mtimes were not compared, so a hypothetical same-size re-flush would not be caught by this diff; the requirement is met either way.
4. **Orography param is dry-run-blind (inherent to the chosen mechanism).** The sidecar path now derives from the Snakefile params string `oro_path = f"{store_dir}/{clim_source}_orography.nc"`, which cannot be exercised end-to-end without chirps data. It is consistent-by-construction with the producer's sidecar path (`extract_historical_climate.py:143`, `dirname(fn_out)/{clim_source}_orography.nc`), and the design explicitly endorsed passing it as a param (§4a) — inherent to the mechanism, not a defect. Unit tests cover the consumer logic.
5. **Commit-1 test fixups for the new layout (in `tests/**` scope):** `test_guard_invalidation.py` — sentinel + keyed `.guard_ok` paths, fixture now returns the keyed guard_ok, `slugify_window` import; `test_workflow_climate_experiment.py` — results path; `test_prepare_climate_data_catalog.py` — chirps tests to the §4a form. All required for the phase-B gates to pass; none changes computed values. **Note:** `test_workflow_climate_experiment.py` is `@pytest.mark.integration` (auto-skipped in the default suite), so its results-path edit was NOT executed by the 287-pass suite — but the gate-3/4 full wf3 smokes populated the same `experiments/<name>/model_results/` target, so the edit is validated by the smokes.

**Boundaries (for the reviewer):** wf1/wf2 Snakefiles (`Snakefile_model_creation`/`Snakefile_climate_projections`) UNTOUCHED (forbidden surface — zero edits needed). No writes into `examples/test_local*` or the frozen `examples/test_local_pre_p31/`. Pre-existing `intake.md` deletion and the two untracked reports left uncommitted. **Commits 5–7 NOT started; the wf3 baseline slice is NOT re-recorded — value-identity is not yet claimable (that is gate 6's job).** No merge, tag, or push.
