# Task Brief — WF2 v2.0 steps 1–2: model-free region + persistent series cache

### Context

Canonical ruleset: `AGENTS.md`. Design: `dev/workflows/wf2-climate-analysis-v2-design.md`
§5.3, §7, §8 (steps 1–2 only). Current-state map:
`dev/workflows/wf2_climate_projections_overview.md`.

- WF2's only cross-workflow input is `hydrology_model/staticgeoms/region.geojson`,
  used solely for `geom.geometry.bounds` + a 1° buffer.
  `snake_utils.climate_store_spec` already produces a **model-free**
  `store_region.geojson` from `shared.basin` + the catalog, and is declared
  identically in `Snakefile_model_creation` (1.10) and
  `Snakefile_climate_experiment` (3.02).
- **Measured 2026-07-29 on `test_case/test_local`:** both polygons bound
  `[9.658333, 0.35, 9.858333, 0.483333]` — identical. The swap therefore selects
  the same GCM cell set and cannot move a number on this fixture.
- All three WF2 intermediate netCDF families are `temp()`, so re-running with one
  changed horizon re-downloads the whole archive slice. Each series file is
  KB-scale; producing it costs remote GCS zarr access.
- `Snakefile_climate_projections` line 119 makes `monthly_stats_fut` depend on
  `monthly_stats_hist`'s output with the comment *"make sure starts with previous
  job"*. `get_stats_climate_proj.py` never opens that file — it reads only
  `input.region_path`. The likely real reason is the unguarded
  `os.mkdir(folder_out)` at lines 179–180.

### Goal

WF2 runs end-to-end with **no `hydrology_model/` on disk**, and a second run with
a changed `future_horizons` entry performs **zero network reads** — with every
manifested output byte-identical to the current baseline.

### Non-goals

- Retiring `historical_year_range` / switching the reference window to
  `shared.historical_window`. That moves the window by a decade and **changes
  every change factor** — design step 5, not this task.
- Collapsing `monthly_stats_hist`/`monthly_stats_fut` into one rule, making
  `members` a wildcard, or renaming outputs (step 3).
- Variable spec, area weighting, monthly change-factor table, report stage
  (steps 4–7).
- A 4th Snakefile entry point; new dependencies; any `save_grids` decision.

### Allowed scope

**Permitted**
- `Snakefile_climate_projections`
- `blueearth_cst/projections/get_stats_climate_proj.py`
- `tests/` — additive only

**Approval-gated** (pause and ask; name the reason)
- `blueearth_cst/shared/snake_utils.py` — `climate_store_spec` is co-owned by
  three DAGs. Prefer consuming it unchanged. If a change is unavoidable, stop at
  Gate 1.
- `config/workflows/*.yml` — only if a new optional key is genuinely required.

**Forbidden**
- `Snakefile_model_creation`, `Snakefile_climate_experiment` (their
  `extract_climate_grid` declarations must not be edited — see Task constraints)
- `dev/baseline/manifest.json`, `pixi.lock`, anything under `project_dir`

### Required changes (checklist)

**Commit 1 — model-free region**

1. Declare `extract_climate_grid` in `Snakefile_climate_projections` from
   `snake_utils.climate_store_spec`, with an input set **identical** to the other
   two declarations (exactly one entry: the catalog file, declared plain, never
   `ancient()`).
2. Point `monthly_stats_hist` / `monthly_stats_fut` at the store's
   `store_region.geojson` instead of `{basin_dir}/staticgeoms/region.geojson`.
   Keep the `ancient()` marking.
3. Add a test asserting the three `climate_store_spec` declarations produce
   identical input sets.

**Commit 2 — persistent series + drop the ordering edge**

4. Remove `temp(...)` from `historical_stats_time_{model}.nc` and
   `stats_time-{model}_{scenario}.nc`. Leave
   `annual_change_scalar_stats-*.nc` as `temp()` for now.
5. Add a `params` digest over `(catalog source entry, region bounds, buffer,
   variables, time window, reducer_version)` to both stats rules, so Snakemake's
   params rerun-trigger re-derives a series when any component changes.
6. Replace `os.mkdir(folder_out)` guarded by `os.path.exists` with
   `os.makedirs(folder_out, exist_ok=True)` in `get_stats_climate_proj.py`.
7. Delete the `stats_time_nc_hist` input from `monthly_stats_fut` (the
   ordering-only edge).

### Validation

Report each rung.

1. **Narrow** — `pytest tests/test_cli.py` (dry-runs all three Snakefiles) after
   each commit.
2. **New behavioural tests** — the identical-input-set test (item 3); a cache
   test: run, touch nothing, re-run, assert zero stats jobs; change one digest
   component, assert exactly the affected series re-derive.
3. **DAG diff** — `snakemake -n` on
   `config/workflows/snake_config_model_test.yml` before and after each commit;
   report the job count and rule set. Expect the hist/fut stages to become
   concurrent after commit 2.
4. **Full gate** — `pytest tests/`; must stay green and purely additive.
5. **Baseline / non-regression** — `pixi run python dev/scripts/check_baseline.py check`
   against `test_case/test_local` after **each** commit, plus
   `dev/scripts/semantic_tree_diff.py` over the WF2 output subtree.
   **This is a local gate — CI cannot run it.**
6. **Region re-check** — re-compare the bounds of `store_region.geojson` and
   `hydrology_model/staticgeoms/region.geojson` before relying on the
   value-neutrality claim. If they differ, STOP (see Gate 2).
7. **Decoupling proof** — move `hydrology_model/` aside and run WF2 to
   completion; restore it afterwards.

### Acceptance criteria

- Both commits pass `check_baseline.py check` with **zero** drift. Any drift on
  either commit means the change was not value-neutral → revert and report.
- WF2 completes with `hydrology_model/` absent.
- Second run after a `future_horizons` edit issues no network reads and re-runs
  only the derive/plot rules.
- `pytest tests/` green, additive only.
- No edits to the WF1/WF3 `extract_climate_grid` declarations.

**Rollback:** any baseline drift, or any WF1/WF3 DAG change visible in their
dry-runs, triggers a revert of the offending commit before continuing.

### Output requirements

- Two commits, scoped by explicit pathspec, in the order above.
- A short note in `dev/working/` recording: the measured region bounds at
  validation time, the before/after job counts, the second-run network-read
  evidence, and each validation rung's outcome.
- **Results delta:** expected to be *empty*. If any manifested value changes,
  stop and report what changed and why rather than re-recording the baseline.

### Task constraints

- **`climate_store_spec` declarations must stay symmetric across all three
  Snakefiles.** Its docstring: *"The input set is exactly one entry — the catalog
  — in both DAGs. An asymmetric input set re-creates the wf1↔wf3 re-extraction
  oscillation (design P2(b) / ext1-02)."* Adding a third declaration inherits
  that constraint verbatim.
- `--dry-run` before running and after editing any rule (`AGENTS.md` § Workflow).
- Do not commit run outputs under `project_dir`; do not hand-edit `pixi.lock`.
- Follow `dev/conventions/naming.md` for any new identifier.

**Human gates**

- **Gate 1** — if `snake_utils.py` must change, PAUSE. A shared-helper edit
  affects three DAGs and needs owner approval before proceeding.
- **Gate 2** — if the two region polygons' bounds differ at validation time,
  PAUSE. Step 1 is then value-changing and needs a re-record decision, not a
  silent continuation.
- **Gate 3** — after commit 1's baseline check, PAUSE and report before starting
  commit 2.
