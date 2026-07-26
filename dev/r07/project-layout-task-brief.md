Task Brief — R07 project layout

### Context

Canonical ruleset: `AGENTS.md` (repo root). Read before starting; its Hard
Constraints bind this task.

- **Authority.** `dev/r07/project-layout-design.md` is the plan; the old→new path
  map is `dev/r07/migration_project-layout.md`. Both are **DRAFT** — see Gate 0.
  Findings provenance: `dev/reviews/2026-07-25_post-r6-assessment.md`.
- **Behaviour-preserving, but not re-record-free.** No computational path changes.
  17 of 18 baseline targets move path; 4 also change content. The manifest is
  re-recorded **exactly once**, at the end.
- **The `"None"` sentinel is the highest-risk detail.** Unquoted `None` in the
  configs parses to the Python **string** `"None"`, not YAML `null`. `null` raises
  `TypeError` at `setup_gauges_and_outputs.py:55` and `plot_results.py:127`. Every
  `None` written must be byte-identical to
  `config/workflows/snake_config_model_test.yml:36-37`.
- **`script:` paths resolve against `workflow.basedir`**, not the CWD; R `shell:`
  bodies are CWD-relative. `--dry-run` is blind to `params:`-string paths and to R
  script bodies — phases B and C need a real run.
- Two engine subtrees (`weather_generator/`, `hydrology_runs/`) share one internal
  shape: `config/ output/ plots/ _work/`.

### Goal

Land the R07 layout across both halves of the system — repository and
`project_dir` — as 13 `r07:` commits, each leaving the tree runnable, with a
single baseline re-record at the end and a clean full-tree semantic diff.

### Non-goals

- The tooling-contract decisions (O-14 `pyproject.toml`, O-15 `ruff`, O-16
  `flit`). Open, unrelated to layout.
- Docker (O-06) and Linux end-to-end (O-18, O-19). Parked.
- Promoting climate analysis to a fourth Snakefile. Separate milestone.
- Any change to a computed value, a Wflow physics parameter, or hydromt's
  internals.

### Allowed scope

**Permitted.** `Snakefile_*` (paths only), `blueearth_cst/**`, `config/**`,
`scripts/**`, `tests/**`, `docs/notebooks/*.ipynb` (DAG cells only), `README.rst`,
`AGENTS.md`, `MIGRATION.md`, `.gitignore`, `dev/r07/**`, `dev/scripts/check_baseline.py`,
`dev/scripts/semantic_tree_diff.py`.

**Approval-gated.** `dev/baseline/manifest.json` — re-record only at commit 12,
after Gate 3. Deleting `data/` and `docs/config/` — released by Gate 1.

**Forbidden.** `pixi.lock`, `Manifest.toml`, `Project.toml`, vendored upstream
packages, anything under `.pixi/`, `Dockerfile` and `scripts/run_snake_docker.sh`
beyond the single `data/` mount removal, `test_case/` contents by hand (runs write
there).

### Required changes (checklist)

Follow the design's commit plan; each numbered item is one commit.

**Phase A — repository (commits 1–5).**
1. Retire `data/`; add `config/templates/observations/{output_locations.csv,
   observations_timeseries.csv,README.md}`; repoint the Linux and `tests/` configs
   to the sentinel; drop the `data/` mount from `run_snake_docker.sh:7`.
2. Delete `docs/config/` (16 files); update `AGENTS.md` `docs/` description and
   `MIGRATION.md:173`.
3. `examples/` → `test_case/`: `.gitignore:124`, all config `project_dir` values,
   `check_baseline.py`.
4. DAG renders → `<project_dir>/dag/`: `run_snake_test.cmd:32` (backslash paths —
   `mkdir` rejects forward slashes), `.gitignore:135-136`, `README.rst:269,285,298`,
   six notebook cells; delete `dag/` and the stray `dag_model.png`.
5. Fix the template `project_dir` default (O-21); add
   `warn_if_project_dir_in_repo()` to `snake_utils.py` beside
   `validate_experiment_name`, called at parse time from all three Snakefiles.
   Containment via `os.path.commonpath`, **not** `startswith`. Warns, never raises.
   Exemption `<repo_root>/test_case` in a module-level constant.

**Phase B — artifact tree (commits 6–11).**
6. Collapse `wf1_raw/` and `<key>/` into one region-keyed store; restore the
   retired `allclose` check as a unit test.
7. Move wflow forcing to `hydrology_model/forcing/`; edit `path_forcing`.
8. Tier `climate_projections/` into `timeseries/` + `summary/` + `plots/`.
9. Restructure the experiment: `weather_generator/` + `hydrology_runs/rlz_<r>/
   {config,output}/`; dissolve `realization_*/`; `stress_test/` → `_work/`;
   `model_results/` → `indicators/`.
10. New climate plot producer under `blueearth_cst/climate_analysis/` reading
    `<key>/extract_historical.nc` (source-grid PET; **need not** match the build's
    PET); declare the missing plot outputs on rules 1.11 and 1.13 (O-24).
11. `experiment_id` suggestion helper — writes `experiment_name` into the config
    **once**; never generated at run time.

**Phase C — machinery and docs (commits 12–13).**
12. Update `check_baseline.py` `TARGETS`, `semantic_tree_diff.py` path map and
    TOML comparator; re-record `dev/baseline/manifest.json`; fill the
    MISSING/EXTRA allowlist in the migration map.
13. Docs: `AGENTS.md` repo map + the invocation-model line (`scripts/` executes
    the pipeline, `dev/scripts/` maintains the repo, `blueearth_cst/` is executed
    by Snakemake), `README.rst`, `MIGRATION.md`.

**Drive-by fixes** — separate small commits, independent of the above: O-07
(`prepare_cst_parameters.py:14` sys.path depth), O-08 (`plot_map.py:28` sentinel
guard), O-09 (`plot_results.py:83` separator docstring), O-10 (`MIGRATION.md`
`__init__.py` list).

### Validation

Report every rung.

1. **Narrow** — `pytest tests/test_cli.py` after each commit that touches a
   Snakefile or a config. Plus the sentinel assertion: both observation keys parse
   to the **string** `"None"` in every edited config.
2. **New behavioural tests** — the `allclose` unit test (commit 6); the
   `warn_if_project_dir_in_repo` cases (fires in-repo, silent for `test_case/` and
   absolute paths); `snakemake --delete-all-output` removes the newly declared
   plot outputs.
3. **Integration** — all three Snakefiles `--dry-run` clean;
   `scripts\run_snake_test.cmd --dry-run` writes only under
   `test_case\test_local\dag\`.
4. **Full gate** — `pytest tests/`; CI baselines must not move (386/30/1
   `windows-latest`, 385/31/1 `ubuntu-latest`).
5. **Baseline / non-regression** — a full three-workflow run on the seed config;
   `semantic_tree_diff.py` full-`project_dir` pre/post with the R07 path map:
   **MISSING/EXTRA empty modulo the written allowlist**, all values identical;
   then `check_baseline` green against the re-recorded manifest.

Capture a pre-R07 reference tree **before commit 1** — the phase-B diff is
worthless without it.

### Acceptance criteria

- 13 `r07:` commits (plus drive-bys), each leaving the tree runnable.
- Full-tree semantic diff clean modulo a written, justified allowlist.
- The **P4 assertion** demonstrated: climate figures produced with no
  `hydrology_model/` present.
- Manifest re-recorded exactly once; `check_baseline` green.
- No file remains at a path the migration map says has moved.
- **Rollback:** if the phase-B diff shows any *value* difference (not path, not the
  adjudicated config snapshots), stop and revert the offending commit — the
  milestone's premise is that no computational path changes.

### Output requirements

- The commits, plus the filled MISSING/EXTRA allowlist in
  `dev/r07/migration_project-layout.md` §4.
- A short results note recording: the diff outcome, the manifest re-record, and
  the CI/suite counts before and after.
- **Results delta:** expected new artifacts are the source-grid climate figures
  and the four newly declared plot outputs. Any *other* EXTRA or any MISSING is a
  defect, not a delta.

### Task constraints

- Do not change any computed value. Path moves, renames, declaration fixes, and
  the added warning only.
- Do not hand-edit `test_case/` contents; regenerate by running.
- Preserve the `"None"` string sentinel exactly.
- Keep `hydrology_model/` as the hydromt `model_root` (design option A) — do not
  nest a `model/` subfolder.
- TOML relative pointers change depth in commit 9; let hydromt re-relativize on
  write and update the comparator — never hand-edit the pointer strings.

**Human gates — PAUSE at each.**

- **Gate 0 — before any commit.** The design is DRAFT. Do not start until the
  owner marks `dev/r07/project-layout-design.md` ACCEPTED. Confirm the four open
  questions are resolved or explicitly deferred.
- **Gate 1 — after the pre-R07 reference capture, before commit 1.** Deleting
  `data/` and `docs/config/` removes tracked files; confirm scope with the owner.
- **Gate 2 — after phase A.** Report rungs 1, 3, 4. Repository half must be green
  before any artifact path moves.
- **Gate 3 — after phase B, before the re-record.** Report rung 5 against the
  pre-R07 reference. The owner adjudicates the MISSING/EXTRA allowlist. **Do not
  re-record the manifest before this gate releases.**
