Task Brief — R07 project layout

### Context

Canonical ruleset: `AGENTS.md` (repo root). Read it before starting; its Hard
Constraints bind this task.

- **Authority.** `dev/milestones/r07/project-layout-design.md` — **ACCEPTED 2026-07-28**,
  after a four-version review (44 findings, all dispositioned). It is the plan;
  do not re-litigate it. Audit trail:
  `dev/milestones/r07/project-layout-design-review-record.md`. Approved framing:
  `dev/milestones/r07/project-layout-intake.md`. Observation register:
  `dev/reviews/2026-07-25_post-r6-assessment.md`.
- **Path authority.** `dev/milestones/r07/migration_project-layout.md` § 7 is the old→new
  path map. **Where a path in this brief disagrees with the map, the map governs
  paths and the design governs behaviour.**
- **Behaviour-preserving, scoped.** No computed value changes **on the baseline
  seed-fixture class**, with three named exceptions (design § "Behaviour-
  preservation stance", exceptions 1–3). Non-seed divergence is the GA-1-accepted
  derivation change, not a regression. Inventory: `check_baseline.TARGETS` holds
  **15 live templates** (the 18-row manifest carries 3 orphans); all 15 change
  manifest key, 10 also change path, 3 change content. **One re-record, at the
  end.**
- **`check_baseline check` is red by construction from commit 4 to commit 14.**
  This is expected, not a regression signal — say so in those commit messages.
  The substitute gates are per-slice `semantic_tree_diff` against a retained
  pre-R07 reference tree, plus the comparator-based discharge anchor. **Both
  holding artifacts are captured in commit 1 and must be preserved for the whole
  milestone — if the reference tree is lost, commits 4–14 have no regression
  detector at all.**
- **`--dry-run` is blind** to `params:`-string paths and to R `shell:` bodies.
  **B1, B4, B5, and B6 need a real run to be proven** (design § "The gate
  blackout", final paragraph). `script:` paths resolve against
  `workflow.basedir`, not the CWD.

### Goal

Land the R07 layout across both halves of the system — repository and
`project_dir` — as **15 `r07:` commits** in the design's order, each leaving the
tree runnable, with a single baseline re-record at commit 14 and a clean
full-tree semantic diff.

### Non-goals

- Tooling-contract decisions O-14 (`pyproject.toml`), O-15 (`ruff`), O-16
  (`flit`). Open, unrelated to layout.
- Docker (O-06) and Linux **end-to-end** validation (O-18, O-19). Parked — but
  Linux *parse-level consistency* is in scope (commits 2 and 4).
- Engine-named subtrees (`models/wflow/`) and the structural half of the
  engine-placement question. Parked beyond R07.
- Promoting climate analysis to a fourth Snakefile. Separate milestone.
- Any change to a computed value, a Wflow physics parameter, or hydromt
  internals.

### Allowed scope

**Permitted.** `Snakefile_*`, `blueearth_cst/**`, `config/**`, `scripts/**`,
`tests/**`, `docs/**` (notebook DAG cells; the new `docs/migration-r06.md`),
`README.rst`, `AGENTS.md`, `MIGRATION.md`, `.gitignore`, `dev/scripts/check_baseline.py`,
`dev/scripts/semantic_tree_diff.py`, `dev/reference/contracts/*-seam.md`,
`dev/reference/naming.md` (§ 7 amendment, commit 15), `dev/milestones/r06/**` (the
reconstructed rename record, commit 15).

**Scoped write.** `dev/milestones/r07/migration_project-layout.md` — **append only**: the
empirically-determined MISSING/EXTRA allowlist, the three expected manifest-row
deletions, the `--forcerun extract_climate_grid` escape-hatch note, the
`COPIED_CONFIG_PATH_MAP` config-path rows, and (only if the bbox branch fires)
the per-edge coordinate deltas. **Do not restructure its § 7 path table.**

**Approval-gated.** `dev/baseline/manifest.json` — re-record only in commit 14,
after Gate 3. Deleting the tracked `data/` tree, `docs/config/` (16 files), and
`blueearth_cst.Rproj` — released by Gate 0.

**Forbidden.** `pixi.lock`, `Manifest.toml`, `Project.toml`, vendored upstream
packages, anything under `.pixi/`. `dev/milestones/r07/project-layout-design.md`,
`dev/milestones/r07/project-layout-intake.md`,
`dev/milestones/r07/project-layout-design-review-record.md`, `dev/roadmap.md` — read-only.
`Dockerfile` and `scripts/run_snake_docker.sh` beyond the two named mount edits.
Hand-editing `test_case/` contents (runs write there). Adding any new dependency
without asking the owner first.

### Required changes (checklist)

One numbered item = one commit, in this order. Design § "Commit plan" is the
authority; the notes below carry the details the review forced.

**Binding list.** `dev/milestones/r07/project-layout-design.md` § "Contract inventory" is the
authoritative per-move table of rules, script modules, **tests**, and **seam
docs**. Update every row's Tests and Seam doc cells **in that move's own commit** —
`tests/` bindings and `dev/reference/contracts/*-seam.md` pins are not optional follow-up.

- [ ] **1. `r07: prepare the baseline machinery for the layout move`** —
  `build_r07_path_map()` / `build_r07_allowlist()`, a generic `--map old=new`
  CLI option, the **declared many-to-one merge class** (`--merge
  <survivor>=<src1>,<src2>`; a merge passes only if the survivor matches
  **every** collapsed source), and new `COPIED_CONFIG_PATH_MAP` entries — all in
  `dev/scripts/semantic_tree_diff.py`. Capture the **pre-R07 reference tree** and
  save `examples/test_local/hydrology_model/run_default/output.csv` to a
  run-local holding path. **`check_baseline.py` is not touched here** — its
  retarget is commit 4 (ext2-03).
- [ ] **2. `r07: retire data/, ship observation templates`** (O-01) — add
  `config/templates/observations/{output_locations.csv,observations_timeseries.csv,README.md}`
  (header-only schemas). **Retarget both live consumers in this same commit, not
  later:** `config/workflows/snake_config_model_test_linux.yml:25-26` → the
  `None` sentinel, and drop the `data/` mount from `scripts/run_snake_docker.sh:7`.
  Also fix `tests/snake_config_model_test.yml:32-33` (O-04 — it points at a
  nonexistent `tests/data/observations/` tree). Add the Linux-config dry-run to
  `tests/test_cli.py`.
- [ ] **3. `r07: delete the docs/config mirror`** (O-05, 16 tracked files) +
  `AGENTS.md`'s `docs/` description + `MIGRATION.md:173`.
- [ ] **4. `r07: rename examples/ -> test_case/`** (O-20) — fixture path,
  `.gitignore:124`, every config `project_dir`, `run_snake_docker.sh`'s
  `examples` mount, the four affected test modules, and — **this commit is the
  sole owner** — `check_baseline.py`'s `TARGETS` templates and
  `PROJECT_DIR_DEFAULT`. **The baseline blackout starts here.**
- [ ] **5. `r07: relocate DAG renders under project_dir`** (O-02) —
  `scripts/run_snake_test.cmd:32` (backslash paths; `mkdir` rejects forward
  slashes), `.gitignore:135-136`, `README.rst:269,285,298`, six notebook cells;
  delete `dag/` and the stray `dag_model.png`.
- [ ] **6. `r07: fix the template project_dir default`** (O-21) +
  `warn_if_project_dir_in_repo(project_dir, repo_root)` in
  `blueearth_cst/shared/snake_utils.py` (O-22), called at parse time from all
  three Snakefiles with `workflow.basedir` as `repo_root`. Containment via
  `os.path.commonpath`, **not** `startswith`. Warns, never raises. Exemption in
  `_PROJECT_DIR_EXEMPT_NAMES = frozenset({"test_case"})`. **Pause point — Gate 1.**
- [ ] **7. `r07: single climate store with a shared region+catalog producer`**
  (B1) — the milestone's most delicate change; read design § B1 in full.
  - `climate_store_spec(project_dir, model_region, clim_source,
    historical_window, data_sources, hydrography, basin_index)` in
    `snake_utils.py` returns the **complete** producer contract (store dir,
    script path, inputs, outputs, params). Both Snakefiles splat every
    spec-driven field.
  - Rule name `extract_climate_grid` in **both** DAGs. Script:
    `blueearth_cst/climate_analysis/extract_historical_climate.py`.
  - **Inputs: exactly one — the data catalog (`project.data_sources`), plain,
    not `ancient()`, declared identically in both DAGs.** Remove rule 1.10's
    `ancient(staticmaps.nc)`, rule 3.02's `ancient(staticgeoms/region.geojson)`,
    **and** rule 3.02's `ancient({store_dir}/.guard_ok)`. An asymmetric input
    set reproduces the ext1-02 oscillation and is forbidden by P2(b).
  - Delineation: `hydromt.model.processes.region.parse_region_basin(...)`, with
    two **new optional** `shared.basin` keys — `hydrography` (default
    `merit_hydro_ihu`) and `basin_index` (default `merit_hydro_index`), both
    catalog entry names — shipped as **commented** template-config lines. Absent
    keys leave the rule-3.00b guard digest byte-identical, so existing configs
    are unaffected.
  - `prepare_build_config.py` (rule 1.02) gains both as params and **raises
    `RuntimeError` naming both files and both values** when the build template's
    `setup_basemaps.hydrography_fn` / `basin_index_fn` disagree with
    `shared.basin`. Do **not** inject the config values into the generated build
    config (rejected alternative).
  - Second declared output `<key>/store_region.geojson`. Standardise the sidecar
    on `orography.nc` and repoint rule 3.08's `oro_path` params string
    (`Snakefile_climate_experiment:331`).
  - Retire rule 1.10, `climate_historical/wf1_raw/`, and
    `blueearth_cst/model/get_region_preview.py` (O-25 — it does **not** import
    on the pinned hydromt 1.3.1; `hydromt.cli.api` was removed in hydromt 1.x).
    Repoint rule 1.11's `_wf1_plot_clim_inputs` to the store.
  - Tests: contract-equality, catalog-staleness, bbox-agreement, hydrography
    cross-check, chirps `oro_path` (see § Validation rung 2).
    `tests/test_extract_climate_wf1.py:24,26` covers the **retired** rule 1.10 —
    retire or rewrite it, do not merely repoint it.
    `tests/test_guard_invalidation.py:97` must stay green, now invalidating via
    the **params** trigger rather than the retired guard edge.
- [ ] **8. `r07: move wflow forcing into the engine subtree`** (B2) →
  `hydrology_model/forcing/inmaps_historical.nc`; the new `path_forcing` is the
  relative `forcing/inmaps_historical.nc`. Edit surface:
  `blueearth_cst/shared/setup_time_horizon.py:51`,
  `Snakefile_model_creation:198,210,305`,
  `tests/test_interchange_contracts.py:529`.
- [ ] **9. `r07: tier climate_projections outputs`** (B3) — `timeseries/` +
  `summary/`. **Only the three summary files move.**
- [ ] **10. `r07: split the project config snapshot into runs/catalogs/templates/generated`**
  (B9) — a **signature change**, not a rename: `copy_config_files.py:47-56,68,80-81`
  goes from one derived `output_dir` to four destinations. Also moves four rule
  path strings (1.02 output, 1.03 input, 1.07 output, 1.08 input) into
  `<project_dir>/config/generated/`.
- [ ] **11. `r07: restructure the experiment into engine subtrees`** (B5, B6,
  B7) — `weather_generator/` + `hydrology_runs/rlz_<r>/{config,forcing,output}/`;
  dissolve `realization_*/`; `stress_test/` → `weather_generator/_work/`
  (**retained on disk, not deleted** — it is the only record of
  `precip_variance` and of monthly structure); `model_results/` → `indicators/`.
  `inmaps_rlz_*_cst_*.nc` are wflow-grid downscaled forcing → 
  `hydrology_runs/rlz_<r>/forcing/inmaps_cst_<c>.nc`, keeping `temp()`;
  `outstates_*` → `hydrology_runs/rlz_<r>/output/`. **Declare `cst_*.csv` as a
  real `input:` on rule 3.11** (today it is an undeclared runtime read at
  `export_wflow_results.py:161`). Edit surface beyond the rules:
  `downscale_climate_forcing.py:72`, `generate_weather.R:68`,
  `export_wflow_results.py:281`.
- [ ] **12. `r07: wf1 evaluation and model figures into the engine subtree`**
  (B10) + climate figures from the store (B4) + declaration fixes (O-24) + the
  `plot_map.py` sentinel guard (O-08).
  - B10 retires `{project_dir}/plots/wflow_model_performance/`: rule 1.11 →
    `hydrology_model/evaluation/` (+ `plots/`), rule 1.12 `basin_area.png` →
    `hydrology_model/plots/`, rule 1.13 → `hydrology_model/forcing/plots/`, rule
    1.14's gather inputs repointed. Three module constants move with them:
    `plot_results.py:108`, `plot_map.py:34`, `plot_map_forcing.py:201`.
  - B4: **new rule 1.15 `plot_climate_source`**, declared in
    `Snakefile_model_creation` **only**, script
    `blueearth_cst/climate_analysis/plot_climate_source.py` (new). Inputs:
    `<key>/extract_historical.nc` (+ `orography.nc` on the chirps branch) **plus
    the data catalog as params** (era5 branch needs `era5_orography`). Outputs
    `source_precip.png`, `source_temp.png`, `source_pet.png` under
    `climate_historical/<key>/plots/`; add them to rule `all` and rule 1.14's
    gather inputs. Source-grid PET **need not** match the build's PET.
  - O-24: declare the **config-invariant subset** of the missing plot outputs on
    rules 1.11 and 1.13.
  - O-08: `plot_map.py:28-31` — guard the `"None"` string so no `gauges_None`
    layer is computed.
- [ ] **13. `r07: experiment_id suggestion helper`** (B8) —
  `suggest_experiment_name` in `snake_utils.py` beside `validate_experiment_name`,
  plus a thin CLI `scripts/suggest_experiment_name.py`. Contract:
  `python scripts/suggest_experiment_name.py <config.yml>` reads
  `project.project_dir`, slugifies its basename (lowercase; every non-`[a-z0-9]`
  → `_`; strip leading non-alphanumerics; collapse `_` runs; truncate to 64),
  passes the result through `validate_experiment_name`, and writes
  `workflows.climate_experiment.experiment_name` **only if the key is absent** —
  an existing value is never overwritten (exit nonzero, naming the present
  value). Never generated at run time.
- [ ] **14. `r07: re-record the manifest`** — **the discharge `compare` gate runs
  first and must exit 0 before `record`** (§ Validation rung 5). Drop the three
  stale rows, listing their paths:
  `climate_experiment/model_results/Qstats.csv`,
  `climate_experiment/model_results/basin.csv`, and the root-level
  `config/snake_config_climate_experiment.yml`. Delete the orphaned sidecar
  `dev/baseline/discharge_ref/1f9f30a367de162f.csv` in the same commit.
- [ ] **15. `r07: docs`** — `AGENTS.md` repo map + the invocation-model line
  (`blueearth_cst/` is executed *by Snakemake*; `scripts/` executes the pipeline;
  `dev/scripts/` inspects or maintains the repository); `README.rst`;
  `MIGRATION.md` → `docs/migration-r06.md`; the `dev/reference/naming.md` § 7
  amendment (two artifact classes — a **required** internal
  `dev/<milestone>/migration_<topic>.md` rename record, and an **optional**
  user-facing guide under `docs/`; the mandated form overrides § 8's kebab-case
  rule for `dev/` markdown). § 7 then also requires **reconstructing R06's
  internal rename record** at `dev/milestones/r06/migration_<PLACEHOLDER-topic>.md` from the
  moved file's rename tables. R07 publishes **no** user-facing guide.

**Drive-by fixes** — separate small commits, independent of the above: O-07
(`prepare_cst_parameters.py:14` sys.path depth), O-09 (`plot_results.py:83`
separator docstring), O-10 (`MIGRATION.md` `__init__.py` list — **land this before commit 15 or against the
new `docs/migration-r06.md` path**), and O-13 (delete the unreferenced
`blueearth_cst.Rproj`). **O-13 was ruled *delete* at G1 but assigned to no
commit** — land it as a drive-by unless the owner directs otherwise. **O-08 is
not a drive-by**; it rides in commit 12.

### Explicit non-moves — do not "fix" these

The review removed work as well as adding it. Doing any of the following will
blow the semantic diff or break a contract:

- **B3:** the three `climate_projections/<clim_project>/plots/` PNGs **do not
  move**.
- **B9:** `experiments/<id>/config/snake_config_climate_experiment.yml` **stays
  where it is** — it does not join `config/runs/`; its content changes only.
- **The `semantic_tree_diff` TOML comparator needs no change.** It already covers
  all five pointer fields generically. B5 needs a new **path map**, not a new
  comparator.
- **Rule 3.00b is untouched.** It keeps both outputs; `{store_dir}/.guard_ok`
  survives as the store-level receipt of the last consistency check. Only its
  **DAG edge** retires.
- **`setup_gauges_and_outputs.py:55` and `plot_results.py:127` are correct.**
  Both read `if X is not None and os.path.<exists>(X):` and short-circuit, so
  `null` raises nothing there. Only `plot_map.py:28-31` is defective (O-08).
- **O-24 is deliberately incomplete.** `plot_basavg` (per `wflow_outvars`),
  `signatures_{station}.png`, and per-station `clim_{station}_{period}.png` stay
  knowingly undeclared. Deriving the list at parse time is a **rejected
  alternative** — it is a rule-shape change, out of scope.
- **`COPIED_CONFIG_PATH_MAP`:** `_is_copied_config` (`semantic_tree_diff.py:576`)
  matches any YAML with a `config` path part, so the new
  `experiments/<id>/weather_generator/config/weathergen_config.yml` is newly
  swept into that directional policy. Intended — state it in the migration map
  rather than letting it read as a diff regression.

### Validation

Report every rung you ran, per commit.

1. **Narrow.** Commit 1: `pytest tests/test_semantic_tree_diff.py
   tests/test_check_baseline_scope.py`. Every commit touching a Snakefile or a
   config: `pytest tests/test_cli.py`. Plus the **corrected** sentinel assertion —
   with `output_locations: None`, `basin_area.png` must **not** be produced from
   a `gauges_None` layer (asserting the string parses is not enough).
2. **New behavioural tests.**
   - Commit 1: merge-class cases in `tests/test_semantic_tree_diff.py`.
   - Commit 6: three `warn_if_project_dir_in_repo` unit cases (in-repo warns;
     `<repo_root>/test_case/...` silent; absolute out-of-tree silent) **plus** one
     `test_cli.py` case asserting the warning text reaches the combined stream.
   - Commit 7: **contract-equality test** — parse both workflows, assert
     `extract_climate_grid` exists in each with an identical normalized contract:
     rule name, script, input set, output paths, params, **and every content- or
     execution-affecting directive** (`conda`, `container`, `envmodules`,
     `wrapper`/`notebook`, `shadow`, `threads`, `resources`, `priority`,
     `retries`, `group`, `cache`, `wildcard_constraints`). **Deny-by-default
     allowed-local set: exactly `message`, `log`, `benchmark` may differ**; a
     non-default value on either declaration for any directive outside the test's
     known universe **fails**. Plus: **catalog-staleness** (edit a relevant
     catalog definition → the next invocation of either workflow schedules
     `extract_climate_grid` **exactly once**; afterwards `--dry-run` in **both**
     schedules nothing), **bbox agreement** (`store_region.geojson` bounds vs
     `staticmaps.nc` bounds, per-edge tolerance 2 × model resolution — this is
     the retired `allclose` check's successor and the configuration-independent
     invariant), **hydrography cross-check raises**, and **chirps-branch
     `oro_path` resolves to the emitted `orography.nc`**.
   - Commit 12: the **P4 assertion** — the three source figures build with
     **neither `hydrology_model/` nor `config/templates/wflow_build_model.yml` on
     disk**; a source-PET unit test; `snakemake --delete-all-output` removes the
     newly declared outputs (claimed for the **seed-config class** only).
   - Commit 13: `suggest_experiment_name` slug cases + refuse-overwrite.
3. **Integration.** All three Snakefiles `--dry-run` clean **independently**,
   under both the Windows seed config and
   `config/workflows/snake_config_model_test_linux.yml` (+ its Linux catalog).
   *If the Linux dry-run is blocked for a path reason, the authorized fallback is
   a referential-integrity test (the config and runner reference no repo path
   that does not exist) — **state the downgrade, never take it silently**.*
   After commit 7: a wf1-only run on a fresh `project_dir` builds the store with
   no `MissingInputException`; then wf1 → wf3 via `scripts/run_workflows.py` must
   report **nothing to be done** for the store rule, **and** a subsequent wf1
   `--dry-run` must schedule nothing (both alternation directions).
   `scripts\run_snake_test.cmd --dry-run` must write only under
   `test_case\test_local\dag\`.
4. **Full gate.** `pytest tests/`; CI green on both legs. The pre-R07 reference
   counts are **386/30/1 `windows-latest`, 385/31/1 `ubuntu-latest`**; commits
   that add tests raise the passed count, so the invariant to hold is **zero
   failures and no new skips**, not literal count equality.
5. **Baseline / non-regression.**
   - **Per slice, after each of commits 7, 8, 11, 12** — not once at the end:
     `semantic_tree_diff.py` against the retained pre-R07 reference tree with the
     R07 path map + merge class.
   - **The B1 merge comparison is exception 3's proof.** The survivor must match
     `wf1_raw/extract_historical.nc` **and** the pre-R07 `<key>/extract_historical.nc`,
     both element-wise via `compare_nc`, **both passing**. The two sides are not
     symmetric: the R07 bbox is bit-identical to today's wf1 bbox, so a failure
     on the `wf1_raw/` side means something *other* than the bbox changed; the
     `<key>/` side carries the risk (that store was cut to 6-dp-rounded region
     bounds). Read a single failure against that asymmetry before invoking the
     branch — and see Gate 2.
   - **Discharge anchor, commit 14, before `record`:**
     `python dev/scripts/check_baseline.py compare --ref <saved-commit-1-path>
     --cur test_case/test_local/hydrology_model/run_default/output.csv` must exit
     0. This is the milestone's real numeric anchor.
   - **Final:** a full three-workflow run on the seed config; full-`project_dir`
     pre/post diff with **MISSING/EXTRA empty modulo the written allowlist**, all
     values identical; then `check_baseline check` green against the re-recorded
     manifest.

### Acceptance criteria

- 15 `r07:` commits (plus drive-bys) in the design's order, each leaving the tree
  runnable. **The milestone has exactly one completed state — after commit 14**
  (docs commit 15 may trail).
- Full-tree semantic diff clean modulo a written, justified allowlist; all values
  identical.
- The **P4 assertion** demonstrated: climate figures produced with neither
  `hydrology_model/` nor the wflow build template on disk.
- The B1 merge comparison passes on **both** sides.
- Discharge `compare` exits 0 before the re-record; manifest re-recorded exactly
  once; `check_baseline check` green afterwards.
- The contract-equality test is in the suite and green — it is the only thing
  standing between a future editor and reintroducing a cross-DAG asymmetry.
- No file remains at a path the migration map says has moved.
- **Rollback.** If a per-slice diff shows any *value* difference outside the
  three named exceptions and the adjudicated config snapshots, revert the
  offending commit — the milestone's premise is that no computational path
  changes. Abandoning mid-flight means reverting the landed `r07:` commits; the
  pre-R07 manifest is then valid against the reverted tree, so **no re-record is
  needed in either direction**.

### Output requirements

- The commits, plus the appended allowlist and notes in
  `dev/milestones/r07/migration_project-layout.md` (see § Allowed scope, Scoped write).
- A short results note — **returned in the final response, not written as a
  file** — recording: the per-slice and final diff outcomes, the
  merge-comparison result on both sides, the discharge `compare` result, the
  manifest re-record, and the test/CI counts before and after.
- **Results delta.** Expected new artifacts: `store_region.geojson`, the three
  `source_*.png` climate figures, and the newly declared plot outputs — all
  allowlisted EXTRA-by-design. Any *other* EXTRA, and any MISSING outside the
  three stated orphan deletions, is a defect, not a delta.

### Task constraints

- Change no computed value beyond the three named exceptions, and read Claim 1 as
  **scoped to the seed-fixture class**.
- Preserve the `"None"` string sentinel **byte-identically** — every existence
  guard downstream depends on it. Unquoted `None` parses to the Python string,
  not YAML `null`.
- `hydrology_model/` stays the hydromt `model_root` (option A). No nested
  `model/` subfolder.
- **Stay within CST's automation scope** (`AGENTS.md` Hard Constraints): consume
  hydromt / hydromt_wflow / Wflow conventions verbatim; never patch a vendored
  package.
- TOML relative pointers change depth in commit 11 — let hydromt re-relativize on
  write and supply a new **path map**. Never hand-edit the pointer strings.
- Do not hand-edit `test_case/` contents; regenerate by running.
- Add no new dependency without asking the owner first.
- Commit messages for commits 4–14 must note that a red `check_baseline check` is
  expected in that window and is not a regression signal.

**Human gates — PAUSE at each.**

- **Gate 0 — before commit 2.** Confirm the **destructive-change set** with the
  owner (not the decision, which is accepted): the tracked `data/` tree,
  `docs/config/`'s 16 files, `blueearth_cst.Rproj`, and the commit-4 fixture
  rename. Requires the commit-1 reference capture to already be on disk.
- **Gate 1 — commit 6, the pause point.** The tree is runnable and `pytest
  tests/` is green, but the baseline gate is red and this is **not** a releasable
  cut. Report rungs 1, 3, 4. If work pauses here, the retained pre-R07 reference
  tree and saved discharge series **must be preserved** for the resume, and the
  commit message must flag the pause as temporary.
- **Gate 2 — commit 7, if either merge comparison fails.** Do not proceed on your
  own judgement: the design's exception-3 branch (a–d) requires listing
  `clim_wflow_1_*` and the **wf3 indicator targets** as expected-to-move,
  recording per-edge deltas, and rewriting the exit adjudication. **If
  `hydrology_model/run_default/output.csv` moves at all, STOP AND ESCALATE** —
  that is outside every branch this design authorises.
- **Gate 3 — after commit 13, before the commit-14 re-record.** Report rung 5
  against the pre-R07 reference. The owner adjudicates the MISSING/EXTRA
  allowlist. **Do not re-record the manifest before this gate releases.**
