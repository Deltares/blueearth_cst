# Task Brief — R11 P2: run identification, the design table, and `st_id`

### Context

Canonical rules: `AGENTS.md`. Scope and rulings:
`dev/milestones/r11/wf3-consolidation-scope.md`. **Normative specification:**
`dev/milestones/r09/wf3-change-requests.md` CR-4 (C22–C28), CR-5b (C34) and F7,
with `wf3-changes-proposal.md` as the reviewable companion. This brief bounds the
unit; it does not restate the spec.

- **P1 landed first and settled the table shape** this phase adds a column to:
  six columns, one table per variable, locations in rows. See
  `phase-1-result-tables-task-brief.md` and
  `migration_indicator-tables.md`.
- **`cst_` → `st_` removes an inconsistency rather than inventing a word.** The
  code already says `st` everywhere that matters — the Snakemake wildcard is
  `st_num`, the count is `ST_NUM`, the helper is `stress_test_grid()`, the config
  section is `stress_test:`. Only the filenames and catalog keys disagree.
- **`rlz_` stays.** Unlike `cst`, it abbreviates a correct term and collides with
  nothing. `mem_` was considered and rejected: it is already WF2's word for CMIP
  members, a genuinely different thing.

### Goal

Members are named `st_<m>` in every filename and catalog key; a
`stress_test_design.csv` answers "what is run 37?"; and the indicator tables carry
`st_id` alongside the perturbation columns, with the consistency check that keeps
a denormalised copy honest.

### Non-goals

- Unit D (config surfaces, C30–C33) — deferred, spec complete in the register.
- The v2 execution model — R12.
- The baseline re-record and the full three-workflow run — **P3**. P2 leaves the
  baseline red, like P1.
- Renaming `rlz_`.

### Allowed scope

**Permitted**
- `Snakefile_climate_experiment` — member filenames, catalog keys, the design
  table's rule, `wildcard_constraints`
- `blueearth_cst/experiment/` — `prepare_cst_parameters.py`,
  `downscale_climate_forcing.py`, `export_wflow_results.py`,
  `prepare_weagen_config.py`, `run_wflow_batch.jl`
- `blueearth_cst/shared/` — `snake_utils.stress_test_grid` (C26 extends it),
  `merge_logs.py`, `interchange_contracts.py`, `indicator_tables.py`
- `blueearth_cst/weathergen/generate_weather.R`
- `blueearth_cst/climate_analysis/prepare_climate_data_catalog.py`
- `tests/`, `dev/reference/contracts/`, `dev/reference/naming.md` §4,
  `dev/scripts/semantic_tree_diff.py` (**`build_project_tree_rules` only** —
  `build_r09_path_map` is frozen; see BOUNDARY below),
  `dev/milestones/r11/`

**Approval-gated**
- `dev/scripts/check_baseline.py` — only if a target path changes shape

**Forbidden**
- `dev/baseline/manifest.json` — P3 owns the single re-record
- `dev/milestones/r09/wf3-change-requests.md` — settled; if wrong, PAUSE
- Anything under unit D

### Required changes (checklist)

1. **`cst_` → `st_`** across member filenames, the WG-5 catalog entry keys
   (`rlz_<n>_cst_<m>`), and the reserved baseline `cst_0` → `st_0`.
   `wildcard_constraints: st_num=[1-9][0-9]*` is unaffected — the wildcard name
   does not change.
2. **`stress_test_design.csv`** (C23): one row per design point, one column per
   stress dimension, plus a row for `st_0` with every change zero.
3. **Two id spaces, not one** (C24): `st_id` is the designed axis, `realization`
   the sampled one. Run identity stays `(rlz, st)`.
4. **Experiment-scoped ids** (C25), written beside the config snapshot.
5. **One enumeration, two consumers** (C26): the routine that expands the DAG also
   writes the table, so they cannot disagree. Extend `stress_test_grid()`.
6. **Id width from the count** (C27), not fixed at three digits.
7. **`st_id` in the results tables** (C28) — the header becomes seven columns —
   **with both obligations, neither optional:**
   - `validate_hm7` asserts `temp_change`/`precip_change` agree with the design
     table's row for that `st_id`. A denormalised copy nothing verifies is a copy
     that eventually lies.
   - The writer **raises** when `stress_test:` gains a third axis, naming C28,
     rather than silently adding a column.
8. **C34** — one recorded decision per unpassed weathergenr argument: surface it
   or accept the default deliberately. Minimum to surface: `save_plots`,
   `pet_method`, `seed`.
9. **F7** — declare `config/templates/weathergen_config.yml` as an `input:` of
   rule 3.04. One line. Orphaned by unit D's deferral; the register says
   explicitly not to let it fall between the two.
10. Update `naming.md` §4, both seam contracts, `rule-index.md`, and
    `build_project_tree_rules` in `semantic_tree_diff.py`. Record the rename per
    §7. Do **not** touch `build_r09_path_map` or any `dev/milestones/**` record —
    see BOUNDARY below.

### Commit plan

The rename is a mechanical transform whose *scope* is the risk; the design table
is new capability. Keeping them apart is what makes a failure attributable.

| # | Subject · paths | Invariant it preserves |
|---|---|---|
| 1 | `refactor(wf3): rename the member token cst_ to st_` · every producer and consumer of a member path or catalog key, in ONE commit | An atomic move+rewrite. A partial rename leaves the DAG referring to files nothing writes, and Snakemake fails at the first missing input rather than at the mistake |
| 2 | `feat(wf3): enumerate the stress-test design table` · `stress_test_grid` + the new rule + tests | The DAG expansion and the table come from one enumeration, so they cannot disagree — the property C26 exists for |
| 3 | `feat(wf3): carry st_id in the indicator tables` · writer + `validate_hm7` + seam doc + tests | HM-7 is a pinned contract: the seventh column and the check that keeps it honest land together |
| 4 | `chore(wf3): record the weathergenr argument decisions; declare the 3.04 template` · C34 + F7 | Independent of the rename; separated so F7's one-line fix is not lost inside it |

### Validation

**The rename's own falsifier matters more than usual.** A rename that half-lands
is the failure mode, and a green unit suite does not detect it.

| Property | Falsifier | Command |
|---|---|---|
| No LIVE member path still says `cst_` | any `cst_[0-9{]` in the renamed surfaces | `git grep -nE "cst_[0-9{]" -- Snakefile_climate_experiment blueearth_cst tests config` |
| The DAG is internally consistent | `--dry-run` names a missing input | `pytest tests/test_cli.py` |
| Catalog keys moved with the files | `validate_wg5_catalog_grid` reports an unexpected key set | `pytest tests/test_interchange_contracts.py -k catalog` |
| `st_id` agrees with the design table | perturb one row's `temp_change`; `validate_hm7` must report it | new test in `test_interchange_contracts.py` |
| A third stress dimension refuses | add one to a synthetic config; the writer must raise naming C28 | new test in `test_export_wflow_results.py` |
| F7 actually re-triggers 3.04 | touch the template; 3.04 must be scheduled | `snakemake --dry-run` before/after |

**BOUNDARY — records are not migrated (ruled 2026-08-08).** `cst_[0-9{]` hits
~65 files, and a large share are RECORDS rather than live surfaces. Rename only:

    Snakefile_climate_experiment, blueearth_cst/**, config/**, AGENTS.md,
    dev/reference/contracts/*-seam.md, dev/reference/naming.md,
    dev/reference/workflows/rule-index.md, dev/scripts/inspect_spatial_ref.py,
    tests/** EXCEPT the three historical-map files below

**Leave untouched**, deliberately: every `dev/milestones/**` design doc,
inventory (`declared_inventory.txt`, `observed_inventory.txt`), probe output
(`_forceall_dryrun.txt`, `_batch_dryrun_demo.txt`) and gate record;
`dev/reviews/**`; `dev/roadmap.md`'s historical phase entries;
`docs/migration-r08-wf2.md` (a past release's user migration note); and
`dev/reference/workflows/climate_experiment.md`, which is registered in
`sealed-records.yml` and whose edit fails `tests/test_sealed_records.py` by
design.

Those describe what was true when written. `AGENTS.md`: *"freshening its paths is
worse than leaving it: the line numbers, rule names and module locations still
lie, while the document now looks maintained."* That is R9 P5 F2 verbatim. The
migration note carries the old→new map, which is how a reader of an old record
translates.

**Anything that TESTS or DOCUMENTS a historical migration map is frozen too** —
found the hard way, 2026-08-08, by sweeping `tests/**` wholesale and having to
revert. `tests/test_r09_path_map.py`, `tests/test_semantic_tree_diff.py` and
`dev/scripts/semantic_tree_diff.py` all exercise or illustrate the P3-1, R07 and
R9 migration maps, whose eras used `cst_`. Renaming their expectations would make
them assert that a frozen map produces `st_`, which it does not — the tests would
fail, and "fixing" them would rewrite the record. They keep the old token, and
that is not a half-landed rename.

**The R9 path map stays frozen.** `build_r09_path_map` keeps `_cst_` on BOTH
sides: it validates a migration between two eras that both used `cst_`, and
`test_r09_path_map.py` exercises it against pre/post-R9 trees, neither of which
is a P2 tree. Only `build_project_tree_rules` — the post-migration inventory —
learns `st_`. Updating both would make the R9 map claim a migration that never
happened in that form.

**`cst_` HAS THREE MEANINGS. Only one of them is the member token.** Established
2026-08-08 after two failed sweeps; do not attempt a regex over bare `cst_`.

| form | meaning | rename? |
| --- | --- | --- |
| `blueearth_cst` | the package | **NEVER** — 874 hits; renaming breaks every import |
| `cst_0`, `cst_{st_num}`, `cst_<m>`, `cst_"+"{…}"`, `cst_*.csv` | the stress-test member | **YES** |
| `cst_calendar`, `cst_raw_digest`, `cst_source`, `cst_acquired`, `cst_region`, `cst_schema`, `cst_series_*`, `cst_time` | **WF2 netCDF attribute prefixes** meaning "written by CST" | **NEVER** — these are provenance attrs in `blueearth_cst/projections/`; renaming corrupts WF2 output |

**Substitute an explicit list of member forms, not a pattern.** The member token
appears as: a digit (`cst_0`), a format placeholder (`cst_{st_num}`, `cst_{c}`,
`cst_{i+1}`), a doc placeholder (`cst_<m>`, `cst_<c>`, `cst_<n>`), a glob
(`cst_*`), and — the one that broke the first attempt — **a string-concatenation
boundary in the Snakefile**: `f"…/rlz_"+"{rlz_num}"+"_cst_"+"{st_num}"+".nc"`,
where the character after `cst_` is a quote.

**Why the first attempt was not caught by the suite.** `cst_[0-9{]` renamed
`prepare_cst_parameters.py` (which writes `st_1.csv`) but NOT the Snakefile's
concatenated `output:` declaration (still `cst_1.csv`). A script's behaviour
disagreeing with its rule's declaration is invisible to `--dry-run`, so
`test_cli` passed against a broken pipeline. **The rename's real falsifier is a
run, not a dry-run** — or, cheaper, a grep asserting the producer and the
declaration agree.

**DANGER — the package is named `blueearth_cst`.** Measured 2026-08-08: **1413**
tracked occurrences of `cst_`, of which **874 are the package name** and only
**65 files** contain a genuine member token. A blanket `cst_` → `st_`
substitution renames the package and breaks every import in the repo. Match
`cst_[0-9{]` (a member index or a wildcard brace), never bare `cst_`, and never
`sed -i` the tree.

**Ladder** — report what each rung *caught*:

| Rung | Scope | Frequency |
|---|---|---|
| 1 | the touched module's own tests | per edit |
| 2 | the new behavioural tests above | per edit |
| 3 | `pytest tests/test_cli.py` — the DAG changes shape | after each commit |
| 4 | `pytest tests/` **from the primary checkout or a SEEDED worktree** — the fixture layer cannot run otherwise (`AGENTS.md`) | once per commit |
| 5 | `pixi run tree-check` — member filenames move, so the path map moves | once, after commit 1 |
| 6 | Baseline: **expected red.** Record which targets moved; do **not** re-record | at phase end |

### Acceptance criteria

- Checklist 1–10 complete; the four commits landed in order
- Rungs 1–5 green; rung 6's red documented target-by-target
- `git grep -nE "cst_[0-9{]"` clean across the LIVE surfaces named above;
  records still carry the old token, deliberately, and that is not a failure
- The package `blueearth_cst` is untouched — `git grep -c "blueearth_cst"` unchanged
- **Rollback:** if the rename cannot be completed atomically in commit 1, revert
  it rather than landing a partial one; a half-renamed tree is worse than none

### Output requirements

A **Results delta** naming each moved target and why, for P3's re-record. Plus
the C34 decision table — one row per unpassed argument, with *surfaced* or
*default accepted deliberately* and the reason.

### Task constraints

- Do not re-open ruled decisions. If the spec looks wrong, PAUSE.
- The rename is one atomic commit. Do not stage it.
- Keep the register and `wf3-changes-proposal.md` in step if either is touched.

**Human gates**

1. **After commit 1, before commit 2** — report the rename's grep evidence and
   `tree-check` result. A rename that half-landed must be caught here, not at
   phase end.
2. **Before touching `check_baseline.py`** — approval-gated; a changed target path
   affects what P3 re-records.
3. **At phase end** — report the baseline delta and STOP. P3 owns the re-record.
