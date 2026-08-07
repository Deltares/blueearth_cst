# Followups — archive

Closed followup items. One brief entry each: what it was, how it ended, and the
date. **Nothing here is open** — an item with anything left to do stays in
[`followups.md`](followups.md), however much of it is already done.

Why this file exists: `followups.md` reached 2,038 lines, of which roughly half
were items already resolved. The forensic write-ups had done their job — every
durable lesson in them had already been promoted into the code, the tests or
`dev/reference/`, so the long entries were duplicates of guidance that lives
somewhere it is actually consulted. They are compressed here rather than
deleted, because the **item IDs are cited from code, tests and Snakefiles** and
must stay resolvable.

**Recovering the full text.** Every entry below existed in full at commit
`fdb1d09` (2026-08-07). Read one with:

```bash
git show fdb1d09:dev/followups.md          # the whole pre-compression file
git log -S'[R10-5]' --oneline -- dev/followups.md
```

---

## Post-R10-design (2026-08-06)

- **[R10-1] Merge rule 1.07 `setup_runtime` into 1.08 `add_forcing`. DONE
  2026-08-06.** Landed as a `script:` driving hydromt's CLI, so the command is
  byte-identical to the old `shell:` rule; 1.07's `gauges_path` input was dead
  and dropped. Executed in the R10 three-workflow run — WF1 17/17, and
  `run_default/output.csv` byte-identical to the baseline.
  (The `tee_to_log` trap it surfaced is documented at `snake_utils._Tee`.)

- **[R10-2] Split rule 1.11 into a metrics rule and a figure rule. DROPPED
  2026-08-06 by owner ruling.** Implementation showed the seam does not exist:
  the metrics are ~5 lines inside the figure loop, behind a shared prefix that
  includes the expensive climate-parity transform. The harm was a wasted re-run,
  not a wrong number. Consequence: `evaluate_` withdrawn as the 19th verb.
  **Accepted gap:** the DAG still cannot express "figure-only change".

- **[R10-3] Two rule consolidations REJECTED 2026-08-06, recorded so they are
  not re-raised.** M2 (`write_outlet_index` into 1.05) would *add* a DAG edge;
  M3 (`gather_benchmarks` + `gather_logs`) would make a partial failure delete
  log parts it then could not rebuild. Also do not merge 3.01c/3.01d — the
  `ancient()`/`temp()` asymmetry *is* the guard.
  Generalized in `dev/reference/workflows/rule-index.md` § M2/M3.

- **[R10-4] Stale rule references in Snakefile comments. DONE** (cleared by the
  R10-5 sweep; verified 2026-08-07 — `prepare_weagen_config_st` survives only in
  an explicit "is GONE" comment, and no `benchmarks.tsv` reference remains).

- **[R10-5] Renumber every rule so `W.NN` follows the DAG. DONE 2026-08-06.**
  47 rule declarations across three workflows, renumbered in one pass keyed on
  (old number, rule name) so a freed number could not be reused mid-sweep; two
  audits confirmed every `W.NN` resolves to a live rule. **Numbers are now
  REUSED**, so a stale reference resolves to a *different* rule — read any
  `W.NN` in `dev/milestones/` as of its date, and do not renumber to insert a
  rule (use a letter suffix). Map: `dev/reference/workflows/rule-index.md`.

- **[R10-7] Rename the shared-rule helpers `_spec` → `_rule`. DONE 2026-08-06**,
  20 files. Both flagged hazards bit and were caught: the symbol inside an error
  *message string*, and `dev/decisions/0003`, where a blanket substitution
  collapsed two old→new arrows — the failure mode of renaming inside a document
  that documents the rename. Convention now in `dev/reference/naming.md` §5.

- **[R10-8] `Snakefile_climate_projections` listed a `LOG_RULES` label with no
  rule. DONE 2026-08-06.** The phantom `2.11_extract_climate_grid` entry made
  every WF2 merged log carry a "no part from this run" section forever. Deleted;
  `tests/test_log_rules_contract.py` confirms no orphan and no unlisted label.

- **[R10-10] `test_model_reference.py`'s `LOG_RULES` slicer stopped at the first
  `]`. DONE 2026-08-06 — folded, not fixed.** The check moved wholesale into
  `tests/test_log_rules_contract.py`, because two modules asserting one property
  by different parsers is how they came to disagree. Folding was strictly
  stronger: the old regex never saw the fan-out rules' concatenated `log:` paths.

- **[R10-11] `pixi run tree-check` could not pass on a correctly-migrated tree.
  FIXED 2026-08-06.** The R9 map runs one way (pre→post), so a live tree matched
  nothing: 153 of 186 paths UNMAPPED, exit 1, on every correct tree. Replaced by
  a post-migration inventory as the default; `--map r09` keeps the one-way map.
  Now 186/186 identity, exit 0. Covered by `tests/test_project_tree_inventory.py`.

## Post-R9 (2026-08-05)

- **[R9-3] The response-surface axis columns held JANUARY, not an annual value.
  FIXED 2026-08-07.** `export_wflow_results.annual_perturbation` now collapses
  each `cst_<m>.csv`'s twelve monthly rows as a month-length-weighted mean
  (owner ruling). The rule was fixed by the CMIP6 overlay, which shares those
  axes with WF2's `_annual` definition. Not a baseline event — every tracked
  config is flat, so the values are unchanged. Contract: HM-7 in
  `dev/reference/contracts/hydrological-model-seam.md`; residual approximation
  on the precip axis is in the function's docstring.

- **[R9-4] R9 moved the project tree but never re-pointed the interchange
  contract tests. FIXED 2026-08-05.** 22 failures on the first post-R9
  `pytest tests/`; three paths sat behind an `os.path.exists` guard and so
  **skipped silently** rather than failing. Fixed by deriving four roots named
  after the Snakefile variables they mirror. Lesson promoted to `AGENTS.md`.

## Post-R7 (2026-07-28/29)

- **[R7-1] `wflow_sbm.toml` was written by five rules and declared by one.
  FIXED 2026-07-29.** Rule 1.03 emits a `touch()` sentinel that 1.04 consumes as
  a **non-ancient** input, so a rebuild re-fires the whole toml-writing chain.
  The obvious fix — dropping `ancient()` on staticmaps — was wrong: 1.04/1.05
  write back into staticmaps, so a plain edge would re-trigger them forever.

- **[R7-2] The store's freshness boundary stops at the catalog file. WON'T FIX,
  2026-07-29 (owner-ruled).** Enumerating catalog-resolved sources as DAG inputs
  means re-implementing hydromt's data resolution, which `AGENTS.md` Hard
  Constraints forbid; and remote sources expose no mtime, so it would silently
  cover only local entries. Escape hatch: `--forcerun extract_climate_grid`.
  Revisit only if hydromt gains a first-class entry→sources API.

- **[R7-3] Which change produced the 134,828-byte `basin_area.png`? ANSWERED
  2026-07-29.** A different branch (`feat/outputs-figures`) wrote it into the
  shared untracked fixture; the artifact outlived the checkout. Not a defect.
  Generalized as R7-21.

- **[R7-4] Import direction in the model-free producer. FIXED 2026-07-29.**
  `climate_parity.py` moved `model/` → `shared/` — misfiled, not miscoupled.
  Pinned by a test that walks the package's ASTs.

- **[R7-6] Declaring `clim_wflow_1_*` made rule 1.11 newly able to fail. FIXED
  2026-08-01.** Subsumed by `snake_utils.MIN_HISTORICAL_YEARS` — one floor for
  the toolbox, checked against the requested window at parse time and against
  the extracted span in the store producer. It still fails loudly; only *where*
  and how legibly changed.

- **[R7-7] The contract-equality test pins Snakemake 9.6.2's directive set.
  ACCEPTED — NO ACTION, 2026-08-02.** The loud failure at a version bump is the
  designed behaviour and is cheaper than the silent hole.

- **[R7-9] Stale benchmark parts survive a rule rename. NO ACTION, 2026-07-29.**
  Self-healing: `merge_benchmarks` deletes every part it merges, so a phantom
  row appears in exactly one report. A guard would mean teaching the merger the
  rule list, which it has no other reason to know.

- **[R7-10] Old-path references in documents commit 15 did not own. FIXED
  2026-07-29.** Includes `dev/milestones/p32a/compare_climate_ladder.py`, marked
  SUPERSEDED rather than repointed — its premise had dissolved, so repointing
  would have left it comparing a store against itself.

- **[R7-11] `plot_map_forcing.py`'s `"None"`-string shape. NO ACTION,
  2026-07-29.** The derived name is consumed by a membership test, which is
  itself the guard, and rule 1.13 passes a real declared path. Structurally
  unlike O-08, where the name was built and used with no check.

- **[R7-12] The tests config warns on every dry-run. WORKING AS INTENDED,
  2026-07-29.** `project_dir: tests/test_project` is in-repo and outside the
  single `test_case/` exemption, so O-22's warning is correct. Not widened.

- **[R7-13] Map §2c's depth arithmetic is off by one. FIXED 2026-07-29**,
  against the emitted TOMLs. Recorded as a correction rather than quietly
  amended: the map deferring to the comparator is what kept the error harmless.

- **[R7-15] Engine-named subtrees. DELIVERED by R9 P2.** The tree is now
  `models/hydrology/wflow/` — domain, then engine — with the symmetric
  `experiments/<id>/{climate/weathergenr,hydrology/wflow}/`. **Not claimed:**
  that a second engine has been tried. R7's narrowing to *separability* still
  describes what is proven.

- **[R7-19] Branch unmerged, tag unapplied, roadmap stale. RESOLVED
  2026-07-29.** Merged `--no-ff` (`0ea3918`), tagged `r07-layout`, pushed; CI
  green on both legs (run 30450296441).

- **[R7-21] The baseline fixture is branch-shared mutable state. MITIGATED
  2026-07-29.** `test_case/test_local` is untracked, so every branch and session
  writes the same tree and `check_baseline check` answers for whichever branch
  ran last. `record` now stamps `recorded_by` (branch, commit, dirty) and
  `check` prints provenance before the verdict — advisory, never changing the
  exit code. Simulated in `tests/test_check_baseline_provenance.py`. The
  underlying sharing is unchanged; branch-derived paths and per-branch
  regeneration remain candidates if misattribution recurs.

## Cross-cutting and pre-R7

- **Baseline rebuilt from a tracked seed config. RESOLVED 2026-07-18.**
  `dev/baseline/manifest.json` re-recorded from the now-tracked seed config
  after a fresh three-workflow run; `record` → `check` round-trips clean.
  Replaces the stale M2b manifest seeded from an untracked `*_local.yml`.

- **The user's gauges were dropped in silence, everywhere. FIXED 2026-08-01.**
  Found on a real basin run and reported by the owner as "output locations
  missing from the spatial plots". hydromt_wflow's `setup_gauges` normalizes the
  basename (`.replace("_", "-")`), so `output_locations.csv` became
  `output-locations` in the staticgeoms layer, the wflow TOML `map`, and the
  parsed output columns.

- **wf1's `| tee {log}` shell rules masked the exit code on failure. RESOLVED
  2026-07-21** (`d13ba37`). The three shell rules route through
  `run_logged.py`, a portable Python tee that keeps live console output, writes
  the log, and exits with the child's own return code. Verified with a
  deliberately-failing child — the old `| tee` masked it to 0 under `cmd.exe`.

- **Redo the M1 warnings triage exhaustively. RESOLVED 2026-07-21.** Swept 82
  captured `.log` files across all three workflows. Bucket 3 (our code): empty.
  Bucket 2: one item, intended hydromt behaviour (a resolution snap), won't-fix.

- **`extract_climate_grid` silently truncated the historical range, and ignored
  the `historical:` config. CLOSED 2026-08-01**, both halves. The truncation now
  emits an advisory when the extracted span falls short of the requested window
  (`ce56bc3`); the hardcoded date range was replaced when R5 wired the window in
  as `params`.

- **weathergenr crashed loading on Windows. RESOLVED 2026-07-17** — root cause
  was conda-forge's r45 `r-waveslim` build, not `ncdf4`. Isolated by loading
  each Import in turn; only `waveslim` overflowed, its Fortran DLL carrying a
  32-bit pseudo-relocation out of range.

- **CMIP6 `precip`/`temp` `.attrs` lost on `monthly_change_scalar_merge`.
  RESOLVED 2026-07-21 — does not reproduce, no fix.** Under the pinned env
  (hydromt 1.3.1) the merged summary carries the full CF attribute set on both
  variables, verified on real-CMIP6-read output and in the recorded manifest.

- **conda-forge ships no `julia` for win-64.** Settled fact, not a task: Julia is
  juliaup-managed and must already be on `PATH`. Recorded in `AGENTS.md` Hard
  Constraints.

## Phase-4 sweep (2026-07-25)

- **CI. DONE 2026-07-25.** `.github/workflows/ci.yml` runs the unit suite on
  push to `main` and on PRs across both pixi platforms, `locked: true` so
  `pixi.lock` drift fails the run. `check_baseline.py` turned out **not** to fit
  — it fingerprints targets inside the untracked fixture tree, so it stays a
  local gate, as does whole-tree `semantic_tree_diff`. Recorded in `AGENTS.md`.

- **Make the projections summary CSV column order deterministic. CLOSED
  2026-07-25** — code fix landed and the manifest re-recorded. wf2 was run to
  completion and the delta proven column-order-only (every value identical when
  matched by label) *before* recording, because a value change would have meant
  something else entirely.

- **`semantic_tree_diff.py` exclusion refinement. CLOSED 2026-07-25 — already
  fixed, no action taken.** P3-1 `576b6a6` had added the run-log file rule;
  verified by calling `_is_excluded` on the three exact paths.

- **Dead-fixture audit: `tests/wflow_build_model.yml`. CLOSED 2026-07-25 —
  confirmed dead, removed.** No config pointed at it, no test loaded it, and it
  was itself broken: a dangling `read_config.config_fn` after the R6 config move.

- **`scripts/run_snake_test.cmd` modernization. CLOSED 2026-07-25 — ported, not
  retired.** `scripts/` is a documented user-facing entry point, so the surface
  was preserved and the hostility fixed: every call goes through `pixi run`,
  `pause` removed, stops on the first failing workflow, arguments forward.

- **Resolve `test_cli` xfails. CLOSED 2026-07-25 — both resolved**, each by one
  of the options the entry proposed. No `xfail` marker remains: the
  `MissingInputException` case took the `config_with_staged_region` fixture
  (R3), the `CyclicGraphException` case a rule-local `wildcard_constraints` (R5).

- **Retire the "CMIP6 GCS throughput regression" follow-up. CLOSED 2026-07-25.**
  The ~6 h estimate came from the slow path; the as-shipped run completes in
  24 min after the eager `.load()` patch in `get_stats_climate_proj.py`.
