# R11 P2 — results delta, for P3's re-record

Phase-end report, 2026-08-08. Merged to `main` as `94ab26e`. This is the input
to **P3's single baseline re-record**; it names what moves, what does not, and
what is still unverified.

## The baseline is GREEN right now, and that is not the same as "unaffected"

Measured, not assumed — `check_baseline.py check` from the primary checkout
after the merge:

    OK - 7 target(s) match manifest.

Nothing on disk changed, because **no WF3 run has happened in P1 or P2**. The
recorded artifacts are byte-identical to what they were before either phase. The
manifest goes red the moment P3 re-runs WF3, and that is the intended sequence —
the register's collect-then-implement rule exists so the contract, the
validators, the migration note and the re-record move **once** rather than per
change.

Stated because "expected red" is easy to read as "currently red". It is not. A
P3 that re-runs and finds these targets unchanged should treat *that* as the
surprise.

## What P3's re-run will move

### WF3 — three manifest entries, all from the table work

| manifest path | what happens | owner |
| --- | --- | --- |
| `experiments/experiment/results/q_indicators.csv` | **shape + values.** Header 6 → **7** columns (`st_id` inserted after `metric`, C28). Already wide → long in P1. Values unrounded since P1 (Q8), so the comparator for this target moves to a **tolerance** rather than a byte match | P1 + P2 |
| `experiments/experiment/results/basin_indicators.csv` | **GONE.** P1 (CR-2) replaced it with one table per variable in `wflow_outvars`. The seed config declares only `river discharge`, so **no basin table replaces it** — the manifest entry must be *removed*, not repointed | P1 |
| `experiments/experiment/config/snake_config_climate_experiment.yml` | the config snapshot is a copy of the tracked config, which P2 did not change. Expected **unchanged** — if it moves, something re-resolved that should not have | — |

### WF1 / WF2 — untouched

The remaining manifest targets (`config/runs/snake_config_*.yml`, the two CMIP6
change-factor CSVs, `run_default/output.csv`, and the four figure targets
excluded by default) are outside P2's blast radius. If any moves, it is not this
phase.

### Member artifacts are NOT in the manifest

Worth stating plainly, because it is the most counter-intuitive part of the
delta: **the `cst_` → `st_` rename and the zero-padding move no baseline target
at all.** No `rlz_*` / `st_*` path is recorded — the manifest covers `rule all`
outputs, and member files are intermediates, most of them `temp()`. The rename's
blast radius is large in the tree and empty in the baseline.

### Not added to the manifest, deliberately

`experiments/<id>/config/stress_test_design.csv` is new and is a `WF3_TARGETS`
entry, so `rule all` demands it — but it is **not** a baseline target.
`check_baseline.py` carries its own target templates, so adding one changes what
P3 records, and the brief gates that edit. It is also already covered by a
stronger check than a byte fingerprint: `validate_hm7` asserts every results row
against the design table's row for its `st_id`.

**P3's decision, not P2's:** whether to add it during the re-record.

## What is unverified, and will first execute in P3

Nothing in P1 or P2 has run the pipeline. Everything above rests on unit tests,
dry-runs at two member-index widths, and greps.

| surface | first executes | why it could not be checked here |
| --- | --- | --- |
| `generate_weather.R`'s new **4-argument** arity (widths passed in) | rule 3.11 | `weathergenr` comes from `pixi run install`; absent in this worktree (`requireNamespace` → FALSE). No R test harness (ruled at R5) |
| `save_plots` wired to the config's `save.plots` | rule 3.11 | same |
| `seed` on the perturbation step | rule 3.12 | same. **If `apply_climate_perturbations` is stochastic, seeding it moves numbers** — expected, absorbed by the same re-record |
| `pet_method` on the perturbation step | rule 3.12 | same |
| every member filename and WG-5 catalog key | rules 3.09–3.16 | the rename's real falsifier is a run; a dry-run cannot see a producer/declaration disagreement |

The standing caveat CR-5 recorded for C29 applies unchanged: **run WF3 from the
primary checkout before treating the R side as done.**

## Two defects the gates caught, not the author

Recorded because both are the kind that recur:

1. **A wildcard constraint that silently voided itself.** `(?!0+$)[0-9]{W}` reads
   correct and passes every anchored unit test. Snakemake embeds a constraint in
   the regex for the WHOLE path, so `$` bound to the end of the path; with `.nc`
   following, `0+$` could never match, the lookahead always succeeded, and the
   constraint degenerated to `[0-9]{W}` — admitting the baseline and making rule
   3.12 a second producer of it. **A 12 × 12 dry-run did not catch it**, because
   where the baseline is also reachable from its plural rule Snakemake prefers
   that one and the ambiguity hides. `test_cross_workflow_inputs` and
   `test_guard_invalidation` caught it as `CyclicGraphException`. Now spelled
   positionally, with a regression test pinning it *embedded mid-path*.
2. **A units disagreement C28's own check forced open.** Commit 2's design table
   wrote the raw precipitation factor (`1.3`); the results writer has always
   written a percent change (`30.0`). The two disagreed *by construction*, so the
   consistency check would have failed on a unit rather than on a defect —
   exactly how such a check rots into noise that gets waved through. Fixed by
   naming the derivation once (`perturbation_axes`).

## Coverage the phase cost, and why

Four Layer-2 integration cases went **passing → skipping**, plus one more from
C34, all on the same principle: the fixture is a pre-P2 tree until P3 re-runs, so
each asserts the POST-change shape and skips on the SPECIFIC pre-change one —
never on a file's absence, which is how R9-4 turned a wrong path into a silent
pass.

| test | skips until | condition |
| --- | --- | --- |
| `test_wg2_integration` | P3's run | old-token twin present where the new artifact is missing |
| `test_hm4_integration` | P3's run | same |
| `test_hm5_integration` | P3's run | same |
| `test_wg5_catalog_grid_integration` | P3's run | every catalog key still carries `_cst_` |
| `test_wg3_integration` | P3's run | `evaluate.model` present, `save.plots` absent |

**P3 should confirm all five un-skip after the re-run.** A green suite that still
reports these skips means the fixture did not regenerate.

## Also landed this phase

- **`t2608081012`** (board) — `tests/test_model_rebuild_cascade.py:67` guards on
  `test_case/test_local/hydrology_model`, a **pre-R9** path. The guarded test
  runs a real `snakemake all -c 1` and has silently skipped since R9. Predates
  P2 (`c058a02`); not fixed here.
- `cst_nc` → `st_nc`, the last seam-shaped residual of the token rename.
- `naming.md` §4 and §7, both seam contracts, `rule-index.md`, and
  `build_project_tree_rules` — updated in the same commits that moved the paths,
  per the scope doc's constraint.
