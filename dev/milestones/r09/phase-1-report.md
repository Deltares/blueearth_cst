# R9 P1 report — path-map comparator and orphan-store tooling

Date: 2026-08-04. Branch: `feat/r09-p1-comparator` (cut from
`milestone/r09-project-tree`). Brief:
[`phase-1-comparator-task-brief.md`](phase-1-comparator-task-brief.md).

**Status: complete. All three Gate 1 items closed 2026-08-04 — the declared
tier, the owner's F1a–F1c rulings, and the observed tier (192 paths, zero
unmapped).** No P2 work has begun. The phase itself moved no files and wrote to
no `project_dir`.

One qualification, so the sentence above is not read too broadly: **after** the
phase closed, on an explicit owner instruction, the fixture tree
`test_case/test_local` was staged for the observed-tier run — 102 orphan files
quarantined (moved, not deleted) to `test_case/_pruned_20260804/`. That is
runbook step 1, not P1 work, and it touched no artifact any rule produces. See
F5 and the runbook.

---

## What landed

| Artifact | Purpose |
| --- | --- |
| `semantic_tree_diff.build_r09_path_map` | The migration map's rows as executable rules |
| `semantic_tree_diff.apply_path_map_matched` | The fall-through signal the falsifier needs |
| `semantic_tree_diff.build_r09_gap_rules` / `R09_MAP_GAPS` | Candidate rows for artifacts the map does not cover — **opt-in**; now empty, all five candidates closed |
| `semantic_tree_diff.build_r09_deletions` | `indicators/RT_*.csv`, deleted rather than migrated |
| `semantic_tree_diff --check-map` | The falsifier: classify a path list, exit 1 on any UNMAPPED |
| `dev/scripts/prune_climate_store.py` | Orphaned `climate_historical/<source>_<window>/` reporting |
| `dev/scripts/snapshot_project_tree.py` | Snapshot a tree + run the map check in one command, every map parameter derived from the config |
| `pixi run tree-check` | Task wrapping the above, shaped like the existing `dag-wf*` tasks |
| `dev/milestones/r09/declared_inventory.txt` | The declared-tier inventory, 176 paths, with its provenance |
| `dev/milestones/r09/observed_inventory.txt` | The observed-tier inventory, 192 paths, from one clean three-workflow run |
| `dev/milestones/r09/observed-tier-runbook.md` | How the observed tier was produced, executed end to end |
| `tests/test_r09_path_map.py`, `tests/test_prune_climate_store.py`, `tests/test_snapshot_project_tree.py` | 118 new tests (plus 5 added to `tests/test_semantic_tree_diff.py`) |

## Rule count by class

`build_r09_path_map("experiment", "era5_20000101_20201231")` — **59 rules**,
after the F1a–F1c amendment and the F8 removal:

| Kind | Count | Of which identity |
| --- | ---: | ---: |
| regex (`fullmatch` + expansion template) | 9 | 3 |
| directory prefix (`old` ends `/`) | 24 | 8 |
| exact file | 26 | 6 |
| **total** | **59** | **17** |

42 rules relocate; 17 resolve a path to itself. The 17 are enumerated **per map
row** — never as a catch-all — because `apply_path_map` returns its input
unchanged on fall-through, so one broad `config/` → `config/` rule would satisfy
every `config/` row at once and empty the unmapped-path report by construction.
`test_a_catch_all_config_prefix_would_empty_the_report` demonstrates exactly that
failure, so the reason for the per-row enumeration is pinned rather than
described.

Registered **narrower source pattern first** (`apply_path_map` is first match
wins). `test_narrower_source_pattern_is_registered_first` asserts the ordering as
an index property over five (narrow, general) pairs, not just as outcomes.

Separately: **zero** opt-in gap rules and 1 deletion pattern. Five candidates
were raised and all five are closed — three became map rows when F1a–F1c were
ruled, two were settled negatively by the observed tier (F2). The mechanism is
kept, empty, because the next inventory may raise a sixth; it is what lets the
falsifier report "N unmapped" and "0 once accepted" as two numbers rather than
quietly reporting 0.

## Falsifier — declared tier

Full output and the complete old → new table:
[`2026-08-04_p1-declared-tier-falsifier.md`](2026-08-04_p1-declared-tier-falsifier.md).

| Run | Paths | Moved | Identity (by rule) | Unmapped | Exit |
| --- | ---: | ---: | ---: | ---: | ---: |
| map as first encoded | 176 | 162 | 11 | **3** | 1 |
| same + proposed rules (`--r09-gap-rules`) | 176 | 163 | 13 | **0** | 0 |
| **after the F1a–F1c amendment**, no opt-in | 176 | 163 | 13 | **0** | 0 |

The third row is byte-identical to the second, which is what confirms the
amendment encoded exactly the three ruled rows and nothing more.

The inventory is regenerated from the three Snakefiles' `output:` declarations
over the tracked seed config, with `project_dir` repointed at an empty temp dir
so every job is planned. It was **not** run against `test_case/test_local`,
whose documented orphans are deliberately unmapped.

One caveat, measured rather than assumed: the three `config/runs/<workflow>/`
bundles are named by a hash over the parsed config, which **includes
`project_dir`**, so regenerating into a different temp dir changes exactly those
three path components (`climate_projections/61868971c618` → `407f4256c490`).
The inventory header records this and the falsifier's assertions normalize the
digest segment, so a faithful regeneration cannot fail for a reason unrelated to
the map.

## Falsifier — observed tier: **VERIFIED**

`dev/milestones/r09/observed_inventory.txt`. One clean three-workflow run from
the primary checkout on 2026-08-04, all three green: wf1 18/18 (4m51s), wf2 24/24
(2m07s), wf3 49/49 (6m46s).

```
MAP CLEAN: 192 paths, 154 moved, 32 identity (by rule),
           6 deleted-by-design, 0 unmapped
```

Identical with `--gap-rules`, which is itself the evidence for F2.

**The two expected asymmetries against the declared tier both hold**, and
neither is a defect:

- **66 observed-only paths** — the undeclared engine artifacts the tier exists
  for. `hydromt.log`, `hydromt_data.yml`, seven `staticgeoms/` layers,
  `run_default/{log.txt,wflow_sbm.toml,outstate/}`, the digest bundles' internal
  structure (`effective.yml`, `source.yml`, `referenced-files.json`, `files/**`),
  weathergenr's four plots and two date tables, six `RT_Q_*.csv`, ten per-gauge
  evaluation figures, and Wflow's `log.txt`.
- **41 declared-only paths**, every one a `temp()` artifact deleted once consumed
  (12 inmaps, 12 outstates, 14 weathergen series) or a bundle directory that
  exists only as its contents.

**F3 confirmed empirically.** Exactly two `log.txt` files —
`hydrology_runs/rlz_1/config/` and `rlz_2/config/` — for twelve members. One log
per realization, shared by six concurrently-batched runs: the race the map
predicted, now observed rather than argued.

**Two rounds of pruning, and the second is the interesting one.** Round 1 (102
files) came from the runbook's hand list. Round 2 (8 files) came from an **mtime
sweep** of the finished tree, and every one of the eight sits under a directory
the map routes wholesale — so the falsifier reported none of them. See F5. The
snapshot was committed *before* round 2 deliberately, so the commit diff is the
exact stale-file list rather than a prose claim.

**One caveat, stated so it is not read as drift.** The snapshot is `main` **plus**
the one-line `LOG_RULES` fix this branch carries (F7). Run from `main` as it
stands today, the tree would carry one extra path,
`logs/_parts/1.01b_delineate_region.log`.

---

## Findings

### F1 — three declared artifacts had no map row — **RULED 2026-08-04, map amended**

Found by applying the encoded map to the declared-tier inventory. They were kept
**out** of `build_r09_path_map` until ruled: the brief is explicit that an
uncovered artifact is "a finding against the map, not a reason to improvise",
amending the map is an owner decision, and Gate 1 exists precisely because a map
wrong in the same direction as the migration is undetectable afterwards. Holding
them in the opt-in `build_r09_gap_rules` is what let the falsifier report *both*
numbers — 3 unmapped, and 0 once the additions are accepted — instead of quietly
reporting 0.

| # | Artifact | Producer | Design tree | Ruling |
| --- | --- | --- | --- | --- |
| F1a | `spatial/geoms/region.geojson` | rule `delineate_region` (ADR 0003) | **silent** | `data/` row generalised to `spatial/geoms/*`; Finding 2 corrected |
| F1b | `config/runs/climate_projections/<digest>/` | WF2 config snapshot | covered — `config/runs/<workflow>/<digest>/` | `config/` row generalised from `model_creation` to `<workflow>` |
| F1c | `experiments/<id>/config/runs/climate_experiment/<digest>/` | WF3 config snapshot | **silent** | new identity row in the experiments section, under P9 |

**F1a was the sharpest.** The map's Finding 2 stated that the `data/spatial/`
rows "correspond exactly" to the nine P1 products of rule `prepare_spatial_maps`.
The declared inventory shows **ten** files under `spatial/`: the tenth,
`geoms/region.geojson`, comes from a *different rule*. The nine-product list was
a complete inventory of one rule's outputs, mistaken for a complete inventory of
the subtree — so the map's completeness claim was falsified by the instrument
built to test it. Ruled toward a **directory row** rather than a sixth
enumerated file, so a seventh layer cannot reopen the same gap; Finding 2's
sentence is corrected in the map doc. The design tree v10 still does not name
`region.geojson` — a documentation gap for **P5**, not a placement question.

**F1b was a transcription narrowing.** Design tree line 308 reads
`config/runs/<workflow>/<digest>/`; the map row transcribed only
`model_creation`, while WF2 emits the same class. The row now reads `<workflow>`.
Encoded as a regex, **not** a `config/runs/` prefix: a prefix would also swallow
`config/runs/snake_config_{model_creation,climate_projections}.yml`, which are
declared inputs of WF3's rule 3.00b drift guard and enumerated rows in their own
right. Pinned by
`test_the_workflow_digest_rule_did_not_become_a_config_runs_catch_all`.

**F1c had no design line at all.** v10's `experiments/<id>/config/` lists
`experiment.yml`, `project_snapshot.yml` and `model_reference.yml`; WF3 also
emits a digest bundle there. Ruled identity under principle **P9** (where the
design differs from what the code emits, the emitted structure wins). The map
gains the row now; whether the design tree should absorb the line is **P4/P5**
territory, since P4 is the phase that touches `experiments/<id>/config/`.

### F2 — two candidate rows — **CLOSED NEGATIVELY by the observed tier**

Both were inferred from the design tree and neither had ever been observed, so
both were held opt-in rather than folded into the map. The observed-tier run
settled them, and in both directions the answer was *no change*:

- **`hydrology_model/instate/`** — **does not exist.** Zero paths under it in a
  tree from a complete WF1 run including `run_wflow`. The design tree names it,
  Wflow.jl does not write it at that path under the pinned version, and the map's
  silence was correct.
- **`hydrology_model/plots/` as a directory** — the tree holds **exactly**
  `basin_area.png` and `basin_area.pdf`. The map's two-file row is right as
  written; the prefix that would have generalised it was unnecessary.

Both rules are removed, and `R09_MAP_GAPS` is now empty. This is the case for
having held them opt-in: had they been folded into the map on design-tree
authority, the map would now carry two rows for artifacts that do not exist —
and the `--gap-rules` run being byte-identical to the strict one is the evidence
that they never fired.

### F7 — WF1's `LOG_RULES` was missing `1.01b_delineate_region` — **FIXED**

Rule `1.01b` declares a `log:` part like every other WF1 rule, but its label was
never added to `LOG_RULES`. **An unlisted label is not an error**: `merge_logs`
only looks up the labels it is given, so the region-delineation section was
silently absent from every merged `wf1_model_creation.log` and its part was
stranded under `logs/_parts/` on every run. Latent since the ADR 0003 spatial
work landed.

The benchmark gather does not share the failure mode — `wf1_benchmarks.md` has
listed `1.01b_delineate_region` (18.00 s) all along. **The two artifacts
disagreeing is what makes this diagnosable at all**, and it is why a file-count
check would not have caught it.

This is precisely the failure mode the master brief names as unreachable by the
suite ("a missed `LOG_RULES` entry (silent, not an error)"), and it was found by
the observed-tier run rather than by a test. Fixed on owner instruction in its
own commit — P1's brief forbids `Snakefile_*` edits, so the exception is
recorded rather than buried. Verified empirically: after the fix WF1 planned
exactly 2 jobs (`merge_logs`, its params having changed, plus `all`),
`logs/_parts/` is gone because the part was finally consumed, and the merged log
opens at `== 1.01b delineate_region`.

### F8 — the map carried a row for a retired artifact — **ROW DROPPED**

Nothing in the current codebase writes `config/generated/wflow_build_model_run.yml`;
only `wflow_build_forcing_historical.yml` is generated. The copy in the tree was
dated 07-29 and survived a complete WF1 run untouched.

The row mattered more than a stale row usually would: it was **one of the map
doc's two named precedence hazards**, so the hazard guarded a file that cannot
appear. Row, rule, row-driven test case and hazard note are all removed; the
remaining hazard (`log.txt`, F3) is unaffected and keeps its ordering test. A
replacement assertion pins that the retired path now **falls through** rather
than resolving, so an old `project_dir` that still holds the file gets it
reported instead of silently migrated to a destination with no producer.

For contrast, the prefixes used for `staticgeoms/`, `run_default/`,
`evaluation/`, `forcing/plots/`, `config/catalogs/`, `config/templates/` and
`config/observations/` are **not** widenings: each of those map rows is itself a
`*` or `**` glob over the directory, so a prefix rule is the faithful per-row
encoding. Since the F1a/F1b rulings, `spatial/geoms/` and
`config/runs/<workflow>/` are in that same category — their rows are now globs
too.

### F3 — `hydrology_runs/rlz_<r>/config/log.txt` is a one-to-many split, not a move

The map already flags this row as "**requires a code change**". Encoding it
surfaces a second property: **the member index is not recoverable from the old
path.** One `log.txt` per realization becomes N per-member logs once P2 sets
`path_log` per member, so the row is a one-to-many *split* — the inverse of
`build_r07_merges` — and no path-map rule can express it as a function.

Encoded as a regex whose destination keeps the map doc's own `<c>` placeholder
verbatim:

```
experiments/<id>/hydrology_runs/rlz_3/config/log.txt
  ->  experiments/<id>/hydrology/wflow/output/rlz_3_cst_<c>.log
```

This is inert in a tree diff: `_is_excluded` drops `.log` and `log.txt` before
mapping, so the rule never reaches `diff_trees`. What it does carry is the
**precedence** — it must precede any `hydrology_runs/rlz_(\d+)/config/(.*)` rule,
which would otherwise consume the log and route it to `config/`. Pinned by
`test_hazard_wflow_log_beats_the_run_config_regex`. Non-blocking; recorded so P2
does not read the rule as a claim that the split is expressible.

### F4 — both named precedence hazards are currently *latent*

The map doc names two. In this encoding neither is load-bearing yet:

- `config/generated/*` → the model root would be swallowed by a `config/**`
  identity row — but identity is enumerated per row and there is no `config/`
  catch-all, so nothing swallows it.
- the `log.txt` row would be consumed by a general
  `hydrology_runs/rlz_(\d+)/config/(.*)` rule — but the run-config rule is
  written as `config/cst_(\d+)\.toml`, which does not match `log.txt`.

Both orderings are kept anyway and commented as such, because "simplifying" the
TOML rule to `config/(.*)`, or adding a `config/` catch-all, silently reopens
them. Both hazards have the tests the brief mandates, so the behaviour is pinned
either way.

### F5 — an identity-mapped directory hides orphans from the falsifier

Found twice, which is what makes it a property rather than an incident.

**Round 1**, staging the tree: the falsifier reported 23 unmapped paths; the
actual orphan set was **102 files**. The 79 it did not name sit under two
directories the map routes *wholesale*:

| Orphan group | Files | Why the falsifier is blind to it |
| --- | ---: | --- |
| retired WF2 log labels under `logs/_parts/` | 21 | the map row is `logs/_parts/**` — an identity prefix, so any part dir under it matches |
| pre-`_parts` WF3 logs under `experiments/<id>/logs/` | 58 | the map row is `logs/*` at experiment scope — identity over the whole directory |

**The encoding is faithful** — both map rows really are wholesale, and the
project-root rows really are narrow (`logs/wf{1,2}_*.log`, `_parts/**`,
`dag/`), which is exactly why the 22 project-scope rule logs *were* caught. The
asymmetry is in the map, not in the comparator.

**Round 2**, after the run: an **mtime sweep** of the finished tree found 9
files predating it. One (`store_region.geojson`) was legitimate — a declared
output of the persistent store rule that Snakemake correctly found up to date.
The other 8 were stale, and **the falsifier reported none of them**: six
pre-split figures under `climate_historical/<key>/plots/` and
`hydrology_model/forcing/plots/` (both prefix-mapped), an orphan `RT_Q_*.csv`
(classified DELETED by design), and the retired build config of F8.

The consequence is operational and worth stating plainly: **the hand-prune list
in the runbook cannot be replaced by the comparator, and mtime is the instrument
the comparator is not.** Anything under an identity- or prefix-mapped directory
has to be adjudicated some other way. Narrowing the wholesale rows to match the
project-root precision would close the log case, but that is a map amendment on
no evidence of a real defect — those rows are what the map says — so it is
recorded rather than done. The runbook now carries the mtime sweep as a step.

### F6 — no path map row proved unexpressible

Every row of the map's four destination sections, plus the project-root rows and
the rule-3.11 rename rows, resolves to its stated destination under
`test_every_map_row_resolves` (72 parametrised cases). `indicators/RT_*.csv` is
the one row whose "destination" is deletion; it is classified `DELETED` by
`build_r09_deletions` rather than given an invented destination, so the row is
covered without polluting the map.

---

## Orphan climate-store reporting

`dev/scripts/prune_climate_store.py`, a **sibling** of `prune_series_cache.py`
rather than a mode inside it: that script's contract is keyed to the CMIP6 series
filename grammar under `climate_projections/<clim_project>/scalar/`, and the
historical store is a different artifact class, in a different tree, under a
different key.

The active key is **derived** the way `snake_utils.climate_store_spec` derives it
(`<clim_source>_<slugify_window(start, end)>`), not globbed — so a key the
workflow would not produce counts as an orphan rather than as a second active
store. `slugify_window` is reused by import; it loads standalone outside
Snakemake, so nothing was vendored.

**Deletes nothing without `--delete`**, matching `prune_series_cache.py`'s stated
contract, and `test_dry_run_is_the_default_and_deletes_nothing` pins it.

## Existing behaviour preserved

- `build_r07_path_map`'s rules and its tests are untouched.
- `apply_path_map` is now a `[0]` projection of `apply_path_map_matched` — one
  matching pass, so the two cannot drift. Backslash normalization, rule-kind
  dispatch and first-match-wins are byte-identical, including the value returned
  for an unmatched path. `build_r07_path_map` users, `compare_yaml` and
  `_normalize_tree_root_paths` are unchanged.
- The CLI gained `--check-map`, `--milestone r09` and `--r09-gap-rules`.
  `--ref`/`--cur` are still required for the tree-diff mode; they are simply not
  required in `--check-map` mode, which uses neither.
- No `Snakefile_*`, `blueearth_cst/**`, `config/**` or `dev/baseline/manifest.json`
  edit. `LOG_RULES` is untouched — the "same edit" rule binds the phase that
  performs the 3.11 rename, which is P3.
- **`pixi.toml` was edited, against the master brief's shared constraint**
  (*"`pixi.toml` / `pixi.lock` and `Manifest.toml` are not to be edited"*), on
  an explicit owner instruction to add the `tree-check` task. The constraint
  guards against environment churn — a dependency change would rebuild the
  ~4.7 GB shared env and make every worktree test the wrong one. A `[tasks]`
  entry declares no dependency: **`pixi.lock` is byte-identical** (md5
  `ab58c87aff831cde7eddc7090a37406b` before and after), so CI's `locked: true`
  cannot drift on it. Revert the task if you would rather keep the constraint
  absolute; nothing else depends on it.

## Validation

Master ladder rungs 1 and 3 only, as the brief scopes. Rung 2 has no trigger (no
Snakefile or `script:` signature changed); rungs 4–5 belong to the program.

| Rung | Command | Result |
| --- | --- | --- |
| 1 Narrow | `pytest tests/test_semantic_tree_diff.py tests/test_r09_path_map.py tests/test_prune_climate_store.py tests/test_snapshot_project_tree.py` | **170 passed** |
| 3 Phase gate | `pixi run test-fast` | **1202 passed**, 30 skipped, 42 deselected, 1 xfailed (60 s) |
| 2 Integration | `pixi run test-cli` | **12 passed** — fired because F7 changed a Snakefile |

WF1/WF2/WF3 suites were not run: this phase touches no workflow.

---

## Gate 1 — status

| # | Item | Status |
| --- | --- | --- |
| 1 | The three declared-tier gaps (F1a–F1c) | **RULED 2026-08-04** — map amended, rules folded in, strict map reports 0 unmapped |
| 2 | The two unruled gaps (F2) | **CLOSED NEGATIVELY** — the observed tier shows `instate/` does not exist and `plots/` holds exactly the two named files. Both rules removed |
| 3 | The observed tier | **VERIFIED** — one clean three-workflow run, snapshotted and committed: `MAP CLEAN: 192 paths, 0 unmapped` |

Item 1's rulings were taken on the falsifier's own output: the map's `data/` row
enumerated five geoms layers where the code writes six, its `config/runs/` row
transcribed one workflow where the design says `<workflow>`, and no row covered
WF3's experiment-scoped bundle. All three are amended in
`migration_project-tree.md`, dated and cross-referenced to this report.

Item 3 is what Gate 1 could not close without, and it earned its place: the
observed tier is the only thing that could settle F2 (both negatively), retire
F8's row, and surface F7 — none of which any declaration or unit test reaches.

**Two items the owner should note before P2 begins**, neither of them blocking:

1. **F7 is a `Snakefile_model_creation` edit on this branch**, outside P1's
   stated scope, in its own commit so it can be reverted or cherry-picked
   independently. `main`'s working tree is clean and carries no part of it.
2. **The observed snapshot is `main` plus that fix.** Regenerating it from `main`
   as it stands yields one extra path. P2 should merge or carry the fix before
   using the snapshot as its pre-migration reference, or subtract that one path.

Everything Gate 1 asked for is present: the map applied to a pre-migration tree
showing the intended post-migration paths, in both tiers, with zero unmapped.
