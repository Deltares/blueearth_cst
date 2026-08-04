# R9 P1 report — path-map comparator and orphan-store tooling

Date: 2026-08-04. Branch: `feat/r09-p1-comparator` (cut from
`milestone/r09-project-tree`). Brief:
[`phase-1-comparator-task-brief.md`](phase-1-comparator-task-brief.md).

**Status: complete. Findings F1a–F1c ruled by the owner 2026-08-04 and folded
into the map; PAUSED at master Gate 1 on the observed tier.** No P2 work has
begun. This phase moved no files and wrote to no `project_dir`.

---

## What landed

| Artifact | Purpose |
| --- | --- |
| `semantic_tree_diff.build_r09_path_map` | The migration map's rows as executable rules |
| `semantic_tree_diff.apply_path_map_matched` | The fall-through signal the falsifier needs |
| `semantic_tree_diff.build_r09_gap_rules` / `R09_MAP_GAPS` | Candidate rows for artifacts the map does not cover — **opt-in**; down to the two unruled ones |
| `semantic_tree_diff.build_r09_deletions` | `indicators/RT_*.csv`, deleted rather than migrated |
| `semantic_tree_diff --check-map` | The falsifier: classify a path list, exit 1 on any UNMAPPED |
| `dev/scripts/prune_climate_store.py` | Orphaned `climate_historical/<source>_<window>/` reporting |
| `dev/scripts/snapshot_project_tree.py` | Snapshot a tree + run the map check in one command, every map parameter derived from the config |
| `pixi run tree-check` | Task wrapping the above, shaped like the existing `dag-wf*` tasks |
| `dev/milestones/r09/declared_inventory.txt` | The declared-tier inventory, 176 paths, with its provenance |
| `tests/test_r09_path_map.py`, `tests/test_prune_climate_store.py`, `tests/test_snapshot_project_tree.py` | 118 new tests (plus 5 added to `tests/test_semantic_tree_diff.py`) |

## Rule count by class

`build_r09_path_map("experiment", "era5_20000101_20201231")` — **60 rules**,
after the F1a–F1c amendment:

| Kind | Count | Of which identity |
| --- | ---: | ---: |
| regex (`fullmatch` + expansion template) | 9 | 3 |
| directory prefix (`old` ends `/`) | 24 | 8 |
| exact file | 27 | 6 |
| **total** | **60** | **17** |

43 rules relocate; 17 resolve a path to itself. The 17 are enumerated **per map
row** — never as a catch-all — because `apply_path_map` returns its input
unchanged on fall-through, so one broad `config/` → `config/` rule would satisfy
every `config/` row at once and empty the unmapped-path report by construction.
`test_a_catch_all_config_prefix_would_empty_the_report` demonstrates exactly that
failure, so the reason for the per-row enumeration is pinned rather than
described.

Registered **narrower source pattern first** (`apply_path_map` is first match
wins). `test_narrower_source_pattern_is_registered_first` asserts the ordering as
an index property over five (narrow, general) pairs, not just as outcomes.

Separately: 2 opt-in gap rules (F2, both directory prefixes, both relocations)
and 1 deletion pattern. Three of the original five became map rows when F1a–F1c
were ruled.

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

## Falsifier — observed tier: **UNVERIFIED**

One clean three-workflow run from the primary checkout, snapshotted as a sorted
path list, does not exist. Producing it is an owner action outside this phase's
scope (map doc, *Sequencing*), and it is the only tier carrying undeclared engine
artifacts — `--dry-run` structurally cannot see them. P1 completes with it
unverified; **Gate 1 may not close that way.**

Two obligations attach to producing it, both from the map doc:

1. **Prune before snapshotting**, or the snapshot bakes the fixture's orphans
   into the contract. Not one command: `prune_series_cache.py --delete` covers
   the WF2 series class, the new `prune_climate_store.py --delete` covers stale
   climate stores, and the log-part and superseded-config orphans are removed by
   hand.
2. Run it from the **primary checkout**, never a worktree.

The undeclared classes the observed tier will exercise, none of which the
declared tier reaches: `hydromt.log`, `hydromt_data.yml`, `staticgeoms/*` beyond
the four declared files, `run_default/*` beyond `output.csv`, `evaluation/*`,
weathergenr's `plots/*.png` and `output/{sim_dates,resampled_dates}.csv`,
`config/generated/wflow_build_model_run.yml`, `store_region.geojson`, and
Wflow's `log.txt`. Each has a map row **and a case in the row-driven test's
`MAP_ROWS` table**; the observed tier is what proves those rows match reality.
`instate/` is the exception and is F2 below — no map row, and not observed
either.

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

### F2 — two further map gaps, observed-tier only

Neither appears in the declared inventory, so neither reds the falsifier; both
would surface the moment the observed tier is snapshotted. Also carried in
`build_r09_gap_rules`.

- **`hydrology_model/instate/`** — Wflow's warm states. The design tree has
  `models/hydrology/wflow/instate/` and the map's models section has no row.
  **Not observed, only inferred**: it appears in no `output:` declaration, and
  the only other evidence in the repository is a path string in an existing
  test fixture. Whether Wflow.jl writes that directory at that path under the
  pinned version is unconfirmed — confirm against the observed tier before
  ruling on the row. The proposed rule is inert if the directory never exists.
- **`hydrology_model/plots/` as a directory** — the map row names
  `basin_area.{png,pdf}` as two files, not a glob. The strict map encodes exactly
  those two exact rules; the prefix that would generalise them is the same class
  of judgment as F1 and is therefore flagged, not folded in.

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

### F5 — no path map row proved unexpressible

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
| 1 Narrow | `pytest tests/test_semantic_tree_diff.py tests/test_r09_path_map.py tests/test_prune_climate_store.py tests/test_snapshot_project_tree.py` | **171 passed** |
| 3 Phase gate | `pixi run test-fast` | **1203 passed**, 30 skipped, 42 deselected, 1 xfailed (41 s) |

WF1/WF2/WF3 suites were not run: this phase touches no workflow.

---

## Gate 1 — status

| # | Item | Status |
| --- | --- | --- |
| 1 | The three declared-tier gaps (F1a–F1c) | **RULED 2026-08-04** — map amended, rules folded in, strict map now reports 0 unmapped |
| 2 | The two unruled gaps (F2) | **Open, not blocking** — neither appears in any declaration; carried opt-in in `build_r09_gap_rules` until the observed tier shows whether they exist |
| 3 | The observed tier | **UNVERIFIED** — an owner action; **this is what keeps Gate 1 open** |

Item 1's rulings were taken on the falsifier's own output: the map's `data/` row
enumerated five geoms layers where the code writes six, its `config/runs/` row
transcribed one workflow where the design says `<workflow>`, and no row covered
WF3's experiment-scoped bundle. All three are amended in
`migration_project-tree.md`, dated and cross-referenced to this report.

**Gate 1 does not close on the declared tier alone**, and P2 does not begin until
it does. The remaining action is the observed tier: prune first
(`prune_series_cache.py --delete`, `prune_climate_store.py --delete`, and the
log-part and superseded-config orphans by hand), then one clean three-workflow
run from the primary checkout, snapshotted as a sorted path list under
`dev/milestones/r09/`, then `--check-map` against it with zero unmapped.
