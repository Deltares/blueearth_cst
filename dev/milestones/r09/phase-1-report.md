# R9 P1 report — path-map comparator and orphan-store tooling

Date: 2026-08-04. Branch: `feat/r09-p1-comparator` (cut from
`milestone/r09-project-tree`). Brief:
[`phase-1-comparator-task-brief.md`](phase-1-comparator-task-brief.md).

**Status: complete, PAUSED at master Gate 1.** No P2 work has begun. This phase
moved no files and wrote to no `project_dir`.

---

## What landed

| Artifact | Purpose |
| --- | --- |
| `semantic_tree_diff.build_r09_path_map` | The migration map's rows as executable rules |
| `semantic_tree_diff.apply_path_map_matched` | The fall-through signal the falsifier needs |
| `semantic_tree_diff.build_r09_gap_rules` / `R09_MAP_GAPS` | Proposed rows for artifacts the map does not cover — **opt-in** |
| `semantic_tree_diff.build_r09_deletions` | `indicators/RT_*.csv`, deleted rather than migrated |
| `semantic_tree_diff --check-map` | The falsifier: classify a path list, exit 1 on any UNMAPPED |
| `dev/scripts/prune_climate_store.py` | Orphaned `climate_historical/<source>_<window>/` reporting |
| `dev/milestones/r09/declared_inventory.txt` | The declared-tier inventory, 176 paths, with its provenance |
| `tests/test_r09_path_map.py`, `tests/test_prune_climate_store.py` | 97 new tests (plus 5 added to `tests/test_semantic_tree_diff.py`) |

## Rule count by class

`build_r09_path_map("experiment", "era5_20000101_20201231")` — **63 rules**:

| Kind | Count | Of which identity |
| --- | ---: | ---: |
| regex (`fullmatch` + expansion template) | 8 | 2 |
| directory prefix (`old` ends `/`) | 23 | 7 |
| exact file | 32 | 7 |
| **total** | **63** | **16** |

47 rules relocate; 16 resolve a path to itself. The 16 are enumerated **per map
row** — never as a catch-all — because `apply_path_map` returns its input
unchanged on fall-through, so one broad `config/` → `config/` rule would satisfy
every `config/` row at once and empty the unmapped-path report by construction.
`test_a_catch_all_config_prefix_would_empty_the_report` demonstrates exactly that
failure, so the reason for the per-row enumeration is pinned rather than
described.

Registered **narrower source pattern first** (`apply_path_map` is first match
wins). `test_narrower_source_pattern_is_registered_first` asserts the ordering as
an index property over five (narrow, general) pairs, not just as outcomes.

Separately: 5 opt-in gap rules (all directory prefixes, 2 of them identity) and
1 deletion pattern.

## Falsifier — declared tier

Full output and the complete old → new table:
[`2026-08-04_p1-declared-tier-falsifier.md`](2026-08-04_p1-declared-tier-falsifier.md).

| Run | Paths | Moved | Identity (by rule) | Unmapped | Exit |
| --- | ---: | ---: | ---: | ---: | ---: |
| strict map | 176 | 162 | 11 | **3** | 1 |
| `--r09-gap-rules` | 176 | 163 | 13 | **0** | 0 |

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

### F1 — three declared artifacts have no map row (needs an owner ruling)

Kept **out** of `build_r09_path_map` deliberately. The brief is explicit that an
uncovered artifact is "a finding against the map, not a reason to improvise", and
amending the map is an owner decision — a rule added quietly would be a fait
accompli, and Gate 1 exists precisely because a map wrong in the same direction
as the migration is undetectable afterwards. The proposed rules live in
`build_r09_gap_rules`, behind `--r09-gap-rules`, so both numbers are reportable.

Split by whether the **design tree v10** fixes the destination:

| # | Artifact | Producer | Design tree | Proposed |
| --- | --- | --- | --- | --- |
| F1a | `spatial/geoms/region.geojson` | rule `delineate_region` (ADR 0003) | **silent** | → `data/spatial/geoms/region.geojson` |
| F1b | `config/runs/climate_projections/<digest>/` | WF2 config snapshot | covered — `config/runs/<workflow>/<digest>/` | identity |
| F1c | `experiments/<id>/config/runs/climate_experiment/<digest>/` | WF3 config snapshot | **silent** | identity |

**F1a is the sharpest and should lead the gate discussion.** The map's Finding 2
states that the `data/spatial/` rows "correspond exactly" to the nine P1 products
of rule `prepare_spatial_maps`. The declared inventory shows **ten** files under
`spatial/`: the tenth, `geoms/region.geojson`, comes from a *different rule*. The
map's own completeness claim for the spatial subtree is therefore falsified, and
the design tree does not name the file either. The destination is not in doubt —
it is the same directory as the other five geoms — but the row is a design gap,
not just a transcription gap.

**F1b is a transcription narrowing.** Design tree line 308 reads
`config/runs/<workflow>/<digest>/`; the map row transcribes only
`model_creation`. WF2 emits the same class, and WF1's row already rules it
unchanged.

**F1c has no design line at all.** v10's `experiments/<id>/config/` lists
`experiment.yml`, `project_snapshot.yml` and `model_reference.yml` — an
experiment-scoped `runs/<workflow>/<digest>/` bundle is not among them, yet WF3
emits one. Identity is proposed under principle **P9** (where the design differs
from what the code emits, the emitted structure wins). If the owner would rather
the design absorb it, that is a design edit, not a map edit — and P4 is the phase
that touches `experiments/<id>/config/`.

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
`evaluation/`, `forcing/plots/`, `config/catalogs/`, `config/templates/`,
`config/observations/` and `config/runs/model_creation/` are **not** widenings:
each of those map rows is itself a `*` or `**` glob over the directory, so a
prefix rule is the faithful per-row encoding.

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
`test_every_map_row_resolves` (69 parametrised cases). `indicators/RT_*.csv` is
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

## Validation

Master ladder rungs 1 and 3 only, as the brief scopes. Rung 2 has no trigger (no
Snakefile or `script:` signature changed); rungs 4–5 belong to the program.

| Rung | Command | Result |
| --- | --- | --- |
| 1 Narrow | `pytest tests/test_semantic_tree_diff.py tests/test_r09_path_map.py tests/test_prune_climate_store.py` | **150 passed** |
| 3 Phase gate | `pixi run test-fast` | **1182 passed**, 30 skipped, 42 deselected, 1 xfailed (48 s) |

WF1/WF2/WF3 suites were not run: this phase touches no workflow.

---

## Gate 1 — what the owner is being asked to rule

1. **The three declared-tier gaps (F1a–F1c)** — accept the proposed rows into
   the migration map, or rule differently. Until then the strict map reports 3
   unmapped and the falsifier's headline is *zero unmapped once the three
   enumerated additions are accepted*. F1a additionally warrants a design-doc
   correction: Finding 2's "correspond exactly" claim is false.
2. **The two observed-tier gaps (F2)** — same question, one milestone earlier
   than they would otherwise surface.
3. **The observed tier itself** — an owner action. Gate 1 does not close on the
   declared tier alone.

**Do not begin P2 until 1–3 are ruled.**
