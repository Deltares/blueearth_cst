Task Brief — R9 P1: path-map comparator and orphan-store tooling

### Context

Canonical ruleset: `AGENTS.md`. Master brief:
`dev/milestones/r09/project-tree-task-brief.md`.

- The old → new path map is complete at
  `dev/milestones/r09/migration_project-tree.md`; the target tree is
  `project-tree-design.md` v10.
- `dev/scripts/semantic_tree_diff.py` already carries `build_r07_path_map`
  (regex rules around lines 329–350) as the pre-R7 → R7 map. It is the template
  for this one, and the direction is **inverse**: R7 moved indices from filename
  into a directory, R9 moves them back.
- `dev/scripts/prune_series_cache.py` reports orphaned WF2 series only. R9 keeps
  the climate store's `<source>_<window>` cache key, so a changed window now
  strands its predecessor with nothing to report it.
- This phase touches **no** `project_dir` and moves **no** files. It builds the
  instrument that will detect whether P2's move was faithful.

### Goal

A tested `build_r09_path_map` that transforms any pre-migration path to its v10
destination, plus orphan climate-store reporting, so P2 has a regression detector
before it starts.

### Non-goals

- No Snakefile edits, no artifact moves, no `project_dir` writes.
- Not the R7 map: `build_r07_path_map` stays, unmodified, for pre-R7 trees.
- No deletion behaviour by default in the new reporting.

### Allowed scope

**Permitted** — `dev/scripts/semantic_tree_diff.py`,
`dev/scripts/prune_series_cache.py` (or a sibling script if the orphan-store
logic does not fit its contract), `tests/test_semantic_tree_diff.py`, and new
test modules.

**Forbidden** — all three `Snakefile_*`, `blueearth_cst/**`, `config/**`,
anything under a `project_dir`, `dev/baseline/manifest.json`.

### Required changes (checklist)

1. `build_r09_path_map(experiment, ...)` covering **every** row of the path map's
   four destination sections (`config/`, `data/`, `models/`, `experiments/`,
   plus project-root `logs/`/`benchmarks/`), mirroring `build_r07_path_map`'s
   exact/prefix/regex rule forms.
2. Regex rules for the two index relocations: `hydrology_runs/rlz_<r>/<kind>/…`
   → `hydrology/wflow/<kind>/rlz_<r>_cst_<c>.…`, including the `inmaps_` and
   `outstates_` prefixes.
3. Identity rules where the map says identity — notably every `config/` row —
   so an unmapped path is distinguishable from a deliberately unchanged one.
4. Orphan climate-store reporting: list `climate_historical/<source>_<window>/`
   directories not matching the config's active key. **Dry run by default**;
   deletion only behind an explicit flag, matching `prune_series_cache.py`'s
   stated contract.
5. Tests: one per relocation class, plus a case asserting an unmapped path is
   reported rather than silently passed through.

### Validation

Ladder rungs 1 and 2 only — this phase changes no runtime behaviour, so rungs
3–5 do not apply and `pytest tests/` is not required here.

1. **Narrow** — `pytest tests/test_semantic_tree_diff.py` (per edit).
2. **New behavioural tests** — the new cases above (per edit).

**Falsifier for the property this phase asserts.** The claim is *"the map covers
every artifact."* A passing unit test cannot show that. Apply the map to a
materialized pre-migration tree and assert **zero unmapped paths**; any path the
map does not recognise must appear in the report. That output is Gate 1's
evidence.

### Acceptance criteria

- Every path map row has a rule; applying the map to a materialized
  pre-migration tree yields the v10 tree with zero unmapped paths.
- `build_r07_path_map` and its tests are unchanged.
- Orphan-store reporting deletes nothing without an explicit flag.
- **PAUSE at master Gate 1.** Do not proceed to P2.

### Output requirements

A phase report naming: the map's rule count by class, the unmapped-path result
against the materialized tree, and any path map row that could not be expressed
as a rule — that last is a finding against the map, not a reason to improvise.

### Task constraints

- Do not move, rename, or delete any artifact.
- If a path map row proves unexpressible or wrong, **stop and report**; the map
  is the contract, and amending it is an owner decision.
