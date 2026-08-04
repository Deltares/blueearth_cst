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

**Explicitly authorized:** the shared `apply_path_map` may gain a **default-off**
reporting parameter (or a sibling returning `(new, matched)`) so a fall-through is
distinguishable from an identity match. Today it returns its input unchanged when
no rule fires (`semantic_tree_diff.py:461`), which makes the falsifier below
inexpressible. Existing call sites — `build_r07_path_map`, `compare_yaml`,
`_normalize_tree_root_paths` — must keep their current behaviour bit-for-bit. The
non-goal below binds `build_r07_path_map`'s **rules**, not the shared applier.

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
   **Enumerate them per row.** A catch-all `config/` → `config/` prefix satisfies
   every row at once and silently empties the falsifier's report, since a
   fall-through and an identity match then look the same.
4. Register the rules in **precedence order — narrower source pattern first**. The
   map's tables are grouped by destination root for reading, not for encoding
   (map doc, *Rule precedence — the tables are not the rule order*). Include a
   test per named hazard, asserting the path resolves to the narrow destination:
   `config/generated/wflow_build_model_run.yml` →
   `models/hydrology/wflow/config/build_model.yml`, and
   `hydrology_runs/rlz_<r>/config/log.txt` →
   `hydrology/wflow/output/rlz_<r>_cst_<c>.log`.
5. Orphan climate-store reporting: list `climate_historical/<source>_<window>/`
   directories not matching the config's active key. **Dry run by default**;
   deletion only behind an explicit flag, matching `prune_series_cache.py`'s
   stated contract.
6. Tests: one per relocation class, plus a case asserting an unmapped path is
   reported rather than silently passed through.

### Validation

Master ladder rungs 1 and 3 only. This phase changes no runtime behaviour: rung
2 has no trigger (no Snakefile or `script:` edit), and rungs 4–5 belong to the
program, not to this phase.

**Named scope — run this and nothing else:** `pytest tests/test_semantic_tree_diff.py`
plus the new test module. Do not run WF1/WF2/WF3 suites; this phase touches no
workflow.

1. **Narrow** — the named scope (per edit).
2. **Phase gate** — `pixi run test-fast` once, at phase end.

**Falsifier for the property this phase asserts.** The claim is *"the map covers
every artifact."* A passing unit test cannot show that. Apply the map to the
**declared-tier inventory** — the paths derived from the three Snakefiles'
`output:` declarations, expanded over the seed config (map doc, *The inventory the
map is validated against*) — and assert **zero unmapped paths**. Any path the map
does not recognise must appear in the report rather than pass through silently.
That output is Gate 1's evidence for the declared tier.

The **observed tier** — one clean three-workflow run from the primary checkout,
snapshotted as a sorted path list — is an owner action and is out of this phase's
scope; it is the only tier carrying undeclared engine artifacts. If it does not
exist when this phase runs, run against the declared tier and **name the observed
tier as unverified in the phase report**. Gate 1 closes on both.

Do **not** run this falsifier against `test_case/test_local`. It is a mixed-era
tree whose documented orphans are deliberately unmapped (map doc, *Orphans in the
fixture — do NOT map*), so it fails by construction on paths the map is right to
reject.

### Acceptance criteria

- Every path map row has a rule, registered narrower-first, and both named
  precedence hazards resolve to their narrow destination.
- Applying the map to the declared-tier inventory yields v10 paths with **zero
  unmapped paths**; the observed tier is either clean or reported as unverified.
- `build_r07_path_map`'s rules and tests are unchanged, and every existing
  `apply_path_map` call site behaves exactly as before.
- Orphan-store reporting deletes nothing without an explicit flag.
- **PAUSE at master Gate 1.** Do not proceed to P2.

### Output requirements

A phase report naming: the map's rule count by class, the unmapped-path result
against the declared-tier inventory, the observed tier's status (snapshotted and
clean, or **unverified**), and any path map row that could not be expressed as a
rule — that last is a finding against the map, not a reason to improvise.

### Task constraints

- Do not move, rename, or delete any artifact.
- If a path map row proves unexpressible or wrong, **stop and report**; the map
  is the contract, and amending it is an owner decision.
