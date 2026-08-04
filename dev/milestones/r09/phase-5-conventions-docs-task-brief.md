Task Brief — R9 P5: conventions and documentation

### Context

Canonical ruleset: `AGENTS.md`. Master brief:
`dev/milestones/r09/project-tree-task-brief.md`. Runs last, after P1–P4 land.

- `dev/reference/naming.md` §8 assigns generated outputs to "owning workflow
  contract — **varies**" and states that the guide "does not unify" file classes.
  R9 adopts a real rule, so the guide must gain one or the tree and the
  convention drift apart. The design records this as a neutral obligation.
- §7's carve-out permits scientific abbreviations in *user-facing output
  filenames* (`Qstats.csv`, `Tlow`, `Tpeak`, `BFI`). After P3 the only survivors
  are config keys and column labels — not filenames — so the carve-out narrows
  rather than being repealed.
- `migration_project-tree.md` already **is** the §7 internal rename record. Its
  scope is exactly two files: `Qstats.csv` and `basin.csv`.

### Goal

The repository's conventions and user-facing documentation describe the tree R9
actually produces, so the next contributor names a new artifact from a rule
rather than by analogy.

### Non-goals

- **The R10 verb vocabulary for rule identifiers** — that belongs to
  `dev/milestones/r10/rule-naming-design.md`, not here.
- No `docs/migration-r09.md` user guide. Pre-existing `project_dir` trees are
  unsupported and a fresh run is required (R7 ruling GA-2, restated for R9), so
  there is nothing for a user to migrate.
- No code changes.

### Allowed scope

**Permitted** — `dev/reference/naming.md`, `README.rst`, `AGENTS.md`,
`docs/**`, `dev/reference/contracts/*-seam.md`.

**Forbidden** — `blueearth_cst/**`, all `Snakefile_*`, `config/**`,
`dev/baseline/**`, and `dev/milestones/r10/**`.

### Required changes (checklist)

1. `naming.md` §8: replace the generated-outputs row's "varies" with the real
   rule — lowercase `snake_case` for locally minted names under `project_dir`,
   with the two exemptions stated (upstream-owned names and embedded tier-1
   identifiers; config keys and data labels are out of the rule's reach).
2. `naming.md` §7: narrow the scientific-abbreviation carve-out to config keys
   and column/row labels, since no filename relies on it after P3.
3. `naming.md` §8: keep `dev/` markdown kebab-case explicitly. The new rule is
   **class-scoped**; a reader must not generalise it into a repo-wide sweep.
4. Record that `migration_project-tree.md` satisfies §7 for R9, and that its
   scope is the two result tables only.
5. Update `README.rst` and `AGENTS.md` where they describe the output tree, and
   the two seam contracts where they name moved paths.
6. Note in `naming.md` §9 that the `W.NN` number is a **stable identifier
   assigned at rule creation**, not a position — WF2 already defines rules out of
   numeric order with gaps at 1.14, 2.05, 3.12, so the current wording is false.


### Discovered during P1–P4 — verified, with evidence

Added 2026-08-04, after the four implementation phases ran. Each was found by
running the pipeline rather than by reading, and each is verified against the
code or the built tree rather than asserted. They are **additions** to the checklist above, not
corrections to it.

7. **`AGENTS.md:114` documents the wrong DAG-render path.** It says
   `config/dag/<project_name>_wf<N>_dag.png`; P2 commit 4 moved renders to
   `logs/dag/` (WF1/WF2) and `experiments/<id>/logs/dag/` (WF3), under design
   principles P4 and P7. Agent-facing, so a stale command here misdirects the
   next session rather than merely confusing a reader.

8. **`AGENTS.md:136` is factually wrong about the pixi environment.** It states
   the env is shared and "a worktree resolves to the primary's copy instead of
   building its own". It does not: each worktree carries its own tracked
   `pixi.toml`, so pixi creates a separate `.pixi/` beside it. Measured in P2 —
   WF3 failed in a task worktree with `there is no package called 'weathergenr'`
   because that package comes from `pixi run install` (remotes), not from
   `pixi install`.
   The passage also *advises* on the strength of that mechanism — telling a task
   that changes `pixi.toml`/`pixi.lock` to build its own env "rather than
   inherit" — so the advice is premised on an inheritance that does not happen.
   Correct both the claim and the advice; the disk-cost figure needs re-checking
   too.

9. **`project-tree-design.md:129` asserts a file does not exist where it does.**
   It states `region.geojson` "exists only as
   `models/hydrology/wflow/staticgeoms/region.geojson` and the store's
   `store_region.geojson`". `data/spatial/geoms/region.geojson` is written by
   rule `delineate_region` (ADR 0003) — `snake_utils.py:836` — and is present in
   the declared inventory, the observed inventory and the built tree.
   This is the root of P1's F1a, where the migration map's `data/` row
   enumerated five geoms layers and the code writes six. The map was amended;
   **the design doc was not**, and it is the more authoritative document.
   This is a design-record correction rather than documentation polish, so rule
   it explicitly rather than editing in passing.

10. **`AGENTS.md` does not mention P1's tooling.** `dev/scripts/prune_climate_store.py`,
    `dev/scripts/snapshot_project_tree.py` and the `pixi run tree-check` task are
    absent from both the Repo Map and Key Commands. The Repo Map currently names
    `prune_series_cache.py` as though it were the only pruning helper.

11. **Extend the grep falsifier's term list.** P3 renamed the rule identifier and
    P2 moved `data_catalog_climate_experiment.yml`, so add `export_wflow_results`,
    `basin.csv`, `config/dag/`, and `indicators/` to the terms below. `Qstats`
    and `RT_` are already listed and remain correct.

**One caution for the grep falsifier.** Its surviving hits will include this
milestone's own reports and briefs, which describe the old tree deliberately and
at length — `phase-1-report.md` alone discusses `hydrology_model/` throughout.
Those are records of what was done, not documentation of the current tree, and
justifying them one by one would swamp the report. Justify by CLASS: sealed
records under `dev/milestones/r0[1-8]/`, R9's own phase reports and briefs, and
the comparator's path-map source side (`build_r09_path_map`, both inventories,
and their tests) all legitimately name old paths. Anything outside those classes
is a real hit.

### Validation

This phase changes no behaviour, so it has no narrow test scope — the grep below
*is* its check. It is also the last phase, so the program's single full-gate run
lands here.

1. **Narrow** — the grep falsifier below (per edit).
2. **Phase gate** — `pixi run test-fast`, to catch any doctest or path assertion
   that reads documentation.
3. **Full gate** — `pixi run test-full`. **The program's one run**, at the
   landing gate. If red, bisect with `pixi run test-fast`.

**Falsifier.** Claim: *no documentation still describes the old tree*. A read-
through cannot establish that. Grep the tracked tree for `hydrology_model/`,
`climate_historical/`, `climate_projections/`, `hydrology_runs/`, `indicators/`,
`weather_generator/`, `Qstats`, and `RT_` and require every surviving hit to be
justified in the phase report — sealed milestone records legitimately keep them;
current docs do not.

### Acceptance criteria

- §8 carries a generated-outputs rule; §7's carve-out is narrowed; the class
  scoping is explicit.
- The grep falsifier's surviving hits are each accounted for.
- `pixi run test-full` green — the program's single full-suite run.
- Sealed milestone records under `dev/milestones/r0[1-8]/` are **unmodified** —
  a sealed record stays as it was written.

### Output requirements

A phase report carrying the grep results with a justification per surviving hit,
and the `naming.md` diff summarised by section.

### Task constraints

- Do not edit sealed milestone records to make the grep clean. Justify, don't
  rewrite history.
- Keep documentation concise and close to the code it describes; no tutorial
  material.
