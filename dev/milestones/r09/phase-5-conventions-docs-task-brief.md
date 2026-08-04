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
- `pytest tests/` green.
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
