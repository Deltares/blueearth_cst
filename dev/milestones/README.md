# dev/milestones/

One folder per milestone, holding its design, plan, review, migration,
and evidence documents. Narrative and rationale live in
[`../roadmap.md`](../roadmap.md); this is the file index.

Most folders here are sealed milestones. A folder is also created when a
milestone is **registered** — its design accepted but not yet implemented — so
the accepted design has a source-of-record home instead of sitting in
`../working/`. The `Sealed` column says which is which.

These folders moved here from the `dev/` root on 2026-08-02. That was a path
change only — no file was renamed, split, or edited beyond its path prefix, and
the folders' internal grammar is deliberately **not** normalised. A sealed
milestone's record stays as it was written.

## Index

| Folder | Milestone | Sealed | Tag |
|---|---|---|---|
| `phase-1/m01/` | M01 — Replication baseline | 2026-05-07 | `m01-replication` |
| `phase-1/m02/` | M02 — Pixi env + install | 2026-05-07 | `m02-pixi` |
| `phase-1/m02b/` | M02b — Library upgrades | 2026-05-07 | `m02b-upgrades` |
| `phase-1/m02c/` | M02c — Test coverage | 2026-05-08 | `m02c-tests` |
| `r01/` | R1 — Modularity contracts | 2026-07-18 | `r01-contracts` |
| `r02/` | R2 — Naming conventions | 2026-07-19 | `r02-naming` |
| `r03/` | R3 — Workflow 1: model builder | 2026-07-19 | `r03-model-builder` |
| `r04/` | R4 — Workflow 2: climate projections | 2026-07-20 | `r04-projections` |
| `r05/` | R5 — Workflow 3: climate experiment | 2026-07-20 | `r05-experiment` |
| `r06/` | R6 — Structural refactor | 2026-07-23 | `r06-refactor` |
| `p31/` | P3-1 — Project/experiment structure | 2026-07-24 | `p31-experiments` |
| `p32a/` | P3-2a — Model-independent climate analysis | 2026-07-24 | `p32a-climate-analysis` |
| `p32b/` | P3-2b — Model-swap interchange contracts | 2026-07-24 | `p32b-interchange-contracts` |
| `p33/` | P3-3 — Performance passes | — | `p33-performance` |
| `r07/` | R7 — Project layout | 2026-07-29 | `r07-layout` |
| `r08/` | R8 — WF2 v2.0: GCM projections analysis | 2026-07-31 | `r08-wf2-projections` |
| `r09/` | R9 — Generated project tree | — *(open)* | `r09-project-tree` |
| `r10/` | R10 — Rule naming | — *(open)* | `r10-rule-naming` |

Phase grouping: `phase-1/` is Phase 1 (Foundation); `r01`–`r06` are Phase 2
(Refactor, complete 2026-07-23); `p31`–`p33` are Phase 3 (Usability &
flexibility, complete 2026-07-25); `r07` is Phase 4 (Layout consolidation);
`r08` is Phase 5 (Workflow rework); `r09` is Phase 6 (Project tree redesign,
open); `r10` is Phase 7 (Naming coherence, open).

`p33` carries a tag but no `sealed` line in `roadmap.md` — it is recorded there
as *scoped 2026-07-24*, so the date column is left blank rather than guessed.

## Conventions

- A milestone folder is the default home for that milestone's **promoted
  working notes** — see the promotion rule in [`../README.md`](../README.md).
  `r08/` is the fullest example: eighteen falsifier and validation notes, cited
  from shipped modules and tests.
- Filenames vary by era and are grandfathered. Later milestones converge on
  `<topic>-design.md`, `<topic>-task-brief.md`, `<topic>-intake.md`,
  `<topic>-design-review-record.md`, `migration_<topic>.md`, and
  `baseline_diffs.md`.
- New milestones get a folder here, not at the `dev/` root.
