# dev/reviews/

The **decaying** tier: process reviews and post-milestone self-check registers.
Snapshots of a system that keeps moving. This is the only folder under `dev/`
that is meant to be pruned — the three-part deletion test is in
[`../README.md`](../README.md) § The retention rule.

## Retention ledger

Assessed 2026-08-02. Re-check the `Cites` column with `git grep -l <filename>`
before acting on it; this table goes stale exactly like its contents.

| File | Lines | Retention | Why |
|---|---|---|---|
| `2026-07-25_post-r6-assessment.md` | 833 | **Keep whole** | 23 individual `O-` numbers are cited from R7 docs, `roadmap.md`, and `pyproject.toml`. Citers reference the detail, not the file |
| `2026-07-29_post-r7-self-check.md` | 330 | **Keep — not closed** | `S7-01` and `S7-02` are still `Status: discussed`, never resolved. Unfinished, not stale |
| `2026-07-30_wf2-v2-process-review.md` | 264 | **Keep — partly superseded** | Sections 1, 3, 5 replaced by `-r2`; 2, 4, 6 stand. Banner in the file says which |
| `2026-07-30_wf2-v2-process-review-r2.md` | 232 | **Keep** | Cited by two R8 records and the WF2 v2.0 design |
| `2026-07-30_process-review-critique-fable.md` | 405 | **Keep** | `-r2`'s evidence base — it argues every concession from these |
| `2026-07-30_process-review-critique-gpt.md` | 102 | **Keep** | Same |
| `2026-07-30_design-loop-efficiency-review-brief.md` | 236 | **Keep** | Cited by `critique-fable.md`; also the record of the blind two-reviewer protocol |
| `2026-07-31_post-r8-self-check.md` | 625 | **Keep whole** | Closed and fully dispositioned, but its own outcome section states it is "a derived overview, not a substitute" for the entries |

Removed on 2026-08-02:

- `2026-07-21_adr-0001-constant-pars.md` → **promoted** to
  `../decisions/0001-restore-wflow-constant-parameters/review-record.md`. It is
  an ADR audit trail, so it inherits the ADR's permanence rather than this
  folder's decay.
- `2026-07-28_design-review-loop-process-observations.md` → **deleted**. Passed
  all three tests: uncited, its run complete, and its lessons graduated into
  five skills in `brain` (`0b174e82`).

## What this folder is not

Not a place for anything durable. An ADR audit trail belongs with its ADR; a
milestone's evidence belongs in `../milestones/<id>/`; a lesson about how to
work belongs in the skill it changes, in `brain`. If a file here is the only
copy of something that must survive, it is in the wrong folder.
