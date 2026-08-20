# WF3 v2.0 execution model — consolidated review record

Audit trail for the `design-review-loop` run `wf3-experiment-v2`, which produced
`design-v4.md` — **ratified as an architectural input, NOT promoted to an
implementable design** (G2, 2026-08-08).

This record is the durable artifact; the per-round scratch (`design-v1.md` …
`design-v4.md`, `status.md`, `ledger.md`, the reviews and briefs) lives in git
history under `dev/working/design-runs/wf3-experiment-v2/` at tag
`archive/wf3-experiment-v2` (commit `531bcc6`, also the head of branch
`docs/wf3-redesign`). Cite the TAG, not the branch: a branch is a mutable
pointer and this one exists on no remote, so until the tag was cut on
2026-08-19 a single branch deletion would have destroyed this record.
Not to be merged — see
"Why nothing was promoted" below.

Deliberate difference from the WF2 record this one is modelled on: WF2's run
promoted an accepted design into `dev/reference/workflows/`. This one has no
promoted design. R12 produces it.

## Run summary

| | |
|---|---|
| Run | `wf3-experiment-v2` |
| Genre | workflow spec |
| Variant | full |
| Ran | 2026-08-01 → 04 |
| Versions | `design-v1.md` (1903 lines) → `v2` (2969) → `v3` (3195) → `v4` (3420) |
| External rounds | 2 of 2 — cap reached, resolved by owner arbitration |
| Findings | 65 unique; 9 re-raised (each faulting a prior-round resolution) |
| Dispatches | opus 6, fable 2 (tier trigger fired twice) |
| Gates | G1 approved 2026-08-01; **G2 ratified 2026-08-08** |
| Scoped verification | PASS — 0 blocking, 0 major, 7 minor (`sv-1`..`sv-7`) |

## Verdict table

| Stage | Reviewer | Doc version | Verdict | blocking | major |
|---|---|---|---|---|---|
| External round 1 | `codex exec` (clean-room, read-only) | `design-v2.md` | `revise` | 1 | 5 |
| Convergence r1 | driver | — | NOT CONVERGED → revision | — | — |
| External round 2 | `codex exec` (clean-room, read-only) | `design-v3.md` | `revise` | 2 | 5 |
| Convergence r2 | driver | — | NOT CONVERGED, cap reached → arbitration | — | — |
| Arbitration | owner | `design-v3.md` | all 7 accepted, fix required | — | — |
| 6a revision | author (Fable) | `design-v4.md` | scope check PASSED, 47 hunks / 24 sections | — | — |
| Scoped verification | reviewer | `design-v4.md` | **PASS** | 0 | 0 |
| G2 | owner | `design-v4.md` | **ratified as input; not implementable** | — | — |

Convergence was never reached mechanically — round 2 returned `revise` on the
last available round — so the owner's arbitration rulings stand in for the
verdict the cap foreclosed, per the loop contract. `design-v4.md` was authored
against those rulings and verified without a further review round.

## G2 ruling (2026-08-08)

Presented: approve design-v4; ratify the risk-7 part-3 rejection; sv-minor
handling; body budget (1903 → 3420).

| item | ruling |
|---|---|
| **design-v4** | Ratified as an **architectural input to R12**, not as a spec to implement. The run is closed; the document is not superseded by a better design but by a changed tree. |
| **risk-7 part 3** (data-derived wall-clock ceiling) | **Rejection ratified.** Independent of everything R9/R10/R11 moved. `ext1-2`'s objection to the accepted parts 1–2 is on the record and travels with this entry; it is not resolved, it is carried. |
| **sv-1..sv-7** | Author's discretion, as filed. Re-derivation supersedes minor editorial findings. |
| **body budget** | Noted, not accepted as a floor. The document nearly doubled while its subject was being invalidated beneath it; R12's re-derivation is **permitted to be shorter**, and should not treat v4's length as a baseline to carry forward. |
| **branch** | `docs/wf3-redesign` stays a branch, cited by path. Not merged. |

## Why nothing was promoted

The run's process was sound and its output is unimplementable, which are not in
tension: the tree it was designed against no longer exists. It ran 2026-08-01→04
against the pre-R9 layout; R9 moved the project tree, R10 renamed rules, and R11
changed what WF3 emits and what its members are called. `main` is 289 commits
ahead of the branch head.

Measured against `design-v4.md` on 2026-08-08 (the roadmap's earlier "~74" was
taken 2026-08-07, before R11 P2/P3 landed):

| stale identifier | hits | invalidated by |
|---|---|---|
| `cst_` | 98 | R11 P2 — member token is `st_` |
| `Qstats` | 25 | R9 — now `q_indicators.csv` |
| `aggregate_rlz` | 15 | R11 P1 — **retired as a hard error** |
| `RT_` | 15 | R9 — tables deleted, nothing replaces them |
| `export_wflow_results` | 14 | R9 — rule is `derive_wflow_indicators` |
| `hydrology_runs` | 11 | R9 — tree move |
| `stress_test_design` | 0 | R11 P2 added it; the design has no concept of it |

**The count is not the reason.** Two structural facts are:

1. **`aggregate_rlz` is load-bearing.** `member_id` is `"agg/cst_<m>"` when it is
   true (`:704`); the cell-completeness predicate branches on it (`:683`); the
   `allow_partial` hole-shape is defined under it (`:664`). A manifest-and-ledger
   architecture is an identity scheme for members plus a predicate for when one
   is done — and both are expressed here in terms of a key that is now a
   parse-time error. There is no substitution: the key was not renamed, the
   distinction it drew was dissolved.
2. **A named gate falsifier now expects the wrong result.** `design-v4:2183-2186`
   states `K = 12` for the tracked fixture and that *"`K = 14` … assumed
   `run_historical: true`, which no tracked config sets"*. R11 P3 set it, because
   leaving it false silently cost two of eleven metrics. `GF-21` therefore builds
   a scratch config to reach a state the tracked config now reaches by default —
   and asserts the reduce emits **no `cst_0` row**, which R11 inverted on
   `[R9-5]`. A falsifier that passes only when the implementation reproduces
   behaviour R11 deliberately removed is worse than no falsifier.

Both are correct-as-written and wrong-as-applied, which is exactly the case the
repo's sealing convention exists for: a document made stale by the world moving,
whose paths must not be freshened because freshening produces something that
*looks* current while its data model still assumes a retired key.

It is **not** registered in `dev/reference/sealed-records.yml`, deliberately. That
registry asserts `(REPO / path).is_file()`, so it can only hold documents in the
tree; registering this one would require merging the run onto `main` first. The
WF2 precedent is the right shape — durable record here, scratch in git history.

## What survives, and what R12 inherits

**Survives** (naming-independent, expensive to reproduce): the four-stage
manifest + ledger architecture; `member_hash` identity; resumable sweeps;
epochs and transition legality; the quarantine model with an explicit inventory;
checked atomic publication (`publish_file`); the counterbalanced AB/BA timing
protocol with a median gate; and the 65-row finding ledger with its dispositions.

**Does not survive**: paths, rule names, artifact names, the member-identity
scheme, the completeness predicate, `K`, and every falsifier that names a
pre-R11 artifact.

**R12's first task is the re-derivation**, and its deliverable is a written
mapping from each surviving finding to its post-R11 expression. That mapping is
what makes two external review rounds portable instead of merely archived.

Two findings deserve carrying forward by name, because R11 gave them evidence the
run did not have:

- **`ext2-1`** (baseline lifecycle) — R11 established that whether the baseline
  member exists at all is toggled by `run_historical`/`ST_START`, and that it had
  been silently absent. The design's treatment of `cst_0` as an asymmetric special
  case is superseded by `[R9-5]`: it is an ordinary member.
- **`ext2-7`** (timing gate) — the counterbalanced protocol stands untouched by
  any of this and is directly reusable.

## Provenance

Branch `docs/wf3-redesign` @ `531bcc6` (`chore(dev): preserve the in-flight wf3
v2 design-run record`). Run directory:
`dev/working/design-runs/wf3-experiment-v2/` — `design-v1..v4.md`, `intake.md`,
`status.md`, `ledger.md` (77 rows), `external-review-r1.md`,
`external-review-r2.md`, `internal-review-{architecture,repo-fit,risk,index}.md`,
`scoped-verification-6a.md`, `review-brief.md`, `v2-change-summary.md`,
`observations.md`. Scope authority:
`dev/workflows/wf3-climate-experiment-v2-intake.md` @ `edc0689`.

Assessment behind the G2 ruling: `dev/milestones/r12/g2-assessment.md`.
