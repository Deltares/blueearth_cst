---
run: wf2-climate-analysis-v2
target-repo: blueearth_cst
genre: workflow-spec
author-binding: generic
started: 2026-07-29
variant: lite
stage: 4-external-r1
external-rounds-completed: 0
dispatches:
  opus: 0
  fable: 1
gates:
  G1: approved 2026-07-29
  G2: pending
flags: [owner-requested-fable-lens]
---

## Stage log

- [done] 0-intake — outputs: intake.md
- [done] 1-seed — outputs: design-v1.md (seeded from the landed
  `dev/workflows/wf2-climate-analysis-v2-design.md` @ f5cd5ff; not re-authored)
- [done] G1 — approved 2026-07-29. Owner directed review of this doc as written,
  so framing is settled to the extent recorded in the review brief's
  settled-framing block. Provisional selected alternative: the three-stage
  architecture (store / reduce / derive+report) of §5.1. OQ-1..OQ-8 remain
  **open** and reviewer input on them is explicitly wanted.
- [open] 2-internal-panel (lite: single lens) — reviewer: `critical-thinker`
  on **Fable**, owner-requested
- [open] 4-external-r1 — reviewer: `codex exec` / `gpt-5.6-sol`

## Variant note

**Lite variant** — single internal lens + 1 external round, per the owner's
request for two individual reviews before finalizing. Gates, ledger,
convergence, and arbitration are unchanged.

**Promotion trigger:** any `blocking` finding, or non-convergence after this one
external round, escalates to the full variant — the remaining two lenses
(architecture, repo-fit) spawn on the current version and the external cap
reverts to 2.

## Tier deviation (logged)

The skill rations Fable to revision spawns answering an external review that
re-raised a prior finding. Here the owner requested a Fable lens directly, which
overrides the default. Counted honestly: `fable: 1`.

## Preflight

`codex exec --sandbox read-only --ephemeral -c approval_policy=never
-m gpt-5.6-sol` — banner verified 2026-07-29: `approval: never`,
`sandbox: read-only`, `model: gpt-5.6-sol`. Fail-closed control confirmed
before dispatch.
