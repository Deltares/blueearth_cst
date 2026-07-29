---
run: wf2-climate-analysis-v2
target-repo: blueearth_cst
genre: workflow-spec
author-binding: generic
started: 2026-07-29
variant: lite  # PROMOTION TRIGGERED -> full (blocking findings)
stage: 5-convergence-r1
external-rounds-completed: 1
dispatches:
  opus: 0
  fable: 1
gates:
  G1: approved 2026-07-29
  G2: pending
flags: [owner-requested-fable-lens, promoted-lite-to-full, blocking-findings-open]
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
- [done] 2-internal-panel (lite: single lens) — outputs: internal-review-risk.md
  (verdict: revise on design-v1.md; 1 blocking, 5 major, 3 minor). Reviewer:
  `critical-thinker` on **Fable**, owner-requested.
- [done] 4-external-r1 — outputs: external-review-r1.md (doc_version:
  design-v1.md; verdict: revise; 3 blocking, 6 major, 1 minor), codex-transcript.txt
- [done] 5-convergence-r1 — NOT converged. 3 distinct blocking defects
  (risk-01 = ext1-01; ext1-02; ext1-03). Promotion trigger fired: lite -> full,
  external cap now 2. Index: review-index.md
- [open] G1-return — four findings are owner decisions, not author fixes
  (review-index.md § "Findings that are owner decisions"); a ruling on the
  reference-series construction or the v2 scope claim changes the selected
  alternative and must precede any revision spawn.

## Driver premise verification (2026-07-29)

The driver checks facts, never authors. Three findings' premises verified
against the repo before arbitration:

- **risk-01 — HOLDS. Regression, not pre-existing.**
  `blueearth_cst/projections/get_stats_climate_proj.py:156` hardcodes
  `time_tuple_all = ("1950-01-01", "2014-12-31")` for cmip6 historical, and
  `config/catalogs/cmip6_data.yml` resolves historical under
  `gs://cmip6/CMIP6/CMIP/{model}/historical/`. `shared.historical_window` ends
  2020-12-31, overrunning the source by six years. The *current* code's
  `historical_year_range: [1990, 2010]` fits inside the historical experiment,
  so this defect is introduced by the design's G3, not inherited.
- **risk-04 — HOLDS.** `dev/baseline/manifest.json` pins exactly 7 WF2 targets,
  all under `climate_projections/cmip6/`: 3 PNGs, `annual_change_scalar_stats_summary.{nc,csv}`,
  `..._summary_mean.csv`, and the config snapshot. No monthly intermediates are
  covered, so a green `check_baseline` constrains the final scalar summary only.
- **risk-09 — HOLDS.** 1+9+1+1+1+1 = 14, not the 13 stated in §5.1/§7; and the
  reduce count omits the observed series that §5.2 routes through stage A.

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
