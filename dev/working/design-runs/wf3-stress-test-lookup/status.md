---
run: wf3-stress-test-lookup
target-repo: blueearth_cst
genre: workflow-spec
author-binding: cst-architect
started: 2026-08-15
variant: full
stage: 1-draft
external-rounds-completed: 0
dispatches:
  opus: 1
  fable: 0
cost:
  expensive-checks: 0
  doc-lines: "-> -"
findings:
  unique: 0
  re-raised: 0
gates:
  G1: pending
  G2: pending
flags: [seeded-from-existing-draft, stage-1-authorized-alone]
---

# Run state — wf3-stress-test-lookup

## Stage log

- [done] 0-intake — outputs: intake.md, run dir, status.md

  Driver-only; no dispatches. Materialized the change request, the six scope
  gaps, the settled-constraints table (11 rows, all owner rulings), decision and
  success criteria, non-goals, the **evidence register** (E1–E10; eight verified
  this session, two carried as explicit hypotheses), **three
  framework-feasibility probes** (P1–P3), the **gate-materialization check**, the
  **derived-artifact register** (9 artifacts), the genre mapping, and the seeding
  decision.

  Two stage-0 findings worth carrying to G1 rather than burying in the register:

  1. **`check_baseline.py` needs a pre-change re-record before the first
     implementation commit.** The gate will fail by design (the indicator tables
     lose two columns), and two open board items say the current baseline cannot
     serve as the "before": `t2608131718` and `t2608121258`. A comparison gate
     cannot be applied retrospectively, so every step landing before the
     re-record is permanently ungateable.
  2. **The artifact this design replaces is outside the numerical gate
     entirely.** `stress_test_design.csv` was deliberately kept out of the
     baseline manifest by an R11 ruling, so `check_baseline` says nothing about
     the correctness of the replacement. Carry to G2 as a named gap.

- [open] 1-draft — author dispatched 2026-08-15 (`cst-architect`, opus, 1 spawn)

  Structural checks on the seed source FAIL (no `## Alternatives considered`, no
  genre sections), so stage 1 takes the restructure path rather than the
  mechanical copy.

  **Authorized alone**, not as part of the whole loop: the user approved stage 1
  only, so G1 is seen before the panel spends three further dispatches. Nothing
  beyond this spawn is authorized; stages 2+ need their own decision.

  The brief holds two obligations together, per `observations.md` O1: preserve
  the eleven settled constraints verbatim in substance, **and** write the six
  declared scope gaps as new normative content. It also carries P1–P3 (specify
  or run), and the instruction not to promote E9/E10 from hypothesis to fact.

  Marked `[open]` until `design-v1.md` is on disk — write-then-mark.

## Variant

`full`, not `lite`. The change is not contained: it alters a data contract
consumed across Python (`prepare_cst_parameters.py`, `export_wflow_results.py`,
`shared/interchange_contracts.py`), R (`impose_climate_change.R`), a Snakefile
(rules 3.09 / 3.12 / 3.16), and a normative contract document (HM-7) — and it adds
a reporting layer that has no existing pattern in the repo to extend.

## Entry criteria

Met on two of three counts: the change alters data contracts across more than one
tool/stage, and the axis-declaration layer is a new direction with no repo pattern
to extend.

## Dispatch plan, if authorized

| Stage | Spawns | Notes |
|---|---|---|
| 1 — seed/restructure | 1 author | `cst-architect`; content preserved verbatim |
| 2 — internal panel | 3 lenses | risk (`critical-thinker`), architecture, repo fit |
| 3 — revision r1 | 1 author | fresh spawn, ledger rows for every finding |
| 4 — external r1 | 1 `codex exec` | clean-room, on `review-brief.md` |
| 5 — convergence | 0 | driver |
| 6 — revision r2 + round-2 trigger check | 0–1 author, 0–1 `codex exec` | fired on evidence, not by default |

Floor is round 1 plus its revision: **6 dispatches**. Cap is 2 external rounds:
**8**. Everything else is driver work.
