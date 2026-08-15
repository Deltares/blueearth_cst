---
run: wf3-stress-test-lookup
target-repo: blueearth_cst
genre: workflow-spec
author-binding: cst-architect
started: 2026-08-15
variant: full
stage: 2-internal-panel
external-rounds-completed: 0
dispatches:
  opus: 4
  fable: 0
cost:
  expensive-checks: 3      # P1, P2, P3 — all executed, not argued
  doc-lines: "1231 -> 1231"
findings:
  unique: 0
  re-raised: 0
gates:
  G1: approved 2026-08-15
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

- [done] 1-draft — outputs: design-v1.md (1231 lines, D1–D24, S1–S11 carried)

  **Driver structural checks pass.** Alternatives section non-empty (5 entries,
  each naming the condition under which it would become preferable); all eleven
  settled constraints present as S1–S11; version series append-only; no findings
  or verdicts yet, so the ledger checks are vacuous at this stage.

  **Genre deviation, recorded not corrected.** `status.md` declared
  `workflow-spec`; the author wrote the repo's own design house style (§ Problem
  / Goals / Selected approach / Alternatives / Consequences / Migration /
  Validation / Open questions), citing `design-document`'s software-system clause
  and the p32b precedent, on the ground that `workflow-spec`'s `Owner role` and
  `Roles, skills` headings have no honest content for a data-contract change.
  The enum stays as declared per `run-artifacts.md` (note the mapping, do not
  invent a value); the shape maps to `decision-record`. Driver accepts as a fact
  check, not as authorship.

  **All three feasibility probes executed** rather than argued — the point of the
  stage-0 register. P1 required a second, faithful synthetic after the first
  failed to reproduce the cycle it was testing for; the author rebuilt it rather
  than bank the non-result.

  **E9 and E10 both settled** with recorded observations, so no hypothesis was
  promoted to a fact.

  One item returns to the owner as **OQ-1** — the migration note's path. A
  stage-0 self-containment gap, not a design choice: `naming.md` §7 mandates
  `dev/<milestone>/migration_<topic>.md` and this work lands before R12 with no
  milestone directory.

- [done] G1 — **approved 2026-08-15** (owner)

  ### The G1 record — settled framing for every downstream spawn

  Approved as written in `design-v1.md`: the problem statement (§1), the eleven
  settled constraints (§3, S1–S11), the decision criteria (§4), and the
  **provisional** selected approach (§5, D1–D24). Downstream reviewers receive
  this as settled: a reviewer may argue a *consequence* of any item below, and
  may not re-litigate the item itself.

  | | Approved |
  |---|---|
  | Problem | The experiment and the response surface are fused; the fixed annual collapse misreports a seasonal design (+30% in JJA reads as +7.6%), and `stress_test_design.csv` is a materialized cache of the member files |
  | Constraints | S1–S11 — percent everywhere; the lookup is the source of truth; the lookup determines the axis, not the scenario; no external consumer constrains this; `stress_test_lookup.csv` in `<exp>/config/`; `st_0` is not a surface member; the identity member is simulated like any other; the lookup lands before R12's identity re-derivation; linear statistics only; the overlay inherits the collapse; workflow-engine scope only |
  | Decision criteria | §4 C1–C6 — correctness first; store the finest grain imposed and derive every summary; no new cache of a derivation; a new parameter is a column not a file shape; the migration is executable in one commit; every claimed runtime property has a falsifier |
  | Provisional approach | The `reporting:` top-level section (D8/D9), `months` defaulting to the member-varying set (D11), a library rather than a rule (D14), two-tier enforcement (D16/D17) |

  **Carried forward as still-open, not approved:** OQ-1 (where the migration
  note lands) was put to the owner at this gate and not ruled. It stays an open
  question in §10 and is available to settle at G2, when the note's final content
  is fixed. It blocks nothing in stage 2 — it is a path, not a decision.

- [open] 2-internal-panel — three lenses dispatched 2026-08-15 (opus ×3)

  risk (`critical-thinker`), architecture (`cst-architect`), repo fit
  (`python-engineer`). Fresh spawns, no conversational memory, each given
  `design-v1.md`, `intake.md` and the G1 record above. Marked `[open]` until all
  three verdict files plus `internal-review-index.md` are on disk.

  Authorized as a single increment; nothing beyond stage 2 is authorized.

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
