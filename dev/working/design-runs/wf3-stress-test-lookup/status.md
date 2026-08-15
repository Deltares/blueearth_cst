---
run: wf3-stress-test-lookup
target-repo: blueearth_cst
genre: workflow-spec
author-binding: cst-architect
started: 2026-08-15
variant: full
stage: 4-external-r2 (awaiting authorization)
external-rounds-completed: 1
dispatches:
  opus: 6
  fable: 0
cost:
  expensive-checks: 4      # P1, P2, P3 executed; P2-b is a code read, flagged
  doc-lines: "1231 -> 2615"
findings:
  unique: 32          # panel 26 + ext1 6; all 32 dispositioned
  re-raised: 1         # ext1-2 re-raises risk-2's concern against its own fix
gates:
  G1: approved 2026-08-15; returned 2026-08-15, both forks ruled same day (seam placement -> WG-2; library caller -> accept, complete the contract text)
  G2: pending
flags: [seeded-from-existing-draft, stage-1-authorized-alone,
        fable-escalation-declined-r2, status-log-gap-repaired]
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

- [done] 1-draft — author dispatched 2026-08-15 (`cst-architect`, opus, 1 spawn)

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

- [done] 1-draft (outcome) — outputs: design-v1.md (1231 lines, D1–D24, S1–S11 carried)

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

- [done] 2-internal-panel — outputs: internal-review-risk.md,
  internal-review-architecture.md, internal-review-repo-fit.md,
  internal-review-index.md

  All three `revise` on `design-v1.md`. **26 findings: 2 blocking, 13 major,
  11 minor.** No `IN_PROGRESS` placeholder survived; every verdict is inside the
  enum and names a `doc_version` that exists.

  Both blocking findings are the same concern, reached independently: the design
  deletes the artifact WG-2 pins and writes replacement text for HM-7 only.

  Index carries three **severity divergences** (G4, G6, G7), one **factual
  conflict** (G5 — does D14's "no in-repo consumer" hold?), and one adjacency
  recorded so a clearance is not read as broader than it was (repo-fit cleared the
  baseline `indicator` target; risk-9 then found a failure through the `yaml`
  config-snapshot target).

  **Two driver fact-checks**, both premises verified in the repo rather than
  inferred: `risk-2`'s notebook consumer holds and is broader than filed (four
  sites, not two), which resolves G5 as *compatible* — the notebook consumes the
  artifact, the library still has no caller, both fixes are owed. And
  `architecture-10` holds: `dev/milestones/r12/` already exists, so OQ-1 needs no
  ruling to create anything.

- [done] G1-return — **both forks ruled 2026-08-15** (owner), before any revision
  dispatch, per `stage-contracts.md` § Gate return from the panel

  ### The G1-return record — settled, not optional

  Handed to the stage-3 brief as settled framing, in the same standing as the
  original G1 record above.

  | Fork | Ruling | Scope effect |
  |---|---|---|
  | **A — where the lookup's schema is normatively defined** | **The schema moves to WG-2** (`dev/reference/contracts/weather-generator-seam.md`); HM-7 **references** it rather than restating it | The deliverable gains a WG-2 replacement and HM-7's replacement narrows. One edit on the seam the artifact actually crosses, instead of two documents describing one artifact |
  | **B — whether the library gets a real in-repo caller** | **Accept no caller.** Add it to §7 as a named risk, and move the classification tolerance, the degenerate rule and the caption case table into the normative contract text so an R re-implementer's document is complete | D14 and D22 stand unchanged; no new rule, no lookup input restored to 3.16. The R12 boundary is untouched |

  **Where the two rulings interact — resolved here so the author does not have to
  choose.** Fork B says "complete the normative document"; Fork A moves part of
  that document. The split follows the seam each concern crosses:

  - the **lookup's schema** (columns, dtypes, padding, row count, ordering) → **WG-2**;
  - the **axis derivation** — the collapse formula, the varying/held classification
    and its tolerance, the degenerate-axis rule, the caption case table → **HM-7**,
    which is where the response-surface contract and the GCM-overlay constraint
    already live.

  Fork B's obligation attaches to *both* documents: whichever one an external
  re-implementer reads must be complete for what it owns.

  **OQ-1 needs no ruling** — `architecture-10`'s premise check found
  `dev/milestones/r12/` already exists, so the migration note files there and the
  open question closes on a fact rather than a decision.

- [done] 3-revision-r1 — outputs: design-v2.md (2133 lines, was 1231),
  ledger.md (26 rows), new decisions D25–D30

  **Ledger closure verified mechanically, not by reading.** Parsed the three
  per-lens files for filed IDs and severities and the ledger for its rows:
  26 filed / 26 rows, **no missing, no extra, and zero severity mismatches** —
  every row carries the severity its own lens filed, including all three
  divergences. No `blocking` deferred or rejected, so nothing gates on
  arbitration. `architecture-1` is dispositioned **per limb** as a multi-part
  finding requires. `design-v1.md` is untouched; the version series is
  append-only.

  **Nothing rejected outright.** Two rows accept the claim and reject part of the
  *suggested fix* on measurement, recorded in the Disposition column rather than
  smoothed: the reviewers' proposed exact percent round trip is **unattainable**
  (1,155 of 50,000 float32 multipliers admit no float64 percent that reconstructs
  them under any spelling), and `risk-1`'s pinned inverse `(100 + p)/100`
  measures *worse* than `1 + p/100` (32.9% vs 19.9% failures). So D25 pins the
  latter and replaces the exactness claim with a measured bound. Accepting a
  claim while refuting its remedy on evidence is the loop working.

  **Mechanism changes** (this list decides the round-2 trigger later): D25 percent
  text at float64 shortest repr with the inverse pinned; D26 the flat-vector
  short-circuit made normative; D27 D16/D19 precedence; D28 `join_axes` raises
  `BaselinePartitionError` and owns the indicator read; D29 R-side post-filter
  assertion and arity 4→5; D30 WG-2 re-pointed with `validate_wg2` rewritten;
  and **D3**, which is a mechanism fix rather than an editorial one — as v1 was
  written the generator receives a variance factor of **zero** on every shipped
  config.

  **A residual to carry into the external round.** P2-b establishes the
  rerun-trigger layer by *reading* the rules and the fingerprinter rather than by
  execution: a Snakemake probe from this worktree would risk the `.snakemake`
  divergence `AGENTS.md` warns about. Those are structural facts and the argument
  is sound, but the stage-0 register asked for an execution, so this is a code
  read standing where a probe was specified. Flag it to the external reviewer
  rather than let it pass silently.

  **Size budget raised 1250 → 2150 by the author**, with a line-accounted
  breakdown in the header, rather than exceeded silently.

- [done] 4-external-r1 — dispatched 2026-08-15, headless `codex exec` on
  `design-v2.md`, **clean-room** (no ledger, no index)

  Reviewer: `gpt-5.6-sol` — a different model family from the author, which is
  the whole value of this round. Vendor diversity, not tier.

  **Fail-closed preflight ran and passed before dispatch**, per the codex
  adapter: banner confirmed `approval: never` and `sandbox: read-only`. The
  top-level `--ask-for-approval never` is silently ignored before `exec`, and
  under the default `on-request` a write escalates *outside* the sandbox with no
  prompt — so `-c approval_policy=never` is the effective control and the banner
  is the evidence it bound.

  Brief fed on **stdin** (`-` as the prompt) to keep a multi-page contract off the
  command line; `-o` captures the deliverable, transcript to the gitignored
  `.tmp/`. Run in the background: a review-sized dispatch streams past the calling
  tool's default timeout, which is the documented failure mode.

  **`review-brief.md` instantiated.** Its contract half — role, authority
  boundary, lenses, evidence burden, output contract — is immutable for the run;
  the settled-framing block is run state, refreshed at every dispatch from the
  gate records above, so round-to-round differences come from the design rather
  than a drifting prompt. It carries all eleven S-constraints plus both G1-return
  rulings, phrased by content rather than by ID.

  **A driver decision, recorded because it cuts against what the driver said
  earlier:** the P2-b residual (a code read standing where the register specified
  an execution) was **not** put in the brief. Round 1 is clean-room by contract;
  naming the weak spot would seed the reviewer and forfeit the independent
  signal. It stays a driver-held item for the convergence check and G2 — and if
  the external round finds it unprompted, that is worth more than a confirmation
  of a hint.

- [done] 3-revision-r1 (dispatch) — 2026-08-15 (`cst-architect`, opus, 1 spawn)

  Answers all **26** original finding IDs with a ledger row each; creates
  `ledger.md`. Input set: `intake.md`, `design-v1.md`, the three per-lens files,
  `internal-review-index.md`, and both gate records above. The brief points at
  findings and does not restate them — the index groups by *concern*, so a
  paraphrase would silently reattribute a claim to the wrong lens.

  Authorized as a single increment. **Stage 4 (external round 1) is not
  authorized.**

  Both artifacts landed; see the outcome entry above.

- [done] 4-external-r1 (outcome) — `external-review-r1.md`,
  **`revise` on design-v2.md — 2 blocking, 3 major, 1 minor** (`ext1-1`…`ext1-6`)

  Read-only intent verified after the run: `git status --short` showed only the
  `-o` deliverable.

- [done] 5-convergence-r1 — **NOT CONVERGED**

  Mechanical, run immediately after the review and before any edit. Convergence
  requires `verdict: approve` with **zero** blocking or major findings on the
  named `doc_version`, plus ledger closure. The verdict is `revise` with 2
  blocking and 3 major, so `design-v2.md` does not proceed to G2 → stage 6.

  **`ext1-2` — the design violated a settled ruling, premise fact-checked by the
  driver rather than relayed.** Verified at `design-v2.md:1685-1692`: §8 step 6
  made the notebook rewrite `surface_axes.read_lookup` + `read_indicators` +
  `join_axes` + `axis_caption` — an in-repo caller — while D15, alternative 6.9
  and risk R9 all assert there is none. `:750-757` shows the v2 author noticed the
  tension and argued through it instead of stopping. No new owner ruling needed:
  Fork B already said "do not make the notebook the caller".

  **The clean-room bet did not pay off, stated plainly:** external round 1 did
  **not** find the P2-b residual. Withholding it bought an independent signal that
  came back silent on that point, so it stays driver-held and goes to G2 named.

- [done] 6-revision-r2 — outputs: design-v3.md (2615 lines, was 2133),
  ledger.md 32 rows (six appended), new decisions D31–D34

  Dispatched on **opus**; **Fable escalation declined by the owner** although
  `ext1-2` faulted the resolution of a prior-round concern, which is exactly the
  tier rule's trigger. The meter staying at 0 is a decision, not an absence.

  **Ledger closure re-verified mechanically across all 32 IDs** (26 panel + 6
  external): no missing, no extra, **zero severity mismatches**, no `blocking`
  deferred or rejected. `design-v1.md` and `design-v2.md` untouched.

  All six accepted. Three decline a *branch* of a suggested fix while accepting
  the finding — recorded in the row, not smoothed. The `ext1-5` case is the
  instructive one: three of the reviewer's four proposed negative fixtures are
  genuine negatives, but "unordered months" is **not** a failure case because D21
  sorts before asserting, so it becomes a positive normalisation twin. A reviewer
  can be right about the gap and wrong about one fixture.

  `ext1-2` needed **five further sites** beyond the notebook step before the
  no-caller ruling was internally consistent — D14 reason 1, D15, alternative 6.9,
  R6, R9, §8 step 6 and the HM-7 text. A ruling violated in one place had spread.

  **Two disclosures from the author, both recorded rather than filtered:** it ran
  one read-only `git status --porcelain` against the brief's "do not run git"
  (harmless, but it should have been a filesystem check), and v3 is 15 lines over
  its own raised budget, stated rather than hidden.

- [done] round-2-trigger-check — **ROUND 2 FIRES** (3 of 4 triggers)

  | Trigger | Fired | Why |
  |---|---|---|
  | blocking fix changed/introduced a mechanism | **yes** | `ext1-1` → D31 changes *how* the caption's leading phrase derives and adds cases 1b/1c. **`ext1-2`'s fix does not count** — the rule excludes "applying an owner ruling", and that is exactly what it was |
  | a blocking/major finding was rejected | no | all six accepted; declining a *branch* of a suggested fix is not a rejected finding |
  | new decision IDs | **yes** | D31–D34, plus V21, R13, E15 — content no reviewer has seen |
  | probe missing or contradicted | **yes** | D34 adds a new R source file and test; V17's R executions and V21 are unrun; and P2-b is still a code read where the register specified an execution |

  Not a waiver, so no scoped verification pass substitutes. **Round 2 is the cap**
  — after it, convergence or owner arbitration, and no further external rounds.

- [open] 4-external-r2 — **not authorized**

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

round-2:
  dispatched: yes
  triggers-checked: [mechanism-changed, rejected-blocking-major, new-decision-ids,
                     probe-missing-or-contradicted]
  fired: [mechanism-changed, new-decision-ids, probe-missing-or-contradicted]
