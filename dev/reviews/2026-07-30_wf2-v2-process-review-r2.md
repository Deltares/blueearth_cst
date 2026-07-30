# Process review — WF2 v2.0 (Phase 5 / R8), revision 2

```
Date:       2026-07-30
Supersedes: sections 1, 3 and 5 of dev/reviews/2026-07-30_wf2-v2-process-review.md
            (sections 2, 4, 6 stand, with the corrections in section 1 below)
Inputs:     that review, plus two independent critiques of it —
              dev/working/2026-07-30_process-review-critique-fable.md  (Fable, xhigh)
              dev/working/2026-07-30_process-review-critique-gpt.md    (GPT-5.6)
Written by: the driver being reviewed. Every concession below is argued from the
            critiques' evidence; every disagreement is argued from the repo.
Status:     mid-milestone; steps 4d, 5a-5f, 6a-6c, 7 and the seal remain
```

Both critiques were dispatched blind, from the same brief, without each other's
output. They agree on three corrections and disagree on two remedies. This
revision records the adjudication, one **measured** finding that neither critique
had (§2), the revised actions for the remaining steps (§5), and the revised
skill-improvement candidates (§6), which replace the original §5.

---

## 1. Errata — where revision 1 was wrong

Conceded, all five. Both critiques independently reached E1-E3.

| # | Revision 1 said | Correction | Evidence |
|---|---|---|---|
| E1 | "The implementation received **no review at all**"; "zero code gates" | Wrong as stated. Every one of the defects was caught by an implementation-side gate — unit test, dry-run, forced-rerun experiment, `semantic_tree_diff`, re-read. What was missing was a **second reader** and a **pre-declared falsifier**, not gates | the review's own §1 table, *Caught by* column; suite 524 → 583 over six steps (handoff §2) |
| E2 | "Eight defects" | Seven at most, and not seven independent signals. The bounds row was an **evidence** defect (a rounded-comparison claim), caught by **Gate 2 — a design-mandated gate**, which demonstrates the opposite of the headline. The sanitized-wildcard row is counted twice; the recurrence is one missing representation invariant, not two votes for code review | step-1 validation record §"Gate 2 — fired"; `4e5944e` ("relearned the hard way") |
| E3 | "none design-shaped" | False for the `update()` row. D9's revalidation mechanism was authored in `design-v4.md` — the post-arbitration revision that **no reviewer ever saw** — and Snakemake's output-removal semantics make it unimplementable as written. That is a design-feasibility defect of the same class external round 1 caught twice | step-2b record ("D9 item 3 could never have fired"); review record, Run summary ("approved at G2 without a further review"); `966b514`, `b7698b9` |
| E4 | "the loop paid for itself" (one defect) | The conclusion holds; the argument was the weak one available. The strong measure — how many of the 28 findings changed **code behaviour** rather than document text — was in the ledger and unused (Fable F7b counts ~14). The 2014 earliness-vs-detection question stays a **hypothesis**: `.sel(time=slice(...))` clips silently, so first-run detection is not established either way | `get_stats_climate_proj.py:261`; both critiques, independently |
| E5 | proposed and applied remedies in the same document | Diagnosis, intervention and after-measurement were not separated. Two remedies landed in `b9e8556` while that commit's own fixture gates were outstanding — and one of them is now measured wrong (§2) | `b9e8556`; review §7 |

What survives unchanged: §2 (efficiency — both critiques accept the findings and
fault only the missing cost denominator), §4 (verification ranking — Fable calls
it the best part of the document), §6 (carry-forwards).

## 2. Measured: the cache-invalidation fix is unsafe as landed

Fable F4 argued from source that `kernel_hash` filters *every* string constant.
Probed directly against the landed implementation
(`blueearth_cst/projections/series_identity.py:137-177`, python 3.12.13):

```
MISSED       dim kwarg  time -> month
MISSED       variable key  pr -> tas
MISSED       resample code  MS -> YS
MISSED       date bound  2014 -> 2020
MISSED       default arg  273.15 -> 0.0
INVALIDATES  numeric constant (control)
```

**Measured, not estimated.** Five of five behaviour-changing edit classes reuse
stale cached series silently. Two are live in the enumerated kernel today —
`resample(time="MS")` and `groupby("time.month")` at
`get_stats_climate_proj.py:90-103`, plus `drop_duplicates(dim="time",
keep="first")` at `:298`. The fourth row is the **2014-window class itself**: the
defect the loop is credited with catching is a defect this guard would not
notice. `_is_docstring` excludes constants by type (`isinstance(const, str)`),
and `__defaults__` / `__kwdefaults__` are never hashed.

No stale hit has occurred: `b9e8556` is the last commit touching `blueearth_cst/`
or the Snakefile (`git log -- blueearth_cst/ Snakefile_climate_projections`), so
nothing has been edited under the weakened guard. The guard must be repaired
**before the next reducer edit**, which is step 5a.

The adjudication that neither critique reached: *any* safe scheme reinstates the
error-message invalidation cost that motivated the narrowing — an error string is
a code-level string constant under every proposal on the table. So the conflict
is not safety-versus-speed, it is **ordering**. Sequence:

1. **Now** — narrow the exclusion to the actual docstring (`c is func.__doc__`,
   not `isinstance(c, str)`), hash `__defaults__` / `__kwdefaults__`, fold the
   `pixi.lock` sha256 into the digest (closes a dependency-upgrade hole *file*
   hashing never closed either). Add the five probe cases above as tests — they
   fail today, which makes them the falsification experiment for the guard
   itself. Record as a design amendment against risk-03's disposition and fix §9
   test (e)'s wording. Accepted cost: error-string edits invalidate again.
2. **Then** the fetch → reduce split, at which point re-reduction is seconds and
   that cost stops mattering.
3. **Then, optional** — promote to a docstring-stripped module AST hash (`ast`,
   stdlib), which additionally closes the same-file unlisted-callee hole. Only
   worth it once helpers are extracted from the single 270-line kernel.

This is Fable's mitigation list, in GPT's severity ranking, with GPT's
normalized-AST endpoint deferred until it is free. Reverting to `module_hash`
(GPT's verdict) pays step 1's cost without step 1's env-lock gain.

## 3. Revised diagnosis

Not "design review versus code review". Four distinct gaps, in severity order:

1. **Unreviewed final content.** The accepted design's last revision post-dated
   its last review. Four decisions (D9-D12) entered with zero adversarial
   review; one of them shipped an unimplementable mechanism. Structural, not
   tuning: every capped loop that permits a post-cap revision has this class.
2. **Gates specified but not materialized.** `semantic_tree_diff` had no
   reference tree until step 4b, so steps 1-4a are permanently ungateable; the
   manifest was known thin ("3 PNG file sizes") and was run six times at full
   network cost anyway. Worse, round 1 *had* proposed widening it — risk-04's
   third fix component, "widen manifest WF2 coverage before step 3 lands", is
   absent from the disposition and from the validation plan. The ledger tracks
   findings, not fix components, so a component vanished at "accepted".
3. **Falsifiers improvised rather than declared.** The two highest-value catches
   (`update()`, the zero-network assertion) came from experiments designed on the
   spot. The five-rung ladder's expensive rung fired **zero** times across six
   steps (15/15 every time, handoff §2) while every real catch came from a cheap
   rung or an improvisation.
4. **No cost denominator.** Dispatch counts exist; stage durations, artifact
   growth, expensive-command counts and finding yield do not. "Proportionate"
   was therefore an assumption, and round-cap or lite-variant tuning cannot be
   calibrated from finding counts.

Shared premises defeat model diversity: both reviewers independently found the
2014 defect and both inherited the same rounded-bounds claim. Independence of
*reviewers* is not independence of *evidence*.

## 4. Adjudicated verdicts on the six simplifications

| Simplification | Fable | GPT | Adjudicated |
|---|---|---|---|
| Batch value-neutral structural steps | reject as stated | accept w/ mitigation | **Reframe.** The cost was never commits, it was running the expensive rungs per sub-step. Keep per-cause commits and cheap rungs per commit; run network + baseline + tree-diff at declared batch boundaries. Batching criterion is **same invariant and same cheap falsifier**, not "value-neutral" — never batch cache-identity, provenance or config-contract changes |
| Narrow cache invalidation to bytecode | accept w/ mitigation | reject | **Unsafe as landed — repair per §2, in that order.** Escalated from both by measurement |
| Small fast config for iteration | accept | accept w/ mitigation | **Accept.** Add: the "never point `check_baseline` at it" rule moves into the config file's own header comment; add synthetic ragged / multi-member tests; never re-record a baseline from it |
| Code-review gate instead of a design round | reject replacement, accept addition | reject as blanket rule | **Addition only, and instrumented.** Demoted from revision 1's headline. Run `/code-review` per diff on the remaining steps as a **measured trial**: record its yield per step. Catching an `update()`-class or tree-diff-class defect promotes it; catching only diff-local defects confirms both critiques' estimate (2-4 of 7) |
| Compress the design at seal | accept w/ mitigation | accept w/ mitigation | **Accept with both mitigations.** Mechanical relocation only, no paraphrase of retained text; schemas, formulas, config contracts, §8/§9 verbatim; stable decision IDs plus a decision-to-section map; after the last migration step; the diff gets G2 editorial classification. A 2700 → 600 line rewrite is otherwise the same unreviewed-authoring hole as §3.1 |
| Split fetch → reduce with a raw cache | accept w/ mitigation, sequence after the `ssp585` diagnosis | accept w/ mitigation, benchmark one source first | **Accept, gated on one probe.** Diagnose `ssp585` and benchmark one source before committing to the split — the benefit estimate is dominated by an undiagnosed tail. Then extend the D9/D12 discipline one layer down: raw entries keyed by physical source identity + acquisition window, raw content digest folded into the reduce key, atomic write, coverage/schema validation on read, poisoned-raw-cache test, pruning path. Design amendment to §5.1-5.3 required |

## 5. Revised actions for the remaining steps

Supersedes the handoff's §4 ordering.

1. Clear the blocker; confirm 4c's two gates (handoff §3). Unchanged.
2. **Repair `kernel_hash` per §2 step 1, before any reducer edit.** New, and it
   precedes 4d — the five probe cases become tests in the same commit.
3. **One `ssp585` diagnostic probe** plus a one-source fetch/reduce benchmark.
   Gate on the result: if the reads can be brought near the ~6-minute profile of
   the other six, re-price the split before building it.
4. 4d — stage-B collapse + `composition.csv`. With `/code-review` on the diff,
   yield recorded.
5. Snapshot a fresh reference tree **before** each value-changing step; per-cause
   commits for 5a/5b/5c; cheap rungs per commit, expensive rungs at declared
   batch boundaries.
6. For each remaining step, write the falsifier **into the step's plan before
   coding it**: the observation that would disprove the step's claimed property,
   and the command that produces it. Record red→green catches in the validation
   record, not only terminal passes.
7. Seal per handoff §4.6, with the compression mitigations in §4 above.

## 6. Revised skill-improvement candidates

Replaces the original §5. Ranked by benefit ÷ cost; targets are the canonical
brain artifacts under `~/workspace/brain/artifacts/skills/`.

1. **`task-brief` + `design-review-loop` — claim → falsifier contract.** Every
   claimed runtime property names the observation that would disprove it and the
   command that produces it, as a required Validation line; at Stage 0/G1 every
   blocking-risk empirical premise records source, exact observation, precision,
   reproduction command, confidence. Both critiques rank this first
   independently; it is what actually found this run's two worst defects, by
   improvisation. **Implement first if only one is possible.**
2. **`design-review-loop` — close the post-cap unreviewed-content hole.** A
   post-cap arbitration revision gets a scoped verification pass confined to its
   newly-introduced decision IDs — one reviewer, or a named feasibility probe per
   new mechanism — before G2 or as the first implementation act touching each.
   The delta is small and enumerable by construction (here: four IDs), and it is
   where this run's worst design defect was born.
3. **`design-review-loop` Stage 0 — materialize every gate the plan relies on.**
   List the gates the migration table cites; for each, verify it can execute
   today; if not, create it or re-plan. Subsumes the original §5.3 (snapshot the
   reference tree) and generalizes it to thin manifests and absent fixtures.
   Framework-sensitive mechanisms get a probe, not prose.
4. **`task-brief` — gate schedule, not just gate order.** The validation ladder
   carries a per-rung **frequency**: cheap rungs per edit, expensive rungs at
   named risk boundaries. Never merge commits to save gate cost. Add the batching
   criterion from §4 (same invariant, same cheap falsifier).
5. **`design-document` + `design-review-loop` Stage 7 — split normative contract
   from rationale.** Declare a normative-body budget at intake; rationale and
   superseded alternatives accrue to the companion review record *during* the
   loop, which exists for exactly that. Seal compression then becomes a near
   no-op; any non-mechanical compression is a new version needing equivalence
   review, with stable decision IDs and a decision-to-section map.
6. **`design-review-loop` ledger schema — disposition per fix component.** A
   multi-component suggested fix is dispositioned component by component
   (adopted / rejected-with-reason), so a dropped component is visible at G2
   rather than silently narrowed. §3.2's risk-04 case is the evidence.
7. **`design-review-loop` `run-artifacts.md` + validation-record convention —
   make the next process review auditable.** `status.md` records stage start/end,
   dispatch count per tier, artifact line delta, unique vs re-raised findings by
   severity, owner decisions, and durations of expensive commands; validation
   records log **catches (red→green), not only terminal passes**; post-run
   reviews label every cost and benefit *measured / estimated / assumed*.
8. **`scientific-workflows` — a validation baseline is never a mutation
   target.** A tree that serves as a validation baseline is not also written by
   long-running jobs: run into scratch and promote after gates pass. Reframes the
   original §2.3 lesson from job control (mitigation) to fixture architecture
   (fix) — all three background-kill incidents share that one cause.
9. **`snakemake` — record the output-removal / `update()` interaction.**
   Unchanged from the original §5.5; both critiques endorse it as filed.
10. **`scientific-workflows` — one diagnostic probe before building process
    around a performance asymmetry.** Generalizes the original §2.2.
11. **Global `AGENTS.md` — use Write/Edit for content with escape sequences.**
    Unchanged from the original §5.6, and the weakest item here: one session,
    three occurrences, arguably a one-off.

Dropped from the original §5: "pair the design gate with a code gate" as a
*headline* remedy (demoted to §4's instrumented trial) and the unqualified
document-size cap (subsumed by candidate 5, which fixes the cause rather than
the symptom).

## 7. Left open, and what would settle it

- **The 2014 counterfactual.** Implement v1's 2000-2020 window in a scratch
  branch against one historical store and observe whether anything errors,
  warns, or silently truncates, and whether any existing gate fires. Settles
  earliness-versus-detection either way. Neither critique claims it is settled.
- **Code-review yield.** The §4 trial answers it with data instead of estimates.
- **Loop cost versus implementation cost.** Not recoverable for this run; only
  candidate 7 makes the next one measurable.
- **Round 2's value.** Fable defends it (its regression duty caught an
  inadequate fix: ext1-08 → ext2-02 → D10); GPT proposes one round plus a
  walking skeleton. Unresolved and left so: candidates 1-3 move executable
  evidence earlier, which is the part both agree on, and a second round informed
  by a probe-backed design is a different instrument from the one either critique
  priced. Revisit at the next loop with candidate 7's data in hand.
