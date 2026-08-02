# Critique — Reviewer A (Fable/Claude)

```
Date:    2026-07-30
Subject: dev/reviews/2026-07-30_wf2-v2-process-review.md
Brief:   dev/reviews/2026-07-30_design-loop-efficiency-review-brief.md
Basis:   review record, validation records, git log, skill files, source —
         cited per claim. Cost/benefit statements labelled measured /
         estimated / assumed / hypothesis.
```

## Verdict

Partly right. The review's efficiency findings (§2) are well-evidenced and its
verification ranking (§4) is the best part of the document. But the headline
diagnosis is overstated in both directions: "the code got no gate" is falsified
by the review's own defect table, whose *Caught by* column names a working
implementation-side gate for every defect; and "none design-shaped" is wrong
for at least one of the eight — the `update()` defect was born in design-v4, the
only version the loop never reviewed, which is a structural hole in the loop the
review does not see. The proposed remedy (`/code-review` per diff) is cheap and
worth adding, but it is aimed at the defect class the existing gates were
already catching, not at the two mechanisms that actually let defects through:
an unreviewed post-arbitration revision, and gates that were specified but not
materialized (no reference snapshot, a knowingly-thin manifest). The review is
self-critical about tuning while leaving the loop's structure unexamined — the
classic shape of a self-review that concedes small errors to protect a large
one.

## Where the existing review is wrong or unsupported

**W1 — "The implementation received no review at all" conflates *review* with
*gate*, and the defect table refutes the stronger reading.** Every one of the
eight defects was caught before the milestone seal by an implementation-side
gate: a unit test, a dry-run, a forced-rerun experiment, `semantic_tree_diff`,
or re-reading (review §1 table, *Caught by* column). Tests grew 524 → 583 over
six steps (measured: `dev/milestones/r08/2026-07-30_wf2-r8-handoff.md` §2). What was
missing was a *second reader*, not gates. The distinction matters because the
remedy differs: the gates' catches came overwhelmingly from falsification
experiments and tree-diffs (the review's own §4 ranking), so hardening those is
the load-bearing fix; a code reviewer plausibly catches the `NameError` and
maybe the wildcard round-trip, but not `update()` (requires knowing
`Job.prepare()` removes outputs — the step-2b record shows even a focused
investigation needed a live experiment to find it) and not the merged-attrs
defect (caught by tree-diff, invisible in a diff). Code review as *headline*
action treats a symptom. Estimated: it would have caught 2–4 of the 8, and
neither of the two the review itself calls the biggest.

**W2 — "Eight defects, none design-shaped" is wrong for the `update()` row.**
Design D9's revalidation mechanism was authored in `design-v4.md` — the
arbitration revision produced *after* the external round cap was spent, and
"approved at G2 without a further review"
(`wf2-climate-analysis-v2-design-review-record.md`, Run summary). No reviewer
ever saw D9, D10, D11, or D12. The step-2b record is explicit: "Design D9 item 3
could never have fired" — the design specified a mechanism Snakemake's execution
model defeats. That is a design-feasibility defect of exactly the genre external
round 1 demonstrably catches: ext1-02 was the same class (a mechanism
incompatible with DAG execution semantics), filed blocking, on a version that
*was* reviewed. The honest table entry is not "No" but "plausibly yes, had v4
been reviewed — it wasn't." This flips part of the headline: one of the two
worst code-reaching defects traces to a coverage hole *in the loop*, not to the
absence of a code gate. See finding F1.

**W3 — The bounds-rounding row does not belong in the table.** Defect #5
("identical" bounds rounded to 6 dp) was not a code defect: it was an overstated
claim in the *driver's premise verification* (record, Driver premise
verification: "The original check compared rounded values"), and it was caught
by **Gate 2 — a design-mandated gate** (design §8 row 1: "re-verify bounds
equality"; step-1 record: "Gate 2 — fired"). Counting it as one of eight code
defects in a process with "zero code gates" pads the headline with a row that
actually demonstrates the opposite: a design-review-derived check firing in
implementation. Corrected, the sample is seven, one of which is design-shaped
(W2).

**W4 — The allocation claim is asserted, never costed.** The review says review
effort went "almost entirely" to the design but measures neither side. The
measurable half exists: the loop cost 4 full-context spawns (run summary:
"dispatches: opus 2, fable 2") plus 2 codex dispatches plus driver
orchestration; the implementation's waste was ~8 network re-derivations at ~15
min each (review §2.1, estimated) plus repeated full-suite runs. By the
review's own §2.1, the dominant *session* cost was implementation-side and
self-inflicted. "The loop: proportionate" is the right conclusion, but the
review reaches it by defect-counting when the cost side was available and would
have made the case stronger — and would have shown that the owner's
"overengineered loop?" question is mostly answered by cache-invalidation and an
undiagnosed store asymmetry, not by the loop at all.

**W5 — Against the counterfactual the brief poses (and the review leaves
unresolved): the 2014 defect would plausibly *not* have surfaced on first real
run. The loop bought detection, not earliness.** Mechanism: window selection in
this codebase is xarray `.sel(time=slice(...))`
(`blueearth_cst/projections/get_stats_climate_proj.py:261`), which silently
clips a slice that overruns the time axis — no error, no warning. risk-01's own
text states the consequence: "the GCM reference becomes 2000–2014 while the
design labels it 2000–2020 … the window mismatch … recurs invisibly, now with
wrong provenance metadata." Nor would the migration gates have caught it: the
reference-window change was a *value-changing* step whose gate is "re-record
with a characterized diff" (design §8 row 5e) — the wrong-window diff would have
been recorded *as* the intended change. Every artifact that now surfaces the
effective window (A1's nominal/effective columns, the coverage assertion of
ext1-04's disposition) exists because of the loop. Labelled: hypothesis —
settled by checking that pre-v2 code contains no requested-vs-actual window
assertion (it does not; the hardcoded `time_tuple_all` spans at
`get_stats_climate_proj.py:169-183` are the only window logic). On the
circularity: the "propagated into four documents" cost belongs to the *bounds*
incident, not the 2014 defect (step-1 record: "propagated into the design, the
review record, the framing given to both reviewers, and the task brief";
`04013fc`). That incident is genuinely circular — the error originated in the
loop's own premise verification and its cost fell on loop-created artifacts —
and nets to roughly zero externality. Its real lesson is verification craft:
record measured deltas and tolerances, never unqualified identity claims.

**W6 — "13 migration sub-steps" mis-assigns the decomposition.** The accepted
design's §8 table itself has 15 rows (1, 2a, 2b, 3, 4, 5a–5f, 6a–6c, 7); the
driver added only the 4a–4d split, explicitly justified by the design's own §4
criterion 5 logic (`4ae1542` commit message). The review frames sub-step count
as driver over-caution; most of it was owner-accepted design. The
disproportion critique, where valid, indicts the design review outcome — which
approved that plan — not the driver's execution.

## Findings it missed

**F1 — The round cap creates an unreviewed-content class, and this run's worst
design defect was born in it.** Everything authored in `design-v4.md` — D9
(region revalidation), D10 (spherical weighting), D11 (gridded schema), D12
(store index) — entered the accepted design with zero adversarial review: the
cap was spent, arbitration authorized the revision, G2 approved it unread by
any reviewer (record, Run summary and stage-6a note). D9 shipped an
unimplementable mechanism (`update()`, step-2b record). The review's remedies
are all downstream of this hole; none closes it. Structural, not tuning: any
capped loop whose last revision post-dates its last review has it.

**F2 — The review conflates commit granularity with gate granularity, so its
batching remedy buys less and costs more than the alternative.** The cost of
the 13 sub-steps was not commits — it was running the expensive rungs (full
network re-derivation + `check_baseline`) per sub-step (review §3). Those are
separable: keep per-cause commits (cheap attribution, cheap bisection), run
rungs 1–3 (unit tests, dry-run) per commit, and run rungs 4–5 (network,
baseline, tree-diff) once per batch boundary. Captures essentially all the
saving the review's "batch the structural steps" action captures, without the
bisection cost it accepts. The review adopted the coarser fix without pricing
this one.

**F3 — A reviewer-proposed mitigation that would have strengthened the code
gates was silently dropped, and the review does not notice.** risk-04's
suggested fix had three components; two landed (byte-identical pinned paths via
D3; tree-diff with old→new map), but "Consider widening manifest WF2 coverage
(per-model monthly stats) before step 3 lands" (record, risk-04 verbatim)
appears nowhere in the disposition or the validation plan. The process then ran
a gate it *knew* was thin ("coverage is thin … 3 PNG file sizes", design §8) six
times at full network cost. The review complains about the weak fixture (§3)
without registering that its own round 1 had proposed the fix and the loop's
ledger discipline — which tracks findings, not fix components — let it vanish.

**F4 — The `kernel_hash` fix, made during the review itself, has concrete
silent-staleness holes beyond the acknowledged one, and it is an unreviewed
change to a reviewed contract.** The implementation
(`blueearth_cst/projections/series_identity.py:137-177`) filters *every string
constant* from the hash ("no *string* constant is allowed to affect the hash",
`_is_docstring` docstring). In xarray-style reduction code, strings are
load-bearing: `da.mean(dim="time") → dim="month"`, `ds["pr"] → ds["tas"]`,
`resample(time="MS") → ("YS")`, a changed date bound — all are string-constant
edits with identical `co_code`, so none invalidates the cache. Changed default
argument values (`func.__defaults__`) are also unhashed. The test suite's
negative space mirrors the holes exactly: `tests/test_series_identity.py`
tests formula, numeric-constant and attribute-lookup changes (lines 472–500) —
none of the missed classes. Separately: the accepted design specifies
file-level hashing ("hashes an explicitly enumerated list of reducer module
files", risk-03 disposition) and §9 cache test (e) — "edit an enumerated
reducer module → all series re-derive" — is now false as written for
comment-only edits. `b9e8556` amended an ACCEPTED design's mechanism with no
design amendment, mid-process-review, under an efficiency banner. The irony:
risk-03 was filed precisely against silent stale-cache paths.

**F5 — The background-job lesson misdiagnoses a fixture-architecture problem as
a job-control problem.** All three incidents (two half-written manifest
targets, the still-held handle blocking every 4c gate — review §2.3, §7) share
one structural cause: the validated fixture *is* the live run directory, so any
interrupted process corrupts the evidence base itself. "Prefer bounded
foreground calls" mitigates; it does not fix. The fix the repo already
half-discovered is separation: the reference tree (`test_case/
ref_wf2_pre_valuechange`) is a snapshot precisely because comparison needs a
tree the run cannot touch. Generalization: a tree that serves as a validation
baseline is never also a mutation target of long-running jobs; run into scratch
and promote after gates pass.

**F6 — The review's evidence base for "what verification earned its keep" is
partly unauditable, because only green outcomes were durably recorded.** The
top-ranked catch (merged-timeseries identity attrs, caught by
`semantic_tree_diff`) appears in no commit and no validation record: `4e5944e`
records "0 FAILED" (the post-fix state) and 4c's gates never ran (`b9e8556`).
The catch presumably happened red→green during 4b's development and lives only
in the driver's memory — the same memory now writing its own performance
review. Validation records log final passes; the catches, which are the data a
process review needs, are recorded haphazardly (4a's commit logs two; 4b's logs
none). Lesson: record gate *failures and catches* in validation records, not
only terminal passes.

**F7 — What the review chose not to measure.** (a) Loop cost vs implementation
cost (W4). (b) How many of the 28 findings changed implementation behavior vs
document text — the ledger supports this classification and it is the direct
test of "did the loop pay"; a rough pass says at least risk-01/02/03/05/06,
ext1-02/03/04/05, ext2-01/02/04/05/06/07 changed what the code does, which is a
*stronger* defense of the loop than the review's single-defect argument.
(c) Per-sub-step gate yield: the five-rung ladder's expensive rung
(`check_baseline` after full re-derivation) fired zero times across six steps —
every 15/15 passed (handoff §2 table) — while the catches came from cheap rungs
and improvised experiments. The review gestures at this (§3, §4) but never
states the number that would justify F2's restructuring.

## Risk assessment of the proposed simplifications

**1. Batch value-neutral structural steps into fewer commits — REJECT as
stated; alternative named.** The failure mode is real (a failing
`semantic_tree_diff` on a 4-commit batch cannot be attributed; bisection
re-runs the expensive gate per probe, which is exactly the cost being avoided).
The cheaper variant (F2): keep per-cause commits, batch only the expensive
rungs at batch boundaries; on a batch-boundary failure, bisect using the cheap
rungs first. Same saving, attribution preserved.

**2. Narrow cache invalidation to function-bytecode hashing — ACCEPT WITH
MITIGATION (it has already landed; unmitigated, it should be reverted).**
Failure mode: silently stale series feeding real change factors — presents as
nothing at all, the worst kind. Concrete holes beyond the docstring's
acknowledged unlisted-callee case: all string constants (dim names, subscript
keys, resample codes, date bounds) and default argument values are invisible
(F4); library upgrades were never covered by either mechanism. Mitigations,
all cheap, no new dependency: (a) hash `func.__defaults__`/`__kwdefaults__`;
(b) replace the type-based string filter with docstring-only exclusion — an
error-message edit then costs one invalidation, the honest price — or hash a
docstring-stripped AST; (c) fold an environment fingerprint (the `pixi.lock`
sha, already in-repo) into the digest so a library upgrade re-derives;
(d) add cache tests for the currently-missed classes — a changed string kwarg
test would fail today, which is the falsification experiment for the guard
itself; (e) record the change as a design amendment and fix §9 test (e)'s
wording. With (a)–(d), the residual (unlisted callees, data-dependent
behavior) is the same residual the file-hash had, and I accept it.

**3. Small fast config for iteration — ACCEPT.** The risk (hiding multi-model,
multi-member, ragged-availability bugs) is contained by construction, not by
discipline: gates and the manifest stay pinned to the 3-model seed
(`b9e8556`: "Explicitly NOT a baseline seed"), and the ragged/absence classes
are covered offline by the 19 resolution tests plus the real-catalog negative
probes (`4ae1542`). Named residual: any code path conditional on ensemble
cardinality is exercised only at gate time, later — accepted, since the gates
still run per batch. One hardening: the "never point `check_baseline` at it"
rule currently lives in a handoff note; put it in the config file's own header
comment so it survives the session that learned it.

**4. Replace a further design round with a code-review gate — REJECT the
replacement; accept the addition.** The two catch disjoint classes, on this
run's own evidence: design rounds caught method/contract defects (2014 clip,
daily/monthly units, DAG-incompatible failure handling — none visible in any
diff); code review catches diff-local defects but plausibly misses `update()`
and certainly misses the tree-diff class (W1). And the marginal design review
forgone in this run — a look at v4 — is precisely where a defect slipped (F1).
The cheap form of "more design review" is not another full round: it is a
scoped check of the arbitration revision's new decision IDs (here: four
decisions, enumerable from the record). Both gates, each where it works.

**5. Compress the design document at seal — ACCEPT WITH MITIGATION.** The
stated risk (re-litigating settled decisions) is the smaller one; the review
record already holds the audit trail and rulings. The larger risk is that a
~2700 → ~600 line rewrite of an ACCEPTED document is itself an unreviewed
authoring pass — the same hole as F1 — and that "compression" quietly rewrites
normative surfaces. Mitigations: mechanical relocation only (whole superseded
sections move to the review record; no paraphrase of what remains); schemas,
formulas, config contracts, §8/§9 stay verbatim; compress only after the last
migration step (until then the doc is the plan of record); the diff gets G2's
editorial classification (owner glance). With those, accept.

**6. Split fetch → reduce with a raw local cache — ACCEPT WITH MITIGATION, and
sequence it after the `ssp585` diagnosis.** The desync class is real: a raw
slice that no longer corresponds to its pin or region while the reduce layer
trusts it. But the design already owns the pattern that closes it — content
attributes plus revalidation plus read-time pin verification (D9/D12);
extending the same discipline one layer down (raw slices carry pin + region
fingerprint + acquisition window; reduce asserts them; §9-style cache tests
mirrored, including a poisoned-raw-cache case) adds no new mechanism class.
Conditions: it is a §5.1–5.3 contract change and needs the design amendment
the handoff already concedes (§4 item 3); and the benefit estimate ("~15 min →
seconds", review §2 — estimated) is dominated by the undiagnosed `ssp585` tail
(review §2.2), so spend the one diagnostic probe first — if the reads can be
brought near the ~6-minute profile of the other six, the split's payoff
shrinks materially and the second cache may not be worth its coherence
surface. Unamended and undiagnosed: reject, because it reopens ext2-01/
ext2-04's closed staleness classes one layer up.

## Is the loop's shape right?

Mostly yes — the review is right that the loop earned its rounds, and the
round-2 regression duty catching an inadequate fix (ext1-08 → ext2-02 → D10) is
the loop working as designed. The cap is also right: the surviving round-2
questions were owner-decision-shaped (window semantics, defaults, tiers), which
more rounds re-confirm rather than resolve. Three shape changes, in order of
conviction:

1. **Probe before drafting, not a prototype.** The highest-value round-1
   findings were all of the form "the design asserts X; the repo/store says Y"
   — historical ends 2014, observed is daily/K vs `Amon` monthly/degC, the
   manifest pins 7 thin targets, the job arithmetic (record, "What the review
   changed"). Those are facts a pre-draft probe list produces in under an hour;
   the loop currently discovers them via its most expensive instrument
   (external review) and verifies them via driver premise-verification *after*
   filing. Moving the measured-reality register to stage 0/1 converts round 1
   from fact-checking to judgment — the mid-loop catalog crawl (`f8194e8`,
   R2-cat row) is in-run evidence that measured inputs improve the design more
   per token than argumentation. A full spike is not supported by this
   evidence: for a 1500-line workflow it costs a large fraction of the
   implementation, and the loop's probe-shaped checks (fixture store opened,
   cos-lat residual computed) delivered the same value bounded.
2. **Close the post-cap hole (F1)** — a scoped verification pass on
   arbitration-revision content. This is a shape fix, not tuning: the cap is
   fine; the unreviewed revision behind it is not.
3. **Bound the artifact by contract, not by seal-time compression** — rationale
   and superseded alternatives accrue to the review record *during* the loop
   (it exists for exactly this); the design carries normative content plus
   pointers. Growth-by-accretion (each round "*added* rather than replaced",
   review §3) is a `design-document`/loop contract gap, not an authoring
   accident.

Design-then-implement with review only at the end is strictly dominated here:
it has the F1 hole for the entire document. Fewer design rounds + code gates
trades the loop's demonstrated catches (three blocking mechanism defects in
round 1) for a gate that misses that class (risk row 4). Evidence that would
change my mind is in the last section.

## Generalized recommendations, ranked by expected value

Ranked by (benefit ÷ cost). None repo-specific; none needs a dependency.
Recommend only — no skill edits here.

1. **`task-brief`: every claimed property names its falsification experiment**
   — the observation that would disprove it and how to produce it — as a
   required Validation line. (Endorses review §5.4, which is correct and is
   the single highest defects-per-token change available: the two
   highest-severity defects of this run fell to improvised falsification —
   `update()` to a forced-rerun before/after, G5 to a zero-network assertion.)
   Generalizes: any cached, resumable, or "no-op by design" behavior in any
   repo can only be verified by an experiment designed to fail.
   **If only one change were possible, this is it.**
2. **`design-review-loop`, stage-0 intake: materialize every gate the
   migration plan relies on before the first implementation commit.** A gate
   that cannot yet run — no reference snapshot (steps 1–4a, permanently
   ungateable; step-1 record "Limitation"), a manifest known not to cover what
   the plan restructures (F3) — is an intake finding, not a mid-run discovery.
   Subsumes the review's §5.3 (snapshot rule) and generalizes it: the checklist
   item is "list the gates §8-equivalent rows cite; for each, verify it can
   execute today; if not, create it or re-plan."
3. **`design-review-loop`: post-cap arbitration revisions get a scoped
   verification pass** — one reviewer (or a named feasibility probe per new
   mechanism) confined to the newly-introduced decision IDs, before G2 or as
   the first implementation act touching each. Closes F1. Generalizes: every
   capped adversarial loop that permits a post-cap revision has an unreviewed
   final-content class; the delta is small and enumerable by construction, so
   the check is cheap.
4. **`task-brief` (validation-ladder guidance): decouple gate schedule from
   commit granularity.** Cheap rungs per commit; expensive rungs (network,
   fixture re-derivation, baseline) per declared batch boundary; never merge
   commits to save gate cost (F2). Generalizes to any repo whose full gate is
   slow: the ladder should carry a per-rung *frequency*, not just an order.
5. **`design-review-loop`, ledger schema: disposition multi-component
   suggested fixes per component** (adopted / rejected-with-reason), so a
   dropped component is visible at G2 rather than silently narrowed (F3).
   Generalizes: "finding accepted" currently certifies less than it appears
   to, in any run of the loop.
6. **`design-document`: split normative contract from rationale, with
   rationale accruing to the companion record during a review loop.** Caps
   growth at the source; makes seal-time compression a near-no-op; prevents
   the reading-order inversion the review names (§3). Generalizes to any
   reviewed-design workflow with a durable audit artifact.

Two review proposals endorsed as filed, below the cut for ranking only:
§5.5 (`snakemake` skill: `update()` / output-removal note — correct, narrow,
cheap) and §2.2 as a general rule (one diagnostic probe before building
process around a performance asymmetry — belongs beside `scientific-workflows`
hygiene). One addition too small to rank: validation records should log
red→green catches, not only terminal passes (F6) — it is what makes the *next*
process review auditable.

## Confidence and what would change my mind

High confidence (repo-verifiable): W2/F1 (v4 unreviewed — record states it;
`update()` born in D9 — step-2b record), W3 (Gate 2 provenance), F3 (risk-04
verbatim vs disposition), F4 (source and tests quoted), F2/F7c (gate-yield
from handoff table and commit messages), W6 (design §8 row count).

Medium confidence (mechanism-argued): W1's estimate of what code review would
have caught — assumed from defect character, not measured; a trial period of
per-diff `/code-review` on the remaining steps would settle it, and catching
either an `update()`-class or tree-diff-class defect would move code review up
my ranking. W5 is a labelled hypothesis: implementing v1's window spec in a
scratch branch and watching every existing gate stay green would confirm it;
any gate firing would refute it and restore the brief's "earliness only"
reading, weakening my defense of the loop's cost.

What would change the larger verdict: (a) a future run where pre-draft probes
were done and external round 1 still filed mostly factual-mismatch findings —
would mean probes don't substitute for review rounds and the loop should keep
its current fact-checking role; (b) evidence that the v4-class hole is rare
(several capped runs whose arbitration revisions ship clean) — would demote
recommendation 3 below 4; (c) a measured loop-cost accounting showing the
spawns and driver time rival the implementation's network waste — would revive
the "overengineered loop" reading that, on present evidence, the review was
right to reject and wrong to leave unmeasured.
