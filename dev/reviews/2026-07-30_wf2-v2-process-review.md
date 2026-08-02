# Process review — WF2 v2.0 (Phase 5 / R8), mid-milestone

```
Date:    2026-07-30
Scope:   the design-review-loop run `wf2-climate-analysis-v2` and migration
         steps 1 through 4c
Trigger: owner question -- "is there a more efficient way? did we overengineer
         the design workflow loop?"
Status:  mid-milestone; steps 4d, 5a-5f, 6a-6c, 7 and the seal remain
```

> **SUPERSEDED IN PART.** Two independent critiques (Fable, GPT-5.6) refuted this
> document's headline. Sections **1, 3 and 5** are superseded by
> `2026-07-30_wf2-v2-process-review-r2.md` — read that first: §1 there is the
> errata list, and §2 records that the `kernel_hash` fix credited in §2.1 below is
> **unsafe as landed**. Sections 2, 4 and 6 here stand, with those corrections.

Written while the evidence is fresh rather than at the seal, because three of the
findings change how the *remaining* steps should be run.

---

## 1. The headline finding

**Review effort was spent almost entirely on the artifact that could not be
executed, and almost none on the artifact that could.**

The design received three review rounds, 28 findings, four versions, two model
families and owner arbitration. The implementation received **no review at all**
before being run.

Every defect that reached a running system was a code defect, in a process with
three design gates and zero code gates:

| Defect | Caught by | Would a design review have caught it? |
|---|---|---|
| `NameError` on every cmip6 run (params read after use) | re-reading the file | No |
| `update()` missing → the entire cache silently never fired | a forced-rerun experiment | No |
| Sanitized-wildcard round-trip, at step 3 | dry-run | No |
| The *same* bug again at 4b, in a different rule | dry-run | No |
| "Identical" region bounds that were rounded to 6 dp | Gate 2, after propagating to 4 docs | No |
| Every rename target treated as certified, defeating A3 | a unit test | No |
| `crawled_on` guard firing on every correct run (date vs str) | first real dry-run | No |
| Merged timeseries inheriting one series' identity attrs | `semantic_tree_diff` | No |

Eight defects, none design-shaped. Meanwhile the design review found four real
defects — the 2014 reference window, the daily/monthly reducer mismatch, the
DAG-incompatible failure handling, and the job arithmetic — so it *did* pay. The
error was not running it; it was **not pairing it with a code gate**.

**Action for the remaining steps:** run `/code-review` on each diff before running
the workflow. It is cheap, and it is aimed where the defects actually are.

---

## 2. Efficiency: where the time and tokens went

### 2.1 The dominant cost was network re-derivation, and it was self-inflicted

Any edit to an enumerated reducer *file* invalidated all 9 series, forcing a full
re-download. Because three `ssp585` reads exceed the 10-minute tool timeout, each
re-derivation became 4-6 bounded tool calls of several minutes each.

Rough count for this session: **~8 full or partial re-derivations**, most of them
triggered by edits that changed no arithmetic at all. Step 4c's trigger was an
error-message change.

Fixed during this review:

- `kernel_hash` — invalidation now tracks bytecode, constants and name lookups
  rather than file bytes, so comments, docstrings and error strings are free.
- `snake_config_dev_fast.yml` — 2 series instead of 9 for iteration.

Still to do, and the biggest remaining win:

- **Split stage A into fetch → reduce with a raw local cache.** The design
  conflated the two, so anything invalidating the *reduction* re-triggers the
  *download*. With the split, steps 5a (weighting), 5b (calendar), 5c (rounding)
  and 6b (dry-month rule) all re-reduce from local disk in seconds. At `Amon` over
  a buffered basin bbox the raw slice is single-digit MB.

**Estimate:** those three changes together take the remaining ~6 value-changing
steps from ~15 minutes of network each to well under a minute each.

### 2.2 The `ssp585` slowness was never diagnosed

Three parallel `ssp585` reads exceed 10 minutes; three historical + three `ssp245`
finish in ~6. This asymmetry shaped every run strategy in the session and cost
several killed background jobs — and it was worked *around*, never investigated. A
single diagnosis (chunking? store size? one slow model?) would likely have been
cheaper than the accumulated workarounds.

**Lesson:** when an unexplained performance asymmetry starts dictating process,
spend one probe on the cause before building process around the symptom.

### 2.3 Background jobs were the wrong tool for long runs

Three background runs were killed mid-write. Two left the fixture with a
manifest-pinned target half-written; one left an OS handle on a netCDF that is
**still held**, blocking the fixture as this is written.

Foreground calls with an explicit long timeout, or bounded per-target calls, were
reliable. Background was not.

**Lesson:** for a run that mutates a validated fixture, prefer bounded foreground
calls. A killed background job does not just lose time — it can leave the
validation baseline unverifiable.

### 2.4 Token spend concentrated in three avoidable places

1. **Re-reading large generated artifacts.** The design grew to 2579 lines and was
   grepped repeatedly. Section-targeted `sed`/`awk` reads were much cheaper than
   `Read` on the whole file — that habit arrived late.
2. **Heredoc escape failures.** `\n` inside `<<'PY'` heredocs was mangled **three
   separate times**, each costing a diagnose-and-retry cycle. The Write/Edit tools
   handle escapes correctly; heredocs should be reserved for escape-free content.
3. **Verbose command output.** Piping through a targeted `grep` was adopted
   partway through and should have been the default from the first snakemake call.

---

## 3. Did we overengineer the design loop?

**The loop: proportionate. Its output: not.**

Round 1 found the 2014 reference-window defect independently in both reviewers —
a defect that would have produced silently wrong change factors, which is exactly
the class of thing this repo cannot afford. Round 2 found three more blocking
issues. On evidence, both rounds earned their cost.

What was disproportionate:

- **A 2579-line design for a ~1500-line workflow.** It grew because each round
  *added* rather than replaced: 15 open questions, 12 decisions, 10 named risks,
  6 extension slots, plus alternatives for each. Future readers will read the
  design instead of the code, and it will drift.
- **13 migration sub-steps.** The design specified 5a-5e and 6a-6c; the driver
  then split step 4 into four parts on its own judgement. Per-cause attribution
  genuinely matters for 5a/5b/5c — weighting, calendar and rounding must be
  separable or a diff cannot be explained. It matters much less for structural
  steps whose only claim is "no number moved", and each split cost a full
  re-derivation.
- **A five-rung validation ladder against a fixture that can barely resolve
  anything.** `check_baseline` pins 7 WF2 targets, three of which are *PNG file
  sizes*; the seed has one member and one horizon. The ladder was run per
  sub-step regardless.

**Actions taken:** the remaining structural steps are batched; 5a/5b/5c stay
strictly separate; the design gets compressed to ~600 lines at the seal, with the
audit trail left in the review record where it already lives.

---

## 4. What verification actually earned its keep

Ranked by defects caught per unit cost:

1. **`semantic_tree_diff` against a reference tree** — caught the merged-timeseries
   identity-attribute defect that `check_baseline` and the whole test suite missed.
   Its one weakness: it did not exist for steps 1-4a because no snapshot had been
   taken. **Take the snapshot at the start of a milestone, not when first needed.**
2. **Targeted negative tests against real data** — injecting a bogus model and an
   unpublished scenario proved the config-error/normal-skip split in one call.
   Far more informative per token than another fixture-based unit test.
3. **Deliberate falsification experiments** — the `--forcerun` before/after that
   exposed `update()`. This was the single highest-value verification of the
   session, and it was ad hoc rather than planned.
4. **`check_baseline`** — necessary but weak: it passed *trivially* on every run
   until the workflow was actually re-executed, and its WF2 coverage excludes all
   monthly intermediates.
5. **Dry-runs** — cheap and caught real wiring errors, but they never execute
   script bodies, so they gave false confidence twice (the `NameError`; the
   missing `update()`).

**Lesson:** design one falsification experiment per claimed property, up front. "9
cache_hit / 0 deriving" and "0 reduce jobs on a horizon change" are worth more
than any number of green unit tests, and both were improvised.

---

## 5. Skill improvement candidates

Raised here; each meets the cross-project bar in `AGENTS.md`.

### 5.1 `design-review-loop` — pair the design gate with a code gate

**Gap.** The skill runs adversarial review on a design and then hands off to
`task-brief`, with no equivalent gate on the implementation. This run produced
zero design-shaped defects in code and eight code-shaped ones.

**Proposed change.** Add a stage-7 handoff requirement: the accepted design's task
brief must name a code-review gate per commit (`/code-review`, `code-reviewer`, or
the repo's equivalent), and the loop's Output section should state that design
convergence is not implementation assurance.

### 5.2 `design-review-loop` — cap document growth

**Gap.** Nothing in the loop bounds the design's size, and each round adds. A
2579-line design for a 1500-line change inverts the reading order.

**Proposed change.** At stage 7 (finalize), require a compression pass: the landed
design keeps architecture, rulings, decisions and consequences; superseded
alternatives and per-round argumentation move to the consolidated review record,
which already exists for exactly this.

### 5.3 `design-review-loop` — snapshot the reference tree at run start

**Gap.** `semantic_tree_diff`-style gates need a pre-change tree. This run
discovered that four steps in, and steps 1-4a can never have that gate
retrospectively.

**Proposed change.** Add to the stage-0 intake checklist: if the target repo has a
tree-diff gate, snapshot the reference **before** the first implementation commit,
and record its path in the derived-artifact register.

### 5.4 `task-brief` — require a falsification experiment per claimed property

**Gap.** The validation ladder is a rung list (narrow → new tests → integration →
full → baseline). It does not ask for an experiment that would *disprove* a
claimed property, which is what actually found the two biggest defects here.

**Proposed change.** Add to the Validation section guidance: for each property the
change claims ("no network on re-run", "zero jobs on a horizon change"), name the
observation that would falsify it and how to produce it.

### 5.5 `snakemake` — record the output-removal / `update()` interaction

**Gap.** `Job.prepare()` removes existing outputs before every job, so any
"revalidate an existing output and skip" pattern is silently a no-op without the
`update()` flag. This cost a blocking-class defect that no unit test could catch.

**Proposed change.** Add to the skill's mechanics notes: persistent or
self-validating outputs must be flagged `update(...)`; cite that Snakemake removes
outputs before job execution and that the failure mode is a cache that never
fires rather than an error.

### 5.6 Global / `AGENTS.md` — heredoc escapes

**Gap.** `\n` inside a quoted heredoc was mangled three times in one session.

**Proposed change.** A one-line rule in the operating-environment section: use
Write/Edit for content containing escape sequences; reserve heredocs for
escape-free text.

---

## 6. Carry-forwards already recorded elsewhere

- `semantic_tree_diff` needs a fresh reference snapshot per value-changing step
  (`dev/r08/2026-07-30_wf2-step2b-validation.md`).
- `update()` must be preserved across output renames (design D9's implementation
  note).
- Long runs: bounded per-target foreground calls, **targets before the flags** —
  `--configfile` takes multiple values and will silently swallow a target path.
- `--rerun-incomplete` is the recovery when a kill leaves a pinned target
  half-written.

## 7. Open blocker at time of writing

A killed background job still holds an OS handle on
`test_case/test_local/.../gcm_timeseries.nc`, so the fixture is mid-write and
`check_baseline` / `semantic_tree_diff` cannot run. Clearing it needs a process
kill or a session restart. Step 4c's fixture gates are therefore **outstanding**,
and this is stated in that commit rather than implied.
