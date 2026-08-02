# Task Brief — evaluate and extend a process review of a design-review-loop run

> **Audience.** Two independent reviewers, working separately and without seeing
> each other's output: one Claude/Fable agent, one GPT-5.6 via headless
> `codex exec`. You are not expected to agree with each other or with the
> existing review.
>
> **Delivery.** Artifact. Write one markdown file (path in *Output
> requirements*). Nothing else is expected of you.

### Intended dispatch (for whoever runs this)

Run the two reviewers **independently, in a clean session**, so neither inherits
this milestone's conversational context — the brief is written to be
self-contained precisely so that holds.

**Reviewer A — Fable, at `xhigh` reasoning effort.**

```
Agent(subagent_type="critical-thinker", model="fable", prompt=<this brief>)
```

Effort: `~/.claude/settings.json` already sets `"effortLevel": "xhigh"`, and a
subagent inherits the session effort unless its own definition overrides it.
`.claude/agents/critical-thinker.md` sets only `model:`, so it inherits `xhigh`
and nothing extra is required. Note the plain `Agent` tool exposes `model` but
**not** `effort` — for a genuine per-call override (e.g. `max` for one spawn) use
a Workflow script: `agent(prompt, {model: 'fable', effort: 'xhigh'})`.

**Reviewer B — GPT-5.6 via headless `codex exec`.**

```
codex exec --sandbox read-only --ephemeral -c approval_policy=never -m gpt-5.6-sol \
  - < dev/reviews/2026-07-30_design-loop-efficiency-review-brief.md \
  > <transcript> 2>&1
```

Keep codex **read-only** and land its output file yourself from stdout, rather
than granting it write access for one deliverable. Verify the banner reads
`approval: never` and `sandbox: read-only` before trusting the posture — the
project default is `on-request`, under which a write escalates outside the sandbox
silently. Expect a long stream; run it with a generous timeout or in the
background, and confirm `git status --short` is clean afterwards.

Neither reviewer should see the other's output.

### Context

Canonical repo ruleset: `AGENTS.md` at the repo root. Read it first — it defines
the repository's conventions, hard constraints, and the skill/role vocabulary
used below.

A milestone (Phase 5 / R8, "WF2 v2.0") ran a **`design-review-loop`**: an
adversarial author/reviewer loop that produced an accepted design document, then
an implementation driven step-by-step against it. The loop and the first five
implementation steps are complete. The owner then asked whether the process was
overengineered and whether it could be made faster and cheaper without losing
quality.

The driver (Claude, acting as loop driver and implementer) wrote a self-critical
process review. **Your job is to evaluate that review, find what it missed, and
judge whether the simplifications it proposes are safe.**

Key facts about the subject, so you can calibrate:

- The design document reached **2579 lines** for a workflow whose implementation
  is roughly **1500 lines** of Python plus a Snakefile.
- The loop ran **1 internal lens + 2 external cross-vendor rounds**, produced
  **28 findings across 4 document versions**, hit its round cap unconverged, and
  was closed by **owner arbitration**.
- Implementation was decomposed into **13 sub-steps**, each with a 5-rung
  validation ladder. Six are done.
- The validation fixture is small: 3 climate models, 1 ensemble member, 1 time
  horizon. The baseline manifest pins **7 targets for this workflow, three of
  which are PNG file sizes**.
- Long validation runs are network-bound and slow: one class of remote read
  exceeds a 10-minute tool timeout, so full re-derivations were run as several
  bounded calls.

### Goal

Produce a critique that (a) tests the existing review's conclusions, (b) adds
findings it missed, (c) assesses the **risk** of each proposed simplification,
and (d) recommends changes that **generalize beyond this repository** — i.e.
changes to the reusable skills and to how such a loop is run anywhere, not
repo-specific fixes.

### Non-goals

- Do **not** review the WF2 design's scientific content, its climate methodology,
  or whether its architecture is correct. That was already reviewed to
  convergence. Your subject is the **process**, not the product.
- Do **not** review the implementation code for defects. The question is what
  *process* would have caught them, not what they were.
- Do not propose changes that require a new third-party dependency.
- Do not rewrite the skills yourself. Recommend; do not author.

### Allowed scope

**Read (all read-only; you may skim anything they cite):**

| Path | What it is |
|---|---|
| `dev/reviews/2026-07-30_wf2-v2-process-review.md` | **The review you are evaluating.** Start here |
| `dev/workflows/wf2-climate-analysis-v2-design.md` | The accepted design (2579 lines) — sample it; do not read end to end |
| `dev/workflows/wf2-climate-analysis-v2-design-review-record.md` | The loop's audit trail: verdicts, all 28 findings, owner rulings, arbitration |
| `dev/r08/2026-07-29_wf2-step1-validation.md` | Validation record, step 1 |
| `dev/r08/2026-07-30_wf2-step2b-validation.md` | Validation record, step 2b — includes the `update()` finding |
| `.claude/skills/design-review-loop/` | The skill under evaluation (SKILL.md + `references/`) |
| `~/.claude/skills/task-brief/`, `~/.claude/skills/design-document/` | The two skills it hands off to |
| `git log --oneline` on `main` | The implementation commit sequence; messages record what each gate found |

**Write:** exactly one file, at the path in *Output requirements*. Nothing else.

**Forbidden:** any change to source, config, tests, the design, the skills, or
the fixture under `test_case/`.

### Required analysis (checklist)

Address each. Where you disagree with the existing review, say so plainly and
give your reasoning — a critique that only amplifies it is worth little.

1. **Test the headline claim.** The review concludes that review effort went
   almost entirely to the artifact that could not be executed (the design), that
   the code got no gate, and that all eight defects reaching a running system
   were code-shaped. Is that the right diagnosis? Is "add a code-review gate" the
   right remedy, or does it treat a symptom?

2. **Test a claim the review makes in its own favour, which may not hold.** The
   review asserts the loop "paid for itself" because round 1 found a defect where
   the proposed reference period overran the source data by six years. **Consider
   the counterfactual:** that defect would plausibly have surfaced on the first
   real run regardless. If so, the loop bought *earliness*, not *detection* — and
   the review's main justification for the loop's cost weakens. There is a
   circularity worth examining: the defect's expensive consequence was that a
   wrong claim propagated into four documents, but those documents existed only
   because of the loop. Assess this honestly; the driver flagged it but did not
   resolve it.

3. **Find what the review missed.** It is self-critical but it is also written by
   the party being reviewed. Likely blind spots to probe: what it chose *not* to
   measure; costs it did not attribute; whether its proposed remedies are the
   cheapest available; whether any of its "lessons" are really one-off incidents;
   whether the loop's *structure* (rather than its tuning) is the problem.

4. **Assess the risk of each proposed simplification.** For each, state the
   failure mode it introduces, how it would present, and whether you would accept
   it. Candidate risks are listed below to make the exercise concrete — treat them
   as a floor, not a ceiling, and add your own:

   | Simplification | Candidate risk to evaluate |
   |---|---|
   | Batch the value-neutral structural steps into fewer commits | A failing diff can no longer be attributed to a cause; bisection cost rises |
   | Narrow cache invalidation from file hashing to function-bytecode hashing | It may now **miss** a real behaviour change — a changed unlisted callee, a library upgrade, or data-dependent behaviour. This weakens the guard that existed to prevent silently-stale cached results |
   | Add a small fast config (1 model, 1 scenario) for iteration | Validating on a reduced ensemble may hide multi-model, multi-member and ragged-availability bugs |
   | Replace a further design round with a code-review gate | Code review may catch different defects but miss architectural ones; it also arrives later |
   | Compress the design document at seal | Loss of rationale invites re-litigating settled decisions |
   | Split one pipeline stage into fetch + reduce with a raw cache | More moving parts and a second cache to keep coherent — a new desync class |

5. **Judge whether the loop's shape is right, not just its size.** Alternatives
   worth weighing: fewer design rounds paired with code gates; a
   prototype/spike *before* the design so the design is written against measured
   reality; design-then-implement in one pass with review only at the end; or
   keeping the loop but bounding its output length by contract. Recommend one and
   say what evidence would change your mind.

6. **Generalize.** Give concrete, reusable recommendations aimed at the skills
   (`design-review-loop`, `task-brief`, `design-document`) and at running such a
   loop in any repository. For each: the gap, the proposed rule or checklist item,
   why it generalizes, and where it belongs. Prefer few strong recommendations
   over many weak ones.

7. **Rank by expected value.** Order your recommendations by (benefit ÷ cost) and
   say which single one you would implement first if only one were possible.

### Validation

There is no test suite for this task. Instead:

- Every claim about what happened must cite a **file path, a commit, or a
  quoted line**. The repo is the evidence; do not rely on the review's summary of
  itself.
- Where you assert a cost or a benefit, say whether it is **measured, estimated,
  or assumed**, and label it as such.
- If you cannot substantiate a claim, mark it explicitly as a hypothesis with the
  observation that would settle it.

### Acceptance criteria

- Each of the 7 required-analysis items is addressed.
- At least **three substantive findings the existing review does not contain**.
- At least one place where you **disagree** with the existing review, argued from
  evidence. If you genuinely find none, say so explicitly and explain why — but
  treat that as a surprising outcome worth justifying.
- Every simplification in the table has a risk verdict: *accept*, *accept with
  mitigation* (named), or *reject* (with the alternative).
- Recommendations are generalized: none depends on this repository's specifics.
- No recommendation requires a new dependency.

### Output requirements

Write one markdown file:

- Fable/Claude reviewer → `dev/reviews/2026-07-30_process-review-critique-fable.md`
- GPT reviewer → `dev/reviews/2026-07-30_process-review-critique-gpt.md`

Structure:

```
# Critique — <reviewer id>

## Verdict
One paragraph: is the existing review's diagnosis right, partly right, or wrong?

## Where the existing review is wrong or unsupported
## Findings it missed
## Risk assessment of the proposed simplifications
   (one row/section per simplification, with accept / accept-with-mitigation / reject)
## Is the loop's shape right?
## Generalized recommendations, ranked by expected value
## Confidence and what would change my mind
```

Be concrete and terse. Prefer naming a mechanism over describing a principle.

### Task constraints

- **Read-only on everything except your one output file.**
- Do not coordinate with the other reviewer; independence is the point of using
  two.
- Do not defer to the existing review's framing because it is self-critical.
  Self-criticism can still be self-serving — it can concede small errors to
  protect a large one.
- Do not pad. A short critique with three real findings beats a long one that
  restates the source.
- No nested delegation: do the analysis yourself.
