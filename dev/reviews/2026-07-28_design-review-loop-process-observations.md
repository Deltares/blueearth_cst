# Process observations — r07-project-layout

Process friction only, never design content. Appended by the driver as it
appears; feeds the post-run retrospective. The skill stays unchanged for the
whole run.

## 2026-07-27 — stage 0/1: run seeded from a pre-existing interactive draft

**Deviation, logged.** The skill's stage table has no contract for a run that
begins from a design doc that already exists. `dev/milestones/r07/project-layout-design.md`
was authored interactively with the owner on 2026-07-26 — 414 lines, sixteen
recorded rulings, two logged reversals, one superseded principle — and the
change request is to put *that document* through the loop, not to have an
author re-derive it.

Handling: the driver ran the stage-1 structural checks against the existing
doc **before** deciding. It declares `Genre: decision-record` itself and carries
a non-empty `## Alternatives considered` (8 alternatives, each with a rejection
or withdrawal rationale). Both hard checks pass, so stage 1 was executed as a
**mechanical copy** to `design-v1.md` with no author spawn — a spawn would have
re-authored owner-ruled content, destroying provenance the loop exists to
protect.

Skill-improvement candidate (do not act during the run): stage 1 could name a
"seed" path — when an existing doc passes the structural checks, copy it as v1
and log the deviation; when it fails them, spawn the author scoped to
*restructure to the genre contract preserving all content verbatim*, which is
not a redraft.

## 2026-07-27 — stage 0: derived artifacts declared non-goals up front

Three artifacts derive from this design and would go stale on any revision
(task brief, migration path map, roadmap status line). The skill's stage table
says nothing about derived artifacts, and the p33 precedent (v1 -> v4) shows how
far a design can move. Declared as run non-goals in `intake.md` with named
post-G2 regeneration actions, and author spawns are barred from touching them.

Skill-improvement candidate: stage 0's intake contract could require an explicit
**derived-artifact register** — what depends on this design, and what regenerates
it after G2 — so the review payload stays one document and stale downstream
files are not silently carried forward.

## 2026-07-27 — stage 2: G1 ruling record supplied to the internal panel

**Deviation, logged.** The stage table gives the panel `design-v1.md` + the
verdict contract, "no chat context, no ledger". The G1 ruling record is neither,
but it is also not named as an input — an ambiguity in the contract.

Handling: the panel dispatch includes the three G1 rulings as *settled framing*,
with an explicit instruction not to spend findings re-litigating them, but to
flag any downstream inconsistency a ruling creates in the doc. Rationale is the
p33 precedent: there, v1 predated a G1 ruling, the panel spent a **blocking**
finding (arch-1) plus two majors on the resulting gate-ruling inversion, and the
whole group resolved to "apply the ruling the owner already made". That is a
wasted round, not a review.

The clean-room intent of "no chat context, no ledger" is isolation from *other
reviewers*, not from the *owner's gate decisions* — which the author input set
includes explicitly ("the gate record from `status.md`").

Skill-improvement candidate: stage 2's input column should name the G1 gate
record as a panel input, stated as settled framing. Two runs have now hit the
same edge.

## 2026-07-27 — stage 4 preflight: codex sandbox enforced; stdin recipe is shell-specific

Fail-closed permission preflight passed before any external dispatch. Banner
confirmed `approval: never` and `sandbox: read-only` (codex-cli 0.145.0, model
`gpt-5.6-sol`, ChatGPT-subscription auth). The `-c approval_policy=never` form
is the effective flag, as the codex adapter states.

Friction: the adapter's empty-stdin recipe `'' | codex exec ...` is written for
PowerShell. Run through the Bash tool it emits
`/usr/bin/bash: line 1: : command not found` — bash parses `''` as a command
name, not as an empty string to pipe. It happened to work (the failed command
still closed stdin), but the correct Bash form is `codex exec ... < /dev/null`.

Skill-improvement candidate (`workflow-driver`, codex adapter): label the
invocation recipe as PowerShell and give the Bash equivalent `< /dev/null`
beside it. A Windows repo driven from Claude Code can dispatch through either
shell.

## 2026-07-28 — stage 2: session died mid-panel; all three lenses lost

The 2026-07-27 session ended while the three lens reviewers were in flight. Each
had written its skeleton-first placeholder (`verdict: revise`, `findings: []`,
"review in progress") and nothing more. Resume reconciliation per
`run-artifacts.md`: stage 2 was `[open]` in the stage log and its outputs are
present-but-empty, so the stage re-runs — all three lenses re-dispatched on
`design-v1.md`, overwriting the stubs. No index existed, so nothing downstream
consumed the empty verdicts.

Friction: the skeleton-first discipline (`stage-contracts.md` § Author input
set) is written for **author** spawns and mandates a resumable partial. For a
**reviewer** spawn the same discipline produces a file that is syntactically a
valid verdict — `verdict: revise`, `findings: []` — and therefore indistinguishable
on disk from a genuine "revise with no findings" review. A driver reconciling on
`status.md` alone would have marked stage 2 done and carried three empty
verdicts into the ledger. What saved it here was reading the files, not the
protocol.

Skill-improvement candidate: reviewer placeholders should be **schema-invalid on
purpose** — e.g. `verdict: IN_PROGRESS` — so a partial review can never be
mistaken for a completed one, and the driver's structural checks should reject
any verdict outside the `approve|revise|reject` enum. Alternatively, bar the
skeleton-first rule for reviewer spawns entirely (a lost review costs one
re-dispatch; a silently-empty one corrupts the ledger).

The `IN_PROGRESS` placeholder was mandated in the re-dispatch briefs and all
three lenses honoured it, replacing it with a real verdict on completion. It
cost nothing and closed the hole for the rest of the run.

## 2026-07-28 — stage 2 → G1: the panel forced a framing return, and the skill
handles this only implicitly

The panel's strongest result was a **three-lens convergence on one blocking
finding** (B1's unnamed producer: risk-1, arch-1, repo-2) whose three proposed
resolutions differed in *milestone scope* — widen, narrow, or preserve. The
skill's framing-change rule ("a material change to problem, scope, constraints,
or the selected alternative — at any stage — returns the run to G1") clearly
applies, but it is written as though the *author* discovers the material change
during revision. Here the driver could see it at index time, before spending an
author dispatch, because the three lenses had already enumerated the fork.

Handling: the driver put the fork to the owner as a G1 return **before** stage 3,
with each option's scope effect stated, and recorded the ruling in a
`G1-return ruling record` table in `status.md` mirroring the original G1 table.
The author brief then carries the ruling as settled rather than as an option.
This is strictly cheaper than the alternative — letting the author pick, then
discovering at G2 that the pick changed scope and bouncing the run back through
G1 with a spent revision.

Skill-improvement candidate: stage 5's convergence check has an explicit
"material framing change → return to G1" exit, but stage 2 has none. The stage
table could give the **driver** an explicit gate-return trigger at index time:
when the panel's findings admit resolutions that differ in scope, constraints,
or the selected alternative, return to G1 *before* dispatching the revision.
Two artifacts make this cheap — the index already groups convergent findings,
and the proposed fixes are already in the verdict schema's `suggested_fix`.

## 2026-07-28 — stage 2: lens disagreement is signal, and the index needs a home for it

The architecture and risk lenses reached **opposite conclusions on the same
checkable question** (whether B1's bbox change perturbs values): arch-3 filed it
`major`; the risk lens checked it, found `buffer=1` and a recorded empirical
`allclose` closure, and deliberately declined to raise it — documenting the
non-finding in its notes. There were also three severity divergences across
convergent groups (D, G, H, I).

The skill's aggregation rule says the index "may group duplicate findings;
preserves every original ID, severity, and text by reference; never deletes or
re-grades". That covers *duplicates* but says nothing about *contradictions*,
which are the more valuable output — a disagreement between two independent
lenses is precisely the place a design is under-determined. The driver added a
`## Conflicts between lenses` section recording both readings and, where one
existed, the cheap empirical test that would settle it. The owner then
adjudicated the substantive conflict at the G1 return rather than leaving it to
the author.

Skill-improvement candidate: `run-artifacts.md`'s index row should require a
conflicts section — cross-lens contradictions and severity divergences recorded
with both readings intact and neither resolved by the driver. Note the
"never re-grade" rule already implies the severity half; the factual half is
unaddressed.

## 2026-07-28 — stage 4 dispatch: the immutable brief went stale mid-run

**Deviation, logged.** `review-brief.md` is instantiated once and immutable for
the run. By the time round 1 dispatched, two of its statements were false: its
settled-framing block listed only the four 2026-07-27 rulings (four more had
since been made across two G1 returns), and its task paragraph said the design
"plans the work as 13 commits" (now 15, owner-ruled).

Dispatching it unchanged would have invited exactly the waste the p33 precedent
records — an external round spent re-litigating decisions the owner already made,
with the round cap at 2.

Handling: the brief stayed byte-unchanged as the run's contract; the dispatch
prompt carried the contract verbatim **plus** a labelled addendum with the newer
rulings, and the stale commit count was generalized rather than restated
("a sequence of commits"). This follows the precedent already set at stage 2 in
this run, where the G1 rulings reached the panel through the dispatch rather than
by editing an artifact. The reasoning: immutability protects the *review
contract* — role, lenses, evidence burden, output schema — so rounds stay
comparable; the settled-framing block is run state, which changes by design every
time a gate rules.

Skill-improvement candidate: `run-artifacts.md` describes `review-brief.md` as
"immutable for the run" without distinguishing its two halves. It could separate
the **contract** (immutable, the thing that makes rounds comparable) from a
**settled-framing block** explicitly refreshed at each dispatch from the gate
records in `status.md`. As written, a run with mid-loop gate rulings must either
dispatch a brief it knows is false or violate a stated invariant. Two stages in
this run hit the same edge.

## 2026-07-28 — arbitration: relaying both sides was not enough; the driver had
to supply a third fact

At the round cap the skill's failure-mode row says to "present surviving findings
+ both sides' rationale from the ledger". Followed literally, the owner would have
ruled on ext2-01 (blocking) with only the reviewer's framing available — that the
no-inputs producer creates a silent-staleness path — because the author had not
addressed catalog freshness at all and so had no counter-rationale in the ledger.
There was no "both sides" to present.

The driver checked the repository before putting it to the owner and found the
decisive fact neither party had: today's rule 3.02 already carries the catalog as
`params: data_sources = DATA_SOURCES`, not as a declared input, so the staleness
path is **pre-existing** and R07 is parity on it. That converted the question from
"does R07 introduce a silent-staleness bug" (which reads as an obvious block) into
"should R07 fix a long-standing gap it does not worsen" (a cost question the owner
can weigh). The owner still ruled accept-and-fix, but on accurate grounds, and
attached a verification the finding itself did not call for — that the catalog set
really is identical in both DAGs, since the symmetric-input fix silently depends
on it.

The general point: at the cap, findings are by construction the ones where author
and reviewer disagreed or talked past each other, so the ledger is the *least*
likely place for the deciding fact to already exist. A driver that only relays is
laundering a stalemate into an owner decision.

Skill-improvement candidate: the arbitration row could direct the driver to
**independently verify each surviving finding's factual premise against the
target repo before presenting it**, and to present that verification as a third
input beside the two rationales — explicitly noting where a finding describes a
pre-existing condition rather than a regression introduced by the design. Cheap
(one or two greps here) and it changed the character of a blocking ruling. Note
this does not breach the "driver never writes design content" rule: checking a
factual premise is not authoring, and the ruling stays the owner's.
