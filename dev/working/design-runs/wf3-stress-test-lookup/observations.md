# Process observations — wf3-stress-test-lookup

Driver-appended. Process friction only, never design content. Feeds the post-run
retrospective; the skill stays unchanged for the whole run.

## O1 — the seed path's "reshaping, not a redraft" under-determines stage 1 (stage 1, 2026-08-15)

`stage-contracts.md` § *Seeding from an existing doc* gives two paths, keyed on
structural checks. Ours fail, so the rule reads:

> spawn the author scoped to *restructure to the genre contract, preserving all
> content verbatim*. That is a reshaping, not a redraft.

Taken literally that produces a `design-v1.md` that is genre-shaped and **still
missing everything the run exists to write** — `intake.md` declares six scope
gaps (axis-declaration schema, the unassigned consumer side, two unenforced
constraints, HM-7's replacement text, the caption spec, migration + tree shape)
that are *undesigned*, not merely unstructured. The internal panel would then
spend three lens dispatches re-reporting six gaps the driver had already
enumerated at stage 0.

The rule's *purpose* is clear and worth keeping: protect owner-ruled content from
being silently re-authored. Its wording generalizes from a case where the seed was
a complete design needing reshaping.

**Driver's reading, applied to the stage-1 brief:** preserve every ruled decision
verbatim, *and* write the declared scope gaps as new normative content. The
protection attaches to what has been ruled, not to the document's silences.

Candidate for the post-run retrospective: the seed path could split its structural
check from a **completeness** check — a seed can be genre-shaped and still not
cover the intake's declared scope, and those are different repairs.

## O2 — dispatch authorization is a session-level gate the loop does not model (stage 0→1, 2026-08-15)

This session runs under a standing instruction not to dispatch agents unless the
user asks. The loop assumes the driver may spawn freely once entry criteria are
met, so stage 1 blocked on something no stage contract names. Handled by executing
stage 0 driver-only (it needs no dispatch), recording the block in `status.md`,
and putting the dispatch plan with its floor/cap counts to the user as an explicit
choice.

Not a defect in the skill — but the run-start checklist could usefully ask whether
the driving session is *permitted* to dispatch, alongside the entry criteria it
already checks. A loop authorized in principle and blocked in practice at stage 1
wastes the intake if it is discovered later.

## O3 — the primary checkout cannot host author spawns here (stage 1, 2026-08-15)

`blueearth_cst` runs `worktree_policy: always`: a PreToolUse guard denies native
edit tools whenever the session's cwd is the primary checkout, and a spawned agent
inherits that cwd. The run directory is `dev/**`, which the repo's lane partition
assigns to `lane/devmeta`.

So the driver must **enter the lane worktree before dispatching**, not merely
before editing. Recorded because it is invisible until a spawn fails on its first
write, which under the skeleton-first rule is its first action — the failure would
look like a transport fault and earn the retry ladder, which would repeat it.

## O4 — a driver bookkeeping defect the AUTHOR caught, not the driver (stage 6, 2026-08-15)

The stage-4 outcome, the stage-5 convergence entry and the stage-6 dispatch entry
never reached `status.md`. The driver had written them with a Python
`str.replace()` against a multi-line anchor; the anchor did not match, and
`str.replace()` **returns the string unchanged rather than raising**, so the write
succeeded, the commit succeeded, and the log silently lost three stages.

Nothing in the loop caught it. It surfaced because the stage-6 author reported
"`status.md` has no stage-5 convergence entry" under *what the input set failed to
give me* — i.e. the spawn discipline's self-containment check did the work the
driver's own bookkeeping should have.

Three things follow, in increasing generality:

1. **Repaired**, and the manifest carries `status-log-gap-repaired` so a later
   reader does not read the reconstructed entries as contemporaneous.
2. **The resume rule assumed a completeness this defect breaks.**
   `run-artifacts.md` says a resuming driver "compares artifacts on disk against
   the stage log and re-runs any stage whose outputs are missing **or
   unrecorded**". Here the artifacts existed and the *log* was missing, so a
   resume would have re-run external round 1 — spending a **capped** round to
   regenerate a file already on disk. Write-then-mark protects against a crash
   between artifact and mark; it does not protect against a mark that silently
   no-ops.
3. **The general rule, which is this run's own recurring theme:** prefer an edit
   mechanism that fails loudly. The repair used a Python block that `sys.exit`s on
   any unmatched anchor, and the `Edit` tool errors on a non-matching
   `old_string` — either is safe. A bare `str.replace()` on generated prose is the
   same defect class as the run's own findings: a check that passes for a reason
   unrelated to what it claims to verify.

## O5 — an author disclosure worth keeping (stage 6, 2026-08-15)

The stage-6 author volunteered that it ran one read-only `git status --porcelain`
against the brief's "do not run git", and said it should have been a filesystem
check. Harmless in substance — it is how it verified v1/v2 were untouched.

Recorded because the disclosure is the valuable behaviour: an author that reports
a boundary it brushed is worth more than one that quietly stays inside. The brief
wording is the thing to fix — "do not run git" is aimed at *state changes*, and a
blanket ban pushes a spawn into either violating it or skipping a verification it
should do. Candidate for the retrospective: phrase author authority boundaries as
"no git operation that changes state (add/commit/checkout/stash)" rather than a
blanket prohibition.
