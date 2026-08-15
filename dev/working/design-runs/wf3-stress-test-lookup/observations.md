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
