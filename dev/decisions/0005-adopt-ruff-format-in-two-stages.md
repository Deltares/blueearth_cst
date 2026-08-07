ADR 0005 — Adopt `ruff format`, in two stages split on the Snakemake code rerun trigger

Status: proposed
Date: 2026-08-07
Deciders: Ümit Taner
Consulted: —
Supersedes: none
Revisions:
  - 2026-08-07: initial draft. Raised from board item `t2608071212` (R7-23),
    which had been sitting as `backlog` while blocked on this ruling. Churn
    re-measured on `main` at `5bc3d6a`; the previously recorded ratio is
    corrected below.

### Context

`ruff format` is **configured but has never been run**. O-15 adopted ruff as the
lint gate with `select = ["E4","E7","E9","F"]` pinned explicitly; the formatter
was deliberately left out of that adoption, on the grounds that repository-wide
reformatting is a churn decision on its own merits. That was right. What has
happened since is that the decision was never actually taken, so the repository
carries a configured-but-inert tool and every new file is written in whatever
style its author used.

**The churn is real and growing.** Measured on `main` at `5bc3d6a`
(2026-08-07), against the 202 tracked `.py` files ruff lints (`docs/` and
`test_case/` excluded per `pyproject.toml`):

| Scope | Files reformatted |
|---|---|
| **Whole tracked source set** | **169 of 202 (84%)** — 760 hunks, +3,724 / −1,963 lines |
| `tests/` | 95 |
| `blueearth_cst/` | 50 |
| `dev/` | 21 |
| `scripts/` | 3 |

Only **33 files are already formatted.** The trend on absolute counts is 118
files at R7 (2026-07-29) → 136 (2026-08-02) → 169 today.

**One correction to the record.** R7-23 quoted "136 of 276" and "118 of 262",
which read as roughly half the repository. Those denominators come from running
`ruff format --check .` at the repo root, which today reports 402 files — about
200 of them untracked run detritus under `.snakemake/`, `.tmp/` and similar,
already-formatted by accident and padding the denominator. Against the source
that actually exists, the proportion is 84%, not 45%. The absolute file counts
are comparable across the three measurements; **the ratios were not**, and the
problem is larger than the record suggested.

**The real blocker is not the diff size.** It is Snakemake's `code` rerun
trigger. All 50 `blueearth_cst/` files in the reformat set are the
`script:`-invoked layer, and Snakemake hashes the entire script text, comments
included. R10-14 (`t2608071220`) measured the consequence precisely: a
**one-line** edit inside `blueearth_cst/spatial/delineate_region.py` re-ran that
rule, rewrote `region.geojson`, and scheduled all 17 WF1 jobs plus all 25 WF2
jobs from it. Reformatting all 50 modules at once invalidates every rule in all
three workflows — not only for this repository's fixture, but for **every
`project_dir` on disk**, including a user's, the next time it is touched.

So the choice is not "7,700 diff lines, yes or no". It is whether a formatting
commit is allowed to force a full pipeline re-run everywhere, and if not, how to
take the part that cannot.

Doing nothing has a running cost in both directions: the diff grows with every
unformatted commit, and the code-trigger blast radius grows with it.

### Decision

We will adopt `ruff format` as the repository's formatting gate, applied in
**two stages split on whether a file is read by Snakemake's `code` rerun
trigger**.

**Stage 1 — now, unconditionally.** Reformat the 119 files that no Snakemake
rule declares (`tests/` 95, `dev/` 21, `scripts/` 3) in one mechanical commit
with no other change in it. These files are invisible to the rerun trigger, so
this stage costs no re-run anywhere.

**Stage 2 — the 50 `blueearth_cst/` modules, deferred to a moment when a full
pipeline re-run is already being paid for** — a milestone's baseline re-record,
or any run already scheduled with `--forceall`. It lands as its own mechanical
commit, immediately before that run, so the invalidation it causes is absorbed
by work that was going to re-run everything regardless.

Enforcement lands with stage 1: `ruff format --check` joins the existing ruff
invocation in `.github/workflows/ci.yml`, scoped to the tracked source set, and
**stage 2's files are excluded from the check until stage 2 lands** — via an
explicit, dated exclusion in `pyproject.toml` naming this ADR, not a silent
narrowing of the check's path list.

### Consequences

*Positive*

- 119 of 169 unformatted files — 70% of the churn — are cleared immediately at
  zero re-run cost, and no `project_dir` anywhere is invalidated by it.
- The formatter stops being inert. A configured tool that has never run is
  indistinguishable from an oversight, and the next person to read
  `[tool.ruff]` no longer has to work out whether the omission was deliberate.
- New files land formatted from stage 1 onward, so the stage-2 set stops
  growing. Its size becomes fixed at 50 files rather than tracking the repo.
- The stage-2 cost becomes explicit and schedulable instead of being the reason
  the whole decision stalls.

*Negative*

- Two commits rather than one, against R7-23's "single mechanical commit"
  instruction. That instruction's purpose was reviewability, which two
  mechanical commits each satisfy; but the repository carries a half-formatted
  source tree between them, and `git blame` gains a second sweep line.
- A dated exclusion in `pyproject.toml` is debt with a deadline, and deadlines
  in config files are routinely missed. If stage 2 is never scheduled, the
  exclusion becomes permanent and the repository ends in exactly the
  half-adopted state this ADR set out to end. **This is the residual risk.**
- Stage 2 still costs a full invalidation when it lands. This ADR does not
  remove that cost; it only stops paying it twice.

*Neutral*

- **No baseline re-record is caused by formatting itself.** `ruff format` does
  not change semantics, so no artifact's content changes and
  `check_baseline.py` compares equal. What stage 2 causes is a full *re-run*,
  not a changed *result* — R7-23's note conflated the two. The re-run is the
  cost; the baseline is the check that proves nothing moved, and it should be
  run after stage 2 for exactly that reason.
- Import sorting (`I001`, 63 findings) stays out of scope. It is a separate
  rule family, a separate reviewable commit, and adding it here would make this
  ADR two decisions.
- `dev/` and `tests/` dominate stage 1 (116 of 119 files), so the visible diff
  is mostly non-shipping code.

### Alternatives considered

**One commit, everything at once.** Reformat all 169 files together, as R7-23
originally proposed. Simpler history, one sweep line in `git blame`, no
exclusion debt, no half-formatted interval. Not chosen because it forces a full
re-run of all three workflows on every `project_dir` in existence at a moment
nobody chose — the cost lands on whoever next touches a project, not on whoever
takes the decision. It would be preferred if taken *at* a milestone re-record,
where the full re-run is already scheduled: at that moment stage 1 and stage 2
collapse into one commit and this ADR's split buys nothing. **If such a
milestone is imminent, prefer this alternative and revise this ADR.**

**Adopt for new and changed files only.** Format on save or on touch, letting
the tree converge gradually. Rejected: it produces a permanently mixed tree,
makes every subsequent diff a mix of substance and formatting — the specific
thing a mechanical commit exists to prevent — and never reaches a state where
`--check` can be enforced, so the gate can never be armed.

**Do not adopt; remove the configuration.** Delete `ruff format`'s config and
record that the repository does not use a formatter. Rejected: it discards a
tool the repository has already decided it wants (the config was added
deliberately under O-15) and leaves the 84% inconsistency in place with nothing
to converge toward. It would be preferred if the author found the formatter's
output actively worse than the current hand formatting — a judgement this ADR
cannot make for them, and one they should make by reading the diff before
ruling.

### Related

- `dev/tasks/t2608071212-r7-23.md` (R7-23, `blocked`) — the board item this ADR
  exists to unblock; carries the original framing and the earlier measurements.
- `dev/tasks/t2608071220-r10-14.md` (R10-14, watch) — the measured blast radius
  of the `code` rerun trigger: one line, 42 jobs. Stage 2's entire cost model
  rests on it.
- `dev/tasks/t2608071205-r8-1.md` (R8-1) — why a red ruff gate went unnoticed.
  Arming a second ruff check is worth less until that is answered.
- `pyproject.toml` `[tool.ruff]` — the pinned `select`, the exclusions, and
  O-15's reasoning for pinning explicitly.
- `dev/reviews/2026-07-25_post-r6-assessment.md` § O-14 / O-15 — the tooling
  assessment that adopted ruff as the lint gate and deferred the formatter.
