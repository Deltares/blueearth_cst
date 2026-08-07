# Task Brief — Adopt `todo-board` as this repo's single work-tracking scheme

*Written 2026-08-07 from `main` at `31d9451`, for execution in a clean session.*

### Context

Canonical rules: `AGENTS.md`. Board contract: the **`todo-board` skill** (v0.10.0,
`~/workspace/brain/artifacts/skills/todo-board/SKILL.md`); consumer guidance:
`project-system`.

- **This repo never adopted the board.** `dev/tasks/` holds one *closed* campaign
  record and `dev/README.md` defines it as "one brief record per closed tracked
  task" — the **inverse** of the board model, where `dev/tasks/` holds the open
  notes and `done` removes the note into `dev/LOG.md`. There is no `dev/LOG.md`,
  no `dev/drafts/`, no `.base`; `todo-board` is not vendored, though 19 other
  skills are, under `.claude/skills/`.
- In its place the repo grew two hand-synced tiers: `dev/TODO.md` (6 open rows,
  hand-written) and `dev/followups.md` (39 open items). They drifted — a fixed
  item sat listed as `backlog`, and single items carried two IDs
  (`t260802a` = `R8-1`). **The owner wants one scheme, not two.**
- On 2026-08-07 `followups.md` was cut 2,038 → 819 lines: closed items moved to
  `dev/followups-archive.md` (40 entries), and an open-item index was added.
  That archive is the **sealed pre-board ledger** and is an input here, not a
  target.
- **14 code / test / Snakefile sites cite followup IDs** (`[R9-3]`, `[R7-21]`, …).
  Citations of *archived* IDs already point at `followups-archive.md`; citations
  of *open* IDs still resolve into `followups.md`, which this task deletes.
- Roughly half the open items are deliberately **not work** — parked rulings,
  knowingly-accepted debt, upstream defects with local workarounds. They fail
  `todo-board`'s admission gate on purpose, which is exactly why a second file
  existed. Phase 1 gives them a home on the board instead.

### Goal

Make `dev/tasks/` the single place work is tracked in this repo, after extending
`todo-board` with a non-actionable item type so nothing needs a second backlog
file. `dev/followups.md` ceases to exist.

### Non-goals

- Re-triage or re-diagnose the open items. The classification below is an input.
- Editing `dev/followups-archive.md` content — sealed.
- Moving candidate milestones onto the board; `dev/roadmap.md` owns those.
- An Obsidian `.base` view. This repo is not in a vault, so the generated
  `dev/TODO.md` is the view.
- Any change under `blueearth_cst/`, `dev/decisions/`, or `dev/milestones/`.

### Allowed scope

**Permitted** — `dev/tasks/`, `dev/LOG.md` (new), `dev/TODO.md`,
`dev/followups.md` (deletion), `dev/README.md`, `AGENTS.md` (the `dev/` map
only), `.claude/skills/todo-board/` (vendored copy), and citation-only edits in
`tests/`, Snakefiles and `dev/scripts/`.

**Approval-gated** — `~/workspace/brain/artifacts/skills/todo-board/` (Gate 1);
`dev/roadmap.md` (Gate 2); `dev/tasks/2026-07-21_pre-r6-followups.md` (Gate 3).

**Forbidden** — `dev/followups-archive.md` content, `dev/decisions/`,
`dev/milestones/`, everything under `blueearth_cst/`, and any test logic.

### Required changes (checklist)

**Phase 1 — the skill (brain repo; blocking).**

1. Add a second frontmatter type, `type: watch-item`: durable knowledge with no
   action intended. Overview carries **What / Why / Trigger** — the *condition
   that would turn it into work* — replacing Effort. No `queue`, no
   `## Progress`, never returned by `next`. Closes when the trigger fires
   (convert to `todo-item`) or it stops being true (`LOG.md`, `superseded`).
2. Reword the admission gate so it **routes** rather than only excludes:
   durable-but-not-work → watch-item; ephemeral → nothing.
3. Views: state how each view separates the two types. The Obsidian `.base`
   already filters `type == "todo-item"`, so it needs a second view; the
   generated `dev/TODO.md` needs a second table or a Type column.
   `todoboard list` today filters `--area` / `--status` only —
   `<DECIDE: add --type to the CLI, or keep type filtering view-only>`.
4. Bump `version:` and add a `HISTORY.md` entry.

**Phase 2 — this repo.**

5. Vendor `todo-board` into `.claude/skills/todo-board/` by the same pin
   mechanism the other 19 use (`<PLACEHOLDER: the sync command or ADR pin>`).
6. Create `dev/LOG.md`. Resolve the existing closed campaign record per Gate 3.
7. Create one note per open followup, per the classification below. **Each note
   keeps its legacy `R<n>-<n>` ID** in frontmatter `refs:` or the body, so
   `git grep R9-5` keeps landing.
8. Delete `dev/followups.md` and repoint every citation of an open ID at the
   notes — **in the same commit** (see Commit plan).
9. `todoboard render` → `dev/TODO.md`, with the do-not-edit banner. Delete the
   six `t260802*` rows, superseded by notes.
10. Rewrite `dev/README.md`'s `dev/` grammar rows (`tasks/` inverts meaning; add
    `LOG.md`; `followups-archive.md` becomes the pre-board ledger) and the
    `dev/` bullet in `AGENTS.md`'s Repo Map.

### Commit plan

A commit boundary carries a correctness property here: deleting `followups.md`
breaks every citation of an open ID the instant it lands.

| Subject | Paths | Invariant it preserves |
|---|---|---|
| vendor the board CLI | `.claude/skills/todo-board/` | CLI available before any note is written |
| scaffold the board | `dev/LOG.md`, `dev/tasks/` | empty board round-trips `add` → `next` → `done` → `render` |
| migrate the open set | `dev/tasks/*`, `dev/followups.md` (deleted), citation sites | **every followup ID resolves before and after this commit** — the citation rewrite must ride with the deletion, not follow it |
| conventions | `dev/README.md`, `AGENTS.md`, `dev/TODO.md` | docs describe the tree that now exists |

### Validation

Rungs 1, 3 and 4 apply; rung 5 (baseline) does not — this is documentation and
vendored tooling, no rule or `script:` change, so no number can move.

1. **Narrow** (per edit) — `add` → `next` → `done` → `render` round-trip on a
   scratch `--root`; run `render` twice and confirm the second produces an empty
   diff (it is specified idempotent).
2. **Falsifier for "every cited followup ID still resolves"** (once, before the
   migrate commit lands). The property asserts an *absence* — no dangling ID —
   which no test reaches:

   ```bash
   git grep -ohE '\[R(7|8|9|10)-[0-9]+\]' -- ':!dev/tasks' ':!dev/followups-archive.md' | sort -u
   # each must be found by:
   git grep -l "<id>" dev/tasks dev/followups-archive.md
   ```

   **Disproof:** any ID with no home. This exact check was run on 2026-08-07
   after repointing the archived citations and returned clean — reuse it rather
   than re-deriving it.
3. **Full gate** (once, before merge) — `pytest tests/` and
   `pixi run ruff check .`. Baseline at `31d9451`: **1510 passed, 31 skipped,
   1 xfailed**; ruff **All checks passed!**. Any deviation is caused by this
   task and must be explained, not accepted.

Report what each rung *caught*, not merely that it passed.

### Acceptance criteria

- `dev/followups.md` no longer exists; `dev/tasks/` holds the open set as notes;
  `dev/TODO.md` is generated and banner-marked.
- The ID falsifier is green.
- `pytest tests/` and ruff match the `31d9451` figures above.
- `dev/README.md` and `AGENTS.md` describe `dev/tasks/` as the open board.
- **Rollback:** if the ID falsifier cannot be made green, revert the migrate
  commit and keep `followups.md`. The vendoring and scaffolding commits stand on
  their own and need not be reverted with it.

### Output requirements

Final counts: todo-items created, watch-items created, items promoted to
`roadmap.md`, `LOG.md` rows. Name every item whose classification you changed
from the table below, with the reason.

### Task constraints

- **Gate 1 (blocking, before any file is written).** The owner approves or amends
  the `watch-item` design. Phase 2 cannot start first — watch-items would have
  nowhere to go.
- **Gate 2.** The owner confirms which items go to `dev/roadmap.md` rather than
  the board.
- **Gate 3.** The owner decides whether `dev/tasks/2026-07-21_pre-r6-followups.md`
  becomes a `LOG.md` row or keeps its file with a banner.
- Skill edits go to the **canonical brain artifact**, never the vendored copy —
  a vendored pin is overwritten on resync.
- **Verify skill edits in a NEW session.** Instruction, skill and role content is
  snapshotted at session start, so a same-session subagent reads the pre-edit
  version and its report describes guidance that no longer exists.
- Work in a task worktree (`.git-workflow.yml`: `worktree_policy: concurrent`);
  the primary checkout is integration-only.

---

## Appendix — proposed classification of the 39 open items

The rule: **todo-item** = work intended to be done. **watch-item** = true about
the system, no action intended; its Trigger is what would change that.
**roadmap** = a candidate milestone, not a task.

| Type | Items |
|---|---|
| **todo-item** (~20) | R10-12, R10-13, R10-9 (residual), R10-6 (residual), R9-1, R9-2, R9-5, R8-1 (residual), R7-5, R7-8 (residual), R7-14, R7-16 (residual), R7-20, R7-22, R7-23, "Linter for naming conventions", "wf3 batch-size disk-aware default", "per-cst persistence isolation under batching", "Snakemake `code` rerun-trigger does not reach 2.04", "`tee_to_log` traceback capture" |
| **watch-item** (~13) | R10-14, R7-17, "R testthat coverage", "PARKED per-rule progress messages", "Deferred: Linux replication" (one item, not six), "[2026-08-06] CR-7 / F18 pointer", `hydromt to_yml` strips `preprocess` (trigger: upstream fix — an xfail already fires), `weathergenr::write_netcdf` `spatial_ref`, weathergenr wavelet minimum, `sys.modules.setdefault` test pollution, dask cannot be stubbed, "Outlet station naming convention decision" (trigger: an owner ruling) |
| **roadmap** (~4) | R7-18 (climate analysis as a fourth Snakefile), "Climate analysis as a model-independent subworkflow", "Reconsider the WF1 rule arrangement", `<VERIFY: "setup_constant_pars short names → CSDMS Standard Names" — its task t260719a reads as closed in dev/TODO.md; confirm before creating anything>` |

Counts are approximate because two subsections ("Minor open items", "Deferred:
Linux replication") hold bullets that may collapse into one item each. Settle
them at Gate 2; do not silently split or merge.
