# dev/

Development process and project record. Everything here supports building the
project; none of it ships. The lifecycle that fills these folders is the
`project-system` skill's workflow.

## Layout

Organised by **how long a thing stays true**, because that decides both where
it goes and when it may be deleted.

**Happening now** — small, churns constantly:

| Path | Holds |
|---|---|
| `TODO.md` | Live task board — unfinished work only (`backlog` / `active` / `blocked`) |
| `working/` | Working & handoff notes for **live** work; drained at closure, but see the promotion rule below |

**Stays true** — consulted while working, rewritten rarely and deliberately:

| Path | Holds |
|---|---|
| `roadmap.md` | The phase narrative: what each milestone set out to do and how it landed |
| `followups.md` | Milestone-scoped backlog with reproducible context; cited by live tests. Detail store behind `TODO.md` |
| `reference/` | The rules: `naming.md`, `agent-activation.md`, `git-conventions.md`, `contracts/`, `workflows/` — see its `README.md` |

**Happened** — records of what was done, kept by identity:

| Path | Holds |
|---|---|
| `decisions/` | ADRs — **permanent**. Superseded ones stay with a pointer; evidence in a `<adr-slug>/` sibling folder |
| `milestones/` | Every milestone's design / plan / review / evidence docs — see its own `README.md` for the index |
| `tasks/` | One brief record per closed tracked task |

**Decays** — snapshots of a system that keeps moving:

| Path | Holds |
|---|---|
| `reviews/` | Process reviews and post-milestone self-check registers. **Prunable** — see the retention rule below |

**Pinned by code** — these paths are constructed in Python, so moving them is a
code change, not a documentation change:

| Path | Holds |
|---|---|
| `scripts/` | Developer scripts — inspect or maintain the repo, never part of a run (`scripts/README.md`). `check_baseline.py` derives the repo root from its own depth |
| `baseline/` | Replication baseline fingerprints and the discharge reference series; `MANIFEST_PATH_DEFAULT` hardcodes `dev/baseline/manifest.json` |

**Scratch lives outside `dev/`.** Disposable machine-local output goes in the
repo-root `.tmp/` — one ignored directory for everything: probe output,
`scaffold_project_tree.py`'s default target (`.tmp/scaffold`), and any
`pytest --basetemp`. `dev/tmp/` was merged into it on 2026-08-02; there is no
second scratch location, and nothing in `dev/` is a place to park disposables.
Tool caches (`.pytest_cache/`, `.ruff_cache/`) stay at their own defaults.

Shard `tasks/` or `reviews/` into `<year>/` subfolders only if a flat folder
ever grows unwieldy. Generated results, figures, and model outputs go in the
project-root `output/` (gitignored), not `dev/`. Create optional folders only
when first needed — and put new ones in the table above when you do.

## The promotion rule

**A cited note is a record, not a draft.** Before deleting anything from
`working/` at closure, grep the repo for its filename. If a test, module,
config, or tracked document cites it, it is source-of-record: promote it to the
milestone folder (or `tasks/` / `reviews/` / `decisions/`) and update every
citation in the same commit. Citations live in docstrings and prose, so
deleting one breaks provenance silently and no test fails.

This is not hypothetical. On 2026-08-02, 27 files were promoted out of
`working/` — eighteen to `milestones/r08/` alone — and eight of them were cited
by shipped modules, `Snakefile_climate_projections`, and `pixi.toml`.

Never let `working/` or `tmp/` hold the only copy of a primary source: `tmp/`
is gitignored and one `git clean -fdX` from gone.

## The retention rule

Only `reviews/` is prunable. A process review or a post-milestone self-check is
a snapshot of a system that keeps moving, so the folder would otherwise
accumulate thousands of lines describing a repository that no longer exists.

A review may be deleted when **all three** hold:

1. **Nothing cites it** — including other reviews. Check with
   `git grep -l <filename>`, not intuition.
2. **Its items are dispositioned**, and anything carried forward has landed in
   `followups.md`, `TODO.md`, or a decision.
3. **Its durable output has left** — if the review produced skill or role
   candidates, those changes are committed in `brain`. The lesson's home is the
   skill; the review that prompted it is then spent.

Delete whole files. **Do not collapse a register to its outcome summary** — that
rule was tried on 2026-08-02 and fails in practice, twice over: the post-R6
assessment is cited by 23 individual `O-` numbers from R7 docs, `roadmap.md`,
and `pyproject.toml`, and the post-R8 register's own summary states it is "a
derived overview, not a substitute". Citers reference the detail, not just the
path.

**Partial supersession means annotate, not delete.** A later revision often
supersedes only some sections — `wf2-v2-process-review-r2.md` replaces sections
1, 3 and 5 of its predecessor while 2, 4 and 6 stand. Put a banner in the
superseded file saying exactly which sections went and where; keep both.

Nothing outside `reviews/` is prunable. `decisions/` is permanent by
construction, `milestones/` and `tasks/` are identity-indexed records, and
`reference/` describes the current system.

## Milestone records

`milestones/` holds one folder per milestone — `phase-1/m0x/`, `r01/`..`r08/`,
`p31/`, `p32a/`, `p32b/`, `p33/`. Its `README.md` indexes all sixteen with seal
dates and tags.

Two rules govern them:

- **New milestones get a folder there, not at the `dev/` root.** The thirteen
  that were at the root moved into `milestones/` on 2026-08-02 — a path change
  only, no file renamed, split, or edited beyond the prefix.
- **A sealed milestone's contents are not refactored.** Filenames and internal
  grammar vary by era and stay as written; later milestones converge on
  `<topic>-design.md`, `<topic>-task-brief.md`, `migration_<topic>.md`, and
  friends. A milestone folder is also the default home for that milestone's
  promoted working notes.

## Working rules

- **Admit before you track.** Small work fully explained by its diff and Git
  history creates no task ID or record. Track only work that must stay visible
  beyond the current session.
- **The board holds live work only.** Move closed tasks to `tasks/` and drain
  their working notes — deleting the uncited ones, promoting the rest.
- **Handoffs are self-contained.** A note handed to another session or runtime
  states objective, state, decisions, location, validation, next action, and
  blockers.
- **Record exact validation** — the commands run and their outcomes.
- **Log shipped features in the root changelog** (`CHANGELOG.md`) — feature-level
  entries only, linking `decisions/` or `tasks/` for the detail. It lives at the
  project root, not in `dev/`.
