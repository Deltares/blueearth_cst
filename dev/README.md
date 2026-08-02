# dev/

Development process and project record. Everything here supports building the
project; none of it ships. The lifecycle that fills these folders is the
`project-system` skill's workflow.

## Layout

Live process state — the type-folder grammar:

| Path | Holds |
|---|---|
| `TODO.md` | Live task board — unfinished work only (`backlog` / `active` / `blocked`) |
| `working/` | Working & handoff notes for **live** work; drained at task closure, but see the promotion rule below |
| `tasks/` | One brief record per closed tracked task |
| `decisions/` | Decision records (context, alternatives, consequences); bulky evidence in a `<adr-slug>/` sibling folder |
| `reviews/` | Periodic and milestone review summaries, plus the critiques and briefs behind them |
| `scripts/` | Runnable developer scripts — inspect or maintain the repo, never part of a run (`scripts/README.md`) |
| `tmp/` | Disposable machine-local outputs (gitignored) |

Durable reference — read, rarely rewritten:

| Path | Holds |
|---|---|
| `roadmap.md` | Source of truth: phases, milestones, branching/tagging conventions |
| `followups.md` | Milestone-scoped backlog with reproducible context; cited by live tests. Detail store behind `TODO.md` |
| `branches-and-tags.md` | Inventory of durable refs and what each is for; transient branches excluded |
| `conventions/` | `naming.md` (prescriptive identifier/file style) and `agent-activation.md` (how roles and skills load per runtime) |
| `contracts/` | The two substitution seams — hydrological model, weather generator — pinned as machine-checked contracts (P3-2b) |
| `workflows/` | Per-workflow contract docs (wf1/wf2/wf3) — **live**, cited from module docstrings and config templates |
| `baseline/` | Replication baseline fingerprints (`manifest.json`) and the discharge reference series, read by `scripts/check_baseline.py` |
| `milestones/` | Every milestone's design / plan / review / evidence docs — see its own `README.md` for the index |

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
