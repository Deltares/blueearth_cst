# dev/

Development process and project record. Everything here supports building the
project; none of it ships. The lifecycle that fills these folders is the
`project-system` skill's workflow.

## Layout

| Path | Holds |
|---|---|
| `TODO.md` | Live task board -- unfinished work only (`backlog` / `active` / `blocked`) |
| `working/` | Ephemeral working & handoff notes for **live** work; deleted at task closure unless cited (see below) |
| `tasks/` | One brief record per closed tracked task |
| `decisions/` | Decision records (context, alternatives, consequences); bulky evidence in a `<adr-slug>/` sibling folder |
| `reviews/` | Periodic and milestone review summaries, plus the critiques and briefs behind them |
| `workflows/` | Per-workflow contract docs (wf1/wf2/wf3) -- **live**, cited from module docstrings and config templates |
| `scripts/` | Runnable developer scripts -- build, lint, profile, and exploratory one-offs |
| `tmp/` | Disposable machine-local outputs (gitignored) |

Shard `tasks/` or `reviews/` into `<year>/` subfolders only if a flat folder
ever grows unwieldy. Generated results, figures, and model outputs go in the
project-root `output/` (gitignored), not `dev/`.

**A cited note is a record, not a draft.** Before deleting anything from
`working/` at closure, grep the repo for its filename. If a test, module,
config, or tracked document cites it, it is source-of-record: promote it to the
sealed milestone folder (or `tasks/` / `reviews/` / `decisions/`) and update
every citation in the same commit. Citations live in docstrings and prose, so
deleting one breaks provenance silently and no test fails. Nineteen notes were
promoted this way on 2026-08-02; eight of them were cited by shipped modules,
`Snakefile_climate_projections`, and `pixi.toml`.

Never let `working/` or `tmp/` hold the only copy of a primary source --
`tmp/` is gitignored and one `git clean -fdX` from gone.

## Historical archive (pre-`project-system` convention)

This project predates the type-folder grammar above. Sealed milestone artifacts
keep their original roadmap-driven, milestone-grouped layout -- their *contents*
are **not** refactored. On 2026-08-02 the thirteen milestone folders moved from
the `dev/` root into `milestones/`, a path change only: no file was renamed,
split, or edited beyond the path prefix.

| Path | Holds |
|---|---|
| `roadmap.md` | Source-of-truth fork roadmap: phases, milestones, branching/tagging |
| `followups.md` | Milestone-scoped backlog with reproducible context; referenced by live tests |
| `baseline/manifest.json` | M1 replication baseline fingerprints (read by `scripts/check_baseline.py`) |
| `milestones/` | Every sealed milestone's design / plan / review / evidence docs -- see its `README.md` for the index |

The type-folder grammar governs **new** work; these records stay as-is. A
sealed milestone folder is the default home for that milestone's promoted
working notes -- `milestones/r08/` holds the WF2 v2.0 falsifier and validation
record (R8, sealed 2026-07-31).

## Working rules

- **Admit before you track.** Small work fully explained by its diff and Git
  history creates no task ID or record. Track only work that must stay visible
  beyond the current session.
- **The board holds live work only.** Move closed tasks to `tasks/` and
  delete their working notes.
- **Handoffs are self-contained.** A note handed to another session or runtime
  states objective, state, decisions, location, validation, next action, and
  blockers.
- **Record exact validation** -- the commands run and their outcomes.
- **Log shipped features in the root changelog** (`CHANGELOG.md`) -- feature-level
  entries only, linking `decisions/` or `tasks/` for the detail. It lives at the
  project root, not in `dev/`.

Create optional folders only when first needed.
