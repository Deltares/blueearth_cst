# Git conventions

Last updated: 2026-11-05

Use this reference for current branch names, checkpoint tags and commit messages in this repository.
The [validation ladder](dev-validation.md) describes checks before landing or publishing.

## Branches and remotes

Name task branches `<type>/<slug>`, using a Conventional Commit type and a short kebab-case description:
`fix/fixture-digest-drift-attribution`, `docs/config-modularization` or
`refactor/wf0-rule-label-constants`. Do not put a milestone number in the name. Record the milestone,
when applicable, with the task and commit.

The configured working trunk is `dev`. `main` is the PR-based public branch. `origin` fetches from and
pushes to the Deltares repository; there is no configured `upstream` remote. Check `git branch -a`,
`git remote -v` and `git status --short` when the current checkout matters. Work on a task branch in an
isolated checkout and land verified changes on `dev`; the primary checkout is the integration point.

Existing historical refs may use `base/<start-point>`, `milestone/<NN>-<topic>`,
`milestone/r<NN>-<topic>`, `milestone/p3<N><letter?>-<topic>` or `lane/<territory>`. These are not
patterns for new task branches. Use `git branch -a` to inspect surviving branches; Git history and
tags retain the earlier context.

## Tags and checkpoints

Development checkpoint tags include Phase 1 `m##-<topic>`, Phase 2 `r##-<topic>` and Phase 3
`p3#-<topic>`. Archived work uses `archive/<topic>`. Use `git tag` for the actual inventory. Treat an
existing checkpoint as permanent; never move it. Ordinary work needs no milestone tag. Release tags
follow the separate [versioning reference](dev-versioning.md).

A tag names committed state. It does not preserve uncommitted changes or outputs under `project_dir/`.
The baseline manifest records comparison information for selected outputs; it does not back up those
files. For example, inspect changes since `pre-r01` with:

```powershell
git diff pre-r01 -- <path>
```

`git restore --source pre-r01 -- <path>` replaces that working file with its tagged version. Check the
diff before using it and preserve any work you still need. `git reset --hard pre-r01` discards the
current branch's later commits and working changes. Check the target branch and preserve needed work
before using that destructive command.

For a milestone seal, follow
[record lifecycle](dev-record-lifecycle.md#milestone-records-and-historical-recovery) to preserve
superseded specifications and seal metadata.

## Commit messages

Use a Conventional Commits subject: `<type>(<scope>): <imperative subject>`. Keep it at most 72
characters, with no trailing period. Add `!` before the colon for a breaking change, as in
`refactor(workflows)!: rename the workflows.<name> config keys and every derived path`.

| Type | Use |
|---|---|
| `feat` | New capability |
| `fix` | Defect correction |
| `docs` | Documentation, including `dev/` records and the board |
| `refactor` | Behavior-preserving restructuring |
| `test` | Tests only |
| `chore` | Environment, ignore rules or tooling housekeeping |
| `perf` | Measured performance improvement |
| `style` | Formatting only, with no code change |

Use a lowercase scope for the affected workflow (`wf0`, `wf2`, `wf3`), subsystem (`console`,
`stage`, `climate`, `plotting`, `runner`) or documentation area (`board`, `dev`, `ref`, `agents`,
`tests`). Omit the scope when the change is genuinely repository-wide.

Example subjects:

- `fix(wf2): refuse to attribute drift against a mismatched fixture`
- `refactor(tests): read LOG_RULES from the parsed workflow, and retire the second parser`
- `docs(ref): state the rules in dev-naming-conventions.md, drop the archaeology`

Write a body only when the reason is not clear from the diff. Keep one logical change per commit; split
unrelated changes. Stage and commit only task-owned hunks. An explicit pathspec does not isolate hunks
in a file shared with other work, and `git commit -a` can include unrelated tracked changes. Inspect
the final staged diff and effective author and committer identity before committing.

Do not commit run outputs under `project_dir/`, local-only configs such as `*_local.yml`, secrets,
credentials, large binary fixtures or generated baselines other than `dev/baseline/manifest.json`.
Never hand-edit `pixi.lock` or `Manifest.toml`. Check `git status --short` and `git diff --cached`
before committing, and keep installed hooks enabled. If an unwanted file enters an unpublished commit,
remove it from unpublished history and update `.gitignore` where appropriate.

Default merge-commit messages are sufficient. An annotated development tag message states the
milestone goal in one line.
