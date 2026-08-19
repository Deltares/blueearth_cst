---
title: Rename snake_config_*.yml to project_config_*.yml repo-wide
type: todo-item
status: backlog
effort: 2
area: config / naming
queue:
created: 2026-08-19
updated: 2026-08-19
---

> [!note] Overview
> **What** — Rename every `snake_config_*.yml` seed and template to
> `project_config_*.yml`, and update the `.gitignore` un-ignore glob, tests,
> docs, `AGENTS.md`, `pixi.toml` and every command line that names one.
> **Why** — Snakemake is one tool this repo happens to use; what the file
> configures is the **project** — the basin, the windows, the models, the
> experiment. `project_config_` says what the file *is*; `snake_config_` says
> which program reads it. The second is the less durable fact: the file would
> keep its meaning if the engine changed.
> **Effort** — Large, but by breadth rather than difficulty. 11 tracked files
> to rename, 221 occurrences across 68 files outside `dev/`, and one glob whose
> failure mode is silent.

## The one that will bite

`.gitignore:130-131` reads

```
test_case/*
!test_case/snake_config_*.yml
```

`test_case/` is otherwise ignored wholesale, so the seed configs are tracked
**only** through that un-ignore glob. Rename the files without moving the glob
in the same commit and they become untracked in silence: `git status` reports
the old paths as deleted and never lists the new ones. `AGENTS.md` already warns
about this for *new* configs; a rename is the same trap from the other side.

Do the `.gitignore` edit and the `git mv` in one commit, and verify with
`git ls-files test_case/` — not with `git status`, which is exactly the command
that lies here.

## Blast radius, measured 2026-08-19

```
11  tracked files named snake_config*   (5 archived, 1 template, 4 test_case, 1 tests fixture)
221 occurrences in 68 files outside dev/
771 occurrences repo-wide  (the difference is dev/ records)
```

Re-measured 2026-08-19 after the Quarto documentation site landed, which added
three more consumers. The repo-wide figure FELL (877 -> 771) while the
outside-`dev/` figure rose: `dev/` records were pruned in between. Read the
outside-`dev/` number as the one that sizes the work — the `dev/` share is
mostly sealed records and history that stays as it is.

Outside `dev/`, the categories are: `.gitignore`, `AGENTS.md`, `README.md`, all
four `*.smk`, five `blueearth_cst/` modules, `pixi.toml` tasks, four
`scripts/`, three `docs/` migration guides, the three `docs/notebooks/*.ipynb`,
**three `docs/guide/*.qmd` pages of the documentation site** (configuration,
quick-start, running — each shows a `--configfile` command line), and **34 files
under `tests/`** (33 test modules plus `conftest.py`).

## Progress

- [ ] Decide the scope of the prefix change: only the `snake_config_` file
      prefix, or also identifiers that carry it (`snake_config_fixture`,
      variable and fixture names). Grep first, triage second — not every
      occurrence of the string is a filename.
- [ ] `git mv` the 11 files; `.gitignore` glob in the SAME commit; verify with
      `git ls-files test_case/`.
- [ ] Update `pixi.toml` tasks, `scripts/run_snake_test.cmd`,
      `scripts/run_snake_docker.sh`, `scripts/run_workflows.py`,
      `scripts/suggest_experiment_name.py`.
- [ ] Update the 33 test modules and `tests/conftest.py`.
- [ ] Update `AGENTS.md` (21 occurrences, including the `.gitignore` rationale
      paragraph, which explains the glob by name) and `README.md`.
- [ ] Write the migration note — `AGENTS.md` requires one for a contract-surface
      rename, and an existing project's `--configfile` path breaks. Follow
      `docs/migration-workflow-names.md`, which is the precedent: the four
      `Snakefile_<noun>` → `*.smk` rename of 2026-08-14.
- [ ] `dev/` records: leave the **4 sealed records** alone
      (`dev/reference/sealed-records.yml`; `tests/test_sealed_records.py` fails
      on an edit). All four mention `snake_config` — 11 occurrences that stay
      stale on purpose. Everything else in `dev/` gets its paths kept current, per
      the Conventions rule that a stale path in a document someone reads to do
      their job is a defect.
- [ ] `pytest tests/test_cli.py` plus the full cheap tier — this touches
      Snakefile config plumbing, so `test-full` at the merge, not just at the
      push.

## Sequencing

Land this **before**
[[t2608191733-ship-a-sample-dataset-bundle-so-a-user-needs-no-deltares-p-drive]]
if both are wanted. That item adds a new `snake_config_sample.yml`; doing it
first only widens this rename, and its `dev/scripts/sample_bundle.yml` already
carries a note pointing here.

The three `docs/notebooks/*.ipynb` carry banner SHAs — see
[[t2608132100-re-render-the-workflow-notebooks-when-their-banner-sha-falls-behind]].
Editing them here may re-open that item rather than closing it.

## Refs

- `AGENTS.md` — Repo Map, `config/` paragraph: why the `.gitignore` pattern has
  the shape it does, and why the prefix is load-bearing.
- `docs/migration-workflow-names.md` — the precedent migration note.
- `dev/reference/naming.md` — the naming rules a renamed contract surface
  answers to.
