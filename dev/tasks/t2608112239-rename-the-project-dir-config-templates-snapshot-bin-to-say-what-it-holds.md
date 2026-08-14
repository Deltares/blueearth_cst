---
title: Rename the project-dir config/templates snapshot bin to say what it holds
type: todo-item
status: backlog
effort: 1
area: naming
queue:
created: 2026-08-11
updated: 2026-08-11
---

> [!note] Overview
> **What** — A finished project keeps copies of the inputs its run used, sorted
> into bins under `<project_dir>/config/`: `catalogs/`, `basin_data/`, `runs/`
> and `templates/`. The last one holds the two hydromt build files WF1 actually
> used (`wflow_build_model.yml`, `wflow_update_waterbodies.yml`); rename it to a
> word that describes those files, since nothing about them is a template.
> **Why** — The bin is named after the repository directory those two files used
> to live in, and they no longer live there: the 2026-08-11 split moved them to
> `config/defaults/`, leaving `config/templates/` in the repo holding only
> copy-me scaffolds. Two directories now share one name and mean opposite
> things — one holds scaffolds you edit, the other holds read-only evidence of
> what a run consumed — and a reader inspecting a project tree reasonably
> concludes the run's build config is editable there. Three separate comments
> in the codebase already exist to warn about exactly this.
> **Effort** — Small in code, but it moves a directory inside every project
> tree, so it needs a migration entry and a decision on already-run projects.
> The real unknown is the digest: `"templates"` is also the `kind` string in the
> snapshot reference descriptors, and `kind` is hashed into the run digest, so
> renaming the string alone re-digests every run and orphans every existing
> `config/runs/<workflow>/<digest>/` directory.

## Progress

- [ ] Decide the new name (`build_config/`? `model_config/`?) — it must not read
      as a fourth thing next to `catalogs/`, `basin_data/`, `runs/`. The sibling
      bin was renamed `observations/` → `basin_data/` on 2026-08-14 on the
      content-kind criterion (name the bin for what it holds, not for who
      supplied it or for the repo directory the files came from); apply the same
      test here
- [ ] Decide whether the snapshot `kind` string changes with the directory or
      stays `"templates"` for digest stability; if it changes, say what happens
      to existing `config/runs/` digests
- [ ] Update the writer: `blueearth_cst/model/copy_config_files.py:254,269-270`
- [ ] Update the digest inputs: `Snakefile_model_creation:150-162`
      (`_snapshot_reference("templates", ...)`) — only if the previous step says so
- [ ] Update the tree inventory map so `tree-check` accepts the new shape:
      `dev/scripts/semantic_tree_diff.py:887,1055`
- [ ] Update the tests that assert the old bin: `test_copy_config_files.py`,
      `test_model_creation.py`, `test_project_tree_inventory.py`
      (repo-side `config/templates/` hits in other tests are unaffected)
- [ ] Sweep the prose that documents the collision — `AGENTS.md:69`,
      `config/defaults/README.md:26`, `Snakefile_model_creation:107-110` — and
      delete the warnings the rename makes unnecessary

## Refs

- Surfaced 2026-08-11 while reading `test_case/test_local/config/`; the bin was
  named at R07 B9, when `config/templates/` was the source directory.
- The repo-side split that broke the correspondence: `config/defaults/README.md`.
- `blueearth_cst/shared/provenance.py:224-243` — `kind` is part of the hashed
  reference document, which is what makes the string load-bearing.
