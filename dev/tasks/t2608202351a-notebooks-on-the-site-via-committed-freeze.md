---
title: Put the three pipeline notebooks on the site via a committed docs/_freeze/
type: todo-item
status: backlog
effort: 2
area: docs / site
origin: quarto-docs-site (2026-08-19)
queue:
created: 2026-08-20
updated: 2026-08-20
---

> [!note] Overview
> **What** — Execute the three notebooks under `docs/notebooks/` once locally, commit the resulting `docs/_freeze/`, and add them to the render list in `docs/_quarto.yml`. Ruling O-2(a), taken 2026-08-19 and not yet applied.
> **Why** — The notebooks are the only worked walkthrough of the pipeline, and they are the one part of `docs/` a reader cannot use from the repo: outputs are stripped by policy (`a2596d0`), so the committed `.ipynb` files are code with no results.
> **Effort** — large — the cost is the run, not the docs edit

## Progress

- [ ] Produce or locate a run tree the notebooks can read (see Precondition)
- [ ] Render once with execution enabled, producing `docs/_freeze/`
- [ ] Commit `docs/_freeze/` and add `notebooks/*.ipynb` to the render list and sidebar
- [ ] Re-render from clean with no execution and confirm the figures survive (`freeze: auto` must not re-execute)
- [ ] State in `docs/notebooks/README.md` that the site serves frozen output, so a stripped `.ipynb` is not read as a defect

## Precondition — this needs a run, not a docs edit

Whoever picks this up should expect to spend the time on a pipeline run.
`docs/_quarto.yml` records that a render has no `project_dir`, no CMIP6 access and
no Julia, and a fresh rapid run costs ~73 minutes. The notebooks read real
outputs, so they execute only against a completed run tree. Nothing external
gates this — hence `backlog`, not `blocked` — but it is not a five-minute change.

## The ignore rule already anticipates this — verified 2026-08-20

`docs/_freeze/` is **not** swept by the quarto ignore block, and `.gitignore:123`
says so in as many words: *"`docs/_freeze/` is deliberately NOT ignored: if the
notebooks are ever ..."*. Confirmed with `git check-ignore -v docs/_freeze/test`,
which reports no match. O-2(a) is implementable as written.

## Relationship to the stripped-output policy

This does not reverse `a2596d0`. That commit keeps *notebook state* out of the
tracked `.ipynb` files — one of them was 6.43 MB across 82 blob versions
(`dev/scripts/notebook_outputs.py`). `docs/_freeze/` is separate,
machine-generated publication output whose whole purpose is the site, and it does
not put outputs back into the notebooks. Worth stating in the commit message,
because it is the first question a reviewer will ask.

## Refs

`dev/working/2026-08-19_quarto-docs-site/design.md` §2.4 (the stripped-output
finding), D5, §7 P2, §10 O-2. `docs/notebooks/README.md`,
`dev/scripts/notebook_outputs.py`. Sibling items: [[t2608202351]],
[[t2608202351b]], [[t2608202352]].
