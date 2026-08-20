---
title: Publish the technical background — split the 2025 note into eight chapters
type: todo-item
status: blocked
effort: 2
area: docs / site
origin: quarto-docs-site (2026-08-19)
queue:
created: 2026-08-20
updated: 2026-08-20
---

> [!note] Overview
> **What** — Split `docs/cst-toolbox-technical-note-2025.md` (157 KB, one file) into the eight chapters of the design's §4 and add them to the Quarto render list as a `background/` section. Chapters 7-8 (the web GUI and the wider platform) stay out by ruling O-1(c).
> **Why** — The site publishes the user guide and setup, and nothing about *why* the toolbox computes what it does. The note is the only written method rationale, and today it is reachable only as a 157 KB file in the repo.
> **Effort** — large

## Progress

- [ ] Owner supplies the figure set (see Blocker)
- [ ] Split the note into the eight chapters of design §4
- [ ] Add `background/*.qmd` to the render list and the sidebar in `docs/_quarto.yml`
- [ ] Repoint every in-repo reference to `docs/cst-toolbox-technical-note-2025.md` (AGENTS.md cites it; grep before landing)
- [ ] `pixi run docs-build` renders with no dangling figure reference

## Blocker — the 38 figures do not exist in this repository

Design open question **O-3**, unanswered. The note's prose references 38 figures;
`docs/_images/` holds exactly one file, `CST_scheme.png`. The original source is
named in the design as `CST Development Phase Report_CLEAN_Feb14_2025 V2.docx`.

**Trigger — this unblocks when the owner supplies that image set** (or rules that
the background publishes as text with the figure references stripped, which is a
different and smaller job).

Publishing without them is the thing to avoid rather than a degraded option: with
`number-sections: true` the chapters render as numbered captions under nothing,
which reads as a broken site instead of an incomplete one.

## Scope note — this is not the whole note

Ruling **O-1(c)** (2026-08-19) leaves the GUI walkthrough and the wider-platform
chapters unpublished; they stay in the repo as source. §4 therefore publishes
eight chapters, not ten, and there is no `platform/` directory. That matches the
repo's own scope constraint: this is the workflow engine, and the site should not
describe software the reader did not install.

Ruling **O-5(c)** settles the form: a plain `type: website`, one page per chapter,
section numbering restarting per page, no PDF. No nested `type: book` sub-project.

## Refs

Design and phasing: `dev/working/2026-08-19_quarto-docs-site/design.md` §2.3
(the missing-images finding), §4 (the chapter split), §7 P2, §10 (O-1, O-3, O-5).
Sibling items: [[t2608202351a]], [[t2608202351b]], [[t2608202352]].
