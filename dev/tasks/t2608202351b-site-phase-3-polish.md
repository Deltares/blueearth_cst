---
title: Finish the site's phase-3 polish — landing page, cross-links, theme, 404
type: todo-item
status: backlog
effort: 1
area: docs / site
origin: quarto-docs-site (2026-08-19)
queue:
created: 2026-08-20
updated: 2026-08-20
---

> [!note] Overview
> **What** — The design's phase 3: work the landing page past a stub, cross-link the four guide pages to each other, settle a Deltares-ish theme and favicon, add a 404 page. Plus the two migration notes, which are a render-list line and too small for their own item.
> **Why** — Phase 1 shipped `theme: cosmo` and a placeholder landing page as scaffolding, not as a choice. This is the difference between a site that renders and a site someone is willing to send a link to — and it is the last thing that is cheap to do while the site is still local.
> **Effort** — small

## Progress

- [ ] Landing page (`docs/index.qmd`) — say what the toolbox is and route the three reader types
- [ ] Cross-links between `guide/quick-start`, `configuration`, `running`, `outputs`
- [ ] Theme and favicon; decide whether `cosmo` stays
- [ ] `404.qmd`
- [ ] Add `migration-workflow-names.md` and `migration-r08-wf2.md` to the render list — they were slated for phase 2 and never landed
- [ ] `pixi run docs-build` clean, and `pixi run docs-preview` walked once by eye

## Why the migration notes ride along here

They are two lines in the `render:` list of `docs/_quarto.yml`. On their own they
fail the board's admission gate — a self-contained edit the diff explains — but
they are also the last unlanded piece of phase 2's render list, and the header
comment in `_quarto.yml` still promises them (*"migrations -- the two migration
notes join in phase 2"*). Landing them here keeps that comment from going stale
and saves an item nobody needs.

## Do this before publishing, not after

Not blocked on anything, and worth taking before [[t2608202352]]. Theme, favicon
and 404 are all cheap to change while nobody has the URL and progressively less
so afterwards. The one thing that genuinely cannot be settled locally is
`site-url`, which is that item's whole subject.

## Refs

`dev/working/2026-08-19_quarto-docs-site/design.md` §7 P3. Sibling items:
[[t2608202351]], [[t2608202351a]], [[t2608202352]].
