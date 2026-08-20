---
title: Finish the site's phase-3 polish — landing page, cross-links, theme, 404
type: todo-item
status: active
effort: 1
area: docs / site
origin: quarto-docs-site (2026-08-19)
queue:
created: 2026-08-20
updated: 2026-08-21
---

> [!note] Overview
> **What** — The design's phase 3: cross-link the four guide pages to each other, settle a Deltares-ish theme and favicon, add a 404 page. Plus the two migration notes, which are a render-list line and too small for their own item.
> **Why** — Phase 1 shipped `theme: cosmo` as scaffolding, not as a choice. This is the difference between a site that renders and a site someone is willing to send a link to — and it is the last thing that is cheap to do while the site is still local.
>
> **Correction, 2026-08-21** — this note originally said the landing page was a stub to be worked past. It is not: `index.qmd` already carries the pitch, the reader-routing table and the method framing. Nothing there needed rewriting, and the only change it took was the GitHub URL below.
> **Effort** — small

## Progress

- [x] Landing page — verified substantive; Start-here table resolves (see the correction above)
- [x] Cross-links — `configuration.qmd` and `outputs.qmd` had no exit; both now carry a Next block
- [x] Theme and favicon — brand layer over a light/dark pair, plus a hand-authored SVG favicon
- [ ] **Replace the three approximate brand colours in `docs/theme.scss` with the official Deltares values**
- [x] `404.qmd`
- [x] `migration-workflow-names.md` and `migration-r08-wf2.md` in the render list and sidebar
- [x] `pixi run docs-build` clean — 7 pages before, 10 after, exit 0
- [ ] `pixi run docs-preview` walked once by eye by the owner

## The theme is right in shape and approximate in colour

Ruling, 2026-08-21: a branded `theme.scss` rather than a stock bootswatch. It is
one brand layer applied over **both** bases — `cosmo` in light, `darkly` in dark
— which is why the file sets only accents, type and rules, and never `$body-bg`
or `$body-color`. Both bases would otherwise fight it.

The three colours in it (`$deltares-navy`, `$deltares-blue`, `$deltares-teal`)
were chosen to sit in the right family and are **not** from the Deltares brand
guide. They are marked as approximate in the file header. Swapping them is a
three-line edit that needs no other change, which is the reason the file is
built that way — but until it happens the site is brand-*ish*, not branded, and
that is the one thing keeping this item open.

## The GitHub URL now resolves

`index.qmd` and the navbar both pointed at `github.com/Deltares/blueearth_cst`.
Owner ruling 2026-08-21: point them at `tanerumit`, which is where the code
actually is today. If O-4 moves the repository, [[t2608202352]] sweeps them.

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
