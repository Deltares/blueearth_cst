---
title: Replace the docs site's approximate brand colours with the official Deltares values
type: todo-item
status: blocked
effort: 1
area: docs / site
origin: quarto-docs-site (2026-08-19)
queue:
created: 2026-08-21
updated: 2026-08-21
---

> [!note] Overview
> **What** — Swap the three placeholder colours at the top of `docs/theme.scss` for the values in the Deltares brand guide.
> **Why** — The site is branded in shape and approximate in colour. Until this lands it is Deltares-*ish*, which is the wrong thing to publish under a Deltares URL — and an approximation nobody is tracking is how it becomes permanent.
> **Effort** — small — three lines, once the values exist

## Progress

- [ ] Obtain the official values (see Blocker)
- [ ] Replace `$deltares-navy`, `$deltares-blue`, `$deltares-teal` in `docs/theme.scss`
- [ ] Update the matching hexes in `docs/_images/favicon.svg` — it is hand-authored and does not read the SCSS
- [ ] Delete the "THESE ARE APPROXIMATIONS" banner from the `theme.scss` header
- [ ] `pixi run docs-build`, then look at the rendered page

## Blocker — the values are not in this repository

**Trigger — this unblocks when the owner supplies the Deltares brand palette**
(or rules that the current approximations are good enough, which closes it as
`dropped`).

The three placeholders, chosen 2026-08-21 to sit in the right family and **not**
copied from any brand guide:

| Token | Placeholder | Used for |
|---|---|---|
| `$deltares-navy` | `#0a2e4d` | navbar, footer top rule |
| `$deltares-blue` | `#0b7ebb` | `$primary`, links, sidebar active item |
| `$deltares-teal` | `#00a0aa` | heading hairlines, footer border |

## Why this is three lines and not a re-theme

`docs/theme.scss` was built so this stays cheap: the palette is three variables
at the top of `scss:defaults`, everything downstream refers to them by name, and
nothing else in the file carries a literal colour. The one place a value is
duplicated is `docs/_images/favicon.svg`, which is hand-authored SVG and cannot
read SCSS — hence the explicit step above, and a comment in the SVG saying so.

The file also deliberately never sets `$body-bg` or `$body-color`. That is what
keeps a future dark mode a two-line change in `_quarto.yml`; see [[t2608202351b]]
in `dev/LOG.md` for why dark was removed.

## Refs

`docs/theme.scss`, `docs/_images/favicon.svg`, `docs/_quarto.yml`.
Design: `dev/working/2026-08-19_quarto-docs-site/design.md` §7 P3.
Sibling items: [[t2608202351]], [[t2608202351a]], [[t2608202352]].
