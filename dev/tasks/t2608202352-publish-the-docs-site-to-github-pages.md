---
title: Publish the docs site to GitHub Pages, and reduce the README onto it
type: todo-item
status: blocked
effort: 2
area: docs / site
origin: quarto-docs-site (2026-08-19)
queue:
created: 2026-08-20
updated: 2026-08-21
---

> [!note] Overview
> **What** — The design's phase 4: declare `site-url`, add a `quarto-actions` workflow rendering to `gh-pages`, add `.nojekyll`, and in the same change reduce the README's Running and Configuration sections to summaries pointing at the site (D11).
> **Why** — The site is deliberately local-only today. Everything phases 1-3 build is invisible to anyone who has not cloned the repo, which is most of the audience the user guide was written for.
> **Effort** — large — mostly first-deploy debugging, which is the part that cannot be rehearsed locally

## Progress

- [ ] Owner answers O-4 (see Blocker)
- [ ] `website: site-url:` set to the resolved project-page subpath
- [ ] GitHub Actions workflow using `quarto-dev/quarto-actions`, rendering to `gh-pages`
- [ ] `.nojekyll` at the site root
- [ ] CI must not execute anything — the committed `_freeze/` from [[t2608202351a]], or its absence, is what makes that safe
- [ ] Reduce the README's Running / Configuration sections to summaries linking the site (D11)
- [ ] Re-check `404.html`'s links after setting `site-url` (see below)
- [ ] Sweep for hardcoded `github.com/<org>/blueearth_cst` URLs if O-4 moves the repository — `_quarto.yml`, `index.qmd`, `404.qmd` carry one each
- [ ] Walk the deployed site for broken asset and link paths — the failure this whole item exists to sequence

## Blocker — the URL is not knowable yet

Design open question **O-4**, unanswered: is moving or mirroring this repository
into the **Deltares-research** organisation actually on the table, or is that
simply where Pages rights exist? Today's remotes are `tanerumit` (origin) and
`Deltares` (upstream), and no `deltares-research.github.io` repository exists, so
the result will be a project page on a subpath —
`https://deltares-research.github.io/<repo>/` — and `<repo>` is the unknown.

**Trigger — this unblocks when the owner names the repository's home.**

The subpath is why the answer is load-bearing rather than cosmetic. A wrong or
absent `site-url` breaks absolute links and assets **only once deployed**, and
local preview shows nothing wrong — so this cannot be got right by guessing and
checking locally.

## The 404 page is the concrete case of the missing `site-url` — measured 2026-08-21

`docs/404.qmd` landed in [[t2608202351b]], and rendering it showed exactly how
the missing `site-url` bites. Quarto rewrites a 404 page's links to be
**root-absolute**, because a 404 is served from an arbitrary depth and relative
links would resolve against the wrong directory:

    href="/guide/configuration.html"
    href="/env_setup_notes.html"

Correct at a domain root, wrong on a project-page subpath, where they must read
`/blueearth_cst/guide/...`. Quarto derives that prefix from `site-url` and has
no other source for it.

So this is not a general "check the links after deploying" — it is a specific
page that is *known* to be wrong until `site-url` is set, and which local
preview will keep showing as fine. Verify it deliberately.

## Why the README reduction is a step here and not its own item

Ruling **D11** (2026-08-19) ties them together deliberately: a README pointing at
a URL that does not exist yet is worse than one that repeats itself. The
duplication between README and site is known, accepted and temporary, and it
clears in the same change that adds `site-url`. Splitting the two is how the
README ends up pointing at a 404.

## Refs

`dev/working/2026-08-19_quarto-docs-site/design.md` §7 P4, D11, §10 O-4.
Sibling items: [[t2608202351]], [[t2608202351a]], [[t2608202351b]].
