---
title: Publish the docs site to GitHub Pages, and reduce the README onto it
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

## Why the README reduction is a step here and not its own item

Ruling **D11** (2026-08-19) ties them together deliberately: a README pointing at
a URL that does not exist yet is worse than one that repeats itself. The
duplication between README and site is known, accepted and temporary, and it
clears in the same change that adds `site-url`. Splitting the two is how the
README ends up pointing at a 404.

## Refs

`dev/working/2026-08-19_quarto-docs-site/design.md` §7 P4, D11, §10 O-4.
Sibling items: [[t2608202351]], [[t2608202351a]], [[t2608202351b]].
