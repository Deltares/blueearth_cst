# Design — a Quarto documentation site for the toolbox, local first

Status: DRAFT v1, for owner review. Repository: `blueearth_cst` (BlueEarth
Climate Stress Test). Author: Claude, 2026-08-19. Supersedes: none.
Branch: `docs/documentation-page` (slot `session-2`).

Revisions:
- 2026-08-19: first draft. Toolchain choice (Quarto) was taken by the owner
  before drafting; the local-first phasing likewise. Three blocking findings
  surfaced during inventory (§2.3–§2.5) and are carried as owner questions
  O-1..O-3.

---

## 1. What this designs, and what was already decided

The owner asked for a documentation site covering three documents — **user
guide**, **technical background**, **setup guide** — to be wired later to a
GitHub Pages site under the `Deltares-research` organisation.

Two decisions are already taken and are **inputs**, not alternatives:

1. **Quarto** is the generator. Verified installable in this environment:
   `quarto 1.9.38` resolves on conda-forge for both `win-64` and `linux-64`
   (`pixi search quarto`), pulling `pandoc 3.8.3`, `typst 0.14.2`, `deno` and
   `dart-sass` — about 25 MB. It therefore does **not** become a second
   out-of-band toolchain of the kind `AGENTS.md` already carries for Julia.
   Alternatives that were weighed and set aside are in §8.
2. **Local first.** The site is built and reviewed on the owner's machine
   (`quarto preview`) with no publishing step at all. Deployment is a separate,
   later phase (§7 P4) that changes exactly two things: a `site-url` and a CI
   workflow.

What this document decides is everything else: where the site lives in the tree,
which documents it publishes, how the 1,727-line technical note becomes
navigable chapters, and what the three findings below force.

---

## 2. What exists today — verified, not assumed

### 2.1 The `docs/` inventory, split by fate

| Path | Size | Fate |
|---|---|---|
| `README.md` (repo root) | 568 lines | stays the GitHub front door; **not** republished verbatim (D2) |
| `docs/install.md` | 131 lines | **publish** — the setup guide, near as-is |
| `docs/cst-toolbox-technical-note-2025.md` | 1,727 lines | **publish, split** — the technical background (§4) |
| `docs/notebooks/` | 3 `.ipynb` + 175-line README | **conditional** — see §2.4 / O-2 |
| `docs/env_setup_notes.md` | 12 KB | publish under setup, as troubleshooting |
| `docs/migration-workflow-names.md`, `docs/migration-r08-wf2.md` | 28 KB | publish under a low-traffic "migrations" section |
| `docs/hydromt-user-guide/` | ~230 KB | **exclude** — vendored upstream |
| `docs/hydromt-wflow/` | ~190 KB | **exclude** — vendored upstream |
| `docs/wflow-user-guide/` | ~70 KB | **exclude** — vendored upstream |
| `docs/hydromt-architecture.md` | 16 KB | **exclude** — vendored/derived upstream material |
| `dev/**` | — | **exclude** — explicitly not shipped (`AGENTS.md`, `dev/README.md`) |

The vendored trees are ~500 KB of Deltares' own published documentation, held
here so an agent can read it offline. Republishing them under a *new* Deltares
site creates a second, staler copy of upstream docs at a URL that looks
authoritative. Link out instead.

### 2.2 The three asks map onto existing material unevenly

- **Setup guide** — exists and is good (`docs/install.md`, seven steps, plus
  `env_setup_notes.md` for troubleshooting). Lowest effort of the three.
- **Technical background** — exists as one 1,727-line file, converted from
  `CST Development Phase Report_CLEAN_Feb14_2025 V2.docx`. Needs splitting and
  a scope decision (O-1). Two defects below.
- **User guide** — **does not exist as such.** The material is inside
  `README.md` (§ Running, § Configuration and run provenance, the four
  per-workflow sections) and inside the notebooks. This is the one that must be
  *authored*, not moved.

### 2.3 Finding: the technical note's images are all missing

The note carries **38 `<img>` tags**, every one of the form:

```html
<img src="../_images/cst-toolbox-technical-note-2025/image7.png" style="width:5.45996in;..." />
```

Relative to `docs/`, `../_images/` is the repository root — which has no
`_images/` directory. `docs/_images/` holds exactly one file, `CST_scheme.png`,
referenced from `README.md` and by nothing in the note. A search over the repo
and over `~/workspace/brain/sources/tools/cst/` returns no
`cst-toolbox-technical-note-2025/` image directory anywhere.

So **every figure in the technical background document is broken today**, and
has been since the docx conversion. On GitHub's markdown view a broken `<img>`
is a small grey placeholder that reads as a rendering quirk; on a published
documentation site it is 38 visible failures on the most formal document there.

This is a content blocker, not a tooling one — no generator can render an image
that is not in the repository. It needs the original docx image extracts (O-3).

### 2.4 Finding: notebook outputs are stripped, not committed

Commit `a2596d0` (2026-08-14), *"refactor(notebooks)!: stop committing rendered
outputs; 8.8 MB -> 0.08 MB"*, reversed the earlier policy. Verified: all three
notebooks contain **zero** `output_type` entries. The rule is enforced three
ways — `dev/scripts/notebook_outputs.py --strip`, the `.githooks/pre-commit`
hook, and `tests/test_notebook_outputs.py`.

The consequence for a site: **rendering the notebooks without executing them
produces code listings with no figures, no tables, and no results.** That is a
markedly worse artifact than the notebook is on GitHub, where a reader at least
knows to run it. Executing them at build time is not available either — a
render needs the full toolbox, a populated `project_dir`, and CMIP6 access, and
costs ~73 minutes for a fresh rapid run (measured 2026-08-13). Options and a
recommendation are in D5 / O-2.

Related: the board's `t2608132100` watch-item still describes the *old* policy
("commit their rendered outputs and carry a dated `rendered against <sha>`
banner"). It was created 2026-08-13, one day before `a2596d0` removed its
premise. Out of scope here, but it should be re-triggered or closed.

### 2.5 Finding: the technical note is a *platform* document, not a toolbox document

Its own preface describes CST as three components — "the BlueEarth CST
workflows, ... the CST-API, ... and the CST-frontend" — and its largest section,
`# CST user workflow` (lines 574–928, 355 lines), is a walkthrough of **seven
GUI screens** of the web frontend. `# Technical Description` similarly documents
CST-frontend and CST-API as first-class subsections.

This repository is the workflow engine only; `AGENTS.md` states it outright
("No web/API code belongs here"), and the standing rule is that the web app's
design never constrains this repo. A documentation site *for this repository*
that opens with seven screenshots of a GUI the reader cannot obtain from this
repository misrepresents what they installed.

This is a scope decision the owner has to make (O-1), not one to infer.

---

## 3. Decisions

**D1 — The site is a Quarto website project rooted at `docs/`.**
`docs/_quarto.yml`, output to `docs/_site/`, cache under `docs/.quarto/`, both
gitignored. Rooting at `docs/` rather than the repo root means the render scope
is a directory that already contains only documentation, so the vendored trees
are excluded by three lines rather than the whole repo being excluded by an
allow-list that has to be maintained against every future `dev/` and root `.md`
file. Rejected alternative: a root-level `_quarto.yml` with an explicit
`project: render:` allow-list — it would let `README.md` render as the site
index, at the cost of a second inventory of the repository that silently rots.

**D2 — One canonical home per document; the site never duplicates prose.**
`README.md` stays the repository's front door and is not copied into the site.
The site's landing page is a short, purpose-built `index.qmd` (what CST is, the
three guides, where the code is). Documents already living under `docs/` are
rendered **in place** — Quarto renders plain `.md` alongside `.qmd`, so
`install.md` needs no conversion and no second copy. This is the existing
"Keep configuration references current" rule applied ahead of time: a copied
paragraph is a paragraph that will disagree with its original within a month.

**D3 — The user guide is authored, by extraction from `README.md`.**
Four pages: *Quick start*, *Configuring a project* (the config bins and the
`snake_config_*.yml` choice), *Running the workflows* (the four entry points in
order, plus `run_workflows.py`), *Reading the outputs* (`project_dir` layout).
The README's corresponding sections are then **reduced to a summary plus a link
to the site**, not deleted and not left duplicated. This is the largest content
task in the project.

**D4 — The technical note becomes ten chapters, and stops numbering itself.**
Mapping in §4. Every top-level heading in the source is literally `# 1.` — an
artifact of the docx conversion — and Quarto's `number-sections: true` restores
correct numbering from document order, so the source numbers are stripped rather
than hand-corrected.

**D5 — Notebooks are published as *walkthroughs*, executed locally, with the
`_freeze/` artifacts committed — subject to O-2.**
Quarto's freeze mechanism stores each document's rendered output under
`docs/_freeze/`, so the site builds anywhere without re-executing. That is the
only route that gets figures onto the site without a build-time run. It does
reintroduce committed rendered output, which `a2596d0` deliberately removed —
which is exactly why it is an owner question and not a decision. The fallback,
if the answer is no, is D5b: the notebooks stay out of the site and the user
guide links to them on GitHub with a line saying they must be run to be useful.

**D6 — The technical background publishes the method, and quarantines the
platform — subject to O-1.**
Recommended shape: chapters 1–6 (preface through licence) publish as the
technical background; the GUI walkthrough and the platform requirements move to
a clearly-labelled *"The wider CST platform"* section with a note that those
components live in other repositories. Nothing is deleted.

**D7 — Toolchain lands as a pixi dependency plus two tasks.**
`quarto = "*"` under `[dependencies]` in `pixi.toml`; tasks
`docs-preview` (`quarto preview docs`) and `docs-build` (`quarto render docs`).
This re-locks `pixi.lock` on both platforms — an expected, reviewable diff. The
typst binary comes with the package, so a PDF of the technical note is available
later with no LaTeX install (§9; not in scope now).

**D8 — Nothing about publishing is built in phases 1–3.** No `site-url`, no
GitHub Actions workflow, no `.nojekyll`, no `quarto publish`. See §7 P4 for what
that phase will need, recorded now so the local site is not built in a way that
has to be undone.

---

## 4. The technical note, split

Source line ranges are from the file as it stands at `e64753a`. Front matter
(lines 1–17) is dropped; it is the brain-repo ingest header, not content. The
manual table of contents (lines 37–120) is dropped; Quarto generates one.

| # | Target file | Source lines | Content |
|---|---|---|---|
| 1 | `background/index.qmd` | 19–36 | Preface — what CST is, its funding and deployment history |
| 2 | `background/introduction.qmd` | 121–171 | Overview; intended uses of CST |
| 3 | `background/architecture.qmd` | 172–230 | Toolbox architecture; the three components |
| 4 | `background/tools.qmd` | 231–293 | HydroMT, Wflow, weathergenr |
| 5 | `background/datasets.qmd` | 294–429 | Non-meteorological and meteorological datasets |
| 6 | `background/license.qmd` | 430–573 | The licence governing CST |
| 7 | `platform/gui-workflow.qmd` | 574–928 | Screens 1–7 of the web frontend (**O-1**) |
| 8 | `platform/requirements.qmd` | 929–1076 | User profiles; functional and non-functional limits (**O-1**) |
| 9 | `background/references.qmd` | 1077–1170 | Bibliography |
| 10 | `background/annex.qmd` | 1171–1727 | Annex |

The original file is **deleted** in the same commit that adds the chapters — it
is not kept as a duplicate. Its history stays in git, and the
`docs/migration-r06.md` precedent says migration maps are not carried for their
own sake.

Two mechanical passes apply to every chapter: rewrite the 38 `<img>` tags to
Quarto figure syntax with captions (blocked on O-3), and drop the `1.` prefixes
from headings.

---

## 5. Site tree

```
docs/
  _quarto.yml            # the project
  index.qmd              # landing page (authored)
  guide/                 # THE USER GUIDE (authored, from README)
    quick-start.qmd
    configuration.qmd
    running.qmd
    outputs.qmd
    notebooks.qmd        # renders the three .ipynb, or links out (O-2)
  setup/
    install.md           # rendered in place, unchanged
    env-notes.md         # env_setup_notes.md, rendered in place
  background/            # THE TECHNICAL BACKGROUND (§4 chapters 1-6, 9, 10)
  platform/              # the wider CST platform (§4 chapters 7-8, O-1)
  migrations/            # the two existing migration notes
  _site/                 # generated, gitignored
  .quarto/               # generated, gitignored
  _freeze/               # committed IF O-2 says yes, else absent
  notebooks/             # unchanged; the .ipynb sources stay where they are
  hydromt-user-guide/    # unchanged, EXCLUDED from render
  hydromt-wflow/         # unchanged, EXCLUDED from render
  wflow-user-guide/      # unchanged, EXCLUDED from render
```

`docs/_quarto.yml`, in the shape it takes at the end of phase 2:

```yaml
project:
  type: website
  output-dir: _site
  render:
    - "*.qmd"
    - "guide/"
    - "setup/"
    - "background/"
    - "platform/"
    - "migrations/"
    - "!hydromt-user-guide/"
    - "!hydromt-wflow/"
    - "!wflow-user-guide/"
    - "!hydromt-architecture.md"

website:
  title: "BlueEarth CST"
  navbar:
    left:
      - text: "User guide"
        href: guide/quick-start.qmd
      - text: "Setup"
        href: setup/install.md
      - text: "Technical background"
        href: background/index.qmd
    right:
      - icon: github
        href: https://github.com/Deltares/blueearth_cst
  sidebar:
    style: floating
    contents: auto
  search: true

format:
  html:
    theme: cosmo
    toc: true
    number-sections: true

execute:
  freeze: auto        # never execute during a site build
```

`.gitignore` gains `docs/_site/` and `docs/.quarto/`. (`/site` is already
ignored at line 116 — a different, root-level path; left alone.)

---

## 6. Validation

A documentation site has no numeric output, so the repository's usual ladder
mostly does not apply; `AGENTS.md` already says a docs-only branch skips
`test-fast`. What replaces it:

1. **`quarto render docs` exits clean** — no unresolved cross-references, no
   missing includes. This is the build gate.
2. **No vendored content in the output.** After a render, a search of
   `docs/_site` for `hydromt` / `wflow-user-guide` paths returns nothing. Cheap,
   and it is the one exclusion a future careless `render:` edit would silently
   undo.
3. **No broken images.** Grep the rendered HTML for `img src` targets and stat
   each one. Until O-3 is answered this fails by design, which is the point.
4. **Link check** — `quarto render` reports dead relative links; external links
   are checked once by hand, not in a loop.
5. **Visual review by the owner**, in `quarto preview`. This is the actual
   acceptance gate and the reason for building locally first.
6. `pixi run lint` / `format-check` stay green — the change touches no Python
   beyond `pixi.toml`, but both are CI gates and near-instant.

Not run, deliberately: the baseline, `tree-check`, the workflow-contract tier.
Nothing here touches `project_dir`, a rule, or a number.

---

## 7. Phases

**P1 — Scaffold (local, ~half a day).** `quarto` into `pixi.toml`; the two pixi
tasks; `docs/_quarto.yml`; `index.qmd`; `setup/` wired to the two existing files
rendered in place; gitignore entries. Deliverable: `pixi run docs-preview`
serves a three-page site with working navigation. Nothing is moved or rewritten
yet, so this phase is reversible by deleting four files.

**P2 — Content (the bulk, ~2–4 days).** The technical-note split (§4); the
authored user guide (D3) and the corresponding README reduction; the notebook
decision applied (D5 or D5b); images repaired (O-3).

**P3 — Polish.** Landing page, cross-references between guides, a Deltares-ish
theme, favicon, 404 page.

**P4 — Publish (deferred; not built until the owner says go).** Recorded now so
P1–P3 do not have to be undone:

- The repository, or a docs-only repository, must exist under
  **`Deltares-research`** — today's remotes are `tanerumit` (origin) and
  `Deltares` (upstream), and no `deltares-research.github.io` repository exists,
  so the URL will be a **project page on a subpath**,
  `https://deltares-research.github.io/<repo>/`.
- That subpath must be declared (`website: site-url:`) or absolute links and
  assets break *only once deployed* — a failure invisible in local preview.
- A GitHub Actions workflow using `quarto-dev/quarto-actions` renders and pushes
  to `gh-pages`; a `.nojekyll` file at the site root.
- CI must not execute notebooks — the committed `_freeze/` (or their absence)
  is what makes that safe.

---

## 8. Alternatives considered

**MkDocs + Material** (`mkdocs-material 9.7.7`, `mkdocs-jupyter 0.25.1`, both on
conda-forge). Lighter — pure Python, no binary toolchain, and `docs/` is already
Markdown so migration is close to a `nav:` block. Better search out of the box.
Set aside because it gives no print/PDF output, which the technical note — a
report by origin and by audience — is the natural consumer of. **Preferable if**
the owner later decides the PDF is unwanted and the 25 MB Quarto toolchain is
resented.

**Sphinx + MyST + pydata-sphinx-theme.** Matches HydroMT's house style and is
the strongest option for API reference generated from docstrings. Set aside
because this repository has no public API to document: `blueearth_cst/` modules
are Snakemake `script:` targets, and there is no package CLI. **Preferable if**
the toolbox is ever packaged and importable.

**Publish nothing; keep GitHub-rendered Markdown.** Zero cost and honest. Set
aside because it cannot give a subpath-hosted site to a non-GitHub audience,
which is the request.

House style does not break the tie in either direction: Wflow.jl — the engine
this toolbox runs — is Quarto (`docs/_quarto.yml`), while HydroMT is Sphinx
(`docs/conf.py`). Both are Deltares.

---

## 9. Consequences

Falsifiable, so a later reader can tell whether this worked:

- `pixi.lock` grows by the Quarto toolchain on both platforms; `pixi install`
  in a fresh worktree gets ~25 MB slower. Every session slot pays it once.
- `docs/` stops being a flat pile of Markdown and becomes a project with a
  render scope. A `.md` file dropped into `docs/` **is published by default**
  once it sits in a rendered directory — the inverse of today, where a file
  there is only ever read deliberately. Anyone adding a note under `docs/`
  after P1 must now think about audience.
- The 568-line README shrinks. Its Running / Configuration sections become
  summaries pointing at the site.
- One more thing that can rot without a gate noticing: the site's prose against
  the code. The existing "Keep configuration references current" rule covers it
  in principle; in practice the site adds surface area, and `docs/notebooks/`
  has already demonstrated that documentation nothing executes goes stale
  (three tree-moves' worth, per `t2608131847`).
- The technical note's mixed scope becomes visible rather than buried at line
  574 — a benefit and a small embarrassment. O-1 exists because publishing makes
  the question unavoidable.

---

## 10. Open questions — owner

**O-1 (blocks P2, §4 chapters 7–8).** The technical note documents the whole
three-part CST platform, including a 355-line walkthrough of the web GUI. On a
site for *this repository*, do we: (a) publish it in a separately-labelled
"wider CST platform" section [recommended]; (b) publish it inline as-is,
accepting that the site describes software the reader did not install; or
(c) leave those two chapters unpublished and keep them in the repo as source?

**O-2 (blocks P2, D5).** Notebooks currently ship stripped by policy
(`a2596d0`). Do we (a) render them locally once and **commit `docs/_freeze/`**
so the site shows real figures [recommended — it is scoped, machine-generated
output whose whole purpose is publication, not notebook state re-entering the
tracked `.ipynb`]; or (b) keep them out of the site and link to GitHub?

**O-3 (blocks P2, §2.3).** The technical note's 38 images do not exist in the
repository. Can you supply the image set from
`CST Development Phase Report_CLEAN_Feb14_2025 V2.docx`? Without it the
technical background publishes as text only, and every figure reference in the
prose dangles.

**O-4 (blocks P4 only).** Is moving or mirroring this repository into the
`Deltares-research` organisation actually on the table, or is that org simply
where you have Pages rights? The answer changes nothing before P4, but it
changes the site URL and therefore `site-url`.

---

## 11. Refs

- `AGENTS.md` — repo map (`docs/` vs `dev/`), the docs-only branch exemption in
  the validation ladder, and the workflow-engine scope constraint.
- `docs/notebooks/README.md` § Outputs are NOT committed, § Re-rendering.
- `dev/tasks/t2608132100-re-render-the-workflow-notebooks-when-their-banner-sha-falls-behind.md`
  — stale premise, see §2.4.
- `a2596d0` (2026-08-14) — the commit that stopped committing notebook outputs.
- Deltares precedents: `Deltares/Wflow.jl` (`docs/_quarto.yml`),
  `Deltares/hydromt` (`docs/conf.py`).
