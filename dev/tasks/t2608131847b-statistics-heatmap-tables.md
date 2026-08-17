---
title: Add statistics heatmap tables, ported from the fao branch
type: todo-item
status: backlog
effort: 1
area: plotting
origin: fao branch assessment (2026-08-13)
queue:
created: 2026-08-13
updated: 2026-08-13
---

> [!note] Overview
> **What** — Port `plot_table_statistics.py` from the upstream `fao` branch: a heatmap rendering of an indicator table, so a reader sees at a glance which indicators move which way rather than reading a CSV.
> **Why** — The cheapest genuinely useful thing on the whole `fao` branch. Pure pandas/seaborn, no model API, so it drops in — and it is useful to WF2's change factors and WF3's indicator tables whether or not the workflow split has landed.
> **Effort** — Small. 293 lines, three entry points, no new dependency (`seaborn` is already in `pixi.toml`).

## What it is

`upstream/fao:src/plot_utils/plot_table_statistics.py`. Three functions over a
DataFrame:

| Function | For |
|---|---|
| `plot_table_statistics` | single-index table → heatmap |
| `plot_table_statistics_multiindex` | multi-index (e.g. scenario × horizon) |
| `plot_table_statistics_absolute` | absolute values rather than relative change |

Useful options already present: `invert_cmap_for` (indicators where "up" is bad
read in the opposite direction), `bold_keyword`, and explicit `vmin`/`vmax` so a
diverging map is anchored at zero rather than at the data range.

**It ports cleanly.** Pure `pandas` / `seaborn` / `matplotlib` on a DataFrame —
no hydromt, no `WflowModel`, no xarray. This is the exception on a branch whose
code is otherwise pinned to hydromt 0.9 and Wflow.jl v0.

## Where it lands

`blueearth_cst/shared/` — it has more than one consumer by construction, which
is what puts it there rather than in a workflow package. Candidate callers:

- WF3's indicator tables (`shared/indicator_tables.py` decides the set; this
  renders one).
- WF2's change-factor tables (`projections/change_factor_table.py`).

Do not wire both in the same change. Land the module with unit tests, then add
one caller.

## Adapt on the way in, do not copy verbatim

- **House plotting style.** `shared/plot_style.py` exists and the repo is
  mid-sweep on standardizing figures
  ([[t2608091006-standardize-plotting-across-the-toolbox-with-shared-templates-then-sweep-the-existing-figures-onto-them]],
  currently `active`). Coordinate — do not add a figure that the sweep then has
  to re-do. `RdBu` is the `fao` default; the repo's palette decision wins.
  **The blocking half of that coordination cleared 2026-08-17**: the WF2 figure
  set was ruled ADOPTED and the cloud orientation ruled unchanged, so the page
  contract this port must build on is settled — no titles, `a)`/`b)` panel
  labels, `_publication_rc()` + `series_figure_size()` + constrained layout +
  `supxlabel(wrap=True)`. A heatmap is not a series figure, so take the
  typography and export settings from `plot_style.py` and derive the rest;
  do not invent a second page spec.
- **Mutable default arguments.** `invert_cmap_for: List[str] = []` is a mutable
  default; ruff will flag it (B006).
- **Naming** per `dev/reference/naming.md`.
- **Diverging colormaps and CVD.** A red–blue diverging map is the right *form*
  here (signed change about a meaningful zero), but check the palette against
  the `data-visualization` skill's guidance before fixing it.

## The gate is the figure gate, not the ladder

`AGENTS.md` § Figures are terminal artifacts. No rule consumes a `.png` under
`project_dir`, so once this has a caller: (1) the module's unit tests, (2) it
renders without an exception, (3) **the rendered PNG is published as an Artifact**
for visual inspection. No baseline run, no full suite, no byte comparison.

The trap named in that same section applies the moment a caller is wired: a
shared helper edited in service of a plot is a contract surface with other
callers and takes the normal ladder. The module itself is figure-only; a change
to `indicator_tables.py` or `change_factor_table.py` to feed it is not.

## Progress

- [x] Port the module into `blueearth_cst/shared/`, adapted per above —
      **done 2026-08-17 (`3c47b31`)**, `shared/statistics_heatmap.py`.
- [x] Unit tests over a synthetic frame — including `invert_cmap_for` and the
      multi-index path. 22 tests.
- [ ] **Wire ONE caller — NEEDS AN OWNER CALL, see below.** Then render and
      publish the PNG as an Artifact for review.
- [ ] Wire the second caller once the first is accepted.

## Refs

- `dev/reviews/2026-08-13_fao-branch-assessment.md` harvest #7.
- `upstream/fao:src/plot_utils/plot_table_statistics.py`.
- `AGENTS.md` § Figures are terminal artifacts.
- Related: [[t2608091006-standardize-plotting-across-the-toolbox-with-shared-templates-then-sweep-the-existing-figures-onto-them]]
  — that sweep owns the shared templates this should be built on, not beside.


## What landed, and the one thing it deliberately did not decide

**The module is in; the caller is not, and that is a decision rather than an
omission.** Wiring either candidate adds a declared figure to a workflow's
output contract, and the WF2 figure set was ruled on the same day
([[t2608091006]]) — adding a sixth figure to a set the owner had just adopted is
not a call to make inside an implementation commit. The two candidates read
differently:

- **WF2 change factors** (`projections/change_factor_table.py`). The monthly
  table is 12 months × 6 combinations, which is the shape a heatmap is best at,
  and it complements the monthly line figure that landed today. But it reopens a
  contract settled hours earlier.
- **WF3 indicator tables** (`shared/indicator_tables.py`). More indicators, and
  the audience is the stress-test reader. But the WF3 axis is the temp×precip
  grid rather than a scenario, so "indicator × st_id" is a less natural heatmap
  than it first sounds.

**Four departures from upstream** are recorded in the module docstring; the two
worth knowing here are that nothing is transposed internally (the original
flipped the frame, so a caller who had already oriented the table got it rotated)
and that `sns.set_theme` is gone (it rewrites the GLOBAL rc, so any figure drawn
later in the same process inherits seaborn's styling instead of the toolbox page
contract — a real hazard in a job that draws more than one family).

**This module is now what keeps `seaborn` in `pixi.toml`.** The WF2 integration
replaced the change-factor `JointGrid`, which was seaborn's last usage; the
dependency was left declared with nothing importing it — precisely the state that
got `flit` removed under ADR 0008 the same day. If this module is ever deleted,
take `seaborn` with it or give it another job.
