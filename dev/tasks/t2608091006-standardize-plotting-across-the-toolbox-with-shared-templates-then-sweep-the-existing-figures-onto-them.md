---
title: Standardize plotting across the toolbox with shared templates, then sweep the existing figures onto them
type: todo-item
status: active
branch: feat/plotting-standardization
effort: 2
area: plotting
queue:
created: 2026-08-09
updated: 2026-08-11
---

> [!note] Overview
> **What** — Two halves. (1) Introduce shared plotting templates/style primitives so figures across the toolbox agree on typography, palette, figure size, colourbar placement and export settings, instead of each module deciding independently. (2) Sweep the existing figure producers onto them. Current plotting surface -- 8 modules under blueearth_cst/ plus 2 dev scripts. climate_analysis/climate_figures.py, climate_analysis/plot_climate_source.py, model/plot_results.py, projections/get_change_climate_proj_summary.py, projections/plot_proj_timeseries.py, shared/func_plot_signature.py, shared/plot_map.py, shared/snake_utils.py (plotting primitives), dev/scripts/basin_map_example.py, dev/scripts/preview_basin_map.py. No R-side plotting exists.
> **Why** — Owner request 2026-08-09. Figures are the deliverable a reader actually sees, and they are styled per-module today, so one basin assessment ships figures that do not look like each other. Design before implementing -- a shared template module is a contract surface every plot imports, and AGENTS.md is explicit that a shared helper edited in service of a plot is NOT a figure-only change. It takes the full validation ladder, while the figures themselves are verified by rendering and looking at them.
> **Effort** — large

## Progress

**Half 1 is complete; half 2 now covers the climate-map and WF2 projection
families.** The climate-map sweep supplied the worked example, and `6d3ec75`
then extracted the shared page and typography contract to
`shared/plot_style.py`.

Read the order as accidental rather than planned: the map family got swept first
because that is where the owner was looking, and the template it implies has to
be extracted from what those commits converged on rather than designed ahead of
them. That is a fine way round, but it means half 1 is now an EXTRACTION job
with a worked example, not a greenfield design.

- [x] Frame both climate map families on the basin — `11ffcc4`
- [x] Share colourbar levels across each source/forcing pair — `403b55e`
- [x] Bring the annual and monthly charts into the figure family — `d8f8ca8`
- [x] Colourbar title breaks at its unit; seasonal key dropped — `ab31ecc`
- [x] Retire the subcatchment climate plots (ADR 0006) — `700caa8`
- [x] Draw `basin_area` from the spatial foundation (ADR 0007) — `8cf2901`
- [x] **Half 1: extract the shared template/style module** — `6d3ec75` added
      `shared/plot_style.py` for page size, typography and export settings
- [ ] **Half 2, remainder — the surfaces still styled independently:**
  - [x] `projections/get_change_climate_proj_summary.py` — faceted
        scenario-colour cloud on the shared page/export conventions
  - [x] `projections/plot_proj_timeseries.py` — combined annual panels and
        horizon-specific monthly panels on the shared conventions
  - [ ] `shared/func_plot_signature.py` (`plot_signatures`, `plot_hydro`,
        `plot_basavg` — `plot_clim` is gone, ADR 0006)
  - [ ] `dev/scripts/basin_map_example.py`
  - [ ] `dev/scripts/preview_basin_map.py`

**Two traps this item must respect**, both already paid for once:

1. `shared/cartographic_map.py` and `shared/plot_map.py` are contract surfaces —
   rule 1.12's basin map and rule 1.13's three forcing maps both draw through
   them. AGENTS.md § *Figures are terminal artifacts* is explicit that a shared
   helper edited in service of a plot is NOT a figure-only change and takes the
   full validation ladder. A template module every plot imports is that, maximally.
2. Anything assembled from the tunable block must be derived in a FUNCTION, not
   frozen into a module constant — a constant snapshots its inputs at import, so
   `preview_basin_map.py --set` would silently do nothing.

## Refs

- `dev/decisions/0006-retire-subcatchment-climate-plots.md`,
  `dev/decisions/0007-draw-basin-area-from-the-spatial-foundation.md` — the two
  rulings taken during the map sweep. 0007's cost is tracked as [[t2608091730]].
- `dev/scripts/preview_plots.py`, `dev/scripts/preview_basin_map.py` — render a
  family without a workflow run. The figure gate is: render it, publish the PNG
  as an Artifact, look at it. Never byte-compare, never run the baseline.
- Closed as superseded during this work: `t2608091028` (see `dev/LOG.md`) — its
  signature-figure redesign had already landed, and ADR 0006 then retired the
  figure. Relevant here because the sweep reaches `func_plot_signature.py`.
