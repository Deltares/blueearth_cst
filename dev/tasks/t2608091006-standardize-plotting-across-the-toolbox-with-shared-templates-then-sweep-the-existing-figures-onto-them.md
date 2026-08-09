---
title: Standardize plotting across the toolbox with shared templates, then sweep the existing figures onto them
type: todo-item
status: backlog
branch: feat/plotting-standardization
effort: 2
area: plotting
queue:
created: 2026-08-09
updated: 2026-08-09
---

> [!note] Overview
> **What** — Two halves. (1) Introduce shared plotting templates/style primitives so figures across the toolbox agree on typography, palette, figure size, colourbar placement and export settings, instead of each module deciding independently. (2) Sweep the existing figure producers onto them. Current plotting surface -- 8 modules under blueearth_cst/ plus 2 dev scripts. climate_analysis/climate_figures.py, climate_analysis/plot_climate_source.py, model/plot_results.py, projections/get_change_climate_proj_summary.py, projections/plot_proj_timeseries.py, shared/func_plot_signature.py, shared/plot_map.py, shared/snake_utils.py (plotting primitives), dev/scripts/basin_map_example.py, dev/scripts/preview_basin_map.py. No R-side plotting exists.
> **Why** — Owner request 2026-08-09. Figures are the deliverable a reader actually sees, and they are styled per-module today, so one basin assessment ships figures that do not look like each other. Design before implementing -- a shared template module is a contract surface every plot imports, and AGENTS.md is explicit that a shared helper edited in service of a plot is NOT a figure-only change. It takes the full validation ladder, while the figures themselves are verified by rendering and looking at them.
> **Effort** — large

## Progress

- [ ] <first step>
