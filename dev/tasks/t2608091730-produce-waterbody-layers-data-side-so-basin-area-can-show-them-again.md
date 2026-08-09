---
title: Produce waterbody layers data-side so basin_area can show them again
type: todo-item
status: backlog
effort: 2
area: plotting
queue:
created: 2026-08-09
updated: 2026-08-09
---

> [!note] Overview
> **What** — Lakes, reservoirs and glaciers reach `models/hydrology/wflow/staticgeoms/` from rule 1.08 `add_reservoirs_lakes_glaciers`, a MODEL rule. The shared spatial foundation (`data/spatial/geoms/`) carries none of them. ADR 0007 moved `basin_area` onto that foundation, so the figure no longer draws waterbodies at all. Fix: produce those layers data-side, from the same catalog source, in rule 1.03/1.06 territory, and have rule 1.08 CONSUME them rather than produce them. Then add `lakes`/`reservoirs`/`glaciers` back to `SPATIAL_MAP_LAYERS` in `shared/plot_map.py` — `plot_raster_map` already accepts all three and needs no change.
> **Why** — Recorded as the known cost of ADR 0007 rather than left implicit. On a basin with a major reservoir, `basin_area` is arguably the one figure that must show it, and today it silently would not. The obvious shortcut — have rule 1.08 also write into `data/spatial/geoms/` — is WRONG and is why this is its own item: a model rule writing into `data/spatial/` makes that tree model-dependent, which re-couples `basin_area` to the model build and undoes the move. The producer has to relocate, not fan out.
> **Effort** — medium. One producer relocation plus a consumer change in 1.08; the figure side is a three-line dict edit.

## Progress

- [ ] Decide which rule owns the data-side waterbody layers (1.03 delineate_spatial_units vs 1.06 prepare_spatial_maps)
- [ ] Relocate the producer and repoint rule 1.08 to consume
- [ ] Add the three layers to `SPATIAL_MAP_LAYERS`; verify on a basin that HAS a reservoir
- [ ] Note the closure in ADR 0007's consequences

## Refs

- `dev/decisions/0007-draw-basin-area-from-the-spatial-foundation.md` — records the move and names this as the gap.
- `blueearth_cst/shared/plot_map.py::load_spatial_basin_layers` — its docstring points here.
- The test fixtures have no waterbodies, so a green suite says nothing about this. Verify on a real basin with one.
