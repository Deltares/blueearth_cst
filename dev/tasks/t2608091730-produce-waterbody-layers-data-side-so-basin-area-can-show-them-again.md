---
title: Produce waterbody layers data-side so basin_area can show them again
type: todo-item
status: blocked
effort: 1
area: plotting
queue:
created: 2026-08-09
updated: 2026-08-11
---

> [!note] Overview
> **What** — `basin_area.png` is the study-area map: elevation, the basin outline, rivers and gauges. It draws no lakes, reservoirs or glaciers, because the only thing that produces those layers is rule 1.08 `add_reservoirs_lakes_glaciers`, which runs inside the wflow model build and writes into the model's own `staticgeoms/`. The figure was moved off the model onto the shared spatial foundation (`data/spatial/`, ADR 0007), and that foundation carries no waterbodies. Fix: add a data-side producer that clips the same three catalog sources (`hydro_reservoirs`, `hydro_lakes`, `rgi`) to the basin and writes them into `data/spatial/geoms/`, then have the figure draw them.
> **Why** — On a basin with a major reservoir, the study-area map is arguably the one figure that must show it, and today it silently would not. The obvious shortcut — letting rule 1.08 also write into `data/spatial/` — is WRONG and is why this is its own item: a model rule writing there makes the shared foundation model-dependent, so the figure could no longer be drawn before a model exists, which is the property ADR 0007 bought.
> **Effort** — Small now that the design is settled (2026-08-11 ruling below): one new producer following the `rivers` precedent, plus a three-line dict edit on the figure side. Rule 1.08 is not touched at all, so the model build and the baseline cannot move. The one remaining design question is which rule owns the producer, and it is not a free choice — see Open question below.

## Progress

*Blocked — the three source datasets are not on this machine. `C:\data\wflow_global\hydromt\hydrography\` holds only `rivers_lin2019`. Without them no producer can emit a layer and the figure gate (render it and look at it) cannot be met. Unblocks when that data root is restored.*

**The blocker is older than this note said, and the log evidence for "they were
there on 2026-08-10" does not hold (corrected 2026-08-12).** hydromt logs the
RESOLVED URI before opening it, and then treats a missing source file exactly
like an empty result. Today's WF1 run shows all three "Reading …" lines followed
by `Skipping method, as no data has been found`, with the files provably absent:
`hydrography/` has a LastWriteTime of 24 July, and a recursive search of
`C:\data` finds no `reservoir-db.gpkg`, `lake-db.gpkg` or `rgi.gpkg` anywhere.
So a "Reading X from Y" line is not evidence that Y exists. See
`t2608121606` — that conflation is a defect in its own right, and it is why the
producer written here must fail loudly on a missing SOURCE while writing an
empty layer for an empty RESULT.

- [x] Decide whether the figure shows physical or modelled waterbodies — owner ruling 2026-08-11: **physical, unfiltered**. Rule 1.08 keeps naming the catalog sources exactly as today and is not modified
- [x] Decide which rule owns the data-side producer — owner ruling 2026-08-12: **1.03 `delineate_spatial_units`**
- [ ] Write the producer: clip the three sources to `basins`, write `geoms/{reservoirs,lakes,glaciers}.geojson`, register them in `spatial_catalog.yml` beside the existing layers
- [ ] Add the three layers to `SPATIAL_MAP_LAYERS` in `shared/plot_map.py` — `plot_raster_map` already accepts all three keys and needs no change
- [ ] Verify by rendering `basin_area` on a basin that HAS a reservoir; the fixture basin has none, so a green suite says nothing
- [ ] Correct ADR 0007's consequences, which still describe the rejected "1.08 consumes them" plan

## Which rule produces them — RULED 2026-08-12: 1.03

**`1.03 delineate_spatial_units`**, and the deciding argument is not the one this
section framed it on. Consistency-versus-read-cost was never a fair trade,
because the WF1-only option reproduces the exact failure mode this whole task
exists to remove: `data/spatial/geoms/` would hold a different layer set
depending on which workflow last wrote it, while the figure reads that directory
by name — so a WF2-first project silently draws a study-area map with no
waterbodies, which is where we came in.

Two facts settled it, both measured on 2026-08-12:

- **The blast radius is real.** `SPATIAL_UNITS = spatial_units_rule(...)` is
  defined in all three Snakefiles (`Snakefile_climate_projections:228`,
  `Snakefile_climate_experiment:198`, and WF1) and splatted into a rule in each,
  so producing waterbodies in 1.03 does produce them in WF2 and WF3.
- **The read cost cannot be measured on this machine, and no number should be
  quoted from it.** Not only are the three sources missing — the local `rivers`
  source is a 0.5 MB test extract, so 1.03's measured 33.4 s says nothing about
  what three real global geopackages would cost. Expectation, explicitly not a
  measurement: the clips are geometry-filtered and index-accelerated, so cost
  scales with features-in-basin rather than file size. Confirm on real data
  before anyone relies on it.

## Original framing — kept for the record

Not a free choice, because the obvious home is shared:

- **1.03 `delineate_spatial_units`** already does exactly this for rivers
  (`spatial/products.py:726`, `catalog.get_geodataframe(rivers_source, geom=basins)`)
  and already writes `data/spatial/geoms/`. But its inputs/params/outputs are
  splatted from one `SPATIAL_UNITS` definition into **2.03 and 3.04 as well**, so
  producing waterbodies here produces them in WF2 and WF3 too, and charges every
  workflow the catalog read.
- **1.06 `prepare_spatial_maps`** is WF1-only, which avoids that cost — but then
  `data/spatial/geoms/` holds a different set of layers depending on which
  workflow last wrote it, and the figure reads that directory by name.

Consistency probably wins over the read cost, but the cost has not been measured
and the WF2/WF3 blast radius has not been checked.

## Refs

- `dev/decisions/0007-draw-basin-area-from-the-spatial-foundation.md` — records the move and names this as its known cost. Its consequences still say the fix is to have 1.08 consume the layers; the 2026-08-11 ruling supersedes that, and the ADR needs the one-line correction listed above.
- `blueearth_cst/shared/plot_map.py::load_spatial_basin_layers` — its docstring points here.
- **`lakes` is a stale name model-side.** hydromt_wflow 1.0.2 has no `lakes` geom: `setup_lakes` became `setup_reservoirs_no_control`, and the geoms it writes are `meta_reservoirs_no_control`, `meta_reservoirs_simple_control` and `glaciers`. Data-side the names come from the SOURCES instead (`hydro_lakes` → `lakes`, `hydro_reservoirs` → `reservoirs`, `rgi` → `glaciers`) and are physically meaningful — a further argument for drawing the figure from the foundation rather than from the model's vocabulary.
- Rule 1.08 does far more than emit vectors: it derives rating curves, storage curves and demand parameters onto `staticmaps.nc`. Leaving it untouched is what keeps this task off the baseline.
