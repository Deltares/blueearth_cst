---
title: Hoist the categorical palette out of compare_sources when a second caller appears
type: watch-item
status: backlog
effort: 1
area: wf0 / figures + shared plot style
origin: 2026-08-17 wf0 rule 0.06
created: 2026-08-17
updated: 2026-08-17
---

> [!note] Overview
> **What** — `compare_sources.SOURCE_COLORS` is the toolbox's only qualitative palette, and it lives in the module that uses it rather than in `shared/plot_style.py`.
> **Why** — Deliberate, not an oversight: one caller does not justify a shared constant, and editing `plot_style.py` escalates a figure change to the full validation ladder (`AGENTS.md`, "Figures are terminal artifacts"). But a second caller would make the local copy the wrong home.
> **Trigger** — A second figure family needs to colour by CATEGORY rather than by quantity.

## What exists today

Every other figure in the toolbox colours by QUANTITY, through
`cartographic_map.RASTER_STYLES` — a precipitation ramp, a temperature ramp.
Rule 0.06's comparison figures are the first that colour by CATEGORY: one line
per climate source, where the colour carries identity and nothing else.

The palette is Okabe-Ito, chosen because it stays distinguishable under all
three common dichromacies — the same accessibility standard
`cartographic_map` already applies to its sequential ramps (see the note above
`RASTER_STYLES` on dichromacy testing, measured 2026-08-09).

Assignment is by DECLARATION ORDER, so `shared.clim_historical` always takes the
first colour. That is a property worth preserving in any hoist: the project's
own source should not change colour because a candidate was added ahead of it in
the config.

## Why it is not in `plot_style.py` yet

Two reasons, and the second is the load-bearing one:

1. One caller. A shared constant with a single consumer is indirection, not
   reuse.
2. `shared/plot_style.py` is a contract surface with other callers, so editing
   it is NOT a figure-local change — `AGENTS.md` is explicit that a shared helper
   edited in service of a plot takes the normal validation ladder. Rule 0.06's
   own module can be changed under the figure gate alone.

## Trigger

A second family needs categorical colours — a per-model WF2 series, a per-gauge
evaluation panel, a per-realization WF3 plot. At that point:

- move `SOURCE_COLORS` to `shared/plot_style.py` under a name that says what it
  encodes rather than which caller uses it (`CATEGORICAL_COLORS`, not
  `SOURCE_COLORS`);
- keep the declaration-order property, and state it where the constant lives;
- take the full ladder for the move, since `plot_style.py` is shared.

Until then this is correct as it stands, and the note exists so the next author
finds the reasoning rather than re-deciding it.

## Refs

- `blueearth_cst/climate_analysis/compare_sources.py` — `SOURCE_COLORS`, whose
  docstring points here.
- `blueearth_cst/shared/cartographic_map.py` — `RASTER_STYLES`, the
  colour-by-quantity side.
- [[figure-revision-gate]] — how a figure change is verified.
