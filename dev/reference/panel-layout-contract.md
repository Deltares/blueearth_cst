# Physical panel-layout contract

This contract applies to opted-in regular diagnostic grids in ggplot2 and Matplotlib.
Maps, wavelet composites and explicitly configured layouts retain their own geometry.

## Inputs and selection

Lengths are millimetres; fonts are points. The weather profile uses 70 × 70 mm plotting rectangles,
ordered column candidates `[3, 2]`, a 180 mm maximum complete width, and row-major order.
One panel uses one column and omits its order label and heading. Keep its units in an axis label.
For multiple panels, exclude candidates exceeding the panel count, then try the remaining candidates
in their declared order. Rows equal `ceil(panel_count / columns)`. Never fall back to a one-column
stack, shrink geometry or typography, omit panels, or crop decorations to force a fit.

Measure each candidate's complete canvas with the target fonts and renderer. Count axis labels, ticks,
panel headings, gaps and legends. Select the first measured width no larger than the limit plus
0.000001 mm comparison tolerance. Keep unused trailing cells empty. A no-fit result is an explicit
error with guidance to increase the limit or reduce the requested geometry deliberately.

The adapters return columns, rows, occupied cell assignments, plotting-area dimensions and measured
canvas dimensions. R cell indices are one-based; Python indices are zero-based. Both fill rows left
to right and then down. Do not confuse square physical axes with equal scientific axis scales.
Finite dimensions must be positive, panel counts positive integers, and multi-panel column candidates
distinct integers of at least two. Reject empty panel sets.

## Implementation boundaries

Upstream R: `weathergenr::arrange_weather_panels()` accepts single-panel ggplots and stores measurements
on its returned patchwork. `export_weather_plot()` uses that stored geometry when dimensions are omitted.
Python: `blueearth_cst.plotting.panel_layout.arrange_panels()` accepts axes drawing callbacks over
prepared data and returns the figure, occupied axes and layout measurements. It uses explicit axes
positions and at most eight measurement passes; it does not mix constrained and tight layout engines.

Both adapters currently measure panel-owned decorations, including axes legends. Figure-level shared
captions and legends must be reserved by the owning renderer before export; they are not automatically
allocated by the Python adapter. The weather gallery keeps captions outside the exported PNGs. Do not
add figure-level furniture or tight-crop an already measured figure and claim its stored dimensions
are still valid. This is an explicit adapter boundary for later figure-family adoption.

Default R behavior preserves repeated y titles; the compact weather profile explicitly requests only
the first column's y title while retaining tick values. The Python caller controls axis-label repetition
in its drawing callbacks. Local scientific axis-sharing decisions remain with each renderer.

## Conformance

`tests/fixtures/panel_layout.csv` supplies measured-width planner cases shared byte-for-byte with
weathergenr's `tests/testthat/fixtures/panel-layout.csv`. Given the same measured widths, planners
must choose the same rows and columns. Actual font metrics can differ between engines, especially
near a width boundary; pixel identity and identical complete canvas sizes are not required.

At 180 mm, ordinary 70 mm panels with fitting decorations arrange 1 as 1×1, 2 as 2×1, 3–4 as 2×2,
5–6 as 2×3 and 9 as 2×5 (columns × rows). At 270 mm, nine panels use 3×3 when decorations fit.
Verify these choices, row-major assignments, no-fit errors, stable font/mark sizes, and actual saved
panel geometry. Raster measurements allow 0.12 mm for edge/stroke/pixel quantization; internal physical
layout measurements use 0.001 mm tolerance. Preserve plotted values and RNG state.

## Ownership and adoption

CST owns this language-agnostic specification and its configuration translation. Upstream owns the R
implementation and standalone API documentation; neither adapter depends on the other's runtime.
Presentation tokens remain outside scientific configuration identity. Weather-specific overrides live
in `config/plotting/weathergenr.yml`, layered over the shared chart tokens. Spatial-map configuration
and the shared style reference remain with session-3 during the concurrent map revision.

Initial adoption is weather diagnostics plus isolated Python conformance renders. Existing Python
production figure families have not been migrated automatically. Their owners should opt in when
their geometry and decoration needs match this contract.
