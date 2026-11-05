# Measured panel layout

Last updated: 2026-11-05

Use this contract for regular multi-panel ggplot2 or Matplotlib figures that need fixed physical
plotting areas. It specifies candidate selection, measurement and comparison across renderers.
Maps, wavelet composites and figure families with explicit geometry keep their own layout rules.

## Inputs and grid selection

Supply a nonempty panel set, plotting-area width and height, ordered column candidates, a maximum
complete canvas width, and the target fonts and renderer. Lengths are in millimetres; fonts are in
points. Panel counts must be positive integers, dimensions positive and finite, and column candidates
distinct integers of at least two. Reject invalid inputs before drawing.

For one panel, use one column and omit the order label and heading; retain any units in an axis label.
For multiple panels, discard candidates with more columns than panels. Measure the remaining candidates
in their stated order, with `ceil(panel_count / columns)` rows. Fill cells left to right and then down;
unused cells in the final row stay empty. Select the first candidate whose measured complete width is
at most the limit plus 0.000001 mm comparison tolerance.

Measure with the target renderer and fonts after drawing axis labels, ticks, panel headings, gaps and
legends. Do not fit by shrinking panels or typography, dropping panels, cropping decorations, or
falling back to a one-column stack. If none fits, raise an explicit error directing the caller to
increase the width limit or deliberately reduce the requested geometry.

The layout result contains columns, rows, occupied cell assignments, plotting-area dimensions and
measured canvas dimensions. Square physical axes do not imply equal scientific axis scales.

### Illustrative geometry and Python defaults

With 70 × 70 mm plotting rectangles and column candidates `[3, 2]`, a 180 mm canvas limit gives
1 panel as 1 × 1, 2 as 2 × 1, 3–4 as 2 × 2, 5–6 as 2 × 3 and 9 as 2 × 5 when decorations fit
(columns × rows). At a 270 mm limit, nine panels use 3 × 3 when decorations fit. These are examples
of the selection rule, not required dimensions for other figure families.

The current Python `arrange_panels()` defaults are 70 × 70 mm panels, candidates `(3, 2)`, a
180 mm maximum width, 300 DPI and no single-panel heading. Callers may pass other valid values.
The examples above do not establish ggplot2 defaults or a general R API.

## Renderer responsibilities

The [Python implementation](../panel_layout.py) accepts drawing callbacks over prepared data and
returns a Matplotlib figure, occupied axes and layout measurements. It places axes explicitly and
allows at most eight measurement passes; it does not combine constrained and tight layout engines.
It rejects non-callable callbacks, invalid DPI and a non-Boolean single-panel-heading flag. It raises
an error if fixed-aspect axes change the requested physical size or decoration measurements fail to
converge. Python cell indices are zero-based. `single_panel_heading=True` is an explicit exception to
the default heading omission.

For ggplot2, measure with the graphics device and fonts used for export. A ggplot2 integration owns
its own arrangement and export behavior; this repository does not provide a standalone generic R
adapter. R cell indices are one-based. Whether to repeat axis titles or share scientific scales is
a figure-family choice; Python callers make label choices in their drawing callbacks.

Measure panel-owned decorations, including axes legends. The figure owner must reserve space for
figure-level captions and shared legends before export. Do not add labels or legends, or tightly
crop a measured figure while claiming its stored dimensions still apply.

## Compare implementations and exports

Given the same measured candidate widths, R and Python planners must choose the same columns, rows
and row-major occupied cells. Keep their planner fixture rows byte-for-byte identical. The tracked
[Python planner fixture](../../../tests/fixtures/panel_layout.csv)
and [Python tests](../../../tests/test_panel_layout.py) cover representative choices, no-fit errors
and physical axes geometry. Renderer font metrics can differ, especially near a width boundary;
pixel identity and identical complete canvas dimensions are not required.

Check that fonts and marks retain their physical sizes, panel assignments follow row-major order,
no-fit errors are explicit, and saved plotting areas keep their requested dimensions. Raster
measurements allow 0.12 mm for edge, stroke and pixel quantization; internal physical layout
measurements use 0.001 mm tolerance. Preserve plotted values and random-number-generator state.
