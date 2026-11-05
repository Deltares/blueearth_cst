# Styling weathergenr figures in CST

Last updated: 2026-11-05

Use this reference when styling or exporting ggplot2 figures returned by weathergenr in CST. It covers
appearance, panel layout and export settings for those figures, rather than a general style for all
weather diagnostics. The
[weather-generator interface](../../../dev/reference/toolbox-weather-generator-interface.md) describes
generation inputs, outputs and replacement requirements.

The [CST adapter](../../weathergen/plot_style.R) calls `weathergenr::style_weather_plot()`
on returned plots. It reads [style](../../../config/plotting/style.yml),
[colors](../../../config/plotting/colors.yml), and the weather-only
[overrides](../../../config/plotting/weathergenr.yml) at render time, after scientific identity is resolved.
These presentation settings do not enter generator or scenario identity.

## Figure conventions

- Observations and agreement guides are blue; stochastic traces/ranges are gray; central ensemble
  summaries are black. Significance thresholds and cone-of-influence guides are red. The wavelet
  significance contour stays black and its observed spectrum retains plain blue.
- Use 7 pt labels/headings and 6 pt tick labels from shared chart tokens, with DejaVu Sans where
  configured. Explicit partial-theme entries win. Fonts and marks retain physical sizes across grids.
- Facets use white, left-aligned lettered headings, horizontal major gridlines and darker borders.
  A single panel omits its heading and letter. Follow
  [the panel contract](panel-layout.md).
- Comparison medians use gray-filled, black-outlined 1.9 mm circles (fill alpha 0.30, stroke 0.30 mm).
  Gray ranges use 0.80 mm and alpha 0.85; the blue 1:1 guide is 0.30 mm behind the data. Each panel
  repeats its observed x-axis title. Medians are not renamed as means.
- Boxplots retain their existing statistical definitions and mean diamonds. Observations use blue;
  simulations use gray fills and black outlines. Outlier dots are 1.3 mm at alpha 0.25.
- Annual time series use 0.25 mm gray realization traces at alpha 0.60 and 0.45 mm opaque blue
  observations. Keep only horizontal gridlines, annual minor ticks and five-year major labels.
  Axis labels are `Serial year` and `mm day⁻¹`; the first data value is not forced to zero.
- Monthly cycles use the same trace/observation weights, separate precipitation/temperature units
  in the panel headings, and a `Month` title below each panel. No ensemble mean is added.

## Panel layout in CST

The weather profile sets 70 × 70 mm plotting rectangles, preferred columns `[3, 2]` and a
180 mm maximum complete width in `weathergenr.yml`. The CST adapter passes these settings and
the raster DPI from `style.yml` to `weathergenr::style_weather_plot()`. The generic
[panel contract](panel-layout.md) governs candidate measurement,
fit, dimensions and comparison across renderers.

The pinned weathergenr API uses `arrange_weather_panels()` to arrange ggplot2 panels and attach measured
layout dimensions to the patchwork; `export_weather_plot()` uses those dimensions when width and height
are omitted. The CST adapter passes explicit dimensions for non-faceted plots and uses the stored
dimensions for faceted plots. CST provides configuration translation, not a standalone ggplot2 adapter.
The R and Python implementations do not call each other's runtime.

Default R layout retains repeated y-axis titles. The compact weather profile keeps the y-axis title
only in the first column and retains tick values. Captions remain outside the exported PNGs; the
gallery carries their colour and summary explanations. Weather diagnostics use this measured layout;
Python adoption consists of isolated conformance renders; existing production figure families do not
automatically use this weather profile.

## Preserved figure exceptions

Observed wavelet power keeps its heatmap colors, red dashed guides, black significance contour and
aligned period coordinates across the two panes. The right pane has no y-axis labels/ticks; both
x-axis titles share a baseline, and the panel gap is 1.5 mm. Zero is not added to the period scale.

WARM annual statistics retain four narrow panels in one row, with gray outlined realization points,
blue zero agreement guides and horizontal dodge used only to separate realizations. The WARM global
spectrum uses a light-gray quantile band, black mean, blue observations and red dashed significance;
its axes are `Period (years)` and `mm² day⁻²`. Its vertical extent is derived from the supplied data,
not a basin-specific fixed upper bound.

## Delivery and integration

The 15 diagnostic filenames remain unchanged; exports overwrite the matching PNGs. CST owns export DPI
and non-facet canvas dimensions. Faceted diagnostics use measured stored dimensions. Captions in the
comparison HTML identify colors and summary meanings; they remain outside the exported plot objects.

The [installer](../../../dev/scripts/install_weathergenr.R) owns `required_version` and `required_sha`
and checks installed `Version` and `RemoteSha`. Run `pixi run install-rdeps` to install that pinned
source. The pin is outside `pixi.lock`; the upstream source itself is not part of this repository.
Shared Python map styling remains separate.
