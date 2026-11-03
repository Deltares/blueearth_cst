# Reviewed weather diagnostic profile

The weather export adapter applies `weathergenr::style_weather_plot()` to returned diagnostics.
Generator/evaluation defaults remain unchanged upstream. CST reads `config/plotting/style.yml`,
`colors.yml` and the weather-only `weathergenr.yml` at render time, after scientific identity is resolved.
No YAML parser, CST path, or CST dependency is introduced upstream.

## Approved conventions

- Observations and agreement guides are blue; stochastic traces/ranges are gray; central ensemble
  summaries are black. Significance thresholds and cone-of-influence guides are red. The wavelet
  significance contour stays black and its observed spectrum retains the reviewed plain blue.
- Use 7 pt labels/headings and 6 pt tick labels from shared chart tokens, with DejaVu Sans where
  configured. Explicit partial-theme entries win. Fonts and marks retain physical sizes across grids.
- Facets use white, left-aligned lettered headings, horizontal major gridlines and darker borders.
  A single panel omits its heading and letter. Follow [the panel contract](panel-layout-contract.md).
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

All 15 original filenames remain unchanged; exports overwrite the matching PNGs. CST owns export DPI
and non-facet canvas dimensions. Faceted diagnostics use measured stored dimensions. Captions in the
comparison HTML identify colors and summary meanings; they remain outside the exported plot objects.

CST installs `weathergenr` version 2.1.0 from published commit
`b023e65e5237f681cb96178754488bba84afb2ae`. The planned `v2.1.0` tag is not published;
the installer uses this immutable commit and verifies both `Version` and `RemoteSha` before
accepting an existing installation. Run `pixi run install-rdeps` after updating the toolbox.
The R package source pin is maintained by `dev/scripts/install_weathergenr.R`, outside `pixi.lock`.

The pinned commit contains the approved plotting implementation
`2b2f602cb3f6500fb69e014d7c6650be357f5370`, including the return/style/export interface.
The shared Python map-style work remains separate from this weather adapter.
