# CST scientific plot style

Last updated: 2026-11-08

Use this reference for CST figure appearance and interpretation in WF0, WF1 and WF2. Shared rules apply
unless a figure-specific exception says otherwise. Plotting changes must preserve calculations and
statistical meaning. The [plotting README](../README.md) owns prepared drawing, presets and export;
the [panel contract](panel-layout.md) owns fixed Matplotlib and ggplot2 grid geometry;
[figure naming](figure-naming.md) owns filename grammar; and
[weathergenr styling](weathergenr-style.md) owns its ggplot2 figures in CST.

Descriptions without displayed titles or footers assume `plotting.annotation: none`. `caption` and
`title` add text outside the existing plotting canvas.

## Output paths

- Write graphics under `plots/`. WF0–WF2 retain readable `plots/figure_titles.md` and a
  `figure_titles.json` inventory beside the caption JSON. Title keys match ordinary caption keys.
  Every title names subject and relevant source, period or support. WF0 uses
  `[specific plot title] — [dataset] ([start year]–[end year])`; comparisons name contributing datasets.
  Methods and spatial support remain in captions and metadata.
- `plotting.annotation` selects `none` (default), `caption` or `title` at ordinary PNG paths. Keep
  title and caption inventories in every mode. R/weathergenr figures are excluded. Spatial maps and
  legacy climate adapters without caption inventories still write title files.
- Keep readable captions in `plots/figure_captions.md`. Put machine manifests and internal analysis
  records in the output unit's single root `_engine/` under the
  [output naming rule](../../../dev/reference/dev-naming-conventions.md#generated-outputs-under-project_dir).
  WF2's manifest paths resolve relative to its ensemble `plots/`; existing WF0 `plots/_engine/`
  awaits migration. User-facing JSON stays with its data product.
- Generated names use lowercase snake_case, including `full_period/` and
  `period_<start>_<end>/`. WF2 source-specific names stay in
  `projections/projection_figures.FIGURE_FILENAMES`; the [figure naming reference](figure-naming.md)
  owns shared rules and the workflow profiles documented so far.

## Shared colour defaults

Observed water-balance diagrams retain every available station, with station names and matched-year
counts, adaptive axis limits and water/energy bounds. Unavailable stations are listed with reasons.
Axis labels name both ratios because their meaning is otherwise ambiguous. No reference curve is fitted.

- [`colors.yml`](../../../config/plotting/colors.yml) owns the categorical cycle, dataset and
  evaluation colours, SSP colours, climate-map colormaps and precipitation-anomaly colormap.
  `shared.plot_colors` resolves it relative to the toolbox, validates structure and hex colours, and
  reads it once per process. Restart an interactive Python session after editing. Matplotlib resolves
  colormap names at draw time.
- [`style.yml`](../../../config/plotting/style.yml) owns page/export defaults, typography, shared
  chart dimensions, lines, markers, bands, label backgrounds, `maps` geometry and evaluation-sheet
  appearance. Map feature and stream-order colours remain in `colors.yml`; both reach drawing code as
  `Preset.maps` tokens. `shared.plot_setup`, `shared.plot_preset` and `shared.plot_style` bridge these
  tokens to CST drawing. Figure-specific geometry stays with its renderer.
- The cycle starts blue, Terracotta, green, amber, violet, rose, cyan and slate; YAML owns exact values
  and order. ERA5 uses blue; CHIRPS and simulations use Terracotta. Time-series observations use gray
  under the template below; WF1 station evaluation retains blue dashed observations. Named dataset and
  evaluation entries reference cycle colours. Assign unknown datasets remaining colours in config order,
  skipping named colours already present; change marker shapes on repeat. The palette has no certified
  colour-vision-deficiency claim, so retain markers and labels; data time series use solid lines.
- Both shared Matplotlib contexts apply the cycle. Semantic presets take precedence: IPCC scenario
  colours, climate-map and anomaly colormaps keep their values. `historical_maps` retains WF0/WF1
  reference ramps; `climate_maps` retains the legacy adapter's colormaps. Colours do not alter
  classification, normalization, data or scientific configuration identity.
- WF0–WF2 rules digest both YAML files in renderer params so edits schedule plots; WF1's combined
  evaluation/plotting job also refreshes its indicators. The historical map-reference producer tracks
  them as display dependencies without changing climate extraction or simulation identity. WF3's R
  adapter reads shared tokens and weather-only overrides at render time, outside scenario identity.
  Weather-specific appearance and filenames remain in the
  [weather guide](weathergenr-style.md).

## SSP scenario colors

- WF2 uses the IPCC AR6 palette for eight ScenarioMIP SSPs and LowNTCF. Key by CMIP6 experiment ID;
  SSP family and RCP palettes differ. [`colors.yml`](../../../config/plotting/colors.yml) holds the
  mapping, re-exported by `projections/projection_figures.py`. Unknown experiments keep literal labels
  and a muted fallback colour.

## Typography, axes and layout

- Design WF2 for 4–5 scenarios, 10–20+ models and possibly multiple members per model. Wrap panels,
  adjust member-trace opacity and retain clear summaries. Never silently omit models or scenarios.
- Use 7 pt axis/panel titles and 6 pt ticks/annotations in current scientific figures. Preserve
  physical font and mark sizes across canvases; scale captions and legends together in previews.
  Use modest white padding. Square plotting areas have equal physical axis lengths, not equal data scales.
- Give each labelled tick a visible mark. Annual series retain yearly ticks and, where appropriate,
  annual minor ticks; reduce labels on long records. Anchor major/minor ticks to the same calendar grid.
  Long WF2 records use 20-year major labels with annual minor ticks. Comparable panels share limits and ticks.
- Axis titles show units alone, except WF2 reference-relative axes:
  `Relative to [reference-period] (symbol)`, using the actual period, e.g.
  `Relative to 1985–2014 (°C)` or `Relative to 1985–2014 (%)`. Absolute axes use units alone.
  Abbreviate variables as `Precip.`, `Temp.` and `PET`. Display `CHIRPS` in labels and legends; keep
  version details in source records.
- Keep grids subtle. Label multiple panels `a)`, `b)`, etc. and explain them in captions. Default to
  no displayed figure title. In `title` mode add it above the figure without changing plotting area;
  preserve agreed panel labels in every annotation mode.

## Legends, labels and points

- Default to a stacked legend in the less crowded upper corner inside the axes. Use white background,
  alpha 0.88 and no border. In-plot labels use the same background with padding 1.2 and no edge.
- Trend-interval, SPI-fit and trend-sensitivity legends sit outside to the right,
  top-aligned with the panels. Narrow panels as needed without changing overall canvas width.
- Keep slope labels off their lines and point labels clear of map features. Maps prefer `station_name`;
  missing/blank names fall back to `wflow_id`. Keep IDs in data and model output columns.
- Use contrasting point colours over maps. Scatter points have translucent fills and dark outlines;
  enlarged points use area 36 pt², alpha 0.7 and about 0.7 pt dark edges unless density requires
  adaptation. CHIRPS is Terracotta `#CC593D`; ERA5 is blue `#0D38E0`.

## Color scales and colorbars

- Check keys against the drawn map frame after layout, including titles and ticks. Shorten overflow
  and shift keys inward. Reduce categorical text only if the full legend exceeds available space;
  warn if it still cannot fit. Comparable plots share limits; independent scales use the full
  light-to-dark range and visible intermediate values.
- Spatial maps and heatmaps use palettes tied to the plotted variable, independently of dataset identity.
  Use diverging red–white–blue for signed anomalies with a meaningful zero; use sequential palettes for
  non-negative quantities. Climate-index maps use `colors.yml` → `climate_index_maps`: blue for rain
  amount, wet-day counts and wet spells; yellow–brown for dry spells; yellow–red for heat; blue for cold.
  Dataset colours remain for comparison lines and points. Use rectangular colourbar extensions and
  10 pt title padding.
- Monthly precipitation departure calendars use continuous `RdBu` (red dry, blue wet), matching the
  SPI heatmap palette while retaining continuous normalization, centred at
  zero with symmetric limits over all finite pooled anomalies. Use a three-line colourbar title:
  `Precip.`, `anomaly`, `(mm)`. Comparison datasets sit side by side on a 180 × 80 mm canvas,
  with month labels and ticks on the first panel only. Use the full layout width and zero extra
  inter-panel spacing, retaining preset padding and text sizes.
  Omit cell borders and a monthly-percentile companion. Missing cells stay gray. Remove unavailable
  legend rows and explain missing values in captions.
- SPI heatmaps use boundaries `-3, -2, -1, 1, 2, 3`, no zero tick, and a colourbar 80% of the
  plotting-area height. `diagnostic_figures.spi` sets canvas height to 0.56 times its standard stack
  height; this is a code setting, not a comparison to an earlier export.

## Spatial map template

- Geographic raster maps (WF0 source and WF1 forcing climatologies, WF1 spatial figures, basin map)
  use [`prepared_maps.py`](../prepared_maps.py) with `Preset.maps` tokens. Adapters prepare fields,
  classes, linework and basemap features; `shared/map_classes.py` prepares raster classes, relief
  and locator content.

| Element | Treatment |
|---|---|
| Coordinates | Aspect `1/cos(mid-latitude)`; signed ticks; `Longitude (°)` / `Latitude (°)`. |
| Rivers | Blue, width and shade by stream order, thin reaches first, thin white halo. |
| Basin/divides | Black basin outline over dashed divides; shared divide edges drawn once. |
| Locations | Purple diamonds; bold station names with white halo, clear of overlaps and frame. |
| Long names | Split near middle at `maps.location_label_wrap` characters. |
| Furniture | Latitude-corrected km scale bar; no north arrow; locator only on overviews. |
| Locator | Elevation/subbasin overviews; square, in clear corner, ≤40% of frame height. |
| Keys | Side panel, bottom-aligned with frame; colourbar/class title above. |
| Source credit | Caption, never a figure footnote. |

- Fit the map frame within 150 × 95 mm with geographic aspect preserved. Equal geographic frames have
  equal physical plotting areas regardless of keys. Labels, keys and locators extend the canvas,
  without resizing map or typography. Measure after drawing; leave 3 mm outer canvas padding before
  adapter-specific cropping.
- List only classes present in the drawn area and covering at least 0.1% of the basin. Draw smaller
  classes and name them in the caption. Reduce text if a legend exceeds frame height; warn if still
  too tall. Put colourbar units on a separate title line. Use `fundamentals.style_colourbar`, shared
  with heatmaps.

### Spatial map panels

Use `prepared_maps.draw_raster_panels` and `Preset.map_panels` for continuous spatial panel figures;
do not duplicate panel geometry in workflow adapters. `config/plotting/style.yml` → `maps.panels` owns
the shared CST defaults: four columns (three supported), 190 mm figure width, 1.5 mm horizontal gutters,
2 mm between heading/map row blocks, 3 mm outside padding and 4 mm heading space. Frames keep geographic
aspect; figure height follows their aspect and row count. Fill panels left-to-right, then down, without
dropping fields or creating empty map frames in an incomplete last row. `max_panels_per_figure` defaults
to 12. Larger sets split into ordered figures/pages; a partial final page retains identical map sizes,
geographic extent and colour classes. Export all pages with numbered filenames or to a multipage PDF;
captions identify page number and included fields. Never silently truncate excess panels.

For model-by-scenario or other labelled matrices, use `draw_raster_matrix`. It retains explicit row/column
labels, reserves `row_label_space_mm` (30 mm by default), and splits large matrices into rectangular page
blocks. A row never wraps into another model or category; map sizes and scales stay fixed across pages.
Unavailable combinations occupy explicit labelled tiles and count toward the page cap.
When per-tile headings are empty, omit their reserved height so column headings sit close to the maps.

Omit coordinate titles/ticks, repeated scale bars and location names. Retain basin outlines, dashed divides,
rivers and location markers. Scale markers, marker edges, lines, halos and frame borders by the square root
of panel width relative to 150 mm, bounded to 0.5–1 of full-map sizes. At the default twelve-panel layout,
6 pt diamonds become about 3.1 pt, basin lines 0.47 pt, divide lines 0.37 pt and marker edges 0.31 pt.
Month/panel headings use the readable 6 pt tick-font token, not the geometric scale factor. Colours and
stream-order distinctions are preserved. Full-size standalone maps retain their existing style.

Use one vertical colourbar at the right, aligned with the bottom map frame, with title and units above.
The shared preset reserves 16 mm for the key and labels; the bar is 2.3 mm wide and 64% of canvas height.
Use `style_colourbar` for its typography and outline. Keep identical classes across comparable panels and
figures. Prefer outward-rounded whole-number limits/steps when the quantity supports them (as for t003
delta maps); retain scientifically necessary fractional scales. The renderer consumes an explicit scale
and never reclassifies supplied scientific values. Missing-data meaning belongs in the caption.

## Figure-specific decisions

### Annual spatial anomalies

- Optional historical P/T anomaly maps show every comparison reporting year, six per page in at most
  three columns and two rows. Retain trailing empty slots and labelled unavailable years. Shared
  `prepared_maps.draw_raster_panels_detailed` preserves native cells, geographic aspect and map overlays.
- Use absolute departures (mm, °C), symmetric limits over finite cells intersecting each displayed
  native frame across all sources and comparison years, without clipping. Precipitation uses the
  shared precipitation-anomaly palette; temperature uses `climate_maps.temp_crossing_freezing` as
  the shared cool-negative/warm-positive ramp. Missing cells use the shared map missing colour.
  With all finite departures zero, display ±1 in the relevant units and disclose the display fallback;
  with none finite, omit the numerical key. Captions state exact dates, per-cell reference support,
  completeness/infill, native support, counts and unavailable reasons.

### Standard time-series template

- The approved annual precipitation preview defines the standard for data time series and monthly cycles.
  `style.yml` → `charts.time_series` owns 0.65 pt solid lines, 1.3 pt markers, opacity 0.95
  (CHIRPS 0.8), and 1.5 pt fully opaque gray observations. `colors.yml` → `neutral` owns the
  time-series observation gray (`#777777`) and horizontal guide color (`#c7d1d8`).
  WF1 station evaluation is an approved exception: observed discharge retains blue dashed lines
  at the evaluation line width and blue markers.
- Horizontal major guides use 0.5 pt and opacity 0.55, behind the data. Retain all plot borders,
  axis labels and annual minor ticks. Label calendar-year axes `Year`; align major labels to
  five- or ten-year multiples when the span permits. Short records use closer year labels;
  long projection records may use twenty-year labels. Annual precipitation units are `mm/year`.
- Use color, opacity and markers to distinguish sources. Dashed lines remain available for fitted
  trends, thresholds, 1:1 guides and other reference annotations, rather than source identity.
  Preserve ensemble trace opacity and uncertainty envelopes; synthetic serial years, month cycles,
  elapsed time and non-temporal diagnostics retain their scientifically appropriate axis labels.
- `Preset.time_series`, `fundamentals.time_series_style`, `time_series_grid`, `year_ticks` and
  `date_year_ticks` supply the Python template. The weather-generator adapter applies the same
  physical line weights and grid settings to its annual and monthly time-series products.
- Annual location overlays show every complete reporting year available independently for each source.
  Missing or incomplete years remain gaps, without partial-year totals or extrapolation.
  Pairwise agreement statistics continue to use matched periods; an overlay's extent is not its
  agreement sample size.
- Single-dataset temporal plots omit empty leading/trailing periods and retain internal gaps.
  Comparison plots retain their broader display range. Climate-index sheets share a y-axis unit title
  per compatible row and a `Year` title below each occupied column; only labelled year values have ticks.
  Mixed-unit rows retain separate unit labels.

### Climate-index scopes and availability

WF0 no longer emits precipitation monthly-coverage figures, including location and legacy gauge plots.
Coverage and completeness tables remain available for analysis and figure preparation.

- Split annual and configured-season sheets into dry/wet spells, heavy rainfall, rainfall amount and
  frequency, temperature extremes, and heatwaves. Emit only scopes and indices with finite retained
  values; omit unavailable climate-index maps as well. Keep original definitions and numerical values.
- `INDEX_GROUPS` in `indicators/climate/definitions.py` owns membership. `figure_availability.json`
  records omitted metrics and emitted PNGs. `figure_captions.json` and the title inventories describe
  the available set, inside the workflow-owned `plots/indices/` directory.
- These thematic sheets replace the combined `annual_indices_ts` and `seasonal_indices_ts` exports.
  Earlier run outputs remain historical evidence; use the current render inventory when reviewing a
  new render rather than treating old unavailable sheets as current products.
- Export filenames retain dataset and location while shortening diagnostic tokens; thematic sheets use
  `<dataset>_<scope>_<period>_<location>.png`. The [naming profile](figure-naming.md) owns the mapping
  and migration; precise scientific support stays in captions and metadata.

### Complete wet-spell exceedance

The complete wet-spell exceedance product also uses the shared thin solid source line and soft
horizontal guides. Preserve its empirical step geometry, probability scale and complete-spell policy;
do not smooth or add interpolated values.
Dry-spell exceedance uses the same line/step/grid styling, retaining its logarithmic probability axis,
square plotting area and existing 0.002 display cutoff.

### Rain-gauge comparisons

- Use the shared CST preset for coverage, annual totals, monthly climatology and daily agreement.
  Annual overlays use each source's complete reporting-year totals without fitted trends. Monthly climatology
  averages complete matched months; individual and joint daily coverage are shown separately.
- Retain zero amounts, common-day scatter masks, native-cell support and unavailable-period notes.
  Apply the standard time-series template. Captions carry station identity,
  contributing source names, common counts and interval attribution.

### WF1 forcing QA

- Use WF0 map, annual-series and monthly-climatology styles for precipitation, temperature and PET.
  Read actual model-grid forcing, mask to the model basin and use equal-weight cell means for temporal
  figures. Keep the nine `forcing_<variable>_<kind>.png` names. Identify Wflow forcing in legends and
  captions; use ERA5 blue. This is QA, not a change to forcing or simulation calculations.
- Model-grid maps retain their footprint and basin mask, coordinate labels, palettes, overlays and
  gauge labels. WF0 source and WF1 forcing maps share
  `data/climate/historical/_engine/map_reference.json`. The selected source's outer cell edges define
  the reference frame; historical candidate cells intersecting it define pooled classes. Borrowed
  CHIRPS temperature/PET do not contribute.
- Display native sources at basin bounds with spatial-map padding, clipped to native outer cell edges
  to avoid blank space. Masked model-grid maps keep full padding. Compute ticks and midpoint-latitude
  aspect from that viewport; without basin geometry, use the reference frame. Colours do not imply
  equal periods, eligibility or PET methods: state those differences in captions.
- Show both rectangular extensions on every available reference colourbar. Report strict under/over
  counts in captions only when nonzero, but always in map metadata. Include the upper boundary in the
  last class. Missing cells are gray; outside-model and uncovered areas are white. With no finite
  historical scale, hatch finite cells neutrally and omit the numerical colourbar. Never infer a
  scale from model forcing or invent 0–1. Omit fine forcing-cell outlines; outside basin stays white,
  unavailable within-basin summaries gray.
- Use WF0 start-year labels, completeness rules, Sen trends and monthly means with 10–90th percentile
  bands. Allow at most three missing days per month and fifteen per reporting year, even if WF0 was
  omitted. Temperature uses means; precipitation and PET use totals. Put model corrections and
  statistical caveats in separate captions, without plot footers.

### WF1 station hydrographs

- Keep five panels: daily discharge, annual means, seasonal cycle, daily observed–simulated scatter,
  and log flow duration. Mark scatter unavailable when observations are absent; omit linear flow duration.
  Use common type, frames, soft horizontal grids, short titles and an inside legend. Observations are
  blue/dashed at the evaluation line width; simulation thin Terracotta/solid. Vertical axes show discharge units alone; months use three
  letters; exceedance labels show percentages. Annual labels start at the preceding five-year grid
  mark, or ten-year mark for records over 35 years, with annual minor ticks.
- Put station identity, dates, support and methods in separate station captions, bounded by the
  plotting area's left/right edges in previews.
- Annual means require full calendar-year bounds, at most three missing days per month and fifteen
  total. Seasonal cycles average eligible monthly means equally across years; full months allow at
  most three missing days. Reindex daily dates so absent timestamps count as missing. Use finite,
  nonnegative discharge without infilling.
- With observations, summaries use paired valid dates; the daily panel retains available series for
  context. Scatter uses concurrent finite, nonnegative pairs, 1.3 pt points at alpha 0.4, and a dashed
  1:1 reference. Flow-duration positions are `rank / (n + 1)` over valid values. Omit zero from log
  without changing positions and report its frequency. The log range includes
  all positive values from both series, without percentile clipping. These rules do not replace
  performance-indicator methods.

### WF1 performance sheets

- Pair finite, nonnegative observed and simulated daily values after date alignment for scatter,
  counts and exported indicators. Monthly indicators compare means on those paired days;
  hydrograph completeness thresholds do not gate indicator rows. Undefined scores stay unavailable.
- Keep a separate table-only `performance_<loc>.png` at 180 × 76 mm. Metrics are rows;
  Daily, Monthly and Ideal are columns. Preserve supplied values from `performance_indicators.csv`.
  Keep squared Pearson correlation and sample counts in JSON metadata. Daily scatter belongs in
  hydrograph panel d. Use common type and a separate caption, without displayed title or footer.

### WF1 peaks and lows signatures

- Keep three panels, regular-weight inside letters with equal physical top-left offsets, an inside
  daily-series legend stacked inside the upper-right corner, common type/frames/grids and borderless
  label backgrounds. Panel c shows the full simulation record, calendar-year ticks and pale gray bands
  over wet/dry calendar years. Wet years have mean simulated flow strictly above its 75th percentile;
  dry years strictly below its 25th. Use complete calendar years with the hydrograph completeness
  thresholds, independently of observed gaps. Preserve warm-up and observation-overlap selection.
- Frequency and daily axes use discharge units alone; scatter axes identify observed and simulated.
  Panel a uses blue observed and Terracotta simulated + symbols, 5 pt with 0.9 pt strokes and a local
  upper-right legend. Label the probability axis Return period (years), with ticks and vertical guides
  at 2, 5, 10 and 20 years. Preserve ranked empirical positions and annual-extreme calculations;
  no extreme-value distribution is fitted. Scatter retains translucent circles with dark edges.
  Put station, selected years, percentile threshold, methods and caveats in station captions.

### WF1 hydrological output summaries

- Default to two single-axis products per variable: basin-average annual series and seasonal cycle.
  Optional subbasin figures repeat both individually, never as panels. No displayed titles. Rebuild
  the basin native grid-cell mean from daily zonal means with fixed cell-count weights; a missing
  zone invalidates that day rather than changing spatial support.
- Apply first-year warm-up exclusion and calendar completeness to every variable. Eligible months
  allow three missing days; years need full bounds, all months eligible and at most fifteen missing
  days. Fluxes use monthly/annual totals, filling eligible missing days with that month's available-day
  mean. Rates and stores use means without infilling. Negative recharge remains physically valid.
- Seasonal cycles show equal-year monthly means and 10–90th percentile bands: interannual variation,
  not uncertainty of the mean. Annual series use Sen slopes per decade and WF0 trend treatment.
  Snow is native snowpack liquid-water depth in mm, a store rather than accumulated flux.
- Captions name quantity, support, period and encodings. Machine methods metadata retains support
  counts, completeness thresholds and exclusions.

### Cross-workflow and WF0 forms

| Figure type | Treatment |
|---|---|
| Historical/forcing maps | Basin-focused bounds with spatial-map padding. |
| Map coordinates | Degrees in axis titles; signed numeric ticks, no N/S/E/W. |
| Annual series | Trend line and slope on each applicable series; unit-only y-axis. |
| WF2 annual overview | No legend; caption explains member traces, medians and ranges. |
| WF2 trajectories | Model median/range, distinct historical trace, inside stacked scenario legend. |
| WF2 handover | Dashed at historical–scenario handover, 2014.5, not current year. |
| WF2 seasonal grids | 2 × 2 seasons; model points, median tick, IQR band; unit-only y-axis; no legend. |
| WF2 seasonal cycle | Median lines; shade historical range only; one-row legend above axes. |
| WF2 dry periods | Hatch precipitation %/ratio panels or months retaining at most half the models. |
| Monthly climatology | Inside stacked legend in less crowded upper corner. |
| Cumulative timing | Months above; reporting-day numbers below; `Cumulative precipitation (%)`; range/box opacity 0.8, opaque numeric annotations. |
| Timing boxes | Enlarged unfilled horizontal boxes, thin outlines, no mean dot. |
| Drought events | Dataset colours; legend distinguishes `Complete` and `Truncated`. |
| Truncated events | Dense diagonal hatch with thin strokes, not dashed outlines. |
| Spell exceedance | Square plotting area; `Exceedance Pr. (log)` y-axis; compact padded canvas. |
| Autocorrelation | Dark point edges and approximately 95% limits, qualified in caption. |
| Trend intervals | Thick horizontal intervals/caps, close rows, no dataset row labels. |
| Seasonal P–T | Square panels; `Temperature anomaly (°C)` / `Precipitation anomaly (%)`. |
| SPI fit checks | Dataset-hued p-value heatmaps, minimal titles, no `Month` x-axis title. |
| Trend sensitivity | Slope points and confidence bars for each eligible start year. |

- Trend-sensitivity panels use sparse calendar-aligned integer labels, minor ticks at other eligible
  start years, a shared start-year extent and the fixed end year in the x-axis title. Retain every
  slope and interval regardless of label density.

### WF2 seasonal and monthly products

- Seasonal grids for temperature change, precipitation % change and precipitation totals are 2 × 2
  DJF/MAM/JJA/SON panels, without an annual panel. Change clouds and sign agreement retain Annual.
  Each model is a point, median a thick tick, and prepared IQR a pale band where at least ten models
  contribute. No whiskers. Use single-line scenario labels. If another scenario reaches ten models,
  omit scenarios below ten and name them in the caption.
- Grid y-axis titles show units alone. Put model counts and periods in captions. Use round 1-2-5
  ticks, at most eight labels, ending one step beyond data. Change grids share a y-axis across seasons.
  Seasonal totals (mm per season) use a separate axis per season, with each model's reference-period
  Historical total left of a divider. Precipitation panel titles give reference amounts: the ensemble
  median monthly climatology summed over the season, e.g. `b) MAM · ref. ≈16 mm`.
- Variability grids show detrended temperature SD, precipitation SD as seasonal totals and
  precipitation CV (%) per model, with Historical groups like seasonal-values grids. Keep per-model
  future/reference ratios in analysis tables. Sign agreement has Annual plus four seasons, not months.
- Change strips and monthly-change figures show a flat gray zero-centred band: the 90% range of mean
  change attributable to year-to-year variability in historical runs (`noise_halfwidth`), computed
  from each member's detrended reference SD and then median across models. Use one band per season
  panel or month. Omit it in mostly dry periods and seasonal-values/variability grids.
- Monthly changes remain box plots: tinted IQR, coloured edge, dark median and light uncapped whiskers
  over the inter-model range, on a stepped full-width-month variability band. Wrap scenarios in two
  columns with shared 1-2-5 limits, unit-only y-axis and horizontal month labels. Omit below-ten-model
  scenarios as above; caption their names and counts. Keep climatology and changes separate.

### WF2 annual and variability products

- Calculated dew point and 10 m wind reuse the annual anomaly trajectory, monthly climatology and
  Annual-plus-four-seasons change strip. Use degC and m/s state means and absolute differences.
  Captions distinguish liquid-water dew point, supplied scalar wind speed and magnitude of the monthly
  mean vector; retain method, height, periods/counts and monthly/buffered-bbox approximations.
- Calculated PET uses the annual anomaly trajectory form in mm/year and the monthly climatology
  form in mm/day. Its change strip contains Annual plus four seasons, with existing model points,
  median ticks and IQR marks; use percent for relative changes and integrated mm for absolute changes.
  Label unavailable panels explicitly. Captions name the method, monthly-input approximation and
  shared buffered-bbox support; PET does not enter the temperature–precipitation cloud.
- Annual-overview median lines use 1.0 pt; trajectory medians 1.5 pt. Do not smooth annual values or
  bridge missing-year gaps.
- Show temperature SD, precipitation SD and precipitation CV variability in separate figures, with
  change-strip points/median/band and a ratio-1 reference line. This is descriptive model spread;
  retain numerical sampling benchmarks separately, without significance fills in these figures.
- The annual member-ratio comparison combines all three indicators. Boxes, medians and capped ranges
  summarize model ratios after geometric averaging within each model, with equal model weight. Faint
  member points show individuals without changing summary weights.

### WF2 change clouds, agreement and coverage

- Default GCM diagnostics also include annual model-row/scenario-column spatial matrices and
  monthly spatial sheets. Basin-average monthly changes use the existing box plots; individual
  `cmip6_delta_monthly_*.png` curve sheets are retired and no longer produced.
  Maps summarize available paired members by cell-wise median within each model/scenario. Preserve
  native source grids, including nonuniform Gaussian latitude spacing; clip display at the basin
  outline without resampling. Captions disclose complete-month masks, dry thresholds and member counts.
  Small ensembles remain visible in these diagnostics. Existing ensemble-summary eligibility stays unchanged.
- `colors.yml` → `projection_changes` owns the UNESCO draft report Figure 13 swatches: cyan through
  blue/purple to magenta for precipitation and blue through cyan/yellow to red for temperature.
  Adapt these ramps to signed change values with a common symmetric scale and a visible zero tick.
  Scenario-comparison figures retain IPCC scenario colors.

- Change clouds have square plotting areas and circular, slightly transparent, dark-edged points for
  every scenario. Colour distinguishes scenarios; stack the inset legend in the clearer upper corner.
  Axes read `Mean precipitation change (%)` and `Mean temperature change (°C)`; captions state periods.
- Sign-agreement matrices have temperature and precipitation percentage-change panels, four discrete
  agreement classes with paired blue/red swatches, cells about 7 × 3.6 mm, and horizontal two-line
  SSP labels (SSP1 over 2.6). Rows are Annual and seasons. Panel titles use 6 pt; ticks, signs and
  legend use 5 pt, an exception to 7/6 pt. Show only median sign per cell, with text colour chosen
  by fill luminance; k/n stays in `sign_agreement.csv`. Hatch precipitation cells retaining at most
  half the scenario's models, omit sign and label `Excluded (too few models)`. Directional agreement
  is neither probability nor significance; state this in the caption.
- Coverage is a compact one-line-per-model table with model, family, institution, country and
  historical-reference column, on alternating pale rows. Filled tiles mark loaded members; print a
  count only above one. Empty cells mean none. Deduplicate loaded historical reference series. Omit
  models with no loaded member and count them in the caption. A dash denotes unknown family; cite the
  family mapping and its provisional/unknown marks.

## Scientific interpretation and methodological review

- Name what each point represents: member, model, scenario summary or observation. Keep paired
  variables from the same model, scenario and analysis window. Distinguish within-model member
  variation from between-model spread.
- Reduce members within models before equal-model summaries. Additional members must not raise a
  model's weight. Keep geometric member reduction for variability ratios. Individual member points
  can accompany model boxes when their meanings are explicit.
- Ranges and IQR boxes describe spread, not confidence intervals. Members of one model are dependent;
  shared ancestry can limit independence among models.
- Match reference/future aggregations for changes. Monthly changes use each month's reference, not
  an annual value. A historical change box is zero by construction: use a zero line and a separate
  absolute-climatology companion.
- Temperature variability uses SD; Celsius CV depends on an arbitrary zero. Precipitation SD is
  absolute and CV relative variability. Name aggregation and dry-mean exclusions. The current
  calculations use detrended period mean rates for precipitation SD and detrended period totals
  divided by their original mean for CV; do not imply an SD/CV/mean identity. Unifying these methods
  would change numerical meaning, not style.
- Drought-event legends use coloured dots for datasets, a line for complete events and a hatched
  sample for truncated events. The hatch is eight diagonal strokes with 0.5 pt linewidth. Timing
  boxes are 0.12 high in cumulative-fraction coordinates, with about 0.6 pt outlines/whiskers.
- Autocorrelation limits use the pointwise approximation ±1.96 / sqrt(pairs at lag). Label them
  95% confidence limits and explain the approximation and limits in captions. Bartlett or other
  serial-correlation-aware limits need a separate methodological decision.

## Captions

- Store captions with their figure products in the relevant `figure_captions.md`. Write for reports:
  name variables, encoding, statistical meaning, source support, missing values and material caveats.
  Keep internal rules and gates out. Move plot footers into captions, left-aligned with plotting area,
  slightly smaller than axis text and gray.
- Geographic map captions (`shared/map_frames.map_caption`) are one sentence naming quantity/units,
  grid/period where applicable, and the product credit from `map_frames.SOURCE_CREDITS`. Add a sentence
  only for material map-specific facts: off-scale values, missing scale/cells, source-grid PET method,
  auxiliary field or classes omitted from the key. Keys/overlays explain the rest; metadata retains
  support counts, thresholds and provenance.
- WF2 captions (`projections/projection_captions.py`) stand alone: quantity/units, periods, encoding
  where the key does not explain it, relevant method (detrending, day weighting, dry exclusion) and
  interpretive caveat. Report counts, omitted scenarios/models, shifted/joined windows, hatched dry
  periods and variability bands when relevant. Coverage and agreement captions use two or three
  short sentences. `SUITE_NOTE` states suite-wide caveats once at the head of `figure_captions.md`.
- Span captions across all plotting columns. In captioned exports, add a band below the figure without
  resizing its plotting area. Keep font proportions stable. Set `savefig.dpi` before
  `plotting.fundamentals.add_caption(fig, text)`; it freezes drawn axes/decorations, adds the measured
  band and restores original DPI. Do not relayout or resize afterwards; wording and metadata remain
  workflow-owned.
  Caption measurement runs layout without painting the ordinary canvas; final export paints the completed
  figure.
- Wrap embedded text to measured axes width with the existing note font. Leave 3 mm below the
  caption and 3 mm to the original canvas bottom. Tight-cropped maps keep their geometry and 2 mm
  export border; do not resize maps to fit captions.

## Interactive review process

- Preserve baseline exports and data provenance. Compare at the same physical size and resolution,
  including captions and actual files at intended publication size. If a shared helper changes,
  inspect all affected figure families. The
  [validation ladder](../../../dev/reference/dev-validation.md) owns checks.
- Render retained inputs through the supported entry point:

```powershell
pixi run python scripts/render_figures.py <family> --project <run> --out <dir>
pixi run python scripts/render_figures.py --list
```

- Each family uses its workflow renderer. Add a missing family to this entry point. For WF2 runs whose
  source config later narrowed its scenarios, repeat `--scenario` for each retained SSP needed in a
  comparison. Use [task scratch](../../../dev/reference/dev-repository-layout.md#task-temporary-material)
  for temporary review files.

## Implementation entry points

- The [plotting README](../README.md) covers prepared drawing and explicit-figure export.
  Workflow adapters own calculations, captions, filenames and output records.
- [`shared/plot_preset.py`](../../shared/plot_preset.py) builds appearance tokens from
  [`style.yml`](../../../config/plotting/style.yml) and
  [`colors.yml`](../../../config/plotting/colors.yml). [`prepared_maps.py`](../prepared_maps.py)
  draws adapter-supplied maps. `shared/map_frames.py` prepares frames/overlays;
  `climate_analysis/diagnostic_maps.py` prepares climate fields; `shared/plot_spatial_maps.py`
  prepares spatial keys and locator basemaps.
- WF0 artists/captions live in `climate_analysis/diagnostic_figures.py` and
  `diagnostic_captions.py`; WF2 artists in `projections/projection_suite_figures.py`.
  `scripts/render_figures.py` renders finished runs; its retained-map reader verifies artifact
  integrity, not equivalence with current code.

### Presets

- Use the [plotting README](../README.md#prepared-drawing-presets-and-export) for `use_preset`,
  `NEUTRAL`, `CST_PRESET` and export scope. Enter `use_preset(CST_PRESET)` for every new CST render
  path and keep it active through final export. The core does not load CST YAML.

### Prepared annual and monthly charts

- Use the [prepared drawing guide](../README.md#prepared-drawing-presets-and-export) for `AnnualLine`,
  `TrendSegment`, `MonthlyBand` and drawing ownership. WF0 keeps compatible `View` wrappers; WF1
  forcing and basin/subbasin summaries prepare records directly. Adapters retain scientific
  preparation, captions and filenames.
