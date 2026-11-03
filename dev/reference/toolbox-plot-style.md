# Scientific plot style and visual review

Last updated: 2026-10-20

This shared reference records the agreed WF0, WF1 and WF2 styles and defaults for subsequent figure
revisions. Apply relevant defaults across plot types; preserve the explicit exceptions below. New visual
decisions belong here, alongside changes to the plotting code. A style decision does not authorize changing
the underlying calculation or statistical interpretation.

Descriptions of figures without displayed titles or footers below describe the default `none` mode.
Selecting `caption` or `title` adds the requested text outside the existing plotting canvas.

## Output paths

Use `plots/` for generated graphics throughout the toolbox. WF0–WF2 also write readable titles to
`plots/figure_titles.md` and a `figure_titles.json` inventory beside the corresponding caption JSON.
Title keys match the caption inventory's ordinary figure keys.
Each title identifies the figure's subject and relevant source, period or spatial support.
WF0 titles use `[specific plot title] — [dataset] ([start year]–[end year])`, for example
`Monthly climate data coverage — ERA5 (1971–2020)`. Comparisons list the contributing dataset names;
spatial support and methods remain in captions and structured metadata.
Project-level `plotting.annotation` selects `none` (default), `caption` or `title` at the ordinary PNG paths.
Retain title and caption inventories independently of this setting. R/weathergenr figures are excluded.
Spatial maps and legacy climate figure adapters that have no caption inventory still write title files.

Keep readable captions in
`plots/figure_captions.md`. Machine plot manifests and internal analysis records belong in the cohesive
output unit's single root `_engine/`, following [the output naming rule](dev-naming-conventions.md#generated-outputs-under-project_dir).
WF2 uses the ensemble root's `_engine/`, with manifest figure paths relative to `plots/`.
Existing WF0 `plots/_engine/` paths await explicit migration. JSON containing user-facing data stays with
its owning data product. Generated filenames and directories use lowercase snake_case, including
`full_period/` and `period_<start>_<end>/`. WF2 adopts the WF0 dataset/variable/context/spatial grammar,
with canonical `precip` and `temp` stems. See [naming](dev-naming-conventions.md) for current filename conventions.

## Shared colour defaults

`config/plotting/colors.yml` owns the default categorical cycle, dataset and evaluation colours,
SSP colours, climate-map colormaps and precipitation-anomaly colormap. Read it through
`blueearth_cst.shared.plot_colors`; do not duplicate these values in renderers. The loader resolves the
file relative to the toolbox, validates its structure and hex colours, and reads it once per process.
Restart an interactive Python session after editing it. Matplotlib resolves colormap names when drawing.

`config/plotting/style.yml` owns page/export defaults, typography, shared chart dimensions, lines,
markers, bands, label backgrounds, map feature geometry (`maps`) and evaluation-sheet appearance.
Map feature colours (`map_features`, `river_order`) stay in `colors.yml`; both reach the drawing code as
`Preset.maps` tokens (see [Spatial map template](#spatial-map-template)). `shared.plot_setup` loads these
preferences; `shared.plot_preset` turns them into the CST chart preset, and `shared.plot_style` keeps
the cartographic/evaluation interface. See [Presets](#presets) for how the preset reaches drawing code.
Figure-specific geometry stays with its renderer. Keep shared appearance preferences in this setup
directory instead of adding new literals to individual plots.

The default cycle is Option A, Expanded original, selected on 2026-10-15. Its order is Original blue
`#0D38E0`, Terracotta `#CC593D`, Original green `#00B389`, Amber `#D99A19`, Violet `#8054B3`, Rose
`#BE5084`, Cyan `#169CC4`, Slate `#66768A`. Blue and Terracotta are the first comparison pair.
ERA5 and observed series use blue; CHIRPS and simulated series use Terracotta. Dataset and evaluation
entries refer to named cycle colours so an edit stays consistent across those families. Unknown datasets
take the remaining cycle colours in configuration order, skipping colours used by named datasets present
in the comparison. Marker shapes change when those colours repeat. This palette has not been certified
for colour-vision deficiencies; preserve line styles, markers and labels as additional distinctions.

Both shared Matplotlib style contexts apply the cycle. Explicit semantic presets take precedence:
the IPCC scenario colours and existing climate-map and anomaly colormaps retain their values.
`historical_maps` preserves the custom ramps used by the current WF0/WF1 display reference; `climate_maps`
preserves named colormaps used by the legacy map adapter. Colour defaults do not change classification,
normalization, data, or scientific configuration identity.
WF0–WF2 plotting rules track a digest of both YAML files in renderer params, so an edit schedules
plotting jobs.
WF1 evaluation also refreshes indicators written by its combined evaluation/plotting job.
WF3's R adapter reads shared tokens and weather-only overrides at render time, outside scientific
configuration identity. The [reviewed weather profile](weather-diagnostic-style.md) owns its colours,
marks, typography and figure exceptions. The [physical panel contract](panel-layout-contract.md) owns
measured diagnostic-grid geometry shared by the R and Python adapters. Figures retain their existing
filenames under `evaluation/plots/`; project annotation modes remain Python-only. See the weather
profile's delivery section for the verified upstream implementation and pending tagged dependency.
The historical map-reference producer also tracks the YAML as a display dependency so cached map colours
refresh consistently across WF0 and WF1. This does not change the extracted climate data or simulation identity.

## SSP scenario colors

WF2 uses the IPCC AR6 scenario palette, including all eight ScenarioMIP SSPs and the LowNTCF variant.
Use CMIP6 experiment IDs as keys; SSP family colors and RCP colors are different palettes.
The final TSU hex values are published in
[pyam's IPCC palette](https://pyam-iamc.readthedocs.io/en/stable/tutorials/ipcc_colors.html),
derived from the [IPCC WG1 color maps](https://github.com/IPCC-WG1/colormaps).

| Experiment ID | Display label | Color |
|---|---|---|
| `ssp119` | SSP1-1.9 | `#00a9cf` |
| `ssp126` | SSP1-2.6 | `#003466` |
| `ssp434` | SSP4-3.4 | `#2274ae` |
| `ssp245` | SSP2-4.5 | `#f69320` |
| `ssp534-over` | SSP5-3.4-OS | `#92397a` |
| `ssp460` | SSP4-6.0 | `#b0724e` |
| `ssp370` | SSP3-7.0 | `#df0000` |
| `ssp370-lowNTCF` | SSP3-7.0-LowNTCF | `#e61d25` |
| `ssp585` | SSP5-8.5 | `#980002` |

Keep the mapping in `config/plotting/colors.yml`; `projections/projection_figures.py` re-exports it
for existing callers, and all WF2 suite artists use it.
Unknown experiments retain their literal labels and muted fallback color.

## Typography, axes and layout

- Design WF2 figures for 4–5 scenarios and 10–20+ models, potentially with multiple members per model.
  Small retained examples do not establish that labels, legends or panels remain readable at that scale.
  Wrap panels instead of indefinitely shrinking them; adapt trace opacity to member density and retain
  clear summaries. Never silently omit models or scenarios to make a figure fit.
- Keep font sizes consistent in physical units across different canvas sizes. Current scientific figures
  use 7 pt axis and panel titles, and 6 pt ticks and annotations. Scale every element together in previews,
  including captions, legends and labels; do not enlarge a compact figure to fill its HTML card.
- Use modest white padding around compact figures and adjust the exported canvas dimensions to avoid
  excessive empty space. Square plotting areas mean equal physical axis lengths, not equal data scales.
- Every labeled axis tick has a visible tick mark. Annual series have a tick for each year; reduce labeled
  year density on long records to avoid crowding. Retain annual minor ticks where appropriate.
- Anchor major and minor year ticks to the same calendar-year grid; do not distribute ticks from the
  record's first year or place them at inconsistent intervals near labels. Reviewed long WF2 records use
  20-year major ticks and 5-year minor ticks. Compare equivalent panels on shared limits and tick positions.
- Use units alone for axis titles unless a figure-specific exception needs the variable name.
- Exception: WF2 reference-relative axes retain `Relative to [reference-period] (symbol)`, for example
  `Relative to 1985–2014 (°C)` or `Relative to 1985–2014 (%)`. Use the actual analysis reference period;
  absolute-value axes still show units alone.
- Use `Precip.`, `Temp.` and `PET` when abbreviated variable names are needed. Display `CHIRPS` in legends
  and plot labels; retain version metadata in source records where relevant.
- Keep grid lines subtle and partly transparent. Avoid competing with the data or annotations.
- Always label multiple plotting panels `a)`, `b)`, etc.; explain panel contents in the caption.
- Default to no displayed figure title. With `plotting.annotation: title`, add the title above the figure
  without resizing its plotting area. Preserve explicitly agreed panel labels in every annotation mode.

## Legends, labels and points

- Default to a vertically stacked legend inside the upper-left or upper-right corner. Choose the corner
  with less underlying data and retain flexibility to move it for other input records.
- Use a white legend background with alpha 0.88 and no visible border. Use the same background for
  in-plot labels: white, alpha 0.88, no edge, padding 1.2.
- For the reviewed trend-interval, SPI fit, trend-sensitivity and monthly-coverage figures, place the
  stacked legend outside the panels on the right, with its top aligned to the panels' top border.
  Reduce panel widths as needed while retaining the intended overall canvas width.
- Place slope annotations far enough from their trend lines to avoid overlap. Position point labels to
  avoid overlaps with other labels and underlying map features; use the standard label background.
- Climate and spatial maps prefer each location's `station_name`; missing or blank labels fall back
  to `wflow_id`. Preserve IDs in data products and model output columns.
- Use contrasting point colors over map layers. For scatter plots, use partly transparent fills and
  darker outlines so points remain visible. The reviewed enlarged scatter points use area 36 pt²,
  alpha 0.7 and an approximately 0.7 pt dark edge; adapt where the plot density requires it.
- Keep dataset colors consistent using the shared defaults: CHIRPS Terracotta `#CC593D`, ERA5 blue `#0D38E0`.

## Color scales and colorbars

- Check on-map colorbars and legends against the drawn map frame after layout settles, including titles
  and tick labels. Shorten overflowing colorbars and shift keys inward as needed. Reduce categorical
  legend text only when the full legend exceeds the available area; warn if adjustment cannot fit it.
- For independent color scales, use the full light-to-dark palette range, including its darkest or
  lightest endpoint at the high end. Keep comparable limits when plots share a scale.
- Use perceptually clear gradients that preserve visibility of intermediate values. For dataset-specific
  heatmaps, use the dataset hue with light-to-dark intensity encoding the displayed quantity.
- Use rectangular colorbar extension ends and consistent title spacing (current title pad: 10 pt).
- Monthly precipitation departure calendars use continuous `BrBG` (brown dry, teal wet), centred on
  zero with symmetric limits spanning all finite pooled anomalies. Label the colourbar
  `Precipitation anomaly (mm)` and omit cell borders to reduce visual clutter.
  Omit the retired monthly-percentile companion. Missing values remain gray.
  This supersedes the discrete monthly-anomaly classes; SPI retains its separate conventions below.
- Remove missing/unavailable legend rows from anomaly plots and explain missing-value handling in captions.
- The reviewed SPI heatmaps use boundaries `-3, -2, -1, 1, 2, 3`, without a zero tick, and a colorbar
  height of 80% of the plotting area. Their final figure height is 56% of the original height.

## Spatial map template

Every geographic raster map (WF0 source and WF1 forcing climatologies, the WF1 spatial set, the basin
map) follows one template, `blueearth_cst/plotting/prepared_maps.py`, adopted on 2026-10-29. The core
draws supplied arrays and features with `Preset.maps` tokens; adapters prepare fields, classes, overlay
linework and basemap features; `shared/map_classes.py` classifies rasters and prepares relief and the locator.
The legacy `shared/cartographic_map.py` was retired on 2026-10-29.

| Element | Treatment |
|---|---|
| Coordinates | Plain axes, aspect `1/cos(mid-latitude)`; numeric signed ticks; `Longitude (°)` / `Latitude (°)`. |
| Rivers | Blue, width and shade stepped by stream order, thin reaches first, thin white halo. |
| Basin and divides | Black basin outline over dashed divides; divides merged so shared edges draw once. |
| Locations | Purple diamonds; bold station names with a white halo, no box, placed clear of overlaps and the frame. |
| Long names | Wrap to two lines at the separator nearest the middle (`maps.location_label_wrap` characters). |
| Furniture | Scale bar (km, latitude-corrected) on every map; no north arrow; locator only on overview maps. |
| Locator | Elevation and subbasin overviews only: small, square, inside the map in its clearest corner (≤40% of frame height). |
| Keys | Side panel, bottom-aligned with the frame: colourbar or class legend, title above, never on the map. |
| Source credit | In the caption, never a footnote on the figure. |

Fit the map frame within 150 × 95 mm, preserving its geographic aspect. Equal geographic frames have equal
physical plotting areas, independently of their keys. Coordinate labels, external keys and locators extend
the canvas without resizing the map or its typography. Measure decorations after drawing and leave 3 mm
outer canvas padding before adapter-specific export cropping.

Class legends list only classes present in the drawn area, and only those covering at least 0.1% of the
basin; smaller classes are still drawn and are named in the caption. A legend taller than the frame is fitted by
reducing its text, with a warning if it still does not fit. Colourbar titles break the unit onto its
own line. Map colourbars use `fundamentals.style_colourbar`, the same appearance as heatmap colourbars.

## Figure-specific decisions

WF1 forcing QA uses the reviewed WF0 map, annual-series and monthly-climatology styles for precipitation,
temperature and PET. Read the actual model-grid forcing, mask to the model basin, and use equal-weight
cell means for temporal figures. Keep the nine `forcing_<variable>_<kind>.png` filenames for now.
Identify the data as Wflow forcing in legends and captions; use ERA5 blue for this forcing family.
Maps retain their model-grid footprint and basin mask, with the reviewed coordinate labels, palettes,
spatial overlays and gauge labels. WF0 source and WF1 forcing maps share the historical reference at
`data/climate/historical/_engine/map_reference.json`. The selected source's outer cell edges determine
the scale-reference frame. Historical candidate cells intersecting this frame determine pooled classes;
borrowed CHIRPS temperature/PET do not contribute. Displayed maps zoom to the basin bounds with the same
padding as spatial maps, clipped to a native source grid's outer cell edges so no blank padding shows;
masked model-grid maps keep the full padding. Ticks and midpoint-latitude aspect derive from that
viewport. Without basin geometry, they retain the reference frame. Native source fields and
model-grid basin masks retain their scientific support. Common colors do not establish equal periods,
eligibility policies or PET methods; captions state those differences.

Every available reference colorbar shows both rectangular extensions. Captions report strict under/over
counts only when nonzero; the map metadata always records them. The final class includes its upper
boundary. Missing cells are gray, outside-model and uncovered areas white. With no finite historical scale, finite cells use an unscaled neutral hatch and
there is no numerical colorbar. Never infer a scale from model forcing or invent a default 0–1 scale.
Suppress individual model-cell outlines on the fine forcing grid so they do not obscure the field.
Show cells outside the model basin in white and unavailable summaries inside it in gray.

Use WF0's start-year reporting labels, completeness handling, Sen trends and monthly mean with a
10–90th percentile band. WF1 uses WF0's default tolerances of three missing days per month and fifteen
per reporting year, independently of whether WF0 runs. Temperature uses means; precipitation and PET
use totals. Keep model corrections and statistical caveats in separate captions, without plot footers.
These summaries describe forcing QA; they do not modify the forcing or simulation calculations.

WF1 station hydrographs retain the five-panel layout: daily discharge, annual means, seasonal cycle,
and linear/log flow-duration curves. Apply the common typography, frames, soft horizontal grids,
short panel titles and inside legend. Observations remain blue and dashed; simulation remains
Terracotta and solid. Use discharge units alone on vertical axes, three-letter months and percentage
exceedance labels. Annual labels start at the preceding five-year grid mark (ten years for records
longer than 35 years), retaining annual minor ticks so the left edge has a nearby calendar label.
Station identity, dates, support and methods
belong in separate station captions, bounded by the plotting area's left and right edges in previews.

Annual means require full calendar-year bounds, at most three missing days in each month and fifteen
in total. Seasonal cycles average eligible monthly means equally across years; full calendar months
allow at most three missing days. Reindex to daily dates so absent timestamps count as missing.
Use finite, nonnegative discharge without infilling. With observations, summary panels use paired
valid dates; the daily panel retains the available series for context. Flow-duration ranks use only
valid values and positions `rank / (n + 1)`. Retain zeros on the linear panel, omit them from the log
axis without changing exceedance positions, and report their frequency. The log range includes all
positive values from both series, without percentile clipping. These choices affect hydrograph
summaries only; performance indicators and other evaluation sheets retain their own methods.

WF1 performance sheets use the same finite, nonnegative paired daily support for the scatter, sample
count, cumulative volumes and exported indicators. Align dates before pairing and retain gaps in the
cumulative curves. These volumes integrate paired days only and do not estimate missing-day flow.
Monthly indicators compare means over the same paired days within each month; hydrograph calendar
completeness thresholds do not gate indicator rows. Undefined scores remain unavailable.
Keep squared Pearson correlation and sample counts in JSON metadata, without displaying them on the
performance plot. Label the scatter `a)` and cumulative volume `b)`, with the stacked observed/simulated
line legend inside the upper-left of the scatter, below its panel label. Keep table values identical to
`performance_indicators.csv`. Use common typography, soft frames/grids and a concise publication caption
that explains each labelled panel, without displayed titles or figure footers.

WF1 peaks and lows signature sheets use the same typography, soft frames, horizontal grids and
borderless label backgrounds. Retain the three-panel layout, regular-weight inside panel letters
with equal physical offsets from each panel's top-left corner, and a stacked inside legend for the
daily series. Panel c shows the full supplied simulation record with calendar-year ticks and a
translucent vertical band spanning the selected wettest/driest calendar year. Preserve the existing
warm-up exclusion and observation-overlap selection upstream. Frequency and daily axes show discharge
units alone; scatter
axes retain observed/simulated labels to distinguish their roles. Use translucent points with dark
edges. Station identity, selected year, methods and caveats belong in the station caption inventory,
without displayed titles or figure footers. Preserve the existing annual extremes and plotting positions.

WF1 hydrological output figures default to two single-axis products per variable: a basin-average
annual series and seasonal cycle. Optional subbasin figures repeat both products individually, never
as panels. These figures have no displayed titles. Reconstruct the basin's native grid-cell mean from
daily zonal means using fixed cell-count
weights; missing zones invalidate that basin day rather than changing its spatial support.

Apply the same first-year warm-up exclusion and calendar completeness rules to all hydrological
variables. Eligible months allow three missing days; eligible years have full calendar bounds, all
months eligible and at most fifteen missing days. Flux amounts use monthly/annual totals, with eligible
missing days filled by that month's available-day mean. Rates and stored quantities use means without
infilling. Keep physically valid negative recharge. Seasonal cycles show equal-year monthly means and
10–90th percentile bands, describing interannual variability rather than uncertainty in the mean.
Annual series use Sen slopes per decade and the shared WF0 trend treatment. Snow plots represent the
native snowpack liquid-water depth, a stored quantity in mm, rather than an accumulated flux.
Captions use concise publication prose: quantity, support, period and visual encodings. Keep numerical
support counts, completeness thresholds and exclusion lists in the machine-readable methods metadata.

| Figure type | Agreed treatment |
|---|---|
| Historical climate maps | Zoom to the basin bounds with spatial-map padding. |
| Wflow forcing maps | Use the same basin-focused viewport as historical climate maps. |
| Map coordinates | Axis titles include degrees; tick labels are numeric, signed, without N/S/E/W. |
| Annual series | Trend line and slope annotation on every applicable series; y-axis shows units only. |
| WF2 annual overview | No legend; explain member traces, medians and ranges in the caption. |
| WF2 trajectories | Model median and range; distinct historical trace; stacked inside scenario legend. |
| WF2 handover line | Dashed line at the CMIP6 historical–scenario handover (2014.5), never the current year. |
| WF2 seasonal grids | 2 × 2 seasons; model points, median tick, IQR band; unit-only y-axis; no legend. |
| WF2 seasonal cycle | Median lines; shade historical range only; one-row legend above the axes. |
| WF2 dry periods | Precipitation %/ratio panels or months keeping at most half the models: hatched. |
| Monthly climatology | Stacked inside legend in the less crowded upper corner. |
| Cumulative timing | Bottom month axis; no top-axis title; y-axis `Cumulative (%)`, numeric ticks. |
| Timing boxes | Enlarged unfilled horizontal boxes, thin outlines, no mean dot. |
| Drought events | Dataset colors identify sources; legend distinguishes `Complete` and `Truncated`. |
| Truncated events | Dense diagonal hatching with thin hatch strokes, rather than dashed outlines. |
| Spell exceedance | Square plotting area; y-axis `Exceedance Pr. (log)`; compact padded canvas. |
| Autocorrelation | Dark point edges and approximately 95% confidence limits; qualify these in caption. |
| Trend intervals | Thicker horizontal intervals and caps, closely spaced rows, no dataset row labels. |
| Seasonal P–T | Square panels; `Temperature anomaly (°C)` and `Precipitation anomaly (%)` axes. |
| SPI fit checks | Dataset-hued p-value heatmaps; minimal panel titles; no x-axis title `Month`. |
| Trend sensitivity | Separate slope points with confidence error bars for each eligible starting year. |
| Monthly coverage | Compact dataset rows of monthly cells; complete, gapped and unusable states. |

Trend-sensitivity panels use sparse, calendar-aligned integer year labels with minor ticks at other
eligible starting years. Share the start-year extent across panels and name the fixed end year in the
x-axis title; retain every slope estimate and interval regardless of label density.

WF2 seasonal grids (temperature change, precipitation % change, precipitation seasonal totals) are 2 × 2
panels for DJF, MAM, JJA and SON, without an annual panel; the change cloud and sign agreement keep the
annual view. Each model is a point, the median a thick tick and the prepared IQR a pale band where at least
10 models contribute; no whiskers, so one outlying model does not read as typical spread. Scenarios carry
single-line x-axis labels; scenarios with fewer than 10 models are left out when another reaches 10, and the
caption names them. The y-axis title is the unit only; model counts and periods go in the caption. Axes end
on round 1-2-5 ticks (at most 8 labels) one step past the data. Change grids share one y-axis across
seasons; the seasonal-totals grid (mm, season total) gives each season its own axis and adds a gray
Historical group (each model's reference-period total) left of a divider. Precipitation panel titles give
the reference amount (ensemble-median monthly climatology summed over the season, "b) MAM · ref. ≈16 mm").
The interannual-variability grids (temperature SD, precipitation SD as season totals, precipitation CV in %)
show each model's detrended values with a Historical group, like the seasonal-values grids; the per-model
future/reference ratios stay in the analysis tables. The sign-agreement matrix shows Annual and the four
seasons, not months.

WF2 change strips and monthly-change figures draw a flat gray band around zero: the 90% range of mean
change that year-to-year variability alone gives in the models' historical runs (`noise_halfwidth`,
computed in the analysis step from each member's detrended reference SD; median across models). One band
per season panel and one per month; none in mostly-dry periods or on the seasonal-values and variability
grids.

WF2 monthly-change figures stay box plots: tinted IQR boxes with a colour edge, a darker median bar and
light uncapped whiskers to the inter-model range, over a stepped variability band (full-width month cells).
Scenarios wrap in two columns with shared 1-2-5 y-limits, a unit-only axis title and horizontal month
labels; scenarios under ten models are dropped as on the seasonal grids, and the caption names them and
gives models per scenario. Review climatology and monthly changes as separate groups in the HTML page.

Keep WF2 scenario summaries subordinate to the data: reviewed annual-overview median lines use 1.0 pt,
trajectory median lines 1.5 pt. Retain annual values without smoothing and preserve gaps in missing years.

WF2 interannual variability shows temperature SD, precipitation SD and precipitation CV in separate figures.
Use the change-strip point, median and band style, with a reference line at ratio 1.
Show descriptive model spread; retain numerical sampling-benchmark products separately without displaying
them or significance fills in this figure.

The annual member-ratio comparison shows all three indicators together. Its boxes, medians and capped
ranges summarize model ratios after geometric averaging of members within each model, with equal model
weight. Faint points retain individual members; their density does not set the weight of the summary.

WF2 change clouds retain square plotting areas and circular points for all scenarios, distinguished by
color, with darker edges and slight transparency. Stack the inset legend in the less occupied upper
corner. Use “Mean precipitation change (%)” and “Mean temperature change (°C)” as axis titles; state
reference and future periods in the caption.

WF2 sign-agreement matrices use temperature and precipitation percentage-change panels, four discrete
agreement classes with paired blue/red swatches, and compact cells (about 7 x 3.6 mm) under horizontal
two-line scenario labels (SSP1 over 2.6) with one-letter month labels. As an approved exception to the
7/6 pt sizes, its panel titles use 6 pt and its ticks, cell signs and legend 5 pt, in proportion to the
small cells. Cells show only the median sign, with text color chosen by fill luminance; k/n counts
stay in `sign_agreement.csv`. Precipitation cells keeping at most half the scenario's models are hatched
with no sign and labelled "Excluded (too few models)". The caption states that directional agreement is
neither probability nor significance.

WF2 ensemble coverage uses a compact one-line-per-model table (model, family, institution and country), a
historical-reference column and pale alternating rows. A filled tile marks loaded members, with a count
only where there is more than one; empty cells mean none. Historical counts deduplicate loaded reference
series. Models with no loaded member are left out, with their number in the caption; a
dash marks an unknown family. The caption cites the family mapping and its provisional/unknown marks.

## Scientific interpretation and methodological review

- State what a plotted point represents: member, model, scenario summary or observation. Distinguish
  within-model member variation from between-model spread and preserve paired variables from the same
  model, scenario and analysis window.
- For equal-model summaries, reduce members within models first; additional members must not increase
  a model's weight. For variability ratios, retain the approved geometric member reduction. Individual
  member points can coexist with model-summary boxes when their different meanings are explicit.
- Ensemble ranges and IQR boxes are descriptive spread, not confidence intervals. Members of one model
  are dependent; shared model ancestry can also limit independence between models.
- Changes compare matched reference and future aggregations. Monthly changes use each month's own
  reference, rather than a single annual reference. A historical change box collapses to zero by
  construction; use the zero reference line, with absolute climatologies in a separate companion figure.
- Use SD for temperature variability; Celsius CV depends on an arbitrary zero. For precipitation, SD
  expresses absolute variability and CV relative variability; label their aggregation and dry-mean
  exclusions. Do not imply SD/CV/mean ratio identities when they use different underlying series.

The retained calculations use detrended period mean rates for precipitation SD and detrended period totals
divided by their original mean for CV; keep this distinction explicit when comparing them. Unifying their
aggregation would be a separate numerical-method change, not a styling decision.

For drought events, reserve colored dot legend symbols for datasets; use a line for complete events and
a hatched sample for truncated events. The final reviewed hatch is eight diagonal strokes in Matplotlib's
hatch string, with hatch linewidth 0.5 pt. Timing boxes are 0.12 high in cumulative-fraction coordinates,
with approximately 0.6 pt outlines and whiskers.

Autocorrelation limits currently use the pointwise approximation ±1.96 / sqrt(pairs at lag). Label them
as 95% confidence limits and state the approximation and its limitations in the caption. Bartlett or
other serial-correlation-aware limits require a separate methodological decision; they were not adopted
in this visual review.

## Captions

- Store each figure's caption in its relevant figure-caption Markdown file, alongside the figure products.
- Write captions for publication or reports: describe variables, encoding, statistical meaning, source
  support, missing values and material caveats. Do not describe internal rules, review logic or gates.
- Move footer information from the plot into its caption. Keep captions left-aligned to the plotting
  area, slightly smaller than axis text, in gray.
- Geographic map captions (`shared/map_frames.map_caption`) are one sentence: the quantity with units,
  grid and period where they apply, "from" the product credit in `map_frames.SOURCE_CREDITS`. Add a
  sentence only for a material, map-specific fact: values beyond the colour scale, no scale or no
  drawn cells, a source-grid PET method, an auxiliary field, or classes left out of the key. The key
  and overlays explain the rest; support counts, thresholds and reference provenance stay in metadata.
- WF2 captions (`projections/projection_captions.py`) are self-contained journal-style legends: the
  quantity and units, periods, encoding where no key gives it, the method where it changes the reading
  (detrending, day-weighting, dry exclusion) and the interpretive caveat (inter-model spread is not a
  confidence interval; sign agreement is not significance). Report model counts, omitted scenarios or
  models, shifted or joined windows, hatched dry periods and the variability band where they apply.
  Keep table- and matrix-type figures (coverage, sign agreement) to two or three short sentences.
  Suite-wide caveats are also stated once, in `SUITE_NOTE` at the head of `figure_captions.md`.
- Captions span the full plotting area: from the leftmost axes' left edge to the rightmost axes' right
  edge, across every column of a multi-panel figure, both in captioned exports and on review pages.
- Include captions inside the preview canvas when enabled. Add space below the image without resizing
  the plotting area or changing its proportions. Keep caption-to-axis font proportions consistent.
- `plotting.fundamentals.add_caption(fig, text)` finalizes an already drawn figure for its captioned
  export. Set `savefig.dpi` to the intended export resolution first. It freezes the drawn axes and
  decorations, adds a measured bottom band, and restores the original figure DPI. Do not resume layout
  or resize the figure after this step. Caption wording and metadata remain workflow-owned.
- Wrap embedded captions to the span of the drawn axes, using measured text width and the existing
  note font. Keep 3 mm below the caption and 3 mm between its top and the original canvas bottom.
  Tight-cropped maps retain their original geometry and 2 mm export border; do not resize the map
  canvas to make room for a caption.

## Interactive review process

- Begin each new plot-group review with scientific, methodological and publication-readability
  suggestions before applying changes. Distinguish visual revisions from changes to calculations.
- Confirm the authoritative baseline and its data provenance before revising. Select retained inputs
  appropriate to the review; older production renders can carry superseded styles.
- Start from the tracked `dev/templates/figure-comparison.html` template. See
  `dev/templates/figure-comparison.md` for setup and figure-record fields. Keep session data in scratch;
  update the tracked template when improvements to the review interface are approved.
- Keep the disposable HTML review in `dev/tasks/<id>/.tmp/`; follow `dev-repository-layout.md` for closure cleanup.
- Show only the plot group currently under review, with existing and revised versions side by side.
  Preserve a baseline before revising the group.
- Keep the template's header, toolbar and card layout; fill only its project context, provenance and
  records. Each card heading names the figure and version, for example `CHIRPS – Spell (before)` and
  `CHIRPS – Spell (revised)`, and the card shows the file name with a link to its source folder.
- Keep consistent physical image scaling so fonts do not grow when the canvas gets smaller. The overview
  uses the pair's declared export dpi (600 by default); both images must use that dpi.
- Keep the full selected file path in the page data so the source remains identifiable.
- Render from retained inputs with `scripts/render_figures.py <family> --project <run> --out <dir>`
  (`--list` shows families) and inspect the actual exported figures. Each family calls the rule's own
  rendering function from a re-render adapter beside the rule adapter; do not write ad-hoc render
  drivers. A family missing from the CLI is added there, not scripted in scratch. For a run whose
  source config has since narrowed its SSPs, pass wf2 `--scenario` per kept SSP to preview the rerun.
  Apply reusable decisions to other applicable plots, and render all affected callers when a shared
  figure helper changes.
- Keep review outputs under `dev/tasks/<id>/.tmp/`, outside the persistent test suite. Verify figure changes
  by rendering and visual inspection; `dev/scripts/capture_page.py` screenshots a review page headlessly
  when the browser tool is unavailable. Retain tests for exported numerical indicators and workflow
  contracts separately from figure review checks.
- During an explicitly deferred-check visual session, do not run tests or gates at each revision. Run
  appropriate checks at the end of the session. Commit once the user approves a figure style/type, rather
  than after each incremental revision. Do not land or push without separate authorization.

## Implementation entry points

- Shared drawing fundamentals: `blueearth_cst/plotting/fundamentals.py`.
- Portable presets, `use_preset` and the neutral default: `blueearth_cst/plotting/preset.py`.
- Reviewed CST preset and its token names: `blueearth_cst/shared/plot_preset.py`.
- Prepared annual lines, trend segments and monthly bands: `blueearth_cst/plotting/prepared_charts.py`.
- Explicit-figure export without workflow logging: `blueearth_cst/plotting/export.py`.
- Workflow bundles and current-figure compatibility: `blueearth_cst/shared/snake_utils.py::save_figure`.
- WF0 artists and compatibility aliases: `blueearth_cst/climate_analysis/diagnostic_figures.py`.
- Page, typography and export constants for the evaluation sheets: `blueearth_cst/shared/plot_style.py`.
- Geographic raster maps (WF0 source, WF1 forcing, WF1 spatial set): drawn by
  `blueearth_cst/plotting/prepared_maps.py`; frames, grid edges and overlay features prepared by
  `shared/map_frames.py`; climate fields by `climate_analysis/diagnostic_maps.py`; spatial keys and the
  locator basemap by `shared/plot_spatial_maps.py`.
- WF0 rendering and captions: `diagnostic_render.py` and `diagnostic_captions.py` in the same package.
- Re-rendering from a finished run: `scripts/render_figures.py`, dispatching to
  `climate_analysis/diagnostic_rerender.py`, `model/forcing_rerender.py` and
  `projections/projection_rerender.py`. Retained map references load
  through `shared/retained_reference.py`, which checks integrity but not current-code drift.
- WF2 artists: `blueearth_cst/projections/projection_suite_figures.py`.

For export without workflow logging, pass the figure explicitly and retain the caller's save options:

```python
from blueearth_cst.plotting.export import export_figure

export_figure(fig, "plots/precip_annual_ts.png", dpi=600, metadata={"Software": None})
```

The caller closes the figure after its final export. Existing workflow callers keep using `save_figure`.

### Presets

The core in `blueearth_cst/plotting/` reads sizes, fonts, colours and line weights from the active
`Preset` at call time and never loads `config/plotting/`. `use_preset(preset)` scopes one around
rendering and also applies its rcParams; outside any scope the built-in `NEUTRAL` default applies.
`CST_PRESET` is the authoritative CST appearance, built from `style.yml` and `colors.yml`.

Every CST render entry point draws inside `use_preset(CST_PRESET)` through export. A caller that owns
its rcParams, such as a caption added to a cartographic map, uses `use_preset(CST_PRESET,
rcparams=False)`. A core helper called outside any scope silently draws with `NEUTRAL`, so new CST
render paths must open a scope. Composition code may read the CST token names (`INK`, `W_FULL`, ...)
from `shared.plot_preset`. `dev/scripts/plotting_standalone_demo.py` exercises the core outside CST.

### Prepared annual and monthly charts

`prepared_charts.annual_series` and `monthly_cycle` draw supplied arrays and return a `Figure`.
Use `draw_annual_lines` and `draw_annual_trends` on an existing axis when a composition needs its own
ticks or legend, as WF1 hydrological summaries do. Call within `use_preset(CST_PRESET)` and keep that
scope active through export. The module does not select a backend or install global styles.

Adapters own aggregation, completeness, source order, units, reporting-year alignment and fitted trends.
`AnnualLine` contains values aligned to the supplied years. `TrendSegment` contains already evaluated
endpoints and the formatted label; drawing never fits a trend. `MonthlyBand` contains the mean and
lower/upper arrays in the supplied twelve-month label order. Pass the band's statistical meaning through
`band_label`; the current adapters use `10–90th percentile` for interannual spread. Drawing neither
calculates quantiles nor rotates months. Missing values retain gaps; inputs are not mutated.

WF0 keeps its compatible `View` wrappers. WF1 forcing and basin/subbasin summaries prepare the small
records directly. Existing captions, export options, filenames and scientific calculations remain with
their workflow adapters. These are internal reusable interfaces; external distribution is a later step.

WF2 review uses the newer artists and retained tables in the `wf0-lab` ensemble outputs as its visual
baseline. The annual overview, trajectories, change strips, split seasonal products, variability ratios,
member-ratio comparison, change cloud, sign agreement and coverage table have been reviewed and their approved
styles implemented in the toolbox. Trajectories use the prepared model median, not the model mean. Keep
baseline selection separate from validation of the current toolbox. Final integration evidence lives in
`../records/validation/plot-improvements/verification.md` (private-checkout-only citation).
