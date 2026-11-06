---
title: "Historical climate analysis"
format: html
---

Historical climate analysis builds understanding of baseline climate variability and the climatic
context of hydrological conditions. It compares candidate gridded datasets to support forcing selection,
using maps, time series and diagnostics of coverage, seasonality and extremes.

## Inputs {#inputs}

- **Basin definition.** Geographical extent and analysis resolution specified in the project settings.
- **Historical climate data.** Candidate gridded datasets accessed through the
  [data catalogs](data-catalogs-and-access.md). The
  [dataset reference](historical-meteorological-data.md) lists first-version options and variable requirements.

See [input formats](data-formats-and-conventions.md) for file conventions.

The selected forcing must supply at least 16 years. Source coverage can shorten the requested period;
extraction requires overlap between the source record and that period.

## Configuration {#configuration}

**Project settings** define the shared basin and climate inputs.

| Setting | Purpose |
|---|---|
| `basin.region`, `basin.resolution` | Basin extent and analysis resolution. |
| `climate.sources` | Candidate historical datasets to analyze. |
| `climate.selected` | Forcing source used in subsequent modeling. |
| `climate.window` | Requested historical period. |
| `climate.water_year_start` | Month that begins the reporting year. |

**Workflow settings** control the diagnostics and optional subbasin figures. The annotated template is
`config/templates/example-basin/project_config_analyze_climate.yml`.

| Setting | Purpose |
|---|---|
| `diagnostics.reference_period` | Reference water years for anomalies and SPI fitting. |
| `diagnostics.comparison_period` | Requested period for comparing sources. |
| `diagnostics.max_missing_days_month` | Allowed missing days per month; default 3. |
| `diagnostics.max_missing_days_year` | Allowed missing days per water year; default 15. |
| `diagnostics.wet_day_threshold_mm` | Wet-day threshold; default 1 mm/day. |
| `diagnostics.spi_distribution` | SPI distribution: `gamma` (default) or `pearson3`. |
| `subbasin_figures` | Additional tables and figures for individual subbasins. |

The reference period defaults to the climate window, and the comparison period defaults to the
reference period. Missing-day tolerances must match the project settings in `climate.map_reference`.
See [execution and run management](configuration-and-execution.md) for running the workflow.

## Processing sequence {#rule-flow}

![Main scientific processing steps and conditional paths described below.](assets/workflow-wf0.svg){width=100% fig-alt="Scientific processing from basin and climate inputs to the assessment products described in this section."}

The workflow prepares basin boundaries, extracts each dataset and calculates diagnostics. Maps retain
each source's native grid with shared display scales. Multiple sources are compared over their common
available period; subbasin results follow the same processing steps when enabled.

## Outputs {#outputs}

Results are saved under `data/climate/historical/`, relative to `project_dir`, with a separate folder
for each source. The main outputs are:

- **Diagnostic tables (`tables/`).** Coverage, climate indices, drought events and trend diagnostics,
  including daily basin series in `daily_basin.csv` and annual climatology in `annual_climatology.nc`.
- **Maps and time-series figures (`plots/`).** Precipitation, temperature and potential
  evapotranspiration where available, with titles and captions.
- **Source comparisons (`comparison/`).** Tables and figures comparing multiple datasets, including
  `tables/agreement.csv` and `tables/agreement.md`, produced when multiple sources are selected.
- **Subbasin results (`tables/subbasins/` and `plots/subbasins/`).** Additional tables and figures when
  `subbasin_figures: true`. Basin maps show subbasin outlines regardless of this setting.
- **Diagnostic record (`_engine/diagnostics.json`).** Methods, data availability and provenance.

Two shared outputs support the analysis:

- **Basin geometry (`data/spatial/`).** GeoJSON basin, subbasin, river and location vectors.
- **Map reference (`data/climate/historical/_engine/map_reference.json`).** Shared display extent and
  color classes, also used by model building.

Displayed maps zoom to the basin bounds with spatial-map padding, clipped to each source grid's
outer cell edges. Without basin geometry, they retain the reference frame.

Figure availability depends on source variables and usable data. PNG filenames follow
`<dataset>_<variable>_<context>_<spatial_scope>.png`; `plotting.annotation` controls whether figures
display a title, caption or neither.

## Diagnostics and interpretation {#interpretation}

- **Coverage.** Record completeness determines which periods can support climate diagnostics and fair
  comparisons between datasets. Missing-day tolerances identify usable months and years, while source
  comparisons use their common available record.
- **Seasonality and anomalies.** Seasonal rainfall and temperature patterns help explain the timing of
  water availability, while anomalies identify departures from typical conditions. Monthly totals and
  means describe the seasonal cycle; anomalies are calculated against reference-month means.
- **SPI and drought events.** Precipitation deficits over different durations help characterize the
  climatic conditions associated with drought. The Standardized Precipitation Index (SPI) standardizes
  3-, 6- and 12-month rainfall accumulations against the reference climate to identify dry periods and
  summarize their duration and severity.
- **Rainfall extremes and dry spells.** Heavy rainfall and extended rain-free periods help characterize
  climatic pressures associated with flooding and water shortages. Annual maximum one- and five-day
  rainfall, mean wet-day intensity and consecutive dry-day durations summarize these features.
- **Persistence.** Persistence indicates whether wet or dry conditions tend to continue across months,
  which matters for prolonged water deficits or surpluses. Autocorrelation of monthly precipitation
  anomalies measures this dependence at lags of 1–12 months.
- **Trends.** Historical trends help assess whether the baseline climate has changed over the record.
  Theil–Sen slopes estimate change per decade, and start-year sensitivity tests how strongly that
  estimate depends on the selected period.
- **SPI fit checks.** Drought interpretation depends on how well the fitted distribution represents
  reference-period rainfall. Comparing distribution choices and SPI normality helps identify indices
  that need closer examination before use.

See the [processing rule reference](workflow-rule-reference.md#analyze-climate) for individual steps.

## Optional daily climate indices {#climate-indices}

Set `diagnostics.climate_indices.enabled: true` in the historical-analysis workflow file to calculate
both native-grid indices and indices of the daily basin series. This bundle is disabled by default
because cellwise percentile bootstrap and event retention can be expensive. The configuration starter
contains the settings; `reference_period: null` uses complete calendar years from `climate.window`.
An explicit `{start: 1991, end: 2020}` selects reference calendar years independently of the existing
`diagnostics.reference_period`, which controls water-year anomalies and SPI. Requested reference years
outside the extracted record count as missing; they do not trigger additional acquisition.

`diagnostics.climate_indices.seasons` maps names to ordered contiguous month lists, for example
`{DJF: [12, 1, 2], JJA: [6, 7, 8]}`. No seasons are inferred. Annual periods use the configured water-year
start; annual and seasonal labels use their start year. December 2000–February 2001 is `DJF:2000`.

### Definitions and reference conventions

Daily precipitation is in mm/day and TX/TN are daily maximum/minimum temperature in °C. Wet days have
precipitation at least `diagnostics.wet_day_threshold_mm` (default 1 mm). Temperature requires verified
source-owned TX/TN; mean temperature is not substituted. CHIRPS remains precipitation-only even when
its forcing store contains ERA5 companion temperatures.

| Product | Definition |
|---|---|
| CDD, CWD | Longest observed dry/wet run overlapping the reporting period, in days. |
| Wet-spell mean, median, p90 | Full lengths of complete wet events starting in the period. |
| R10mm, R20mm | Number of days with precipitation ≥10 or ≥20 mm. |
| Rx1day, Rx5day | Maximum daily or complete consecutive five-day rainfall within the period. |
| PRCPTOT, wet days, SDII | Wet-day total, count, and total divided by count. |
| R95p amount | Sum of daily rainfall strictly exceeding the reference wet-day 95th percentile. |
| R95p fraction | R95p amount divided by period wet-day rainfall, in percent; undefined at zero rainfall. |
| TXx, TNx | Maximum daily maximum/minimum temperature in the period. |
| TX90p, TN90p | Percentage of valid days strictly above the calendar-day reference 90th percentile. |
| WSDI | Days overlapping the period in fixed-reference TX exceedance runs of at least six days. |
| Heatwave days, longest heatwave | Total and maximum overlapping days in TX exceedance runs ≥3 days. |
| Heatwave count | Number of qualifying events starting in the period, when their count is known. |

Percentiles use Hyndman–Fan type 8. Precipitation thresholds pool reference wet days. Temperature uses
a five-day calendar window, circular within each reference year. February 29 is excluded from fitting
and uses February 28's threshold during evaluation. These calendar choices follow the
[PCIC implementation](https://github.com/pacificclimate/climdex.pcic); they are explicitly implemented
here rather than assumed equivalent to other packages' defaults.

For TX90p/TN90p within reference years, replace the evaluated reference year with each other reference
year in turn, evaluate the original daily observations, then average exceedance indicators. Out-of-reference
evaluation uses the retained full-reference thresholds. This also supports evaluation wholly inside the
reference period. All replacement replicates must have adequate support; failed replicates are not dropped.
WSDI and the three-day heatwave adaptation use fixed thresholds throughout so their day totals reconcile
with one event catalogue. They are not bootstrap-adjusted WSDI or extended-summer Climpact heatwaves.

`reference_minimum_fraction` defaults to 0.9, a toolbox quality choice rather than a universal ETCCDI rule.
Temperature thresholds need that fraction of expected five-day samples and at least two contributing
years; precipitation needs that fraction of reference observations and at least two wet days. At least
two reference years are required, but short references remain exploratory. Retained reference files
include thresholds, sample support, contributing temperature years and the temperature reference samples.

### Missingness, events and spatial meaning

Missing days break runs and never become dry days. Events are detected on the continuous retained record
before periods are sliced. The catalogue stores inclusive start/end dates, full observed duration,
censoring and heat-event mean/peak temperature. Period day totals use overlap; event counts use starts.
Unknown observations or an overlapping censored heatwave make the physical event count unavailable.
Censored duration metrics are lower bounds; complete-event distributions exclude censored segments and
can therefore be biased toward shorter events. No complete wet spells gives undefined duration summaries.

Period eligibility uses the existing monthly and yearly missing-day tolerances (the yearly tolerance
also applies to a configured season). Absent period edges count as missing. Eligible periods use observed
values without infilling. Reference support is assessed separately. NetCDFs retain valid/expected days
and eligibility per variable; unavailable or undefined values are NaN. Exact threshold equality never
counts as percentile exceedance. The event catalogue spans the full extracted record, including portions
outside the requested reporting periods.

Basin series are equal-weight means of available basin cells each day; contributing-cell counts are
retained. Their indices differ from spatial means of cellwise indices. Maps show means of available annual
cellwise indices on the source's native basin grid. Their source-specific colour scales and percentile
references do not establish agreement between datasets. These are meteorological pressures, not flood
impacts or hydrological drought measurements.

### Retained products and reruns

Each source writes an immutable `indices/<evaluation-id>/` bundle containing `annual_grid.nc`,
`annual_basin.nc`, optional `seasonal_*.nc`, `reference_*.nc`, `events_*.csv`,
`wet_spell_exceedance_*.csv`, `basin_contributing_cells.nc` and `calculation.json`. The completion record
identifies source/input hashes, settings, implementation and numerical-library versions, unavailable
variables, and payload checksums. `_engine/climate_indices.json` points to the completed bundle and is
written last. Readers verify its manifest and payloads before rendering. Changed scientific inputs or
settings create a new bundle; unchanged evaluations reuse verified products. Existing schema-2 diagnostic
tables and formulas retain their previous meaning, including their legacy precipitation totals.

Figures under `plots/indices/` include annual and configured-season basin panels, complete wet-spell
exceedance and one native-grid map per metric, retaining unavailable panels. The adjacent captions JSON
records interpretation and evaluation identity. This first bundle is source/basin based; existing optional
subbasin diagnostics continue separately. Comparison tables do not acquire new climate-index columns.
The retained-source mode of `scripts/render_figures.py` also renders enabled index bundles into its
separate output directory, preserving the original publication and its reference.

Failed calculations leave visibly incomplete staging directories without publishing their pointer.
Integrity failures stop reuse rather than overwriting evidence. Preserve the affected bundle for diagnosis
and move a damaged evaluation out of `indices/` before explicitly rerunning its compute job. Snakemake
tracks the completion pointer; it does not individually schedule missing internal payloads. Large grids
and long reference periods need substantial compute and memory; no locations are silently sampled or capped.
