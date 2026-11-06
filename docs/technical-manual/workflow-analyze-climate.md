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
| `climate.forcing.precip_source` | Optional forcing choice; comparison-only analysis needs none. |
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

### Compare daily rain gauges {#rain-gauges}

Optionally set `rain_gauges` in the historical analysis YAML:

```yaml
rain_gauges:
  stations_path: data/rain_gauge_stations.csv
  precip_path: data/rain_gauge_precip.csv
  time_label: interval_end
```

Omission or `null` disables the comparison. A populated mapping requires exactly these three fields.
Schema 3 resolves paths beside that YAML; schema 2 keeps its existing invocation-relative base.
The station CSV requires `id,x,y`, with optional display `name` and elevation `elevtn` (metres).
IDs are unique nonempty strings, including leading zeros. Coordinates are finite WGS84 longitude and
latitude in degrees. Elevation is metadata only. The precipitation CSV has `time` followed by literal
station-ID columns, in millimetres per daily interval. Names never control joins. Blank cells and
missing rows are missing; zero is dry weather. Duplicate dates/columns, negative/non-numeric values,
infinity and non-midnight timestamps are rejected. Convert provider missing-value codes to blanks.

`interval_end` means midnight UTC ending the accumulated day: `2000-02-01` belongs to January 31.
`interval_start` labels are shifted forward one day before matching. Both declarations require actual
midnight-to-midnight UTC intervals; prepare other accumulation windows upstream. Relabelling a local
09:00 accumulation does not convert it. Extracted source stores retain their interval-end labels.
Clipping and month/reporting-year attribution use the preceding covered day, without modifying stores.
An extraction missing the last required interval leaves the period incomplete.

Every declared precipitation candidate is sampled with HydroMT native-cell point sampling, including
single-source comparisons. Stations must lie inside/on the basin and within the actual source grid.
Outside points, unmatched IDs and nodata cells are reported; there is no interpolation, nearest-valid
search, elevation adjustment or gap filling. Each station uses common valid dates across observations
and all candidates. Coverage retains individual counts, joint counts and the expected calendar.
Daily bias (source minus observed), MAE, RMSE and Pearson correlation use those same dates.
Percentage bias divides the summed difference by summed observations; zero denominators are unavailable.
Correlation is unavailable with fewer than two pairs or constant series.

Monthly and reporting-year totals require every actual day, including leap days. Reporting years use
start-year labels and `climate.water_year_start`. Monthly climatologies average the same complete months
for every series; `monthly.csv` retains contributing months for counting. These strict rules apply only
to rain gauges and do not change basin diagnostics' existing tolerated-gap summaries.

Run the usual historical analysis entry point. Products appear in
`data/climate/historical/rain_gauges/`: `tables/{stations,exclusions,coverage,agreement,monthly,annual}.csv`,
four comparison PNGs per supplied station under `plots/`, and one `_engine/` containing sampled daily
values/masks, evaluation provenance and figure manifests. Station metadata records safe filename tokens,
sampled-cell coordinates/CRS and distance. Figures show unavailable comparisons explicitly.
Titles and captions remain available in `plots/figure_titles.md` and `plots/figure_captions.md`.
Rerender retained products with `scripts/render_figures.py rain-gauges --project <run> --out <review>`.

This comparison supports forcing choice; it does not select or correct forcing. Point/grid support
differs, and gridded products may already incorporate these observations, so agreement is not independent
validation. The starter's two CSVs illustrate the format with synthetic values.

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
