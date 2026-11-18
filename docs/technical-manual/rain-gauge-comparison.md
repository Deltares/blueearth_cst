---
title: "Rain-gauge comparison"
format: html
---

Optionally set `observations.met_stations` in the project file:

```yaml
observations:
  met_stations:
    stations_path: data/rain_gauge_stations.csv
    precip_path: data/rain_gauge_precip.csv
    time_label: interval_end
```

## Settings

Omission or `null` disables observation input; gridded per-location comparison calculations still run when
multiple candidates are selected. Their figures require `plotting.scales: [basin, locations]`. A populated
mapping requires these three fields and accepts optional `temp_path`. Paths resolve beside the project file.

## Input files

- **Station CSV.** Requires `id,x,y`, with optional display `name` and elevation `elevtn` (metres). IDs are
  unique nonempty strings, including leading zeros. Coordinates are finite WGS84 longitude and latitude in
  degrees. Elevation is metadata only.
- **Precipitation CSV.** `time` followed by literal station-ID columns, in millimetres per daily interval.
  Names never control joins. Blank cells and missing rows are missing; zero is dry weather. Duplicate
  dates/columns, negative/non-numeric values, infinity and non-midnight timestamps are rejected. Convert
  provider missing-value codes to blanks.
- **Temperature CSV (`temp_path`).** The same `time` and station-ID columns for daily mean temperature in
  degrees Celsius; negative temperatures are valid. It uses the same time-label declaration as precipitation.

## Time labels

`interval_end` means midnight UTC ending the accumulated day: `2000-02-01` belongs to January 31.
`interval_start` labels are shifted forward one day before matching. Both declarations require actual
midnight-to-midnight UTC intervals; prepare other accumulation windows upstream. Relabelling a local 09:00
accumulation does not convert it. Extracted source stores keep their interval-end labels. Clipping and
month/reporting-year attribution use the preceding covered day, without modifying stores. An extraction
missing the last required interval leaves the period incomplete.

## Sampling and scores

Every declared precipitation candidate is sampled with HydroMT native-cell point sampling, including
single-source comparisons. Stations must lie inside/on the basin and within the actual source grid. Outside
points, unmatched IDs and nodata cells are reported; there is no interpolation, nearest-valid search,
elevation adjustment or gap filling.

Each station uses common valid dates across observations and all candidates. Coverage keeps individual counts,
joint counts and the expected calendar.

- Daily bias (source minus observed), MAE, RMSE and Pearson correlation use those same dates.
- Percentage bias divides the summed difference by summed observations; zero denominators are unavailable.
- Correlation is unavailable with fewer than two pairs or constant series.

## Monthly and annual totals

Monthly and reporting-year totals require every actual day, including leap days. Reporting years use
start-year labels and `climate.water_year_start`. Monthly climatologies average the same complete months for
every series; `monthly.csv` keeps contributing months for counting. These strict rules apply only to
per-location comparisons, including locations without gauges, and do not change basin diagnostics' existing
tolerated-gap summaries. Temperature summaries require complete matched periods independently of
precipitation and use means rather than totals.

## Comparison layout

Run the usual historical analysis entry point. The comparison area has two views:

```text
data/climate/historical/source_comparison/
  locations.csv
  plots/basin_average/
  plots/locations/<id>/
  tables/basin_average/
  tables/locations/        annual, monthly, coverage, agreement, native_cells (.csv)
  _engine/
```

The basin-average view compares gridded basin means. In the location view, each dataset is sampled at the
original coordinates of every declared spatial location and rainfall-gauge site. These are native-cell values,
not subbasin averages. `plotting.scales` continues to control the optional area-average figures inside
individual source stores.

The location tables are long tables keyed by `location_id`; `annual` and `monthly` add a `variable` column
(`precip` with `total_mm`, `temp` with `mean_degC`), and `agreement` a `scope` column (`daily`, `monthly`).
Exclusions are listed once, in `_engine/locations/exclusions.csv`.

## Location index

`locations.csv` maps each location ID to its name, coordinates, location role, spatial location code, Wflow
ID, associated subbasin and available observation columns.

- Declared locations use `location_<wflow_id>`; gauge-only sites use `station_<safe-token>`, preserving
  literal gauge IDs in `gauge_id`.
- A gauge joins an existing location when exactly one declared location is within 0.1 m of its original coordinates, measured on the WGS84 ellipsoid, and that location has no gauge assigned. Ambiguous or unmatched gauges keep separate site IDs. Climate sampling uses the gauge's original coordinates.
- Outside-basin and outside-grid sites remain indexed with explicit exclusions.
- An alternate temperature companion keeps its true source label and its configured preparation; its figures
  identify the saved extraction grid instead of claiming the original companion grid.

## Location figures

Each location has annual overlays and monthly climatology figures. Where observations exist, a monthly
agreement figure adds one panel per gridded dataset, common axis limits, a 1:1 line, Pearson r and the number
of complete paired months. Annual overlays and monthly panels adapt the comparison forms of Figures 8–12 in
*Development of a Climate Stress Test and Case Study for Zimbabwe* (University of Cincinnati, draft final
report to UNESCO, 24 November 2021), using CST presets.

Temperature annual means and climatologies use only genuine temperature carriers; monthly agreement requires
`temp_path`. The temperature scatter explicitly compares monthly means. Correlation includes the seasonal
cycle and does not measure anomaly skill or establish accuracy.

Precipitation monthly-coverage plots are omitted throughout historical climate analysis. Coverage and
completeness remain available in the saved tables. Figures show unavailable comparisons explicitly.

## Location tables and records

Per-location tables keep coverage, daily agreement, monthly/annual precipitation, native-cell coordinates/CRS,
exclusions and optional temperature means. Monthly agreement metrics keep paired-month counts, Pearson r, mean
bias and RMSE; precipitation errors are in mm/month and temperature errors in °C.

Each location's plot directory contains readable `figure_titles.md` and `figure_captions.md`; sampled daily
data live under `_engine/locations/` and machine figure inventories under `_engine/plots/locations/`. Basin
figure inventories live under `_engine/basin_average/`.

## Rerendering

Rerender both views with `scripts/render_figures.py wf0-comparison --project <run> --out <review>`. The
`rain-gauges` family renders the location view.

## Limits

This comparison supports forcing choice; it does not select or correct forcing. Point/grid support differs,
and gridded products may already incorporate these observations, so agreement is not independent validation.
The starter's two CSVs illustrate the format with synthetic values.
