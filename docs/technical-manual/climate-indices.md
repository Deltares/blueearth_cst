---
title: "Daily climate indices"
format: html
---

Set `diagnostics.climate_indices.enabled: true` in the historical-analysis workflow file to calculate
both native-grid indices and indices of the daily basin series. This bundle is disabled by default
because cellwise percentile bootstrap and event retention can be expensive. The configuration starter
contains the settings. Index thresholds are fitted on `diagnostics.reference_period` (default
`climate.window`), read as calendar years with those labels; with a non-January `climate.water_year_start`
the index reference therefore starts in January of the first labelled year. Requested reference years
outside the extracted record count as missing; they do not trigger additional acquisition.

The project file's `climate.seasons` maps names to ordered contiguous month lists, for example
`{DJF: [12, 1, 2], JJA: [6, 7, 8]}`. Omitted, it is DJF, MAM, JJA and SON; `{}` reports annual indices only. Annual periods use the configured water-year
start; annual and seasonal labels use their start year. December 2000–February 2001 is `DJF:2000`.

## Definitions and reference conventions

Daily precipitation is in mm/day and TX/TN are daily maximum/minimum temperature in °C. Wet days have
precipitation at least `diagnostics.wet_day_threshold_mm` (default 1 mm). TX/TN indices require verified
source-owned extrema; mean temperature is not substituted. Mean-temperature counts separately require
the source's actual daily `temp` variable, not an estimate from TX/TN. CHIRPS remains precipitation-only
even when its forcing store contains ERA5 companion temperatures.

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
| Mean-temperature counts | Days with daily mean T <0°C, and T above the configured threshold (default 25°C). |
| TX90p, TN90p | Percentage of valid days strictly above the calendar-day reference 90th percentile. |
| WSDI | Days overlapping the period in fixed-reference TX exceedance runs of at least six days. |
| Heatwave days, longest heatwave | Total and maximum overlapping days in TX exceedance runs ≥3 days. |
| Heatwave count | Number of qualifying events starting in the period, when their count is known. |

### Mean-temperature counts

The mean-temperature counts use strict comparisons and a 25°C default upper boundary. Configure the upper
boundary with `diagnostics.climate_indices.mean_temperature_threshold_degc`; it must be finite. These products
are `mean_temp_below_zero_days` and `mean_temp_above_threshold_days`. They are not minimum-temperature frost
days, maximum-temperature summer days or heatwave events.

They count observed qualifying days without missing-day scaling, use the same period-eligibility rules as
other absolute indices, and need no percentile reference support or bootstrap. The boundary is kept in NetCDF
attributes, publication identity, captions and figure labels. Both annual and configured-season results are
produced.

### Percentile thresholds

Percentiles use Hyndman–Fan type 8. Precipitation thresholds pool reference wet days. Temperature uses
a five-day calendar window, circular within each reference year. February 29 is excluded from fitting
and uses February 28's threshold during evaluation. These calendar choices follow the
[PCIC implementation](https://github.com/pacificclimate/climdex.pcic); they are explicitly implemented
here rather than assumed equivalent to other packages' defaults.

### Evaluation inside the reference period

For TX90p/TN90p within reference years, replace the evaluated reference year with each other reference
year in turn, evaluate the original daily observations, then average exceedance indicators. Out-of-reference
evaluation uses the saved full-reference thresholds. This also supports evaluation wholly inside the
reference period. All replacement replicates must have adequate support; failed replicates are not dropped.

WSDI and the three-day heatwave adaptation use fixed thresholds throughout so their day totals reconcile
with one event catalogue. They are not bootstrap-adjusted WSDI or extended-summer Climpact heatwaves.

### Reference support

`reference_minimum_fraction` defaults to 0.9, a toolbox quality choice rather than a universal ETCCDI rule.
Temperature thresholds need that fraction of expected five-day samples and at least two contributing
years; precipitation needs that fraction of reference observations and at least two wet days. At least
two reference years are required, but short references remain exploratory. Saved reference files
include thresholds, sample support, contributing temperature years and the temperature reference samples.

## Missingness, events and spatial meaning

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
retained. Their indices differ from spatial means of cellwise indices. Basin and upstream-subbasin
indices use daily equal-weight means of available native cells before index calculation. Locations
sample the native cell containing their original coordinates. These are meteorological pressures, not flood
impacts or hydrological drought measurements.

## Saved products and reruns {#retained-products-and-reruns}

### Index table

Each source writes one `tables/climate_indices.csv` containing all registered indices for basin, upstream
subbasins and configured locations, for annual and configured seasonal periods. Rows keep support
identifiers, units, availability and missing-value reasons, including unavailable source variables and
out-of-domain locations. Original daily extraction and source-level `basin_cells.csv` remain the calculation
inputs. Existing diagnostic tables keep their previous formulas and precipitation totals.

### Supporting records

Only `_engine/climate_indices.json` (settings, input/code fingerprints and product hashes) and
`_engine/figure_data/wet_spell_exceedance.csv` (basin plotting data) support this CSV. New calculations save
no gridded index arrays, threshold arrays, event catalogues or daily contributing-cell counts, and create no
hashed directories.

Changed inputs/settings recalculate from the daily source in memory. Products are staged before replacement;
the record is written last. Readers verify code and product hashes.

### Index figures

Figures under `plots/indices/` split annual and configured-season basin panels into dry/wet spells, heavy
rainfall, rainfall amount and frequency, temperature extremes, and heatwaves. Only available indices and
scopes are emitted, alongside complete wet-spell exceedance when available. Compact sheets share unit labels
within compatible rows and year titles by column. Single-source plots omit empty boundary periods while
preserving internal gaps.

`figure_availability.json` lists emitted PNGs and omitted indices; the adjacent captions JSON records
interpretation and evaluation identity. The workflow owns this directory as one availability-dependent
rendering product. Comparison tables do not acquire new climate-index columns.

The saved-source mode of `scripts/render_figures.py` also renders enabled index bundles into its separate
output directory, preserving the original publication and its reference.

### Failures and compute

Calculation and input-drift failures preserve previous outputs. An interruption during replacement can leave
inconsistent products; checksum verification stops rendering until recalculation. Snakemake declares the CSV,
record and small plotting table separately, so missing products are scheduled again. Long reference periods
need substantial compute and memory; no locations are silently sampled or capped.
