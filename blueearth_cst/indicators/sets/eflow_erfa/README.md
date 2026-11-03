# ERFA flow-regime indicators

The ERFA set describes changes in the magnitude, timing, variability and intermittency of river
flow. It calculates 35 annual metrics and reports their medians and interquartile ranges, giving
70 indicators per discharge series.

Use these indicators to compare flow regimes across climate conditions and identify changes that
may warrant ecological investigation. The set does not classify ecological risk or predict
responses of particular species or habitats.

## Indicators by flow characteristic

The table groups all 35 annual metrics. Each is reported as a median (`_med`) and interquartile
range (`_iqr`) across retained water years, giving 70 indicators. The median describes typical
conditions; the interquartile range describes year-to-year variability.

| Category | Indicators | Definition | Units |
|---|---|---|---|
| High-flow magnitude | 1-, 3-, 7-, 30- and 90-day maxima (5) | Annual maximum discharge averaged over each duration | m³/s |
| High-flow timing | Timing of maximum flow (1) | Water-year day of the first annual daily maximum | days |
| High-flow frequency | High-flow pulse count (1) | Number of events above the record's 90th percentile | count/year |
| High-flow duration | High-flow pulse duration (1) | Mean duration of those events within each year | days |
| Seasonality | Mean monthly discharge (12) | Mean flow for each calendar month, January–December | m³/s |
| Low-flow magnitude | 1-, 3-, 7-, 30- and 90-day minima (5) | Annual minimum discharge averaged over each duration | m³/s |
| Low-flow timing | Timing of minimum flow (1) | Water-year day of the first annual daily minimum | days |
| Low-flow frequency | Low-flow pulse count (1) | Number of events below the record's 10th percentile | count/year |
| Low-flow duration | Low-flow pulse duration (1) | Mean duration of those events within each year | days |
| Rate of change | Mean rise and fall rates (2) | Mean positive magnitude of daily increases or decreases | m³/s/day |
| Frequency of change | Rising and falling days (2) | Annual counts of daily increases and decreases | count/year |
| Intermittency | Zero-flow event count (1) | Annual number of sequences with discharge ≤ 0 | count/year |
| Intermittency | Zero-flow event duration (1) | Mean duration of those sequences within each year | days |
| Intermittency | Timing of longest zero-flow event (1) | Water-year day when the first longest sequence begins | days |

Numbers in parentheses count the annual metrics, before the two summaries are calculated.
Timing medians are water-year day numbers; timing interquartile ranges are differences in days.
Counts are recorded as dimensionless counts in the result dictionary; “count/year” above identifies
that events are counted within each annual block. Both summaries retain the underlying units.
Exact result-column names and definitions are listed in `indicators.csv`.

These calculations follow the fixed defaults of the toolbox's reference ERFA implementation.
They are IHA-style flow metrics, not an exact reproduction of every IHA or ERFA formulation.

## Annual periods and completeness

Annual blocks follow the configured water year. A year needs at least 90% finite daily values,
and a monthly mean needs at least 80% of that month's daily values. Leap days are included.
Inspect `erfa_year_coverage` to see which years were retained at each location.

The input must have a daily date sequence, with missing observations represented explicitly.
Missing dates cannot simply be removed. If no year passes the annual coverage check, the calculation
stops. Missing metric values are excluded separately when summarizing each metric; a metric with
no usable values remains undefined.

For a group of realizations, the toolbox averages the individual realization summaries with equal
weights. This is different from pooling all years and calculating one median or interquartile range.
An undefined member propagates to the group result.

## How flow events are identified

High-flow pulses exceed the record's 90th percentile; low-flow pulses fall below its 10th percentile.
Percentiles use linear interpolation and require at least ten finite observations. Thresholds are
estimated from the full supplied record, including data in years later excluded by annual coverage.

Each baseline or scenario series estimates its own thresholds. Pulse changes therefore describe
events relative to that series' flow distribution, not exceedances of a fixed historical threshold.
Use that distinction when interpreting changes across scenarios.

Intermittency calculations treat discharge at or below zero as a zero-flow condition. Event runs
are separated at water-year boundaries and missing values. When no event occurs, its count and
duration are zero; timing of an absent zero-flow event is undefined.

## Timing and rates of change

Complete trailing averages of 3, 7, 30 and 90 days, and daily flow differences, can cross water-year
boundaries. Rates are calculated without smoothing or a minimum-change cutoff; falling rates are
reported as positive magnitudes.

Timing is expressed as the day number within the water year, starting at one. Where an extreme or
longest zero-flow spell occurs more than once, the first occurrence is used. Timing summaries use
ordinary medians and interquartile ranges, not circular statistics, so events close to opposite
ends of the water year need particular care.

## Select and interpret this set

Use `eflow-erfa` as the set name in your simulation configuration and retain river discharge.
Compare both typical conditions and year-to-year spread, checking coverage and event definitions
before drawing ecological conclusions.

See [Choose indicators](../../../../docs/guide/new-case.md#choose-indicators) for configuration and
[Reading the outputs](../../../../docs/guide/outputs.md) for interpreting tables.
