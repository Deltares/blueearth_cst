# Default hydrological indicators

The Default set provides an initial overview of how climate conditions affect river flow and the
water balance. It selects indicators for the model variables retained by your simulation, so the
available results depend on your choice of outputs.

## Indicators by characteristic

The table groups all 18 indicators. Only those supported by the retained model variables are selected.
Except for return levels, results summarize annual values with equal weight for each contributing year.

| Category | Indicators | Definition | Units |
|---|---|---|---|
| Average flow | Annual mean discharge | Mean of annual mean river flow | m³/s |
| High-flow magnitude | Daily and seven-day maxima (2) | Mean annual maximum daily flow and seven-day mean flow | m³/s |
| High-flow magnitude | 95th-percentile discharge | Mean annual daily-flow 95th percentile (5% exceedance) | m³/s |
| Low-flow magnitude | Daily and seven-day minima (2) | Mean annual minimum daily flow and seven-day mean flow | m³/s |
| Low-flow persistence | Baseflow index | Mean annual ratio of minimum seven-day flow to annual mean flow | dimensionless |
| Seasonality | Wettest- and driest-month means (2) | Mean flow in the months selected from the unperturbed reference | m³/s |
| High-flow extremes | 10-year maximum-flow return level | Fitted from pooled annual maximum daily flows | m³/s |
| Low-flow extremes | 2-year seven-day low-flow return level | Fitted from pooled annual minima of seven-day mean flow | m³/s |
| Water balance | Annual precipitation, AET, PET and recharge (4) | Mean annual totals of the four retained variables | mm |
| Snow storage | Annual maximum snow water depth | Mean of annual maximum snow depth expressed as water | mm |
| Surface runoff | Annual mean overland flow | Mean of annual mean overland flow rate | m³/s |
| Thermal conditions | Annual mean air temperature | Mean of annual mean daily air temperature | °C |

AET and PET mean actual and potential evapotranspiration; recharge is signed net groundwater recharge.
The baseflow index here is a low-flow ratio, not a hydrograph-separated baseflow fraction.
Annual depth totals retain units of mm; interpretation as mm/year requires complete annual coverage.

Check `indicators.csv` in your results folder for each selected indicator's exact definition, units
and parameters. Discharge is a flow rate, whereas precipitation, evapotranspiration and recharge
are commonly reported as depths; do not compare their numeric values without checking units.
Annual mean temperature is the average of daily air temperatures, reported in °C, not a sum.

## Periods and data coverage

Annual calculations follow the configured water year. They coincide with January–December only
when the water year starts in January. No additional model warm-up period is removed by this set.

Seven-day calculations require seven consecutive nonmissing daily values. Windows can cross a
water-year boundary but never join the end of one weather realization to the start of another.

Summary indicators can retain partial years and handle missing values differently according to the
calculation. Inspect `block_coverage` in the indicator dictionary for the contributing days and
periods. An annual depth total in mm is equivalent to a mean annual depth in mm/year only when the
contributing years are complete and nonmissing.

## Comparing realizations and climate conditions

For ordinary group summaries, each realization receives equal weight. An undefined realization
can make the group result undefined; it should not be treated as zero or silently excluded.
Return-level calculations instead pool the eligible annual extremes across realizations.

Reference-month indicators use the recorded unperturbed reference at the same location. They
identify reference months from pooled monthly sums; retain the reference simulations when selecting
these indicators.

## Interpreting return levels

High-flow return levels are fitted to annual maxima; low-flow return levels use annual minima of
seven-day averages. Only finite annual values enter the fit. The calculation requires at least
the larger of ten values or the requested return period rounded up to a whole year, but passing
this minimum does not establish that the estimate is reliable. A group below the minimum, or one
whose fit the estimator refuses, is published as `unavailable` with the reason; other groups,
locations and indicators are still calculated. Values pool only members of one group, so five-year
runs give singleton groups five values and no return level.

The supplied method has been checked against a limited set of synthetic cases and supports only
its assessed probability range. It provides no uncertainty intervals and does not establish
accuracy for your basin. Treat the results as screening indicators, not design estimates,
especially for short records or long return periods.

Read the recorded return-level assessment and keep it with the indicator tables when sharing
results. See [return levels and their limits](../../../../docs/guide/outputs.md#return-levels-and-their-limits)
for the fitted parameters and interpretation checks.

## Select and read this set

Select `default` in your simulation configuration. Before running, retain the model variables
needed for your assessment. Start with the
[indicator guide](../../../../docs/guide/new-case.md#choose-indicators) for selection and the
[outputs guide](../../../../docs/guide/outputs.md) for table interpretation.
