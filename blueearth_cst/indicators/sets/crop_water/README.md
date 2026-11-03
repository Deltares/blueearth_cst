# Crop-water screening indicators

The Crop-water set uses simulated precipitation, evapotranspiration, recharge and river discharge
to screen changes in seasonal water availability and water stress. It provides ten indicators
based on six calculations.

These are catchment-scale proxies, not calibrated crop-stress, irrigation-demand, yield or damage
estimates. Use them to identify conditions requiring more detailed agricultural or water-supply analysis.

## Indicators by water-availability characteristic

The table groups all ten indicators. Winter and summer versions are separate results, while low-flow
frequency is reported for summer only and recharge for the October–September year.

P is precipitation, PET is potential evapotranspiration, AET is actual evapotranspiration,
Q is discharge, and Qcrit is the fixed low-flow threshold for each reporting location.

| Category | Indicators | Definition over each reporting period | Units |
|---|---|---|---|
| Climatic water deficit | Winter and summer climatic deficits (2) | Sum of daily max(PET − P, 0) | mm/season |
| Dry-spell duration | Winter and summer longest dry spells (2) | Longest consecutive run with P < 1 mm/day | days |
| Evapotranspiration deficit | Winter and summer relative ET deficits (2) | 100 × (1 − sum(AET) / sum(PET)) | % |
| Evapotranspiration limitation | Winter and summer ET-limitation days (2) | Count AET/PET < 0.5 on days with PET ≥ 1 mm/day | days |
| Low-flow frequency | Summer low-flow days (1) | Count days with Q < Qcrit | days/season |
| Groundwater replenishment | Annual net recharge (1) | Sum signed net recharge from 1 October to 30 September | mm/year |

Numbers in parentheses identify separate indicators. Thresholds shown are supplied defaults;
check your result dictionary if you configure different thresholds. Exact indicator names combine
`winter_` or `summer_` with the measure; annual recharge is reported as `annual_recharge`.

Each reported value is the arithmetic mean of the complete contributing periods.
The indicator dictionary identifies the period and parameters for each selected calculation.

## Seasons and completeness

The supplied seasons are **winter: 1 November–31 May** and **summer: 1 May–31 October**.
Winter is labelled by its ending year. These seasons overlap in May and must not be summed.
Annual recharge uses 1 October–30 September, independently of the project's water-year setting.

These periods are fixed conventions of the supplied set, not a crop calendar inferred for your basin.
Check their relevance to local growing seasons and water management before interpreting them.

Daily values are labelled by the end of their interval. A missing required daily value invalidates
that period; partial periods at the edges of the record are reported and excluded. Dates must form
a regular daily sequence. Inspect the period-coverage diagnostics before comparing scenarios.

## Prepare the low-flow threshold

The low-flow indicator needs a fixed discharge threshold at each location. The toolbox can derive
it from the historical hydrological simulation in workflow (WF) 1. For example, in
`project_config_simulate_system.yml`:

```yaml
indicator_sets:
  - set: crop-water
    summer_low_flow_days:
      thresholds:
        baseline_quantile: {}
```

This requests the default **Q90 exceedance threshold**: the flow exceeded 90% of the time,
equivalent to the 10th percentile. By default, it uses complete May–October periods from all
available historical calendar years in the model's discharge record. It does not automatically
take its period from the project's climate-window setting.

The resulting threshold is held fixed across scenarios. It comes from historical modelled flow,
not observations or future scenario discharge, so its interpretation depends on historical model
performance. Incomplete or missing reference periods are excluded and reported; at least one
complete reference period is required at each location.

You can select particular historical years, another exceedance probability, or an annual reference
period. For example, Q80 uses an exceedance probability of 0.80. Choosing an annual reference
distribution changes the threshold, but the indicator still counts low-flow days during May–October.
See the [indicator-request reference](../../../../docs/technical-manual/indicator-requests.md)
for configuration details.

Run WF1 before evaluating this threshold. When running all workflows together, threshold preparation
takes place after WF1. A first-run preview may report it as pending; indicators-only evaluation
requires the historical model results already to exist.

If the threshold is missing, strict evaluation stops; partial evaluation reports affected locations
as unavailable. Alternatively, exclude the low-flow indicator. A zero threshold makes the strict
below-threshold count uninformative. Explicit operational thresholds can also be supplied, with
their units, source and reference-period information; do not mix them with an automatic baseline rule.

## Interpret the measures in their spatial context

Non-discharge inputs are daily subcatchment means. They do not isolate agricultural land or a
particular aquifer. Discharge represents the selected outlet or gauge and may measure drainage,
rather than water available for irrigation.

Recharge is signed net recharge to the saturated zone, not gross recharge or sustainable pumping
yield. Representation of irrigation return flows and canal leakage depends on the model setup.

AET and PET are modelled landscape quantities. Vegetation coefficients and evaporation components
affect their ratio, so ET deficits are screening proxies. Negative relative deficits are retained
when AET exceeds PET. A zero seasonal PET total makes the relative deficit undefined; no eligible
PET days makes the ET-limitation-day calculation undefined. Low-PET days are excluded from the
daily limitation count, but not from the checks for complete paired AET/PET data.

## Before using the results

Retain precipitation, PET, AET, recharge and discharge for the indicators that require them.
Check the period diagnostics for missing days, reference thresholds, eligible PET days and occasions
when AET exceeds PET. Compare scenarios using the same definitions and reference thresholds,
and keep those assumptions with any reported findings.

See [Choose indicators](../../../../docs/guide/new-case.md#choose-indicators) and
[Reading the outputs](../../../../docs/guide/outputs.md) for selection and table interpretation.
