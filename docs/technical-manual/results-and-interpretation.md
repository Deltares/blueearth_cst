---
title: "Results, evaluation and interpretation"
format: html
---

The results describe historical climate, simulated hydrological behaviour and responses to imposed
climate changes. Their interpretation depends on data coverage, reference conditions and the methods
used to calculate each indicator. This chapter explains these conventions and the evidence available
for evaluating results. The [output guide](../guide/outputs.md) shows how to inspect figures and
connect indicator tables to climate conditions.

## Output structure and result tables

The table below links to the outputs of each analysis. Paths are relative to the project output
directory, `project.project_dir`; each workflow page describes the files and when they are produced.

| Analysis | Main folder | Outputs |
|---|---|---|
| Historical climate | `data/climate/historical/` | [Coverage, diagnostics and comparisons](workflow-analyze-climate.md#outputs) |
| Model and historical run | `models/hydrology/wflow/` | [Model, forcing, responses and evaluation](workflow-build-model.md#outputs) |
| Climate projections | `data/climate/projections/<ensemble>/` | [Changes, coverage and figures](workflow-analyze-projections.md#outputs) |
| Generated weather | `scenarios/` | [Weather series and scenario lookups](workflow-generate-scenarios.md#outputs) |
| Scenario simulations | `experiments/<name>/` | [Daily responses and indicator products](workflow-simulate-system.md#outputs) |

Shared basin maps and vectors are under `data/spatial/`. Configuration copies are under `config/` or
with their scenario/experiment. The execution summary reports completion, warnings and failures;
a folder's existence alone does not establish completeness.

Location and scenario identifiers in CSV files are text labels. NetCDF files record the grid, dates,
variables and units needed to interpret climate and model arrays. Figure captions describe the period
and area represented, and explain missing results. A projection change table containing only column
headings means that no usable comparisons were available; it does not indicate zero climate change.

### Indicator table structure

Indicator results describe either an individual weather realization or a group of realizations.
The accompanying tables identify the climate conditions and simulations behind each value. They are
stored under `experiments/<name>/results/indicator_sets/<set-id>/`:

| File | Meaning |
|---|---|
| `indicators.csv` | Definitions, units and calculation methods, including reasons for missing results. |
| `<token>_run_indicators.csv` | Indicator values for each simulation and reporting location. |
| `<token>_group_indicators.csv` | Indicator values summarizing groups of simulations at each location. |
| `group_members.csv` | Simulations included in each group. |
| `group_metadata.csv` | Climate perturbations and reference status of each group. |

Each value table identifies the indicator and location, followed by `run_id` or `group_id`, and reports
`value`, `status` and `reason`. The definitions in `indicators.csv` explain the quantity being measured,
its units, reporting period, aggregation and treatment of missing data. They also identify requested
calculations that could not produce a value table.

The `block_coverage` field reports available and expected days for each reporting period, simulation
and location. A complete record can still be too short to characterize variability or extremes.
When a calculation is reported under an alternative indicator name, `equivalent_definition` identifies
the original calculation; the values are unchanged.

Scenario lookups connect runs to realizations and perturbations. `perturbation_lookup.csv` expresses
monthly temperature changes in °C and precipitation changes in percentages, although configuration
uses precipitation multipliers. Lookups must belong to the experiment being interpreted: current
exports under `scenarios/` can be replaced by later generation.

See the [table reference](retained-handoffs.md) for column definitions and the
[User Guide example](../guide/outputs.md#join-an-indicator-to-its-climate-settings) for connecting
indicator values to climate conditions.

## Evaluation and diagnostic products

### Historical forcing diagnostics

Historical climate diagnostics help assess whether the selected forcing represents the seasonal
patterns, variability, droughts and extremes relevant to the basin. Missing data and the fit of
statistical distributions affect these comparisons. Agreement between gridded datasets does not
establish accuracy against local measurements.

### Basin representation and historical model performance

Maps of basin boundaries, rivers and reporting locations support assessment of the model's spatial
representation. Forcing plots show the climate inputs applied to that model. Historical discharge
summaries describe daily flows, annual means, seasonal cycles and flow-duration curves, with
additional subbasin figures when enabled.

Comparison with observed discharge provides evidence of model performance where station identifiers,
dates and usable measurements match. Without observations, these figures describe simulated behaviour
but do not establish model skill.

See [model building](workflow-build-model.md) and
[observation formats](data-formats-and-conventions.md#discharge-observations).

### Weather-generator products and checks

Generated weather should represent the seasonality, variability and wet- and dry-spell behaviour
relevant to the assessment. The [weather-generation chapter](weather-generation.md#assumptions-and-applicability)
explains these checks and their limits. Available diagnostics depend on the generator recipe;
historical climate diagnostics are not automatically calculated for every scenario.

Matching the requested realization counts, dates and variables confirms that the expected weather
series are present. It does not establish that those series adequately represent basin climate.

### Checking saved indicators

The inspection command can list saved indicator sets without running Wflow. Replace the example
experiment path:

```powershell
pixi run python scripts/inspect_indicators.py --experiment "C:/cst-cases/runs/my-basin/experiments/stress_test_01" --list
```

To inspect a specific set, add `--manifest` with the path to its `indicators.json`, found under that
experiment's `_engine/indicator_sets/<id>/`. Inspection checks saved file consistency and structure.
The optional `--recheck` performs numerical checks only when the required matching sources, inputs and
dependencies are available; it can fail even when ordinary file inspection succeeds.
See [inspection and numerical rechecking](indicator-requests.md#integrity-and-numerical-rechecking).

## Reading indicator values

### Units, aggregation and reference comparisons

- **Units.** Use the indicator dictionary, rather than the filename: discharge indicators may describe
  duration, timing or variability instead of flow rate.
- **Coverage.** Definitions specify reporting years and incomplete-period handling; `block_coverage`
  records observed and expected days. Annual-total depths in mm equal mean annual depths in mm/yr
  only for complete, nonmissing annual blocks.
- **Aggregation.** Many indicators average realizations equally and propagate missing values;
  qualified return-level calculations instead pool blocks from individual runs. The definition
  determines the method, and `group_members.csv` identifies contributing runs.
- **References.** Generated reference groups have `is_reference=true` and an empty `st_id`.
  They differ from historical simulations, which supply some thresholds such as crop-water preparation.

See [table joins and reference groups](retained-handoffs.md#joins-and-reference-membership).

### Missing, undefined and unavailable results

| Row status | Meaning |
|---|---|
| `available` | A numerical result was calculated. |
| `undefined` | The calculation returned an undefined value; the value cell is empty. |
| `unavailable` | Required location-level input was missing; the reason identifies it. |

Never replace these empty values with zero. An absent response variable can omit the entire calculation
from value tables, while its declaration remains in the dictionary and saved set record.
`strict` evaluation refuses missing required inputs; `partial` allows explicit unavailable results.
Thus a successful partial run can contain no values for a requested indicator.
See [input evaluation](indicator-requests.md#inputs-evaluation-and-recovery).

### Extreme values and return levels

- **Fit evidence.** Keep CSVs with `indicators.json` and `return_level_benchmark.json`. Fit status,
  `return_level_validation` and `shape_coverage` describe benchmark coverage, not basin accuracy.
- **Parameters.** GEV fits use SciPy's `(c, loc, scale)` convention, with conventional shape ξ = −c.
  Results currently have no uncertainty intervals.
- **Eligible estimates.** Unsupported return periods and failed fits are not usable estimates.
  The full `flood_risk` example includes unsupported probabilities; eligible 2- and 10-year declarations
  still require the [documented sample checks](../../blueearth_cst/indicators/sets/flood_risk/README.md).

## Checking which results are current

- **Indicator sets.** Sets finish independently, so failures can leave older results beside new ones.
  Execution summaries and saved set records identify the calculation behind each table.
- **Projections.** Partial reruns can also leave mixed results. Attribution is recorded in
  `data/climate/projections/<ensemble>/_engine/execution/production.json`; the newest log alone is
  insufficient. See [run records](execution-and-provenance.md).
- **Preservation.** Keep derived spreadsheets outside managed result folders, which may be refreshed.
  Retain complete project outputs for later checks and use the
  [projection transfer command](results-transfer.md) for supported transfers.
