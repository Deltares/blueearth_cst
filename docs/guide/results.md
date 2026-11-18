---
title: "Stress-test results"
subtitle: "Read indicator tables and link them to climate perturbations"
---

After hydrological simulation and indicators finish, each [experiment](glossary.md#experiment) holds tables of
[indicator](glossary.md#indicator) values for every simulation and every climate perturbation. This page shows
where the tables are, what each one holds, how to check a value before using it and how to link it to the
climate perturbation it represents. Paths are relative to the [output
directory](glossary.md#output-directory).

## Table location {#where-the-tables-are}

Each selected [indicator set](glossary.md#indicator-set) has its own folder,
`experiments/<name>/results/indicator_sets/<set-id>/`: `default` unless `project_config_simulate_system.yml`
selects others. For example, a `default` set with river discharge indicators contains:

```text
experiments/stress_test_01/results/indicator_sets/default/
├── indicators.csv          # Indicator definitions, units and coverage
├── group_members.csv       # Runs included in each group
├── group_metadata.csv      # Climate perturbation and reference status of each group
├── q_run_indicators.csv    # Indicator values for each run and location
└── q_group_indicators.csv  # Indicator values for each group and location
```

Here, `q` means river discharge. Other selected variables have their own run and group tables.

## Table contents {#what-each-table-holds}

- **`indicators.csv`** defines each indicator: what the value measures, its units, reporting period and
  coverage, that is, how much usable data it rests on. Start here for the indicator you need.
- **`q_run_indicators.csv`** gives one value per [run](glossary.md#run) and reporting location.
- **`q_group_indicators.csv`** gives one value per [group](glossary.md#group): the runs that share one climate
  perturbation across realizations.
- **`group_metadata.csv`** gives each group's climate perturbation and whether it is the unperturbed
  reference; **`group_members.csv`** lists the runs in each group.

For annual totals such as precipitation or evapotranspiration, check `block_coverage` for missing days and
incomplete years. With the default set, totals in mm represent mean annual depth in mm/yr only when all
contributing years are complete and contain no missing values; see the
[default conventions](../../blueearth_cst/indicators/sets/default/README.md). The manual describes
[every column](../technical-manual/results-and-interpretation.md#indicator-table-structure).

## Check `status` before `value` {#check-status-before-value}

Every value has a `status` and, when needed, a `reason`:

- `available`: a value was calculated.
- `undefined`: the calculation is undefined under the set's conventions.
- `unavailable`: no value could be calculated. The reason names a missing parameter, or a data limit such as
  `5 annual blocks; 10 required` for a return level from too-short runs.

An unavailable value is not zero; never compare it as if it were. If a required response variable was not
saved, the calculation is left out of the value tables altogether: check its declaration and explanation in
`indicators.csv` and the `indicators.json` manifest, alongside the selected
[set conventions](new-case.md#choose-indicators). For return levels, read the manual's
[estimation methods and limits](../technical-manual/indicator-calculations.md#return-level-estimation-and-supporting-evidence);
for missing inputs, see
[input evaluation and recovery](../technical-manual/indicator-requests.md#inputs-evaluation-and-recovery).

## Link a value to its climate perturbation {#link-a-value-to-its-climate-perturbation}

Run values link to their scenario through `run_id`; group values link to their climate perturbation through
`group_id`. Use the [scenario set](glossary.md#scenario-set) that your experiment used: the same run or group
label in another experiment may refer to different simulations, and newer scenarios can replace the tables in
`scenarios/`. If that has happened, use the preserved tables at the paths recorded for the experiment.

The examples below read tables only. Replace `run_root`, `experiment`, `set_id` and the scenario set name with
your own:

```python
from pathlib import Path
import pandas as pd

run_root = Path("C:/path/to/case-output")
experiment = "stress_test_01"
set_id = "default"
scenario_set = run_root / "scenarios/stochastic-2040-2060"  # name listed in scenarios/collections.csv
results = run_root / "experiments" / experiment / "results/indicator_sets" / set_id
scenarios = pd.read_csv(
    scenario_set / "scenario_run_lookup.csv", dtype=str, keep_default_na=False
)
perturbations = pd.read_csv(
    scenario_set / "perturbation_lookup.csv", dtype={"st_id": str}, keep_default_na=False
)
```

1. Connect each simulation's indicators to its climate scenario using `run_id`:

   ```python
   runs = pd.read_csv(results / "q_run_indicators.csv", dtype={"run_id": str, "location": str})
   run_settings = runs.merge(scenarios, on="run_id", how="left", validate="many_to_one")
   ```

2. Connect each group's indicators to its climate settings using `group_id`:

   ```python
   groups = pd.read_csv(results / "q_group_indicators.csv", dtype={"group_id": str, "location": str})
   coordinates = pd.read_csv(results / "group_metadata.csv", dtype=str, keep_default_na=False)
   group_settings = groups.merge(coordinates, on="group_id", how="left", validate="many_to_one")
   ```

For either table, `st_id` then joins to `perturbation_lookup.csv` for the monthly changes. That lookup has
twelve rows per perturbed point, so this further join expands rows; it is not a one-row-per-indicator table.
The [unperturbed reference](glossary.md#unperturbed-reference) has an empty `st_id` and `is_reference=true` in
`group_metadata.csv`. Keep it explicitly: an inner join with the perturbations would drop it. Use
`group_members.csv` to inspect the runs behind each group, including single-run groups.

### Opening an indicator table in Excel {#opening-an-indicator-table-in-excel}

Run and group IDs such as `run_01` and `group_01` stay text when the CSV is opened. Import perturbation
`st_id` values as text; in pandas, read lookup tables with `dtype=str` and `keep_default_na=False` to keep the
empty reference `st_id`. See [identifier
conventions](../technical-manual/saved-results.md#joins-and-reference-membership).

## Current limits {#what-the-tables-do-not-give-you-yet}

These joins identify the climate perturbation behind each value; they do not build a
[response surface](glossary.md#response-surface), which needs a separate analysis of the selected indicators
and their scenario coverage.

## Next

- [Reading the outputs](outputs.md) — maps, climate diagnostics and model figures.
- [Running the workflows](running.md#calculate-different-indicators-from-existing-simulations) — calculate
  other indicators from saved simulations.
- [Troubleshooting](troubleshooting.md) — failed or incomplete runs.
