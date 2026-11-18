---
title: "Glossary"
subtitle: "Terms used in the toolbox"
---

Each heading gives the preferred term. Where a setting or file holds the value, it is shown in code font.
"Replaces" lists other names you may meet for the same thing in older pages or messages.

## Case {#case}

The folder for one basin assessment. It holds shared input files in `data/` and one or more trials. Replaces:
application.

## Change factor {#change-factor}

The projected change between a reference period and a future period in a climate projection. By default,
temperature changes are in degrees Celsius and precipitation changes in percent.

## Climate perturbation {#climate-perturbation}

An imposed change to generated weather. Temperature changes are added in degrees Celsius; precipitation is
multiplied by a factor, so 0.7 means 30% drier. Replaces: stressor.

## Climate projection {#climate-projection}

Climate-model output (CMIP6) for future scenarios. It shows which perturbations are plausible; it does not
drive the weather generator. The optional delta-change method applies projected changes directly. Replaces:
GCM.

## Climate risk assessment {#climate-risk-assessment}

The full process of relating climate change to the performance of a water system, from historical analysis to
interpreting indicators. The toolbox supplies evidence for decisions; it does not make them.

## Climate window {#climate-window}

The historical calendar years extracted from the climate datasets (`climate.window`). It is not the model's
simulation period. Replaces: historical period.

## Data catalog {#data-catalog}

A YAML file that tells the toolbox where datasets are stored and how to read them (`project.catalog`).

## Engine folders (`_engine/`) {#engine-folders}

Folders holding the toolbox's own records, such as hashes, markers and run settings. You do not need to open
or edit them; everything you read sits outside them.

## Ensemble, member {#ensemble}

The selected climate models, scenarios and members for projection analysis (`ensemble`). A member is one
simulation of one climate model, such as `r1i1p1f1`.

## Experiment {#experiment}

A set of simulations made with one model, one scenario set and one group of simulation settings, in
`experiments/<name>/`. The first is `stress_test_01`; `--new-experiment` starts the next.

## Forcing recipe {#forcing-recipe}

A named choice of historical forcing datasets. Several recipes (`climate.forcings`, with `default_forcing`)
let one model run with different historical datasets. A single `climate.forcing` block is the recipe
`default`.

## Group {#group}

The runs that share one climate perturbation across all realizations, identified by `group_id`. Groups of
unperturbed runs are marked as reference groups (`is_reference`).

## Historical forcing {#historical-forcing}

The historical weather record chosen to drive the model and to condition the weather generator
(a named set under `climate.forcings`, selected with `climate.default_forcing`).

## Indicator {#indicator}

A number summarizing a simulated time series, such as mean annual flow or a 10-year flood. Replaces: metric.

## Indicator set {#indicator-set}

A named collection of indicators calculated together (`indicator_sets`), such as `default`, `eflow-erfa` or
`crop-water`.

## Layout {#layout}

The version of the output-directory structure, recorded in `config/_engine/layout.json`. The toolbox refuses
an output directory written with a different layout.

## Model {#model}

A Wflow-SBM basin model in its own folder, `models/hydrology/<name>/` (`model.name`, default `wflow`).

## Output directory {#output-directory}

The folder where the toolbox writes all results for a trial (`project.project_dir`). The starter uses the
trial's `outputs/` folder. Replaces: project directory, output root, run root, product root.

## Perturbation grid {#perturbation-grid}

All combinations of the temperature and precipitation levels (`n_levels` per axis), plus the unperturbed
reference. Replaces: stress-test grid.

## Preview (dry run) {#preview}

A command that lists the planned tasks without executing them: `--dry-run`, or `-- --dry-run` with
`scripts/run_workflows.py`.

## Project file {#project-file}

A trial's `project_config.yml`. It sets the basin, data, model and output directory, and selects which
workflows run. Replaces: project configuration.

## Realization {#realization}

One generated daily weather sequence. Several realizations sample natural variability (`n_realizations`).

## Registered model {#registered-model}

A model prepared or changed outside the toolbox and recorded with `scripts/register_model.py register`. The
toolbox uses it but never rebuilds it.

## Reporting location {#reporting-location}

A point where the model saves results, listed in the file named by `basin.output_locations`. Replaces: output
location, station, gauge.

## Response surface {#response-surface}

An indicator displayed across the perturbation grid, showing how it changes with temperature and
precipitation.

## Run {#run}

One simulation of one scenario, identified by `run_id`.

## Run command {#run-command}

A script that starts workflows: `scripts/run_workflows.py` for all enabled workflows,
`scripts/run_workflow.py` for one workflow and `scripts/simulate_system.py` for simulations. Replaces: runner,
owned runner, launcher.

## Run summary {#run-summary}

The `summary.md` file written for each launch in its own folder under `logs/`.

## Scenario {#scenario}

One generated weather series: one grid point, or the unperturbed reference, for one realization.

## Scenario set {#scenario-set}

All scenarios from one generation, in `scenarios/<name>/`. Replaces: scenario collection, collection.

## Schema version {#schema-version}

The format version of the configuration files (`schema_version`). Current case files use 3.

## Source {#source}

A named dataset entry in a data catalog, such as `era5`.

## Stress test {#stress-test}

Generating weather across a grid of imposed temperature and precipitation changes and simulating the basin's
response to each.

## Toolbox folder {#toolbox-folder}

The `blueearth_cst` folder created during installation. Run commands from this folder. Replaces: checkout,
repository.

## Trial {#trial}

One configuration of a case, with its own settings files and outputs. The default trial is named `default`.

## Unperturbed reference {#unperturbed-reference}

Generated weather with no imposed change, used as the baseline for comparison. It is not the historical
record. Result tables mark it with an empty `st_id` and `is_reference=true`. Replaces: reference, baseline.

## Water year {#water-year}

The twelve-month reporting year, starting in the month set by `climate.water_year_start` (default January).
Replaces: reporting year.

## Workflow {#workflow}

One of the five analyses: historical climate analysis, model building, projection analysis, scenario
generation, and hydrological simulation and indicators.

## Workflow file {#workflow-file}

The settings file for one workflow, named under `workflow_configs` in the project file. The starter
names them `project_config_<workflow>.yml`. Replaces: workflow configuration.
