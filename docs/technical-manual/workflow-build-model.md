---
title: "Hydrological model building"
format: html
---

Model building prepares a Wflow-SBM basin model and runs it with historical weather. Maps, simulated
time series and comparisons with available discharge observations help assess its behavior before
stress testing. Construction uses global datasets and includes no local calibration.

## Inputs {#inputs}

- **Spatial data.** Terrain, hydrography, land cover, vegetation, soils and waterbody datasets for
  constructing the basin model.
- **Historical climate data.** Selected meteorological forcing for the historical simulation.
- **Model-building recipes.** HydroMT settings, defaulting to `config/defaults/wflow_build_model.yml`
  and `config/defaults/wflow_update_waterbodies.yml`.
- **Reporting locations.** Optional CSV specifying locations for model outputs.
- **Discharge observations.** Optional CSV files for evaluation, with columns and units matching
  the modeled locations and variables.

See [input formats](data-formats-and-conventions.md) for file conventions and
[model preparation](model-preparation.md) for recipe roles and links to the upstream manuals.

## Configuration {#configuration}

**Project settings** define the basin, historical forcing and simulated variables.

| Setting | Purpose |
|---|---|
| `basin.region`, `basin.resolution` | Basin extent and model resolution. |
| `basin.output_locations` | Reporting locations. |
| `climate.selected`, `climate.window` | Historical forcing source and available record. |
| `model.outvars` | Variables saved for evaluation and later indicator calculations. |

**Workflow settings** control the historical simulation and evaluation. The annotated template is
`config/templates/example-basin/project_config_build_model.yml`.

| Setting | Purpose |
|---|---|
| `simulation_window.start`, `simulation_window.end` | Simulation years within the climate record. |
| `observations` | Observed CSV files associated with modeled variables. |
| `evaluation.subbasin_plots` | Additional evaluation figures for individual subbasins. |
| `engine.build_config` | Alternative HydroMT model-building recipe. |
| `engine.waterbodies_config` | Alternative waterbody-update recipe. |

Wflow treats the starting date as the initial state time. With daily forcing, the first output is one
day later; forcing must cover that first update through the requested end date. See
[execution and run management](configuration-and-execution.md) for running the workflow.

## Processing sequence {#rule-flow}

![Main scientific processing steps and conditional paths described below.](assets/workflow-wf1.svg){width=100% fig-alt="Scientific processing from basin and climate inputs to the assessment products described in this section."}

The workflow prepares basin and thematic data, builds the model and adds historical forcing on the
model grid. It then runs the historical simulation, exports response tables and produces evaluation
figures. Basin and forcing maps can be produced before the simulation finishes; additional climate
sources may be extracted for shared map scales without changing the selected forcing.

## Outputs {#outputs}

Paths below are relative to `project_dir`. The main outputs are:

- **Spatial data (`data/spatial/`).** Basin vectors, reporting-location registry, thematic maps and
  the spatial catalog.
- **Historical climate (`data/climate/historical/`).** The selected climate dataset and shared map
  reference, with additional map candidates where needed.
- **Basin model (`models/hydrology/wflow/`).** Model maps in `staticmaps.nc`, settings in
  `wflow_sbm.toml`, geometries and the HydroMT recipes used to build the model.
  `staticgeoms/outlet_index.csv` links outlets to reporting locations.
- **Historical forcing (`models/hydrology/wflow/forcing/`).** Model-grid forcing in
  `inmaps_historical.nc` and forcing-check figures.
- **Simulated time series (`models/hydrology/wflow/run_default/`).** Native Wflow `output.csv` and
  per-variable `output_<code>.csv` tables, with a gauge crosswalk linking model IDs to station names.
- **Evaluation (`models/hydrology/wflow/evaluation/`).** Model-response figures and scores where
  observations are available, including `performance_indicators.csv` and station discharge
  summaries under `plots/stations/`.

The completed model supplies subsequent hydrological simulations. Logs, benchmarks and configuration
archives are described in [execution records](execution-and-provenance.md).

## Diagnostics and interpretation {#interpretation}

- **Basin and forcing maps.** Maps show the model's spatial inputs and historical forcing.
  They support inspection of the basin setup before interpreting simulated responses.
- **Historical response.** Time series describe model behavior under the selected historical weather.
  Per-variable exports complement the native Wflow output and do not replace it.
- **Observation comparisons.** Available observations support evaluation of simulated behavior at
  matching locations. Without observations, response figures describe the simulation but do not
  establish model skill; a successful run alone does not establish predictive adequacy.

See the [processing rule reference](workflow-rule-reference.md#build-model) for individual steps.
