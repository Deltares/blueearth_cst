---
title: "Workflow rule reference"
format: html
---

This reference lists the processing rules for each workflow. Names match the executable rules;
scenario generation also includes preparation steps performed by the launch command.

## Historical climate analysis {#analyze-climate}

See [historical climate analysis](workflow-analyze-climate.md) for inputs, settings and outputs.

| Rule | Operation |
|---|---|
| `delineate_region` | Define the shared basin region. |
| `delineate_subbasins_and_rivers` | Define basin, subbasin, river and location vectors. |
| `extract_climate_datasets_<source>` | Extract each historical dataset for the basin. |
| `produce_historical_map_reference` | Prepare the shared map extent and color classes. |
| `prepare_historical_map_fields_<source>` | Prepare native-grid map fields for each source. |
| `compute_climate_diagnostics_<source>` | Calculate and save temporal diagnostics. |
| `prepare_annual_spatial_anomalies_<source>` | Retain optional native annual P/T anomalies. |
| `plot_climate_diagnostics_<source>` | Produce source figures and captions. |
| `plot_climate_comparison` | Calculate and plot the common-period comparison. |
| `prepare_location_comparisons` | Sample cells at locations and gauge sites; save matched summaries. |
| `plot_location_comparisons` | Render coverage, annual overlays, climatology and monthly agreement. |

## Hydrological model building {#build-model}

See [hydrological model building](workflow-build-model.md) for inputs, settings and outputs.

| Rule | Operation |
|---|---|
| `delineate_region` | Define the shared basin region. |
| `delineate_subbasins_and_rivers` | Build shared vectors and location registry. |
| `extract_climate_datasets` | Extract the selected historical climate source. |
| `extract_map_candidates_<source>` | Extract candidates for shared historical display scales. |
| `produce_historical_map_reference` | Prepare shared frame/classes without requiring historical climate analysis. |
| `prepare_historical_map_fields_<source>` | Prepare native-grid map fields for each source. |
| `prepare_land_and_soil_maps` | Prepare thematic rasters for model building. |
| `build_wflow_model` | Build the Wflow-SBM model. |
| `add_reservoirs_lakes_glaciers` | Add waterbodies and glaciers. |
| `declare_gauges_and_outputs` | Configure gauges and Wflow output variables. |
| `add_climate_forcing` | Add model forcing and complete the model. |
| `write_gauge_index` | Map Wflow IDs to named stations. |
| `plot_basin_map` | Plot basin and thematic maps. |
| `plot_model_forcing` | Plot the model forcing grid. |
| `evaluate_budyko` | Optional observed upstream water balance against prepared forcing, without simulated Q. |
| `run_historical_simulation` | Run Wflow on the historical window. |
| `export_simulation_tables` | Export per-variable response tables. |
| `plot_model_evaluation` | Plot and score against available observations. |
| `write_run_metadata` | Record run identity and staleness metadata. |

## Climate-projection analysis {#analyze-projections}

See [climate-projection analysis](workflow-analyze-projections.md) for inputs, settings and outputs.

| Rule | Operation |
|---|---|
| `delineate_region` | Define the shared basin region. |
| `delineate_subbasins_and_rivers` | Build shared vectors. |
| `fetch_cmip6_projections` | Fetch raw slices by model, experiment and member. |
| `reduce_to_basin_averages` | Reduce each local slice to a monthly basin series. |
| `derive_change_factors` | Compare future and reference windows and write tables and report. |
| `prepare_projection_analysis` | Compute analysis-window diagnostic tables and availability. |
| `plot_climate_projections` | Render projection figures. |

## Climate-scenario generation {#generate-scenarios}

See [climate-scenario generation](workflow-generate-scenarios.md) for inputs, settings and outputs.

| Rule or launch-command preparation step | Operation |
|---|---|
| `delineate_region` | Define the shared basin region. |
| `extract_climate_datasets` | Extract the selected historical climate store. |
| `prepare_perturbation_grid` | Tabulate monthly temperature and precipitation changes. |
| `snapshot_generation_inputs` (launch-command preparation step) | Save source data and weather-generation settings. |
| `claim_scenario_collection` (launch-command preparation step) | Prepare storage for the scenario experiment. |
| `prepare_weather_generator_settings` (launch-command preparation step) | Write weather-generator input settings. |
| `generate_weather_realizations` | Generate unperturbed series. |
| `perturb_climate_realizations` | Transform every delivered scenario, including unperturbed references. |
| `publish_scenario_collection` | Check that all requested weather scenarios are complete. |

## Hydrological simulation and indicators {#simulate-system}

See [hydrological simulation and indicators](workflow-simulate-system.md) for inputs, settings and outputs.

| Rule | Operation |
|---|---|
| `simulations_only` (target) | Check completion of saved simulated time series |
| `simulations_and_indicators` (target) | Check completion of requested indicators |
| `write_model_fingerprint` | Record which basin model is used |
| `check_model_unchanged` | Check that the basin model has not changed |
| `snapshot_simulation_inputs` | Checkpoint: save selected scenarios and model settings |
| `downscale_scenario_series` | Prepare forcing per run |
| `run_wflow_simulations_batch_<batch>` | Run Wflow for groups of weather scenarios |
| `publish_wflow_outputs` | Save and check the simulated variables |
| `prepare_indicator_plan` | Checkpoint: check inputs for indicator calculations |
| `derive_system_indicators` | Calculate indicators and record data availability |
