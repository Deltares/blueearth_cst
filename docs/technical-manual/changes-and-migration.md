---
title: "Changes and migration"
format: html
---

This reference covers moving existing cases and output directories to the current toolbox structure,
renamed configuration settings, and the rules that apply when a trial is copied. The User Guide explains
how to [create a case and add a trial](../guide/new-case.md#create-the-case-files).

## Copying a trial

[Trial](../guide/glossary.md#trial) copying with `--from-trial` refuses inputs it cannot carry over safely.
Missing inputs, links, cross-trial inputs outside shared `data/`, retained-product references and schema-2
sources are refused with guidance. Migrate a schema-2 copy explicitly before using it as a source.
Review catalog-internal dataset paths and access separately; catalogs do not bundle their datasets.
Keep catalogs with relative dataset locators together with their datasets in shared case `data/`.
Copying a trial-local catalog with relative dataset dependencies is refused, rather than rewriting
HydroMT's catalog semantics or silently leaving datasets behind.

## The case README trial table

Creation adds one row to the case README's Trial/Purpose/Status table, initially `working`. No central
registry is updated. Change status manually to `retained` or `complete`; complete means the declared scope was
reviewed, not that every workflow ran. Preserve saved evidence for new
[experiments](../guide/glossary.md#experiment). Toolbox revision, source trial and run-note fields stay with
each trial.

## Copy and verify an existing case

Existing schema-2 and schema-3 split layouts remain runnable. The scaffold refuses a new case name
already present under `applications/` or `runs/` to avoid creating an ambiguous second case.
Migration is a separate, non-destructive operation:

1. Copy authored configurations and sidecars into a new case directory with a `default/` trial.
   Preserve the original configurations and results until verification is complete.
2. For schema 2, resolve each file path using its original base before explicitly changing the copy
   to schema 3. Review the [path reference](configuration.md#how-paths-are-read).
3. Set the copied trial's output root to `outputs`, and update copied YAML-relative input references.
   Keep shared inputs in case `data/`, with distinct inputs under the trial. Retain `toolbox:` locators.
4. Compose every workflow, inspect resolved input paths, and preview with the owned runner from a
   different invocation directory. Run a bounded workflow to check downstream execution paths.
5. Add the minimal README trial row and output ignore rule. Review and archive evidence separately;
   do not move or rewrite immutable product records as part of configuration migration.

An evidence archive preserves results, checks, configuration snapshots and run records. A rerunnable
archive additionally requires the referenced datasets, catalogs and software environment. Record missing
external dependencies. Git ignore rules are not backups, and moving YAML alone does not relocate runtime
locators embedded in historical outputs.

## Renamed configuration settings {#renamed-configuration-settings}

The current configuration uses simpler names and moves some settings into the project file. A settings
file that still uses an earlier key is refused with a message naming its replacement. Settings that were
only renamed keep their meaning, so completed results are reused.

| File | Earlier setting | Current setting |
|---|---|---|
| Project | `workflows.<name>: {enabled, config_path}` | `workflows.<name>: true` or `false`, and `workflow_configs.<name>: <path>` |
| Project | `climate.sources`, `climate.forcing`, `climate.selected` | Named sets under `climate.forcings`, and `climate.default_forcing` |
| Project | Model building `observations`; historical climate analysis `rain_gauges` | `observations.river_discharge` (`path`, `time_label`) and `observations.met_stations` |
| Project | Historical climate analysis `diagnostics.climate_indices.seasons` | `climate.seasons` (default DJF, MAM, JJA, SON; `{}` for annual only) |
| Historical climate analysis | `diagnostics.max_missing_days_*`; `pt_temperature_source` | Project `climate.map_reference`; removed, ERA5 temperature is always used |
| Historical climate analysis | `plotting.comparison`, `.source`, `.location`; `diagnostics.annual_spatial_anomalies` | `plotting.view` (`comparison`, `individual` or `both`) and `plotting.scales`; `plotting.annual_anomaly_maps` |
| Projection analysis | `catalog`, `reference_window`, `future_windows` (list) | Optional `projections_catalog`, optional `reference_period`, `future_periods` (named mapping) |
| Projection analysis | `variables` as a mapping of settings | `variables` as a list, with `variable_options` |
| Model building | `simulation_window`; `engine.build_config`, `engine.waterbodies_config` | `run_period`; `hydromt.build_config_path`, `hydromt.waterbodies_config_path` |
| Model building | `plotting.spatial`, `.forcing`, `.location`; `evaluation.budyko` settings | `plotting.input_maps`, `.forcing_figures`, `.scales`, `.view`; `evaluation.budyko: true` |
| Scenario generation | `simulation_window`, `weathergen_config` | `future_period`, `generator_config_path` |
| Scenario generation (delta change) | `projections: {ensemble, future_windows}`, `pet: {mode: projection_ratio}` | `future_periods`, `scale_pet: true`; `schema_version` is optional |
| Hydrological simulation | `operations`, `indicator_sets`, `indicator_set_definitions`, `indicator_selection`, `indicator_evaluation` | `steps: {simulations, indicators}` and `indicators: {sets, custom_sets, options, subset, evaluation}` |
| Advanced settings | `runtime.julia_threads` | `runtime.historical_julia_threads` |

The earlier `workflows.<name>: {enabled, config_path}` form is still accepted, but one project file cannot
mix it with the current form.

Some changes cause one-time reruns or changed outputs:

- A project that compared several datasets but used only one as forcing now declares one forcing set per
  dataset. Model building reruns once in the [per-set layout](workflow-build-model.md#named-recipes); start a
  new simulation experiment afterwards.
- Every compared dataset can now drive the model, so each must cover at least 16 years.
- Climate indices gain seasonal rows; set `climate.seasons: {}` to keep annual values only. Historical
  climate analysis always calculates the subbasin tables.
- Model building recalculates the forcing comparison once. With several forcing sets, the default
  `view: comparison` omits the per-run reporting-location figures.
- The bundled weather-generator settings file is now `config/defaults/weathergenr_config.yml`. Rerun
  scenario generation before running hydrological simulation on its own.

## Output-directory layout versions

A new output directory records the [layout](../guide/glossary.md#layout) it was written with in
`config/_engine/layout.json` (layout 5 for this toolbox version); a directory without it uses layout 1. The
[run commands](../guide/glossary.md#run-command) refuse a directory written with a different layout: set
`project.project_dir` to a new folder and rerun. Existing directories are not converted.
