---
title: "Changes and migration"
format: html
---

This reference covers moving existing cases and output directories to the current toolbox structure,
renamed and deprecated configuration settings, outputs written by earlier versions, and the rules that apply
when a trial is copied. The User Guide explains
how to [create a case and add a trial](../guide/new-case.md#create-the-case-files).

## Copying a trial

[Trial](../guide/glossary.md#trial) copying with `--from-trial` refuses inputs it cannot carry over safely.
Missing inputs, links, cross-trial inputs outside shared `data/`, references to saved products and schema-2
sources are refused with guidance. Migrate a schema-2 copy explicitly before using it as a source.
Review catalog-internal dataset paths and access separately; catalogs do not bundle their datasets.
Keep catalogs with relative dataset paths together with their datasets in shared case `data/`.
Copying a trial-local catalog with relative dataset dependencies is refused, rather than rewriting
HydroMT's catalog semantics or silently leaving datasets behind.

## The case README trial table

Creation adds one row to the case README's Trial/Purpose/Status table, initially `working`. No central
registry is updated. Change status manually to `retained` or `complete`; complete means the declared scope was
reviewed, not that every workflow ran. Preserve saved evidence for new
[experiments](../guide/glossary.md#experiment). Toolbox revision, source trial and run-note fields stay with
each trial.

## Copy and verify an existing case

Existing schema-2 and schema-3 split layouts remain runnable. Case creation refuses a new case name
already present under `applications/` or `runs/` to avoid creating an ambiguous second case.
Migration is a separate, non-destructive operation:

1. Copy authored configurations and accompanying files into a new case directory with a `default/` trial.
   Preserve the original configurations and results until verification is complete.
2. For schema 2, resolve each file path using its original base before explicitly changing the copy
   to schema 3. Review the [path reference](configuration.md#how-paths-are-read).
3. Set the copied trial's output root to `outputs`, and update copied YAML-relative input references.
   Keep shared inputs in case `data/`, with distinct inputs under the trial. Keep `toolbox:` paths.
4. Combine every workflow's settings, inspect resolved input paths, and preview with the all-workflow run
   command from a different working directory. Run a bounded workflow to check downstream execution paths.
5. Add the minimal README trial row and output ignore rule. Review and archive evidence separately;
   do not move or rewrite immutable product records as part of configuration migration.

An evidence archive preserves results, checks, configuration snapshots and run records. A rerunnable
archive additionally requires the referenced datasets, catalogs and software environment. Record missing
external dependencies. Git ignore rules are not backups, and moving YAML alone does not relocate runtime
paths embedded in historical outputs.

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
mix it with the current form. Archives and the saved forcing-set settings of earlier runs keep their original
bytes and meaning.

Some changes cause one-time reruns or changed outputs:

- A project that compared several datasets but used only one as forcing now declares one forcing set per
  dataset. Model building reruns once in the [per-set layout](forcing-comparison.md#named-recipes); start a
  new simulation experiment afterwards.
- Every compared dataset can now drive the model, so each must cover at least 16 years.
- Climate indices gain seasonal rows; set `climate.seasons: {}` to keep annual values only. Historical
  climate analysis always calculates the subbasin tables.
- Model building recalculates the forcing comparison once. With several forcing sets, the default
  `view: comparison` omits the per-run reporting-location figures.
- The bundled weather-generator settings file is now `config/defaults/weathergenr_config.yml`. Rerun
  scenario generation before running hydrological simulation on its own.

## Deprecated settings {#deprecated-settings}

These settings are still accepted; replace or remove them.

- **Historical climate analysis.** Old top-level `source_figures`, `location_figures` and `subbasin_figures`
  are still accepted with deprecation warnings (`view: both` and `locations`); conflicting declarations are
  errors.
- **Model building plots.** Legacy `forcing_plots`, `evaluation.station_plots` and `evaluation.subbasin_plots`
  are accepted with deprecation warnings; both evaluation switches map to the `locations` scale. Conflicting
  declarations are errors.
- **Model building `historical_run`.** The former `historical_run` setting, which saved named copies under
  `simulations/historical/`, is no longer used. It is accepted with a deprecation warning and ignored; remove
  it from the configuration.
- **Forcing sets.** `climate.forcings.<name>.temp_pet_source` is still read as `temp_source`; supplying both
  spellings is refused. With one forcing set, reading the configuration warns `temp_pet_source is deprecated;
  use climate.forcing.temp_source`; with several forcing sets it is accepted without a warning.
- **Scenario-set selection.** `scenario_collection.manifest_path` in the simulation file is deprecated.

## Schema 2 and earlier file forms {#schema-2-path-rules}

- `schema_version: 2` keeps file bases relative to the directory the command runs from. `workflow_configs`
  paths are always relative to the project file, including schema 2. Rain-gauge files under
  `observations.met_stations` also keep that base.
- Schema 2 uses these legacy path rules; changing the version alone does not convert paths.
- Legacy flat stochastic files remain supported under project schemas 2 and 3. Nested stochastic settings
  normalize to the same effective generation inputs. Existing stochastic collection selection and reservation
  behavior are preserved.
- Legacy source files remain supported with migration notices, and run-record archives keep their original
  bytes.

## Outputs from earlier versions {#outputs-from-earlier-versions}

### Historical climate analysis {#earlier-historical-climate-outputs}

- Older saved figure metadata remains readable in place. Rerendering reads projects written before layout 3
  from their per-table files.
- Rerunning annual departure maps retires previously managed PNGs from the older
  `plots/annual_anomalies/page_001/` layout after writing the replacement figures directly under `plots/`.
- The rain-gauge location view replaces the former comparison `plots/subbasins/`. The `rain-gauges` rerender
  family continues reading legacy `source_comparison/stations/` (`comparison/stations/` before layout 3) or
  `historical/rain_gauges/` products. Older flat basin tables are readable by the comparison rerender command.
  Rerunning historical analysis writes the new layout; obsolete generated directories from earlier runs should
  be retired separately after verifying the replacement products.
- Legacy climate-index bundles remain readable and are preserved; the current calculation does not delete
  existing case evidence. Spatial index maps are no longer generated; previously inventoried index maps are
  retired after successful rendering, and unrelated files are preserved.

### Map preparation after an upgrade {#map-preparation-after-an-upgrade}

Map preparation records normalize Python source line endings to LF, so equivalent toolbox folders on Windows
share code hashes. Input data, catalogs and environment locks keep byte-exact hashes. After upgrading from
byte-exact Python hashes, regenerate the prepared fields once. If historical climate analysis reports
`stale preparation identity`:

1. Run the single-workflow run command with your usual config and output directory, targeting
   `--target produce_historical_map_reference` and forwarding
   `-- --forcerun prepare_historical_map_fields_era5 prepare_historical_map_fields_chirps` (include only
   sources declared in your project).
2. Preview with `--dry-run`, then execute without it.
3. Resume the normal workflow command.

This rebuilds map fields and their reference without forcing climate extraction.

### Projection analysis {#earlier-projection-outputs}

- Individual `cmip6_delta_monthly_*.png` curve sheets are no longer produced. Previously managed copies are
  removed when their figure directory is regenerated.
- For older runs without saved gridded changes, figure rerendering reconstructs spatial changes from their
  local raw slices and scalar run records in the review destination; those inputs must still be available.
- The catalog adapter mappings added for projected PET can invalidate
  earlier raw-slice records, including P/T-only caches, under the existing catalog-integrity policy; unchanged
  output values do not imply unchanged cache keys across toolbox revisions.
- Delta-change generation needs gridded changes of inventory schema 2 or later. Run projection analysis alone
  with `-- --forcerun prepare_projection_analysis` to rewrite older gridded changes (inventory schema 1).

### Scenario sets {#earlier-scenario-sets}

- Sets from earlier versions stay where they are (`scenarios/_engine/data/<id>/`) and are listed in the index
  as `retained-<id>`.
- Projects generated before the current scenario-set layout may still hold top-level copies and
  `scenarios/_engine/current.json`; these are no longer refreshed and can be deleted.

### Hydrological simulation {#earlier-simulation-outputs}

- Old native-only results are not upgraded or rewritten: create new historical runs/experiments.
- Existing native-grid inventories keep their original checksum and reader semantics.
- Older run records without measured elapsed seconds remain unavailable.

## Delta-change/1 and /2 {#delta-change-1-and-2}

Versions 1 and 2 (explicit historical and projection files, `reader`, `simulation_window`) are read-only:
their saved collections under `scenarios/_engine/collections/` stay readable and selectable by
`scenario_collection: {name: retained-<id>}`, `{id: ...}` or `manifest_path`, but new generation refuses
them, and `scripts/generate_delta_scenarios.py` is retired. To regenerate:

1. Run `analyze_projections` for the ensemble and horizons you used as projection files.
2. Replace the settings with the `delta-change/3` block in
   [delta-change scenarios](delta-change-scenarios.md#configure). `historical_path`, `source_like`, `reader`,
   `reference_period`, `future_period`, `members` and `factor_layout` are removed; `simulation_window`
   becomes the projection-analysis reference period (or `historical_override.period`).
3. Generate through the run command. Periods and members now follow projection analysis, so the new set
   differs from the old one.

Unreleased `delta-change/3` sets written with `projections.window_id` before this layout must be
regenerated.

## Output-directory layout versions

A new output directory records the [layout](../guide/glossary.md#layout) it was written with in
`config/_engine/layout.json` (layout 5 for this toolbox version); a directory without it uses layout 1. The
[run commands](../guide/glossary.md#run-command) refuse a directory written with a different layout: set
`project.project_dir` to a new folder and rerun. Existing directories are not converted.
