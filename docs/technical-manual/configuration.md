---
title: "Configuration reference"
format: html
---

This reference specifies how project and workflow settings are combined, how file paths are resolved,
and which constraints apply. [Shared configuration](shared-configuration.md) introduces the assessment
choices; [Creating a project](../guide/new-case.md) explains how to prepare a case.
The annotated starter is under `config/templates/example-basin/`; `config/defaults/` contains the build
configuration files and generator settings read during execution.

## How configuration settings are combined {#composition-and-precedence}

### Project and workflow files

The project top level accepts only `schema_version`, `project`, `basin`, `climate`, `model`, `plotting`,
`workflows` and `workflow_configs`. `workflows.<name>` is a YAML boolean that selects the workflow;
`workflow_configs.<name>` is the optional path of its settings file. Keep declared files readable even
when their workflow is disabled. For the earlier form, see
[renamed settings](changes-and-migration.md#renamed-configuration-settings).
Omitting the file gives an empty settings mapping; workflow-required fields still apply when enabled.

Workflow files hold that workflow's settings. Shared section names and shared leaf names cannot be placed
there to override the project. Combining the settings reads each declared file and adds the workflow's switch
as `enabled` to the effective workflow mapping; the file path is not part of it. Workflow files cannot
override arbitrary project settings. Duplicate or unknown fields are rejected in sections with a fixed set of
allowed fields. Disabling a workflow leaves its previous files in place.

### Read scope

Combining the settings has an explicit read scope. A workflow's own file and sections declared by that entry
point are required to load; an unreadable unrelated file can be skipped with a diagnostic. Shared-name and
content checks still inspect every declared file that does load. Tools that inspect the complete configuration
load all readable declared files. This tolerance is not permission to omit an enabled workflow's required
inputs or bypass configuration capture and input checks.

### Advanced settings

`config/advanced_settings.yml` is loaded for every project. Its `constraints` are hard limits, `defaults`
provide values that supported project settings may override, and `runtime` specifies required software
versions. It is never a `--configfile` target. Only its listed keys are accepted.

## How paths are read

### Path bases

`schema_version: 3` resolves enumerated file fields beside the YAML declaring them. `workflow_configs` paths
are always relative to the project file. Supported schema numbers are integers 2 and 3; other values fail.
For schema 2, see [schema 2 path rules](changes-and-migration.md#schema-2-path-rules).

| Declaring file | File fields resolved by schema 3 |
|---|---|
| Project | `project.project_dir`, `project.catalog`, `project.reuse_data_from`, `basin.output_locations`, file-valued region entries, `observations` paths |
| model building | `hydromt.build_config_path`, `hydromt.waterbodies_config_path` |
| projection analysis | `projections_catalog` |
| scenario generation | `generator_config_path` |
| hydrological simulation | `scenario_collection.manifest_path`, `indicators.custom_sets.<id>` |

### Path forms

- Use `toolbox:config/...` for resources in the toolbox folder in use. A `toolbox:` path must stay
  inside that folder.
- Absolute paths remain supported; a foreign-platform absolute path is rejected.
- URLs and predefined HydroMT catalog names keep their spelling. Catalog-internal paths keep HydroMT
  semantics.
- There is no environment-variable expansion or search for a matching filename. The resolver expands `~` and
  normalizes slash-separated local paths.

### Example

`hydromt: {build_config_path: hydromt/build.yml}` in `settings/build-model.yml` reads
`settings/hydromt/build.yml`. A bundled build configuration file can be
`toolbox:config/defaults/wflow_build_model.yml`. Moving YAML files does not relocate saved outputs or
datasets. Archives preserve exact source bytes and resolved runtime paths.

### Case creation {#case-scaffolder}

The case creation command creates `<case>/<trial>/project_config.yml`, defaulting to trial `default`. A first
trial may have another name, but later creation then needs `--from-trial` until a default trial exists. It
writes `project.project_dir: outputs` and a shared outlet reference `../data/output_locations.csv`.

`--from-trial` requires schema 3, preserves scientific settings and workflow enablement, and copies
referenced authored trial inputs. Shared case data stay shared; external inputs are reported. References to
saved products must be removed from a working source copy before creating a fresh trial. `--catalog` uses
destination-YAML-relative paths or `toolbox:` resources. See [case creation](../guide/new-case.md).

## Shared fields

| Field | Meaning and constraints |
|---|---|
| `project.project_dir` | Writable product root; keep production cases outside the toolbox |
| `project.catalog` | One or more HydroMT catalogs, in the supplied order |
| `project.reuse_data_from` | Development aid: other projects' `project_dir` paths to reuse data from; see [data reuse](#data-reuse) |
| `basin.region` | HydroMT region mapping or its literal string representation |
| `basin.resolution` | Model grid spacing in degrees |
| `basin.sources` | Hydrography and matching basin-index catalog entry names |
| `basin.output_locations` | Optional location CSV; [format and snapping](input-formats.md#output-locations) |
| `basin.delineation` | Spatial-unit controls, including `snap_tolerance_m`; starter comments describe the options |
| `climate.forcings` | Named forcing sets; historical comparison uses their datasets; see below |
| `climate.default_forcing` | The set scenario generation uses and the default for model building and simulation |
| `climate.window` | Inclusive calendar-year `start`/`end`; historical record, not hydrological run period |
| `climate.water_year_start` | Reporting-year month, default January |
| `climate.map_reference` | Historical completeness for maps and historical-analysis diagnostics; default 3 missing days/month, 15/year |
| `climate.seasons` | Named contiguous month lists for climate indices; default DJF, MAM, JJA, SON |
| `observations.river_discharge` | Observed daily-mean discharge `path` and optional `time_label` (default `interval_end`); model evaluation, see [observations](#observations) |
| `observations.met_stations` | Station `stations_path`, `precip_path`, optional `temp_path` and required `time_label`; historical analysis, see [rain gauges](rain-gauge-comparison.md) |
| `model.name` | Folder under `models/hydrology/` that model building creates or runs and hydrological simulation uses; default `wflow` |
| `model.outvars` | Saved model variables; declarations must use supported Wflow display names |
| `plotting.annotation` | `none` (default), `caption` or `title` on ordinary PNG paths |

The climate window must span at least the 16 years set by `constraints.min_historical_years` in
`config/advanced_settings.yml`. Extraction uses the available overlap, warns on narrowing, and refuses no
overlap or insufficient selected-source coverage. CHIRPS selection requires overlap with its companion
variables (ERA5 by default). Model building checks forcing coverage for its declared simulation window,
including Wflow's initial-state time convention.

`climate.map_reference` accepts only `max_missing_days_month` (integer 0–27) and
`max_missing_days_year` (nonnegative integer). Booleans and fractions fail. Historical climate analysis
diagnostics use these values. Model building forcing summaries keep their independent default 3/15 policy.
Completeness and temporal aggregation are distinct from the inclusive extraction dates.

### Observations {#observations}

Observed series live in the project file, because several workflows read them:

```yaml
observations:
  river_discharge:
    path: data/observed_discharge.csv
    time_label: interval_end   # or interval_start
  met_stations:
    stations_path: data/rain_gauge_stations.csv
    precip_path: data/rain_gauge_precip.csv
    time_label: interval_end
```

Discharge is daily mean m³/s over midnight-to-midnight UTC days, one column per `wflow_id`
([format](input-formats.md)); `interval_start` labels are shifted one day before the model evaluation,
historical forcing comparison and Budyko screening read them. Discharge evaluation needs
`river discharge` in `model.outvars`. Unknown entries and keys are refused.
Earlier locations of these settings are refused; see
[renamed settings](changes-and-migration.md#renamed-configuration-settings).

### Historical forcing sets {#historical-forcing-recipe}

`climate.forcings` names one or more forcing sets; each body lists its datasets (keys below). Historical
comparison analyzes each dataset the sets use once per variable: every `precip_source` and every
temperature/PET source (ERA5 unless a set names another), with native/default source preparation.
Projection analysis does not use the sets. Precipitation must be `era5`, `chirps` or `chirps_global`;
unsupported datasets fail before execution. Omission and explicit defaults are equivalent. The
[annotated starter](../../config/templates/example-basin/project_config.yml) shows the keys.

```yaml
climate:
  forcings:
    era5: {precip_source: era5}
    chirps: {precip_source: chirps, temp_source: era5}
  default_forcing: era5
```

Keep the sets in the project file: model building and scenario generation share them.
`climate.default_forcing` names the set that drives scenario generation and the default for model building
and simulation. It is optional with one set, and with several sets while only historical comparison and
projection analysis run; choose it after reviewing the comparison. Model building, scenario generation
and simulation refuse several sets without it. Order never selects the default.

Defaults fill omitted settings, never missing data. ERA5 supplies all required companion fields by default,
including for CHIRPS precipitation. A different `temp_source` must supply every required field; individual
variables are not silently borrowed from ERA5. The forcing set derives seven total fields for de Bruin, six
for Makkink and precipitation/temperature/PET for direct mode. Supplied nulls, wrong types, unknown keys and
empty names are errors. Entry-point summaries show resolved choices before execution.

### Several forcing sets {#named-historical-forcing-recipes}

Names match `[a-z][a-z0-9_]{0,31}`; `default` is reserved, and two names may not describe the same forcing
set. With one set, model outputs use the single-set layout (`historical/default/`, `wflow_sbm.toml`). With
several, model building simulates `build_model.historical_forcings` (default: the default set) in the per-set
layout described in [forcing-set comparison](forcing-comparison.md#named-recipes), and simulation selects one
set through `simulate_system.historical_forcing`.

#### Earlier keys

The earlier `climate.sources`, `climate.selected` and `climate.forcing` keys are refused;
[renamed settings](changes-and-migration.md#renamed-configuration-settings) explains the conversion and its
one-time reruns.

#### Forcing-set keys

| Key | Default and allowed meaning |
|---|---|
| `precip_source` | Required precipitation dataset: `era5`, `chirps` or `chirps_global` |
| `temp_source` | `era5`; temperature, extrema, pressure and radiation required by the selected PET method |
| `orography_source` | `era5_orography`; source elevation entry; required explicitly for another companion |
| `temp_correction` | `true`; boolean controlling lapse correction at extraction and model preparation |
| `press_correction` | `true`; boolean controlling pressure correction during computed-PET preparation |
| `pet.mode` | `computed` or `direct` |
| `pet.method` | `debruin` (default) or `makkink`; computed mode only |
| `pet.source` | Required catalog entry for supplied reference PET in direct mode; forbidden in computed mode |

#### Direct PET

Direct mode forbids both `pet.method` and an explicitly supplied `press_correction`, even if false.
It prepares temperature and transports the supplied PET through HydroMT-Wflow's `setup_pet_forcing`.
Supply reference PET, not actual evapotranspiration. Direct PET is available for historical analysis and
model forcing; scenario generation and scenario simulation reject it before preparation. No PET
perturbation model is implied.

#### Companion fields

Computed-PET forcing sets require companion `temp`, `temp_min`, `temp_max`, `press_msl` and `kin`, plus
`kout` for de Bruin. Direct-PET forcing sets require companion `temp` and supplied `pet`. Precipitation
defines the extraction grid. Sources are clipped to their actual daily overlap; gaps, duplicates, unsupported
calendars, absent variables and invalid required-domain values fail instead of being padded. Supplied
PET must be finite and nonnegative inside the required basin; masked cells outside it are allowed.

#### Temperature and pressure corrections

When a different companion is regridded, its declared source elevation references the lapse correction
onto the hydrography's accompanying file. Subsequent model preparation uses that file. Disabling temperature
correction disables both steps. Same-source ERA5 extraction reads native temperature without applying
lapse correction; its configured elevation is used during preparation. Pressure is corrected only during
PET preparation.

#### Alternative inputs

The reviewed ERA5/CHIRPS catalog arithmetic remains supported. An alternative canonical-unit input must
declare `metadata.time_label: interval_end` in its catalog and actually deliver daily midnight fields in
mm/day, degrees Celsius, hPa and W/m² as applicable. This declaration describes the daily interval ending
at each timestamp; regular midnight timestamps alone do not establish it.

Required-variable unit scaling must be identity arithmetic, and variable attributes are checked. Supplied PET
may also use HydroMT's native `mm` label for a daily total; it is equivalent to mm/day only under this
enforced one-day interval contract. No additional conversion profile or new data product is supplied.
Controlled fixtures establish the mechanism; real alternative companions and independent PET products require
separate source qualification.

#### Extraction record {#extraction-identity}

The extraction record contains sources, required variables and corrections performed during extraction.
Non-default extraction settings select a separate store. Preparation-only pressure settings, and
same-source elevation/lapse settings, do not change that store or the generator's record of inputs.

Scenario simulation compares extraction settings with the selected collection's archived creator settings and
applies the current preparation settings. Non-default preparation records use version 3; saved version-2
records keep their original correction settings. The generating code remains part of the scenario set's record
of inputs. Numerical backward comparisons therefore use equivalent fixed inputs and seeds.

#### CHIRPS diagnostics

CHIRPS diagnostics remain precipitation-only contributors even when their forcing stores borrow
temperature/PET fields. Captions identify the companion or supplied-PET source when those fields are shown.

## Reuse data from other trials {#data-reuse}

For quick development runs, a trial can copy data a sibling trial already extracted instead of reading the
catalog sources again. Name the siblings' product roots in the project file, or add them for one run:

```yaml
project:
  project_dir: outputs
  reuse_data_from: [../v2/outputs]
```

```powershell
python scripts/run_workflows.py --config showcase/project_config.yml --reuse-data-from v2/outputs
```

Sources are only read.

- A historical climate store is reused when its extraction record names the same extraction code, source,
  forcing set, hydrography entry, region geometry and catalog entries, and its data cover the requested
  window; the requested window is then sliced from it and has the content of a fresh extraction.
- Catalog entries are compared without the catalog root, but the shipped P-drive and local-mirror catalogs
  describe ERA5 and CHIRPS differently, so reuse works between trials that use the same catalog.
- Stores extracted before records existed, or by other extraction code, are never reused.
- A CMIP6 raw slice is reused when its recorded hash matches, including slices fetched before this feature.

Each decision is logged by the extraction or fetch rule; reuse changes no rerun decision and no run's record
of inputs.

## Workflow settings

Use the complete annotated starter files for supported optional fields and example values.
The following links explain the settings for each workflow.

| Workflow | Settings and detailed reference |
|---|---|
| historical climate analysis | `diagnostics`, workflow-local `plotting`: [historical analysis](workflow-analyze-climate.md#configuration) |
| model building | `run_period`, `hydromt` build configuration files, workflow-local `plotting`, `evaluation`: [model](workflow-build-model.md#configuration) |
| projection analysis | `projections_catalog`, `ensemble`, `models`, `scenarios`, `members`, `variables`, `variable_options`, `reference_period`, `future_periods`, `stats`: [projections](workflow-analyze-projections.md#configuration) |
| scenario generation | `seed`, `n_realizations`, `future_period`, monthly perturbations, `generator_config_path` and capacity: [generation](workflow-generate-scenarios.md#configuration) |
| hydrological simulation | Collection, experiment, `steps`, `indicators` and `compute`: [simulation](workflow-simulate-system.md#configuration) |

`experiment_name` is optional. An omitted name selects family `stress_test`, initially `stress_test_01`;
a custom base selects its own numbered family. Plain execution selects the highest allocation, irrespective
of completion. `--new-experiment` advances it; an exact numbered name selects that experiment. Previewing with
`scripts/suggest_experiment_name.py` reserves nothing. Changed scientific simulation inputs require a new
experiment. See [saved results](saved-results.md#identities-and-reuse).

## Scalar response precision {#csv-precision}

Model building and hydrological simulation each accept this setting in their own workflow YAML:

```yaml
output:
  csv:
    decimal_places: 5
```

`decimal_places` defaults to 5 and accepts integers 0–15, excluding booleans. These are decimal places, not
significant digits. Historical model evaluation and thresholds, and the indicators of hydrological
simulations, use the rounded saved time series. Values smaller than half the last decimal unit can become
zero; use greater precision when very small flows matter. Missing values remain missing. Indicator-summary
formatting and native NetCDF precision are independent. The CSV and NetCDF sections can be used together.

Native Wflow CSVs are temporary conversion inputs. Verified per-variable CSVs and their metadata are the saved
format; there is no option to keep native CSVs or a compatibility option. Increasing precision after
conversion requires another simulation. New simulation experiments freeze precision in their response request.

## Optional native model grids {#netcdf-grid}

The model-building and hydrological-simulation workflow files each accept their own
`output.netcdf_grid` request. Omission or `null` disables it. Settings are independent: enabling
historical grids does not enable ensemble grids. The toolbox manages file paths.

```yaml
output:
  netcdf_grid:
    compressionlevel: 1
    variables:
      snowpack_dry_snow__leq_depth: snow
      snowpack_liquid_water__depth: snowwater
      river_water__volume_flow_rate: discharge
```

### Grid variables

Keys under `variables` are installed Wflow-SBM standard names; values are unique lowercase
snake_case NetCDF names. Coordinate names are reserved. Variables must exist in enabled model
components and have a native writer-supported shape; these checks run before timestepping.
Native soil-layer dimensions are preserved. Empty mappings, unknown keys and user-specified
paths reject. `compressionlevel` defaults to 1 and accepts integers 0–9, excluding booleans.

### File contents

Files follow the native simulation timestep, period and calendar, with interval-end labels.
Snow depths are end-of-step storages in mm: dry-snow water equivalent and liquid water,
respectively; neither is fractional snow cover. Discharge is the native timestep-average flow
in m3/s. Other variables keep Wflow's own units and temporal meaning. Native values and masks
are written by Wflow; metadata adds units from its standard-name API and CRS from staticmaps.

The [output guide](../guide/outputs.md#native-grids) gives the file locations. Open the files with xarray or
GIS software. One file holds all requested variables, including native soil layers where present.

### Storage

Grid storage scales with rectangular cells × timesteps × variable layers × runs. The run command reports a
conservative raw float64 estimate with no assumed compression saving and reserves space for saved grids and
recovery copies. Compression is lossless but its benefit depends on the data.

### Changes and archiving

- Changing historical grid variables requires rerunning the historical simulation; CSVs cannot reconstruct
  maps.
- Historical reruns replace their configured outputs, so archive completed cases that must be kept.
- Disabling output removes the declaration and preserves old files; an old file is not evidence that the
  current request produced a grid.
- Completed ensemble experiments freeze the request: use `--new-experiment` for a different request. Keep the
  complete experiment, including its inventory, when archiving grids.

### Grid checks

Before they are written or reused on retry, grids must contain finite values on each variable's native
support. Masked cells outside that support remain valid. Units, coordinates, CRS and layers are checked
against expectations captured directly from the initialized Wflow model in `<run>.toml.grid.toml`, beside its
run settings. The saved inventory embeds these expectations, so archived reads do not need Julia or the
original model-input paths.

## Plotting resources

Python plotting adapters load `config/plotting/colors.yml` and `style.yml` from the toolbox. These affect
every project using that toolbox folder. The [local plotting reference](../../config/plotting/README.md)
explains how to change these settings. Appearance changes schedule affected plots without changing the
scenario or simulation record of inputs; model building's combined evaluation job also refreshes its
indicators. R generator figures keep their own settings. Changing colors does not change normalization,
classification or scientific missingness rules.
