---
title: "Configuration reference"
format: html
---

This reference specifies how project and workflow settings are combined, how file paths are resolved,
and which constraints apply. [Shared configuration](shared-configuration.md) introduces the assessment
choices; [Creating a project](../guide/new-case.md) explains how to prepare a case.
The annotated starter is under `config/templates/example-basin/`; `config/defaults/` contains recipes
read during execution.

## How configuration settings are combined {#composition-and-precedence}

`blueearth_cst/shared/config_composition.py` combines the configuration files. The project top level accepts
only
`schema_version`, `project`, `basin`, `climate`, `model`, `plotting` and `workflows`.
Each `workflows.<name>` stanza accepts only `enabled` and optional `config_path`.
Use YAML booleans for enablement. Keep declared files readable even when their workflow is disabled.
Omitting the file composes an empty settings mapping; workflow-required fields still apply when enabled.

Workflow files hold that workflow's settings. Shared section names and shared leaf names cannot be placed
there to override the project. Composition reads each declared file, adds its project-level `enabled`, and
removes `config_path` from the effective workflow mapping. Workflow files cannot override arbitrary project
settings. Duplicate or unknown fields are rejected in sections with a fixed set of allowed
fields. Disabling a workflow leaves its previous files in place.

Composition has an explicit read scope. A workflow's own file and sections declared by that entry point
are required to load; an unreadable unrelated file can be skipped with a diagnostic. Shared-name and
content checks still inspect every declared file that does load. Tools that inspect the complete configuration
load all readable declared files. This tolerance is not permission to omit an enabled workflow's required
inputs or bypass
configuration capture and input checks.

`config/advanced_settings.yml` is loaded for every project. Its `constraints` are hard limits, `defaults`
provide values that supported project settings may override, and `runtime` specifies required software
versions.
It is never a `--configfile` target. Exact keys are closed by `_ADVANCED_SETTINGS_SCHEMA` in
`blueearth_cst/shared/advanced_settings.py`.

## How paths are read

`schema_version: 3` resolves enumerated file fields beside the YAML declaring them.
`schema_version: 2` retains invocation-relative file bases. Workflow `config_path` is always relative to
the project file, including schema 2. Supported schema numbers are integers 2 and 3; other values fail.

| Declaring file | File fields resolved by schema 3 |
|---|---|
| Project | `project.project_dir`, `project.catalog`, `basin.output_locations`, file-valued region entries |
| model building | `engine.build_config`, `engine.waterbodies_config`, each `observations` value |
| projection analysis | `catalog` |
| scenario generation | `weathergen_config` |
| hydrological simulation | `scenario_collection.manifest_path`, `indicator_set_definitions.<id>.config_path` |

Use `toolbox:config/...` for resources in the executing toolbox checkout. A toolbox locator must stay
inside that checkout. Absolute paths remain supported; a foreign-platform absolute path is rejected.
URLs and predefined HydroMT catalog names keep their spelling. Catalog-internal paths retain HydroMT
semantics. There is no environment-variable expansion or search for a matching filename.
The resolver expands `~` and normalizes slash-separated local paths.

For example, `observations: {river discharge: data/q.csv}` in `settings/wf1.yml` reads
`settings/data/q.csv`. A bundled recipe can be `toolbox:config/defaults/wflow_build_model.yml`.
Moving YAML files does not relocate retained outputs or datasets. Archives preserve exact source bytes
and resolved runtime locators. `config_paths.py` and `tests/test_config_paths.py` implement these rules.

The case scaffolder creates `<case>/<trial>/project_config.yml`, defaulting to trial `default`.
It writes `project.project_dir: outputs` and a shared outlet reference `../data/output_locations.csv`.
`--from-trial` requires schema 3, preserves scientific settings and workflow enablement, and copies
referenced authored trial inputs. Shared case data stay shared; external inputs are reported.
Retained-product references must be removed from a working source copy before scaffolding a fresh trial.
`--catalog` uses destination-YAML-relative paths or `toolbox:` resources. Existing layouts and schema-2
execution are unchanged; see [case creation and migration](../guide/new-case.md).

## Shared fields

| Field | Meaning and constraints |
|---|---|
| `project.project_dir` | Writable product root; keep production cases outside the toolbox |
| `project.catalog` | One or more HydroMT catalogs, in the supplied order |
| `basin.region` | HydroMT region mapping or its literal string representation |
| `basin.resolution` | Model grid spacing in degrees |
| `basin.sources` | Hydrography and matching basin-index catalog entry names |
| `basin.output_locations` | Optional location CSV; [format and snapping](input-formats.md#output-locations) |
| `basin.delineation` | Spatial-unit controls, including `snap_tolerance_m`; starter comments describe the options |
| `climate.sources` | Candidate historical catalog sources |
| `climate.forcing.precip_source` | Precipitation choice from comparison candidates; infer only a sole candidate |
| `climate.forcing` | Optional selected-source companion, elevation, PET and correction recipe; see below |
| `climate.window` | Inclusive calendar-year `start`/`end`; historical record, not hydrological run period |
| `climate.water_year_start` | Reporting-year month, default January |
| `climate.map_reference` | Historical-map completeness; default 3 missing days/month, 15/year |
| `model.outvars` | Retained model variables; declarations must use supported Wflow display names |
| `plotting.annotation` | `none` (default), `caption` or `title` on ordinary PNG paths |

The historical-window parser is `historical_window_bounds` in `shared/wf3_science.py`.
`snake_utils.validate_historical_window` enforces the configured 16-year minimum. Extraction uses the
available overlap, warns on narrowing, and refuses no overlap or insufficient selected-source coverage.
CHIRPS selection requires overlap with its companion variables (ERA5 by default). Model building checks forcing
coverage for its declared simulation window, including Wflow's initial-state time convention.

`climate.map_reference` accepts only `max_missing_days_month` (integer 0–27) and
`max_missing_days_year` (nonnegative integer). Booleans and fractions fail. Historical climate analysis
diagnostic tolerances must
equal these shared values. Model building forcing summaries retain their independent default 3/15 policy.
Completeness and temporal aggregation are distinct from the inclusive extraction dates.

### Historical forcing recipe

`climate.forcing` applies when preparing the selected source for model building, scenario generation
or simulation. Historical comparison and projection workflows do not resolve this choice; historical
comparisons analyze the candidate datasets using their native/default source preparation.
Selected precipitation remains `era5`, `chirps` or `chirps_global`. Omission and explicit defaults are
equivalent. The [annotated starter](../../config/templates/example-basin/project_config.yml) shows the keys.

Keep this block in the project file: model building and independent scenario generation share it.
Comparison-only historical analysis needs no forcing selection; omit the block or use `{}`. When forcing
is needed, the whole block may be omitted only with one valid candidate in `climate.sources`.
With several candidates, set `climate.forcing.precip_source` explicitly. Candidate order and earlier runs
never select forcing. Empty, duplicate or unsupported candidates fail before execution.

Defaults fill omitted settings, never missing data. ERA5 supplies all required companion fields by default,
including for CHIRPS precipitation. A different `temp_source` must supply every required field; individual
variables are not silently borrowed from ERA5. The recipe derives seven total fields for de Bruin, six for
Makkink and precipitation/temperature/PET for direct mode. Supplied nulls, wrong types, unknown keys and
empty names are errors. Entry-point summaries show resolved choices before execution.

### Named historical forcing recipes {#named-historical-forcing-recipes}

To run one fixed model with several recipes, replace `climate.forcing` with a named map and a default:

```yaml
climate:
  sources: [era5, chirps]
  forcings:
    era5: {precip_source: era5}
    chirps: {precip_source: chirps, temp_source: era5}
  default_forcing: era5
```

Each body is an ordinary `climate.forcing` recipe and is validated the same way. Names match
`[a-z][a-z0-9_]{0,31}`; `default` is reserved. `forcings` and `default_forcing` are declared together, never
alongside `climate.forcing` or `climate.selected`, and two names may not describe the same recipe. Order never
selects the default. Scenario generation always uses the default. Model building simulates
`build_model.historical_forcings` (default: the default recipe), and simulation selects one recipe through
`simulate_system.historical_forcing`. Single-recipe projects need no migration. To adopt names, move the
`forcing` block under a name and set `default_forcing` to it. Outputs then use the per-recipe layout described in
[model building](workflow-build-model.md#named-recipes); earlier `run_default` results are not relabelled.

Legacy `climate.selected` and `climate.forcing.temp_pet_source` remain accepted with deprecation notices.
Do not supply both spellings of a field, even with equal values. A legacy null selection remains unselected;
null under a new key is invalid. Archives and scientific recipe serialization retain their original bytes
and meaning. Equivalent old/new or inferred/explicit choices have the same scientific identity within one
software candidate; producer-code integrity can change identities across revisions.

| Key | Default and allowed meaning |
|---|---|
| `precip_source` | Candidate precipitation source; required explicitly when several candidates exist |
| `temp_source` | `era5`; temperature, extrema, pressure and radiation required by the selected PET recipe |
| `orography_source` | `era5_orography`; source elevation entry; required explicitly for another companion |
| `temp_correction` | `true`; boolean controlling lapse correction at extraction and model preparation |
| `press_correction` | `true`; boolean controlling pressure correction during computed-PET preparation |
| `pet.mode` | `computed` or `direct` |
| `pet.method` | `debruin` (default) or `makkink`; computed mode only |
| `pet.source` | Required catalog entry for supplied reference PET in direct mode; forbidden in computed mode |

Direct mode forbids both `pet.method` and an explicitly supplied `press_correction`, even if false.
It prepares temperature and transports the supplied PET through HydroMT-Wflow's `setup_pet_forcing`.
Supply reference PET, not actual evapotranspiration. Direct PET is available for historical analysis and
model forcing; scenario generation and scenario simulation reject it before preparation. No PET
perturbation model is implied.

Computed recipes require companion `temp`, `temp_min`, `temp_max`, `press_msl` and `kin`, plus `kout`
for de Bruin. Direct recipes require companion `temp` and supplied `pet`. Precipitation defines the
extraction grid. Sources are clipped to their actual daily overlap; gaps, duplicates, unsupported
calendars, absent variables and invalid required-domain values fail instead of being padded. Supplied
PET must be finite and nonnegative inside the required basin; masked cells outside it are allowed.

When a different companion is regridded, its declared source elevation references the lapse correction
onto the hydrography sidecar. Subsequent model preparation uses that sidecar. Disabling temperature
correction disables both steps. Same-source ERA5 extraction reads native temperature without applying
lapse correction; its configured elevation is used during preparation. Pressure is corrected only during
PET preparation.

The reviewed ERA5/CHIRPS catalog arithmetic remains supported. An alternative canonical-unit input must
declare `metadata.time_label: interval_end` in its catalog and actually deliver daily midnight fields in
mm/day, degrees Celsius, hPa and W/m² as applicable. This declaration describes the daily interval ending
at each timestamp; regular midnight timestamps alone do not establish it. Required-variable unit scaling
must be identity arithmetic, and variable attributes are checked. Supplied PET may also use HydroMT's
native `mm` label for a daily total; it is equivalent to mm/day only under this enforced one-day interval
contract. No additional conversion profile or new data product is supplied. Controlled fixtures establish
the mechanism; real alternative companions
and independent PET products require separate source qualification.

Extraction identity contains sources, required variables and corrections performed during extraction.
Non-default extraction settings select a separate store. Preparation-only pressure settings, and
same-source elevation/lapse settings, do not change that store or its generator input identity.
Scenario simulation compares extraction recipes with the selected collection's archived creator recipe
and applies the current preparation settings. Non-default preparation records use version 3; retained
version-2 records retain their original correction settings. Producer-code integrity remains part of
scenario identity. Numerical backward comparisons therefore use equivalent fixed inputs and seeds.

CHIRPS diagnostics remain precipitation-only contributors even when their forcing stores borrow
temperature/PET fields. Captions identify the companion or supplied-PET source when those fields are shown.

## Workflow settings

Use the complete annotated starter files for supported optional fields and example values.
The following links explain the settings for each workflow.

| Workflow | Settings and detailed reference |
|---|---|
| historical climate analysis | `diagnostics`, workflow-local `plotting`, optional `rain_gauges`: [historical analysis](workflow-analyze-climate.md#configuration) |
| model building | `simulation_window`, engine recipes, `observations`, workflow-local `plotting`, `evaluation`: [model](workflow-build-model.md#configuration) |
| projection analysis | `catalog`, `ensemble`, `models`, `scenarios`, `members`, `variables`, windows, `stats`: [projections](workflow-analyze-projections.md#configuration) |
| scenario generation | `seed`, `n_realizations`, `simulation_window`, monthly perturbations, generator and capacity: [generation](workflow-generate-scenarios.md#configuration) |
| hydrological simulation | Collection, experiment, operations, indicator requests and `compute`: [simulation](workflow-simulate-system.md#configuration) |

`experiment_name` is optional. An omitted name selects family `stress_test`, initially `stress_test_01`;
a custom base selects its own numbered family. Plain execution selects the highest allocation, irrespective
of completion. `--new-experiment` advances it; an exact numbered name selects that experiment. Previewing with
`scripts/suggest_experiment_name.py` reserves nothing. Changed scientific simulation inputs require a new
experiment. See [retained identities](retained-handoffs.md#identities-and-reuse).

## Scalar response precision {#csv-precision}

WF1 and WF4 each accept this setting in their own workflow YAML:

```yaml
output:
  csv:
    decimal_places: 5
```

`decimal_places` defaults to 5 and accepts integers 0–15, excluding booleans. These are decimal places,
not significant digits. WF1 evaluation and historical thresholds, and WF4 indicators, use the rounded
retained time series. Values smaller than half the last decimal unit can become zero; use greater
precision when very small flows matter. Missing values remain missing. Indicator-summary formatting
and native NetCDF precision are independent. The CSV and NetCDF sections can be used together.

Native Wflow CSVs are temporary conversion inputs. Verified per-variable CSVs and their metadata are
the retained format; there is no native-retention or compatibility option. Increasing precision after
conversion requires another simulation. New WF4 experiments freeze precision in their response request.
Old native-only publications are not upgraded or rewritten: create new historical runs/experiments.

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

Keys under `variables` are installed Wflow-SBM standard names; values are unique lowercase
snake_case NetCDF names. Coordinate names are reserved. Variables must exist in enabled model
components and have a native writer-supported shape; admission occurs before timestepping.
Native soil-layer dimensions are preserved. Empty mappings, unknown keys and user-specified
paths reject. `compressionlevel` defaults to 1 and accepts integers 0–9, excluding booleans.

Files follow the native simulation timestep, period and calendar, with interval-end labels.
Snow depths are end-of-step storages in mm: dry-snow water equivalent and liquid water,
respectively; neither is fractional snow cover. Discharge is the native timestep-average flow
in m3/s. Other variables retain Wflow's own units and temporal meaning. Native values and masks
are written by Wflow; metadata adds units from its standard-name API and CRS from staticmaps.

The [output guide](../guide/outputs.md#native-grids) describes placement, storage and reruns.

## Plotting resources

Python plotting adapters load `config/plotting/colors.yml` and `style.yml` from the toolbox.
These affect every project using that checkout. The
[local plotting reference](../../config/plotting/README.md) explains how to change these settings.
Appearance changes schedule affected plots without changing scenario/simulation identity; model building's
combined
evaluation job also refreshes its indicators. R generator figures retain their own settings.
Changing colors does not change normalization, classification or scientific missingness rules.

Tests: `tests/test_config_composition.py`, `tests/test_config_paths.py` and
`tests/test_v2_config_shape.py` cover composition/path and shared-setting boundaries.
