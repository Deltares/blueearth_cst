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
| `climate.selected` | Selected forcing; must belong to candidates; null is allowed for historical climate analysis-only analysis |
| `climate.window` | Inclusive calendar-year `start`/`end`; historical record, not hydrological run period |
| `climate.water_year_start` | Reporting-year month, default January |
| `climate.map_reference` | Historical-map completeness; default 3 missing days/month, 15/year |
| `model.outvars` | Retained model variables; declarations must use supported Wflow display names |
| `plotting.annotation` | `none` (default), `caption` or `title` on ordinary PNG paths |

The historical-window parser is `historical_window_bounds` in `shared/wf3_science.py`.
`snake_utils.validate_historical_window` enforces the configured 16-year minimum. Extraction uses the
available overlap, warns on narrowing, and refuses no overlap or insufficient selected-source coverage.
CHIRPS selection requires overlap with its ERA5 companion variables. Model building independently checks
forcing
coverage for its declared simulation window, including Wflow's initial-state time convention.

`climate.map_reference` accepts only `max_missing_days_month` (integer 0–27) and
`max_missing_days_year` (nonnegative integer). Booleans and fractions fail. Historical climate analysis
diagnostic tolerances must
equal these shared values. Model building forcing summaries retain their independent default 3/15 policy.
Completeness and temporal aggregation are distinct from the inclusive extraction dates.

## Workflow settings

Use the complete annotated starter files for supported optional fields and example values.
The following links explain the settings for each workflow.

| Workflow | Settings and detailed reference |
|---|---|
| historical climate analysis | `diagnostics`, `subbasin_figures`: [historical analysis](workflow-analyze-climate.md#configuration) |
| model building | Required `simulation_window`; optional engine recipes and `observations`: [model](workflow-build-model.md#configuration) |
| projection analysis | `catalog`, `ensemble`, `models`, `scenarios`, `members`, `variables`, windows, `stats`: [projections](workflow-analyze-projections.md#configuration) |
| scenario generation | `seed`, `n_realizations`, `simulation_window`, monthly perturbations, generator and capacity: [generation](workflow-generate-scenarios.md#configuration) |
| hydrological simulation | Collection, experiment, operations, indicator requests and `compute`: [simulation](workflow-simulate-system.md#configuration) |

`experiment_name` is optional. An omitted name selects family `stress_test`, initially `stress_test_01`;
a custom base selects its own numbered family. Plain execution selects the highest allocation, irrespective
of completion. `--new-experiment` advances it; an exact numbered name selects that experiment. Previewing with
`scripts/suggest_experiment_name.py` reserves nothing. Changed scientific simulation inputs require a new
experiment. See [retained identities](retained-handoffs.md#identities-and-reuse).

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
