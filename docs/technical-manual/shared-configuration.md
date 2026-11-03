---
title: "Shared configuration"
format: html
---

The project file, `project_config.yml`, defines the basin, data sources and model outputs shared across workflows.
Workflow YAML files hold the settings specific to each analysis. This page introduces the shared
choices; [execution and run management](configuration-and-execution.md) covers running, resuming
and monitoring the assessment.

## Project configuration {#project-structure-and-configuration-files}

The [Workflows overview](scientific-workflows.md#project-structure-and-configuration) introduces the
project file, workflow files and output directory. Copy the complete
`config/templates/example-basin/` starter or follow the [new-case guide](../guide/new-case.md), keeping
`project_config.yml` and its five workflow files together. The starter's default catalog requires access
to the Deltares P-drive.

The following excerpt belongs in the case folder's `project_config.yml`. It shows the configuration
version, output directory, data catalog and one enabled workflow; the full starter also contains the
shared basin, climate and model settings.

```{.yaml filename="project_config.yml (excerpt)"}
schema_version: 3
project:
  project_dir: "../../runs/my-basin"
  catalog: toolbox:config/catalogs/deltares_data_pdrive.yml
workflows:
  analyze_climate:
    enabled: true
    config_path: project_config_analyze_climate.yml
```

Here, `config_path` points to `project_config_analyze_climate.yml` beside `project_config.yml`.
That workflow file contains the climate-analysis settings; `enabled` controls whether it runs.
The output directory is separate from these configuration files, and disabling a workflow does not
delete its existing products.

### How paths are read

With `schema_version: 3`, supported file paths resolve beside the YAML that declares them.
For example, `observations: {river discharge: data/q.csv}` in `settings/build-model.yml` reads
`settings/data/q.csv`. A workflow's `config_path` resolves beside the project file.

Use `toolbox:config/...` for bundled resources or absolute paths for other locations.
HydroMT catalogs follow their own internal path rules. Environment variables are not expanded.
Schema 2 uses legacy path rules; changing the version alone does not convert paths.
See the [configuration reference](configuration.md#how-paths-are-read) for the supported file fields.

## Key shared settings

Shared assessment choices belong in the project YAML. Each workflow page describes its own YAML
settings and the shared parameters it uses:

| Assessment choice | Where to set it |
|---|---|
| Output storage and data access | Project `project.project_dir`, `project.catalog`. |
| Basin boundary and grid spacing | Project `basin.region`, `basin.resolution` (degrees). |
| Reporting locations and saved variables | Project `basin.output_locations`, `model.outvars`. |
| Historical sources and selected forcing | Project `climate.sources`, `climate.selected`. |
| Historical record and reporting year | Project `climate.window`, `climate.water_year_start`. |

The historical extraction window, historical model run, GCM comparison windows and generated scenario
period are separate choices. The [workflow pages](scientific-workflows.md) explain their relationships.
Select saved model variables before simulation: a later indicator request cannot recover an unsaved
variable.

### Plot appearance {#plot-annotations}

Set project `plotting.annotation` to `none`, `caption` or `title` for Python plots.
Captions and title inventories remain available beside the figures regardless of this choice.
Shared colours and styles are described in the [plotting guide](../../config/plotting/README.md).
These settings do not apply to R/weathergenr plots.

### Shared toolbox resources

Most assessment choices belong in your case files. Resources inside the toolbox can affect all cases
using that installation: `config/catalogs/` describes data access; `config/defaults/` contains live
model and generator recipes; `config/plotting/` controls Python appearance.
`config/templates/` contains copyable starters. Normally leave `config/advanced_settings.yml` as supplied.
Pass the project file to the command, never the advanced-settings file.
