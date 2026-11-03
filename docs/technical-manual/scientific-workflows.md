---
title: "Workflows"
format: html
---

The toolbox organizes climate risk assessment into five workflows, from historical data analysis to
hydrological simulation and indicator calculation. Each addresses a distinct part of the assessment
and can run separately when its required inputs are available. The linked pages describe their inputs,
processing steps, configuration and outputs.

- **[Historical climate analysis](workflow-analyze-climate.md):** builds understanding of baseline
  climate variability and the climatic context of hydrological conditions, and supports selection
  of the gridded dataset used as forcing.
- **[Model building](workflow-build-model.md):** prepares the basin model from spatial and climate data,
  runs the historical simulation and produces diagnostics for evaluating model behavior.
- **[Climate-projection analysis](workflow-analyze-projections.md):** estimates temperature and
  precipitation changes to provide context for interpreting the stress-test conditions.
- **[Scenario generation](workflow-generate-scenarios.md):** generates weather realizations across
  analyst-defined temperature and precipitation changes, including an unperturbed reference.
- **[Hydrological simulation and indicators](workflow-simulate-system.md):** runs the prepared model
  with generated weather and summarizes responses at the assessment locations.

## How the workflows connect

Historical climate analysis establishes baseline context and supports forcing selection. Model building
and scenario generation supply the model and weather for hydrological simulation; scenario generation
and simulation form the stress-testing part of the climate risk assessment. Projections provide context
without determining the scenarios. See the [conceptual overview](overview.md#components).

The decision context sets the geographical scope, planning horizon and objectives—for example,
catchment flash floods, basin-wide drought or reduced irrigation supply. These concerns guide the
analysis resolution, climate ranges, indicators and performance thresholds, expressed through project
and workflow settings.

## Project structure and configuration

Each assessment has a project YAML file and separate YAML files for the five workflows. The starter
keeps them together in a case folder, alongside any input files supplied by the user:

```text
my-basin/
├── project_config.yml                       # Shared project settings
├── project_config_analyze_climate.yml       # Historical climate analysis
├── project_config_build_model.yml           # Model building and historical simulation
├── project_config_analyze_projections.yml    # Climate-projection analysis
├── project_config_generate_scenarios.yml     # Weather scenario generation
├── project_config_simulate_system.yml        # Hydrological simulation and indicators
└── data/
    └── output_locations.csv                 # Reporting locations supplied by the user
```

- **Project configuration (`project_config.yml`).** Defines shared basin, climate and model settings,
  data access and output storage. It also selects which workflows run and points to their YAML files.
- **Workflow configurations (`project_config_<workflow>.yml`).** Define settings specific to each
  workflow, such as analysis periods, climate perturbations or indicator selections. Each workflow
  page explains its own settings and the shared project parameters it uses.
- **Project output directory (`project.project_dir`).** Stores prepared data, models, generated
  scenarios and experiment results. This directory can be separate from the case folder above;
  its `data/` folder contains prepared data rather than the user-supplied files shown here.

The [new-case guide](../guide/new-case.md) explains how to create this structure from the starter.
[Shared configuration](shared-configuration.md) introduces the key project settings and file paths
before the individual workflows. [Execution and run management](configuration-and-execution.md)
covers running, resuming, monitoring and documenting the assessment.
