---
title: "Software and execution"
format: html
---

Execution and automation tools connect the modeling tools, datasets and scientific workflows. They read
the project settings, check inputs, schedule calculations and keep records of each run. This allows an
assessment to be configured and executed through command-line instructions.

## Software environment

Pixi manages the Python/R environment through `pixi.toml` and `pixi.lock`. Julia is managed by juliaup and
must be available on `PATH`; `Project.toml` and `Manifest.toml` describe the Julia project. Do not install
Julia into Pixi or edit generated locks by hand. Use the [installation guide](../guide/install.md) for setup.

### Roles of Python, R and Julia {#language-boundaries}

| Runtime | Operations | Authoritative files |
|---|---|---|
| Python | Runners, configuration, HydroMT preparation, reductions and plots | `pixi.toml`, `pixi.lock` |
| R | `weathergenr` and its generation scripts | `dev/scripts/install_weathergenr.R`, Pixi tasks |
| Julia | Wflow execution | `Project.toml`, `Manifest.toml`, `config/advanced_settings.yml` |
| Snakemake | Workflow scheduling | Five root `.smk` entry points and included rule files |

`config/advanced_settings.yml` declares required software versions and default resources. Its closed schema
is implemented in `blueearth_cst/shared/advanced_settings.py`. `tests/test_julia_runtime.py` checks Julia pin
consistency. The installation script specifies the installed weather-generator revision.

Run supported commands with `pixi run`. Snakemake Python steps read `snakemake.input`, `output` and `params`.
R workflow scripts use `Rscript --vanilla` and positional arguments. netCDF carries gridded climate and
forcing between languages. Hydrological simulation retains rounded per-variable CSV responses with explicit
interpretation metadata.

## Run commands

A **runner** ([run command](../guide/glossary.md#run-command)) is a Python script launched from the command
line to start and coordinate an analysis. `scripts/run_workflows.py` runs the workflows enabled in a project
configuration, while `scripts/simulate_system.py` runs hydrological simulation directly. Each runner checks
the required inputs, starts the requested processing and records its outcome. The
[running guide](../guide/running.md) gives commands and examples.

The runner's final verdict includes result publication. Progress or a completed simulation batch alone is
not confirmation that every requested indicator set is current.

## Snakemake

Snakemake schedules individual processing steps according to their dependencies and available resources.
For example, a simulation can start only after its model and forcing inputs are ready. Independent steps
can run concurrently within the configured resource limits. Use the supported runners to execute scenario
generation and hydrological simulation, including their preparation and result-saving steps.

## Data access and environment records {#availability-and-provenance}

Installing the runtime does not install the external forcing or model datasets. Catalog roots, network
access, storage and project write access are separate prerequisites. Successful runtime checks do not
establish data coverage, scientific suitability or a qualified model.

Configuration snapshots and collection/simulation/indicator records capture environment and source
identities for each processing stage. These identities detect incompatible reuse; they do not recreate
an environment on another machine. Projection analysis bundles likewise record lock hashes and toolbox
revision without installing dependencies. See [execution records](run-records.md) and
[results transfer](results-transfer.md).

## Configuration and run management

The [project configuration](configuration.md) specifies the basin, data sources, analysis periods,
climate perturbations and workflows to run.

Use [run records](run-records.md) to establish what ran and which inputs were used.
[Computation time and resource requirements](runtime-and-resources.md) explains the timing and resource
reports. [Saved results](retained-handoffs.md) describes output structure and reuse, while
[transferring climate-projection results](results-transfer.md) explains how to move projection analysis
products between projects.
