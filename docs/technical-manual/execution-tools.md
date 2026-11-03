---
title: "Workflow execution and automation"
format: html
---

Execution and automation tools connect the modeling tools, datasets and scientific workflows. They read
the project settings, check inputs, schedule calculations and keep records of each run. This allows an
assessment to be configured and executed through command-line instructions.

## Python runners

A **runner** is a Python script launched from the command line to start and coordinate an analysis.
`scripts/run_workflows.py` runs the workflows enabled in a project configuration, while
`scripts/simulate_system.py` runs hydrological simulation directly. Each runner checks the required inputs,
starts the
requested processing and records its outcome. The [running guide](../guide/running.md) gives commands
and examples.

## Snakemake

Snakemake schedules individual processing steps according to their dependencies and available resources.
For example, a simulation can start only after its model and forcing inputs are ready. Independent steps
can run concurrently within the configured resource limits. Use the supported runners to execute scenario
generation
and hydrological simulation, including their preparation and result-saving steps.

## Configuration and run management

The [project configuration](configuration.md) specifies the basin, data sources, analysis periods,
climate perturbations and workflows to run. The [software environment](environment.md) supplies the
Python, R and Julia dependencies used by those workflows.

Use [run records](execution-and-provenance.md) to establish what ran and which inputs were used.
[Computation time and resource requirements](runtime-and-resources.md) explains the timing and resource
reports. [Saved results](retained-handoffs.md) describes output structure and reuse, while
[transferring climate-projection results](results-transfer.md) explains how to move projection analysis products between
projects.
