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

The all-workflow runner checks general workflow configuration at startup. It validates forcing and
shows **Historical forcing settings** only when the first enabled forcing consumer is reached:
model building, scenario generation, or simulation. Historical comparisons and projection analysis
can finish without a forcing choice. If forcing is ambiguous or invalid, the runner stops before
the consumer starts; completed comparison outputs remain available for review. It does not pause
interactively or choose a dataset from the comparison results.

Experiment admission also waits until that consumer boundary, before any model or scenario producer
writes. An indicators-only simulation needs experiment admission but no forcing selection or settings
block. Destructive rebuilds retain their full validation before erasing outputs.

The forcing settings have no timestamp or subsystem prefix and appear once per invocation.
The block uses a bold gray title followed immediately by flush-left `>` items in gray. Redirected
output and terminals with color disabled retain the same layout without styling codes.
This styling is limited to Historical forcing settings, whose lines fit 80 columns or the narrower
terminal width. Settings, workflow sequence, handoffs, commands, summaries and rebuild previews keep
their existing formatting. Command options and paths retain indented continuations without `>` markers.
The preparation progress line is left aligned in the terminal's normal color.
Timestamped subsystem rows report processing within a workflow; repeated configuration reads do
not print another summary. The summary names rainfall and temperature datasets, the potential
PET method and planned elevation adjustments, distinguishing project settings
from toolbox defaults. It describes the configuration; data coverage is checked during processing.

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
