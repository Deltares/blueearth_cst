---
title: "Execution and run management"
format: html
---

Execution connects the configured analyses to their inputs, computing resources and saved results.
This chapter explains scheduling, reuse and run records. For commands, follow
[Running workflows](../guide/running.md); for errors and interrupted runs, use
[Troubleshooting](../guide/troubleshooting.md).
[Shared configuration](shared-configuration.md) and the individual
[workflow pages](scientific-workflows.md) describe the settings consumed by execution.

## Supported entry points {#running-the-assessment}

The runners prepare inputs and check results as well as schedule Snakemake rules.

| Entry point | Scope |
|---|---|
| `scripts/run_workflows.py` | Enabled workflows in the project file, including scenario generation. |
| `scripts/run_workflow.py` | One explicitly named workflow; accepts all five workflow names. |
| `scripts/simulate_system.py` | Hydrological simulations and/or indicators for a selected experiment. |

The all-workflow runner follows a fixed order: historical climate analysis, model building, projection
analysis, scenario generation and hydrological simulation. It skips disabled workflows and stops at
the first workflow failure. Disabled workflows do not supply missing prerequisites, although their
compatible existing products can be reused. Model building and scenario generation supply the model
and weather required for simulation; projections do not determine the generated scenarios.

For ordinary use, the User Guide recommends the all-workflow runner, with only the required workflows
enabled. It also documents the dedicated hydrological simulation command. The single-workflow script
accepts all five names; for `generate_scenarios` and `simulate_system`, it delegates to their dedicated
runners rather than launching Snakemake directly.

Bare Snakemake execution omits required preparation and is not supported for scenario generation or
hydrological simulation. Simulation targets are inferred from `operations` and indicator selection;
routine requests need no target override.

A dry run does not generate missing inputs. Some downstream checks remain pending until earlier
products exist, so a successful preview does not establish that every later calculation can succeed.
The [command reference](execution-and-provenance.md#launcher-arguments-and-safety-boundaries)
describes supported options and forwarding restrictions.

## Computation time and resource requirements

Model-cell count, simulation length, perturbation levels and weather realizations determine the main
workload. Storage includes source extracts, the model, weather series, saved responses and readable
exports; scenario exports occupy space alongside their saved source copies.

`--cores` is a positive integer with default 3. Hydrological simulation groups runs according to basin
size and available resources. Its optional `compute` settings include `julia_threads`,
`max_parallel_batches`, `batch_size_max`, `batch_size` and `disk_headroom_gb`. Increasing concurrency
can increase memory and disk demands. Julia's first simulation in a process includes startup and
compilation costs.

Benchmark job durations can overlap, so their sum is not total elapsed time; peak memory for one job
is not total concurrent memory. [Runtime and resource guidance](runtime-and-resources.md) explains
the measurements and workload adjustments. Case measurements are more informative than a universal
runtime estimate.

## Completion and failure status {#monitoring-progress-and-diagnosing-failures}

Each launch produces a linked `logs/<UTC timestamp>__<workflow scope>/summary.md`, with `console.log`,
workflow logs and benchmark reports where available. The summary distinguishes requested work,
outcomes and warnings. A successful preview or an up-to-date run may perform no new computations.

Hydrological simulation saves each completed indicator set independently. A failed multi-set run can
therefore leave current results for some sets beside older results for others. Command completion and
the presence of CSV files alone do not establish that every requested set is current. The
[results chapter](results-and-interpretation.md#checking-which-results-are-current) explains how to
identify the calculation behind a product; [Troubleshooting](../guide/troubleshooting.md) gives
diagnostic and recovery steps.

## Reuse and experiment compatibility {#resuming-runs-and-reusing-results}

Completed work is reusable when its saved inputs and settings remain compatible with the request.
An experiment identifies the model, climate scenarios and scientific simulation settings used for
its responses. Changed scientific inputs require a new experiment; forcing jobs to rerun or
unlocking the project does not make those inputs compatible.

Indicator changes can use saved responses through `operations: [indicators]` when the completed
experiment contains the required variables, locations and periods. Missing response variables require
new simulations. Resource changes alone do not define a new scientific experiment.
The [saved-results reference](retained-handoffs.md#identities-and-reuse) specifies reuse requirements;
[indicator settings](indicator-requests.md) covers selection and handling incomplete inputs.

Use the User Guide to
[select or create an experiment](../guide/running.md#select-resume-or-create-an-experiment) or
[recalculate indicators](../guide/running.md#calculate-different-indicators-from-existing-simulations).
An interrupted allocation has separate reservation-recovery checks; recovery does not repair partial
weather generation or damaged responses. A rebuild permanently removes generated products and all
experiments, including disabled workflows' outputs. Its preview and deletion boundaries are described
beside the [rebuild procedure](../guide/running.md#rebuild-outputs-for-testing).

## Run records and reproducibility

The toolbox saves source configurations and execution records, but does not copy every external
dataset. Reproducing an assessment therefore requires the source case files, complete output directory,
software environment and accessible source data.

Historical climate analysis, model building and projection analysis configuration copies are under
`config/<workflow>/`; scenario and experiment settings accompany their respective outputs. These
copies document previous runs. Changes belong in the source YAML rather than the saved copies.

Run summaries link attempts and timings; product records identify which attempt produced a result.
The newest log alone need not describe every file after a partial rerun.
[Run records and reproducibility](execution-and-provenance.md) documents configuration history,
product attribution and simulation resource measurements.

## Portability and results transfer {#saving-and-transferring-assessment-results}

Moving a case requires compatible software, accessible catalogs and observations, and valid output
paths. Relative paths resolve beside their declaring YAML; absolute paths may need updating.
Indicator CSVs support table reading but do not contain everything required to rerun an assessment.

The supported ZIP transfer covers completed climate-projection products. Import checks file integrity,
basin geometry and applicable configuration, and does not merge into a different existing projection
product. Imported products support reading; they do not recreate the source environment or guarantee
reuse by a later projection-analysis run. See the
[transfer procedure](../guide/running.md#transfer-climate-projection-results) and
[transfer reference](results-transfer.md) for commands and destination requirements.
