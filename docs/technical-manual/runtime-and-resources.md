---
title: "Computation time and resource requirements"
subtitle: "Use run summaries to plan larger assessments and investigate slow runs"
---

Use the toolbox's timing reports to see where a run spends time and to judge whether your machine
can support a larger basin or scenario ensemble. Start with measurements from a completed run
that resembles the assessment you are planning.

## Open the execution summary

Follow the closing console link to `logs/<UTC timestamp>__<workflow scope>/summary.md`.
For a command running several workflows, the summary links to their individual reports.

First check whether the attempt executed new work, previewed tasks, failed or reused existing
outputs. A quick rerun that reuses completed results is not a measure of how long a fresh run takes.
Then open the linked `wfN-benchmarks.md` report for a readable overview, or its `.tsv` file
for individual measurements. Here, `N` identifies the workflow.

Missing measurements mean that timing information is unavailable, not that a task took zero time.
An unfinished run provides only partial measurements.

## Compare measurements correctly

The reports describe several different quantities:

| Measurement | How to use it |
|---|---|
| Workflow elapsed time | Time taken by that recorded workflow attempt, including overlapping jobs |
| Individual job duration | Helps identify slow tasks; parallel job times cannot simply be added |
| Peak individual-job RAM | Memory used by one job, not the combined peak of all concurrent jobs |
| Julia loading and first-member time | Includes startup and compilation that later members may avoid |
| Allocated bytes | Total allocation activity, not memory retained or peak RAM |

When comparing runs, note the basin size, grid resolution, period, scenario count, retained model
variables and CPU settings. Also check which outputs were reused. A larger ensemble may spread
startup costs over more simulations, so multiplying a small run's duration is only a rough estimate.

Compilation and garbage collection are already included in wall time; do not add them again.
Older records without measured elapsed seconds remain unavailable.

## Inspect simulation resource use

Workflow (WF) 4 records resource diagnostics during execution. Dry runs do not collect them.
Find the hydrological simulation child invocation's `resources` links and read the traces alongside its logs
to
distinguish forcing preparation, hydrological simulation and indicator evaluation.

Use these measurements to identify likely pressure on CPU, memory or storage. They are sampled
approximately every two seconds, so short peaks can be missed. Summed process memory may count
shared pages more than once, and system-wide measurements include other applications.
Logical I/O may include cached operations; reported CPU frequency alone does not diagnose throttling.

## Adjust the workload

If memory is limiting the run, reduce `--cores` before increasing concurrency. For hydrological simulation,
review
the optional `compute` settings in `project_config_simulate_system.yml`: batch size, concurrent
batches, Julia threads and disk headroom. Start with automatic allocation and change settings
in response to measured constraints.

Larger basins, longer periods, more realizations and additional retained variables all affect
resource requirements. Test a manageable configuration before expanding the assessment, then
inspect the new summary to see whether the adjustment helped.

These reports measure execution, not scientific adequacy. Use the model and indicator checks in
[Reading the outputs](../guide/outputs.md) before interpreting the results.

For exact file names, fields and collection limits, see the
[execution reference](execution-and-provenance.md).
