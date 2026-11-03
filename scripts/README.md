# scripts/

What a user runs to execute the pipeline. Snakemake-executed modules live in `blueearth_cst/`;
repository inspection and maintenance tools live in `dev/scripts/` and are never part of a run.

## Runners

| Script | What it does |
|---|---|
| [`run_workflows.py`](run_workflows.py) | Run every enabled workflow in fixed order; the contract is its module docstring. |
| [`run_workflow.py`](run_workflow.py) | Run one of the five workflows through its owned runner. |
| [`generate_scenarios.py`](generate_scenarios.py) | Owned two-phase WF3 launcher: prepare sources, freeze a plan, then generate. |
| [`simulate_system.py`](simulate_system.py) | WF4 runner: validates the operation and targets, then starts one Snakemake call. |

## Case setup

| Script | What it does |
|---|---|
| [`create_case.py`](create_case.py) | Create an external case from this checkout's `config/templates/example-basin/`. |
| [`suggest_experiment_name.py`](suggest_experiment_name.py) | Preview numbered selection without changing configs or reserving a name. |
| [`calibrate_batching.py`](calibrate_batching.py) | Time WF4 batch settings on your basin and machine. |
| [`plot_workflow_dag.py`](plot_workflow_dag.py) | Render a workflow's DAG under the project's own `logs/dag/`. |
| [`render_figures.py`](render_figures.py) | Re-render a finished run's figures from its retained products into a separate folder; `--list` shows families. |
| [`maintain_config_archives.py`](maintain_config_archives.py) | Inspect generated configuration history; `--apply` retains snapshots and removes completed launch bookkeeping. |
| [`transfer_results.py`](transfer_results.py) | Export a WF2 product with its configuration and provenance as one bundle, or import a bundle into another project. |


## Procedures and result inspection

Start with [running](../docs/guide/running.md), [new cases](../docs/guide/new-case.md) or
[indicator selection](../docs/guide/indicators.md); this file is an entry-point index, not a flag reference.
Use `inspect_indicators.py` for saved-set integrity inspection and optional numerical rechecking;
the [technical request reference](../docs/technical-manual/indicator-requests.md) explains their different
requirements and limits. Inspection alone does not make stale outputs reusable.
