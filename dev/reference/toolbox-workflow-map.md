# Workflow implementation map

Last updated: 2026-11-05

Use this map to find a rule, its implementation and the products it reads or writes. The Snakefiles
own exact rule names, numbers, inputs, outputs and dependencies. The
[architecture reference](toolbox-architecture.md) explains workflow responsibilities, configuration
and retained artifacts.

## Start at the workflow entry point

| Workflow | Rule definitions |
|---|---|
| WF0 | [analyze_climate.smk](../../analyze_climate.smk) |
| WF1 | [build_model.smk](../../build_model.smk) |
| WF2 | [analyze_projections.smk](../../analyze_projections.smk) |
| WF3 | [generate_scenarios.smk](../../generate_scenarios.smk) |
| WF4 | [simulate_system.smk](../../simulate_system.smk) |

Use the [single-workflow runner](../../scripts/run_workflow.py) for one workflow or the
[all-workflow runner](../../scripts/run_workflows.py) for an enabled sequence. WF4 also has a
[direct simulation runner](../../scripts/simulate_system.py).

WF3's owned source and generation phases are coordinated in
[`scripts/generate_scenarios.py`](../../scripts/generate_scenarios.py), called by the all-workflow
runner. WF4 includes either
[`simulate_and_indicators.smk`](../../blueearth_cst/experiment/rules/simulate_and_indicators.smk)
or [`indicators_only.smk`](../../blueearth_cst/experiment/rules/indicators_only.smk) according to
the validated operation, then uses shared indicator-publication rules in `simulate_system.smk`.
Read the launchers as well as the DAG when changing preparation, target selection or retained-input
admission. Bare WF3/WF4 Snakemake launches are not the supported execution path.

## Trace a rule and its consumers

1. Find the rule in its Snakefile. Follow its `script:`, `shell:` or helper call to the implementation.
   Check the selected project and workflow configuration to see which jobs are enabled.
2. Read `input:`, `output:`, helper-generated declarations and terminal targets together. A file
   written inside a script but absent from `output:` is not automatically a schedulable dependency.
3. Search outside the DAG for retained-artifact readers, verification tools and other consumers.
   For optional or dynamically named products, inspect the producer's inventory and completion
   contract instead of assuming a fixed file list.

Rule labels and log order come from each workflow's `RuleRegistry` registrations and `LOG_RULES`.
Each logged rule must also declare `message: <registered identity>.banner()` so its start is shown
as a compact `RUN` row. Without it, Snakemake prints its full job-details block. The parsed contract
in `tests/test_log_rules_contract.py` guards both registration and message presence.
[Rule numbering](dev-naming-conventions.md#9-rule-numbering-wnn-reference-scheme) defines the label
convention. Numbers are reading aids, not execution order; inspect dependency edges.
Gather rules must wait for all terminal work they own, including independent branches, before
combining logs or benchmarks.

## Shared producers and cross-workflow inputs

WF0 and WF1 include the same
[`source_map_preparation.smk`](../../blueearth_cst/climate_analysis/rules/source_map_preparation.smk)
definition for native-grid map fields. Its `tables/annual_climatology.nc` is a numerical product
separate from diagnostic tables and display-reference preparation. A display-only change should not
rerun that numerical producer. WF1 retains `historical/default/run/output.csv` so export and evaluation can
consume the historical simulation without rerunning it.

[`cross_workflow_leaves.py`](../../blueearth_cst/shared/cross_workflow_leaves.py) names the static
WF1 model leaves that WF4 simulation consumes. WF2 and WF3 build their own DAGs without WF1 leaves.
WF4 also reads a validated scenario collection; its typed reader checks content beyond the static
leaf list. Indicators-only uses retained simulation and response records. The
[cross-workflow tests](../../tests/test_cross_workflow_inputs.py) exercise these boundaries.
For mutable model directories, follow the
[model-root ordering rule](wf1-model-construction.md#mutable-model-root-ordering).

## Check an affected graph

Dry-run the affected configured graph through its supported launcher after a rule change. The
[DAG plotting tool](../../scripts/plot_workflow_dag.py) provides a graph view. Select affected
consumer and contract checks through the [validation ladder](dev-validation.md); useful entry
points include [`test_cross_workflow_inputs.py`](../../tests/test_cross_workflow_inputs.py),
[`test_log_rules_contract.py`](../../tests/test_log_rules_contract.py) and the runner tests.
The [reference index](README.md) points to the owning scientific and implementation contracts.
