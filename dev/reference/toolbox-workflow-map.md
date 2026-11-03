# Workflow implementation map

Last updated: 2026-10-20

Use this map before changing a rule: find its executable definition, shared producers and contract owner.
The Snakefiles own exact rule names, numbers, inputs, outputs and dependencies; this document does not
maintain a second inventory. [Toolbox architecture](toolbox-architecture.md) owns workflow boundaries
and configuration ownership.

## Find the implementation

Paths below are relative to the repository root. Follow a rule's `script:`, `shell:` or helper call to
its implementation; inspect the selected configuration when determining which jobs exist.

| Workflow | Rule definitions |
|---|---|
| WF0: historical climate analysis | [analyze_climate.smk](../../analyze_climate.smk) |
| WF1: model construction and historical run | [build_model.smk](../../build_model.smk) |
| WF2: projection analysis | [analyze_projections.smk](../../analyze_projections.smk) |
| WF3: scenario generation | [generate_scenarios.smk](../../generate_scenarios.smk) |
| WF4: simulation and indicators | [simulate_system.smk](../../simulate_system.smk) |

WF3 also has preparation steps in [the all-workflow runner](../../scripts/run_workflows.py).
WF4 selects [simulation rules](../../blueearth_cst/experiment/rules/simulate_and_indicators.smk) or
[indicators-only rules](../../blueearth_cst/experiment/rules/indicators_only.smk), then shares indicator
publication rules in its top-level Snakefile.
[The simulation runner](../../scripts/simulate_system.py) owns operation and target validation,
including multiple indicator requests. Read these launchers as well as the DAG when changing preparation
or retained-artifact admission.

For rule labels and log ordering, inspect each workflow's `RuleRegistry` registrations and `LOG_RULES`.
[Naming][rule-numbering] owns numbering conventions and the pointer for interpreting old
numbered citations. Numbers are labels, not a substitute for inspecting dependencies.

[rule-numbering]: dev-naming-conventions.md#9-rule-numbering-wnn-reference-scheme

## Find the current contract

| Change concerns | Owning reference |
|---|---|
| Model-free regions, vectors and location identities | [Spatial foundation](toolbox-spatial-foundation.md) |
| Model construction, forcing and mutable-root ordering | [Model construction](wf1-model-construction.md) |
| CMIP6 grid readability and source-version admission | [CMIP6 readability](wf2-cmip6-source-readability.md) |
| Projection windows, reductions and diagnostic availability | [WF2 guide][wf2-guide] |
| Scenario collections, retained responses and indicator sets | [Retained handoffs][handoffs] |
| Replacement weather generators | [Generator interface](toolbox-weather-generator-interface.md) |
| Replacement hydrological models | [Model interface](toolbox-hydrological-model-interface.md) |
| Figure ownership, annotation and export conventions | [Plot style](toolbox-plot-style.md) |
| Supported execution and log interpretation | [Workflow guide][workflow-guide] |

[wf2-guide]: ../../docs/site/toolbox-reference/workflow-analyze-projections.qmd
[handoffs]: ../../docs/site/toolbox-reference/retained-handoffs.qmd
[workflow-guide]: ../../docs/site/toolbox-reference/workflows.qmd

## Check dependencies and consumers

Read `input:`, `output:`, helper-produced declarations and terminal targets together. A file written
inside a script but absent from `output:` is not automatically a schedulable dependency. Search callers
outside the DAG as well: verification tools and retained-artifact readers can consume workflow products.

For mutable model directories, follow the [model-root ordering contract][model-ordering].
For optional or dynamically named products, inspect the producer's inventory and terminal completion
contract rather than assuming a fixed list of files. Gather rules must wait for all owned terminal work,
including independent branches, before merging execution records.

WF0 and WF1 share the native-grid map-field producer in
`blueearth_cst/climate_analysis/rules/source_map_preparation.smk`. It seals
`tables/annual_climatology.nc` separately from numerical diagnostic tables and display-reference
preparation. Rendering changes must not rerun this numerical producer. Historical WF1 simulation
responses in `run_default/output.csv` persist so exports and evaluation can rerun independently.

Validate the affected configured graph with the supported runner's dry-run interface.
[The DAG plotting tool](../../scripts/plot_workflow_dag.py) provides an executable graph view.
Select consumer checks through the [validation ladder](dev-validation.md); examples include
`tests/test_cross_workflow_inputs.py`, `tests/test_log_rules_contract.py` and the affected runner tests.
These are navigation pointers, not an additional mandatory suite.

[model-ordering]: wf1-model-construction.md#mutable-model-root-ordering
