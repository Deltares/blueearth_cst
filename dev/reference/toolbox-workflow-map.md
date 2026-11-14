# Workflow implementation map

Last updated: 2026-11-14

Use this map to find a rule, its implementation and the products it reads or writes, and to name or
number a new rule. The Snakefiles own exact rule names, numbers, inputs, outputs and dependencies. The
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
[Rule numbers](#rule-numbers) define the label convention. Numbers are reading aids, not execution
order; inspect dependency edges.
Gather rules must wait for all terminal work they own, including independent branches, before
combining logs or benchmarks.

## Rule names

Action rules use `<verb>_<noun>`, verb first; aggregate targets such as `all` may use a noun
([naming conventions §2](dev-naming-conventions.md#2-per-language)).
Choose the verb by action class from this table.

| Verb | Action class |
| --- | --- |
| `fetch_` | acquire from an external source |
| `extract_` | subset or derive from a larger source already present |
| `delineate_` | derive a catchment boundary from hydrography and an outlet |
| `prepare_` | **compute or assemble** something a later rule needs |
| `build_` | construct a model from inputs |
| `add_` | mutate an existing model in place by adding **data** (a hydromt `update`) |
| `declare_` | change what an engine will **emit**, adding no model data |
| `write_` | **emit a record or index** — the emission *is* the work |
| `generate_` | stochastic or synthetic production |
| `downscale_` | resolution transform |
| `perturb_` | apply a climate perturbation to an existing series |
| `run_` | invoke an external engine |
| `reduce_` | **intermediate** aggregation that feeds a later rule |
| `produce_` | shared preparation and publication consumed by more than one workflow |
| `derive_` | compute a workflow's **terminal product** from reduced inputs |
| `plot_` | render a figure |
| `check_` | validate, fail loud |
| `publish_` | **validate a finished product and mark it ready** — the ready marker is written last |
| `claim_` | reserve a durable identity before anything is written under it |
| `snapshot_` | freeze or copy inputs for provenance, so a later step cannot drift from them |
| `gather_` | merge parts |

Use these distinctions when choosing a verb:

- **`reduce_` vs `derive_`:** both aggregate inputs, but `reduce_` feeds another rule while
  `derive_` produces the terminal product (`derive_change_factors`, `derive_system_indicators`).
- **`prepare_` vs `write_`:** use `prepare_` when computation or assembly remains beyond file writing;
  use `write_` when producing the record or index is the work. Downstream use does not decide the verb:
  `write_climate_data_catalog` remains `write_` because emitting the catalog is the work.

Use full nouns except established `gcm`, `cmip6`, `wflow`, `rlz` and `st`. Do not introduce ad hoc
contractions such as `weagen` or `proj`; existing `proj` names await their own migration. Put
qualifiers after the noun and spell them out.

Add a verb only for a distinct action class with a rule that uses it.

Check the verb against the rule's script or shell body when work moves; grammatical naming alone can
misdescribe the action.

## Rule numbers

Each rule in the five workflow definitions has a `W.NN` reference number. `W` is the workflow ID:
`0` analyze_climate, `1` build_model, `2` analyze_projections, `3` generate_scenarios or
`4` simulate_system. `NN` is its zero-padded position in logical order: data, build, run, records.

It exists in exactly two places:

- A comment header above each rule, such as `# 1.06  build_wflow_model`.
- **The `log:` / `benchmark:` filename prefix** — `1.06_build_wflow_model.log` or `.tsv`.
  Managed launches place parts under `logs/<execution>/_parts/wfN-<child-id>/{logs,benchmarks}/`;
  a bare `snakemake` WF0–WF2 run uses `logs/_parts/` and `logs/benchmarks/_parts/`.
  For wildcard rules the prefix goes on the subdirectory (`4.05_run_wflow_simulations/batch_{b}.log`).
  The `W` digit keeps rule labels disambiguated and sorts them by workflow then step within each namespace.

Numbering conventions:

- **Logical order**, with gaps after removals and letter suffixes for insertions; do not close gaps by
  renumbering surviving rules. Target rules (`all`, `simulations_only`, `simulations_and_indicators`)
  carry no number: they aggregate work rather than execute a pipeline step.
- Every dependency points from a lower to a higher number, including `ancient()` inputs.
  `ancient()` suppresses a timestamp rerun trigger, not the DAG edge.

Rule numbers have been reused. Interpret an older `W.NN` citation against the Snakefile revision it
describes (`git show <commit>:<snakefile>`); do not read it as a current rule ID or rewrite retained evidence.

Rules:

- Rule *identifiers* are not numbered (MUST). Snakemake identifiers cannot contain a leading digit or
  dot, and they are CLI targets. Put `W.NN` only in comments and log or benchmark paths.
- `W.NN` is a reference and reading aid, not execution order. Snakemake uses the DAG; independent
  branches can run concurrently. Low-to-high means a lower-numbered rule cannot depend on a higher one.
- Source definition order may differ from numbering because module-level code depends on placement.
  `LOG_RULES` must remain in number order (`tests/test_log_rules_contract.py`): it controls workflow
  log merge order.
- In prose and commits use “Rule 1.3”; use padded `1.03` in sortable filenames.
- **Do not renumber to insert a rule.** Use a letter suffix (`1.09b`) until a deliberate workflow-wide
  migration. Numbers appear in `LOG_RULES`, log and benchmark paths, `rule_banner`, comments and prose.
  An omitted `LOG_RULES` label silently drops its log section. A suffix retains sorting order:
  `"1.09" < "1.09b" < "1.10"`.
- "Rule 1.5" decimals are review shorthand for *talking about* an insert, never a permanent identifier.

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
