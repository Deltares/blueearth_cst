# Current development and toolbox reference

Last updated: 2026-10-20

This is the shared source of current guidance for developers and agents: what applies in this checkout,
how the toolbox is organized, and where to check compliance. Use this map to find the owning document
before writing. Update that document for revisions; do not create a separate file per decision or task.

## Development practices

| When the change concerns | Update or consult |
|---|---|
| Repository placement, config directories or local scratch | [dev-repository-layout.md](dev-repository-layout.md) |
| Documentation authority, promotion, retention or sealing | [dev-record-lifecycle.md](dev-record-lifecycle.md) |
| File, identifier or output naming | [dev-naming-conventions.md](dev-naming-conventions.md) |
| Verification, evidence reuse, landing or CI | [dev-validation.md](dev-validation.md) |
| Git branch, tag or commit conventions | [dev-git-conventions.md](dev-git-conventions.md) |
| Release/version decisions | [dev-versioning.md](dev-versioning.md) |
| User documentation structure and authoring | [dev-documentation.md](dev-documentation.md) |
| Editorial preferences and collaborative page review | [Revision guidance](documentation-revision-guidance.md) |

- Task admission, sessions, board coordination, handoffs or Herdr worker coordination:
  private/dev-task-and-session-workflow.md (private-checkout-only citation)
- Project agent activation and diagnostics: private/dev-agent-activation.md (private-checkout-only citation)

## Architecture and implementation

| When the change concerns | Update or consult |
|---|---|
| Workflow boundaries, configuration ownership or retained handoffs | [toolbox-architecture.md](toolbox-architecture.md) |
| Shared regions, spatial identities or map ownership | [toolbox-spatial-foundation.md](toolbox-spatial-foundation.md) |
| Wflow construction, constant parameters or model-root ordering | [wf1-model-construction.md](wf1-model-construction.md) |
| Forcing catalog admission and unit interpretation | [toolbox-forcing-units.md](toolbox-forcing-units.md) |
| Generation inputs, outputs or replacement engines | [toolbox-weather-generator-interface.md][generator] |
| Simulator preparation, responses or replacement engines | [toolbox-hydrological-model-interface.md][simulator] |
| Rule definitions, dependencies and consumers | [toolbox-workflow-map.md](toolbox-workflow-map.md) |
| CMIP6 grid handling or source-version selection | [wf2-cmip6-source-readability.md](wf2-cmip6-source-readability.md) |
| Output-variable names and indicator terminology | [toolbox-indicator-glossary.md](toolbox-indicator-glossary.md) |

## Plotting and presentation

| When the change concerns | Update or consult |
|---|---|
| Scientific figure appearance, export or review practice | [toolbox-plot-style.md](toolbox-plot-style.md) |
| Reviewed WF3 weather diagnostics and figure exceptions | [weather-diagnostic-style.md](weather-diagnostic-style.md) |
| Measured R/Python diagnostic-grid geometry | [panel-layout-contract.md](panel-layout-contract.md) |
| WF0 diagnostic figure filename grammar | [wf0-figure-naming.md](wf0-figure-naming.md) |
| A reusable report or comparison interface | [templates/README.md](../templates/README.md) |

## Find guidance by workflow

Use this lookup to reach the relevant topic owners; a workflow does not need its own reference file.
[toolbox-architecture.md](toolbox-architecture.md) owns shared boundaries, and the
[toolbox-workflow-map.md](toolbox-workflow-map.md) explains how to inspect rules and consumers.

| Workflow | Development guidance | User guide | Rule definitions |
|---|---|---|---|
| WF0: historical climate | [toolbox-spatial-foundation.md][spatial], [wf0-figure-naming.md][wf0-figures] | [workflow-analyze-climate.qmd][guide0] | [analyze_climate.smk][rules0] |
| WF1: model construction | [wf1-model-construction.md][model], [toolbox-forcing-units.md][forcing] | [workflow-build-model.qmd][guide1] | [build_model.smk][rules1] |
| WF2: projections | [wf2-cmip6-source-readability.md][cmip6], [toolbox-spatial-foundation.md][spatial] | [workflow-analyze-projections.qmd][guide2] | [analyze_projections.smk][rules2] |
| WF3: scenarios | [toolbox-weather-generator-interface.md][generator], [retained-handoffs.qmd][handoffs] | [workflow-generate-scenarios.qmd][guide3] | [generate_scenarios.smk][rules3] |
| WF4: simulation | [toolbox-hydrological-model-interface.md][simulator], [toolbox-indicator-glossary.md][indicators] | [workflow-simulate-system.qmd][guide4] | [simulate_system.smk][rules4] |

[spatial]: toolbox-spatial-foundation.md
[wf0-figures]: wf0-figure-naming.md
[model]: wf1-model-construction.md
[forcing]: toolbox-forcing-units.md
[cmip6]: wf2-cmip6-source-readability.md
[generator]: toolbox-weather-generator-interface.md
[simulator]: toolbox-hydrological-model-interface.md
[handoffs]: ../../docs/site/toolbox-reference/retained-handoffs.qmd
[indicators]: toolbox-indicator-glossary.md
[guide0]: ../../docs/site/toolbox-reference/workflow-analyze-climate.qmd
[guide1]: ../../docs/site/toolbox-reference/workflow-build-model.qmd
[guide2]: ../../docs/site/toolbox-reference/workflow-analyze-projections.qmd
[guide3]: ../../docs/site/toolbox-reference/workflow-generate-scenarios.qmd
[guide4]: ../../docs/site/toolbox-reference/workflow-simulate-system.qmd
[rules0]: ../../analyze_climate.smk
[rules1]: ../../build_model.smk
[rules2]: ../../analyze_projections.smk
[rules3]: ../../generate_scenarios.smk
[rules4]: ../../simulate_system.smk

## How to maintain this collection

Keep this collection flat and organize navigation here by reader question.
Use `dev-` for repository practices, `toolbox-` for shared technical concerns, and `wfN-` for
workflow-specific topics. Keep `README.md` as the index.
A new file needs a distinct reader question and maintenance boundary, not merely a new task or decision.
Add its ownership to this map and remove any duplicate account in the same change.

Exact defaults, schemas, inventories and checks remain with their executable owners. References explain
meaning and relationships and point to those owners. If code and guidance disagree, resolve the discrepancy
before claiming compliance; acceptance of a historical design does not prove implementation.

Proposals belong with `dev/tasks/`, reusable artifacts in `dev/templates/`, and dated evidence in
records/README.md (private-checkout-only citation: `../records/README.md`). sealed-records.yml
(private-checkout-only citation: `../records/sealed-records.yml`) owns preservation metadata. Keep revision
histories, superseded specifications and session progress out of this folder. The
[dev-record-lifecycle.md](dev-record-lifecycle.md) owns the detailed maintenance policy.
