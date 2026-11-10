# Current development and toolbox reference

Last updated: 2026-11-05

Use this index to find the current public reference for a development or toolbox question. Update the
owning reference when behavior changes; executable sources retain exact schemas, defaults and commands.

## Development practices

| When the change concerns | Update or consult |
|---|---|
| Placement, config directories, scratch | [dev-repository-layout.md][layout] |
| Document authority and retention | [dev-record-lifecycle.md][lifecycle] |
| File, identifier or output naming | [dev-naming-conventions.md](dev-naming-conventions.md) |
| Verification, evidence reuse, landing or CI | [dev-validation.md](dev-validation.md) |
| Workflow console orientation, commands and work plans | [dev-console-style.md](dev-console-style.md) |
| Gate receipts, Windows tests, E2E bundles | [dev-validation-procedures.md][procedures] |
| Git branch, tag or commit conventions | [dev-git-conventions.md](dev-git-conventions.md) |
| Release/version decisions | [dev-versioning.md](dev-versioning.md) |
| CST documentation content, presentation and publishing | [dev-documentation.md](dev-documentation.md) |

## Architecture and implementation

| When the change concerns | Update or consult |
|---|---|
| Workflow boundaries and retained handoffs | [toolbox-architecture.md][architecture] |
| Regions, spatial IDs, map ownership | [toolbox-spatial-foundation.md][spatial] |
| Wflow build and model-root ordering | [wf1-model-construction.md][model] |
| Forcing catalog admission and unit interpretation | [toolbox-forcing-units.md](toolbox-forcing-units.md) |
| Generation inputs, outputs or replacement engines | [toolbox-weather-generator-interface.md][generator] |
| Simulator preparation and responses | [toolbox-hydrological-model-interface.md][simulator] |
| Rule definitions, dependencies and consumers | [toolbox-workflow-map.md](toolbox-workflow-map.md) |
| CMIP6 grids and source versions | [wf2-cmip6-source-readability.md][cmip6] |
| Indicator terminology | [toolbox-indicator-glossary.md][indicators] |

## Plotting and presentation

| When the change concerns | Update or consult |
|---|---|
| CST figure appearance and interpretation | [cst-style.md][cst-style] |
| Weathergenr figures in CST | [weathergenr-style.md][weather-style] |
| Measured ggplot2/Matplotlib panel geometry | [panel-layout.md][panel-layout] |
| Figure filenames and profiles | [figure-naming.md][figure-naming] |
| Reusable plotting helpers | [plotting/README.md](../../blueearth_cst/plotting/README.md) |
| A workflow report template | [templates/README.md](../templates/README.md) |

## Find guidance by workflow

Use this lookup to reach the relevant topic owners; a workflow does not need its own reference file.
[toolbox-architecture.md](toolbox-architecture.md) owns shared boundaries, and the
[toolbox-workflow-map.md](toolbox-workflow-map.md) explains how to inspect rules and consumers.

- **WF0 historical climate:** [spatial foundation][spatial], [figure naming][wf0-figures];
  [user guide][guide0]; [rules][rules0].
- **WF1 model construction:** [model construction][model], [forcing units][forcing];
  [user guide][guide1]; [rules][rules1].
- **WF2 projections:** [CMIP6 source readability][cmip6], [spatial foundation][spatial];
  [user guide][guide2]; [rules][rules2].
- **WF3 scenarios:** [weather-generator interface][generator], [retained handoffs][handoffs];
  [user guide][guide3]; [rules][rules3].
- **WF4 simulation:** [hydrological-model interface][simulator], [indicator glossary][indicators];
  [user guide][guide4]; [entry point][rules4].

[spatial]: toolbox-spatial-foundation.md
[layout]: dev-repository-layout.md
[lifecycle]: dev-record-lifecycle.md
[procedures]: dev-validation-procedures.md
[architecture]: toolbox-architecture.md
[cst-style]: ../../blueearth_cst/plotting/docs/cst-style.md
[weather-style]: ../../blueearth_cst/plotting/docs/weathergenr-style.md
[figure-naming]: ../../blueearth_cst/plotting/docs/figure-naming.md
[wf0-figures]: ../../blueearth_cst/plotting/docs/figure-naming.md#historical-climate-diagnostics-wf0
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
[rules4]: ../../scripts/simulate_system.py
[panel-layout]: ../../blueearth_cst/plotting/docs/panel-layout.md

## Maintaining references

Keep one current home per topic. Add a reference when it answers a distinct developer question; link it
here and remove duplicate rules. [dev-record-lifecycle.md](dev-record-lifecycle.md) explains how current
guidance relates to retained records. Check source and tests before stating that a design is implemented.
