---
title: "Quick start"
subtitle: "Run the example basin and find its results"
---

Run the five-workflow rapid example on the supplied basin to see the output
layout before configuring your own case. Allow at least an hour; data access,
downloads, and local compute capacity affect the time.

First complete the [Windows setup guide](../site/setup/install.md), including
Pixi, juliaup-managed Julia 1.11.7, `pixi install`, and `pixi run install`.
From the repository root, confirm that the rapid config's `project.catalog`
entries and their data roots are available. Its base catalog expects the
Deltares P-drive at `P:/wflow_global/hydromt`; the ERA5 override also uses
that drive. CMIP6 acquisition needs network access. Installation alone does
not supply these data. See [Configuring a project](configuration.md)
if your data live elsewhere.


## The shape of a run

Five workflows share one project configuration set. The arrows show the
all-workflow runner's execution order. Select a workflow to see what it does.

<iframe src="assets/shape-of-a-run.html" title="Interactive diagram of the five workflows in execution order" style="width:100%;height:460px;border:0;" loading="lazy"></iframe>

[Open the interactive diagram](assets/shape-of-a-run.html) if your Markdown editor does not show it inline.

The model and generated climate scenarios are both needed for the stress test.
The historical climate comparison is optional; run it on its own when choosing
the climate data for the model. CMIP6 projections help judge which tested
climate changes are plausible; they do not determine the scenarios to simulate.

## Pick a configuration

Two example configuration sets ship under `test_case/`. Each has a project file
and one settings file per workflow; both describe the same basin:

| Config | Use it for |
|---|---|
| `test_case/project_config_rapid.yml` | First execution check: 2 × 2 perturbation grid plus the reference |
| `test_case/project_config_baseline.yml` | 2 × 3 grid plus the reference; not needed for a first run |

Both sets use two weather realizations and the 2046–2054 simulation window;
the rapid set reduces the perturbation grid. Its return levels are only an
execution check, not basin estimates. Review the period, stressor range, and
sample size for an assessment. The [configuration guide](configuration.md)
explains the project and workflow files.

## Run it

From the repository root, preview the enabled workflows first. The dry run
may show a partial DAG until earlier producers have created their inputs:

```powershell
pixi run python scripts/run_workflows.py --config test_case/project_config_rapid.yml --project-dir test_case/test_rapid --cores 3 -- --dry-run
```

Then run them:

```powershell
pixi run python scripts/run_workflows.py --config test_case/project_config_rapid.yml --project-dir test_case/test_rapid --cores 3
```

The command runs the workflows enabled in the project file, in this order:
historical climate, model, projections, scenario generation, and simulation.

WF1 builds the model and WF3 generates its independent climate series. WF4
uses both. WF2 compares projections with the stress-test range; it does not
feed WF3. Use the [running guide](running.md) for one-workflow commands and
experiment selection.

## What you get

The rapid config writes under `test_case/test_rapid/`. After a successful
run, open the exact `logs/<timestamp>__all-workflows/summary.md` named by
the closing console message. It links the workflow logs and benchmark reports.
The main products are:

```text
test_case/test_rapid/
  data/spatial/                       basin geometry and locations
  data/climate/                       historical forcing and projection products
  models/hydrology/wflow/             model, historical run, and evaluation
  scenarios/                          exported current collection and lookup tables
  scenarios/_engine/data/<id>/        immutable collection data
  experiments/experiment_rapid/       retained responses and indicator sets
  logs/<timestamp>__all-workflows/    summary, logs, and benchmarks
  config/                             configuration archives and invocation records
```

Find the hydrological indicators under
`experiments/experiment_rapid/results/indicator_sets/default/`, starting
with `indicators.csv` for definitions and units. Run-level tables join through
`scenarios/scenario_run_lookup.csv`; group-level tables join through
`group_metadata.csv` and `scenarios/perturbation_lookup.csv`. See
[Reading the outputs](outputs.md) for the exact joins and interpretation limits.

## When something goes wrong

- **The catalog root is missing** — check that the P-drive is mounted and the
  catalog source paths resolve. The [setup guide](../site/setup/install.md)
  shows the bounded path check.
- **The project is locked** — confirm that no other run is active, then use
  the [workflow recovery instructions](running.md#run-one-workflow).
- **`weathergenr` is missing** — complete `pixi run install` as described in
  the [setup guide](../site/setup/install.md).
- **Julia not found** — Julia is deliberately not in the pixi environment. It is
  managed by juliaup and must be on your `PATH`; see the
  [Setup guide](../site/setup/install.md).

## Next

- [Configuring a project](configuration.md) — point the toolbox at your own basin.
- [Running the workflows](running.md) — the flags, the wrapper, and experiments.
- [Reading the outputs](outputs.md) — what a run wrote, and whether it is current.
