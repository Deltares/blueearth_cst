---
title: "Workflow structure and dependencies"
format: html
---

The usual execution order is historical climate analysis → model building → projection analysis → scenario
generation → hydrological simulation. Model building and scenario generation independently supply
hydrological simulation.
Projection analysis is a projection plausibility overlay; its CMIP6 changes never determine generated weather.
All products are written below `project.project_dir`.

| Workflow | Entry point | Principal product |
|---|---|---|
| [historical climate analysis](workflow-analyze-climate.md) | `analyze_climate.smk` | Historical source diagnostics and comparisons |
| [model building](workflow-build-model.md) | `build_model.smk` | Completed model and historical simulation/evaluation |
| [projection analysis](workflow-analyze-projections.md) | `analyze_projections.smk` | Basin change factors and projection diagnostics |
| [scenario generation](workflow-generate-scenarios.md) | `generate_scenarios.smk` | Ready scenario collection |
| [hydrological simulation](workflow-simulate-system.md) | `simulate_system.smk` | Retained responses and named indicator sets |

## Historical reference and model setup

Historical climate analysis describes baseline climate variability and the climatic context of
hydrological conditions, supporting selection of the gridded forcing dataset. It shares basin and
historical climate preparation with the other workflows. Model building constructs the model from
catalog data and simulates the historical hydrological response. Observations enable evaluation where
supplied; no local calibration is performed.

## Projection context and stress-test experiment

Projection analysis compares CMIP6 historical and future windows. Scenario generation conditions a weather
generator on the selected
historical source, then perturbs generated realizations over the configured climate grid. Scenario generation
needs no
Wflow model. Hydrological simulation needs a completed prepared hydrological model and explicit runs from a
ready collection. It consumes no projection analysis
product. An indicators-only request can reuse compatible retained hydrological simulation responses.

Use `scripts/run_workflows.py` for the configured pipeline, including scenario generation preparation and
publication.
Use `scripts/simulate_system.py` for direct hydrological simulation. Bare Snakemake is not a supported
scenario generation and hydrological simulation execution route.
The [running guide](../guide/running.md) explains commands and recovery; [configuration](configuration.md)
explains how settings are combined and paths are resolved. Rule numbers are reference labels, not a serial
schedule: independent branches
can run concurrently.

The runner's final verdict includes result publication. Progress or a completed simulation batch alone is
not confirmation that every requested indicator set is current. Exact records and console conventions are
described in [run records and reproducibility](execution-and-provenance.md).
