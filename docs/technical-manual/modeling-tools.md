---
title: "Modeling and analysis tools"
format: html
---

The CST Toolbox combines tools for model preparation, weather generation, hydrological simulation and
indicator calculation. Each tool contributes a different part of the assessment; the scientific workflows
connect their inputs and outputs.

## Model preparation: HydroMT

HydroMT and its HydroMT-Wflow plugin prepare spatial and climate data and construct the basin model.
They connect cataloged datasets to the model's required inputs. The
[model construction workflow](workflow-build-model.md) describes the data preparation and configuration.

## Hydrological simulation: Wflow-SBM

Wflow-SBM simulates hydrological processes under historical and generated weather conditions. Model building
builds
the model and runs the historical simulation; hydrological simulation uses the completed model for the
stress-test scenarios.
See [model construction](workflow-build-model.md) and [system simulation](workflow-simulate-system.md).

## Weather generation: weathergenr

The stochastic weather generator, `weathergenr`, produces weather realizations conditioned on historical
climate. The scenario-generation workflow applies the specified climate perturbations. These scenarios
are generated independently of the climate projections used for the plausibility comparison. See
[scenario generation](workflow-generate-scenarios.md) for inputs, settings and outputs.

## Indicator calculation

The indicator module summarizes saved model time series into measures of water availability, flow extremes
and selected impacts. The following chapters explain [indicator selection](indicator-requests.md) and
[custom calculations](indicator-extension.md). The indicator-set appendices describe formulas, units,
data requirements and interpretation limits.

Software requirements are described under [workflow execution and automation](execution-tools.md).
