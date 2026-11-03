---
title: "Overview"
format: html
---

The toolbox combines external modelling and data-preparation tools with modules for indicator
calculation and visualization. Together they prepare the basin model, generate weather scenarios,
simulate hydrological responses and present results for interpretation. The linked pages explain each
component's role, methods and limitations.

- **[HydroMT and HydroMT-Wflow](model-preparation.md):** a model-building framework and its Wflow-specific
  plugin. They prepare spatial and climate data and construct the basin model.
- **[Wflow and Wflow-SBM](hydrological-simulation.md):** hydrological modelling software and the model
  used by the toolbox to simulate catchment processes under historical and generated weather.
- **[weathergenr](weather-generation.md):** a stochastic weather-generation package that produces
  realizations conditioned on historical climate for the specified climate experiment.
- **[Indicator module](indicator-calculations.md):** toolbox calculations that convert simulated time
  series into measures of water availability, flow regimes, extremes and selected impacts.
- **[Visualization module](results-and-interpretation.md#evaluation-and-diagnostic-products):** toolbox
  plotting functions that produce maps, diagnostic plots and figures for interpreting results.
