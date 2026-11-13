---
title: "Wflow"
format: html
---

## Processes and model structure

Wflow-SBM is a distributed hydrological model implemented in Julia. It calculates water storage and
fluxes across a grid and routes water through the drainage network. Its soil representation includes
unsaturated and saturated storage, with vertical exchanges and lateral subsurface flow. Rainfall
interception, evapotranspiration, infiltration, runoff generation and routing connect atmospheric
forcing to river discharge.

The process diagram shows the connections between atmospheric inputs, surface and subsurface storage,
evapotranspiration and routed flow. It provides a conceptual guide to the model's water balance.

![Hydrological stores and fluxes represented by Wflow-SBM. Rainfall and snowmelt feed surface and soil stores; evapotranspiration returns water to the atmosphere, while runoff and subsurface flow feed the river network. Glacier and waterbody contributions depend on the configured model features.](assets/wflow-sbm-processes.png){width=100% fig-alt="Wflow-SBM process diagram showing interception, snow and glacier melt, evaporation, transpiration, infiltration, unsaturated and saturated soil stores, capillary rise, overland flow, subsurface flow and river routing." style="background-color: white;"}

Snow accumulation and melt influence the seasonal water balance where temperatures cross the snow
thresholds. Glacier and waterbody processes contribute where the model contains those features.
The supplied waterbody recipe distinguishes reservoirs with simple control from waterbodies without
that control. These representations should be inspected in regulated basins; they do not establish
that actual operating rules, abstractions or transfers are reproduced.

The toolbox runs daily simulations. River discharge is reported at selected outlets or gauges;
other selected variables, such as precipitation, evapotranspiration and recharge, are retained as
subcatchment means. These spatial supports matter when comparing a discharge record with an areal
water-balance quantity or interpreting a result for a particular site.

## Parameters and spatial discretization

The model grid determines the spatial units over which processes are calculated. Topography and
hydrography determine drainage connections, while soil and land-cover information provide spatially
varying properties. The recipe also fixes parameters such as snow thresholds and melt coefficients,
infiltration controls and the ratio of horizontal to vertical saturated hydraulic conductivity.

Choosing a finer grid increases the number of cells and computational demand. It can improve the
representation of basin geometry where the source data support it, but it does not add information
to coarse meteorological forcing or uncertain soil estimates. Model resolution should therefore be
considered together with source resolution, basin size and the locations whose responses matter.

The current build uses Wflow's standard parameter names and configuration conventions. Some parameters
are stored as spatial maps and others as scalar values in the model configuration. When reviewing a
model, examine both; the maps alone do not contain every active parameter choice.

## Initial conditions and warm-up

Initial soil moisture, groundwater storage, snow and channel storage can influence the early part of
a simulation. Wflow supports a cold start using its initialization settings or a warm start from saved
model states. The active model configuration determines which is used; the presence of a state-file
path alone does not mean that the states are read.

For historical evaluation, the toolbox excludes the first simulation year from its hydrological
summaries when the record contains more than 365 time steps. Shorter records retain the initial period
and are reported as such. This is an evaluation convention, not evidence that every basin has reached
an adequate initial state after one year.

Stress-test simulations (WF4) set initialization through `simulate_system.initialization`:

- **State.** `auto` (default) keeps a warm state configured in the model TOML; otherwise it starts every
  run from WF1's historical end state (`historical/default/run/outstate/outstates.nc`). `historical` requires that
  end state and `cold` uses Wflow's cold initialization; either is refused when the model TOML already
  configures a warm state. The end state's bytes are part of the simulation identity, so a rerun WF1
  history requires a new experiment.
- **Warm-up.** `warmup_years` (default 1) simulates that many years before the configured window by
  repeating the run's own first days, so stores adjust to the run's perturbed climate. Response reading
  discards these days. Responses therefore keep the configured calendar, and indicators never see the
  warm-up. Leap days can shift the repeated block's seasonal alignment by at most one day per year.

The state's date need not match the stress-test calendar. Its quality depends on WF1: a short historical
run that itself started cold yields a near-cold end state. In wrt-egypt-v3, cold starts drained for about
seven years and a -40% rainfall response ranged from -10% to -82% depending on the first analysed year.
Starting from the historical end state made it -82% to -83%. Wet perturbations still filled stores for
about a year, which the default warm-up covers.

Slowly responding stores, typically groundwater, can keep adjusting to a perturbed climate after the
warm-up; in wrt-egypt-v3, recharge still drifted by 1-2 mm/yr after 20 years. This drift is part of the
simulated transient response. Check storage-sensitive indicators against it, particularly for short
windows and basins with large groundwater or lake storage.

## Assumptions and applicability

Wflow-SBM provides a physically structured representation suitable for examining basin-scale
hydrological responses. Its application here remains conditional on forcing quality, parameter
estimation and the adequacy of the represented processes. Successful execution does not establish
agreement with observations or fitness for a particular planning decision.

Evaluate more than mean discharge where observations are available. Seasonal timing, high and low
flows, persistence and water-balance plausibility may respond differently to model errors. A model
that reproduces annual flow can still misrepresent drought duration or peak flows. Daily discharge
also cannot resolve subdaily flood peaks, inundation or damage without further analysis.
