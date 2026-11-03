---
title: "Uncertainty and Limitations"
format: html
---

The toolbox automates basin preparation and climate stress testing from global data. Its checks can
detect incompatible inputs, but a successful run does not establish local hydrological accuracy.
The following limitations follow from the data and methods used in this implementation.

**Historical forcing.** The same selected climate record drives the historical Wflow run and
conditions weathergenr, so rainfall bias can affect both. CHIRPS replaces precipitation while ERA5
still supplies temperature, pressure and radiation; it is not an independent replacement of the full
meteorological record. Use historical climate analysis to compare candidate sources and check them against
local information.

**Spatial detail.** The starter's approximately 1 km model grid combines inputs ranging from finer
land-cover and soil maps to much coarser ERA5 meteorology. Resampling does not resolve local rainfall
or soil variability. The generator uses equal weights for climate cells intersecting the basin,
without fractional-area weighting; small basins and strong climate gradients need particular care.
Inspect `location_registry.csv`, basin boundaries and the spatial maps, especially near confluences.

**Model parameters and water management.** HydroMT-Wflow derives parameters from global inputs;
the workflow does not calibrate them to the basin. The build recipe estimates river depth using a
power-law method and soil hydraulic conductivity using the Brakensiek pedotransfer function, rather
than local channel surveys or soil measurements. Lake and reservoir routines are included, but the
supplied recipes select reservoirs of at least 1 km², lakes of at least 10 km² and glaciers of at least
1 km². Inventory-based storage and simple reservoir controls do not establish actual operating rules
or reproduce unconfigured abstractions and transfers. Check the active model configuration, not just
the waterbodies shown on the map.

**Weather generation.** weathergenr resamples historical daily fields and adjusts their statistics.
More realizations explore its stochastic variability; they do not add independent evidence about
unobserved extremes or remove forcing and model-parameter uncertainty. The supplied evaluation recipe
plots at most 25 grid cells, so larger domains require additional spatial review. Compare generated
unperturbed conditions with the historical record before interpreting perturbed scenarios.

**Projection coverage.** projection analysis selects from the cataloged models and available preferred
members;
the configured ensemble may be much smaller than the archive. Monthly precipitation and temperature
changes do not describe changes in daily extremes, and ensemble spread is not a calibrated probability
of basin outcomes. Compare consistent model/member coverage and reference periods across scenarios.

For a sensitivity assessment, compare defensible forcing sources or model configurations as separate
runs while retaining the same stress-test and indicator definitions. Record the differences alongside
the results. Input hashes, catalog checks and saved run records support reproducibility; they do not
replace evaluation against observations or establish that an indicator is adequate for a decision.
