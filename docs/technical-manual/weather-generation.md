---
title: "weathergenr"
format: html
---

## Historical conditioning and weather variability

The R package `weathergenr` generates daily weather sequences conditioned on the historical climate
record. Each realization is a different possible sequence under the chosen statistical conditions;
its dates label the simulation period rather than predict the weather on particular future days.
The toolbox generates precipitation and mean, minimum and maximum temperature over the climate grid.

With the default spell factors, an unperturbed realization represents the generator's historical-climate
conditions. It is distinct from the observed or reanalysis record used to condition the generator and
from a hydrological model run driven directly by that record. Comparing generated unperturbed
conditions with historical data helps identify generator effects before interpreting the additional
effect of climate perturbations. If spell factors are changed, they also affect the root realizations
from which the temperature–precipitation grid is produced.

## Generation methods and spatial dependence

The generator combines three statistical operations. A wavelet autoregressive model represents
low-frequency variation in area-average annual precipitation. A Markov-chain component represents
daily weather-state transitions, and nearest-neighbour resampling produces daily, multivariable
weather fields consistent with the generated annual conditions. Distributional adjustments then
apply the specified climate perturbations.

Resampling related variables and grid cells together retains relationships present in the historical
record, rather than generating independent weather at every cell. Nevertheless, spatial dependence,
temperature–precipitation relationships and the persistence of wet and dry periods remain properties
to evaluate. Longer synthetic sequences explore combinations permitted by the generator; they do not
create new historical evidence about rare events or multidecadal variability.

The generation workflow produces diagnostic plots. The supplied recipe limits the evaluation plots
to 25 grid cells, so those plots do not constitute an exhaustive spatial check for larger domains.
Review basin-wide statistics and the cells or subregions relevant to the assessment, particularly
where climate gradients are strong.

## Climate perturbations and temporal trajectories

The main stress-test grid combines temperature and precipitation levels. Each level carries twelve
monthly values, allowing changes to differ by season. Temperature changes are additive in degrees
Celsius. Precipitation mean changes are multipliers: 0.8 specifies a 20% reduction and 1.2 a 20%
increase. For each month, `n_levels` evenly spaces values between that month's minimum and maximum,
including both bounds. The same level index selects all twelve months, so a level is a seasonal
pattern, not twelve independent choices. Every temperature level is paired with every precipitation
level. With one level, the minimum is used.

Temperature offsets are added equally to daily mean, minimum and maximum temperature; the supplied
recipe therefore leaves the daily temperature range unchanged. Precipitation is adjusted through
Gamma quantile mapping of positive daily amounts, fitted using the method of moments. The supplied
recipe enforces the target wet-day mean, disables additional extreme-tail amplification and sets no
precipitation cap or floor. Its intensity threshold is zero; this differs from historical climate analysis's
default 1 mm/day
threshold for reporting wet days.

**Variance control.** In the supplied recipe, `scale_var_with_mean: true` sets the target wet-day
variance factor to the square of the precipitation mean factor. For a constant mean factor of 1.2,
the variance factor is 1.44. Separate `precip.variance.min/max` values are passed to the generator but
ignored under this setting. To use those bounds, set `scale_var_with_mean: false` in the generator
recipe. Their levels follow the precipitation level index; they do not add a third grid axis.
These are distributional targets, not guarantees that a finite generated sample has exactly the
specified variance.

**Constant and transient changes.** A constant trajectory applies the selected offset or factor
throughout the run. In weathergenr 2.0.1, a transient temperature offset ramps linearly from zero to
twice the selected offset across the calendar years represented in the generated dates. A
precipitation factor ramps from 1 to twice the selected factor minus 1. Thus, the selected values
describe the arithmetic mean of the annual imposed settings, not their final values or a guaranteed
change in the realized whole-period climate. Changes are constant within each calendar year for a
given month; they are not interpolated daily.

The variance factor is ramped separately: with default mean-linked variance, its endpoint is twice
the squared mean factor minus 1. The generator floors precipitation mean and variance factors at
0.01. Strong drying can reach that floor and break the intended average-change relationship. A
single-calendar-year transient run receives the initial, unchanged settings. Whole-period indicators
from transient runs combine changing conditions and cannot represent a constant endpoint climate.

### Example: from monthly bounds to scenarios

Suppose all twelve monthly temperature bounds are 0 and 2 °C with three levels, and precipitation
bounds are 0.8 and 1.2 with three levels. This creates nine paired climate conditions:

| Setting | Sampled levels |
|---|---|
| Temperature offset | 0, 1, 2 °C |
| Precipitation mean factor | 0.8, 1.0, 1.2 |
| Default constant variance factor | 0.64, 1.00, 1.44 |

For the +1 °C, 1.2 precipitation condition, a constant run applies +1 °C and a 20% wet-day mean
increase throughout. A five-calendar-year transient run instead applies temperature offsets of
0, 0.5, 1, 1.5 and 2 °C, and precipitation factors of 1, 1.1, 1.2, 1.3 and 1.4.
Its default variance factors run from 1 to 1.88, rather than being the square of each annual mean
factor. The lookup still records the selected +1 °C and +20% settings, so read it together with
the saved trajectory and generator recipe.

`perturbation_lookup.csv` contains twelve monthly rows per design point and expresses precipitation
mean and variance settings as percentage changes. `scenario_run_lookup.csv` links each daily NetCDF
series to its design point and realization. Two realizations in this example produce eighteen grid
series plus two generated reference series. A grid point with unchanged factors remains distinct
from the separately retained reference. See [scenario generation outputs](workflow-generate-scenarios.md#outputs).

Optional monthly dry- and wet-spell factors modify mean spell lengths. They apply to weather generation
without adding another grid dimension. Unchanged factors are one. Precipitation distribution
adjustments and spell controls have different purposes: the same mean rainfall can accompany
different event intensities or persistence, with different hydrological consequences.

The unperturbed realization is retained alongside the perturbed scenarios. The random seed and
generation settings are recorded so the weather experiment can be identified and reused. Neither
CMIP6 model selection nor emissions-scenario selection defines these perturbations.

## Assumptions and applicability

The historical record must represent the climate characteristics needed by the generator. The toolbox
requires at least 16 inclusive historical years, but this minimum does not ensure that low-frequency
variability or extremes are well characterized. Missing data, changing station support and biases in
the gridded source can influence both the unperturbed sequences and their perturbations.

Compare generated and historical seasonality, distributions, wet- and dry-spell behavior and spatial
relationships. Under large perturbations, examine whether combinations of temperature and rainfall
remain physically credible for the intended experiment. The generator explores prescribed statistical
changes; it does not simulate atmospheric circulation or establish their future likelihood.
