---
title: "Indicator module"
format: html
---

## Hydrological and impact indicators

Indicators translate retained daily model responses into measures that can be compared across climate
conditions. Select them according to the assessment question and retain the model variables they
require. The three supported sets have different purposes:

- **Default** summarizes average conditions, high and low flows, seasonality and water-balance
  components. Its selection depends on the variables retained by the simulation.
- **ERFA** describes flow magnitude, timing, variability and intermittency through 35 annual metrics,
  each summarized by its median and interquartile range. These measures support flow-regime comparison;
  they do not classify ecological risk or predict species responses.
- **Crop-water** uses precipitation, evapotranspiration, recharge and discharge to screen seasonal water
  deficits and low-flow conditions. Its catchment-scale proxies are not crop-yield or irrigation-demand
  estimates, and its supplied seasonal periods need to be checked against local conditions.

## Definitions, units and aggregation

The indicator dictionary records the selected definitions, parameters and units. Discharge is a flow
rate in m³/s, while precipitation and evapotranspiration are depths accumulated over specified periods.
Recharge is signed net recharge. A depth total from an incomplete year cannot be interpreted as a
complete annual water balance.

Annual calculations generally follow the configured water year, but individual sets may define their
own periods. Crop-water, for example, uses fixed winter and summer periods that overlap in May and an
October–September recharge year. Missing-data and completeness rules also differ by calculation;
coverage diagnostics show which periods contribute.

Ordinary group summaries give each realization equal weight. ERFA therefore averages the medians and
interquartile ranges calculated for individual realizations; it does not calculate a single median
over all pooled years. Undefined member results propagate to the group rather than being silently
treated as zero. Return levels use a separate pooled-extremes calculation.

Reference definitions also matter. Default reference-month indicators use the recorded unperturbed
simulations. ERFA pulse thresholds are estimated separately from each supplied series. The Crop-water
low-flow calculation instead uses a fixed location-specific threshold, which can be derived from
historical simulated discharge or supplied with its provenance. These measures answer different questions
even when they all describe low-flow change.

## Return-level estimation and supporting evidence

The Default set includes a 10-year maximum-flow return level and a 2-year return level for seven-day
low flows. The calculation extracts annual extremes separately within each realization, pools eligible
annual values across realizations and fits a generalized extreme-value distribution using L-moments.
Seven-day windows never join the end of one realization to the beginning of another.

The fitting step requires at least the larger of ten annual values or the requested return period
rounded up to a whole year. Partial annual blocks can contribute under the supplied extraction rules,
so inspect coverage as well as the number of fitted values. Passing the sample-size check is an
operational requirement, not a reliability guarantee.

The estimator has supporting checks over a bounded set of synthetic cases and probability levels.
It reports fitting information and refusals but provides no uncertainty intervals or basin-specific
validation. Pooling generated years increases the number of simulated extremes, not the length of
the independent historical record. For transient runs, the pooled sample also combines changing
climate conditions. Treat these return levels as screening measures and examine the retained fit
evidence before using them in a decision.

### Benchmark records and fitted parameters

Read `return_level_validation` in `indicators.json` to identify the return periods and conditions
assessed by the supplied benchmark. `return_level_benchmark.json` holds that assessment. These checks
do not establish accuracy for a particular basin; the return levels are not design estimates.

The recorded generalized extreme-value (GEV) parameters use SciPy's `(c, loc, scale)` convention.
The shape parameter often written as ξ is `-c`; check this sign when comparing fits with another tool.
Each fit also reports `shape_coverage`, including how far its estimated `c` falls outside the shapes
tested in the benchmark. A fit outside that range has not been assessed by this benchmark. Small
departures are not strong evidence on their own because shape estimates from short records are uncertain.

Keep an indicator table together with its `indicators.json` and `return_level_benchmark.json` when
sharing or comparing return levels. The CSV alone does not identify the estimator or its assessment.

## Custom indicator calculations

Additional indicators can be implemented in the toolbox's Python indicator module. Each calculation
declares its required variables and units, parameters, output definition, assumptions and handling of
missing data. Calculations using several variables require aligned dates, locations and time steps.
The implementation should include an independent expected-value check and appropriate diagnostics.

Configuration selects installed calculations; a YAML file alone does not define a new formula.
An indicators-only evaluation can reuse retained model responses when they contain all required
inputs. A measure requiring an unsaved model variable needs a new simulation. The
[indicator module reference](../../blueearth_cst/indicators/README.md) describes the authoring interface;
the [results chapter](results-and-interpretation.md) explains how indicators inform the assessment.
