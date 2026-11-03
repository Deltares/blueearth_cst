# Default indicators

The toolbox's registry and calculation scripts define the default indicators and formulas.
See the [advanced authoring contract](../../README.md#add-a-calculation) for implementation details.

Use these conventions when reading results from the default set:

- **Summary statistics:** annual means, maxima, minima, p95, basin totals, rolling-seven-day extrema,
  baseflow ratio and reference month means retain native pandas missingness and partial-block weighting.
- **Water years:** the configured water-year anchor defines the blocks.
  January–December applies only to a January water year.
- **Rolling windows:** seven-day windows require seven nonmissing days.
  They cross water-year boundaries, but never member boundaries.
- **Reference months:** these use pooled monthly sums and explicit reference membership/location.
- **Warm-up:** no new warm-up is removed.
- **Temperature:** the token is `temp` and the units are `degC`.
  `temp_annual_mean` averages daily air temperature within water-year blocks, then averages the block
  means with equal weight.
  It never sums daily values.
  The default preset selects it only when temperature is configured or retained for that simulation.
- **Member groups:** run indicators support equal-weight member means with NaN propagation.
  Return levels pool blocks instead.
- **Return levels:** these use member-local annual maxima or rolling-seven-day annual minima.
  Only finite block values are used.
  The sample is screened using max(ceil(T),10).
  The estimator's source must pass qualification verification.
  Only probabilities allowed by the supplied domain are accepted.
  Fit-block counts differ from retained daily-input counts.

## Advanced implementation details

The [module authoring contract](../../README.md#add-a-calculation) describes registration, formulas,
callable-keyed parameters, fixed assumptions and independent checks.
The following details apply specifically to the default set:

- `registry.py` retains the production default indicator names and declares fixed `ASSUMPTIONS`.
- Units, reference behavior and accepted parameters appear next to each registration and formula.
- `rolling_annual_min` describes the fixed seven-day calculation.
  Annual maximum fits have no unused rolling-window parameter.
