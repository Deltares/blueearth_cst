# Flow regime

Select this installed set with `flow_regime` in the
[WF4 request](../../../../docs/technical-manual/indicator-requests.md).

The optional production flow-regime additions are mean annual 30-day maximum, 30-day minimum and 90-day
minimum discharge, plus mean annual rise/fall magnitudes. They use configured water-year blocks and native
NaN behavior. Trailing windows cross water-year boundaries, never members; differences belong to the later
day, zeros are excluded and blocks without a qualifying change remain undefined. Annual means skip undefined
blocks. Member-group means use equal weights and propagate undefined members.

The set also exposes the matched prototype aliases `q_annual_mean`, `q_baseflow_index`, `flood_peak_10yr`,
`high_flow_p95`, `low_flow_7day`, and `recharge_depth`. These reuse the default set's formula callables.
`whole_period_mean` averages all retained days directly, so leap years and partial years receive their
actual daily-sample weights. It is separate from the equal-block-weight annual mean.

`registry.py` defines names, units and fixed assumptions. `calculations.py` implements formulas, accepted
parameters and conditional rules; `checks.py` provides independent expected values. Follow
[the custom-calculation instructions](../../README.md) for extensions. This set does not alter the default
suite.
