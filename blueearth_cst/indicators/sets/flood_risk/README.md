# Flood discharge hazards (opt-in)

The [WF4 request reference](../../../../docs/technical-manual/indicator-requests.md#standalone-definitions)
shows how to select the ten-year return level. Select only return periods supported by the available
evaluation evidence, as described below.

The five example return periods are 2, 5, 10, 15 and 20 years. These are annual maximum discharge hazard
levels, not exposure, damages or a fully qualified flood-risk assessment. They are never enabled by default.
The production estimator and its source qualification are unchanged. Member-local finite annual maxima are
pooled under the configured water-year anchor; partial edge blocks are retained. Every location requires at
least max(ceil(T),10) usable fit blocks. Retained daily-input counts are separate from fit-block counts.

The shipped GF15 evidence bounds probabilities to .5 and .9, so only the 2- and 10-year examples are eligible
under that evidence. The 5-, 15- and 20-year declarations refuse with an explicit qualification error.
Supplying different periods never expands the benchmark domain. A 25-year request on 20 usable blocks is
screened for insufficient sample before its probability is considered. No new benchmark qualification is
claimed. Failed fits retain structured failure details; unexpected estimator faults propagate.

`registry.py` defines indicators and fixed assumptions; `calculations.py` specifies accepted parameters;
`checks.py` provides independent inverse-quantile checks. See [the module README](../../README.md) for plain
Python calls, input semantics and extensions. WF4 supplies its verified qualification declaration explicitly.
