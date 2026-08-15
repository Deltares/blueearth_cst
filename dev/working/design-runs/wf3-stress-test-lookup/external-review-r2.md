## Verdict
verdict: revise
doc_version: design-v3.md

## Findings
### ext2-1 [blocking]
- section: §5.3 The consumer side — what derives an axis (D28); §5.8 HM-7 replacement text
- finding: The report-time partition check is only one-directional. `join_axes` requires indicator ids absent from the lookup to equal the baseline token and requires a non-empty surface, but never requires every lookup member to appear in the indicator table. `validate_hm7` specifies bidirectional completeness, but it is test-time-only and therefore does not close this runtime gap.
- rationale: If one or more surface members are missing from a stale or partial indicator table, every remaining nonzero id still belongs to the lookup, the absent-id set is still exactly the baseline, and the surface is non-empty. `join_axes` therefore returns an incomplete response surface silently, producing missing grid cells or a biased surface rather than the mismatch error D28 was introduced to provide.
- suggested_fix: Make `join_axes` assert equality between the lookup member-id set and the nonbaseline indicator member-id set before joining. Mirror that requirement in HM-7’s report-time join semantics and add a V18 case where one valid lookup member is missing from the indicators.

### ext2-2 [blocking]
- section: §5.2 The surface declaration — config schema and tiers; §5.3 The consumer side — what derives an axis (D14/D33)
- finding: The schema permits `x` and `y` to declare the same `variable`, but the result representation cannot express that configuration. Each axis independently accepts `temp | precip`; no distinctness rule exists, while `SurfaceJoin.axes` is keyed by variable and derived columns are named through `AXIS_COLUMN[variable]`.
- rationale: A declaration such as JFM temperature on `x` and JJA temperature on `y` passes the specified schema, but one `AxisResult` overwrites the other in the dictionary and both target the same `temp_change` column. The implementation must either discard an axis or return an object that violates its declared API, so an admitted configuration cannot be implemented correctly.
- suggested_fix: Require at parse time that `{x.variable, y.variable} == {"temp", "precip"}`—allowing orientation reversal but refusing duplicate variables. State this in the surface schema and add a negative parser test.

### ext2-3 [major]
- section: §5.1 The lookup table — D25; §5.7 WG-2 replacement text
- finding: The normative “at most one `float64` ulp of the level” reconstruction bound is unqualified, although its evidence covers only multipliers in `[0.5, 1.6]` and the design specifies no matching admissible range. The bound is false over positive values otherwise permitted by the document: for the float32-shortest level `0.013596006`, the specified conversion writes `-98.6403994` and reconstructs `0.013596005999999883`, a difference of 68 float64 ulps.
- rationale: WG-2 makes this bound a pinned cross-language contract, while V16 and V20 use it as an acceptance threshold. A low but positive multiplier can therefore follow the specified formulas exactly and still fail the contract and migration gate; the accepted resolution of the prior precision finding is not valid over its declared domain.
- suggested_fix: Either impose and parse-time validate a multiplier domain for which the one-ulp bound is proved, or replace it with a domain-qualified numerical-error bound. Extend V20 across the full admitted domain, including values near its lower boundary.