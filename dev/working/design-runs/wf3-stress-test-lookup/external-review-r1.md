## Verdict
verdict: revise
doc_version: design-v2.md

## Findings
### ext1-1  [blocking]
- section: 5.5 The derived caption — algorithm
- finding: The caption algorithm ignores an explicitly declared month subset when all months vary. D16 permits `M` to be a proper subset of the varying set, but §5.5 selects case 1 whenever `H` is empty and emits `mean change over the year`. Thus a uniform experiment declared with `months: [1, 2, 3]` is labelled annual even though its axis—and eventually its projection overlay—is collapsed over JFM.
- rationale: The resulting figure makes a false statement about the plotted quantity and violates C1. The error is especially consequential for the projection overlay because its JFM collapse can differ materially from its annual collapse even when the stress-test member values do not.
- suggested_fix: Derive the leading phrase from `M` in every non-degenerate case (`mean change over <label(M)>`), independently of the global varying/held classification. Define how varying months outside `M` are described or omitted, and add a test for an all-month-varying design with explicit `M = JFM`.

### ext1-2  [blocking]
- section: 8. Migration plan
- finding: The design contradicts the settled ruling that the new axis-derivation library has no in-repo caller. D15, alternative 6.9, and R9 assert that `axis_values`, `axis_caption`, and `join_axes` execute on no repository path, while migration step 6 requires the repository notebook to call `surface_axes.read_lookup`, `read_indicators`, `join_axes`, and `axis_caption`.
- rationale: The implementation cannot satisfy both instructions. Following step 6 violates the owner-approved boundary and invalidates R9’s risk analysis; following the ruling leaves the notebook migration specified by R6 incomplete and potentially broken after removal of the old columns.
- suggested_fix: Align the notebook migration with the settled no-caller ruling, such as by making it a contract-based external-consumer example that does not import the library. Otherwise return the proposed exception to the owner gate before continuing.

### ext1-3  [major]
- section: 5.3 The consumer side — what derives an axis
- finding: The degenerate-axis contract is not implementable unambiguously. D27 admits degenerate axes whose months are held at several different offsets, but says to “return the constant for those months” and bypasses step 3, where the weighted-collapse formula is defined. In addition, `axis_values` returns only a `pd.Series` and `join_axes` only two data frames, although D19 requires the consumer to receive `degenerate = True`; D28 also requires normalization using `ST_NUM`, which none of the normative signatures accepts.
- rationale: Independent implementations can legitimately choose different scalar values for a multi-offset degenerate axis, and the specified Python caller has no defined channel for the metadata needed to render that value as an annotation rather than a plot dimension. Key-width inference is likewise left implicit, weakening the partition check intended to prevent silent misjoins.
- suggested_fix: Define the degenerate scalar explicitly by applying the same flat-vector/weighted-mean formula over `M`, noting that the result is constant across members rather than necessarily equal across months. Replace the Series-only API with an explicit result object carrying values, caption, `degenerate`, and key-width context, or add equivalent explicit parameters and return values.

### ext1-4  [major]
- section: 7. Consequences and risks
- finding: Consequence 2 conflates the bound on the reconstructed precipitation multiplier with a bound on hydrological indicator values. D25 can bound the forcing-parameter difference to one `float64` ulp, but it cannot establish that indicator outputs move by at most that amount or remain within the baseline comparator’s tolerance.
- rationale: Weather-generation transformations, thresholds, quantile mapping, and hydrological simulation can amplify or discontinuously respond to a tiny parameter change. V20 tests only reconstruction, while V4 uses shipped levels that reconstruct exactly, so no validation falsifies the stated output claim for a non-exact grid.
- suggested_fix: State the one-ulp guarantee only for the reconstructed multiplier. Limit indicator-equality claims to exactly reconstructing shipped configurations, or add an end-to-end non-round-grid experiment with an empirically justified output tolerance.

### ext1-5  [major]
- section: 9. Validation plan
- finding: V17 does not test the failure behaviour it claims to validate. Its falsifier is the R script proceeding with a missing, partial, duplicate, or unordered member slice, but its assigned check is only a WF3 run on the valid rapid configuration.
- rationale: The guard can be absent or incorrectly expressed while the proposed gate remains green. A malformed or mismatched lookup can then reach R vector recycling and produce silently wrong climate perturbations—the exact cross-language failure D29 is intended to prevent.
- suggested_fix: Add negative executions of `impose_climate_change.R` using lookup fixtures with a missing month, duplicate month, wrong member token, and unordered months, asserting a nonzero exit and the member-specific diagnostic.

### ext1-6  [minor]
- section: 9. Validation plan
- finding: V15 lists "`validate_wg2` green on a `12 × ST_NUM` lookup" as a falsifier even though that is the expected valid case.
- rationale: Read literally, the gate treats correct validator acceptance as failure, making its pass criterion internally inconsistent.
- suggested_fix: Change the first clause to "`validate_wg2` not green on a valid `12 × ST_NUM` lookup"; retain green results on the malformed variants as falsifiers.