## Verdict
verdict: revise
doc_version: design-v3.md

## Findings
### ext2-01  [blocking]
- section: 5.3 The GCM series store — product, identity, caching
- finding: The cache incorrectly assumes that the region configuration fully determines `store_region.geojson`. The polygon also depends on the delineation catalog, its underlying data, and producer code. Yet the region input is marked `ancient()` and the series digest contains only configuration parameters, not the polygon’s content or its producing inputs.
- rationale: After a relevant catalog or delineation change rewrites the polygon, existing series remain eligible for reuse and stage B recomputes the same expected digest. It can therefore accept basin averages calculated for the old polygon, producing wrong change factors; recording the old bounds only makes the defect auditable after the fact.
- suggested_fix: Use `store_region.geojson` as an ordinary input, or introduce a content fingerprint that participates in both Snakemake invalidation and `cst_series_digest`. Stage B must also verify the current polygon fingerprint against every series.

### ext2-02  [blocking]
- section: 5.3 The GCM series store — Spatial reduction
- finding: The proposed geometry check does not make cosine-latitude weighting valid. Strictly monotonic 1-D coordinates can still be irregularly spaced or Gaussian; their cells require latitude and longitude widths in addition to `cos(latitude)`. Such grids currently pass the check even though OQ-10 explicitly acknowledges non-uniform spacing.
- rationale: Accepted non-uniform grids receive incorrect spatial weights and hence wrong basin means and change factors. This means the resolution of round-one finding `ext1-08` is incomplete, with exposure increased by the expanded generated catalog.
- suggested_fix: Either reject grids whose latitude and longitude spacing is not sufficiently uniform, or derive approximate cell edges and weight by spherical cell area. Retaining bounds and using bounds-derived areas is preferable. Add a non-uniform rectilinear test that either verifies correct areas or verifies fail-fast refusal.

### ext2-03  [blocking]
- section: 5.8 The optional gridded branch
- finding: The gridded change-field contract is under-specified. Cellwise changes require the historical and scenario grids to be compatible, but the design defines neither coordinate/CRS equality requirements nor regridding behavior. It also provides no schema for whether `grids/change/*.nc` contains annual changes, monthly changes, which statistics, or how dry-reference statuses and absolute fallbacks are represented.
- rationale: Historical and scenario publications may use different grid labels or coordinates. Implicit xarray alignment can produce empty, sparse, or mismatched fields, while alternative implementations could emit incompatible products. Ruling R2’s declared gridded output therefore cannot be implemented reliably as specified.
- suggested_fix: Define the complete gridded-change schema and require exact CRS and coordinate compatibility before cellwise arithmetic, failing fast when it is absent. If differing grids must be supported, specify one existing-dependency regridding method. Add shifted-grid, mismatched-CRS, monthly/annual, and dry-cell tests.

### ext2-04  [major]
- section: 5.3 The GCM series store — Cache key; D8 — time-axis uniqueness
- finding: The value called the “resolved URI” is not a physical source identity: substituting `{member}` still leaves `{variable}/*/*`, including wildcard grid label and publication version. The digest also excludes read-relevant metadata such as `metadata.crs`.
- rationale: A newly published version under the same glob can be read by a fresh project while an existing cache silently retains the prior publication under an unchanged digest. Provenance cannot identify which physical zarr stores supplied the values, and metadata corrections can alter interpretation without invalidation. D8 detects duplicate timestamps only when a source is reread; it does not repair cache identity.
- suggested_fix: Close OQ-14 before implementation. Have the generator pin and record the exact physical zarr path selected for each variable, and include those paths plus all read-affecting metadata in the digest and provenance.

### ext2-05  [major]
- section: 5.4 Region and reference window — The reference window length
- finding: The asserted 30-year 1985–2014 reference conflicts with the complete-hydrological-year policy. Whenever `start_month_hyd_year` is not January, this calendar window contains only 29 complete hydrological years; the partial years at both ends are dropped.
- rationale: Annual and potentially monthly statistics would use fewer years and a different effective period than the owner-approved “30 years, 1985–2014,” while the acceptance test checks only that warnings do not fire. Reported sample length and scientific interpretation can therefore be misleading.
- suggested_fix: Define whether the ruling denotes 30 calendar years or 30 complete hydrological years. Then either use calendar years for WF2 or construct exactly 30 hydrological years and report their actual date bounds. Add a non-January acceptance test asserting `n_years`, effective dates, and dropped months.

### ext2-06  [major]
- section: 5.6 Change factors — Dry-month / near-zero denominator rule
- finding: The dry-reference policy remains scientifically incomplete because `relative_change.min_reference` is outcome-determining but its default is still OQ-9. The contract likewise does not settle whether thresholds and `relative_change.max_flagged_months` are required or defaulted.
- rationale: Implementers can produce different `value`, `status`, and report-warning outputs from identical inputs by selecting different undocumented thresholds. Shipped configurations and boundary tests cannot be finalized, so the claimed resolution of `risk-05`/`ext1-05` is not yet complete.
- suggested_fix: Before implementation, either choose and justify explicit per-variable defaults or make both thresholds required and populate every shipped configuration. Add tests immediately below, at, and above each threshold.

### ext2-07  [major]
- section: 5.5 Variable specification — declaring the quantity, not the aggregator
- finding: The configurable variable contract is broader than the catalog’s availability contract. Resolution certifies only `pr` and `tas`, yet a requested `kin` or `press_msl` combination is marked resolved and converted into a job even when the corresponding store is predictably absent.
- rationale: A large run can spend hours completing other network jobs before halting on a missing configured variable. `composition.csv` will classify the combination as resolved even though the generated snapshot lacked the requested input, undermining the design’s central separation between “not published” and “failed to read.”
- suggested_fix: For v2.0, either reject variables other than `precip` and `temp` at DAG build, or make the generator publish per-variable member availability and resolve against all requested variables. Do not represent catalog-known absence as a runtime read failure.

### ext2-08  [minor]
- section: 5.7 Source resolution and failure semantics — D4
- finding: The statement that the resolved combination set “is now written down in `composition.csv` before any job runs” is impossible under the specified DAG: `composition.csv` is a stage-B output and stage B cannot run until every required reducer succeeds.
- rationale: A failed run has the DAG-build stderr summary and logs, but no composition artifact. This contradicts the round-one `ext1-02` disposition, which correctly limited provenance to successful runs.
- suggested_fix: State explicitly that `composition.csv` describes completed runs. If a durable pre-execution resolution manifest is required, specify it as a separate earlier artifact rather than assigning that behavior to stage B.

### ext2-09  [minor]
- section: 9. Validation plan — No aggregation
- finding: The assertion that no row may equal the mean of other rows is not a valid no-aggregation invariant; a legitimate member value can coincide numerically with such a mean.
- rationale: The test can reject a correct implementation by coincidence and can still miss aggregation if synthetic values are poorly chosen.
- suggested_fix: Assert tuple cardinality, unique keys, direct equality to independently computed per-series results, and absence of cross-combination reduction operations. If using sentinel values, construct them so no aggregate can equal an original value.