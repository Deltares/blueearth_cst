## Verdict
verdict: revise
doc_version: design-v1.md

## Findings
### ext1-01  [blocking]
- section: 5.3 Region and baseline — solved structurally, not by validation; 5.5 Change factors — annual and monthly
- finding: The reference-period contract cannot be realized as written. `shared.historical_window` is 2000–2020, while CMIP6 historical runs end in 2014; the design does not specify joining each historical series to its scenario series for 2015–2020, including gap, overlap, and calendar handling. It also independently proposes a 30-year reference default, although the shared baseline currently spans 21 years.
- rationale: Implementations will either truncate the GCM reference to 2000–2014, fail on missing years, or silently compare unequal periods. Any of these violates G3 and changes the change factors. Making the shared window 30 years instead would also change the shared climate store used by WF1 and WF3, a cross-workflow consequence absent from the migration plan.
- suggested_fix: Define the reference as exactly `shared.historical_window`; specify historical-to-scenario concatenation per `(model, member, scenario)` with coverage, duplicate, gap, and calendar checks; and separate the future-horizon length decision from the reference-window contract. If concatenation is rejected, require the shared reference to end by 2014 and revise G3.

### ext1-02  [blocking]
- section: 5.6 Report stage; 5.7 Source resolution at DAG-build time
- finding: Runtime source-failure handling is incompatible with the proposed DAG. Every resolved source is a required Stage-A netCDF input to Stage B, but a remote read failure is supposed to produce no empty file and instead be recorded later in `provenance.json`.
- rationale: In Snakemake, the failed reducer job or its missing declared output stops the DAG before Stage B or the report can run. Consequently, failed-source provenance, configurable minimum-source enforcement, and continuation with the surviving ensemble cannot work as specified.
- suggested_fix: Choose and document either fail-fast semantics, or a failure-tolerant artifact contract in which every source job emits a required status artifact and successful data are discovered through a checkpoint or manifest. Move provenance/minimum-source validation ahead of Stage B in the tolerant design.

### ext1-03  [blocking]
- section: 5.2 The unified series store — the central idea; 5.4 Variable specification — replacing name-based dispatch; 5.5 Change factors — annual and monthly
- finding: A variable-level `aggregate: sum` cannot uniformly reduce the daily observed inputs and monthly-mean CMIP6 inputs into a comparable precipitation series. Summing daily precipitation produces a monthly accumulation, whereas summing an Amon series with one value per month merely preserves a monthly mean rate; the proposed common `units: mm/day` would therefore label unlike quantities as equivalent.
- rationale: Observed-versus-GCM diagnostics would compare incompatible values, and annual or monthly precipitation products would depend on source frequency rather than climate. The later reference to month-length weighting does not define the conversions needed to repair the Stage-A store.
- suggested_fix: Specify a canonical monthly quantity for each variable—such as mean rate in mm/day or accumulated depth in mm/month—and define source-specific conversion using units, sampling interval, temporal bounds, and calendar before spatial reduction. Reject inputs whose temporal semantics cannot be established and test equivalent daily and monthly synthetic inputs.

### ext1-04  [major]
- section: 5.2 The unified series store — the central idea; 7. Consequences and risks
- finding: G5 conflicts with the reducer cache key. The reducer digest includes `window`, but the document does not distinguish acquisition coverage from the analysis horizons that users may change.
- rationale: If `window` follows `future_horizons`, changing a horizon invalidates the persistent series and repeats the expensive network reads, directly falsifying consequence 2. If it instead means full source coverage, the required coverage and associated cost are undefined.
- suggested_fix: Give Stage A a stable acquisition window independent of reference and future analysis windows—normally the complete required CMIP source span—and make only Stage B depend on `future_horizons`. Record acquisition coverage in each series and fail when a requested analysis window is not fully covered.

### ext1-05  [major]
- section: 5.5 Change factors — annual and monthly
- finding: The change-factor method lacks normative formulas and edge-case policies for relative changes, incomplete hydrological years, missing months, and non-Gregorian calendars.
- rationale: Monthly precipitation reference values can be zero or near zero, producing infinite or unstable percentage changes. Incomplete first or last hydrological years and truncated series can also enter statistics with fewer months, while unspecified calendar weighting makes results differ by model for procedural rather than climatic reasons.
- suggested_fix: State equations and units for every supported variable/statistic; define denominator thresholds and NA/status behavior; require complete hydrological years and minimum coverage; and specify calendar-aware interval weighting. Add tests for dry months, missing months, partial years, leap years, and 360-day calendars.

### ext1-06  [major]
- section: 5.8 Extension slots (named, not built); 10. Open questions
- finding: The claim that every extension slot is “a read, not a pipeline” is false for the proposed store. S2 requires extraction and fan-out for multiple observed products although Stage 0 creates only the single configured `shared.clim_historical` store; S4 requires a new daily acquisition and storage branch; and credible long-term S1 trends may require coverage beyond the project baseline window.
- rationale: Implementing these advertised extensions would change the producer graph, cache schema, configuration, and provenance contracts, so G9 is not delivered by the selected architecture. The widened “general climate analysis” framing would therefore create downstream redesign rather than the promised stable extension surface.
- suggested_fix: Either narrow v2’s claim to monthly basin-series projection analysis, documenting S1/S2/S4 as future architecture changes, or generalize the source registry and series identity now to include observed-source and temporal-resolution axes with independently configured acquisition windows.

### ext1-07  [major]
- section: 5.5 Change factors — annual and monthly
- finding: The ensemble contract does not define the sampling unit after `member` becomes a wildcard. A threshold of 10 and envelopes “across models” are ambiguous when models contribute different numbers of members; institution counts alone do not prevent pseudoreplication.
- rationale: Adding members from one model could give that model disproportionate influence and could trigger percentile envelopes without adding independent model diversity. Reported uncertainty would then change because of configuration multiplicity rather than a broader ensemble.
- suggested_fix: Resolve OQ-6 before specifying ensemble summaries. Define thresholds using unique models, show members hierarchically, and either average members within each model before equal-model summaries or document another explicit weighting rule. Until then, emit individual model/member traces without an aggregate envelope.

### ext1-08  [major]
- section: 5.2 The unified series store — the central idea
- finding: A cosine-latitude weight is not generally a cell-area weight for the catalog’s heterogeneous CMIP grids. It is valid only under restrictive rectilinear, regularly spaced coordinate assumptions that the design neither checks nor records.
- rationale: On Gaussian, irregular, or curvilinear grids, model-to-model differences can partly reflect grid geometry. Because the catalog currently drops coordinate bounds, the proposed reducer may be unable to establish correct areas while still claiming area-weighted results.
- suggested_fix: Define supported grid geometries and compute areas from retained bounds or derived cell edges, including longitude wrapping and missing-cell treatment. Validate assumptions and fail or explicitly label an approximation when exact areas cannot be established.

### ext1-09  [major]
- section: 5.1 Architecture — three stages, fan-out only where it pays; 8. Migration + commit plan; 10. Open questions
- finding: OQ-8 must be resolved before this architecture can be implemented: Stage A discards spatial dimensions, so Stage B cannot reproduce the existing `save_grids` products.
- rationale: Preserving `save_grids` requires an additional gridded artifact path and declared optional rules; retiring it is a breaking behavior change requiring migration and acceptance coverage. Leaving the choice open makes the rule graph, output contract, job count, and step-4 value-neutrality claim indeterminate.
- suggested_fix: Either explicitly retire `save_grids` with a migration note and characterized loss of functionality, or specify a separate optional gridded branch with declared outputs, cache behavior, and validation.

### ext1-10  [minor]
- section: 5.1 Architecture — three stages, fan-out only where it pays; 7. Consequences and risks
- finding: The seed job-count prediction is arithmetically inconsistent. Three models require three historical GCM series, six future model-scenario series, and one observed series: ten reducer jobs, not nine. With store, derive, report, config-copy, and benchmark-gather jobs, the stated architecture totals 15 jobs rather than 13.
- rationale: The falsifiable 13-job consequence and its validation gate will fail even if the implementation matches the architecture.
- suggested_fix: Derive expected counts from the resolved source manifest in tests instead of hard-coding 13, and update the design’s illustrative count.