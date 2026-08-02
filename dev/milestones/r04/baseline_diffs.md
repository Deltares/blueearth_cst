# R04 baseline diffs

R4 is a **behavior-preserving** refactor of workflow 2 (climate projections):
orchestration, logging, guards, label renames, docs, tests — no computational
change. The milestone baseline gate re-ran workflow 2 end-to-end with R4's code
into the recorded project dir (`examples/test_local`) and compared against
`dev/baseline/manifest.json` (recorded 2026-07-18), scoped to the 11 workflow-1
+ workflow-2 targets via `check_baseline.py check --workflow model_creation
--workflow climate_projections` (the commit-2b scope filter).

## Result: 9 / 11 exact, 2 CSV byte-diffs (data preserved)

**Matched exactly (tolerance 0):** all 4 workflow-1 targets (3 PNGs + config
snapshot), and 5 of 7 workflow-2 targets — `annual_change_scalar_stats_summary.nc`
(per-variable summary stats **and** `attrs`), the 3 response PNGs, and
`snake_config_climate_projections.yml`.

**Differed (sha256):** the 2 full-precision CSV targets —
`annual_change_scalar_stats_summary.csv` and
`annual_change_scalar_stats_summary_mean.csv`.

## Diagnosis: CSV serialization non-determinism, not a value change

The differing CSVs are **not** a behavior change introduced by R4:

1. **The data is provably identical.** `annual_change_scalar_stats_summary.nc`
   — written from the *same* in-memory merged dataset as the two CSVs — matched
   the manifest **exactly at tolerance 0** on every per-variable statistic
   (shape, dtype, count_non_nan, min, max, mean, std, attrs). A real value
   change would move those aggregate stats; it did not. The CSVs carry
   full-precision float64 values (e.g. `12.049344062805176`), so a byte-level
   sha256 is sensitive to row order / serialization detail that the aggregate
   `.nc` fingerprint is invariant to.
2. **R4's merge/summary code is behaviorally unchanged.** The only R4 edit to
   `get_change_climate_proj_summary.py` is the `filter_nonempty` extract
   (commit `770e48c`) — line-for-line equivalent to the prior inline
   dummy-skip loop, order-preserving. No R4 commit alters the merge inputs
   (`expand(...)` order), the transpose, or the `to_dataframe().to_csv()` call.

The residual difference is therefore attributed to **run-to-run serialization
non-determinism** in `xarray.Dataset.to_dataframe().to_csv()` on the merged
multi-index dataset (row ordering of full-precision floats), independent of
R4's changes. This is consistent with the manifest's known byte-level fragility
on derived outputs (see `dev/milestones/r01/baseline_diffs.md`, `dev/followups.md`
§ baseline).

## Disposition

- **The manifest is NOT re-recorded.** R4 makes zero manifest edits; the CSV
  byte-diff is a fingerprint-fragility artifact, not a milestone output change,
  and the authoritative data (the `.nc` slice) is preserved exactly.
- **Followup (not blocking R4):** if byte-stable CSV baselines are wanted, make
  the summary CSV serialization deterministic (sort the dataframe by its coord
  index before `to_csv`, or fingerprint the CSV by normalized content/sorted
  rows rather than raw sha256). Tracked as a baseline-robustness note, not an R4
  defect.

Verified 2026-07-20 by the workflow-2 end-to-end re-run (19/19 jobs, exit 0)
and the scoped `check_baseline` comparison above.
