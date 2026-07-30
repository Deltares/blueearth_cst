# Falsifier — step 5d, default statistic set

```
Written: 2026-07-30, BEFORE any 5d code exists
Design:  wf2-climate-analysis-v2-design.md §5.6 "Statistics", §8 row 5d
Ref:     test_case/ref_wf2_pre_5d  (post-5c; CLEAN against itself, 126 files)
```

## What 5d claims

Today's eight statistics — `mean, std, var, median, q_90, q_75, q_10, q_25` — are
computed over a ~20-year window, which makes `q_90` "effectively the
second-highest of 20 values". v2.0 emits **`mean`, `median`, `std`** by default;
tail quantiles become **opt-in** and, when emitted, are **labelled with their
effective sample size**.

§8 classifies this as an **output-set change**, not a value change, and
**manifest-unclean** because the summary CSVs lose columns.

## J1 — the default set must be exactly three

**Falsified if** a run on a shipped config emits `var` or any `q_*`.

## J2 — columns must be LOST, not renamed

**Falsified if** the summary CSVs carry the same number of statistics as before.
Renaming or hiding is not removing; the design's stated reason is that a 20-year
window cannot support a tail quantile, and a label change would leave the
unsupportable number in the product.

## J3 — the RETAINED statistics must not move at all

This is the sharpest property in the step, and the one that distinguishes an
output-set change from a value change.

**Falsified if** `mean`, `median` or `std` differs — by any amount — from
`ref_wf2_pre_5d` for any (model, scenario, horizon, variable). 5d drops
statistics; it does not touch the arithmetic that produces the survivors.

If a retained value moves, something in the aggregation changed under cover of a
"set" change, and 5d is no longer attributable to its stated cause. Checked
directly on the summary `.nc`, not through the tree diff, because the tree diff
reports the file as FAIL either way.

## J4 — the quantiles must remain reachable, not deleted

Opt-in means the capability survives. **Falsified if** there is no supported way
to request `q_90` after this step, or if requesting it fails.

## J5 — an emitted quantile must carry its effective sample size

The design requires the label in **both** the CSV and the report.

**Falsified if** an opted-in `q_90` appears without the sample size that makes it
readable — the whole point being that "the second-highest of 20" should be
self-evidently that.

Note this is **not gate-visible on shipped configs**, because they will not opt
in. It must be covered by a unit test, not by the tree diff. Recorded so its
absence from the diff is not mistaken for absence of the feature.

## J6 — the config snapshot must NOT change

The opt-in key is **optional and unset** in shipped configs, so the config
snapshot — one of the 15 manifest targets, fingerprinted by sha256 — must be
byte-identical.

**Falsified if** the config target's sha256 moves. That would mean 5d altered a
config contract, which is **5e's** job (`variable spec`, `save_gridded` rename),
and batching the two would make neither attributable.

## J7 — stage B only

**Falsified if** the dry-run schedules any `fetch_gcm_raw` or `reduce_gcm_series`
job. The statistic set is applied in stage B; the series are untouched.

This also predicts which files may move: `summary/*` only. Not `raw/`, not
`series/`, not `timeseries/` — `gcm_timeseries.nc` plots the series, not the
statistics.

## Order of work

1. Change the default set; add the optional opt-in key without setting it in any
   shipped config.
2. Unit tests for J4 and J5 (no fixture).
3. Dry-run: assert J7.
4. Run; check J1, J2, J3, J6 directly on the artifacts.
5. Re-record only after the column diff is shown, and snapshot `ref_wf2_pre_5e`.

---

## Outcome — 2026-07-30, all seven discharged

| | Result |
|---|---|
| J1 | default set is exactly `mean, median, std` |
| J2 | 8 → 3; `var, q_10, q_25, q_75, q_90` **dropped**, not renamed |
| J3 | retained statistics **bit-identical**, max delta `0.000000000000` |
| J4/J5 | 12 unit tests; `q_90` opted in reads `q_90[n=20]` |
| J6 | config snapshot untouched — the 2 failing manifest targets are the summary `.csv` and `.nc` |
| J7 | 4 jobs total: stage B, plot, gather, all. No fetch, no reduce |
| `check_baseline` | FAILED on 2 targets, then re-recorded OK 15/15 |

**J3 is the property that made this step safe to make.** An "output-set change"
is only that if the survivors do not move, and the check was run directly on the
summary `.nc` rather than through the tree diff, which reports FAIL either way and
would have hidden a value drift inside a column removal.

**Two results were sharper than the falsifier required.**
`annual_change_scalar_stats_summary_mean.csv` did not change at all — it selects
`mean`, which J3 pins as unchanged — so only 2 of the 3 summary artifacts moved.
And J7's prediction that `timeseries/` would be untouched held: `gcm_timeseries.nc`
plots the series, not the statistics.

**J6 was worth stating separately.** It would have been easy to add the `stats:`
key to the shipped configs "for discoverability", which would have moved the
config snapshot and merged 5d's diff with 5e's config-contract changes. The key is
optional and unset; discoverability belongs in the template config, which is not a
manifest target.
