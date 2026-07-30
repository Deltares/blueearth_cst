# Falsifier — step 5c, drop the stage-A 2-decimal rounding

```
Written: 2026-07-30, BEFORE any 5c code exists
Design:  wf2-climate-analysis-v2-design.md §5.3 "Precision", §8 row 5c
Ref:     test_case/ref_wf2_pre_5c  (post-5b; CLEAN against itself, 126 files)
```

## What 5c claims

Stage A stops rounding to 2 decimals. Today `get_stats_climate_proj.py:130`
applies `.round(decimals=2)` to the spatial mean, which on a mm/day precipitation
rate is a **0.005 mm/day floor — about 0.15 mm/month**.

Scope is **stage A only**, per the design. Two rounding sites exist there:

* `:130` — the scalar series. In scope.
* `:137` — the gridded monthly means, `save_grids`-only. In scope, but cold on
  shipped configs (`save_grids: false`), so no diff evidence will come from it.

Out of scope, deliberately: `plot_proj_timeseries.py:239-240` rounds `precip` and
`temp` to 2 dp when building `gcm_timeseries.nc`. That is a *display* product, the
design's precision clause names stage A, and changing it in the same commit would
make the diff unattributable. Recorded here so its survival is a decision rather
than an oversight.

## H1 — the floor must actually be gone

**Falsified if** every value in a re-derived series is still an exact multiple of
0.01. That is the signature of the rounding surviving somewhere upstream of the
write — and it is the failure mode where the commit looks applied but is not.

## H2 — the move must be bounded by half the rounding step

This is 5c's strongest property, and it is exact rather than statistical.
`|unrounded − round(x, 2)| ≤ 0.005` for every value, by definition.

**Falsified if** any series value moves by **more than 0.005** from the
`ref_wf2_pre_5c` tree. A larger move means something other than the rounding
changed, and the step is no longer attributable to its stated cause.

This is the check that distinguishes "rounding removed" from "rounding removed
plus something else came along" — the failure that 5a's dtype upcast and 5b's
`.rename` would both have produced.

## H3 — dtype must NOT change

`weighted_spatial_mean` casts back to the input dtype (float32); dropping the
round must not disturb that. Whether the series *should* be float64 is a real
question, and the 5a comment deferred it here — **this falsifier answers it: not
in this step.** Promoting dtype and dropping rounding in one commit produces a
diff attributable to neither.

**Falsified if** any series variable changes dtype.

## H4 — the change must reach the change factors

**Falsified if** `semantic_tree_diff --ref test_case/ref_wf2_pre_5c` shows no
value change in the summary artifacts. A 0.005 mm/day floor on monthly values
propagates to annual aggregates and thence to the change factors; if nothing
moves there, the series never fed the aggregation.

## H5 — the cache must actually invalidate, and be seen to

The 5b lesson: a change outside the functions `kernel_hash` enumerates is
invisible, Snakemake schedules the job, the job revalidates its own digest, finds
it unmoved and returns — `utime`-ing the file without rewriting it. Silent
success.

`.round(decimals=2)` at `:130` **is** inside `get_stats_clim_projections`, which
is in `REDUCER_KERNEL`, so `REDUCER_HASH` should move on its own and no
`SCHEMA_VERSION` bump should be needed.

**Falsified if** after the run a series still satisfies H1's "all multiples of
0.01" — i.e. the job ran and skipped. **Verify positively**, do not assume:
compare `cst_reducer_module_hash` before and after; it must differ.

## H6 — no network

Stage A re-derives from local raw slices. **Falsified if** the dry-run schedules
any `fetch_gcm_raw` job.

## H7 — prediction for the plot product

`gcm_timeseries.nc` is built from the series and then rounded to 2 dp by the plot
code, which 5c does not touch. Since the underlying values move by ≤ 0.005, most
rounded values will be **unchanged**, and a minority will flip by exactly 0.01
where the unrounded value crossed a midpoint.

**Falsified if** `gcm_timeseries.nc` moves by more than 0.01 anywhere — that would
mean the plot path is doing something other than re-rounding a slightly different
input.

Stated as a prediction because it is the one place where "almost no change" is the
*correct* result and could otherwise be misread as the step failing to apply.

## Order of work

1. Drop `.round(decimals=2)` at both stage-A sites.
2. Assert H5 positively (`cst_reducer_module_hash` moves) before trusting any diff.
3. Run; check H1, H2, H3 on the series directly, not via the tree diff.
4. Characterize H4/H7 in the diff.
5. Re-record the baseline only after that, and snapshot `ref_wf2_pre_5d`.

---

## Outcome — 2026-07-30, all seven discharged

| | Result |
|---|---|
| H1 | floor gone: `2.4213858`, no longer multiples of 0.01 |
| H2 | max delta **0.00500107** across all 9 series |
| H3 | dtype `float32` preserved on both variables |
| H4 | 12 failed: 9 series + 3 summary; **zero `raw/`** |
| H5 | reducer hash `104c9613…` → `caa6dfff…` — moved, no schema bump needed |
| H6 | zero `fetch_gcm_raw` jobs |
| H7 | `gcm_timeseries.nc` **0 of 16308** values changed |
| `check_baseline` | FAILED on 3 targets, then re-recorded OK 15/15 |

**H2's bound needed one correction, and it is worth stating rather than
quietly widening.** `|unrounded − round(x, 2)| ≤ 0.005` is exact in real
arithmetic, and the measured worst case exceeded it by 1.07e-6 — on a `temp`
value of 25.695, where one float32 ULP is 1.907e-6. The excess is smaller than the
representation step, so the bound holds; what was wrong was the falsifier stating
an exact-real bound for a float32 product. A stricter reading would have condemned
a correct step, which is the same trap F6 set in 5a.

**H7 was sharper than predicted.** The prediction allowed a minority of values to
flip by 0.01 as the plot re-rounds a shifted input. None did — because
`round(x, 2)` for `|x − round(x, 2)| ≤ 0.005` returns the *same* 2-dp value except
at exact midpoints, which no value hit. "Almost no change" was correct; "no change
at all" is correcter, and would have been easy to misread as the step failing to
apply had the prediction not been written down first.

**H5 confirmed positively, not assumed.** The 5b lesson was that a change outside
the hashed kernel is invisible and fails as silent success. `.round(decimals=2)`
sits inside `get_stats_clim_projections`, which is enumerated in `REDUCER_KERNEL`,
so `REDUCER_HASH` moved on its own — verified by comparing the recorded hash
before and after, not inferred from the code's location.
