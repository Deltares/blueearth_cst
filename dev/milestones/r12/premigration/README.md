# Pre-migration artifacts — the WF3 stress-test lookup

Recorded at **step 0** of `dev/milestones/r12/stress-test-lookup-task-brief.md`
(board item `t2608152230`), from `test_case/snake_config_baseline.yml` in the
primary checkout, on 2026-08-16 at `ed436c7`.

**These exist because the migration destroys both of them**, and two entries of
the design's §9 validation plan compare against them *after* they are gone.
Neither is recoverable once its step lands.

| path | what it is | which check needs it |
|---|---|---|
| `member_grid/st_<m>.csv` | the six per-member parameter files as rule 3.09 wrote them, in **multiplier** units, header `month,temp_mean,precip_mean,precip_variance` | **V16** — the R side must reconstruct these factors from the lookup's percent columns to within one `float64` ulp. Step 5 deletes `<exp>/climate/weathergenr/_work/` outright |
| `stress_test_design.csv` | the derived design table the lookup absorbs, header `st_id,temp_change,precip_change,precip_variance_change` | context for V16/V4; it is the artifact whose annual collapse the migration abolishes |
| `indicator_ref/<hash>.csv` | the step-0 baseline reference for `q_indicators.csv`, at the **seven-column** shape `metric,location,st_id,rlz_id,temp_change,precip_change,value` | **V4** — its procedure drops `temp_change`/`precip_change` from *this stored copy* and compares the remainder against the post-migration table. Step 7's re-record overwrites `dev/baseline/indicator_ref/` in place |

The whole member set is kept rather than one file: V16 says "for one member",
but a single member gives no way to check a second if the first happens to sit
in D25's exactly-invertible set — and `snake_config_baseline.yml`'s levels all
do, which is exactly why V4 alone cannot observe the conversion residual (§9,
"V4 procedure").

## What the step-0 run established

A full WF1 + WF2 + WF3 re-run, with `experiments/experiment/` deleted and
rebuilt from scratch, reproduced **every numeric baseline target**: the wflow
discharge series, `q_indicators.csv`, and both CMIP6 change-factor tables. Only
the three flat config copies moved, and they moved together to one sha
consistent with the current seed — which is `t2608131718`, closed by this
record rather than by a fix.

So the "before" is attributable: it is what the code at `ed436c7` produces, not
an inherited tree.
