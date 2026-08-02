# Falsifier — step 6a (three commits: tidy annual table, monthly table, provenance)

```
Written: 2026-07-31, BEFORE any 6a code
Design:  §5.9 (report-stage artifacts), §8 row 6a — "Additive", manifest-unclean
Ref:     test_case/ref_wf2_pre_5f_fixed  (post-5f + kernel_hash fix)
```

6a is the largest remaining step and splits into three independently gateable
commits:

| | Change | New computation? |
|---|---|---|
| i | `change_factors/annual.csv` + `.nc` in tidy long format | no — reshapes what stage B already produces |
| ii | `change_factors/monthly.csv` | **yes** — monthly change factors do not exist today |
| iii | `provenance.json` | no — assembles facts already computed |

## Two scope corrections, measured rather than assumed

**`n_models_in_summary` does not exist.** §8 says "tidy CSV schema (no
`n_models_in_summary`)"; grep finds it in the design only. design-v2 *proposed*
the column and it was never implemented, so there is nothing to remove. Recorded
because a commit claiming to remove it would be claiming work that did not happen.

**The current CSV is wide, not merely untidy.** One row per
`(stats, clim_project, model, scenario, horizon, member)` with `precip` and `temp`
as **columns**, plus a stray `spatial_ref` coordinate. The tidy schema is one row
per `(…, variable, statistic)`, so variables become rows.

## M1 — the tidy table must be long, one row per the full key

Design §5.9: one row per
`(dataset, institution, scenario, member, horizon, period, variable, statistic)`.

**Falsified if** `precip`/`temp` remain columns, or if any key field is missing.
On the seed that means **6 combinations × 1 horizon × 2 variables × 3 statistics
= 36 rows**, against today's 6.

## M2 — `spatial_ref` must not appear

A CRS coordinate is not a change factor. **Falsified if** it survives into the
tidy table — it is in today's CSV only because `to_dataframe()` dumps every
coordinate.

## M3 — the values must be IDENTICAL to the wide table

6a is additive: it reshapes, it does not recompute.

**Falsified if** any `(combination, horizon, variable, statistic)` value differs
from the corresponding cell of `annual_change_scalar_stats_summary.csv`. Checked
by joining the two, not by eyeballing row counts — a reshape that silently drops
or duplicates rows would still produce a plausible-looking file.

## M4 — the provenance columns must carry the EFFECTIVE window, not the nominal

§5.9 requires both, plus `n_years` and `n_years_dropped`.

**Falsified if** `n_years` reports the nominal count. After the 2026-07-30 fix the
seed's effective count is **21**, and the nominal window `[1990, 2010]` would also
read as 21 — a coincidence that makes this untestable on the seed alone, so it is
covered by a unit test on a window where they differ.

## M5 — the monthly table must be a real computation, not annual repeated

**Falsified if** the 12 monthly rows for one `(combination, horizon, variable,
statistic)` are all equal, or equal the annual value. Monthly change factors are
computed per calendar month over the window, which is a different aggregation.

**Also falsified if** the monthly table's relative changes are computed against a
near-zero denominator without a status — that is 6b's dry-month rule, and 6a must
leave the door open for it rather than silently emitting infinities. Until 6b,
`status` is `ok` for every row and the column exists.

## M6 — `provenance.json` must be reconstructible, not decorative

**Falsified if** it omits any of: the resolved sources with their **verified
physical store paths** (D12), the region fingerprint (D9), nominal *and* effective
windows with per-end dropped months and the clip flag (A1),
`shared.historical_window` with the alignment result, the reducer module hash, the
variable spec, and the catalog/index `crawled_on`.

Each of those already exists somewhere in the pipeline. The falsifier is that
`provenance.json` **assembles** them rather than recomputing — a second derivation
is a second thing to drift.

## M7 — additive means additive

**Falsified if** any existing artifact's **values** move. The manifest is unclean
only because new files are added and the summary CSVs may be re-pinned, not
because numbers change.

**Falsified (silent-success check) if** the dry-run schedules no job at all — the
fourth cache defect of this milestone was exactly that, so the trigger is verified
positively before the diff is trusted.

## Order of work

1. 6a-i: tidy annual table + `.nc`; M1–M4; gate.
2. 6a-ii: monthly table; M5; gate.
3. 6a-iii: `provenance.json`; M6; gate.
4. M7 checked at each; re-record once per commit, snapshot at the end.

---

## Outcome — 2026-07-31, all seven discharged across three commits

| | Result |
|---|---|
| M1 | 36 annual rows (6×1×2×3); variables are rows |
| M2 | `spatial_ref` dropped |
| M3 | all 36 rows **joined** against the wide table, 0 mismatched |
| M4 | caught a live drift — see below |
| M5 | 432 monthly rows; −1.76 % (Mar) to +29.34 % (Dec) against 10.98 % annual |
| M6 | 18 unit tests; every required fact present and **sourced, not recomputed** |
| M7 | 3 EXTRA files, no value moved, `check_baseline` OK 15/15 throughout |

### One quantity, three homes, three chances to disagree

M4 was written because the seed makes nominal and effective year counts coincide.
It caught the drift **twice more than expected**:

1. **6a-i** — the tidy table took `n_years` from the run-level reference record
   and reported **20** beside a `composition.csv` reporting **21**.
2. **6a-iii** — `provenance.json` reported `reference_window_effective =
   1990-2010` (the calendar span) where the other two say
   `1990-01-01 / 2010-12-01`.

Both have the same root: **`reference_window.n_years` is a calendar span
(`end − start`) while `hydrological_year_bounds` counts complete hydrological
years.** Two correct functions, two different quantities, one name.

The fix in both cases was to take the value the composition record already used
rather than compute a fresh one — and in `provenance.json` to move it to the
**source** level, because the effective window depends on the data each
combination has and a run-level field would have been a third definition.

Counting the calendar defect at 5b and the `n_years` mismatch here, **a value
recorded in two places has disagreed four times in this milestone.** The pattern
is not carelessness in any one commit; it is that every new artifact is a new
opportunity, and the only reliable defence has been to pass values rather than
recompute them.
