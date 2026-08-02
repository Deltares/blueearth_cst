# Falsifier — step 5b, calendar-aware month-length weighting

```
Written: 2026-07-30, BEFORE any 5b code exists
Gate:    R8 handoff §4 item 5 — falsifier before implementation
Design:  wf2-climate-analysis-v2-design.md §5.6 "Calendar-aware weighting", §8 row 5b
Ref:     test_case/ref_wf2_pre_5a  (post-A3 state: 18 attrs-only diffs, values clean)
Depends: A3 (fbcd4df) — the true calendar now exists on the raw slices
```

## What 5b claims

The annual aggregate weights each month by its length **in the model's own
calendar**. Today `get_change_annual_clim_proj` treats all twelve months alike:
`.resample(YS-…).sum()` for precip, `.mean()` for temp. The design's stated
purpose is that this "makes annual means comparable across models with different
calendars rather than differing for procedural reasons".

Aggregation becomes, per hydrological year:

* **temp** (mean-aggregated): `Σ(T_m · d_m) / Σ d_m` — a duration-weighted mean
* **precip** (sum-aggregated): `Σ(P_m · d_m)` — the annual total of a rate

## G0 — prerequisite: the series must carry the calendar

Measured 2026-07-30: it does **not**. `cst_calendar` is on the raw slices (A3) but
is not propagated to `series/*.nc`, and stage B reads the series, not the raw.

**Falsified if** 5b lands without `cst_calendar` on the series — the weighting
would then be reading a calendar from somewhere other than the model.

## G1 — the lengths must come from the CALENDAR, never from the axis

This is the trap A3 exists to prevent, and it is easy to walk back into. The
series' time axis is `datetime64` (Gregorian) because `harmonise_dims` converted
it. `time.dt.days_in_month` on that axis returns **Gregorian** lengths — 29 days
for February in a leap year — for a model that has no such day.

**Falsified if** a `noleap` series spanning a leap year (e.g. 2000, 2004) gives
February a weight of 29. It must be 28, always, for every year.

The observation: build the weights for a `noleap` calendar over 2000–2004 and
assert every February is 28. Using `time.dt.days_in_month` fails this; deriving
from `(calendar, year, month)` passes.

## G2 — on a 360-day calendar, 5b must be a NO-OP

Every month is 30 days, so the weights are uniform and a weighted mean is the
unweighted mean. This is 5b's analogue of 5a's strict-generalization claim.

**Falsified if** a `360_day` series produces non-uniform weights, or if its
weighted annual aggregate differs from the unweighted one beyond float tolerance.

## G3 — on noleap/standard it must NOT be a no-op

**Falsified if** a `noleap` or `standard` series yields a weighted annual mean
equal to the unweighted one. Month lengths there are 31/28/31/30/… — genuinely
non-uniform — so equality means the weights never reached the aggregation.

## G4 — analytic correctness, not merely "different"

**Falsified if** for a synthetic year with known monthly values the weighted
result differs from the hand-computed `Σ(v_m · d_m) / Σ d_m`.

"Changed" is not "correct"; G3 only proves something moved.

## G5 — an unweightable calendar must RAISE, naming the series

The design: "stage B raises on a calendar it cannot weight." A3 made this
reachable by introducing `CALENDAR_UNKNOWN` as a refusable sentinel.

**Falsified if** a series carrying `cst_calendar = "unknown"` (or an unrecognised
name) is silently weighted with Gregorian lengths. That is precisely the failure
A3 was written to end, re-entering through the back door.

## G6 — the purpose claim, which none of the above tests

G1–G5 test mechanics. The design's *reason* is cross-model comparability.

**Falsified if** two synthetic series representing the **same underlying climate**
on different calendars (`360_day` vs `noleap`) do not become *closer* in annual
mean after weighting than before. If weighting does not reduce the purely
procedural gap between calendars, the step does not do what it is for.

This is the test that would catch a weighting that is self-consistent but wrong.

## G7 — the tree diff, and why 5b is gateable where 5a was not

Unlike 5a, this step **must move values on this fixture**. All three fixture
models are `noleap` (A3, `fbcd4df`), whose month lengths are non-uniform, so the
annual aggregates genuinely change.

- **Falsified (no-op) if** `semantic_tree_diff --ref test_case/ref_wf2_pre_5a`
  reports no value differences in the change factors and summary.
- **Falsified (contaminated) if** the `raw/` slices change. 5b is stage B only;
  it touches no series and no raw slice, so a raw diff means something unrelated
  moved.
- **Expected:** `summary/*`, `timeseries/*` and the change factors move; `raw/*`
  identical; `series/*` identical (5b does not re-derive them).

`check_baseline` **will legitimately fail** here — 15 targets include the summary
artifacts. That is the first real re-record of the milestone, and it must happen
only after the diff is characterized.

## G8 — no re-derivation, and no network

5b changes stage B only. Stage B has no cache and re-runs every invocation, and
`REDUCER_KERNEL` covers stage A.

**Falsified if** the dry-run schedules any `fetch_gcm_raw` or `reduce_gcm_series`
job. That would mean the change leaked into the series identity, and 5b would be
paying a re-derivation it has no reason to pay.

## Outcome — 2026-07-30, all nine discharged

| | Result |
|---|---|
| G0 | series now carry `cst_calendar='noleap'` at schema 4 |
| G1–G6 | 16 unit tests pass, no fixture, no network |
| G7 | 126 compared, 21 failed: 18 **attrs only** (9 raw + 9 series), 3 summary with **real value changes** |
| G8 | dry-run scheduled **zero** `fetch_gcm_raw` from the 5b logic itself |
| `check_baseline` | **FAILED on 3 targets** as predicted, then re-recorded OK 15/15 |

The measured effect, which is what 5b is for:

```
precip mean  12.57397183 -> 12.54277532
precip std   19.87138831 -> 20.00744164
temp   mean   1.332215015 ->  1.332610508
```

### Three defects the falsifier caught, none of which were the weighting

1. **The test helper lied.** `pd.date_range("2001-01-15", freq="MS")` snaps
   FORWARD to the next month start, so every synthetic axis began in February and
   each index assertion was off by one. Surfaced as G1 reporting "February = 31".
2. **`.rename` twice.** xarray drops the name on any binary op between
   differently-named operands, so `da * w` is already unnamed — the
   *multiplication* loses it, not just the division. The first fix renamed only
   the division branch and precip still failed, because precip never divides.
3. **The schema bump was mandatory, not tidiness.** The attribute stamping lives
   in the snakemake body, OUTSIDE the functions `kernel_hash` enumerates, so
   `REDUCER_HASH` did not move. Snakemake scheduled all 9 reduce jobs (it saw the
   script change), each revalidated its own cache, found the digest unmoved, and
   returned — `utime`-ing the file without rewriting it. Observed directly: series
   "rewritten" at 21:19 still carrying schema 3 and no calendar. `SCHEMA_VERSION`
   3→4 is what actually applied the change.

Defect 3 is the same class as 5a's `REDUCER_KERNEL` omission: **a change outside
the hashed surface is invisible to the cache**, and the failure is silent success.

## Order of work

1. Propagate `cst_calendar` raw → series (G0). Costs a series re-derivation —
   from local raw, no network.
2. `month_length_weights(times, calendar)` + G1–G5 unit tests. No fixture.
3. Wire into the annual aggregation; G5's raise.
4. Dry-run: assert G8 (no fetch, no reduce beyond step 1's).
5. Run; characterize G7's diff **before** re-recording.
6. Re-record the baseline, and take a `ref_wf2_pre_5c` snapshot — the pre-5a
   reference is now two value-changing steps behind.
