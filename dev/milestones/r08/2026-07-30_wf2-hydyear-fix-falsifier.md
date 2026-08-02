# Falsifier — the hydrological-year off-by-one fix (option A)

```
Written: 2026-07-30, BEFORE the fix
Blocker: dev/milestones/r08/2026-07-30_wf2-5f-hydyear-offbyone.md (owner chose A)
Design:  §5.4 A1 — "30 complete hydrological years only when start_month is
         January; for any other start month, 29"
Ref:     test_case/ref_wf2_pre_5f  (post-5e; CLEAN against itself)
Order:   BEFORE 5f, so 5f's A1 acceptance test can be written once, truthfully
```

## The claim

A hydrological year `Y` runs `[Y-{month}-01, (Y+1)-{month}-01 − 1 month]`. It is
**complete** iff both ends lie inside the data. The effective window is the span
of the complete years, and `n_hyd_years` counts them.

The current implementation ends the window at `{last_year}-{month} − 1 month`,
i.e. it assumes the year *starting* in `last_year` is always incomplete. That is
right when the data stops mid-year and wrong when it runs through that year's end.

## L1 — the two cases A1 names, which the fix exists to satisfy

Over a series spanning exactly `1985-01 … 2014-12`:

| `start_month` | effective | `n_hyd_years` |
|---|---|---|
| `Jan` | 1985-01-01 → 2014-12-01 | **30** |
| `Oct` | 1985-10-01 → 2014-09-01 | **29** |

**Falsified if** either differs. The October value is the one to watch: it is
**already correct today**, so a naïve `+1` would break it. The fix must be
conditional on whether the data covers the final year's end, not unconditional.

## L2 — the seed must gain exactly one year

`[1990, 2010]` slices `1990-01 … 2010-12`.

**Falsified if** the effective window is not `1990-01-01 → 2010-12-01` with
`n_hyd_years = 21`. Today it is `1990-01-01 → 2009-12-01`, 20.

## L3 — a mid-year end must still drop the partial year

**Falsified if** data ending `2014-06` with a January start reports 2014 as
complete. The final year is genuinely partial and must be dropped — this is the
property the current code gets right and the fix must preserve.

## L4 — a late start must drop the leading partial year

**Falsified if** data starting `1985-03` with a January start counts 1985. The
1985 hydrological year began before the data did.

## L5 — no complete year must fail loudly

**Falsified if** a span shorter than one hydrological year returns a zero-length
or inverted window rather than raising. An empty reference propagates as an empty
denominator into every relative change factor — the same failure D1's
"entirely after 2014" guard exists to prevent.

## L6 — values MUST move, and only where the window reaches

This is a value-changing step by construction: the reference gains a year, so
every change factor is referenced against more data.

**Falsified (no-op) if** the summary artifacts do not move.

**Falsified (contaminated) if** `raw/` or `series/` move. The window is applied in
stage B; stage A is window-independent by design (G5 — analysis windows are
deliberately excluded from the series identity so a window change schedules zero
reduce jobs). If a series moves, that separation has broken.

That last one is worth stating as a positive prediction: **this step should
schedule no `reduce_gcm_series` job at all**, which is the clearest evidence G5
still holds after five value-changing steps.

## L7 — the composition record must follow

`composition.csv` carries `reference_window_effective` and
`n_hyd_years_reference`.

**Falsified if** they still report the old window after the fix — the record must
track the arithmetic, which is the whole reason 4d extracted
`hydrological_year_bounds` into one shared definition rather than two.

## Order of work

1. Fix `hydrological_year_bounds`; unit tests L1–L5 (no fixture, no network).
2. Dry-run: assert L6's no-reduce prediction.
3. Run; characterize the diff; check L7 in `composition.csv`.
4. Re-record, snapshot `ref_wf2_pre_5f_fixed`, then proceed to 5f.

---

## Outcome — 2026-07-30, all seven discharged

| | Result |
|---|---|
| L1 | Jan → 30 (1985-01-01→2014-12-01); Oct → 29 (1985-10-01→2014-09-01) |
| L2 | seed → `1990-01-01 / 2010-12-01`, **21 years** (was 20) |
| L3/L4 | mid-year end and late start still drop their partial years |
| L5 | a sub-year span raises rather than returning an empty window |
| L6 | 4 files: 3 summary + `composition.csv`. **Zero `raw/`, zero `series/`** |
| L7 | composition tracked the arithmetic with no separate edit |
| `check_baseline` | FAILED on 3 targets, re-recorded OK 15/15 |

13 unit tests, including A1's exact claim parameterised across five start months:
30 for January, 29 for every other.

Measured effect — a 20-sample reference gaining a 21st year moves dispersion most:

```
precip mean   12.091314 -> 10.981394   (-1.109920)
precip std    16.175005 -> 11.505303   (-4.669702)
temp   mean    1.175584 ->  1.180171   (+0.004587)
```

### L6 caught something bigger than it was checking for

The prediction was "no `reduce_gcm_series` job". The dry-run said **"Nothing to be
done"** — no job at all. The fix was correct, its unit tests passed, and the
workflow would not have applied it.

Snakemake's code trigger tracks a rule's **script**, not the modules it imports.
The fix lives in `get_change_climate_proj.py`; stage B's script is
`derive_change_factors.py`. Stage A has `REDUCER_HASH` for exactly this. Stage B
had nothing, on the belief — stated in this repo's own commit messages, mine
included — that it "has no cache and re-runs every invocation". It re-runs when
its script, inputs or params change, and an imported module is none of those.

`STAGE_B_HASH` now closes it, enumerated on the same discipline as
`REDUCER_KERNEL`: name every function whose arithmetic matters, because
`kernel_hash` follows no call graph.

**This is the third instance of one failure mode in this milestone** — 5a's
unlisted `REDUCER_KERNEL` callees, 5b's stamping outside the hashed kernel, and
now stage B having no hashed kernel at all. Each time the symptom was identical:
the change is right, the tests pass, and the artifacts silently do not move.

### G5 confirmed rather than assumed

Zero reduce jobs on a step that changes the analysis window is the design's G5
holding — analysis windows are deliberately excluded from series identity so a
window change schedules no re-derivation. Six value-changing steps in, it still
does.
