# BLOCKER — the reference window is one hydrological year short (January starts)

```
Found:  2026-07-30, writing 5f's A1 acceptance test, before any 5f code
Status: OPEN — needs an owner decision; it is value-changing, so it cannot ride 5f
Scope:  every run. The seed's reference window is 20 years where the config asks
        for 21.
```

## The contradiction

Design A1 (§5.4, arbitration ext2-05) states, of OQ-4's `1985–2014` window:

> 1985–2014 is 30 calendar years. The complete-hydrological-year policy then
> yields 30 complete hydrological years **only when `start_month_hyd_year` is
> January**; for any other start month the window contains **29**.

Measured against a monthly series spanning exactly `1985-01 … 2014-12`
(360 months = 30 calendar years):

| `start_month_hyd_year` | effective window | `n_hyd_years` | A1 expects |
|---|---|---|---|
| `Jan` | 1985-01-01 → **2013-12-01** | **29** | **30** |
| `Oct` | 1985-10-01 → 2014-09-01 | 29 | 29 ✓ |

The non-January case is right. **January is one year short.**

## Why

`hydrological_year_bounds` (`get_change_climate_proj.py`) ends the window at

```python
end = pd.to_datetime(f"{last_year}-{start_month}") - pd.DateOffset(months=1)
```

i.e. it assumes the hydrological year *starting* in `last_year` is incomplete.
That is correct whenever the data stops mid-year — the October case above — and
wrong when the data runs through the end of that hydrological year, which is
exactly what a January start over data ending in December does. The 2014
hydrological year (2014-01 … 2014-12) is complete in the data and is discarded.

## It is not confined to the template window

The **seed** config has the same shape. `historical_year_range: [1990, 2010]`
slices `1990-01 … 2010-12` — 21 complete January-start years — and the effective
window is `1990-01-01 → 2009-12-01`, **20 years**. The composition record has been
reporting `n_hyd_years_reference = 20` since 4d, faithfully, for a window the
config asked to be 21.

So every change factor this repo has produced is referenced against one year less
data than configured. The numbers are *self-consistent* — nothing is wrong twice —
but the window is not what the config says.

## Why it cannot ride 5f

§8 classifies 5f as **value-neutral**: "the seed fixture keeps `[1990, 2010]`, so
no number moves". Fixing the off-by-one adds a year to the reference window on the
seed, which moves every change factor. That is a value-changing step, needing its
own falsifier, its own diff characterization and its own re-record.

Landing 5f's template change while writing an acceptance test that asserts **29**
for January would encode the defect as intended behaviour — the acceptance test
A1 asked for is precisely the thing that would have to lie.

## Options

| | Approach | Consequence |
|---|---|---|
| **A** | Fix `hydrological_year_bounds` to include the final year when the data covers it; land as its own value-changing step with a gate | A1 satisfied; seed reference becomes 21 years; every change factor moves once |
| **B** | Amend A1: declare `[start, end]` to mean `end` **exclusive**, so 29 is correct for both start months | No value moves; but "1985–2014 is 30 calendar years" in the design becomes false, and the config's own arithmetic stops matching a reader's expectation |
| **C** | Land 5f's template + docs now, record this for a later step | 5f ships honestly; the template's advertised 30-year window silently delivers 29 until fixed |

**Recommendation: A, as its own step immediately after 5f.** B makes the design
text wrong to keep the code right, and the config reads as an inclusive range to
anyone who has not read A1. C is acceptable only if A follows, and the note must
say the template's 30 is really 29 until then.

Whichever is chosen, the A1 acceptance test 5f calls for should be written
**against the decision**, not against current behaviour.

## What is NOT affected

The values produced so far are internally consistent: the same window feeds
reference and scenario, and `composition.csv` has always reported the effective
window truthfully alongside the nominal one. This is a window-length defect, not
an arithmetic one — which is precisely why the nominal/effective split introduced
at 4d is what surfaced it.
