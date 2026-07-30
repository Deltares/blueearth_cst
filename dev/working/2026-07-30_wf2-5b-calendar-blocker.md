# BLOCKER — the pipeline replaces the model calendar with a false one

```
Found:  2026-07-30, while drafting step 5b's falsifier (before any 5b code)
Status: OPEN — needs an owner decision; 5b cannot proceed as designed
Scope:  larger than 5b. Existing raw slices and series carry a wrong calendar.
```

## The finding

Step 5b weights each month by its length **in the model's own calendar**, which the
design specifies is "taken from the decoded `cftime` axis (the catalog sets
`decode_times: true`)".

That axis does not survive. Measured on the fixture:

| | GFDL-ESM4 historical | INM-CM5-0 historical |
|---|---|---|
| **Store** (`gs://cmip6`, `.zmetadata` → `time/.zattrs`) | `calendar = 'noleap'` | `calendar = 'noleap'` |
| **Our raw slice** (`raw/*.nc`, `time.attrs` + `encoding`) | `calendar = 'proleptic_gregorian'` | `calendar = 'proleptic_gregorian'` |
| **`cst_calendar` attribute** | `''` (empty) | `''` (empty) |
| In-memory index type at read | `DatetimeIndex` (`datetime64[ns]`) | `DatetimeIndex` |

The design's own inventory records these models as `noleap`. The store confirms it.
Our artifacts assert `proleptic_gregorian`.

**This is worse than losing the calendar.** A missing value is detectable and can
raise. A confidently recorded *wrong* value cannot: every series claims a calendar
it does not have.

## Why it blocks 5b specifically

1. **The weights would be wrong in exactly the way 5b exists to fix.** Month
   lengths taken from this axis are Gregorian — February gets 28 or 29 days by
   year. A `noleap` model has no 29 February at all. 5b would introduce a
   procedural discrepancy between models while claiming to remove one.
2. **The specified guard can never fire.** "Stage B raises on a calendar it cannot
   weight" is unreachable when every series reports `proleptic_gregorian`, a
   calendar it *can* weight — incorrectly.
3. **`cst_calendar` is empty**, so nothing downstream can even detect the problem.
   `fetch_gcm_raw` reads it off the in-memory index (`getattr(index, "calendar",
   "")`), and a `DatetimeIndex` has no `calendar`, so it records `""`.

## Scope beyond 5b

All **9 raw slices and 9 series currently on disk** — and the reference snapshot
`test_case/ref_wf2_pre_5a` taken from them — carry the false calendar. This is a
provenance defect in artifacts already produced and gated, not only a future
problem. It does not invalidate the *values* computed so far (nothing has used the
calendar yet), but it does invalidate the `cst_calendar` field and anything that
would trust it.

## Where it is lost — not yet pinned

The conversion happens upstream of `fetch_gcm_raw`'s `.load()`: the index is
already `DatetimeIndex` when that module sees it. Candidates, in order of
likelihood, none confirmed:

* xarray decoding to `datetime64` rather than keeping `cftime` for a `noleap` axis;
* the catalog's `preprocess: harmonise_dims`;
* `to_netcdf` stamping a default calendar when writing a `datetime64` axis.

Note the third would explain the *written* attribute even if the first is the real
cause of the type change. Pinning this needs one instrumented fetch.

Per AGENTS.md the fix belongs in our own code — `blueearth_cst/`, the Snakefiles or
the catalog — never in a vendored hydromt.

## Options, with their costs

| | Approach | Cost | Consequence |
|---|---|---|---|
| **A** | Preserve the true calendar at fetch (`use_cftime`, or read `time/.zattrs` from the store and stamp `cst_calendar` honestly) | Pins the loss point; likely re-fetch of all 9 raw slices (9 remote opens, ~10 min) | 5b implementable as designed |
| **B** | Stamp the calendar from the store index at fetch **without** changing the time axis | Cheaper; one extra metadata read per source (~0.3 s, the `.zmetadata` read used above) | `cst_calendar` becomes truthful; month lengths still Gregorian, so 5b needs the axis too — partial |
| **C** | Amend the design: accept Gregorian month lengths, document the approximation | No code cost | 5b's stated purpose — "makes annual means comparable across models with different calendars" — is not achieved. A `360_day` model would be weighted 28/31 |
| **D** | Defer 5b until the calendar is preserved | No cost now | 5c–5f can proceed; 5b returns after A |

**Recommendation: A, preceded by the one instrumented fetch that pins where the
conversion happens.** B alone is not sufficient for 5b, though it is worth doing
regardless because a truthful `cst_calendar` is what makes the guard in (2)
reachable. C defeats the step's purpose on precisely the models the inventory
names.

## What is NOT affected

No committed value is wrong. 5a's weights are spatial and calendar-independent;
`check_baseline` 15/15 and the 5a diff stand. The defect is confined to calendar
provenance and to any step that would consume it — 5b being the first.
