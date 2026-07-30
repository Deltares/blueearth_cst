# BLOCKER — the pipeline replaces the model calendar with a false one

```
Found:  2026-07-30, while drafting step 5b's falsifier (before any 5b code)
Status: RESOLVED 2026-07-30 by A3 (owner-chosen). 5b is unblocked.
        All 9 raw slices now record `cst_calendar = noleap`, read from the store.
        Gates after the corrective re-fetch: check_baseline OK 15/15;
        semantic_tree_diff 126 compared, 18 failed, 0 missing, 0 extra — the 18
        being the 9 raw + 9 series, ATTRS ONLY (zero value or dtype differences).
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

## PINNED 2026-07-30 — and it is our own catalog's doing

Two measurements settle it:

1. **Plain xarray preserves the calendar.** Opening the same zarr with
   `xr.open_zarr(consolidated=True)`, no hydromt, gives `CFTimeIndex` with
   `calendar='noleap'`.
2. **`harmonise_dims` destroys it.** `hydromt/data_catalog/drivers/preprocessing.py:66`:

   ```python
   # Time
   if ds.indexes["time"].dtype == "O":
       ds = to_datetimeindex(ds)
   ```

`preprocess: harmonise_dims` is requested by **our** generated catalog, in the
shared defaults anchor, on all 289 entries
(`dev/scripts/generate_cmip6_catalog.py`, `DEFAULTS_BLOCK`). So this is not an
upstream defect to work around — it is our configuration selecting a preprocess
that bundles a lossy time conversion with the lon/lat harmonisation we do want
(0–360 → −180–180, and S→N → N→S orientation).

**What is NOT corrupted.** `to_datetimeindex` maps monthly midpoints one-to-one
and order-preserving; no step is dropped and no value is touched. Year and month
survive exactly. Only the *calendar identity* is lost. That matters because month
length is a function of (calendar, year, month) — so recovering the calendar NAME
is sufficient; the converted axis can still supply year and month.

### Preferred implementation: A3

| | Approach | Why not |
|---|---|---|
| A1 | Drop `harmonise_dims`, reimplement lon/lat harmonisation ourselves | Gives up two corrections we rely on, to fix a third |
| A2 | Capture the calendar before the preprocess runs | Not reachable — the preprocess runs inside the driver |
| **A3** | **Read the calendar from the store's `.zmetadata` at fetch and stamp `cst_calendar` truthfully; derive month lengths from (calendar, year, month), not from the axis** | — |

A3 costs one extra metadata read per source (~0.3 s, the same `.zmetadata` read
used to produce the table above), keeps `harmonise_dims`' benefits, and gives 5b
exactly what it needs.

**One-time cost.** Existing raw slices carry `cst_calendar = ''`, and the raw
digest does not include the calendar, so a cache hit would skip re-stamping them.
Either fold the calendar into the raw digest components or bump
`series_identity.SCHEMA_VERSION` — both invalidate the 9 slices and re-fetch them
once (~10 min, 9 remote opens). That is the honest price of having recorded a
false value; there is no way to correct an artifact without rewriting it.

## Where it is lost — superseded by the section above

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
