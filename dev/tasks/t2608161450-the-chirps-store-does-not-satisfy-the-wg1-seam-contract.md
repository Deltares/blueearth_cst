---
title: The chirps store does not satisfy the WG-1 seam contract, and nothing reports it
type: todo-item
status: backlog
effort: 2
area: wf0 / climate store + interchange contracts
origin: 2026-08-16 wf0 two-source run
queue:
created: 2026-08-16
updated: 2026-08-16
---

> [!note] Overview
> **What** — A chirps climate store fails `validate_wg1` on eight counts. WF0 draws its figures and exits 0 regardless, because nothing in WF0 validates WG-1 -- so a candidate source can WIN a forcing comparison and then fail when it is promoted to `shared.clim_historical` and WF3 reads it.
> **Why** — The failure is invisible at exactly the moment the decision is made. WF0 exists to choose a forcing dataset; a candidate that cannot serve as one should not pass its evaluation silently.
> **Effort** — Medium: the dtype and attribute rows are a producer fix; the units row needs checking against values before anything is changed.

## Measured

First end-to-end run of the wf0 multi-source path, 2026-08-16, against
`test_case/snake_config_rapid.yml` with `candidate_sources: [chirps]`. Both
stores checked with `validate_wg1` after the run.

| Contract term | Expected | era5 | chirps |
|---|---|---|---|
| dims / coords | `(time, latitude, longitude)` | ok | ok |
| coord dtype | `float32` | ok | **`float64`** |
| `temp` / `temp_min` / `temp_max` dtype | `float32` | ok | **`float64`** |
| global attr `crs` | `4326` | ok | **absent** |
| global attr `category` | `meteo` | ok | **absent** |
| `precip` units | `mm d**-1` | ok | **`mm`** |
| `spatial_ref`, seven variables | present | ok | ok |

era5: **PASS**. chirps: **8 diffs**.

## Why it was never seen

`validate_wg1`'s own docstring says it: *"chirps-branch facts (precip-only + the
orography sidecar) are NOT checked here -- no chirps fixture exists (design R2);
this validator is era5-grounded."* The branch had additionally never run to
completion -- the same run had to fix a hardcoded `merit_hydro` (`47b80c6`) and a
`lat`/`lon` coordinate spelling (`d5d9415`) before it produced a store at all.

## The units row is the one that is not merely cosmetic

`mm` against a contract of `mm d**-1` is either a wrong label or a wrong
magnitude, and the two are not distinguishable from the attribute. **Check the
values before changing anything.** The catalog's `chirps` entry carries
`unit_add: {time: 86400}` and `rename: {precipitation: precip}` and no unit
conversion, and CHIRPS is natively mm/day, so a label defect is the likely
answer -- but "likely" is not the standard for a quantity that multiplies
through the whole stress test.

The dtype and global-attr rows are producer-side and cheap: the chirps branch
builds its dataset by hand (`extract_historical_climate.py`), while the era5
branch gets its dtypes and attrs from the hydromt read.

## Both precip-only sources are in scope

`chirps_global` is admitted by the same `_SUPPORTED_SOURCES` list and is equally
unexercised -- it has no local staging, so it has never been run at all. Fixing
only the source that happened to be staged would leave the identical defect
behind the other name.

## Two directions, not yet chosen

1. **Fix the producer** so the chirps branch emits a WG-1-conforming store. This
   is the substantive answer and it is what a promotion needs.
2. **Make WF0 report it** -- validate each candidate store against WG-1 and
   surface the diffs beside the comparison, so a source that cannot be promoted
   says so where the choice is made. Cheap, and useful even after (1), since it
   generalises to the next source.

They are complementary; (2) is what stops this class recurring.

## Refs

- `dev/reference/contracts/weather-generator-seam.md` -- WG-1.
- `blueearth_cst/shared/interchange_contracts.py::validate_wg1`.
- Landed in the same session: `47b80c6`, `d5d9415`, `1cf303f`, `640f24d`.
- [[skip-outputs-for-missing-variables]] is the adjacent ruling from the same
  run -- reporting must reflect what a dataset actually carries.

## Progress

- [ ] Decide between the two directions above (or take both).
- [ ] Check `precip` units against the VALUES before touching the attribute.
