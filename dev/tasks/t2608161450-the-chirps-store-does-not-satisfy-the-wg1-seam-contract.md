---
title: The chirps store does not satisfy the WG-1 seam contract, and nothing reports it
type: todo-item
status: backlog
effort: 2
area: wf0 / climate store + interchange contracts
origin: 2026-08-16 wf0 two-source run
queue:
created: 2026-08-16
updated: 2026-08-17
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

## The two absent global attrs are the visible part of a wider loss (2026-08-17)

The chirps store carries **no catalog metadata at all**. Measured on the same
fixture while building rule 0.06, its entire attribute set is:

```
{'region_bbox': array([9.658, 0.35, 9.858, 0.483])}
```

The era5 store carries eight: `crs`, `category`, `notes`, `paper_doi`,
`paper_ref`, `source_license`, `source_url`, `source_version` — the catalog
entry's `metadata:` block, which hydromt attaches to what it returns and
`prep_historical_climate` writes through.

**The cause is now known, and it is one line.** The era5 branch fetches a whole
Dataset and keeps its attrs; the chirps branch fetches ONE variable and calls
`.to_dataset()` on the DataArray (`extract_historical_climate.py`, the
`clim_source == "chirps"` branch), and the entry's metadata does not survive
that. So `crs` and `category` were never two isolated omissions — they are the
two rows `validate_wg1` happens to check out of eight that are all missing for
the same reason. A producer fix that stamps the metadata back onto `ds` before
`to_netcdf` closes all eight at once.

**Rule 0.06 works around it rather than waiting for the fix**: its comparison
table reads provenance from the store and falls back to the catalog entry
(`compare_sources._catalog_metadata`), because without that the Reference and
Version columns are blank for exactly the precipitation-only sources a
comparison is run to judge. The workaround is deliberate and documented, and it
should be REMOVED once the producer is fixed — it is a second path to the same
fact, which is the thing this repo generally refuses. It stays until then
because a blank Reference column is a defect a reader sees.

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

## Two directions — RULED 2026-08-17: take BOTH

> **Ruling (owner).** Both, not one. (1) is what a promotion needs; (2) is what
> stops the class recurring for the next source admitted to
> `_SUPPORTED_SOURCES`. Neither alone was accepted: (1) alone leaves the next
> unexercised source failing silently in the same place, and (2) alone leaves
> chirps unusable while reporting that it is.

1. **Fix the producer** so the chirps branch emits a WG-1-conforming store. This
   is the substantive answer and it is what a promotion needs.
2. **Make WF0 report it** -- validate each candidate store against WG-1 and
   surface the diffs beside the comparison, so a source that cannot be promoted
   says so where the choice is made. Cheap, and useful even after (1), since it
   generalises to the next source.

**Order is not free.** (2) is the falsifier for (1): wire the WF0 validation
first so it FAILS on the current chirps store, then fix the producer and watch
it pass. Fixing first leaves the check asserting a condition already true, which
is the weaker gate. The units row is exempt from both and comes first — see
below; it is a values question, not a producer-conformance one.

`chirps_global` is in scope with `chirps`, per *Both precip-only sources* above:
it is admitted by the same list and has never been run, so fixing only the
staged name leaves the identical defect behind the other one.

## Refs

- `dev/reference/contracts/weather-generator-seam.md` -- WG-1.
- `blueearth_cst/shared/interchange_contracts.py::validate_wg1`.
- Landed in the same session: `47b80c6`, `d5d9415`, `1cf303f`, `640f24d`.
- [[skip-outputs-for-missing-variables]] is the adjacent ruling from the same
  run -- reporting must reflect what a dataset actually carries.

## Progress

- [x] Decide between the two directions above (or take both) — **both, ruled
      2026-08-17.** See the ruling callout for the order and why.
- [ ] Check `precip` units against the VALUES before touching the attribute.
      Independent of the ruling and first in line: `mm` vs `mm d**-1` is either
      a label defect or a magnitude that multiplies through the whole stress
      test, and the attribute cannot tell them apart. Needs a chirps store on
      disk, which this slot has not got (`test_case/test_rapid` is absent here).
- [ ] Wire the WF0 candidate-store WG-1 validation and watch it FAIL on the
      current chirps store, surfacing the diffs beside rule 0.06's comparison.
- [ ] Stamp the catalog metadata and the `float32` dtypes back onto the chirps
      branch's dataset before `to_netcdf` — one producer fix closes all eight
      rows — and confirm the check from the previous step turns green.
- [ ] Drop rule 0.06's catalog fallback (`compare_sources._catalog_metadata`),
      which exists only to paper over the missing metadata.
- [ ] Cover `chirps_global` by the same fix; it has never been run at all.
