---
title: 40% of CMIP6 models are refused by hydromt's regular-grid check
type: todo-item
status: active
effort: 2
area: wf2 projections / remote store
origin: 2026-08-18 stage_cmip6 run
queue: 1
created: 2026-08-18
updated: 2026-08-18
---

> [!note] Overview
> **What** — `27 of 67` CMIP6 models cannot be read at all: hydromt's `.raster`
> accessor requires UNIFORMLY spaced coordinates, and these models publish on a
> Gaussian grid whose latitudes are Legendre roots and so vary by ~1%. They fail
> with `ValueError: The 'raster' accessor only applies to regular grids`.
> **Why** — The 27 are not a fringe: CanESM5, all five EC-Earth3 variants,
> MPI-ESM1-2-HR/LR, CNRM-CM6-1, CNRM-ESM2-1, MIROC6, MRI-ESM2-0, BCC-CSM2-MR.
> They are **silently absent from every WF2 ensemble run so far**, which makes
> this a fact about past results, not only a tooling annoyance.
> **Effort** — Medium-large. The read path is shared with WF2, so it touches
> `fetch_gcm_raw`, the raw digest, and every cached slice.

## Progress

- [x] Decide the approach — **option 1**, as an IRREGULAR-ONLY branch: the
      regular path is untouched down to the cell selection, so nothing that
      works today changes
- [x] Implement on the ONE fetch path — `fetch_gcm_raw.fetch_raw_slice`, which
      WF2 rule 2.04 and `dev/scripts/stage_cmip6.py` both call
- [x] Decide what happens to slices already cached: **nothing invalidates**, and
      `SCHEMA_VERSION` does not move. The regular path is byte-identical, and
      the models this rescues never had a cached slice to begin with
- [x] Acceptance — CanESM5 (Gaussian) staged end to end, with GFDL-ESM4
      (regular) as the control. See "What landed" below
- [ ] Re-run the full ensemble once and confirm no model in the 27 still fails.
      A **spot check is not the measurement**: the branch was verified on one
      Gaussian model, and 26 others have never been read

## What landed

`fetch_raw_slice` wraps its `get_rasterdataset` call and, on hydromt's own
`only applies to regular grids` phrase only, re-reads the SAME store without a
bbox and applies the bbox itself:

- `_slice_spatial_dimensions` is the only step in hydromt's read path that needs
  an evenly spaced grid, and it runs **only when a bbox is passed**. Verified
  against a live CanESM5 store before the branch was written: the un-clipped
  read succeeds, and rename, unit conversion, CRS and nodata are still
  hydromt's.
- `bbox_index_slice` reproduces `.raster.clip_bbox` in index space, off
  `grid_weights.midpoint_edges` rather than an affine transform. Pinned against
  `clip_bbox` itself on a uniform grid across 3 bboxes × 3 buffer values, which
  is what makes the **`buffer` = CELLS** reading evidence rather than assertion.
- Latitude is handled in index space because `harmonise_dims` leaves it N->S: a
  label slice `sel(lat=slice(south, north))` would return nothing, and an empty
  spatial selection reduces to NaN rather than raising.

**The downstream link was checked, not assumed.** A slice that lands on disk but
dies in the reduce stage would leave the 27 models just as absent. The reducer
opens `raw/*.nc` with plain `xr.open_dataset`, resolves its dims name-based
through `_spatial_dim`, and reduces with `grid_weights.weighted_spatial_mean`,
whose D10 weights need ordered but not evenly spaced axes -- so nothing on that
path reaches `.raster.res`. `.raster.vars` (`get_stats_climate_proj:357`,
`derive_change_factors:194`) is `list(data_vars)` and touches no geometry.
Confirmed on the staged CanESM5 slice: geometry check passes, the area-weighted
series is 26.43 degC with no NaN.

Measured 2026-08-18, `--workers 1`, region `test_case/test_local`:

| model | grid | result |
|---|---|---|
| CCCma/CanESM5 | Gaussian | staged, 2x2 cells, 780 steps, 26.4 degC / 9.59 mm day-1 |
| NOAA-GFDL/GFDL-ESM4 | regular (control) | staged, 2x2 cells, 780 steps, 24.5 degC / 9.41 mm day-1 |

CanESM5 cost 1:25 against the control's 0:22 — the branch pays a second open
(~19 s here) on a model that previously produced nothing at all. A cheap
pre-probe of the grid was considered and rejected: it would have to guess which
variable's store to read and could then DISAGREE with hydromt's own verdict.

## Not done here, and why

- **The `buffer_degrees` name is wrong.** hydromt has always read `buffer` as
  "resolution multiplicity" — cells (`data_catalog.py:1370`) — while
  `analyze_projections.smk:402` calls it `REGION_BUFFER_DEGREES` and passes
  `1.0`. Every slice ever fetched used the cell reading, so the branch matches
  it deliberately; correcting the NAME is safe, correcting the SEMANTICS would
  invalidate every cached slice and change every change factor. Its own item.
- **`CAS/FGOALS-g3`** (`dlat 2.0253 .. 5.1811`, a real variable-resolution grid)
  needs no separate answer after all: the branch requires the axis to be
  ordered, not evenly spaced, so it reads like any Gaussian model. Unverified —
  it is in the 26 the sweep above still has to cover.
- **The two `error` models** (MPI-M/ICON-ESM-LR, UA/MCM-UA-1-0) carry no `lat`
  variable and still fail, before any grid question is asked.

## The measurement

Run 2026-08-18 with `dev/scripts/probe_cmip6_grids.py`, which reads only the
`lat`/`lon` coordinates of one `Amon/tas` store per model and applies hydromt's
own test. One store per model, 67 models.

```
regular     38 of 67
IRREGULAR   27 of 67
error        2 of 67   (MPI-M/ICON-ESM-LR, UA/MCM-UA-1-0 — no `lat` variable)
```

**Longitude is regular in every one of the 27.** Only latitude fails:

| model | n lat | dlat range |
|---|---|---|
| CCCma/CanESM5 | 64 | 2.7673 .. 2.7906 |
| EC-Earth-Consortium/EC-Earth3 | 256 | 0.6959 .. 0.7018 |
| MPI-M/MPI-ESM1-2-HR | 192 | 0.9272 .. 0.9351 |
| CNRM-CERFACS/CNRM-CM6-1 | 128 | 1.3890 .. 1.4008 |
| AWI/AWI-CM-1-1-MR | 192 | 0.9272 .. 0.9351 |

That is a **Gaussian grid**, and the variation is ~0.008 deg against hydromt's
tolerance of 5e-4. No Gaussian grid can ever satisfy that test — the data is
well-formed and the tolerance is simply tighter than the grid type allows.

**One genuine outlier**, which is NOT Gaussian: `CAS/FGOALS-g3` at
`dlat 2.0253 .. 5.1811` — a real variable-resolution grid. It looked at the time
as though it needed its own answer; it does not (see "Not done here").

## The check being failed

`hydromt/gis/raster.py:441-444` (hydromt 1.3):

```python
xreg = np.allclose(dxs, dxs[0], atol=5e-4)
yreg = np.allclose(dys, dys[0], atol=5e-4)
if not xreg or not yreg:
    raise ValueError("The 'raster' accessor only applies to regular grids")
```

## Three options, with the one that looks right

1. **Do not route these reads through `.raster`.** Nothing about clipping to a
   bounding box requires even spacing — `xarray.sel(lat=slice(...),
   lon=slice(...))` subsets a Gaussian grid correctly. The constraint belongs to
   hydromt's accessor, not to the data or to what WF2 needs from it.
   **Preferred**, and the least lossy.
2. **Regrid on read** onto a uniform grid. Correct but heavier, and it inserts a
   resampling step into a path whose output feeds a digest — so it changes what
   "the slice" IS, not just how it is fetched.
3. **Skip them.** Rejected: 40% of the ensemble, including the models reviewers
   expect. Only acceptable as a stopgap, and if taken it must be LOUD.

**`gr` cannot rescue this.** CMIP6 lets a model publish a regridded variant, but
only 17 models in the index do, and neither `AWI/AWI-CM-1-1-MR` nor
`NUIST/NESM3` is among them. Grid labels across the whole store index:
`gn` 3364, `gr` 1702, `gr1` 84.

## Refs

- `dev/scripts/probe_cmip6_grids.py` — the probe; re-run it to re-measure.
  Reads coordinates only, so it is far cheaper than a hydromt open.
  **It cannot be the acceptance check**, which this note originally said it was:
  it applies hydromt's tolerance to the store's own coordinates and never touches
  `fetch_gcm_raw`. The grids stay Gaussian, so it will report the same 27
  IRREGULAR forever. It measures grid GEOMETRY, not readability — staging a
  slice is what measures readability.
- `blueearth_cst/projections/fetch_gcm_raw.py` — `fetch_raw_slice`, the single
  read path both WF2 and the staging tool use. Whatever is done goes here.
- [[t2608172138]] — the wf1 preflight, same "refuse it up front" instinct that
  `stage_cmip6.plan` now applies to absent members and scenarios. The grid
  problem CANNOT be pre-filtered the same way: the catalog records which stores
  exist, not their grid geometry, so this only surfaces at fetch time.
