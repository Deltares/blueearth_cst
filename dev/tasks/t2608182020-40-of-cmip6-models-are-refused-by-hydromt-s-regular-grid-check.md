---
title: 40% of CMIP6 models are refused by hydromt's regular-grid check
type: todo-item
status: backlog
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

- [ ] Decide the approach (see the three below); it is a design call, not a patch
- [ ] Implement on the ONE fetch path — `fetch_gcm_raw.fetch_raw_slice`, which
      WF2 rule 2.04 and `dev/scripts/stage_cmip6.py` both call
- [ ] Decide what happens to slices already cached under the old path: whether
      the digest changes, and therefore whether existing `raw/` caches invalidate
- [ ] Re-run `dev/scripts/probe_cmip6_grids.py` afterwards and confirm the
      refused set is empty (bar the genuine outliers below)

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

**One genuine outlier**, which is NOT Gaussian and needs its own answer:
`CAS/FGOALS-g3` at `dlat 2.0253 .. 5.1811` — a real variable-resolution grid.

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
- `blueearth_cst/projections/fetch_gcm_raw.py` — `fetch_raw_slice`, the single
  read path both WF2 and the staging tool use. Whatever is done goes here.
- [[t2608172138]] — the wf1 preflight, same "refuse it up front" instinct that
  `stage_cmip6.plan` now applies to absent members and scenarios. The grid
  problem CANNOT be pre-filtered the same way: the catalog records which stores
  exist, not their grid geometry, so this only surfaces at fetch time.
