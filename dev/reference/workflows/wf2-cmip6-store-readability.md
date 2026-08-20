# WF2 — which CMIP6 stores this toolbox can read

```
Status:   reference measurement (promoted from board items t2608182020 and
          t2608191613 on 2026-08-20, when both closed)
Dates:    grid geometry   — probed 2026-08-18, staged 2026-08-18/19
          version pinning — measured 2026-08-19 over the 2026-07-29 crawl
Source:   gs://cmip6 (Pangeo/Google public CMIP6 zarr mirror), and
          `config/catalogs/cmip6_store_index.json`
Method:   `dev/scripts/probe_cmip6_grids.py` for geometry; `dev/scripts/stage_cmip6.py`
          for readability; a pass over the store index for version multiplicity
```

Companion to `wf2-cmip6-store-inventory.md`, which records what **exists** in the
store. This one records what this toolbox can **read** out of it, and why some
stores it could not.

Both sections below describe the same failure shape, and that is why they share a
document: a model that cannot be read is **silently absent from an ensemble**.
Nothing in WF2 reports a model that never arrived, so an absence is a fact about
past results, not only a tooling annoyance. Both causes are now handled, and both
are worth re-measuring after a catalog re-crawl.

---

## 1. Grid geometry — 27 of 67 models publish on a Gaussian grid

### The check being failed

`hydromt/gis/raster.py:441-444` (hydromt 1.3):

```python
xreg = np.allclose(dxs, dxs[0], atol=5e-4)
yreg = np.allclose(dys, dys[0], atol=5e-4)
if not xreg or not yreg:
    raise ValueError("The 'raster' accessor only applies to regular grids")
```

The `.raster` accessor requires **uniformly spaced** coordinates. A Gaussian
grid's latitudes are Legendre roots and so vary by ~1% — about `0.008` degrees
against a tolerance of `5e-4`. **No Gaussian grid can ever satisfy that test.**
The data is well formed; the accessor's precondition is simply tighter than the
grid type allows.

### The measurement, 2026-08-18

One `Amon/tas` store per model, 67 models, coordinates only:

```
regular     38 of 67
IRREGULAR   27 of 67
error        2 of 67   (MPI-M/ICON-ESM-LR, UA/MCM-UA-1-0 — no `lat` variable)
```

The 27 are not a fringe. They include CanESM5, all five EC-Earth3 variants,
MPI-ESM1-2-HR/LR, CNRM-CM6-1, CNRM-ESM2-1, MIROC6, MRI-ESM2-0 and BCC-CSM2-MR.

**Longitude is regular in every one of the 27.** Only latitude fails:

| model | n lat | dlat range |
|---|---|---|
| CCCma/CanESM5 | 64 | 2.7673 .. 2.7906 |
| EC-Earth-Consortium/EC-Earth3 | 256 | 0.6959 .. 0.7018 |
| MPI-M/MPI-ESM1-2-HR | 192 | 0.9272 .. 0.9351 |
| CNRM-CERFACS/CNRM-CM6-1 | 128 | 1.3890 .. 1.4008 |
| AWI/AWI-CM-1-1-MR | 192 | 0.9272 .. 0.9351 |

**One genuine outlier, which is not Gaussian:** `CAS/FGOALS-g3` at
`dlat 2.0253 .. 5.1811`, a real variable-resolution grid. It needed no separate
answer — the branch below requires the axis to be *ordered*, not evenly spaced,
so it reads like any Gaussian model.

### `gr` cannot rescue this

CMIP6 lets a model publish a regridded variant, but only 17 models in the index
do, and neither `AWI/AWI-CM-1-1-MR` nor `NUIST/NESM3` is among them. Grid labels
across the whole store index: `gn` 3364, `gr` 1702, `gr1` 84.

### What the toolbox does — the irregular branch

`fetch_gcm_raw.fetch_raw_slice` catches hydromt's own phrase and re-reads the
**same** store without a bbox, then applies the bbox itself in index space
(`bbox_index_slice`). `_slice_spatial_dimensions` is the only step in hydromt's
read path needing an evenly spaced grid, and it runs only when a bbox is passed —
so rename, unit conversion, CRS and nodata are still hydromt's. The regular path
is untouched down to the cell selection, so nothing cached invalidates. The
mechanism and its rationale live at the code, not here.

Rejected: a cheap pre-probe of the grid, which would have to guess which
variable's store to read and could then **disagree with hydromt's own verdict**.

### Readability, measured by staging

Grid geometry is not readability. The probe applies hydromt's tolerance to a
store's coordinates and will report the same 27 forever; **staging a slice is
what measures readability**, and it was measured — 29 models attempted (the 27
plus the probe's 2 errors), each staged slice then opened, checked for the digest
and both variables, and reduced through the real reducer:

```
27 slices written, all 27 through the irregular branch
27 of 27 reduce cleanly: 2x2 to 3x3 cells, 23.5-26.5 degC, 4.4-9.9 mm day-1, no NaN
 2 failures, NEITHER of them a grid problem
```

Cost, `--workers 1` against `test_case/test_local`: CanESM5 1:25 against the
regular control GFDL-ESM4's 0:22 — a second open (~19 s) on a model that
previously produced nothing at all.

The two failures, and why neither is a grid problem:

- **`MPI-M/ICON-ESM-LR`** — `ValueError: x dimension not found`. An unstructured
  (ICON) mesh with no x/y axis at all, so there is no bbox to take. It fails
  before any grid question is asked, and needs mesh handling rather than a clip.
- **`MIROC/MIROC-ES2H`** — `NoDataException: No data left after temporal
  slicing`. Its `historical/Amon` store holds **12 months (1850)** under every
  one of its three members. Asked for a window the store does cover, it reads
  through the branch normally (2x3 cells, 25.13 degC), so **its grid is fine**.

**The probe undercounted by one.** `UA/MCM-UA-1-0` reads perfectly well: it
spells its axes `latitude`/`longitude`, which hydromt's `set_spatial_dims`
accepts and the probe's hardcoded `ds["lat"]` did not. Its grid *is* irregular,
so it takes the branch like the rest. The probe now tries both spellings — a
probe stricter than the thing it probes invents refusals, which is the one result
a diagnostic must not produce. **The branch rescues 28 models, not 27.**

### One data condition, unrelated to the grid

`EC-Earth3 r101i1p1f1` covers 1970-2014, giving 540 steps against the other
models' 780. A store legitimately starting after the window is a data condition,
not a fault (`assert_raw_coverage`); a different member would cover more.

---

## 2. Published-version multiplicity — 221 of 2426 member combinations

### The mechanism

`series_identity.pinned_uri` narrows the catalog's trailing `/*/*` to the one
`<grid_label>/<version>` the store index recorded. When the pins could not name
one physical location it returned `None` — "keep the glob" — and the globbed URI
`.../{variable}/*/*` then matched **every** version of **both** variables, so
four stores went into the combine.

`fetch_gcm_raw.check_time_axis` carries a guard for exactly this ambiguity, but
it never ran: the merge raised first, inside the driver.

### Three failure faces, and a fourth outcome that is not a failure

This is why the bucket is matched on a phrase the toolbox builds rather than on
an exception type:

1. `MergeError: conflicting values for variable 'pr'` — the two versions disagree
   on values. Observed on `CAS/CAS-ESM2-0 historical r1i1p1f1`.
2. `OutOfBoundsDatetime: Out of bounds nanosecond timestamp: 2262-04-16` — the
   versions cover **different spans**, so aligning their indexes goes through
   `pandas.Index.union`, which upcasts to nanoseconds and overflows on the one
   running past 2262. Observed on `CSIRO-ARCCSS/ACCESS-CM2 ssp585 r1i1p1f1`.
   This looks like the 2262 defect fixed in `423af1f` and is not: a single 2300
   store stages fine (`CCCma/CanESM5 ssp585`), because there is no union.
3. A duplicated time axis, where the values agree well enough to combine — the
   one face the existing guard did catch.
4. **Not a failure:** two versions differing only in metadata merge cleanly and
   produce a correct slice. That is why the diagnostic wraps a *failed* read
   instead of refusing up front.

### The measurement, over the 2026-07-29 index

| member combinations | count |
|---|---|
| one version per variable — pinned, fast path | 2205 |
| more than one version each — globbed | 105 |
| `pr` and `tas` versions differ — globbed | 116 |
| **entries with at least one globbed member** | **46 of 289** |

### The rule — the newest version wins (owner ruling, 2026-08-19)

`pinned_uri` takes the newest version per variable and requires every variable to
land on the same location:

| member combinations | before | after |
|---|---|---|
| pin cleanly | 2205 | **2387** |
| refused, globbed | 221 | **39** |

The 39 residual are where `pr`'s newest and `tas`'s newest are *different*
locations: one URI carries one `{variable}` placeholder expanded inside a single
path, so it cannot address both. They keep the diagnostic, which names every
version per variable.

It also **refuses to choose between grid labels** (`gn` vs `gr`). No member in
today's index pins two, so it changes nothing now; it exists because a plain
`max()` would make that choice silently the moment one appeared, and picking a
regridding is not picking a revision.

`SCHEMA_VERSION` moved 5 -> 6 with the rule. The digest carries the pins, never
the rule applied to them, so a slice built by merging two versions and a slice
read from the newest are indistinguishable by digest. One property of the
artifacts follows: `cst_source_paths` records the **candidate** list, not the
resolution, so an ambiguous source's slice still shows both versions — which one
was read is recoverable from the schema version alone.

### Why per-entry catalog pinning is not available

Pinning the version in `cmip6_data.yml` looks like the cleaner fix and cannot be
expressed. The catalog has one uri per entry with a `{member}` placeholder, but
**the version varies by member**: `CAS-ESM2-0 historical` records `gn/v20200302`
for `r1i1p1f1` and `gn/v20200303` for `r2i1p1f1`, both alongside `gn/v20201227`.
Measured: **136 of 289 entries have no single `<grid>/<version>` covering every
member and variable.** Per-member pinning already has exactly one home — the
store index, read by `pinned_uri` — and reaching it needs no re-crawl.

---

## Re-measuring

- `dev/scripts/probe_cmip6_grids.py` — grid **geometry**, coordinates only, far
  cheaper than a hydromt open. It cannot measure readability; the grids stay
  Gaussian, so it reports the same 27 forever.
- `dev/scripts/stage_cmip6.py` — **readability**, by staging real slices. This is
  the acceptance check for anything in section 1.
- Section 2's counts come from a pass over `config/catalogs/cmip6_store_index.json`
  and move only when that index is re-crawled.
