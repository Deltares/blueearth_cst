# WF2 — inventory of the Google Cloud CMIP6 store

```
Status:   reference snapshot (input to the WF2 v2.0 design)
Date:     2026-07-29
Source:   gs://cmip6 (Pangeo/Google public CMIP6 zarr mirror), crawled live
Method:   anonymous gcsfs directory listing of
          cmip6/CMIP6/{CMIP,ScenarioMIP}/{institution}/{source}/{experiment}/
          {member}/{Amon,day,3hr}/  — 2 459 (model, experiment, member) triples,
          4 575 (…, table) rows. Cross-checked against the published index
          https://storage.googleapis.com/cmip6/pangeo-cmip6.csv (514 818 rows,
          Last-Modified 2022-06-28).
```

Companion documents: `wf2_analyze_projections_overview.md` (rule-level map of WF2
today), `../../milestones/r08/wf2-climate-analysis-v2-design.md` (the rework this feeds).

---

## 1. Why the numbers below come from a live crawl

The published index (`pangeo-cmip6.csv`) is **incomplete at the store level**. Its
source list matches the bucket exactly (51 ScenarioMIP sources, zero difference
either way), so it is not stale in the sense of missing models — but individual
zarr stores are absent from it. Confirmed example:
`gs://cmip6/CMIP6/ScenarioMIP/NCC/NorESM2-LM/ssp585/r1i1p1f1/Amon/pr/gn/` exists
in the bucket and is **not** a row in the index, while the sibling `tas` store is.
Counts derived from the index alone therefore understate availability (the index
gives 46 models for ssp585 monthly `pr`+`tas`; the live crawl gives 48).

All model/scenario/member tables in §3–§5 are from the live crawl. The
frequency-family table in §6 is index-derived and marked as such — it is used only
for order-of-magnitude realm coverage, not for availability decisions.

## 2. How WF2 consumes the store (constrains what "available" means)

- `config/catalogs/cmip6_data.yml` declares hydromt entries keyed
  `cmip6_{model}_{scenario}_{member}`, where `{model}` is
  `institution/source_id` (e.g. `NOAA-GFDL/GFDL-ESM4`) and the URI globs
  `.../Amon/{variable}/*/*` — so grid label and version are resolved
  automatically, but the **frequency is hardwired to `Amon`** (monthly).
- **The `*/*` glob is not guaranteed to match exactly one store.** Checked across
  all 69 declared combinations × {`pr`,`tas`}: 137 resolve to a single
  `{grid}/{version}` pair, one does not — `NCC/NorCPM1` historical `tas`
  publishes both `gn/v20190914` and `gn/v20200724`, so the glob matches two zarr
  stores with fully overlapping time. It lands on `tas`, which is aggregated with
  `.mean()`, so the effect is muted; the same situation on a `precip` entry would
  feed duplicate months into an aggregation. The `drop_duplicates(dim="time")` in
  the per-variable fallback path (`get_stats_climate_proj.py:232`) only runs when
  the combined open has already raised. Worth an explicit uniqueness check in v2.
- Variables are renamed in the adapter: `pr→precip`, `tas→temp`, `rsds→kin`,
  `psl→press_msl`. Only those four CMIP6 variables are reachable today.
- `analyze_projections.smk` has no `{member}` wildcard.
  `blueearth_cst/projections/get_stats_climate_proj.py` **loops** the configured
  `members:` list and merges the results along a `member` dimension, so
  multi-member ensembles are supported by the code.
- **Silent-empty failure mode.** For each member the script builds the entry name
  and checks `if entry in data_catalog.sources`. A `(model, scenario, member)`
  combination that the catalog does not declare produces an empty `xr.Dataset()`
  and no error — the run completes with a silently thinner ensemble. All 69
  combinations currently declared in `cmip6_data.yml` were verified live to
  resolve to real `Amon` `pr`+`tas` stores (see §4), so this is latent, not
  active.

## 3. Overview A — models, scenarios, ensemble members (monthly, `Amon`)

Cell = number of members carrying **both** `pr` and `tas`. Full machine-readable
table: `wf2-cmip6-monthly-members.csv` (daily equivalent:
`wf2-cmip6-daily-members.csv`).

| institution | source | hist | ssp119 | ssp126 | ssp245 | ssp370 | ssp434 | ssp460 | ssp534-over | ssp585 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| AS-RCEC | TaiESM1 | 2 | 0 | 1 | 1 | 1 | 0 | 0 | 0 | 1 |
| AWI | AWI-CM-1-1-MR | 5 | 0 | 1 | 1 | 5 | 0 | 0 | 0 | 1 |
| AWI | AWI-ESM-1-1-LR | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| BCC | BCC-CSM2-MR | 3 | 0 | 1 | 1 | 1 | 0 | 0 | 0 | 1 |
| BCC | BCC-ESM1 | 3 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| CAMS | CAMS-CSM1-0 | 3 | 2 | 2 | 2 | 2 | 0 | 0 | 0 | 2 |
| CAS | CAS-ESM2-0 | 4 | 0 | 2 | 2 | 2 | 0 | 0 | 0 | 2 |
| CAS | FGOALS-f3-L | 3 | 0 | 1 | 1 | 1 | 0 | 0 | 0 | 1 |
| CAS | FGOALS-g3 | 6 | 1 | 4 | 4 | 5 | 1 | 1 | 1 | 4 |
| CCCR-IITM | IITM-ESM | 1 | 0 | 1 | 1 | 1 | 0 | 0 | 0 | 1 |
| CCCma | CanESM5 | 65 | 50 | 50 | 50 | 50 | 5 | 5 | 5 | 50 |
| CCCma | CanESM5-CanOE | 3 | 0 | 3 | 3 | 3 | 0 | 0 | 0 | 3 |
| CMCC | CMCC-CM2-HR4 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| CMCC | CMCC-CM2-SR5 | 3 | 0 | 1 | 1 | 1 | 0 | 0 | 0 | 1 |
| CMCC | CMCC-ESM2 | 1 | 0 | 1 | 1 | 1 | 0 | 0 | 1 | 1 |
| CNRM-CERFACS | CNRM-CM6-1 | 29 | 0 | 6 | 10 | 6 | 0 | 0 | 0 | 6 |
| CNRM-CERFACS | CNRM-CM6-1-HR | 1 | 0 | 1 | 1 | 1 | 0 | 0 | 0 | 1 |
| CNRM-CERFACS | CNRM-ESM2-1 | 10 | 5 | 5 | 10 | 5 | 5 | 5 | 5 | 5 |
| CSIRO | ACCESS-ESM1-5 | 40 | 0 | 40 | 40 | 40 | 0 | 0 | 0 | 38 |
| CSIRO-ARCCSS | ACCESS-CM2 | 5 | 0 | 5 | 5 | 5 | 0 | 0 | 0 | 5 |
| DKRZ | MPI-ESM1-2-HR | 0 | 0 | 1 | 2 | 10 | 0 | 0 | 0 | 1 |
| DWD | MPI-ESM1-2-HR | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 1 |
| E3SM-Project | E3SM-1-0 | 5 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| E3SM-Project | E3SM-1-1 | 1 | 0 | 0 | 10 | 0 | 0 | 0 | 0 | 1 |
| E3SM-Project | E3SM-1-1-ECA | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| EC-Earth-Consortium | EC-Earth3 | 72 | 51 | 57 | 96 | 57 | 50 | 0 | 50 | 58 |
| EC-Earth-Consortium | EC-Earth3-AerChem | 2 | 0 | 0 | 0 | 2 | 0 | 0 | 0 | 0 |
| EC-Earth-Consortium | EC-Earth3-CC | 1 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 1 |
| EC-Earth-Consortium | EC-Earth3-Veg | 8 | 3 | 7 | 8 | 6 | 0 | 0 | 0 | 8 |
| EC-Earth-Consortium | EC-Earth3-Veg-LR | 3 | 3 | 3 | 3 | 3 | 0 | 0 | 0 | 3 |
| FIO-QLNM | FIO-ESM-2-0 | 3 | 0 | 3 | 3 | 0 | 0 | 0 | 0 | 3 |
| HAMMOZ-Consortium | MPI-ESM-1-2-HAM | 3 | 0 | 0 | 0 | 3 | 0 | 0 | 0 | 0 |
| INM | INM-CM4-8 | 1 | 0 | 1 | 1 | 1 | 0 | 0 | 0 | 1 |
| INM | INM-CM5-0 | 10 | 0 | 1 | 1 | 5 | 0 | 0 | 0 | 1 |
| IPSL | IPSL-CM5A2-INCA | 1 | 0 | 1 | 0 | 1 | 0 | 0 | 0 | 0 |
| IPSL | IPSL-CM6A-LR | 32 | 6 | 6 | 11 | 11 | 2 | 7 | 1 | 6 |
| IPSL | IPSL-CM6A-LR-INCA | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| KIOST | KIOST-ESM | 1 | 0 | 1 | 1 | 0 | 0 | 0 | 0 | 1 |
| MIROC | MIROC-ES2H | 3 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| MIROC | MIROC-ES2L | 31 | 10 | 10 | 30 | 10 | 0 | 0 | 1 | 10 |
| MIROC | MIROC6 | 50 | 1 | 50 | 50 | 3 | 1 | 1 | 1 | 50 |
| MOHC | HadGEM3-GC31-LL | 5 | 0 | 1 | 4 | 0 | 0 | 0 | 0 | 4 |
| MOHC | HadGEM3-GC31-MM | 4 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 4 |
| MOHC | UKESM1-0-LL | 16 | 5 | 16 | 16 | 16 | 5 | 0 | 5 | 5 |
| MPI-M | ICON-ESM-LR | 5 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| MPI-M | MPI-ESM1-2-HR | 10 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| MPI-M | MPI-ESM1-2-LR | 10 | 0 | 10 | 10 | 10 | 0 | 0 | 0 | 10 |
| MRI | MRI-ESM2-0 | 12 | 1 | 5 | 10 | 5 | 1 | 1 | 1 | 6 |
| NASA-GISS | GISS-E2-1-G | 46 | 6 | 11 | 30 | 27 | 6 | 6 | 11 | 11 |
| NASA-GISS | GISS-E2-1-G-CC | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| NASA-GISS | GISS-E2-1-H | 25 | 0 | 5 | 10 | 6 | 0 | 0 | 0 | 4 |
| NASA-GISS | GISS-E2-2-H | 5 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| NCAR | CESM2 | 11 | 0 | 5 | 6 | 8 | 0 | 0 | 0 | 5 |
| NCAR | CESM2-FV2 | 3 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| NCAR | CESM2-WACCM | 3 | 0 | 1 | 5 | 3 | 0 | 0 | 3 | 5 |
| NCAR | CESM2-WACCM-FV2 | 3 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| NCC | NorCPM1 | 30 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| NCC | NorESM2-LM | 3 | 0 | 1 | 13 | 3 | 0 | 0 | 0 | 1 |
| NCC | NorESM2-MM | 3 | 0 | 1 | 2 | 1 | 0 | 0 | 0 | 1 |
| NIMS-KMA | KACE-1-0-G | 3 | 0 | 3 | 3 | 3 | 0 | 0 | 0 | 3 |
| NIMS-KMA | UKESM1-0-LL | 3 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 |
| NOAA-GFDL | GFDL-CM4 | 1 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 1 |
| NOAA-GFDL | GFDL-ESM4 | 3 | 1 | 1 | 3 | 1 | 0 | 0 | 0 | 1 |
| NUIST | NESM3 | 5 | 0 | 2 | 2 | 0 | 0 | 0 | 0 | 2 |
| SNU | SAM0-UNICON | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| THU | CIESM | 3 | 0 | 1 | 1 | 0 | 0 | 0 | 0 | 1 |
| UA | MCM-UA-1-0 | 2 | 0 | 1 | 1 | 1 | 0 | 0 | 0 | 1 |

**Totals (monthly `Amon`, `pr`+`tas`):**

| experiment | models | members | largest single ensemble | models having `r1i1p1f1` |
|---|---:|---:|---:|---:|
| historical | 65 | 634 | 72 (EC-Earth3) | 55 |
| ssp119 | 14 | 145 | 51 | 9 |
| ssp126 | 46 | 332 | 57 | 34 |
| ssp245 | 47 | 469 | 96 | 36 |
| ssp370 | 41 | 326 | 57 | 32 |
| ssp434 | 9 | 76 | 50 | 5 |
| ssp460 | 7 | 26 | 7 | 5 |
| ssp534-over | 12 | 85 | 50 | 7 |
| ssp585 | 48 | 333 | 58 | 36 |

Model counts here are `(institution, source)` rows, so the duplicated
`MPI-ESM1-2-HR` and `UKESM1-0-LL` publications count twice; by distinct
`source_id` the historical count is 64 and ssp585 is 47.

**Scenario coverage.** Eight SSPs plus `historical`. Tier-1 (ssp126/245/370/585)
is well populated (41–48 models); tier-2 (ssp119/434/460/534-over) is thin
(7–14 models) and dominated by a few large-ensemble groups (EC-Earth3, CanESM5,
GISS-E2-1-G, CNRM-ESM2-1). 38 models provide all four tier-1 SSPs plus
historical — that is the practical ensemble ceiling for a balanced multi-scenario
overlay.

**Member labels are not uniformly `r1i1p1f1`.** 9 historical and 12 ssp585 models
have no `r1i1p1f1` at all. Physics/forcing variants: `r1i1p1f2` (CNRM-*,
MIROC-ES2L, UKESM1-0-LL, MCM-UA-1-0), `r1i1p1f3` (HadGEM3-GC31-*), `r1i1p2f1`
(CanESM5-CanOE), plus GISS-E2-1-G's `p3f1`/`p5f1`/`p5f2`. CESM2 and ACCESS-ESM1-5
have `f1` members but not at `r1i1` for some scenarios. A configuration pinned to
`members: [r1i1p1f1]` silently drops these models (§2).

**Two naming traps for the `institution/source` catalog key.** `MPI-ESM1-2-HR`
is published under three institutions (`MPI-M` historical, `DKRZ` and `DWD` for
SSPs), and `UKESM1-0-LL` under both `MOHC` and `NIMS-KMA`. The correct prefix
differs per scenario for the same model.

## 4. What the repo catalog exposes

**Before 2026-07-29** the catalog was a hand-curated subset: 8 scenario-level
entries with shared `model`/`member` placeholders, 69 resolved sources, all
pinned to `r1i1p1f1`. All 69 were verified live (no dead entries), but coverage
was 23 of 64 models for historical, 9 of 45 for ssp126, 12 of 46 for ssp245,
12 of 41 for ssp370, 9 of 47 for ssp585, 2 of 14 for ssp119, 1 each for
ssp434/ssp460, and nothing for ssp534-over. CanESM5, EC-Earth3, MIROC6,
MPI-ESM1-2-*, MRI-ESM2-0, UKESM1-0-LL, CNRM-*, HadGEM3, GISS-E2-1-G were all
unreachable.

**Now** `config/catalogs/cmip6_data.yml` is generated by
`dev/scripts/generate_cmip6_catalog.py` from the crawl in §1 and covers the store
in full: **289 entries, 2 426 sources** — every (model, scenario, member) in §3.

Structure, and why it is not the old shape: hydromt expands `placeholders` into
the full cross-product, so a scenario-level entry with a shared `member` list
would register combinations that do not exist, and the
`if entry in data_catalog.sources` guard (§2) would wave them through into the
empty-dataset branch — phantom members on the `member` coordinate. Instead there
is one entry per (model, scenario), keyed
`cmip6_<institution>/<source>_<scenario>_{member}`, whose `member:` placeholder
lists exactly the members that exist. The first entry carries a YAML anchor and
the rest pull the driver/adapter/metadata block in with a merge key, so the file
is ~3 900 lines rather than ~10 000.

Verified on regeneration: all 69 previously declared sources resolve under the
new catalog with byte-identical URIs, so existing configs are unaffected
(`pytest tests/test_cli.py`: 9 passed).

**Member selection is a config-side decision, and a global list cannot be
optimal.** `analyze_projections.smk` reads one `members:` list and passes
it to every model job, so the catalog alone does not remove the `r1i1p1f1` pin.
Two facts constrain the default:

1. A label a model does not publish is skipped cleanly. Verified against the
   real store: GFDL-ESM4 (`f1` only) and UKESM1-0-LL (`f2` only) run with the
   same list, and each output carries only its own member on the `member`
   coordinate — no all-NaN phantom slice.
2. But `get_change_climate_proj.py` raises `asymmetric hist/clim members` when a
   model's historical and future member sets differ (guard added at t260720d,
   D-MEM, precisely to stop xarray's inner join from silently shrinking the
   ensemble). Widening the list widens that mismatch.

Measured over the generated catalog, across the four tier-1 SSPs:

| `members:` | usable (model, scenario) pairs | models reached | pairs that raise |
|---|---:|---:|---:|
| `[r1i1p1f1]` (old) | 134 | 38 | 0 |
| `+ r1i1p1f2` | 148 | 42 | 18 |
| `+ r1i1p1f3` | 153 | 44 | 18 |
| `+ r1i1p2f1` | **157** | **45** | 18 |
| full 11-label union | 149 | 43 | 26 |

`snake_config.template.yml` therefore ships
`[r1i1p1f1, r1i1p1f2, r1i1p1f3, r1i1p2f1]` — the maximum. The wider union is
strictly worse: the extra physics variants (`p3f1`, `p5f1`, `r1i1000p1f1`, …)
create more asymmetry than coverage. The 18 raising pairs are CAMS-CSM1-0,
CanESM5 (tier-2 only), EC-Earth3 (ssp245), GISS-E2-1-G, GISS-E2-1-H,
IPSL-CM6A-LR (ssp434/460), MCM-UA-1-0, NorESM2-LM (ssp245); they fail loud with
the unshared labels named, and the fix is to narrow `members:` for that run.

A per-model `members:` mapping would remove the tradeoff entirely — it is a
config-schema plus script change, so it is recorded here for v2 rather than done.

The small `snake_config_model_test*.yml` fixtures stay pinned to `r1i1p1f1`:
their three models all publish it, and the gate is meant to stay minimal.

## 5. Overview B — temporal resolutions

### 5.1 Frequency × realm families present in the bucket

Store counts are index-derived (lower bounds; see §1) and are given only to show
relative size. `table_id` encodes frequency *and* realm.

| frequency | atmosphere | ocean / sea-ice | land | other | ≈ stores |
|---|---|---|---|---|---:|
| fixed (time-invariant) | `fx` | `Ofx` | — | `Efx`, `IfxGre` | 18.8 k |
| 1-hourly | — | — | — | `AERhr`, `E1hrClimMon` | 11 |
| 3-hourly | `3hr`, `CF3hr` | — | — | `E3hr` | 2.0 k |
| 6-hourly | `6hrLev`, `6hrPlev`, `6hrPlevPt` | — | — | — | 0.3 k |
| daily | `day`, `CFday`, `AERday` | `Oday`, `SIday` | — | `Eday`, `EdayZ` | 39.7 k |
| **monthly** | **`Amon`**, `AERmon`, `AERmonZ`, `CFmon` | `Omon`, `SImon` | `Lmon`, `LImon`, `ImonGre` | `Emon`, `EmonZ` | **415 k** |
| yearly / decadal | — | `Oyr`, `Odec` | — | `Eyr` | 37.0 k |
| climatology | `Aclim` | `Oclim`, `SIclim` | — | `Eclim` | 1.6 k |

(The eight rows sum to the index's 514 818 stores.)

The archive is overwhelmingly monthly. WF2 reads `Amon` only.

### 5.2 Monthly vs daily vs 3-hourly coverage (live crawl, `pr`+`tas`)

Monthly and daily columns require `pr`+`tas` on the same member; the 3-hourly
column counts members carrying `pr` **or** `tas` (few carry both).

| experiment | monthly models / members | daily models / members | 3-hourly models / members |
|---|---|---|---|
| historical | 65 / 634 | 48 / 456 | 33 / 98 |
| ssp119 | 14 / 145 | 10 / 58 | 1 / 1 |
| ssp126 | 46 / 332 | 36 / 197 | 17 / 41 |
| ssp245 | 47 / 469 | 38 / 314 | 17 / 41 |
| ssp370 | 41 / 326 | 33 / 231 | 18 / 60 |
| ssp434 | 9 / 76 | 5 / 14 | 0 / 0 |
| ssp460 | 7 / 26 | 3 / 8 | 0 / 0 |
| ssp534-over | 12 / 85 | 5 / 11 | 0 / 0 |
| ssp585 | 48 / 333 | 41 / 252 | 23 / 45 |

Restricted to the combination WF2 actually needs (historical + ssp245 + ssp585,
`pr`+`tas`, same model): **46 models at monthly, 35 at daily**; every daily model
is also available monthly. The 11 monthly-only models — those that would be lost
by moving WF2 to daily forcing: AWI-CM-1-1-MR, CAS-ESM2-0, CIESM, CNRM-CM6-1-HR,
CanESM5-CanOE, E3SM-1-1, FGOALS-f3-L, FIO-ESM-2-0, GISS-E2-1-G, GISS-E2-1-H,
MCM-UA-1-0.

3-hourly is a thin, uneven subset: 35 models publish something 3-hourly, but only
8 publish 3-hourly `pr` for historical (most 3-hourly entries are `tas`/`huss`),
and tier-2 SSPs have none. It is not a viable ensemble basis. 6-hourly is
negligible for `pr`/`tas`.

### 5.3 Variable availability by frequency (number of distinct models)

Monthly (`Amon`):

| variable | hist | ssp126 | ssp245 | ssp370 | ssp585 |
|---|---:|---:|---:|---:|---:|
| pr | 64 | 45 | 46 | 42 | 47 |
| tas | 64 | 45 | 46 | 41 | 47 |
| tasmax / tasmin | 41 | 32/33 | 33 | 29 | 34 |
| rsds | 57 | 40 | 40 | 36 | 42 |
| psl | 64 | 44 | 44 | 41 | 47 |
| sfcWind | 58 | 42 | 43 | 39 | 44 |
| huss | 60 | 41 | 43 | 39 | 44 |
| hurs | 54 | 38 | 39 | 37 | 40 |
| evspsbl | 62 | 42 | 42 | 41 | 44 |
| ps | 63 | 43 | 45 | 41 | 46 |
| mrro (runoff) | 0 | 0 | 0 | 0 | 0 |

Daily (`day`):

| variable | hist | ssp126 | ssp245 | ssp370 | ssp585 |
|---|---:|---:|---:|---:|---:|
| pr | 51 | 35 | 37 | 33 | 39 |
| tas | 49 | 37 | 38 | 35 | 39 |
| tasmax / tasmin | 43 | 35/34 | 36/35 | 30/29 | 37/36 |
| rsds | 42 | **1** | 32 | **1** | 33 |
| psl | 42 | 28 | 32 | 30 | 33 |
| sfcWind | 43 | 27 | 28 | 23 | 28 |
| huss | 40 | 27 | 32 | 28 | 31 |
| hurs | 38 | 28 | 31 | 29 | 33 |
| mrro | 36 | 10 | 12 | 10 | 12 |
| evspsbl, ps | 0 | 0 | 0 | 0 | 0 |

Two asymmetries matter for a PET-capable or weather-generator-facing WF2: daily
`rsds` is essentially absent for ssp126 and ssp370 (1 model each) though present
for ssp245/ssp585, and `evspsbl` has no `day` entry at all. The zero rows are
scoped to the three tables crawled: monthly runoff lives in `Lmon`, not `Amon`
(hence `mrro`'s zero row above), and `ps` has ~240 `CFday` stores across these
experiments in the index. Neither table was crawled, so a zero here means
"absent from `Amon`/`day`/`3hr`", not "absent from CMIP6".

## 6. Caveats the store listing cannot answer

- **Calendars differ across models** and are not in any index. Verified samples:
  `noleap` (GFDL-ESM4, INM-CM5-0, CanESM5), `360_day` (UKESM1-0-LL),
  `proleptic_gregorian` (MPI-ESM1-2-LR, EC-Earth3). This does **not** propagate
  through WF2's monthly aggregation the way the code comment at
  `get_stats_climate_proj.py:76-85` implies: that comment describes daily input,
  but the `Amon` input already has one value per month, so
  `resample("MS").sum("time")` is an identity on the values (it only relabels the
  timestamp to month start) and `precip` stays a monthly-mean rate in mm day⁻¹,
  not a monthly total. The real approximation is downstream:
  `get_change_climate_proj.py` builds annual values by summing (precip) or
  averaging (temp) the 12 monthly values **unweighted**, so month length is
  ignored. That makes the change factors calendar-*insensitive* by construction,
  at the cost of not being true annual totals — acceptable for a ratio, but it
  should be stated rather than assumed.
- **Time extents vary.** Historical is uniformly 1850-01…2014-12; SSPs are
  usually 2015-01…2100-12 but some run to 2300 (CanESM5 ssp585: 3 432 months).
  WF2's hardcoded `("2015-01-01","2100-12-31")` slice already normalises this.
- **Grid labels vary** (`gn` native, `gr`/`gr1` regridded). The catalog's `*/*`
  glob picks whatever exists; where both are published, resolution and grid
  differ between models. Not a failure, but an unstated heterogeneity in the
  ensemble.
- **Multi-member runs make the `member` dimension ragged** (CanESM5 50 members,
  INM-CM4-8 one), and `get_change_climate_proj_summary.py` /
  `plot_proj_timeseries.py` compute ensemble statistics across it. Every config
  to date has had exactly one member per model, so that path has never been
  exercised — whether percentiles should weight models or members is an open
  question for the v2 design, not something the catalog decides.
- The crawl covered `Amon`, `day`, `3hr` under `CMIP/historical` and
  `ScenarioMIP/ssp*` only. Other activities (`CMIP` piControl/1pctCO2/abrupt-4xCO2,
  `DAMIP`, `HighResMIP`, `LUMIP`, …) and other realms (`Lmon`, `Omon`) are in the
  bucket but out of WF2's scope.

## 7. Reproducing this inventory

The catalog generator (`dev/scripts/generate_cmip6_catalog.py`) performs the same
crawl and is the maintained entry point; run it with `--dry-run` for the counts
without rewriting the catalog. The wider crawl behind §5 (adding `day` and `3hr`)
was a one-off probe: list
`cmip6/CMIP6/{CMIP,ScenarioMIP}/*/*/{experiment}/*/{table}/` with
`gcsfs.GCSFileSystem(token="anon")` and pivot member counts per
`(institution, source, experiment)` filtered on the required variable set. Set
`GCSFS_EXPERIMENTAL_ZB_HNS_SUPPORT=false` first (same reason as
`get_stats_climate_proj.py:14`).

## 8. Glossary

### Projects and infrastructure

| term | meaning |
|---|---|
| CMIP6 | Coupled Model Intercomparison Project, phase 6 — the coordinated multi-model climate experiment protocol whose output this store holds. |
| activity_id | The MIP sub-project an experiment belongs to. WF2 touches two: `CMIP` (the DECK/historical core, which owns `historical`) and `ScenarioMIP` (which owns every `ssp*`). |
| Pangeo / `gs://cmip6` | The Google Cloud public mirror of CMIP6 as zarr stores, maintained by the Pangeo project. Path layout: `CMIP6/{activity}/{institution}/{source}/{experiment}/{member}/{table}/{variable}/{grid}/{version}/`. |
| `pangeo-cmip6.csv` | The published flat index of that bucket. Incomplete at store level (§1) — do not treat as authoritative. |
| zarr | Chunked array format used by the mirror; read remotely over `gcsfs` without downloading whole files. |
| hydromt | The data-catalog and model-building library WF2 reads the store through. A *catalog entry* maps a source name to a URI plus unit/rename handling. |
| placeholder | A hydromt catalog mechanism that expands `{...}` in an entry name and URI into concrete sources — as a **cross-product** over all listed values, which is why per-model member lists matter (§4). |

### Identifiers in a store path

| term | meaning |
|---|---|
| institution_id | The modelling centre that published the run (`NOAA-GFDL`, `MOHC`, `NCC`). Not fixed per model — see the `MPI-ESM1-2-HR` / `UKESM1-0-LL` trap in §3. |
| source_id | The climate model itself (`GFDL-ESM4`, `CanESM5`). "Model" in this document means source_id. |
| experiment_id | The forcing scenario the run applies: `historical` (1850–2014, observed forcings) or an SSP (2015 onward). |
| member_id / variant label | One ensemble member, formatted `r<N>i<N>p<N>f<N>`: **r**ealization (initial-condition perturbation), **i**nitialization method, **p**hysics variant, **f**orcing variant. `r1i1p1f2` is the first realization under an alternative forcing dataset. Models with no `r1i1p1f1` (CNRM-\*, UKESM1-0-LL, HadGEM3-\*) publish only a variant. |
| table_id | Frequency + realm of the output (§5.1). `Amon` = atmosphere, monthly. |
| grid_label | `gn` = native model grid; `gr`/`gr1` = regridded to a common grid by the publisher. |
| version | Publication date directory (`v20190726`). The catalog's `*/*` glob spans grid and version and assumes exactly one of each (§2). |

### Scenarios

| term | meaning |
|---|---|
| SSP | Shared Socioeconomic Pathway — the socioeconomic storyline. In `sspXYZ`, `X` is the SSP narrative and `YZ` the ~2100 radiative forcing in W m⁻² × 10. |
| ssp126 / ssp245 / ssp370 / ssp585 | The four tier-1 SSPs: strong mitigation (~1.9–2.6 W m⁻²), middle-of-the-road, regional rivalry / high aerosol, fossil-fuelled high end (8.5 W m⁻²). Best-populated, hence the practical ensemble basis. |
| ssp119 / ssp434 / ssp460 | Tier-2: 1.5 °C-compatible, and two intermediate forcing levels. Thin (7–14 models). |
| ssp534-over | Overshoot — follows ssp585 to ~2040 then declines to 3.4 W m⁻². Absent from the pre-2026-07 catalog. |
| historical | The 1850–2014 forced run every scenario branches from; WF2's reference period. |
| tier-1 / tier-2 | ScenarioMIP's own priority split, which is why coverage differs so sharply between them. |

### Variables (CMIP6 name → WF2 name)

| CMIP6 | WF2 | quantity | units in store → after adapter |
|---|---|---|---|
| `pr` | `precip` | precipitation flux | kg m⁻² s⁻¹ → mm day⁻¹ (`unit_mult: 86400`) |
| `tas` | `temp` | near-surface (2 m) air temperature | K → °C (`unit_add: -273.15`) |
| `rsds` | `kin` | surface downwelling shortwave radiation | W m⁻² |
| `psl` | `press_msl` | sea-level pressure | Pa → hPa (`unit_mult: 0.01`) |
| `tasmax` / `tasmin` | — | daily max / min near-surface temperature | K |
| `sfcWind` | — | near-surface wind speed | m s⁻¹ |
| `huss` / `hurs` | — | near-surface specific / relative humidity | 1 / % |
| `evspsbl` | — | evaporation including sublimation | kg m⁻² s⁻¹ |
| `mrro` | — | total runoff (a land-realm variable, hence absent from `Amon`) | kg m⁻² s⁻¹ |
| `ps` | — | surface air pressure | Pa |

Only the first four are reachable through the catalog today; the rest are listed
because §5.3 reports their coverage as input to a PET- or weather-generator-facing
v2.

### Terms specific to this workflow

| term | meaning |
|---|---|
| WF1 / WF2 / WF3 | The three Snakemake workflows: model creation, climate projections, climate experiment (the stress test). |
| change factor | The per-(model, scenario, horizon) shift between the reference and future window — multiplicative for precipitation, additive for temperature. WF2's product. |
| plausibility overlay | The role those change factors play: they locate the WF3 perturbation grid in projection space. They never drive stress-test runs. |
| horizon | A named future window (`near: [2030, 2060]`, `far: [2070, 2100]`) over which a change factor is computed. |
| ensemble | Here, the set of (model, scenario, member) combinations a run aggregates over. "Ragged" means members differ in number per model (§6). |
| response surface | WF3's map of system performance over the temperature × precipitation perturbation grid; the surface the overlay is drawn on. |
