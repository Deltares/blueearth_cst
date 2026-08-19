# Shipped sample dataset — scope

**Status:** scoped 2026-08-19, not implemented. Owner rulings recorded below.
**Input contract:** `dev/scripts/sample_bundle.yml` (authored with this note).

## The promise, stated honestly

"No internet" is unreachable: `pixi install` solves from conda-forge,
`pixi run install` pulls `weathergenr` through `remotes::install_github`, and
Julia comes from juliaup. What the bundle delivers is **no credentials and no
Deltares `P:` drive — one anonymous download, then every pipeline run offline.**

Scope the work against that promise. Judged against "no internet at all" it
fails on the first line of the install guide.

## Why this is mostly packaging, not new capability

Four things already exist and do the hard parts:

| Piece | What it already does |
|---|---|
| `dev/scripts/stage_data.py` | mirrors a bbox-clipped subset of the hydromt tree — **the bundle producer** |
| `dev/scripts/stage_cmip6.py` | writes WF2-cache-compatible CMIP6 slices, digest-stamped, explicitly built for reuse across projects |
| `config/catalogs/deltares_data.yml` | already points at a staged local root (`C:\data\wflow_global\hydromt`), not `P:` |
| hydromt `meta.roots` | first path that `exists()` wins (`data_catalog.py:744`); with **no** `root`/`roots`, the root falls back to `dirname(yml)` |

`config/basemap/` is the precedent for vendoring a static asset, with its
reasoning already written down.

## Measured sizes

```
C:\data\wflow_global\hydromt   98.0 MB / 827 files
  meteo/era5_daily.zarr        48.2 MB   (staged 1970-2020 — far wider than any config needs)
  meteo/chirps_africa_daily    25.7 MB   (staged 1990-2020)
  soil/soilgrids_v1.0          19.9 MB
  topography, landuse, hydro    4.2 MB
CMIP6 raw slices (test_local)   0.6 MB / 9 files
```

Re-staged at the sample window (2000-2016, clearing `min_historical_years`
= 16) the bundle lands near **45-55 MB**. Time range dominates cost, not bbox —
`stage_data.yml`'s own header says so, and the chunk geometry is why.

## Findings that change the work

**1. Today's staged root is not a complete sample.** `stage_data.yml`'s dataset
list is what someone staged while iterating, not a proven-sufficient set. Rule
1.08 (`add_reservoirs_lakes_glaciers`) reads three catalog entries absent from
it:

```
hydro_lakes       hydrography/lakes/lake-db.gpkg
hydro_reservoirs  hydrography/reservoirs/reservoir-db.gpkg
rgi               hydrography/rgi/rgi.gpkg
```

`setup_reservoirs_lakes_glaciers.py:121` catches `NoDataException` and
`FileNotFoundError` and records `status: skipped`. The `test_local` fixture's
sentinel records `ok` for all three — it was built when the catalog still
reached the full source. **On today's staged root the sample would silently run
a degraded WF1, and nothing fails.** Ruled: stage them clipped.

**2. Pre-seeding WF2's `raw/` cache is the designed path, not a hack.** Rule
2.04's output is wrapped in `update(...)` so `Job.prepare()` cannot delete it,
and `fetch_gcm_raw.py:412` revalidates the digest *before* any network call.
Verified by reading both; a pre-seeded slice makes the rule a no-op.

**3. The raw digest is coupled to two generated files.** `raw_components`
includes the store-index pins, and `cmip6_data.yml` / `cmip6_store_index.json`
are crawl products AGENTS.md tells people to regenerate. Regenerate them
without rebuilding the bundle and every shipped slice's digest goes stale →
`fetch_gcm_raw` misses cache → ~1142 s per source → the offline promise dies
with no error message. **Pin both into the bundle** and have the sample config
read the pinned copies.

**4. The slices are basin-locked, by design.** The digest covers
`region_fingerprint`, so they cache-hit only for the sample basin's exact
`region.geojson`. Fine for a fixed demo; it must be stated, not left implicit.

**5. Stale reference, found in passing.** `dev/scripts/stage_cmip6.yml` still
carries `buffer_degrees: 1.0` and cites `REGION_BUFFER_DEGREES
(analyze_projections.smk:402)`. The rename to `buffer_cells` /
`REGION_BUFFER_CELLS` landed in t2608182238. `stage_cmip6.py` reads
`buffer_cells` and its `load_config` uses `setdefault` without rejecting
unknown keys, so the line is silently inert and agrees only by coincidence of
value. Fix with the builder.

## Routing — the recommendation and the trap

Ship `deltares_data_sample.yml` **inside the bundle** with no `root:` and no
`roots:`. hydromt then resolves the root to the catalog file's own directory,
so it can only ever point at the bundle, on any OS, wherever the user unpacked
it. The sample project config selects it explicitly via `project.data_sources`.

**Do not** append the sample root to the shipped `deltares_data.yml`'s `roots:`
list. `_determine_catalog_root` takes the first path that exists, so any user
who has the bundle would silently get the clipped Gabon-extent tree for a real
basin.

Free win: one self-locating v1 catalog removes the windows/linux fork for the
sample case. (`deltares_data_linux.yml` is still v0 — `kwargs`,
`driver: vector`, singular `meta.root`.)

## Owner rulings (2026-08-19)

| Question | Ruling | Rejected |
|---|---|---|
| Hosting | **GitHub Release asset** | Zenodo DOI; Git LFS (every clone pays, billed bandwidth); Docker as primary |
| Waterbody sources | **Stage them clipped** | accept `skipped`; audit-first |
| Coverage | **All four workflows, rapid-scale** | wf0+wf1 only; a second baseline-scale variant |

Release assets are mutable and deletable, so provenance cannot come from the
host: pin `sample_data.sha256` **in the repo** and have the fetcher verify
before unpacking. That buys Zenodo's tamper-evidence without the deposition
step, and the same bundle can be mirrored to Zenodo later without changing
anything on this side.

## Layout, and why each piece sits where it does

Three homes, split by invocation model (AGENTS.md), applied:

- **producers** — `dev/scripts/stage_data.py`, `stage_cmip6.py`,
  `build_sample_bundle.py`: maintainer-only, never part of a run.
- **fetcher** — `scripts/fetch_sample_data.py`: a user runs it, so it is
  user-facing.
- **unpack target** — a gitignored top-level `sample_data/`. Not under
  `test_case/`, whose un-ignore patterns are already three deep; a fourth to
  keep 50 MB of binaries out of git is a trap waiting to be sprung.
- **run config** — `test_case/snake_config_sample.yml`. The `snake_config_`
  prefix is mandatory or the file is silently untracked. See the pending
  rename to `project_config_`.

## Acceptance test — the only check that proves the claim

Fresh checkout + fetched bundle + `pixi run install`, then all four workflows on
the sample config **with the network disabled**, run from the primary checkout
(not a slot worktree — `.snakemake` divergence).

## Cut by YAGNI

Arbitrary user basins (the CMIP6 cache is region-fingerprinted; a second basin
doubles the bundle for no demo value), a second sample basin, a baseline-scale
variant, the `deltares_data_linux.yml` v0→v1 rewrite beyond the sample catalog,
and Docker as primary channel.

## Open item for the implementer

The **source-completeness audit**. The waterbody gap was found by inspection;
the manifest still needs a systematic cross-check of every catalog name
reachable from `spatial_sources` defaults, `clim_historical`,
`candidate_sources` and the two `config/defaults/*.yml` against what
`stage_data.yml` stages. That audit is what finally sizes the bundle.
