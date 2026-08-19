---
title: "Ship a sample dataset bundle so a user needs no Deltares P: drive"
type: todo-item
status: backlog
branch: feat/scope-shipped-sample-dataset
effort: 2
area: distribution / sample data
queue:
created: 2026-08-19
updated: 2026-08-19
---

> [!note] Overview
> **What** — Package a bbox-clipped sample of the hydromt tree plus
> WF2-cache-compatible CMIP6 slices as a downloadable bundle, and route a sample
> project config at it, so all four workflows run with no Deltares `P:` access
> and no cloud credentials.
> **Why** — Today a new user cannot run anything without `P:` drive rights. The
> staging tools that produce the bundle already exist; what is missing is
> packaging, routing and provenance.
> **Effort** — Large. One new builder, one new fetcher, one new config, a
> release process, and a source-completeness audit that has to happen first.

**Scoped 2026-08-19.** Design and rulings:
`dev/working/2026-08-19_shipped-sample-dataset/design.md`.
Input contract, already authored: `dev/scripts/sample_bundle.yml`.

## The promise, stated honestly

"No internet" is unreachable — `pixi install` solves from conda-forge,
`pixi run install` pulls `weathergenr` via `remotes::install_github`, and Julia
comes from juliaup. The deliverable is **no credentials and no `P:` drive: one
anonymous download, then every pipeline run offline.** Do not let the item be
judged against a promise it cannot keep.

## Progress

- [ ] **Source-completeness audit — do this first, it sizes everything else.**
      Cross-check every catalog name reachable from `spatial_sources` defaults,
      `shared.clim_historical`, `analyze_climate.candidate_sources` and the two
      `config/defaults/*.yml` against what `stage_data.yml` actually stages. The
      list in that file is what someone staged while iterating, not a proven
      set — see the waterbody finding below.
- [ ] `dev/scripts/build_sample_bundle.py` — reads `sample_bundle.yml`, drives
      `stage_data.py` and `stage_cmip6.py`, writes the self-locating catalog,
      the pinned crawl products, `BUNDLE.md` and the checksum. **The two stagers
      are not peers: stage spatial → derive `region.geojson` → stage CMIP6.**
      The polygon is delineated by hydromt from `merit_hydro_ihu` +
      `merit_hydro_index` and the slices are fingerprinted on it, so the other
      order fingerprints every slice on a polygon the sample config does not
      produce — a silent full re-fetch for the user.
- [ ] **Empty-clip fidelity.** `stage_data.py:791` writes NO FILE when a vector
      clip has zero features, which turns rule 1.08's `ok` (source readable,
      nothing in this basin) into `skipped` (source absent). The builder must
      write a valid empty GeoPackage instead, or the sample's sentinel
      misreports which methods ran.
- [ ] `scripts/fetch_sample_data.py` — user-facing (three-homes rule: a user
      runs it). Downloads the release asset, **verifies `sample_data.sha256`
      before unpacking**, unpacks to a gitignored top-level `sample_data/`.
- [ ] `test_case/snake_config_sample.yml` — derived from `sample_bundle.yml`,
      selecting the bundle's catalog via `project.data_sources`.
- [ ] `.gitignore` entry for `sample_data/`, and the tracked
      `sample_data.sha256`.
- [ ] Docs: install guide gets the sample path; `BUNDLE.md` carries provenance,
      per-source licences and what was cut.
- [ ] **Acceptance test** — fresh checkout + fetched bundle + `pixi run
      install`, then all four workflows on the sample config **with the network
      disabled**, from the PRIMARY checkout (a slot worktree gets its own
      `.snakemake`).

## Owner rulings (2026-08-19)

| Question | Ruling | Rejected |
|---|---|---|
| Hosting | **GitHub Release asset** | Zenodo DOI; Git LFS (every clone pays, billed bandwidth); Docker as primary |
| Waterbody sources | **Stage them clipped** | accept `skipped`; audit-first |
| Coverage | **All four workflows, rapid-scale** | wf0+wf1 only; a second baseline-scale variant |

Release assets are mutable and deletable, so provenance cannot come from the
host: `sample_data.sha256` is tracked **in the repo** and the fetcher verifies
against it. Mirroring to Zenodo later needs no change on this side.

## Five things that will bite, all verified by reading the code

1. **Today's staged root ships a silently degraded WF1** — but check the
   ruling's scope, `rgi` is the exception (see the design note: an equatorial
   basin's glacier clip is empty, so listing it stages nothing). Rule 1.08
   (`add_reservoirs_lakes_glaciers`) reads `hydro_lakes`, `hydro_reservoirs` and
   `rgi`; none is in `stage_data.yml`.
   `setup_reservoirs_lakes_glaciers.py:121` catches `NoDataException` /
   `FileNotFoundError` and records `status: skipped`. The `test_local` fixture's
   sentinel says `ok` for all three — it predates the switch to a staged root.
   Nothing fails; the sample just quietly does less.
2. **Pre-seeding WF2's `raw/` is the designed path.** Rule 2.04's output is
   wrapped in `update(...)` so `Job.prepare()` cannot delete it, and
   `fetch_gcm_raw.py:412` revalidates the digest before any network call.
3. **The raw digest is coupled to two GENERATED files.** `raw_components`
   includes the store-index pins, and `cmip6_data.yml` /
   `cmip6_store_index.json` are crawl products. Regenerate them without
   rebuilding the bundle and every shipped slice goes stale → ~1142 s per source
   re-fetch → the offline promise dies with no error. Pin both into the bundle —
   which works for free, because `analyze_projections.smk:312` derives
   `STORE_INDEX` as a SIBLING of `data_sources_climate`. There is no second
   config key; the price is that the pinned index must be named exactly
   `cmip6_store_index.json` and sit beside the catalog.
4. **Catalog routing has a silent trap.** Ship the sample catalog inside the
   bundle with **no** `root:`/`roots:` — hydromt falls back to `dirname(yml)`,
   so it can only ever point at the bundle. Do **not** add the sample root to
   the shipped `deltares_data.yml`'s `roots:`: `_determine_catalog_root` takes
   the first path that `exists()`, so a user with the bundle would silently get
   the clipped Gabon-extent tree for their own basin.
5. **`dev/scripts/stage_cmip6.yml` is stale.** It still says
   `buffer_degrees: 1.0` and cites `REGION_BUFFER_DEGREES`; the rename to
   `buffer_cells` / `REGION_BUFFER_CELLS` landed in t2608182238.
   `stage_cmip6.py`'s `load_config` uses `setdefault` and does not reject
   unknown keys, so the line is inert and agrees only by coincidence of value.
   Fix it with the builder.

## Measured

```
C:\data\wflow_global\hydromt   98.0 MB / 827 files   (era5 zarr 48.2, chirps 25.7, soilgrids 19.9)
CMIP6 raw slices               0.6 MB / 9 files
```

Re-staged at 2000-2016 the bundle lands near **45-55 MB**. Time range dominates
cost, not bbox.

## Explicitly NOT in scope

Arbitrary user basins (the CMIP6 slices are region-fingerprinted, so they
cache-hit only for the sample basin's exact polygon), a second sample basin, a
baseline-scale variant, the `deltares_data_linux.yml` v0→v1 rewrite beyond the
sample catalog, and Docker as the primary channel.

## Refs

- `dev/working/2026-08-19_shipped-sample-dataset/design.md` — the scope, the
  rulings, and the evidence behind each finding above.
- `dev/scripts/sample_bundle.yml` — the parameter template this item builds from.
- [[t2608191733a-rename-snake-config-yml-to-project-config-yml-repo-wide]] —
  decides whether the sample config lands as `snake_config_sample.yml` or
  `project_config_sample.yml`. Land the rename FIRST if both are wanted; adding
  a new `snake_config_*` file only widens that rename.
