Status: accepted (§1–7, implemented); **proposed** (§8–11, the vector-foundation split)
Date: 2026-08-02
Deciders: Ümit Taner
Consulted: gabon_0108 run (2026-08-02) — geometry comparison showing
           `store_region.geojson` and `spatial/geoms/basins.geojson` are the
           same polygon; `Snakefile_climate_projections` §2.11 comment block
           (design D2/A1, "the accepted, stated price"); `series_identity.py`
           module docstring (design D9 / ext2-01)
Supersedes: none
Revisions:
  - 2026-08-02: initial record, accepted and implemented in the same session
    (fast-track, no staged review). One `delineate_region` rule, declared
    identically in all three workflows from `snake_utils.region_spec`, produces
    `spatial/geoms/region.geojson`; rule 1.02 and the climate store consume it
    instead of delineating; WF2 drops the climate-store producer entirely;
    `store_region.geojson` is retired and the store's extent moves into
    `extract_historical.nc` attributes.
  - 2026-08-06: **subject broadened from the region polygon to the shared
    spatial foundation**, and the title with it. Adds §8–11 (proposed): split
    `prepare_spatial_maps` at its thematic-raster seam so the vector layers —
    basins, subbasins, catchments, rivers, locations, registry — become a third
    shared spec declared in all three workflows, letting WF2 and WF3 consume
    basin and subbasin boundaries without dragging in the raster stack. The
    "point at `basins.geojson`" alternative rejected in the original record is
    marked revisited, because this split removes its disqualifying factor.

# ADR 0003 — Spatial artifacts delineated once per project, shared across workflows

### Context

`shared.basin.region` is delineated **twice** per project, by two rules that
never compare notes:

| Caller | Rule | Output |
|---|---|---|
| `blueearth_cst/spatial/products.py::_region_geometry` | 1.02 `prepare_spatial_maps` | `spatial/geoms/basins.geojson` (exploded, with ids) |
| `blueearth_cst/climate_analysis/extract_historical_climate.py::delineate_store_region` | 1.10 / 2.11 / 3.02 `extract_climate_grid` | `climate_historical/<key>/store_region.geojson` |

Both call hydromt's `parse_region_basin` with the same `shared.basin.region`,
the same catalog, and the same `hydrography`/`basin_index` entry names. Neither
takes `clim_source`. Measured on the gabon_0108 project (2026-08-02), the two
outputs are the same polygon:

```
store_region ∪  ==  basins ∪            → True
bounds        [9.65833, 0.35, 9.85833, 0.48333]  (identical)
```

So `store_region.geojson` is written once **per store key** — once for era5,
again for chirps, again for any other dataset or window — and every copy holds
what `spatial/geoms/` already holds.

The duplication is not the expensive part. `Snakefile_climate_projections`
declares the whole climate-store producer **purely to obtain that polygon**:

> WF2 declares this producer to obtain `store_region.geojson` — a model-free
> delineated polygon … The gridded extraction it also produces is NOT read by
> wf2 v2.0 (design N7); that cost is the accepted, stated price of A1.

A projections-only run therefore pays for a full multi-decade climate
extraction to learn a basin outline it could have read from a 3 kB file. That
price was accepted because, at the time, the store producer was the only
model-free source of the polygon. It no longer has to be.

The reason the store producer exists in this shape at all is worth keeping in
view: R07 B1 made the climate store **model-free** on purpose, replacing
derivations that read the built model's `staticmaps.nc` (wf1) or
`staticgeoms/region.geojson` (wf3). Any replacement must preserve that — the
region must come from config plus catalog, never from a hydrology build.

### Context — second pressure (2026-08-06)

The region polygon is now shared; **nothing else spatial is.** WF2 and WF3
declare `delineate_region` and no other `spatial/` rule, and neither workflow's
scripts read a vector layer — `export_wflow_results.py` and
`plot_proj_timeseries.py` contain no reference to basins, subbasins or the
location registry. Figures and metrics in both workflows want basin and subbasin
boundaries: a context map beside the change-factor plots, and the option of
subbasin-resolved indicators instead of today's basin averages.

The obvious move — declare `prepare_spatial_maps` in all three workflows, as
`delineate_region` is declared — repeats the trade this record removed. That rule
produces nine outputs across two separable jobs
(`spatial/products.py::prepare_spatial_products`):

| job | outputs | needed by |
|---|---|---|
| **vectors + hydrography** — read the hydrography raster, derive flow direction and accumulation, delineate parent basins, snap gauges, partition subbasins | `geoms/{basins,subbasins,catchments,rivers,locations}.geojson`, `location_registry.csv` | WF1, and now WF2 + WF3 |
| **thematic raster stack** — `_thematic_maps` reads and reprojects LULC (`vito`), LAI (`modis_lai`) and soil (`soilgrids`) onto the grid | folded into `spatial_maps.nc` | WF1 only — it exists to parameterise Wflow |

A projections-only run would resample three global raster sources to draw a
subbasin outline. That is the same shape as the cost this record repaid, one
level down: WF2 paying for a large derived product to obtain a small geometric
one. The seam is clean in the code — `_thematic_maps` is a single call and
nothing in the vector path depends on it.

Owner ruled 2026-08-06 that WF2 and WF3 need **no DEM or raster layer**, only the
boundaries.

### Decision

Introduce **one region artifact per project**, produced by one small rule that
all three workflows declare identically.

1. **`snake_utils.region_spec(project_dir, model_region, hydrography,
   basin_index, data_sources)`** returns a `RegionSpec` — `script`, `inputs`,
   `outputs`, `params` — exactly mirroring `climate_store_spec`. Its single
   output is:

   ```
   <project_dir>/spatial/geoms/region.geojson
   ```

   It sits in `spatial/geoms/` beside `basins.geojson`, `catchments.geojson`,
   `locations.geojson` and the rest, because that is where this project keeps
   the vector description of where the model is.

2. **Rule `delineate_region`** — `1.01b` / `2.03b` / `3.01b`, following the
   existing `3.00b` letter-suffix precedent — runs
   `blueearth_cst/spatial/delineate_region.py`, which calls `parse_region_basin`
   and writes the GeoJSON. Model-free by construction: its only input is the
   data catalog, its only params are the region spec and the two catalog entry
   names. The three declarations are byte-identical except `message`, `log`, and
   `benchmark`, enforced the same way `tests/test_climate_store_contract.py`
   enforces the store's.

3. **Rule 1.02 consumes it.** `prepare_spatial_maps` takes
   `region.geojson` as a declared input and reads the polygon instead of calling
   `parse_region_basin` itself. `_region_geometry` keeps every validation it
   performs today (non-empty, CRS present, explode, non-overlap) — it stops
   delineating, not checking.

4. **The climate store consumes it.** `climate_store_spec` gains
   `region_geojson` as an **input** and loses it as an **output**;
   `extract_historical_climate.py` reads the polygon's bounds instead of
   delineating. `delineate_store_region` moves to
   `blueearth_cst/spatial/delineate_region.py` as the shared producer's
   implementation.

5. **WF2 stops declaring the climate store.** Rule 2.11 `extract_climate_grid`
   is removed. WF2 declares `delineate_region` and reads `region.geojson`
   directly.

6. **`store_region.geojson` is retired.** The store's extent provenance moves
   into `extract_historical.nc` global attributes — `region_geojson_sha256`,
   `region_bbox`, and `region_source` — so the extraction still records the
   extent it was cut to, by content rather than by an adjacent copy.

7. **`series_identity` repoints.** The polygon content fingerprint
   (`polygon_sha256`, design D9 / ext2-01) reads `spatial/geoms/region.geojson`.
   The fingerprint stays **content-based**: a catalog change can still rewrite
   the polygon while `shared.basin.region` is unchanged, and that is exactly the
   case a specification-based digest misses.

**§8–11 are PROPOSED, not implemented.** They extend the same pattern from the
region polygon to the vector foundation.

8. **Split `prepare_spatial_maps` at the thematic seam**, into two rules:

   - **`delineate_spatial_units`** — region polygon + hydrography catalog +
     `shared.basin.gauge_points` → `geoms/{basins,subbasins,catchments,rivers,
     locations}.geojson` and `location_registry.csv`. Model-free and
     engine-neutral, exactly as the whole rule is today.
   - **`prepare_spatial_maps`** (retained name and job) — consumes those, adds
     the thematic layers → `spatial_maps.nc`, `spatial_catalog.yml`,
     `spatial_report.yml`. **WF1 only.**

   The split is a decomposition of one existing function, not new logic: the
   vector half is `prepare_spatial_products` up to and including
   `_delineate_spatial_units`, the raster half is `_thematic_maps` onward.

9. **`snake_utils.spatial_units_spec(...)`** returns a `SpatialUnitsSpec` —
   `script`, `inputs`, `outputs`, `params` — mirroring `region_spec` and
   `climate_store_spec`. All three workflows declare `delineate_spatial_units`
   from it, byte-identical but for `message`/`log`/`benchmark`.

10. **WF2 and WF3 consume the vectors as declared inputs** of the figure and
    metric rules that use them. *Which* rules is deliberately left open — see
    *Open questions*; making the artifacts reachable is this decision, using them
    is the next one.

11. **`automatic_subbasins.max_count` gains a per-basin spelling.** A gauge-free
    project must be able to ask for subbasin == basin. Today the ceiling is
    **global across parents** and `allocate_automatic_subbasin_budgets` raises
    when `len(parent_areas) > max_count`, so "one per basin" is expressible only
    by setting `max_count` to a parent count the config author cannot know before
    delineation runs. The default of 20 also means a gauge-free project currently
    gets twenty automatic subbasins, not one.

### Consequences

*Positive*

- A projections-only run no longer triggers a climate extraction. On a config
  whose store is cold, WF2's cost drops from a multi-decade multi-variable
  extraction to one delineation — observable as rule 2.11 disappearing from
  `snakemake -n` output for `Snakefile_climate_projections`.
- `parse_region_basin` is called **once** per project instead of twice, and the
  two callers can no longer disagree. Today nothing checks that they don't; the
  agreement is coincidence maintained by both reading the same config.
- One region file per project instead of one per store key. A project with era5
  and chirps stores holds one polygon, not three.
- The region becomes available to any future rule without dragging either
  `spatial_maps.nc` or a climate extraction into the DAG.

*Negative*

- WF2 and WF3 gain a dependency on a `spatial/` artifact; today they reference
  `spatial/` nowhere. The dependency is on one small model-free rule, not on
  1.02's raster products, but it is new coupling and it is real.
- `climate_historical/<key>/` stops being self-describing as a directory. Its
  extent is recoverable from the netCDF attributes rather than from a sibling
  file — better provenance (content-addressed, and it travels with the data),
  but it is no longer visible by listing the directory.
- Existing projects carry a stale `store_region.geojson` that nothing reads.
  Harmless, and not deleted automatically: removing a file a previous run wrote
  is the owner's call, and `dev/scripts/prune_series_cache.py` is the precedent
  for that being explicit.
- The WF2 series cache invalidates once, because `polygon_sha256` now reads a
  different file. The bytes are the same polygon but not the same GeoJSON
  serialization, so the digest changes and every cached series re-derives on the
  next run. One-time, and loud rather than silent — `series_identity`'s backstop
  raises on mismatch rather than reusing.

*Neutral*

- Rule count rises by one per workflow and falls by one in WF2 (2.11 removed),
  so WF2 is net zero and WF1/WF3 each gain a rule that runs in seconds.
- The store key (`<clim_source>_<window>`) is unchanged, so store reuse across
  experiments (P3-1 §4) behaves exactly as before.

#### Consequences of §8–11

*Positive*

- WF2 and WF3 can draw or aggregate on basin and subbasin boundaries with **no
  built model and no thematic raster read**. Observable as
  `snakemake -n -s Snakefile_climate_projections` listing
  `delineate_spatial_units` while `vito`, `modis_lai` and `soilgrids` appear in
  no job's inputs.
- `rivers.geojson` and `location_registry.csv` come with them, so a WF2 context
  map and a WF3 station-labelled indicator table need no further plumbing.
- The alternative this record originally rejected — point a consumer at
  `basins.geojson` — becomes viable, because its disqualifying factor was the
  raster stack behind rule 1.02 and §8 removes it.

*Negative*

- A **third** shared spec, and therefore a third byte-identity contract test
  beside `test_region_spec.py` and `test_climate_store_contract.py`. The
  duplication-by-construction cost of this pattern is now paid three times.
- WF2 gains a hydrography raster read (`buffer=10`) plus flow-direction and
  accumulation derivation that it does not pay today. Much cheaper than the
  thematic stack but **not free, and not yet measured** — see *Open questions*.
- `shared.basin.gauge_points` becomes a rerun trigger for WF2 and WF3, which
  reference it nowhere today.
- WF1 gains a rule. Interacts with the renumbering in `dev/followups.md`
  `[R10-5]`: land the split first or the numbers move twice.

*Neutral*

- `spatial_catalog.yml` currently enumerates all five geoms, the registry and
  `spatial_maps`. After the split its producer must be chosen — either the vector
  rule writes a vector-only catalog that the raster rule extends, or the catalog
  stays whole in the raster half and WF2/WF3 read the geojsons by path.
- Migration: the two rules and their call sites land in **one commit**. Splitting
  a `script:` module into two entry points leaves the tree un-runnable between a
  bare move and its reference rewrite.

### Alternatives considered

- **Point the climate store at `spatial/geoms/basins.geojson`.** No new artifact
  at all — the polygon already exists under that name. **Rejected**: it makes
  every workflow depend on rule 1.02, whose real product is `spatial_maps.nc`
  and the raster stack behind it. WF2 would then run the spatial foundation to
  get a polygon — the same trade it makes today with the climate store, moved
  rather than removed. `basins.geojson` is also the *exploded, id-carrying*
  form, which is a P1 domain product, not a plain extent.

  > **Revisited 2026-08-06.** The first objection no longer holds: §8 splits the
  > raster stack out, so depending on the vector layers no longer means depending
  > on `spatial_maps.nc`. The **second** objection stands and is why §8 keeps
  > `region.geojson` as a separate artifact rather than folding it into
  > `basins.geojson` — a plain extent and an exploded id-carrying product are
  > different things, and the climate store wants the former.
- **Add `region.geojson` beside `basins.geojson` with no rule change**, written
  by 1.02 as an extra output, and have the store read it. Cheaper, but leaves
  WF2 and WF3 depending on 1.02 for the same reason as above, and leaves the
  duplicate delineation in place whenever the store runs first.
- **Keep `store_region.geojson` as a copy** of the shared artifact, so the store
  directory stays self-describing. **Rejected** as the wrong fix for the loss it
  addresses: a copy is a second source of truth that can drift, and the store's
  extent belongs *in* the extraction, not beside it. The netCDF attributes carry
  the same information and cannot be separated from the data.
- **Do nothing.** The duplication is harmless in itself — both callers read the
  same config and produce the same polygon. Preferred only if WF2's extraction
  cost were not real; it is, and it is stated in the code as an accepted price.

#### Alternatives to §8–11

- **Declare `prepare_spatial_maps` unsplit in all three workflows.** One more
  shared spec, no decomposition, symmetric with `delineate_region`. **Rejected**:
  every projections-only run would resample `vito`, `modis_lai` and `soilgrids`
  to obtain vector boundaries — the cost this record exists to have removed, at
  larger scale. Preferred if the thematic read were cheap, or if WF2/WF3 wanted
  the DEM; the owner ruled on 2026-08-06 that they do not.
- **Declare the geojsons as plain inputs in WF2/WF3, with no producing rule.**
  The smallest possible change. **Rejected**: WF2 stops bootstrapping itself. It
  runs today from a cold project because it declares the rules that build what it
  reads; under this option it fails with missing inputs unless WF1 ran first.
  Preferred only if WF1-first were already mandatory — which is what the next
  alternative would make it.
- **Move the shared rules into a preparation workflow (`WF0`).** `snapshot_config`,
  `delineate_region`, `delineate_spatial_units` and `extract_historical_climate`
  become a fourth Snakefile that runs before the other three, deleting the
  shared-spec duplication entirely — one declaration each instead of three plus a
  byte-identity test. **Deferred, not rejected**: it is a larger architectural
  change (a fourth entry point against `AGENTS.md`'s stated three, plus
  `run_workflows.py`, its `workflows.<name>.enabled` schema, `tests/test_cli.py`,
  `plot_workflow_dag.py`, the R9 path map and `README.rst`), and it removes each
  workflow's ability to bootstrap itself. It is also **not blocked by this
  decision**: WF0 would carry the vector half and leave the raster half in WF1,
  which is the same seam §8 cuts, so §8 is a prerequisite either way. Raise it as
  its own record when the duplication cost of three shared specs is felt.

### Validation

1. `tests/test_region_spec.py` — the shared spec's shape, and that the three
   workflow declarations of `delineate_region` differ only in
   `message`/`log`/`benchmark` (mirroring `test_climate_store_contract.py`).
2. `tests/test_delineate_region.py` — the producer writes a GeoJSON whose
   geometry equals what `parse_region_basin` returned, with the CRS preserved.
3. `tests/test_spatial_products.py` — 1.02 reads the polygon from the declared
   input and still raises on empty geometry, missing CRS, and overlapping
   parents.
4. `tests/test_climate_store_contract.py` — `region_geojson` is an input, not an
   output; the three store declarations stay symmetric; WF2 declares no store.
5. `tests/test_series_identity.py` — `polygon_sha256` reads the new path and
   still changes when the polygon's content changes.
6. `pytest tests/test_cli.py` — all three Snakefiles parse and dry-run.
7. Live: WF1 on `C:/TESTS/CST/config_gabon0108.yml`, and
   `snakemake -n -s Snakefile_climate_projections` showing no
   `extract_climate_grid` job.

#### Validation of §8–11

1. `tests/test_spatial_units_spec.py` (new) — the spec's shape, and that the
   three declarations of `delineate_spatial_units` differ only in
   `message`/`log`/`benchmark`. Mirrors `test_region_spec.py`.
2. `tests/test_spatial_products.py` — the vector half writes the same six
   artifacts it writes today, with the same schemas; the raster half still
   validates the ID joins across raster, vector and registry.
3. `pytest tests/test_cli.py` — all three Snakefiles parse and dry-run.
4. Live: `snakemake -n -s Snakefile_climate_projections` lists
   `delineate_spatial_units` and **no** job whose inputs include `vito`,
   `modis_lai` or `soilgrids`. This is the assertion that the split achieved its
   purpose; without it the change is indistinguishable from the rejected
   unsplit alternative.
5. `check_baseline.py check` — the vector outputs are byte-identical to
   pre-split, so the baseline passes unchanged. A diff here means the split
   changed behaviour, which it must not.

### Open questions — §8–11

- **What do WF2 and WF3 actually plot or aggregate?** §10 leaves the consuming
  rules unnamed. Subbasin-resolved WF3 indicators would change what
  `basin_indicators.csv` means, which is a separate decision.
- **Measured cost of the hydrography read in WF2.** Asserted cheaper than the
  thematic stack, not measured. If it is not, the split has moved the cost rather
  than removed it.
- **Who writes `spatial_catalog.yml`** after the split — see *Consequences*.
- **Spelling for §11's per-basin ceiling** (`max_count: per_basin`, a per-parent
  ceiling, or a separate `subdivide: false`), and whether the default of 20
  changes. Separable from the split: it is what makes a gauge-free project
  *usable* from WF2/WF3, not what makes the boundaries *reachable*.

### Related

- `blueearth_cst/shared/snake_utils.py::climate_store_spec` — the shared-spec
  pattern this mirrors, and the store contract being changed.
- `Snakefile_climate_projections` §2.11 comment block — design D2/A1, the
  "accepted, stated price" this record repays.
- `blueearth_cst/projections/series_identity.py` — design D9 / ext2-01, why the
  polygon is fingerprinted by content.
- `dev/milestones/r07/` — R07 B1, which made the climate store model-free; this
  record keeps that property while removing the duplicate delineation.
- `blueearth_cst/spatial/products.py::_region_geometry` — the other delineation.
