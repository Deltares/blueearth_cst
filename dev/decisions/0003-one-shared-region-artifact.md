Status: accepted
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

# ADR 0003 — One shared region artifact, delineated once per project

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

### Alternatives considered

- **Point the climate store at `spatial/geoms/basins.geojson`.** No new artifact
  at all — the polygon already exists under that name. **Rejected**: it makes
  every workflow depend on rule 1.02, whose real product is `spatial_maps.nc`
  and the raster stack behind it. WF2 would then run the spatial foundation to
  get a polygon — the same trade it makes today with the climate store, moved
  rather than removed. `basins.geojson` is also the *exploded, id-carrying*
  form, which is a P1 domain product, not a plain extent.
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
