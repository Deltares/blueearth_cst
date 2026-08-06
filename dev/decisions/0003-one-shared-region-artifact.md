Status: accepted (§1–7, implemented); **accepted (§8–11, implemented 2026-08-06)**;
        **proposed** (§12, the `wflow_id` renumbering)
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
    spatial foundation**, and the title with it. Adds §8–12 (proposed): split
    `prepare_spatial_maps` at its thematic-raster seam so the vector layers —
    basins, subbasins, catchments, rivers, locations, registry — become a third
    shared spec declared in all three workflows, letting WF2 and WF3 consume
    basin and subbasin boundaries without dragging in the raster stack. The
    "point at `basins.geojson`" alternative rejected in the original record is
    marked revisited, because this split removes its disqualifying factor.
    Also ruled: the shared-rule helpers drop the `_spec` suffix for `_rule`
    (`[R10-7]`). §1–7 below still name `region_spec` / `climate_store_spec`,
    which is what the **implemented** code calls them until that sweep lands.
    §11 rewritten and §12 added the same day: the automatic-subbasin ceiling
    becomes per-basin at a default of 11, and `wflow_id` is renumbered into
    per-basin blocks of 100. §12 is the one part of this record that moves
    outputs — it renames every gauge column in `output.csv` and requires a
    baseline re-record.
  - 2026-08-06: **§8–10 implemented and moved from proposed to accepted.**
    Three commits, each leaving the tree runnable: the `products.py` split with
    WF1 declaring rule `1.01c`; the WF2 (`2.03c`) and WF3 (`3.01f`)
    declarations; the R9 path-map row for the seam. §11–12 stay **proposed** —
    they move outputs and are separate landings. See *Landed state (§8–10)*
    below for what the implementation settled that the design did not say.
  - 2026-08-06: **§11 implemented and accepted, as ONE landing rather than the
    two §11 specifies.** Measurement collapsed the split — see *Landed state
    (§11)*. §12 stays proposed.

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

**§8–12 are PROPOSED, not implemented.** They extend the same pattern from the
region polygon to the vector foundation.

8. **Split `prepare_spatial_maps` at the thematic seam**, into two rules:

   - **`delineate_spatial_units`** — region polygon + hydrography catalog +
     `shared.basin.gauge_points` → `geoms/{basins,subbasins,catchments,rivers,
     locations}.geojson` and `location_registry.csv`. Model-free and
     engine-neutral, exactly as the whole rule is today.
   - **`prepare_spatial_maps`** (retained name and job) — consumes those, adds
     the thematic layers → `spatial_maps.nc`, `spatial_catalog.yml`,
     `spatial_report.yml`. **WF1 only.**

   The function boundary is `prepare_spatial_products` up to and including
   `_delineate_spatial_units` versus `_thematic_maps` onward — but **the seam is
   not free**, and an earlier draft of this section wrongly called it a pure
   decomposition. What crosses it in memory is the whole hydrography grid stack:
   `_thematic_maps(catalog, config, basins, maps["flow_direction"])` uses
   `flow_direction` as the resample template, and `maps` by then carries the
   derived `basin_id` and `subbasin_id` rasters the raster half also writes into
   `spatial_maps.nc`.

   **8a. The vector rule therefore writes a seventh artifact**: the hydrography
   grid stack (`flow_direction`, `flow_accumulation`, `upstream_area`,
   `river_mask`, `basin_id`, `subbasin_id`) as a netCDF the raster rule declares
   as an input. Recomputing instead would make WF1 read the hydrography twice
   with two grids that can drift; rasterizing back from the geojsons adds new
   logic and precision loss. The intermediate is the only option that keeps one
   producer per value.

   It is a **seam intermediate, not a product**: it stays out of
   `spatial_catalog.yml`, and it needs a row in the R9 path map and a
   `tree-check` entry.

   **8b. The vector rule's inputs and params must be a pure function of
   `project` + `shared.basin`.** Today rule 1.02 declares
   `config_snake = config_path` as an *input* and passes the whole
   `workflows.model_creation` section as *params*. Neither can survive a
   three-way shared declaration: the five projections-only configs contain no
   `workflows.model_creation` keys at all, so the params payload would differ per
   invoking workflow — the input/params asymmetry `ext1-02` forbade for the
   climate store — and `config_path` itself differs between a full config and a
   single-workflow one, so as a declared input it would thrash the shared rule on
   every WF1/WF2 alternation.

   So the vector rule drops `config_snake` and narrows its params to the
   `SpatialConfig` fields resolved from `shared.basin` alone. Consequence: the
   one-release `workflows.model_creation.output_locations` fallback in
   `resolve_gauge_points_path` cannot feed the shared rule. Acceptable — it is
   deprecated compat — but it must be stated rather than discovered. What makes
   this safe is rule 3.00b, which already guarantees `shared.basin` agrees across
   the workflows that share the rule.

9. **`snake_utils.spatial_units_rule(...)`** returns a `SpatialUnitsRule` —
   `script`, `inputs`, `outputs`, `params` — mirroring the other two shared-rule
   helpers. All three workflows declare `delineate_spatial_units` from it,
   byte-identical but for `message`/`log`/`benchmark`.

   The suffix is **`_rule`, not `_spec`** (ruled 2026-08-06): the object holds a
   rule's script, inputs, outputs and params, so it *is* a rule definition minus
   its labels, and "the region rule" reads to someone who does not write
   software. `region_spec` → `region_rule` and `climate_store_spec` →
   `climate_store_rule` rename with it so the trio stays consistent —
   `dev/followups.md` `[R10-7]`, folded into the R10 sweep. `_contract` was
   rejected because this repo already uses "contract" for interchange surfaces
   (`dev/reference/contracts/`, `SPATIAL_CONTRACT_VERSION`).

10. **WF2 and WF3 consume the vectors as declared inputs** of the figure and
    metric rules that use them. *Which* rules is deliberately left open — see
    *Open questions*; making the artifacts reachable is this decision, using them
    is the next one.

11. **`automatic_subbasins.max_count` becomes `max_per_basin`, a PER-BASIN
    ceiling, default 20 → 11.** Today it is one **global** budget shared across
    parents:
    `allocate_automatic_subbasin_budgets` gives each fallback parent one unit,
    distributes the remainder by largest-remainder weighted on upstream area, and
    **raises outright** when `len(parent_areas) > max_count`. Per-basin removes
    that failure mode and makes a multi-basin project's partitions comparable —
    every parent gets the same ceiling however many parents there are.

    `allocate_automatic_subbasin_budgets` is then **deleted, not adapted**: with
    an equal per-parent ceiling there is nothing left to allocate.

    Safe because `select_automatic_subbasins` treats the count as an **upper
    bound** — it binary-searches for the smallest area threshold whose outlet
    count is `<= max_count` — so a small parent simply yields fewer subbasins and
    never errors. The area weighting being dropped therefore costs less than it
    sounds: a small basin was already going to produce fewer units than a large
    one at the same ceiling.

    **The key is renamed, not just redefined** (ruled 2026-08-06). `max_count`
    changing meaning in place would silently triple a three-basin project's
    partition — 20 subbasins total becomes 20 *per basin* — with no error and no
    diff in the config. `shared.basin`'s schema is **not** closed (unlike
    `advanced_settings`, whose `_ADVANCED_SETTINGS_SCHEMA` rejects unknown keys),
    so a leftover `max_count` would be ignored in silence and the project would
    run at the new default instead of the value its author wrote. The rename must
    therefore come with an **explicit rejection** of the old key in
    `parse_spatial_config`, naming the replacement — not merely a new key that
    happens to be read.

    **11, not 13.** Twelve is the practical ceiling for a qualitative colour ramp
    a reader can tell apart (ColorBrewer `Set3` and `Paired` both stop at 12), so
    11 keeps *one basin's* subbasin map legible with a legend entry per unit; 13
    forces a palette that repeats or interpolates. The argument holds **per
    basin only** — a three-basin project reaches 33 units and exceeds any
    qualitative ramp regardless, so per-basin colouring is the figure's problem
    to solve, not the default's. `MAX_LOCAL_SUBBASIN_NUMBER = 99` stays as the
    hard cap: this is a default, not a limit.

    **§11 splits into two landings, because it is NOT behaviour-preserving.** An
    earlier draft claimed §8–11 left outputs byte-identical. False, twice:

    - **11a — rename only.** Ship the tracked seed config as
      `max_per_basin: 20`, preserving the current value. Expected baseline delta:
      exactly one YAML key in the config snapshot, which `check_baseline`
      fingerprints by normalized SHA-256. Note that without this edit the shipped
      config **fails to parse** the moment the old-key rejection lands.
    - **11b — default 20 → 11**, as its own baseline event. The shipped test
      config has `gauge_points: null`, so the fixture runs the *automatic* path:
      lowering the ceiling changes the threshold `select_automatic_subbasins`
      selects whenever the count at 20 exceeded 11, which moves the subbasins,
      the registry, the gauge map and the `_basavg` columns. Whether this
      fixture actually crosses 11 can only be answered in the primary checkout.

    *Withdrawn:* an earlier draft of this section proposed "no gauges → one
    subbasin per basin". The owner retracted it on 2026-08-06. A gauge-free
    project **should** be subdivided; it should just not be subdivided twenty
    ways.

12. **`wflow_id` becomes a basin-grouped, subbasin-structured integer**, so
    Wflow's output columns group by basin without losing what they point at.
    Today two unrelated formulas share the column:

    | location | today | basin 1 example |
    |---|---|---|
    | subbasin primary | `wflow_id = subbasin_id = basin_id*100 + local_subbasin_number` | `101` |
    | any additional point | `1_000_000 + subbasin_id*100 + local_location_number` | `1_010_102` |

    A seven-digit id sits beside a three-digit one in the same column, for points
    a user thinks of as siblings.

    Target: **`wflow_id = basin_id*1000 + local_subbasin_number*10 + m`**, with
    `m = 0` for the subbasin's primary location and `m = 1…9` for additional
    locations within it. Basin 1 reads `1010, 1011, 1020, 1030…`; basin 2 reads
    `2010, 2011, …`. Grouped by basin, ordered by subbasin, and the subbasin
    stays legible in the flat id.

    **Rejected: `basin_id*100 + k` with a flat sequential `k`** (the first draft
    of this section, and the owner's initial proposal). Two defects, both raised
    in review and accepted 2026-08-06:

    - It *near-collides* with `subbasin_id` at an off-by-one — S01's primary
      would be `100` while its `subbasin_id` is `101`, and `103` could be an
      additional gauge while subbasin `103` is S03. Two 3-digit namespaces with
      different meanings and no visual tell.
    - It **loses information the current scheme carries**. `1_010_102` decodes to
      "subbasin 101, location 02"; a flat `103` decodes to nothing and can only
      be resolved through the registry. Ugly ids that encode structure beat tidy
      ids that do not.

    **Cost, accepted:** nine additional locations per subbasin (today 99), and
    four-digit ids. A `basin_id*10000 + subbasin*100 + m` variant lifts the limit
    to 99 at five digits and is the fallback if a real project ever needs it.

    **`location_code` does not change.** It stays hierarchical (`B001-S01-L01`).
    Codes are for reading, `wflow_id` is the integer for joining and for scanning
    a CSV header.

    **12a. `build_wflow_model` must repeal an enforced invariant.**
    `_validate_registry` raises `"Every primary location wflow_id must equal its
    subbasin_id"` (`build_wflow_model.py:157`). Under §12 a primary is
    `basin*1000 + sub*10` while its `subbasin_id` is `basin*100 + sub`, so **WF1
    cannot build at all** until that check is rewritten. An earlier draft called
    this a silent breakage to grep for; it is a designed-in invariant this
    decision must consciously repeal, and it is loud.

    **12b. Observation files are a user-data migration.** Observation timeseries
    columns are `wflow_id` values (`observation_validation.py`,
    `plot_results.py`), so after §12 every project's observation CSV header is
    wrong and `validate_observation_station_ids` raises on the missing ids. The
    migration note must cover observation files, not only `gauge_points` pinning.

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

#### Consequences of §8–12

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

- **§11b and §12 both move outputs.** Only §8–10 and §11a are
  behaviour-preserving; an earlier draft of this record claimed §8–11 were, and
  that was wrong (see §11). Land each output-moving step in its own commit, or an
  intended diff becomes indistinguishable from a regression.
- **§12's baseline mechanism, precisely.** The baseline's discharge anchor
  resolves the primary outlet column through `outlet_index.csv` →
  `subcatchment_id` → `Q_<subcatchment_id>` (`check_baseline.py:380-405`), and
  outlet-map values are **subbasin ids**, which §12 does not renumber. So the
  anchor column keeps its name. What moves is the **gauge** columns
  (`Q_<wflow_id>` / `P_<wflow_id>` from the `gauges_locations` map), the WF3
  indicator tables, the registry, and — via re-pinning — the config snapshots.
  State this in the re-record commit, or the diff will not match its message.
- **The outlet/gauge id overlap inverts, and may change the CSV *schema*.**
  `plot_results.py:43-51` documents today's normal case, where the outlet's
  subcatchment id equals the outlet gauge's `wflow_id` and the two coincide in
  one column. §12 removes that equality, so the same physical point may appear
  as two columns. Whether `output.csv` gains a column — a schema change, not a
  rename — must be pinned by a fixture run before the baseline is re-recorded.
- **`wflow_id == subbasin_id` is repealed, not broken** — see §12a. It is an
  enforced validator, so the failure is a hard build error, not a silent drift.
- Projects that pin `wflow_id` in `gauge_points` must re-pin, and every
  observation file's column headers must be rewritten (§12b). The mismatch check
  in `assign_location_ids` and `validate_observation_station_ids` both raise by
  name, so the failures are loud — but this is user data, not repo data.
- The `warn_if_low_gauge_ids` advisory is **moot** under §12 rather than needing
  a new threshold: generated ids start well above it, and a user-pinned id below
  the floor dies at the mismatch check first.

*Neutral*

- **`spatial_catalog.yml` stays whole in the raster (WF1-only) half.** It is the
  model-build interface — `build_wflow_model` resolves every entry through it —
  and `_catalog_dict()` is static, so the raster rule can list the vector entries
  it consumed. WF2 and WF3 need no hydromt catalog to read a geojson by path. A
  vector-only catalog *extended* by the raster rule would mean two writers of one
  file, which is the R7-1 anti-pattern.
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

#### Alternatives to §8–12

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
  its own record when the duplication cost of three shared rules is felt.

  **Tripwire, so this is not re-litigated every milestone:** adopt WF0 when a
  **fourth** shared rule appears, or when a byte-identity contract test catches
  real drift twice. Until then the duplication is held safe by tests that already
  exist, and the deferral costs nothing.

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

#### Validation of §8–12

1. `tests/test_spatial_units_rule.py` (new) — the helper's shape, and that the
   three declarations of `delineate_spatial_units` differ only in
   `message`/`log`/`benchmark`. Mirrors `test_region_rule.py` (itself renamed
   from `test_region_spec.py` by `[R10-7]`).
2. `tests/test_spatial_products.py` — the vector half writes the same six
   artifacts it writes today, with the same schemas; the raster half still
   validates the ID joins across raster, vector and registry.
3. `pytest tests/test_cli.py` — all three Snakefiles parse and dry-run.
4. Live: `snakemake -n -s Snakefile_climate_projections` lists
   `delineate_spatial_units` and **no** job whose inputs include `vito`,
   `modis_lai` or `soilgrids`. This is the assertion that the split achieved its
   purpose; without it the change is indistinguishable from the rejected
   unsplit alternative.
5. `check_baseline.py check` — for §8–10 the vector outputs are byte-identical
   to pre-split, so the baseline passes unchanged. A diff there means the split
   changed behaviour, which it must not.
6. §11b and §12 are **expected** to move outputs and are validated separately:
   - §11b — on a multi-basin fixture, each parent's automatic partition is capped
     at `max_per_basin` independently, and a parent count exceeding it no longer
     raises. `tests/test_delineation.py` loses the global-allocation cases with
     `allocate_automatic_subbasin_budgets`.
   - §12 — `tests/test_identity.py`: every `wflow_id` decodes as
     `basin_id*1000 + local_subbasin_number*10 + m`; each subbasin's primary ends
     in `0`; a tenth location in one subbasin raises; values are unique. Then
     **re-record the baseline**, stating in the commit that the *gauge* column
     names changed by design while the outlet anchor column did not.
7. **§8's acceptance gate is a measurement, not an assertion. PASSED
   2026-08-06.** Measured by
   `dev/decisions/0003-one-shared-region-artifact/probe_split_cost.py`, which
   times the two halves exactly as `prepare_spatial_products` calls them, against
   the real `deltares_data.yml` catalog:

   | | fixture config | `config_gabon0108.yml` |
   |---|---|---|
   | already paid today (`delineate_region`) | 5.58 s | 5.86 s |
   | **§8 adds to WF2/WF3** | **3.65 s** | **3.89 s** |
   | unsplit would add instead | 13.03 s | 13.17 s |
   | **§8 avoids** | **9.37 s — 72.0%** | **9.28 s — 70.5%** |

   `delineate_region` is **excluded from the comparison**: WF2 and WF3 already
   declare it under §1–7, so it is paid either way. The gate is what §8 *adds*
   against what declaring the rule unsplit would add — and it avoids ~71% of
   that. **§8 is worth doing; the unsplit alternative is not.**

   **Caveat, unresolved:** both configs resolve to the same 384-cell basin
   (16 × 24), where both halves are dominated by fixed overhead — opening VRTs
   and tifs, parsing the catalog — rather than per-cell work. No larger basin
   config exists on this machine. Which half scales worse with basin size is
   still unmeasured, so re-run the probe when a large basin is available. The
   direction is unlikely to reverse: the thematic half's three global-source
   reads are per-run overhead that a larger basin only adds to.

### Landed state (§8–10), 2026-08-06

What the implementation settled, in the places the design left underspecified or
was wrong. §11 and §12 are untouched and still proposed.

- **The seam carries the WHOLE grid stack, not §8a's six layers.** §8a lists
  `flow_direction`, `flow_accumulation`, `upstream_area`, `river_mask`,
  `basin_id`, `subbasin_id`. `spatial_maps.nc` also holds `cell_area`,
  `river_order`, `elevation` and `slope`, all from `prepare_hydrography`, so a
  six-layer seam would have forced the raster half to re-derive them — the
  second hydrography read §8a rejects. §8's own prose ("the whole hydrography
  grid stack") is the binding statement; the enumeration is illustrative.
- **`<project_dir>/data/spatial/hydrography.nc`**, confirmed by the owner.
  Absent from `spatial_catalog.yml`, and a test pins the absence: that file is
  the model-build interface, and an entry would advertise an intermediate to
  `build_wflow_model`.
- **Two non-obvious things were needed for byte-identity**, both found by
  hashing all nine artifacts before and after on the synthetic fixture rather
  than by reasoning:
  - the seam is read with `mask_and_scale=False`. Every layer stores its nodata
    as a `_FillValue` **attribute**, so default CF decoding recasts the array to
    float — `basin_id` and `subbasin_id` would have returned float64 with NaN at
    the fill, and `spatial_maps.nc` would have shipped float identifier rasters
    while every value still compared equal;
  - the read dataset is rebuilt **coords-first** and its CRS re-anchored on its
    EPSG code. netCDF stores variables in creation order and `open_dataset`
    yields data variables before coordinates; and a CRS rebuilt from stored WKT
    re-emits a poorer WKT than the catalog's (`DATUM[...]` where the catalog had
    `ENSEMBLE[... MEMBER ...]`, `CONVERSION["unnamed"]`). Each was a byte diff on
    a file whose every value already matched.
- **`delineation_method_by_basin` is derived, not carried.** The raster half
  recovers it from `subbasins`, where every row already records its parent's
  method, and raises if a parent carries two. Pinned on the multi-basin fixture,
  since a one-basin fixture cannot tell a per-basin mapping from a constant.
- **The rule numbers are `1.01c` / `2.03c` / `3.01f`.** Not `3.01c`: WF3 had
  already taken c/d/e, and renumbering is `[R10-5]`. The WF3 rule is therefore
  defined after `3.01e`, keeping numbers in definition order.
- **Reachability needed two edges per workflow, which §10 did not anticipate.**
  With no consumer the rule is a leaf, so it needs (a) a `rule all` target entry
  or it is never scheduled, and (b) an edge into `gather_logs` **and**
  `gather_benchmarks` or it runs parallel to the merge and strands its log part
  under `_parts/` — the defect this repo's `LOG_RULES` blocks already record
  three times. Ruled 2026-08-06 that the project-scoped vectors join WF3's
  otherwise experiment-scoped `rule all`: they depend on `shared.basin` alone,
  which rule 3.00b already guarantees agrees across workflows.
- **The §8 acceptance assertion is now a test, not a one-off dry-run.**
  `_thematic_maps` — the only reader of `vito`, `modis_lai` and `soilgrids` —
  has exactly one entry point, so "WF2 no longer reads the thematic sources" is
  decidable by asking which workflows declare a rule running
  `prepare_spatial_maps.py`. Checking the script rather than grepping a job list
  for `vito` also survives a config that names those sources explicitly.
  Measured on the fixture config, WF2's dry run schedules 26 jobs across 10
  rules and `prepare_spatial_maps` is not among them.
- **§8b's cost is larger than §8b says, and it is silent.** The section states
  the deprecated `workflows.model_creation.output_locations` fallback "cannot
  feed the shared rule" and calls that acceptable, but never says what happens
  downstream. Rule 1.02 previously received `model_config` and so honoured the
  legacy key on its way to the partition. A config setting ONLY the legacy key
  now reaches the vector rule with no gauge points, falls back to the
  **automatic** partition, and produces different subbasins, a different
  `location_registry.csv` and different `wflow_id` values — with **no error**,
  since an absent gauge file is a legitimate configuration. Rule 1.05 would then
  add gauges to a model whose subbasins were derived without them.

  No tracked config is affected: every one sets `shared.basin.gauge_points`
  (checked 2026-08-06), so the baseline seed is unaffected. But the
  compatibility release's promise is narrower than it reads — a migrating
  project must **move** the key, not merely be warned the old one is deprecated.
  Whoever removes the fallback should say so in the migration note.
- **Not yet verified:** the thematic clip geometry now arrives from
  `basins.geojson` (EPSG:4326) reprojected onto the grid CRS, where the unsplit
  rule passed an in-memory frame already in that CRS. A no-op whenever the
  hydrography is geographic, which merit_hydro is — but the fixture catalog
  ignores `geom=`, so only `check_baseline.py check` from the primary checkout
  can close it.

### Landed state (§11), 2026-08-06

- **One landing, not two — and the reason is a measurement.** §11 splits itself
  into 11a (rename, value preserved) and 11b (default 20 → 11) so "an intended
  diff becomes distinguishable from a regression". That reason does not apply
  here, because **11b moves nothing on the baseline fixture.** Replaying
  `select_automatic_subbasins` against the fresh fixture's own hydrography seam,
  at every ceiling from 1 to 25:

  | ceiling | 1 | 2 | 3 | 4 | **5–25** |
  |---|---|---|---|---|---|
  | subbasins | 1 | 1 | 3 | 4 | **5** |

  The partition **saturates at 5** from ceiling 5 upward, so 11 and 20 are the
  same number to this fixture and the single baseline diff is attributable to
  the key rename alone. This answers §11b's own open question — "whether this
  fixture actually crosses 11 can only be answered in the primary checkout" — in
  the negative: the 384-cell basin never binds the ceiling at all.

  **The corollary matters more than the collapse.** This fixture is *structurally
  incapable* of validating any ceiling above 5, so no baseline run can ever test
  §11b. Its real coverage is the synthetic multi-basin case in
  `tests/test_spatial_products.py`, which is what validation item 6 already
  asked for and which now carries two tests: the ceiling applies per parent
  independently, and more parents than the ceiling no longer raises.

- **The rename is wider than §11 states: FIVE configs, not one.**
  `config/workflows/{snake_config_model_test,..._linux,dev_fast,template}.yml`
  and `tests/snake_config_model_test.yml` all carry the key, and all five must
  move in the same commit or the old-key rejection makes them unparseable. All
  five went to `max_per_basin: 11`, the new default, rather than to a preserved
  20 — which the table above shows is the same partition.

- **THREE baseline entries move, not one.** §11 says "exactly one YAML key in
  the config snapshot". The manifest holds three config-snapshot targets —
  WF1's, WF2's and WF3's — all sharing hash `48242f48…` because all three are
  copies of the seed config. A re-record touches all three.

- **The internal name moved with the config key.** `SpatialConfig.
  max_automatic_subbasins` → `max_subbasins_per_basin`, and
  `select_automatic_subbasins`'s `max_count` parameter → `max_subbasins`.
  Leaving the code name reading "global count of automatic subbasins" after the
  meaning became per-basin would be the same silent redefinition §11 forbids at
  the config layer, one level down. The rule's params key moved too, so the
  params rerun-trigger fires once.

- **`shared.basin` is inside rule 3.00b's guard digest**, so renaming the key
  flips `guarded_sections_digest` and the drift guard re-runs. Harmless when the
  workflows run in order (WF1 re-snapshots first); a WF3-only run against a
  stale WF1 snapshot fails loud, which is the guard working.

### Open questions — §8–12

- **What do WF2 and WF3 actually plot or aggregate?** §10 leaves the consuming
  rules unnamed. Subbasin-resolved WF3 indicators would change what
  `basin_indicators.csv` means, which is a separate decision.
- **Ordering of additional locations within `m`** (§12). These become Wflow
  column names, so the tie-break must be deterministic — today
  `assign_location_ids` sorts on `station_name, snapped_row, snapped_col,
  original_x, original_y` within a subbasin, which carries over, but say so.
- **Does the `outlets` map get renumbered too?** It must not — the baseline's
  discharge anchor depends on outlet values staying subbasin ids — but §12 never
  says so explicitly, and an implementer renumbering "all the ids" would break
  the gate.
- **Nine additional locations per subbasin** is the §12 cap. Fine unless a real
  deployment exceeds it, in which case the `basin*10000 + sub*100 + m` variant
  applies. Check against a real gauge list before implementing, not after.

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
