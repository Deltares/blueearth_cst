# Rule index — WF1, WF2 and WF3

Every rule in `build_model.smk`, `analyze_projections.smk` and `run_stress_test.smk`:
what each one does, what it writes, and how they connect.

> **WF0 (`analyze_climate.smk`) is not covered here.** Its ten numbered rules are
> documented only by the comment headers in the Snakefile itself.

Each workflow gets a diagram, a one-line summary table, then one section per rule.
**Does** is the rule's job; **Writes** transcribes its `output:` block, so the claim can be
checked against the Snakefile rather than believed.

Rule numbers are reused, so any `W.NN` written before 2026-08-06 names a different
rule — translate through [What changed](#what-changed) before reading one in
`dev/milestones/`, `dev/decisions/`, `dev/LOG.md` or a dated migration record.

## On the numbers

`W.NN` is the rule's position in its workflow's **logical order**: data first,
then model build, then run, then records. Numbering is contiguous within each
workflow and every dependency points from a lower number to a higher one, so a
rule can never depend on something numbered after it.

**"Every dependency low→high" is checked against `input:`, `ancient()`
included.** `ancient()` suppresses the timestamp rerun-trigger; it does not
remove the DAG edge. Two rules move further than a reader of the previous map
would expect for exactly this reason — see the note under the WF1 table.

**Going forward: do not renumber to insert a rule.** Use a letter suffix
(`1.09b`) until the next deliberate sweep. Renumbering is a migration, not an
edit.

## What changed

The only place this page names old numbers and names. Everything after it is the current state.

### Renumbering

Read this table before interpreting any `W.NN` in a document written before
2026-08-06. 47 declarations: 18 in WF1, 10 in WF2, 19 in WF3. The `was`
column carries the old name too wherever the rename and the renumber coincide,
so one lookup answers both.


**WF1** — `build_model.smk`

| new | rule | was |
|---|---|---|
| 1.00 | `all` | 1.00 |
| 1.01 | `snapshot_config` | 1.01 |
| 1.02 | `delineate_region` | 1.01b |
| 1.03 | `delineate_spatial_units` | 1.01c |
| 1.04 | `extract_historical_climate` | 1.10 `extract_climate_grid` |
| 1.05 | `plot_climate_source` | 1.15 |
| 1.06 | `prepare_spatial_maps` | 1.02 |
| 1.07 | `build_wflow_model` | 1.03 |
| 1.08 | `add_reservoirs_lakes_glaciers` | 1.04 |
| 1.09 | `declare_wflow_outputs` | 1.05 `add_gauges_and_outputs` |
| 1.10 | `add_climate_forcing` | 1.08 `add_forcing` (+ 1.07, merged in) |
| 1.11 | `write_outlet_index` | 1.06 |
| 1.12 | `plot_basin_map` | 1.12 `plot_map` |
| 1.13 | `plot_forcing` | 1.13 |
| 1.14 | `run_wflow` | 1.09 |
| 1.15 | `plot_wflow_evaluation` | 1.11 `plot_results` |
| 1.16 | `gather_benchmarks` | 1.14 |
| 1.17 | `gather_logs` | 1.16 |

> **Two rules moved further than the previous draft of this table had them, and
> not because of the new rule.** `write_outlet_index` and `plot_basin_map` both
> declare `ancient(<model>/.model_final)`, and that sentinel is written by
> `add_climate_forcing` — so both are downstream of it, and the earlier map,
> which placed them at 1.07 and 1.10 against `add_climate_forcing` at 1.11, had
> two dependencies pointing high→low. The cause is ADR 0004, which moved the
> model root's terminal anchor onto the forcing rule after that map was drawn.
> `ancient()` is why it was easy to miss: it hides the rerun-trigger, not the
> edge.

**WF2** — `analyze_projections.smk`

| new | rule | was |
|---|---|---|
| 2.00 | `all` | 2.00 |
| 2.01 | `snapshot_config` | 2.03 |
| 2.02 | `delineate_region` | 2.03b |
| 2.03 | `delineate_spatial_units` | 2.03c |
| 2.04 | `fetch_gcm_slice` | 2.01 `fetch_gcm_raw` |
| 2.05 | `reduce_gcm_series` | 2.02 |
| 2.06 | `derive_change_factors` | 2.04 |
| 2.07 | `plot_gcm_timeseries` | 2.06 `plot_climate_proj_timeseries` |
| 2.08 | `gather_benchmarks` | 2.10 |
| 2.09 | `gather_logs` | 2.07 |

> **WF2's two gather rules swap relative order**, which is the one cell in this
> table that is a convention choice rather than a derivation. The two are
> parallel leaves — identical `input:` sets, neither consumes the other — so no
> dependency decides it. WF1 and WF3 both define benchmarks first; WF2 defined
> logs first, and nothing recorded why. Ruled 2026-08-06 to follow the other two
> workflows, so the three read alike and `gather_logs` is the last-numbered rule
> everywhere.

**WF3** — `run_stress_test.smk`

| new | rule | was |
|---|---|---|
| 3.00 | `all` | 3.00 |
| 3.01 | `check_project_consistency` | 3.00b |
| 3.02 | `snapshot_config` | 3.01 |
| 3.03 | `delineate_region` | 3.01b |
| 3.04 | `delineate_spatial_units` | 3.01f |
| 3.05 | `write_model_reference` | 3.01c |
| 3.06 | `check_model_reference` | 3.01d |
| 3.07 | `write_experiment_config` | 3.01e |
| 3.08 | `extract_historical_climate` | 3.02 `extract_climate_grid` |
| 3.09 | `prepare_stress_test_grid` | 3.03 `climate_stress_parameters` |
| 3.10 | `prepare_weathergen_config` | 3.04 `prepare_weagen_config` |
| 3.11 | `generate_weather_realizations` | 3.06 `generate_weather_realization` |
| 3.12 | `perturb_climate_realization` | 3.07 `generate_climate_stress_test` |
| 3.13 | `write_climate_data_catalog` | 3.08 `climate_data_catalog` |
| 3.14 | `downscale_climate_realization` | 3.09 |
| 3.15 | `run_wflow_batch_<b>` | 3.10 |
| 3.16 | `derive_wflow_indicators` | 3.11 |
| 3.17 | `gather_benchmarks` | 3.12 |
| 3.18 | `gather_logs` | 3.13 |

> **`3.01f` is gone, and that is what the renumber was for.** The vector rule
> answered to `3.01f` only because `3.01c`–`3.01e` were already taken, so a rule
> that belongs beside `delineate_region` sorted five letters away from it. It is
> now `3.04`, adjacent to `3.03` in all three workflows.

### Twelve renames

| rule | was |
|---|---|
| `declare_wflow_outputs` | `add_gauges_and_outputs` |
| `add_climate_forcing` | `add_forcing` |
| `extract_historical_climate` | `extract_climate_grid` |
| `plot_wflow_evaluation` | `plot_results` |
| `plot_basin_map` | `plot_map` |
| `fetch_gcm_slice` | `fetch_gcm_raw` |
| `plot_gcm_timeseries` | `plot_climate_proj_timeseries` |
| `prepare_stress_test_grid` | `climate_stress_parameters` |
| `prepare_weathergen_config` | `prepare_weagen_config` |
| `generate_weather_realizations` | `generate_weather_realization` |
| `perturb_climate_realization` | `generate_climate_stress_test` |
| `write_climate_data_catalog` | `climate_data_catalog` |

## Conventions

- Every rule also writes a `log:` part and a `benchmark:` part under
  `logs/_parts/` and `benchmarks/_parts/`. Uniform, so not repeated per rule.
- **Writes (undeclared)** is a real disk write that Snakemake does not know
  about. These matter: they are invisible to `--dry-run`, not cleaned by
  `--delete-all-output`, and unusable as a dependency. Three rules mutate
  `wflow_sbm.toml` or `staticmaps.nc` this way, by design — the sentinel pattern
  in the Snakefile comments exists precisely because of it.
- `temp(...)` outputs are deleted once consumed. Sentinels (`.model_built`,
  `.outputs_configured`, `.project_consistency_ok`, `.model_reference_ok`,
  `.guard_ok`) are outputs but not products.

Paths are relative to `project_dir`, with these shorthands:

| shorthand | path |
|---|---|
| `<model>/` | `models/hydrology/wflow/` |
| `<spatial>/` | `data/spatial/` |
| `<store>/` | `data/climate/historical/<clim_source>_<window>/` |
| `<proj>/` | `data/climate/projections/<clim_project>/` |
| `<exp>/` | `experiments/<experiment_name>/` |
| `<wg>/` | `<exp>/climate/weathergenr/` |
| `<runs>/` | `<exp>/hydrology/wflow/` |

---

# WF1 — model creation (`build_model.smk`)

Builds a distributed Wflow-SBM model from global datasets via hydromt and runs it
once on historical forcing. No calibration — rapid deployment.

An arrow is a **declared** dependency; rules on separate branches run
concurrently. The stages read **data → model → run → records**: nothing that
does not need a built model appears after one, and the numbers now follow.

```
STAGE 1 — DATA   (no model exists yet)
──────────────────────────────────────────────────────────────────
                    config + data catalogs
                              │
      1.01 snapshot_config ───┤
                              ▼
                    1.02 delineate_region ──► region.geojson
                              │
              ┌───────────────┴───────────────┐
              ▼                               ▼
    1.04 extract_historical_climate   1.03 delineate_spatial_units
      (SHARED store, = WF3 3.08)       (SHARED vectors, = 2.03/3.04:
              │                         basins, subbasins, rivers,
              ▼                         locations, the registry)
    1.05 plot_climate_source                  │
                                              ▼
                                    1.06 prepare_spatial_maps
                                     (thematic rasters, WF1 only)
STAGE 2 — MODEL BUILD                         │
──────────────────────────────────────────────────────────────────
                                              ▼
                                    1.07 build_wflow_model
                                              │
                                              ▼
                                  1.08 add_reservoirs_lakes_glaciers
                                              │
                                              ▼
                                   1.09 declare_wflow_outputs
                                              │
                                              ▼
                                   1.10 add_climate_forcing
                                     (LAST writer of the model
                                      root — ADR 0004's sentinel)
                                              │
              ┌───────────────┬───────────────┼───────────────┐
              ▼               ▼               ▼               ▼
   1.11 write_outlet   1.12 plot_basin   1.13 plot_forcing  (to stage 3)
        _index              _map

STAGE 3 — RUN + EVALUATE
──────────────────────────────────────────────────────────────────
                         1.14 run_wflow
                               │
                               ▼
               1.15 plot_wflow_evaluation ◄── the store (1.04)

STAGE 4 — RUN RECORDS
──────────────────────────────────────────────────────────────────
      1.16 gather_benchmarks · 1.17 gather_logs   (last: every terminal)
```

**Stages are a reading aid, not a barrier.** Stage 1's climate branch (1.04,
1.05) runs concurrently with everything below it — a cold store extracts while
the model builds. Only the arrows constrain order.

**Three rules hang off 1.10 through `ancient()`, and the diagram draws those
edges as real** — because they are. 1.11, 1.12 and 1.14 all declare
`ancient(<model>/.model_final)`, the terminal build sentinel 1.10 writes.
`ancient()` suppresses the timestamp rerun-trigger and nothing else; the
dependency stands, which is exactly why 1.11 and 1.12 are numbered after 1.10
and not beside 1.07. 1.11 also reads `outlets.geojson` (1.07) and the registry
(1.03), and 1.12 reads `staticmaps.nc` (1.07) — those are the edges the diagram
omits to stay legible, and none of them contradicts the numbering.

**The five leaves.** 1.05, 1.11, 1.12, 1.13 and 1.15 have no downstream rule.
All are members of `WF1_TERMINALS`, so all are `rule all` targets and inputs of
the two gather rules — that is the edge the stage-4 line stands in for. Four are
figures, which are expected to terminate (no rule consumes a `.png`). **1.11 is
the one data leaf**, and its real consumer sits outside the workflow: see its
section below.

`WF1_TERMINALS` has a **sixth** member that is not a leaf —
`<spatial>/spatial_catalog.yml`, listed as one representative of 1.06's
multi-output set so the gather rules wait for it. Its producer feeds 1.07, so it
is a terminal in the target-set sense without being a graph leaf.

**What is NOT a dependency, despite reading like one.** 1.10 does not consume the
climate store: it reads source climate through the data catalog (`-d`), and its
only declared input is 1.09's sentinel — it assembles the forcing recipe itself.
The store reaches WF1's *figures* (1.05, 1.15), never its forcing.

| # | rule | in one line |
|---|---|---|
| 1.00 | `all` | Target aggregator. |
| 1.01 | `snapshot_config` | Snapshots the config and everything it references. |
| 1.02 | `delineate_region` | Delineates the one project extent. |
| 1.03 | `delineate_spatial_units` | The shared vector foundation, and where gauges enter the workflow. |
| 1.04 | `extract_historical_climate` | The shared historical-climate store (= WF3 3.08). |
| 1.05 | `plot_climate_source` | Climate figures on the source grid. |
| 1.06 | `prepare_spatial_maps` | The thematic raster stack and the model-build interface. |
| 1.07 | `build_wflow_model` | Parameterises Wflow-SBM, and where gauges enter the model. |
| 1.08 | `add_reservoirs_lakes_glaciers` | Adds waterbodies. |
| 1.09 | `declare_wflow_outputs` | Declares the `[output.csv]` block: which timeseries Wflow emits. |
| 1.10 | `add_climate_forcing` | Assembles the hydromt recipe and applies it: builds the forcing. |
| 1.11 | `write_outlet_index` | Crosswalk from Wflow outlet IDs to named stations. |
| 1.12 | `plot_basin_map` | Every figure the spatial foundation supports: the basin/DEM map and the thematic family. |
| 1.13 | `plot_forcing` | The same figures on the model's own forcing grid. |
| 1.14 | `run_wflow` | Runs Wflow.jl once. |
| 1.15 | `plot_wflow_evaluation` | The evaluation figures, and the metrics table. |
| 1.16 | `gather_benchmarks` | Merges the timing parts. |
| 1.17 | `gather_logs` | Merges the log parts. |

## WF1 rule detail

#### 1.00 · `all`

**Does.** Target aggregator — declares the WF1 target set (the terminals, plus
the config snapshot, the merged log and the benchmark table) so one
`snakemake all` builds the workflow.

**Writes.** Nothing of its own.

#### 1.01 · `snapshot_config`

**Does.** Copies the config and every file it references into the project,
routed by kind, and writes an immutable content-addressed bundle of the
effective settings (merged config + advanced settings + manifest) so a finished
project can say what it was run with.

**Writes.** `config/runs/snake_config_build_model.yml` ·
`config/runs/build_model/<digest>/` (bundle dir).

**Writes (undeclared).** Copies into `config/templates/` (build + waterbodies),
`config/catalogs/` (data catalogs) and `config/basin_data/` (the two optional
basin data inputs — gauge/output locations and the observed series — which live
outside the repo *and* outside `project_dir`).

#### 1.02 · `delineate_region`

**Does.** Delineates the one project **extent** from `shared.basin.region` plus
the data catalog, via hydromt `parse_region_basin` (ADR 0003). Catalog in,
polygon out — model-free. It splits and names nothing: one or several parent
features, no IDs, no gauges. Every downstream extent comes from this artifact,
never from a built model.

**Writes.** `<spatial>/geoms/region.geojson`.

#### 1.03 · `delineate_spatial_units`

**Does.** The **shared** vector foundation — the same rule WF2 declares as 2.03
and WF3 as 3.04, splatted from one `spatial_units_rule` helper so the three
declarations cannot drift. Partitions the region into the vector layers every
later join is keyed on, and is where **gauge points enter the workflow**: it
snaps `shared.basin.gauge_points` to the river network and partitions each
parent basin into incremental subbasins — gauge-driven where `control` points
exist, automatic otherwise, chosen per basin — creating the `basin_id` →
`subbasin_id` → `wflow_id` identity hierarchy.

Its params are a pure function of `project` + `shared.basin` (ADR 0003 §8b), and
that is a requirement: the projections-only configs carry no
`workflows.build_model` section at all, so a payload drawn from one would
differ per invoking workflow.

**Writes.** `<spatial>/geoms/{basins,subbasins,catchments,rivers,locations}.geojson` ·
`<spatial>/location_registry.csv` · `<spatial>/hydrography.nc`.

`hydrography.nc` is the **seam intermediate** (§8a), not a product: the whole
whole hydrography grid stack crosses the vector/raster boundary in memory, and
re-deriving it in 1.06 would make WF1 read the hydrography twice with two grids
that can drift. It is deliberately absent from `spatial_catalog.yml`.

#### 1.04 · `extract_historical_climate`

**Does.** The **shared** historical-climate store producer — the same rule WF3
declares as 3.08, splatted from one `climate_store_rule` helper so the two
declarations cannot drift. Extracts the configured historical climate for the region and
window, on the source grid, model-free. Its declared inputs are the data catalog
— the store's freshness boundary — and the region polygon.

**Writes.** `<store>/extract_historical.nc`, plus `<store>/orography.nc` on the
chirps branches. The extraction records its own extent in netCDF attributes
(`region_geojson_sha256`, `region_bbox`, `region_source`) rather than in a
sidecar file.

#### 1.05 · `plot_climate_source`

**Does.** The canonical climate figure set on the **source** grid, straight from
the shared store, before any regridding to the model. Its whole subgraph is 1.02
+ 1.04 + this rule, so the figures build with no `<model>/` on disk at all.

**Writes.** `<store>/plots/` — the `figure_names("source")` set.

#### 1.06 · `prepare_spatial_maps`

**Does.** The **raster half** of the spatial foundation, and WF1-only: folds the
thematic layers (`vito` land cover, `modis_lai`, `soilgrids`) onto the grid 1.03
handed it, and writes the model-build interface. The vector layers and the
registry come from 1.03 — declaring the unsplit rule instead would have made a
projections-only run resample all three thematic sources to draw a subbasin
outline (measured 2026-08-06: the split avoids ~71% of that).

The name is narrow — it names one of its three outputs — and was kept
deliberately; `build_spatial_foundation` read less clearly, and `build_` is
reserved here for constructing a *model* (`rule-naming-design.md` amendment 2).

**Writes.** `<spatial>/spatial_maps.nc` · `<spatial>/spatial_catalog.yml` ·
`<spatial>/spatial_report.yml`.

#### 1.07 · `build_wflow_model`

**Does.** Parameterises Wflow-SBM on that spatial foundation via hydromt, then
reopens the written model and verifies its grid and IDs against the spatial
products. Also where **the gauges enter the model**: `setup_gauges` /
`setup_outlets` write the `gauges_locations` and `outlets` maps into staticmaps,
both with `toml_output=None` — maps only, no output declarations. No snapping and
no subcatchment derivation: 1.03 did both.

**Writes.** `<model>/staticmaps.nc` · `<model>/wflow_sbm.toml` ·
`<model>/staticgeoms/region.geojson` · `<model>/staticgeoms/outlets.geojson` ·
`<model>/.model_built` (sentinel).

`wflow_sbm.toml` is created here and then mutated in place by 1.08, 1.09 and
1.10, none of which declare it. That is what the `.model_built` sentinel exists
to handle.

#### 1.08 · `add_reservoirs_lakes_glaciers`

**Does.** Adds waterbodies to the built model (a hydromt update). A temporary
hydromt workaround; can fold back into 1.07 when upstream supports it.

**Writes.** `<model>/staticgeoms/reservoirs_lakes_glaciers.txt`.

**Writes (undeclared).** `<model>/staticmaps.nc` — it commits the waterbody
layers back into the model. That undeclared write is part of why the model-root
readers need a sentinel: Snakemake attributes `staticmaps.nc` to 1.07, so
declaring it there orders nothing after *this* rule.

#### 1.09 · `declare_wflow_outputs`

**Does.** Declares which timeseries Wflow emits — the `[output.csv]` block — for
`outlets` (Q), `gauges_locations` (Q, P) and basin means of any extra
`wflow_outvars`. It adds **no model data**: 1.07 created both gauge maps with
`toml_output=None`, deferring exactly this step. It also re-checks that the
model's gauge IDs still equal `location_registry.wflow_id`, and fails if either
map is absent.

`declare_` is the verb table's 18th entry, added for this rule: 1.08 and 1.10
add model *data* (waterbody layers, forcing grids), while this changes only what
the engine will emit.

**Writes.** `<model>/.outputs_configured` (sentinel).

**Writes (undeclared).** `<model>/wflow_sbm.toml` — the `[output.csv]` block
itself, via `mod.write()` — and `<model>/staticmaps.nc`, which `mod.close()` must
commit or hydromt leaves the new variables stranded in a `staticmaps_<hash>.nc`
temp file.

The gauge-ID re-check is not redundant with 1.07's identical comparison
(`build_wflow_model.py::_validate_written_model`): 1.08 mutates `staticmaps.nc`
in between, so this copy is what catches corruption from that step.

#### 1.10 · `add_climate_forcing`

**Does.** Two steps in one rule. First assembles the hydromt
recipe: a `steps:` YAML holding `setup_config` (`time.starttime`,
`time.endtime`, `time.timestepsecs`, `input.path_forcing`),
`setup_precip_forcing` and `setup_temp_pet_forcing`, with the PET method and
orography source branched off `clim_historical` and the chunksize sized by
opening the model's staticmaps. Then applies it via `hydromt update wflow_sbm`,
which builds the forcing for the model grid and — through the recipe's
`setup_config` step — writes the run window and forcing pointer into the model
TOML.

The window is `workflows.build_model.simulation_window`, falling back to
`shared.historical_window` when unset (2026-08-10). That is the SIMULATION
period, not the extraction period, and it must sit inside the record.

**Reads the climate store, not the catalog.** This rule declares
`extract_historical.nc` as an input and generates a one-entry catalog
(`config/climate_store_catalog.yml`) pointing hydromt at it, so the forcing is
built from the extraction rule 1.04 already made rather than from a second full
pass over the global dataset. `dem_forcing_fn` still resolves from the main
catalog — the store holds no orography.

**Writes.** `<model>/forcing/inmaps_historical.nc` ·
`<model>/config/build_historical_forcing.yml` (the recipe, kept as provenance of
the model it built) · `<model>/.model_final` (sentinel).

**Writes (undeclared).** `<model>/wflow_sbm.toml` — `time.*` and
`input.path_forcing`.

**This rule is the LAST WRITER of the model root, and `.model_final` is what
says so** (ADR 0004). `hydromt update wflow_sbm` calls `mod.write()`, which
rewrites the whole root — staticmaps, the TOML, every `staticgeoms/` layer. Four
rules (1.11, 1.12, 1.13, 1.14) declare that sentinel `ancient()` to order
themselves behind it, and **that is why they are numbered after this rule**: an
`ancient()` input is a real DAG edge with the timestamp trigger suppressed, not
an absent one. **Residual risk, stated because no test can catch it:** the
sentinel is correct only while this rule remains the last writer. A new rule
that mutates the model after it must take the sentinel with it.

#### 1.11 · `write_outlet_index`

**Does.** Joins Wflow's outlets to the deterministic basin/subbasin/location
identities, so a model output can be traced back to a named station. hydromt
labels outlets with basin-derived subcatchment IDs, which are not the registry's
IDs — this is the crosswalk between them, rebuilt on every run.

**Writes.** `<model>/staticgeoms/outlet_index.csv`.

**Consumed by** — and this is why the rule has no downstream node: **no rule
declares this file as an input.** Its consumers are outside the DAG.

- `dev/scripts/check_baseline.py::_read_discharge_series` uses it to resolve
  *which* `Q_*` column of `output.csv` is the primary outlet, once a project has
  more than one gauge station. It matches `compat_station_name == "wflow_1"`,
  takes that row's `subcatchment_id`, and requires `Q_<id>` to be present.
  Without the file a multi-gauge project fails the baseline gate outright. The
  path is **derived** from the run CSV's location
  (`<run>/../staticgeoms/outlet_index.csv`), not passed — which is exactly why
  Snakemake cannot see the edge.
- `dev/reference/contracts/hydrological-model-seam.md` pins it in `validate_hm3`
  as a persisted model-root artifact.

It is a member of `WF1_TERMINALS`, so it is a `rule all` target and an input of
both gather rules. **Do not prune it as stray output** — `check_baseline.py`'s
own module docstring records that it is fingerprinted beyond `rule all` for this
reason.

**Not merged into 1.09**: its inputs are
`outlets.geojson` and the registry, so it runs in parallel with the waterbody
and output-declaration rules. Merging would serialise a cheap pandas join behind
a hydromt `r+` mutation — it *adds* an edge.

#### 1.12 · `plot_basin_map`

**Does.** Draws every figure the shared spatial foundation supports: `basin_area`
— basin, rivers, gauges and the DEM on one map — plus the thematic family beside
it, the subbasin delineation, land cover, leaf area index and the topsoil
properties.

ONE rule for both, because they are one deliverable. The thematic maps draw the
same vector overlay and deliberately suppress its legend *because* `basin_area`
carries the key; a run that produced one set without the other would ship ten
maps whose linework nothing explains. Both halves are leaves, so splitting them
would duplicate the vector inputs and the plots directory for no scheduling gain.

**Reads.** The spatial foundation only — `hydrography.nc` and the vector layers
from 1.03, and `spatial_maps.nc` from 1.06. The 1.06 edge arrived with the
thematic family; before it, this rule depended on 1.03 alone. It costs nothing,
since 1.06 is upstream of the model build and nothing downstream waits on this
leaf.

ADR 0007 already retired this rule's `staticmaps.nc` read, and with it the
`.model_final` sentinel edge that existed to stop a concurrent `-c 3` run
aborting below Python on an unlocked HDF5 read. It opens no model file at all.

**Writes.** `data/spatial/plots/`, **PNG only**. The vector
deliverable was dropped across the whole figure set because nothing in the
toolbox or the platform read it; 600 dpi at 180 mm carries the figure everywhere
it is used, and not serialising each one twice halves the rule's render time.
`figure_paths()` still takes a `formats` argument, so a caller preparing a
manuscript can ask for a PDF. The thematic list is declared from that function,
the same contract 1.13 has with `climate_figures.figure_names()`.

Not every figure is *declared*. A figure whose source variable is specific to one
catalog source — `soil_depth_to_bedrock`, which reads soilgrids v1.0's own
`BDTICM` filename and has no soilgrids_2020 equivalent — is drawn but left out of
`output:`. Declaring it would fail the RULE on a project that merely chose the
other soil source, which is a workflow crash in exchange for one figure.
`data/spatial/plots/` is a directory in the tree inventory, so the undeclared
renders are still accounted for.

**No title, no overlay key**, on the thematic half only. The filename names the
figure and `basin_area` carries the legend; see `shared/plot_spatial_maps.py`.

#### 1.13 · `plot_forcing`

**Does.** Draws the canonical climate figure set for the model's own forcing —
the same figures 1.05 draws for the source grid, so the two directories answer
"what did the downscaling change?" side by side.

**Writes.** `<model>/forcing/plots/` — the full variable × kind cross-product
from `climate_figures.figure_names("forcing")`, all declared.

#### 1.14 · `run_wflow`

**Does.** Runs Wflow.jl once on that historical forcing, driven by the model's
own TOML.

**Writes.** `<model>/run_default/output.csv` — `temp()`, so a successful run
does not leave it: rule 1.14b derives the readable per-variable tables from it
and rule 1.15 reads it for the metrics, then Snakemake drops it. Run with
`--notemp` to keep it (the baseline gate pins it, and iterating on a 1.15 figure
otherwise re-runs the whole model).

#### 1.15 · `plot_wflow_evaluation`

**Does.** Scores the Wflow run against observations where they exist and draws
the evaluation figures — four sheets per station plus the basin-average series.
One rule, both products.

**Writes.** `<model>/evaluation/performance_metrics.csv` · one
`<var>_basavg.png` per basin-average `wflow_outvars` entry.

**Writes (undeclared).** Per station, keyed by `wflow_id`:
`hydrograph_<id>`, `signatures_peaks_<id>`, `signatures_lows_<id>` and
`performance_<id>`, one PNG each. Their count is a product of the model
build (outlets and subcatchments), not of config, so they cannot be enumerated
at parse time — and neither can their NAMES, because a wflow_id
is assigned by rule 1.03. Only the hydrograph is drawn without observations; the
other three need them, and the two extremes sheets additionally need a run
longer than a year.

**Why no figure is declared** — figures name a point by its `wflow_id`, as every
other artifact of a run does, so no single figure filename is predictable at parse
time. The metrics table is this rule's terminal in `WF1_TERMINALS` instead. The rename also closed two defects the old key hid: an
outlet and a gauge on one cell were plotted twice under two names, and an
observation at the basin outlet matched nothing at all, because observations are
keyed by `wflow_id` while the outlet series is keyed by its subcatchment id.
`plot_results.resolve_stations` is where both are now handled.

**Why the metrics and the figures share a rule** — `performance_metrics.csv` is
baseline-covered data while the figures are excluded, and the DAG cannot express
that distinction; splitting anyway is not affordable, because the metrics are one `compute_metrics` call *inside* the
figure loop, downstream of the model open, the gauge-name resolution, the merge,
the alignment and the climate-parity transform. Splitting means either
duplicating the parity work or adding a declared intermediate, and the harm it
fixes is a wasted re-run, not a wrong number. **Consequence to know when reading
the `AGENTS.md` validation ladder:** a plot-only edit here still re-runs the
whole rule and rewrites identical metrics, so the gate passes but is not free.

#### 1.16 · `gather_benchmarks`

**Does.** Merges the per-rule timing parts into one table with a rule column and
a TOTAL row, rewritten fresh each run. Takes the terminal set as input, which is
what schedules it last.

**Writes.** `benchmarks/wf1_benchmarks.md`.

#### 1.17 · `gather_logs`

**Does.** Merges every WF1 log part into one workflow log in rule order, then
deletes the parts it consumed and prunes the emptied directories. After a
**partial** re-run the untouched sections are marked "no part from this run" —
the artifact describes the run that produced it, not an accumulated history.

**Writes.** `logs/wf1_build_model.log`.

## Two meanings of "subbasin"

In `shared.basin.region` (rule 1.02) `subbasin:` is **hydromt's** region
keyword — "everything upstream of this point, above `uparea`" — and it selects
the project extent. CST's `subbasins.geojson` (rule 1.03) is a different thing:
the incremental partition *within* that extent. A project can be
`{'basin': ...}` at 1.02 and still have twelve subbasins at 1.03.

## Where a gauge point lives, rule by rule

`shared.basin.gauge_points` (`station_name, x, y, location_role[, wflow_id]`) is
consumed once, by 1.03, and everything after that reads its derived identities:

| stage | rule | what happens to the point |
|---|---|---|
| enters | 1.03 `delineate_spatial_units` | snapped to a river cell, given `location_id`/`wflow_id`; a `control` point also becomes a subbasin outlet |
| enters the model | 1.07 `build_wflow_model` | written into `staticmaps.nc` as `gauges_locations`, no TOML output |
| becomes an output | 1.09 `declare_wflow_outputs` | named in `[output.csv]`, so Wflow emits its timeseries |
| becomes joinable | 1.11 `write_outlet_index` | `outlet_index.csv` maps Wflow's subcatchment IDs back to the named station |

---

# WF2 — climate projections (`analyze_projections.smk`)

A plausibility overlay, not a driver. Computes monthly CMIP6 change factors that
situate the stress-test grid in projection space. **Nothing here feeds a
stress-test run.**

No model anywhere in this workflow — it is data end to end.

```
STAGE 1 — DATA
──────────────────────────────────────────────────────────────────
                        config + catalogs
                                │
        2.01 snapshot_config ───┤
                                ▼
                      2.02 delineate_region
                                │  region.geojson
              ┌─────────────────┼─────────────────┬──────────────────┐
              │                 │                 │                  │
  CMIP6 store │ (gs://cmip6)    │                 │                  ▼
              ▼                 │                 │      2.03 delineate_spatial
    2.04 fetch_gcm_slice        │                 │           _units
    (one raw slice per member;  │                 │      (SHARED, = 1.03/3.04.
     the ONLY remote read)      │                 │       A LEAF here: nothing
              │                 │                 │       in WF2 consumes it yet)
              ▼                 │                 │                  │
    2.05 reduce_gcm_series ◄────┘                 │                  │
    (one job per series key, full fan-out)        │                  │
              │                                   │                  │
STAGE 2 — PRODUCT                                 │                  │
──────────────────────────────────────────────────────────────────   │
              ▼                                   │                  │
    2.06 derive_change_factors ◄──────────────────┘                  │
    (ONE job — the workflow's answer)                                │
              │                                                      │
              ├──► summary/*_change_factors_{annual,monthly}.csv     │
              │    composition.csv · provenance.json · report.md     │
              │    plots/overview/change-factor-cloud[-combined].png │
              ▼                                                      │
STAGE 3 — FIGURES + RECORDS                                          │
──────────────────────────────────────────────────────────────────   │
    2.07 plot_gcm_timeseries   (reads 2.05's series, not 2.06)       │
              │                                                      │
              ▼                                                      ▼
    2.08 gather_benchmarks · 2.09 gather_logs ◄───────────────────────
```

The region polygon feeds **four** rules — 2.03, 2.04, 2.05 and 2.06 all declare
it — because stage B recomputes every expected digest, including the polygon
fingerprint. 2.07's edge from 2.06 is an **ordering edge only**; it plots the
per-member series from 2.05 and never opens the change-factor table.

**2.03 is a leaf, and both gather rules declare it explicitly.** Nothing in WF2
consumes the vector layers yet (ADR 0003 §10 leaves the consuming rules
unnamed), so without that edge it would run in parallel with the merge and
strand its log part under `_parts/` — the defect the `LOG_RULES` comments record
three times over. A `rule all` target entry is separately what makes it
reachable at all: an undeclared leaf is simply never scheduled.

| # | rule | in one line |
|---|---|---|
| 2.00 | `all` | Target aggregator. |
| 2.01 | `snapshot_config` | As WF1 1.01. |
| 2.02 | `delineate_region` | As WF1 1.02 — the same artifact. |
| 2.03 | `delineate_spatial_units` | As WF1 1.03 — the same artifacts. A leaf here. |
| 2.04 | `fetch_gcm_slice` | Acquires one raw CMIP6 slice. The only remote read. |
| 2.05 | `reduce_gcm_series` | Stage A: one local slice → one monthly series. |
| 2.06 | `derive_change_factors` | Stage B, one job. WF2's terminal product. |
| 2.07 | `plot_gcm_timeseries` | The eight projection figures. |
| 2.08 | `gather_benchmarks` | Merges the timing parts. |
| 2.09 | `gather_logs` | Merges the log parts. |

## WF2 rule detail

#### 2.00 · `all`

**Does.** Target aggregator — the change-factor summaries plus the projection
plots, the merged log and the benchmark table.

**Writes.** Nothing of its own.

#### 2.01 · `snapshot_config`

**Does.** As WF1 1.01, with the WF2 bins.

**Writes.** `config/runs/snake_config_analyze_projections.yml` ·
`config/runs/analyze_projections/<digest>/` (bundle dir).

**Writes (undeclared).** Catalog copies into `config/catalogs/`.

#### 2.02 · `delineate_region`

**Does.** As WF1 1.02 — the same one project region artifact, from the same
shared spec (ADR 0003), which is why a projections-only run does not trigger
a full climate extraction just to learn a basin outline.

**Writes.** `<spatial>/geoms/region.geojson`.

#### 2.03 · `delineate_spatial_units`

**Does.** As WF1 1.03 — the same shared vector foundation, from the same helper.
WF2 declares the **vector half only**: the thematic raster stack stays WF1-only,
so a projections-only run obtains basin and subbasin boundaries without reading
`vito`, `modis_lai` or `soilgrids` at all. That is the whole point of ADR 0003
§8's split — `snakemake -n` on this file must list `delineate_spatial_units` and
no job whose inputs mention those three sources, which is §8's acceptance
assertion.

What it buys WF2: a context map beside the change-factor plots, and the option
of subbasin-resolved indicators. It does not yet **consume** them (§10).

**Writes.** As 1.03.

#### 2.04 · `fetch_gcm_slice`

**Does.** Acquires one raw CMIP6 slice for a (model, scenario, member) key.
**The only rule that reads the remote store.** Split from the reduction because
the costs differ by four orders of magnitude — measured 2026-07-30: ~1142 s to
open a remote source, ~19 s to transfer, ~0.2 s to reduce — so a reducer edit
must not re-download. Its params carry `raw_digest_components`, deliberately
excluding the reducer hash; passing the full set here would silently undo the
split while every test still passed.

**Writes.** `<proj>/raw/<series_key>.nc` — persistent and `update()`-flagged,
because Snakemake removes outputs in `Job.prepare()` and the revalidate-and-skip
cache would otherwise never fire.

#### 2.05 · `reduce_gcm_series`

**Does.** Stage A. Reduces one **local** raw slice to a monthly series over the
region polygon, for its (model, scenario, member) key. One job per key, no edges
between series, no network call.

**Writes.** `<proj>/scalar/<series_key>.nc` — persistent + `update()`, same
reason as 2.04.

#### 2.06 · `derive_change_factors`

**Does.** Stage B, a **single job**: turns every reduced series into the change
factors per model, scenario and horizon. Asserts that the set of series it opens
equals its declared input list, so a model dropped from the config cannot rejoin
through a leftover file, and recomputes every expected digest including the
polygon fingerprint. WF2's terminal product — and, despite the `derive_` name, it
also renders one figure and writes the run's provenance and human-readable
report. Kept as one rule deliberately: the design gives stage B no fan-out.

**Writes.** `<proj>/summary/<clim_project>_change_factors_annual.csv` ·
`_monthly.csv` · `<proj>/summary/composition.csv` ·
`<proj>/summary/provenance.json` · `<proj>/report.md` ·
`<proj>/plots/overview/change-factor-cloud.png`, plus
`overview/change-factor-cloud-combined.png` when more than one horizon is
configured (a single horizon has no cloud travel to show).

#### 2.07 · `plot_gcm_timeseries`

**Does.** Draws the two annual overviews (absolute and anomaly panels) from the
per-member series of 2.05, and one monthly change-factor figure per configured
horizon. Since 2026-08-17 the monthly figures RENDER
`summary/*_change_factors_monthly.csv` rather than recomputing it: that table is
a real input, opened and read. The annual table stays an **ordering edge only**.

**Writes.** Eight PNGs under `<proj>/plots/`, named
`<clim_project>_{precip,temp}_{annual,monthly}_{absolute,change}.png`. All eight
are declared, so none is invisible to Snakemake.

#### 2.08 · `gather_benchmarks`

**Does.** As WF1 1.16, for WF2.

**Writes.** `benchmarks/wf2_benchmarks.md`.

#### 2.09 · `gather_logs`

**Does.** As WF1 1.17, for WF2. Replaces two per-stage gathers that merged only
the fan-out rules, so following one run meant opening five files and knowing
their order.

**Writes.** `logs/wf2_analyze_projections.log`.

---

# WF3 — climate experiment (`run_stress_test.smk`)

The stress test itself. Generates stochastic weather realizations, perturbs each
across a temperature × precipitation grid, runs every member through Wflow, and
reduces the runs to the indicator tables that form the response surface.

Every climate artifact is generated **before** the model is used: 3.14 is the
first rule to put the model to work, and the whole stress-test ensemble already
exists by then.

```
STAGE 1 — GUARD + PROVENANCE   (config and hashes only)
──────────────────────────────────────────────────────────────────
   config ──► 3.01 check_project_consistency   (drift guard, fails loud)
                 │         .project_consistency_ok
                 │
                 │   ┌─────────────┬─────────────┬─────────────┐
                 └──►│             │             │             │
                     ▼             ▼             ▼             ▼
              3.02 snapshot   3.07 write_    3.09 prepare  3.10 prepare_
                 _config      experiment_    _stress_test  weathergen_
                              config             _grid        config

   config ──► 3.03 delineate_region      (guard-independent: its only
                     │  region.geojson    input is the data catalog, and
                     │                    the byte-identity contract with
       ┌─────────────┴───────────┐        WF1/WF2 forbids adding one)
       ▼                         ▼
  3.04 delineate_        3.08 extract_historical_climate
       _spatial_units         (SHARED with WF1 1.04)
   (SHARED, = 1.03/2.03;
    a LEAF here too)

   model  ──► 3.05 write_model_reference  (inputs: WF1's .outputs_configured
                     │                     + wflow_sbm.toml, both ancient)
                     ▼
              3.06 check_model_reference   (verdict consumed by 3.14)

STAGE 2 — CLIMATE DATA   (the model is fingerprinted, never used)
──────────────────────────────────────────────────────────────────
   3.08 extract_historical_climate      3.10 prepare_weathergen_config
              │  extract_historical.nc          │  weathergen_config.yml
              └────────────────┬────────────────┘
                               ▼
              3.11 generate_weather_realizations
                               │  rlz_1..R_st_0.nc   (unperturbed)
   3.09 prepare_stress_test_grid        │
              │  stress_test_lookup.csv │
              └────────────────┬────────┘
                               ▼
                  3.12 perturb_climate_realization
                         │  rlz_<n>_st_<m>.nc   (perturbed)
                         ▼
                  3.13 write_climate_data_catalog
                         │
STAGE 3 — MODEL RUN   (first use of the built model)
──────────────────────────────────────────────────────────────────
                         ▼
       3.14 downscale_climate_realization ◄── model + 3.06's verdict
                         │  inmaps + per-member TOML
                         ▼
                  3.15 run_wflow_batch_<b>   (B members per Julia session)
                         │  per-member run CSVs
                         ▼
STAGE 4 — PRODUCT + RECORDS
──────────────────────────────────────────────────────────────────
                  3.16 derive_wflow_indicators
                         │  q_indicators.csv · basin_indicators.csv
                         ▼
                  3.17 gather_benchmarks · 3.18 gather_logs
```

**The store feeds 3.11, not 3.09.** 3.09 enumerates the stress-test grid from
the config alone — it needs no climate data at all, and runs concurrently with
the extraction. The historical climate is what the *generator* resamples.

**The guard's fan-out is 3.01 → {3.02, 3.07, 3.09, 3.10}** — the four rules that
declare `consistency_ok`. An earlier version of this diagram drew it reaching
`delineate_region` and `write_model_reference` as well; neither declares it, and
`delineate_region` structurally *cannot* — its input set is splatted from the
shared rule helper and adding a WF3-only input would break the byte-identity
contract `test_region_rule.py` enforces. The same is true of 3.04, for the same
reason. `write_model_reference` hangs off the built model instead
(`.outputs_configured` + `wflow_sbm.toml`, both `ancient()`).

**3.04 is a leaf here as it is in WF2**, and both gather rules declare it for
the same reason. Note the scope mismatch, ruled 2026-08-06: everything else in
`WF3_TARGETS` is experiment-scoped and this one is **project**-scoped, because
the vectors depend on `shared.basin` alone — which 3.01 guarantees agrees across
workflows. Two experiments on one project share one copy, and that is what makes
the shared declaration safe.

| # | rule | in one line |
|---|---|---|
| 3.00 | `all` | Target aggregator. |
| 3.01 | `check_project_consistency` | Startup drift guard against the wf1/wf2 snapshots. |
| 3.02 | `snapshot_config` | As WF1 1.01, kept inside the experiment. |
| 3.03 | `delineate_region` | As WF1 1.02 — the same artifact. |
| 3.04 | `delineate_spatial_units` | As WF1 1.03 — the same artifacts. A leaf here. |
| 3.05 | `write_model_reference` | Records which model state this experiment used. |
| 3.06 | `check_model_reference` | Refuses to simulate if that model has changed. |
| 3.07 | `write_experiment_config` | Records the experiment's own parameters. |
| 3.08 | `extract_historical_climate` | The shared climate store (= WF1 1.04). |
| 3.09 | `prepare_stress_test_grid` | **Creates** the stress test: one lookup table, twelve rows per grid point. |
| 3.10 | `prepare_weathergen_config` | The one weather-generator config. |
| 3.11 | `generate_weather_realizations` | All `RLZ_NUM` unperturbed realizations, in one call. |
| 3.12 | `perturb_climate_realization` | **Applies** one grid point to one realization. |
| 3.13 | `write_climate_data_catalog` | Catalogs every generated climate file. |
| 3.14 | `downscale_climate_realization` | One member onto the Wflow grid: forcing + TOML. |
| 3.15 | `run_wflow_batch_<b>` | Runs Wflow.jl, `B` members per Julia session. |
| 3.16 | `derive_wflow_indicators` | The indicator tables, one per output variable. WF3's terminal product. |
| 3.17 | `gather_benchmarks` | Merges the timing parts. |
| 3.18 | `gather_logs` | Merges the log parts. |

## WF3 rule detail

#### 3.00 · `all`

**Does.** Target aggregator — the two indicator tables, the three config
records, the merged log and the benchmark table.

**Writes.** Nothing of its own.

#### 3.01 · `check_project_consistency`

**Does.** Startup drift guard. A WF3 config is a *full* config, so its
project-level sections must describe the same project the built model came from;
this fails loud on divergence, **naming the diverging key**, rather than letting
the experiment silently reuse a model built under other settings. Runs at rule
time, not parse time, so `--dry-run` and `--unlock` stay usable.

**Writes.** `<exp>/.project_consistency_ok` (per-experiment sentinel, a fresh
input of the per-experiment roots) · `<store>/.guard_ok` (store-level receipt,
consumed `ancient()` and keyed identically for every experiment sharing dataset +
window, so the shared rule's input set never varies across experiments).

#### 3.02 · `snapshot_config`

**Does.** As WF1 1.01, but the snapshot stays **inside the experiment** rather
than joining `config/runs/`.

**Writes.** `<exp>/config/snake_config_run_stress_test.yml` ·
`<exp>/config/runs/run_stress_test/<digest>/` (bundle dir).

**Writes (undeclared).** Catalog copies into `<exp>/config/catalogs/`.

#### 3.03 · `delineate_region`

**Does.** As WF1 1.02 — the same one project region artifact.

**Writes.** `<spatial>/geoms/region.geojson`.

#### 3.04 · `delineate_spatial_units`

**Does.** As WF1 1.03 — the same shared vector foundation, from the same helper,
and byte-identical to the other two declarations but for
`message`/`log`/`benchmark` (`tests/test_spatial_units_rule.py` fails on any
other difference).

What it buys WF3: the subbasin partition and the location registry as
**project**-scoped artifacts — the option of subbasin-resolved indicators and a
station-labelled indicator table — without a built model and without the
thematic raster stack. It does not yet **consume** them (ADR 0003 §10).

**Writes.** As 1.03.

#### 3.05 · `write_model_reference`

**Does.** Records **which model state** this experiment used: the model's
relative path, a pointer-derived digest, and the per-input hashes behind it. Not
a copy — a hash answers the question a duplicated staticmaps would, and the
per-input hashes are kept so a later mismatch can *name* what changed. Its model
inputs are `ancient()` on purpose: if the reference were rewritten whenever the
model changed it would always match, and 3.06's comparison would be decorative.

**Writes.** `<exp>/config/model_reference.yml`.

#### 3.06 · `check_model_reference`

**Does.** The other half: recomputes the fingerprint and refuses to simulate if
the live model has changed since the experiment was recorded. Its sentinel is a
declared input of 3.14 — the first rule to touch the model — because a check
after the work is a post-mortem, not a guard.

**Writes.** `<exp>/.model_reference_ok` — `temp()`, and that is the trigger, not
an optimisation. A persisted sentinel would satisfy 3.14's edge with a **stale
verdict**: the check passed once, the file remains, and 3.14 is free to
re-simulate against a model that changed afterwards. Deleting it on consumption
forces the next invocation to re-evaluate. A guard evaluates; it does not cache
an answer.

**Do not merge 3.05 and 3.06.** They read as an obvious pair and merging them
destroys the guard — the `ancient()` / `temp()` asymmetry above *is* the
mechanism, not an accident.

#### 3.07 · `write_experiment_config`

**Does.** Records the experiment's own parameters, separately from the project
ones. Generated, never authored — a hand-written file here would be a second
source of truth competing with the `--configfile`. Immutable from the first
*successful* run, keyed off the merged workflow log's existence, since editing an
experiment's parameters before it has produced anything is ordinary work and
afterwards would silently redefine what the existing results mean.

**Writes.** `<exp>/config/experiment.yml`.

#### 3.08 · `extract_historical_climate`

**Does.** The shared historical-climate store producer — the same rule as WF1
1.04, byte-identical but for `message`/`log`/`benchmark`, with
`tests/test_climate_store_contract.py` failing on any other difference. Usually
already current when run in pipeline order.

**Writes.** `<store>/extract_historical.nc`, plus `<store>/orography.nc` on the
chirps branches.

#### 3.09 · `prepare_stress_test_grid`

**Does.** Enumerates the configured temperature × precipitation grid and writes
ONE lookup table at monthly grain: twelve rows per stress-test point, carrying
the temperature change and the precipitation mean and variance changes.
**This is what creates the stress test.**

**Writes.** `<exp>/config/stress_test_lookup.csv` — `12 × ST_NUM` rows keyed
`(st_id, month)`, `st_id` zero-padded to a width derived from `ST_NUM` (so
`01 … 12` on a twelve-point grid) and textually identical to the member token.
**No `st_0` row**: the table is the parameter grid, and the reserved unperturbed
baseline has no parameters. Values are PERCENT for both precipitation columns
and additive °C for temperature. Still one loop, so the enumeration that names
the members and the one that describes them cannot disagree (C26).

Replaced `<wg>/_work/st_<m>.csv` plus `<exp>/config/stress_test_design.csv` on
2026-08-16; `_work/` is gone. Migration record:
`dev/milestones/r12/migration_stress-test-lookup.md`.

#### 3.10 · `prepare_weathergen_config`

**Does.** Assembles the one weather-generator config from the shipped template
plus the project settings — the year arithmetic (middle year, simulation length)
and the two transient-change flags. The template is a **declared input**: until
2026-08-05 it was a params-only read, so editing it changed nothing until
something else forced a rerun, and 3.11 kept generating from superseded settings.

**Writes.** `<wg>/config/weathergen_config.yml`.

#### 3.11 · `generate_weather_realizations`

**Does.** Runs weathergenr **once** to produce all `RLZ_NUM` stochastic
realizations of the historical climate — the unperturbed `st_0` baselines. The
plural is load-bearing: number carries meaning here, with 3.11 plural (all in one
job) against 3.14 singular (wildcarded, one job per member).

**Writes.** `<wg>/output/rlz_1_st_0.nc` … `rlz_<RLZ_NUM>_st_0.nc`, all
`temp()`.

**Writes (undeclared).** Four generator diagnostic figures moved into
`<wg>/plots/` (`obs_power_spectra.png`, `warm_annual_precip.png`,
`warm_annual_stats.png`, `warm_annual_wavelet.png`) and weathergenr's date CSVs
left in `<wg>/output/`.

#### 3.12 · `perturb_climate_realization`

**Does.** Takes one unperturbed realization and one stress-test point and
applies that perturbation — precipitation mean and variance factors, temperature
delta, transient flags, PET recompute. **It applies the stress test; 3.09 creates
it.** Its `st_num` wildcard is constrained to ≥ 1 so it can never become a second
producer of the reserved `st_0` baseline, which would surface as a cyclic-graph
error.

Its parameter input is the **constant** lookup, not a per-member file: the member
id arrives as a positional argument and `read_member_grid.R` filters on it,
stopping unless the slice is twelve rows in month order. That guard exists
because the migration turned a structural `MissingInputException` into a quiet
data condition — a join matching nothing yields a zero-length vector, and R's
recycling makes a silent wrong answer at least as likely as an error. It also
converts both percent columns back to the generator's multiplier form.

**Writes.** `<wg>/output/rlz_<n>_st_<m>.nc`, `temp()`.

#### 3.13 · *(removed 2026-08-18)*

Was `write_climate_data_catalog`: it enumerated every generated climate file —
perturbed and unperturbed — into ONE hydromt catalog the downscaling step read a
single entry out of. The fan-in was the cost: no member could be downscaled
until every member had been perturbed, and the perturbed NCs are `temp()`, so
all of them had to coexist on disk until this rule had read them. 3.14 now
writes its own one-entry catalog per member. The number is not reused — `W.NN`
is an id, not a position (`dev/reference/naming.md` §9).

#### 3.14 · `downscale_climate_realization`

**Does.** Downscales one perturbed realization onto the Wflow grid via hydromt,
producing that member's forcing and its run TOML. The first rule to touch the
model, which is why 3.06's guard sentinel is a declared input here. Writes the
member's own one-entry hydromt catalog first: a bare path cannot carry
`preprocess=harmonise_dims` or `crs=4326`, because hydromt_wflow's setup methods
pass no `source_kwargs`.

**Writes.** `<runs>/forcing/inmaps_rlz_<n>_st_<m>.nc` (`temp()`) ·
`<runs>/config/rlz_<n>_st_<m>.toml` · `<runs>/config/rlz_<n>_st_<m>.yml`
(`temp()`).

#### 3.15 · `run_wflow_batch_<b>`

**Does.** Runs Wflow.jl for every member, `B` per Julia session to amortise
startup, through a parse-time loop of one anonymous rule per batch with static
per-member input/output lists. `B` defaults from `-c N` and is clamped by
`batch_size_max`; `batch_size: 1` restores one job per member. Rule identifiers
are per batch while the log label stays the singular `3.15_run_wflow` —
deliberately, so **this rule is exempt from the rename call-site rule**. P3-3
keys logs by batch id, not by rule identifier; applying the six-call-site rule
mechanically here would rename a `LOG_RULES` entry that has no rule to match and
break the merge.

**Writes.** `<runs>/output/rlz_<n>_st_<m>.csv` per member ·
`<runs>/output/outstates_rlz_<n>_st_<m>.nc` per member (`temp()`).

#### 3.16 · `derive_wflow_indicators`

**Does.** Reduces every member's run to the indicator tables that form the
response surface — one per configured output variable. WF3's terminal product.
Reads **no parameter artifact at all**: it needed the per-member grid for the
axis values, which are now derived at reporting time from the lookup (HM-7), and
the design table for the id width, which comes from `index_width(st_num)`.
Verifies before any reduction work that the members which actually RAN cover
`ST_START..ST_NUM` — what ran, rather than what was declared.

**Writes.** `<exp>/results/<token>_indicators.csv`, five columns
(`metric, location, st_id, rlz_id, value`). The axis columns were removed on
2026-08-16: they held an annual collapse of twelve monthly perturbations, which
misreports any seasonal design.

#### 3.17 · `gather_benchmarks`

**Does.** As WF1 1.16, for WF3.

**Writes.** `benchmarks/wf3_benchmarks_<experiment>.md`, merging
`benchmarks/_parts/<experiment>/3.*`.

#### 3.18 · `gather_logs`

**Does.** As WF1 1.17, for WF3 — where the merge earns most: 3.12 and 3.14 write
one part per (rlz, cst) and 3.15 one per batch, so the part dir held hundreds of
files across several subdirectories. A clean full run leaves one.

**Writes.** `logs/wf3_run_stress_test_<experiment>.log`, merging
`logs/_parts/<experiment>/3.*`.

> Both are **project-scoped**, keyed by experiment in the filename, so every
> workflow's run records sit in one `logs/` and one `benchmarks/`. The
> scratch `_parts/` stay experiment-scoped one level down — WF3 part names are
> rule numbers, identical across experiments, so a shared part dir would let one
> experiment's stranded part be merged into another's log.

---

## Do not merge these rules

Each pairing looks mergeable and is not. Stated so the case is not re-raised.

| Pairing | Why it stays split |
|---|---|
| `write_outlet_index` into `declare_wflow_outputs` | Paired thematically, not structurally. `write_outlet_index` reads only `outlets.geojson` and `location_registry.csv`, so it runs in parallel with the waterbody and output-declaration rules. Merging serialises a cheap pandas join behind a hydromt `r+` mutation — it *adds* an edge |
| `gather_benchmarks` with `gather_logs` | Both merge functions call `_remove_parts`, deleting the parts they consumed. In one rule, a failure in the second half strands the first half's already-deleted parts, and the re-run degrades that artifact to "no part from this run". Split, either one succeeding means its output survives the other's failure |
| `write_model_reference` with `check_model_reference` | The `ancient()` / `temp()` asymmetry *is* the guard. See 3.06 |
| `plot_wflow_evaluation` into a metrics rule and a figure rule | The seam is not there: the metrics are one call inside the figure loop, downstream of the model open, the merge, the alignment and the parity transform. Splitting costs either a duplicated parity transform or a new declared artifact. See 1.15 |

**Two rules for judging any such candidate:**

1. Two rules being small, adjacent and thematically similar is not an argument for merging them. Check what each actually **depends on**, and whether either **destroys its own inputs**.
2. **A function boundary is not a data boundary.** Before splitting a rule, list what the second half would have to **reload or recompute** — not which functions it would call. A split is affordable only when that list is short or the intermediate is worth declaring.

## Where the rules meet the artifacts

For what each rule reads and writes, rather than what it does:

- `dev/reference/workflows/model_creation.md`, `climate_experiment.md` — per-workflow detail.
- `dev/reference/contracts/weather-generator-seam.md`, `hydrological-model-seam.md` — the pinned interchange surfaces.
- `dev/milestones/r09/wf3-changes-proposal.md` appendix — the WF3 chain step by step, with the declared inputs of each stage.
- `dev/milestones/r10/rule-naming-design.md` — the verb vocabulary and the rename rationale.
