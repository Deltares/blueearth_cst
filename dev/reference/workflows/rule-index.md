# Rule index — every Snakemake rule, all three workflows

One page listing every rule in `Snakefile_model_creation`,
`Snakefile_climate_projections` and `Snakefile_climate_experiment`, what each
one does, what it writes, and how they connect.

> **This page describes what is on disk.** The R10 step-6 sweep landed
> 2026-08-06: the twelve renames (`dev/milestones/r10/rule-naming-design.md`,
> record `dev/milestones/r10/migration_rule-names.md`) and the positional
> renumber (`dev/followups-archive.md` `[R10-5]`).
>
> **Any `W.NN` written before that date means a different rule.** Translate
> with [What changed](#what-changed) — it is the permanent translation table,
> not a transitional note — before reading a rule number in `dev/milestones/`,
> `DEVLOG.md`, `dev/decisions/` or a dated migration record.

Each workflow gets a diagram, a one-line summary table, then one section per
rule. **Does** is the rule's job; **Writes** transcribes its `output:` block —
so the claim can be checked against the Snakefile rather than believed.

## On the numbers

`W.NN` is the rule's position in its workflow's **logical order**: data first,
then model build, then run, then records. Numbering is contiguous within each
workflow and every dependency points from a lower number to a higher one, so a
rule can never depend on something numbered after it.

That is a change of policy, ruled 2026-08-06. The number used to be a *stable
identifier assigned at rule creation*, which left it uncorrelated with order —
gaps at 1.07/2.05/2.08/2.09/3.05, WF2 defined out of numeric order, letter
suffixes stacked five deep beside 3.01, and `gather_benchmarks` sitting at 2.10
beside siblings at 1.14 and 3.12. See [What changed](#what-changed) for what
this costs.

**"Every dependency low→high" is checked against `input:`, `ancient()`
included.** `ancient()` suppresses the timestamp rerun-trigger; it does not
remove the DAG edge. Two rules move further than a reader of the previous map
would expect for exactly this reason — see the note under the WF1 table.

**Going forward: do not renumber to insert a rule.** Use a letter suffix
(`1.09b`) until the next deliberate sweep. Renumbering is a migration, not an
edit — see `[R10-5]`.

## What changed

The only place this page names the old numbers and names. Everything after this
section is the target state.

### Renumbering (`[R10-5]`)

Read this table before interpreting any `W.NN` in a document written before
2026-08-06. **47 identifiers**: 18 in WF1, 10 in WF2, 19 in WF3. The `was`
column carries the old name too wherever the rename and the renumber coincide,
so one lookup answers both.

(An earlier draft of this table published 45. It reconciles as −1 for
`evaluate_wflow_run`, which `[R10-2]`'s drop means never exists, and +3 for the
`delineate_spatial_units` rule `[R10-6]` §8 added to each workflow.

`rule-naming-design.md` says **34**, and both numbers are right: it counts
distinct *identifiers*, this page counts *declarations*. Seven rules are
declared in more than one workflow — `all`, `snapshot_config`,
`delineate_region`, `delineate_spatial_units`, `gather_logs` and
`gather_benchmarks` three times each, `extract_historical_climate` twice — which
is 13 declarations beyond their first. 47 − 13 = 34.)

**WF1** — `Snakefile_model_creation`

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

**WF2** — `Snakefile_climate_projections`

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

**WF3** — `Snakefile_climate_experiment`

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

> **The cost, stated plainly: numbers are REUSED, so old references now resolve
> to the wrong rule.** New 1.07 is `build_wflow_model`; old 1.07 was
> `setup_runtime`, the rule `[R10-1]` merged away. New 1.11 is
> `write_outlet_index`; old 1.11 was `plot_results`. New 3.05 is
> `write_model_reference`; old 3.05 was the deleted `prepare_weagen_config_st`.
> Sharpest of all, new 3.10 is `prepare_weathergen_config` where old 3.10 was
> `run_wflow` — a stale reference to "3.10" now points from the model run to a
> config-assembly rule. Under the previous policy a retired number stayed a gap
> and a stale reference was merely dangling — obvious. Now it silently resolves
> to a different rule.
>
> Every `W.NN` in `dev/milestones/`, `DEVLOG.md`, `dev/decisions/` and the
> Snakefile comments predates this table and must be read **as of its date**.
> That is the price of positional numbers, and it was accepted knowingly.

### Twelve renames (R10)

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

### One structural change

| | change | why |
|---|---|---|
| `[R10-1]` | **`setup_runtime` merges into `add_climate_forcing`** (old 1.07 into old 1.08, now 1.10) | it wrote a hydromt forcing build recipe whose only consumer was the next rule. Two rules, one job — and a recipe that never leaves the pair needs no name of its own, so the naming problem disappears with the rule instead of being renamed around |

**A second one was accepted and then dropped.** `[R10-2]` would have split
`plot_results` into a metrics rule and a figure rule, so that a figure-only
change became visible as one to Snakemake. Implementation found the seam was not
there: the metrics are one call *inside* the figure loop, downstream of a model
open, a gauge-name resolution, a merge, an alignment and the climate-parity
transform, so the "metrics half" is ~5 lines and the "figure half" is the
module. Dropped 2026-08-06 — the harm it fixed is a wasted re-run, not a wrong
number. **The verb `evaluate_` was withdrawn with it**, and 1.15 keeps the name
`plot_wflow_evaluation`, which was always the figure half's. What stays true is
recorded in `[R10-2]`: the DAG still cannot express the figure-vs-data
distinction the `AGENTS.md` validation ladder turns on. That is a known accepted
gap, not an open task.

**The rule set gained one member since the design was written**, from a
different item: `[R10-6]` §8 split the vector half out of `prepare_spatial_maps`
into `delineate_spatial_units`, declared in all three workflows from one shared
helper — 1.03 / 2.03 / 3.04 here. A structural change, but not one of this
milestone's; it is listed so the identifier count reconciles.

**Two merges were considered and rejected**, recorded in `[R10-3]` so they are not
re-raised: `write_outlet_index` into `declare_wflow_outputs` (would *add* a DAG
edge) and the paired `gather_*` rules (both delete the parts they consume, so
merging creates a partial-failure path that silently degrades the merged log).

**Three rules kept names the audit questioned.** `prepare_spatial_maps` (names one
of nine outputs — but the alternative read less clearly), `derive_change_factors`
(also renders a figure and writes a report) and `write_outlet_index`. Reasoning in
`rule-naming-design.md` amendment 2.

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

# WF1 — model creation (`Snakefile_model_creation`)

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
| 1.12 | `plot_basin_map` | Basin, rivers, gauges and DEM on one map. |
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

**Writes.** `config/runs/snake_config_model_creation.yml` ·
`config/runs/model_creation/<digest>/` (bundle dir).

**Writes (undeclared).** Copies into `config/templates/` (build + waterbodies),
`config/catalogs/` (data catalogs) and `config/observations/` (the two optional
observation inputs, which live outside the repo *and* outside `project_dir`).

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
`workflows.model_creation` section at all, so a payload drawn from one would
differ per invoking workflow.

**Writes.** `<spatial>/geoms/{basins,subbasins,catchments,rivers,locations}.geojson` ·
`<spatial>/location_registry.csv` · `<spatial>/hydrography.nc`.

`hydrography.nc` is the **seam intermediate** (§8a), not a product: the whole
hydrography grid stack used to cross the vector/raster boundary in memory, and
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

**Does.** Two steps that used to be two rules. First assembles the hydromt
recipe: a `steps:` YAML holding `setup_config` (`time.starttime`,
`time.endtime`, `time.timestepsecs`, `input.path_forcing`),
`setup_precip_forcing` and `setup_temp_pet_forcing`, with the PET method and
orography source branched off `clim_historical` and the chunksize sized by
opening the model's staticmaps. Then applies it via `hydromt update wflow_sbm`,
which builds the forcing for the model grid and — through the recipe's
`setup_config` step — writes the run window and forcing pointer into the model
TOML.

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

**Not merged into 1.09, and `[R10-3]` records why**: its inputs are
`outlets.geojson` and the registry, so it runs in parallel with the waterbody
and output-declaration rules. Merging would serialise a cheap pandas join behind
a hydromt `r+` mutation — it *adds* an edge.

#### 1.12 · `plot_basin_map`

**Does.** Plots basin, rivers, gauges and DEM on one map. Reads `staticmaps.nc`
straight off disk, so it is ordered behind 1.10's `.model_final` sentinel —
without that anchor a concurrent `-c 3` run aborts below Python on an unlocked
HDF5 read, with no traceback, because the pixi env sets
`HDF5_USE_FILE_LOCKING="FALSE"`.

**Writes.** `<model>/plots/basin_area.pdf` and `basin_area.png` — one render, two
formats: the PDF is the publication deliverable (vector, embedded fonts), the PNG
the preview every other consumer reads.

#### 1.13 · `plot_forcing`

**Does.** Draws the canonical climate figure set for the model's own forcing —
the same figures 1.05 draws for the source grid, so the two directories answer
"what did the downscaling change?" side by side.

**Writes.** `<model>/forcing/plots/` — the full variable × kind cross-product
from `climate_figures.figure_names("forcing")`, all declared.

#### 1.14 · `run_wflow`

**Does.** Runs Wflow.jl once on that historical forcing, driven by the model's
own TOML.

**Writes.** `<model>/run_default/output.csv`.

#### 1.15 · `plot_wflow_evaluation`

**Does.** Scores the Wflow run against observations where they exist and draws
the evaluation figures — hydrographs, climate comparison at model parity,
basin-average series, and signature plots. One rule, both products.

**Writes.** `<model>/evaluation/performance_metrics.csv` ·
`<model>/evaluation/plots/hydro_wflow_1.png` ·
`clim_wflow_1_{month,year}.png` · one `<var>_basavg.png` per basin-average
`wflow_outvars` entry.

**Writes (undeclared).** `hydro_<station>.png`,
`clim_<station>_{month,year}.png`, `signatures_<station>.png`. Their count is a
product of the model build (outlets and subcatchments), not of config, so they
cannot be enumerated at parse time; `signatures_*` also needs observations and a
run longer than a year.

**Why the metrics and the figures share a rule** — `[R10-2]` proposed splitting
them, since `performance_metrics.csv` is baseline-covered data while the figures
are excluded from the baseline, and the DAG cannot express that distinction. The
split was **dropped**: the metrics are one `compute_metrics` call *inside* the
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

**Writes.** `logs/wf1_model_creation.log`.

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

# WF2 — climate projections (`Snakefile_climate_projections`)

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
              │    plots/*_change_factor_cloud.png                   │
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

**Writes.** `config/runs/snake_config_climate_projections.yml` ·
`config/runs/climate_projections/<digest>/` (bundle dir).

**Writes (undeclared).** Catalog copies into `config/catalogs/`.

#### 2.02 · `delineate_region`

**Does.** As WF1 1.02 — the same one project region artifact, from the same
shared spec. Since ADR 0003 this is why a projections-only run no longer triggers
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
`<proj>/plots/<clim_project>_change_factor_cloud.png`.

#### 2.07 · `plot_gcm_timeseries`

**Does.** Plots the projected series — absolute levels and changes, annual and
monthly, for temperature and precipitation — from the per-member series of 2.05.
Its stage-B input is an **ordering edge only**; this rule never opens it.

**Writes.** Eight PNGs under `<proj>/plots/`, named
`<clim_project>_{precip,temp}_{annual,monthly}_{absolute,change}.png`. All eight
are declared; five used to be written but undeclared, and so were invisible to
Snakemake.

#### 2.08 · `gather_benchmarks`

**Does.** As WF1 1.16, for WF2.

**Writes.** `benchmarks/wf2_benchmarks.md`.

#### 2.09 · `gather_logs`

**Does.** As WF1 1.17, for WF2. Replaces two per-stage gathers that merged only
the fan-out rules, so following one run meant opening five files and knowing
their order.

**Writes.** `logs/wf2_climate_projections.log`.

---

# WF3 — climate experiment (`Snakefile_climate_experiment`)

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
              │  st_1..N.csv            │
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
| 3.09 | `prepare_stress_test_grid` | **Creates** the stress test: one CSV per grid point. |
| 3.10 | `prepare_weathergen_config` | The one weather-generator config. |
| 3.11 | `generate_weather_realizations` | All `RLZ_NUM` unperturbed realizations, in one call. |
| 3.12 | `perturb_climate_realization` | **Applies** one grid point to one realization. |
| 3.13 | `write_climate_data_catalog` | Catalogs every generated climate file. |
| 3.14 | `downscale_climate_realization` | One member onto the Wflow grid: forcing + TOML. |
| 3.15 | `run_wflow_batch_<b>` | Runs Wflow.jl, `B` members per Julia session. |
| 3.16 | `derive_wflow_indicators` | The two indicator tables. WF3's terminal product. |
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

**Writes.** `<exp>/config/snake_config_climate_experiment.yml` ·
`<exp>/config/runs/climate_experiment/<digest>/` (bundle dir).

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
one file per stress-test point: twelve monthly rows of temperature delta,
precipitation mean factor and precipitation variance factor. **This is what
creates the stress test.**

**Writes.** `<wg>/_work/st_1.csv` … `st_<ST_NUM>.csv` (zero-padded to a
width derived from `ST_NUM`, so `st_01 … st_12` on a twelve-point grid) ·
`<exp>/config/stress_test_design.csv` — one row per design point plus the
`st_0` baseline, written from the SAME loop, so the members and the table
describing them cannot disagree (C23/C26).

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

**Writes.** `<wg>/output/rlz_<n>_st_<m>.nc`, `temp()`.

#### 3.13 · `write_climate_data_catalog`

**Does.** Enumerates every generated climate file — perturbed and unperturbed —
into a hydromt data catalog the downscaling step reads, with the orography
sidecar path passed in explicitly rather than reconstructed by walking up from a
realization file.

**Writes.** `<exp>/config/catalogs/data_catalog_climate_experiment.yml`.

#### 3.14 · `downscale_climate_realization`

**Does.** Downscales one perturbed realization onto the Wflow grid via hydromt,
producing that member's forcing and its run TOML. The first rule to touch the
model, which is why 3.06's guard sentinel is a declared input here.

**Writes.** `<runs>/forcing/inmaps_rlz_<n>_st_<m>.nc` (`temp()`) ·
`<runs>/config/rlz_<n>_st_<m>.toml`.

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

**Does.** Reduces every member's run to the two indicator tables that form the
response surface. WF3's terminal product. Takes the stress-test grid as a
declared input — it was an undeclared runtime read until R07 B6, and so invisible
to `--dry-run`.

**Writes.** `<exp>/results/q_indicators.csv` ·
`<exp>/results/basin_indicators.csv`.

#### 3.17 · `gather_benchmarks`

**Does.** As WF1 1.16, for WF3.

**Writes.** `<exp>/benchmarks/wf3_benchmarks.md`.

#### 3.18 · `gather_logs`

**Does.** As WF1 1.17, for WF3 — where the merge earns most: 3.12 and 3.14 write
one part per (rlz, cst) and 3.15 one per batch, so the experiment's `logs/` held
hundreds of files across several subdirectories. A clean full run leaves one.

**Writes.** `<exp>/logs/wf3_climate_experiment.log`.

---

## Consolidations — all four decided

The name audit raised four structural candidates. All were ruled on 2026-08-06;
none is an R10 item. **One landed. The other three did not** — S1 was accepted
and then dropped the same day on implementation evidence.

| # | candidate | verdict |
|---|---|---|
| M1 | merge the forcing-recipe rule into `add_climate_forcing` | **accepted and landed** — `[R10-1]` |
| S1 | split `plot_results` into metrics + figures | **accepted, then DROPPED** — `[R10-2]`. The seam it assumed is not there: the metrics are one call inside the figure loop, downstream of the model open, the merge, the alignment and the parity transform. Splitting costs either a duplicated parity transform or a new declared artifact; the harm it fixes is a wasted re-run. `evaluate_` was withdrawn as a verb with it. See 1.15 |
| M2 | merge `write_outlet_index` into `declare_wflow_outputs` | **rejected.** Paired thematically, not structurally: `write_outlet_index` reads only `outlets.geojson` and `location_registry.csv`, so it runs in parallel with the waterbody and output-declaration rules. Merging would serialise a cheap pandas join behind a hydromt `r+` mutation — it *adds* an edge |
| M3 | merge `gather_benchmarks` + `gather_logs` per workflow | **rejected.** Both merge functions call `_remove_parts` — they delete the parts they consumed. In one rule, a failure in the second half strands the first half's already-deleted parts and the re-run degrades that artifact to "no part from this run". Today either succeeding independently means its output survives the other's failure |

**Also do not merge `write_model_reference` and `check_model_reference`** — the
`ancient()` / `temp()` asymmetry *is* the guard. See 3.06's section.

**Two general lessons**, both earned by designs that looked right on paper:

1. Two rules being small, adjacent and thematically similar is not an argument
   for merging them. Check what each actually **depends on**, and whether either
   **destroys its own inputs**. (M2 and M3.)
2. **A function boundary is not a data boundary**, and this is the lesson both
   splits taught — one by surviving it, one by not. `[R10-6]` §8 read as
   "vectors, then thematic rasters" until the whole hydrography grid turned out
   to cross the seam in memory; it landed anyway, by promoting that grid to a
   declared seam intermediate (`hydrography.nc`). `[R10-2]` read as "metrics,
   then figures" until the metrics proved to be one call inside the figure loop,
   downstream of a model open, a gauge-name resolution, a merge, an alignment
   and a parity transform; there was no equivalent intermediate worth its price,
   and it was dropped. Before splitting a rule, list what the second half would
   have to **reload or recompute** — not which functions it would call.

## Where the rules meet the artifacts

For what each rule reads and writes, rather than what it does:

- `dev/reference/workflows/model_creation.md`, `climate_experiment.md` — per-workflow detail.
- `dev/reference/contracts/weather-generator-seam.md`, `hydrological-model-seam.md` — the pinned interchange surfaces.
- `dev/milestones/r09/wf3-changes-proposal.md` appendix — the WF3 chain step by step, with the declared inputs of each stage.
- `dev/milestones/r10/rule-naming-design.md` — the verb vocabulary and the rename rationale.
