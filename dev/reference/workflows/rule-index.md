# Rule index — every Snakemake rule, all three workflows

One page listing every rule in `Snakefile_model_creation`,
`Snakefile_climate_projections` and `Snakefile_climate_experiment`, what each
one does, what it writes, and how they connect.

> ## ⚠ This page describes the TARGET state, not what is on disk today
>
> Nothing below is implemented. It shows the rule set after the R10 rename sweep
> (`dev/milestones/r10/rule-naming-design.md`), after the two accepted structural
> changes (`dev/followups.md` `[R10-1]`, `[R10-2]`), and after the renumbering
> ruled on 2026-08-06 (`[R10-5]`).
>
> **Almost every rule answers to a different number today, twelve to a different
> name, one still exists and one does not exist yet.** Translate with
> [What changed](#what-changed) before typing a `snakemake <rule>` target or
> reading a `W.NN` reference in any older document. Delete this banner and that
> section once all of it lands.

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
gaps at 2.05/2.08/2.09/3.05, WF2 defined out of numeric order, and
`gather_benchmarks` sitting at 2.10 beside siblings at 1.14 and 3.12. See
[What changed](#what-changed) for what this costs.

**Going forward: do not renumber to insert a rule.** Use a letter suffix
(`1.09b`) until the next deliberate sweep. Renumbering is a migration, not an
edit — see `[R10-5]`.

## What changed

The only place this page names the old numbers and names. Everything after this
section is the target state.

### Renumbering (`[R10-5]`)

Read this table before interpreting any `W.NN` in a document written before
2026-08-06.

**WF1** — `Snakefile_model_creation`

| new | rule | was |
|---|---|---|
| 1.00 | `all` | 1.00 |
| 1.01 | `snapshot_config` | 1.01 |
| 1.02 | `delineate_region` | 1.01b |
| 1.03 | `extract_historical_climate` | 1.10 |
| 1.04 | `plot_climate_source` | 1.15 |
| 1.05 | `prepare_spatial_maps` | 1.02 |
| 1.06 | `build_wflow_model` | 1.03 |
| 1.07 | `write_outlet_index` | 1.06 |
| 1.08 | `add_reservoirs_lakes_glaciers` | 1.04 |
| 1.09 | `declare_wflow_outputs` | 1.05 |
| 1.10 | `plot_basin_map` | 1.12 |
| 1.11 | `add_climate_forcing` | 1.08 (+ 1.07, merged in) |
| 1.12 | `plot_forcing` | 1.13 |
| 1.13 | `run_wflow` | 1.09 |
| 1.14 | `evaluate_wflow_run` | 1.11 |
| 1.15 | `plot_wflow_evaluation` | 1.11b |
| 1.16 | `gather_benchmarks` | 1.14 |
| 1.17 | `gather_logs` | 1.16 |

**WF2** — `Snakefile_climate_projections`

| new | rule | was |
|---|---|---|
| 2.00 | `all` | 2.00 |
| 2.01 | `snapshot_config` | 2.03 |
| 2.02 | `delineate_region` | 2.03b |
| 2.03 | `fetch_gcm_slice` | 2.01 |
| 2.04 | `reduce_gcm_series` | 2.02 |
| 2.05 | `derive_change_factors` | 2.04 |
| 2.06 | `plot_gcm_timeseries` | 2.06 |
| 2.07 | `gather_logs` | 2.07 |
| 2.08 | `gather_benchmarks` | 2.10 |

**WF3** — `Snakefile_climate_experiment`

| new | rule | was |
|---|---|---|
| 3.00 | `all` | 3.00 |
| 3.01 | `check_project_consistency` | 3.00b |
| 3.02 | `snapshot_config` | 3.01 |
| 3.03 | `delineate_region` | 3.01b |
| 3.04 | `write_model_reference` | 3.01c |
| 3.05 | `check_model_reference` | 3.01d |
| 3.06 | `write_experiment_config` | 3.01e |
| 3.07 | `extract_historical_climate` | 3.02 |
| 3.08 | `prepare_stress_test_grid` | 3.03 |
| 3.09 | `prepare_weathergen_config` | 3.04 |
| 3.10 | `generate_weather_realizations` | 3.06 |
| 3.11 | `perturb_climate_realization` | 3.07 |
| 3.12 | `write_climate_data_catalog` | 3.08 |
| 3.13 | `downscale_climate_realization` | 3.09 |
| 3.14 | `run_wflow_batch_<b>` | 3.10 |
| 3.15 | `derive_wflow_indicators` | 3.11 |
| 3.16 | `gather_benchmarks` | 3.12 |
| 3.17 | `gather_logs` | 3.13 |

> **The cost, stated plainly: numbers are REUSED, so old references now resolve
> to the wrong rule.** New 1.07 is `write_outlet_index`; old 1.07 was
> `setup_runtime`. New 3.05 is `check_model_reference`; old 3.05 was the deleted
> `prepare_weagen_config_st`. Under the previous policy a retired number stayed a
> gap and a stale reference was merely dangling — obvious. Now it silently
> resolves to a different rule.
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
| `plot_wflow_evaluation` | `plot_results` — **and see the split below** |
| `plot_basin_map` | `plot_map` |
| `fetch_gcm_slice` | `fetch_gcm_raw` |
| `plot_gcm_timeseries` | `plot_climate_proj_timeseries` |
| `prepare_stress_test_grid` | `climate_stress_parameters` |
| `prepare_weathergen_config` | `prepare_weagen_config` |
| `generate_weather_realizations` | `generate_weather_realization` |
| `perturb_climate_realization` | `generate_climate_stress_test` |
| `write_climate_data_catalog` | `climate_data_catalog` |

### Two structural changes

| | change | why |
|---|---|---|
| `[R10-1]` | **`setup_runtime` merges into `add_climate_forcing`** (old 1.07 into old 1.08, now 1.11) | it wrote a hydromt forcing build recipe whose only consumer was the next rule. Two rules, one job — and a recipe that never leaves the pair needs no name of its own, so the naming problem disappears with the rule instead of being renamed around |
| `[R10-2]` | **`plot_results` splits** into 1.14 `evaluate_wflow_run` (metrics) and 1.15 `plot_wflow_evaluation` (figures) | `performance_metrics.csv` is baseline-covered data; the figures are excluded from the baseline. One rule producing both left the DAG unable to express the distinction the validation ladder turns on |

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
    1.03 extract_historical_climate    1.05 prepare_spatial_maps
      (SHARED store, = WF3 3.07)        (engine-neutral maps,
              │                          gauges, identities)
              ▼                               │
    1.04 plot_climate_source                  │
                                              │
STAGE 2 — MODEL BUILD                         │
──────────────────────────────────────────────────────────────────
                                              ▼
                                    1.06 build_wflow_model
                                              │
                              ┌───────────────┴──────────┐
                              ▼                          ▼
                 1.07 write_outlet_index    1.08 add_reservoirs_
                                                  _lakes_glaciers
                                                       │
                                                       ▼
                                          1.09 declare_wflow_outputs
                                                       │
                                       ┌───────────────┴──────────┐
                                       ▼                          ▼
                            1.10 plot_basin_map      1.11 add_climate_forcing
                                                                  │
                                                     ┌────────────┴───────────┐
                                                     ▼                        ▼
                                          1.12 plot_forcing            (to stage 3)

STAGE 3 — RUN + EVALUATE
──────────────────────────────────────────────────────────────────
                         1.13 run_wflow
                               │
                               ▼
                     1.14 evaluate_wflow_run
                               │
                               ▼
               1.15 plot_wflow_evaluation ◄── the store (1.03)

STAGE 4 — RUN RECORDS
──────────────────────────────────────────────────────────────────
      1.16 gather_benchmarks · 1.17 gather_logs   (last: every terminal)
```

**Stages are a reading aid, not a barrier.** Stage 1's climate branch (1.03,
1.04) runs concurrently with everything below it — a cold store extracts while
the model builds. Only the arrows constrain order.

**The five leaves.** 1.04, 1.07, 1.10, 1.12 and 1.15 have no downstream rule.
All are members of `WF1_TERMINALS`, so all are `rule all` targets and inputs of
the two gather rules — that is the edge the stage-4 line stands in for. Four are
figures, which are expected to terminate (no rule consumes a `.png`). **1.07 is
the one data leaf**, and its real consumer sits outside the workflow: see its
section below.

**What is NOT a dependency, despite reading like one.** 1.11 does not consume the
climate store: it reads source climate through the data catalog (`-d`), and its
only declared input is the forcing recipe it assembles itself. The store reaches
WF1's *figures* (1.04, 1.15), never its forcing.

| # | rule | in one line |
|---|---|---|
| 1.00 | `all` | Target aggregator. |
| 1.01 | `snapshot_config` | Snapshots the config and everything it references. |
| 1.02 | `delineate_region` | Delineates the one project extent. |
| 1.03 | `extract_historical_climate` | The shared historical-climate store (= WF3 3.07). |
| 1.04 | `plot_climate_source` | Climate figures on the source grid. |
| 1.05 | `prepare_spatial_maps` | The spatial foundation, and where gauges enter the workflow. |
| 1.06 | `build_wflow_model` | Parameterises Wflow-SBM, and where gauges enter the model. |
| 1.07 | `write_outlet_index` | Crosswalk from Wflow outlet IDs to named stations. |
| 1.08 | `add_reservoirs_lakes_glaciers` | Adds waterbodies. |
| 1.09 | `declare_wflow_outputs` | Declares the `[output.csv]` block: which timeseries Wflow emits. |
| 1.10 | `plot_basin_map` | Basin, rivers, gauges and DEM on one map. |
| 1.11 | `add_climate_forcing` | Assembles the hydromt recipe and applies it: builds the forcing. |
| 1.12 | `plot_forcing` | The same figures on the model's own forcing grid. |
| 1.13 | `run_wflow` | Runs Wflow.jl once. |
| 1.14 | `evaluate_wflow_run` | Scores the run against observations: the metrics table. |
| 1.15 | `plot_wflow_evaluation` | The evaluation figures. |
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

#### 1.03 · `extract_historical_climate`

**Does.** The **shared** historical-climate store producer — the same rule WF3
declares as 3.07, splatted from one `climate_store_spec` so the two declarations
cannot drift. Extracts the configured historical climate for the region and
window, on the source grid, model-free. Its declared inputs are the data catalog
— the store's freshness boundary — and the region polygon.

**Writes.** `<store>/extract_historical.nc`, plus `<store>/orography.nc` on the
chirps branches. The extraction records its own extent in netCDF attributes
(`region_geojson_sha256`, `region_bbox`, `region_source`) rather than in a
sidecar file.

#### 1.04 · `plot_climate_source`

**Does.** The canonical climate figure set on the **source** grid, straight from
the shared store, before any regridding to the model. Its whole subgraph is 1.02
+ 1.03 + this rule, so the figures build with no `<model>/` on disk at all.

**Writes.** `<store>/plots/` — the `figure_names("source")` set.

#### 1.05 · `prepare_spatial_maps`

**Does.** Builds the engine-neutral spatial foundation — the maps a model is
parameterised on, before any Wflow-specific step. Also where **gauge points
enter the workflow**: it snaps `shared.basin.gauge_points` to the river network
and partitions each parent basin into incremental subbasins — gauge-driven where
`control` points exist, automatic otherwise, chosen per basin — creating the
`basin_id` → `subbasin_id` → `wflow_id` identity hierarchy every later join uses.

**Writes.** `<spatial>/spatial_maps.nc` ·
`<spatial>/geoms/{basins,subbasins,catchments,rivers,locations}.geojson` ·
`<spatial>/location_registry.csv` · `<spatial>/spatial_catalog.yml` ·
`<spatial>/spatial_report.yml`.

#### 1.06 · `build_wflow_model`

**Does.** Parameterises Wflow-SBM on that spatial foundation via hydromt, then
reopens the written model and verifies its grid and IDs against the spatial
products. Also where **the gauges enter the model**: `setup_gauges` /
`setup_outlets` write the `gauges_locations` and `outlets` maps into staticmaps,
both with `toml_output=None` — maps only, no output declarations. No snapping and
no subcatchment derivation: 1.05 did both.

**Writes.** `<model>/staticmaps.nc` · `<model>/wflow_sbm.toml` ·
`<model>/staticgeoms/region.geojson` · `<model>/staticgeoms/outlets.geojson` ·
`<model>/.model_built` (sentinel).

`wflow_sbm.toml` is created here and then mutated in place by 1.08, 1.09 and
1.11, none of which declare it. That is what the `.model_built` sentinel exists
to handle.

#### 1.07 · `write_outlet_index`

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

#### 1.08 · `add_reservoirs_lakes_glaciers`

**Does.** Adds waterbodies to the built model (a hydromt update). A temporary
hydromt workaround; can fold back into 1.06 when upstream supports it.

**Writes.** `<model>/staticgeoms/reservoirs_lakes_glaciers.txt`.

**Writes (undeclared).** `<model>/staticmaps.nc` — it commits the waterbody
layers back into the model. That undeclared write is what forced 1.10's ordering
fix: Snakemake attributes `staticmaps.nc` to 1.06, so declaring it there orders
nothing after *this* rule.

#### 1.09 · `declare_wflow_outputs`

**Does.** Declares which timeseries Wflow emits — the `[output.csv]` block — for
`outlets` (Q), `gauges_locations` (Q, P) and basin means of any extra
`wflow_outvars`. It adds **no model data**: 1.06 created both gauge maps with
`toml_output=None`, deferring exactly this step. It also re-checks that the
model's gauge IDs still equal `location_registry.wflow_id`, and fails if either
map is absent.

`declare_` is the verb table's 18th entry, added for this rule: 1.08 and 1.11
add model *data* (waterbody layers, forcing grids), while this changes only what
the engine will emit.

**Writes.** `<model>/.outputs_configured` (sentinel).

**Writes (undeclared).** `<model>/wflow_sbm.toml` — the `[output.csv]` block
itself, via `mod.write()` — and `<model>/staticmaps.nc`, which `mod.close()` must
commit or hydromt leaves the new variables stranded in a `staticmaps_<hash>.nc`
temp file.

The gauge-ID re-check is not redundant with 1.06's identical comparison
(`build_wflow_model.py::_validate_written_model`): 1.08 mutates `staticmaps.nc`
in between, so this copy is what catches corruption from that step.

#### 1.10 · `plot_basin_map`

**Does.** Plots basin, rivers, gauges and DEM on one map. Reads `staticmaps.nc`
straight off disk, so it is ordered behind 1.09's sentinel — without that anchor
a concurrent `-c 3` run aborts below Python on an unlocked HDF5 read.

**Writes.** `<model>/plots/basin_area.pdf` and `basin_area.png` — one render, two
formats: the PDF is the publication deliverable (vector, embedded fonts), the PNG
the preview every other consumer reads.

#### 1.11 · `add_climate_forcing`

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
the model it built).

**Writes (undeclared).** `<model>/wflow_sbm.toml` — `time.*` and
`input.path_forcing`.

#### 1.12 · `plot_forcing`

**Does.** Draws the canonical climate figure set for the model's own forcing —
the same figures 1.04 draws for the source grid, so the two directories answer
"what did the downscaling change?" side by side.

**Writes.** `<model>/forcing/plots/` — the full variable × kind cross-product
from `climate_figures.figure_names("forcing")`, all declared.

#### 1.13 · `run_wflow`

**Does.** Runs Wflow.jl once on that historical forcing, driven by the model's
own TOML.

**Writes.** `<model>/run_default/output.csv`.

#### 1.14 · `evaluate_wflow_run`

**Does.** Scores the Wflow run against observations where they exist, producing
the evaluation metrics. Split from the figures (`[R10-2]`) because the metrics
table is **baseline-covered data** while the figures are explicitly excluded from
the baseline — one rule producing both left the DAG unable to express the
distinction the validation ladder turns on.

**Writes.** `<model>/evaluation/performance_metrics.csv`.

#### 1.15 · `plot_wflow_evaluation`

**Does.** Draws the evaluation figures — hydrographs, climate comparison at model
parity, basin-average series, and signature plots where observations exist.
Terminal: no rule consumes a figure, so a change here cannot propagate into a
number.

**Writes.** `<model>/evaluation/plots/hydro_wflow_1.png` ·
`clim_wflow_1_{month,year}.png` · one `<var>_basavg.png` per basin-average
`wflow_outvars` entry.

**Writes (undeclared).** `hydro_<station>.png`,
`clim_<station>_{month,year}.png`, `signatures_<station>.png`. Their count is a
product of the model build (outlets and subcatchments), not of config, so they
cannot be enumerated at parse time; `signatures_*` also needs observations and a
run longer than a year.

**Open at implementation:** how the two halves share loaded data — a re-read of
`output.csv` in this rule, or a declared intermediate from 1.14. And whether
`evaluate_` earns a 19th verb; see `[R10-2]`.

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
the project extent. CST's `subbasins.geojson` (rule 1.05) is a different thing:
the incremental partition *within* that extent. A project can be
`{'basin': ...}` at 1.02 and still have twelve subbasins at 1.05.

## Where a gauge point lives, rule by rule

`shared.basin.gauge_points` (`station_name, x, y, location_role[, wflow_id]`) is
consumed once, by 1.05, and everything after that reads its derived identities:

| stage | rule | what happens to the point |
|---|---|---|
| enters | 1.05 `prepare_spatial_maps` | snapped to a river cell, given `location_id`/`wflow_id`; a `control` point also becomes a subbasin outlet |
| enters the model | 1.06 `build_wflow_model` | written into `staticmaps.nc` as `gauges_locations`, no TOML output |
| becomes an output | 1.09 `declare_wflow_outputs` | named in `[output.csv]`, so Wflow emits its timeseries |
| becomes joinable | 1.07 `write_outlet_index` | `outlet_index.csv` maps Wflow's subcatchment IDs back to the named station |

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
              ┌─────────────────┴─────────────────┐
              │                                   │
  CMIP6 store │ (gs://cmip6)                      │
              ▼                                   │
    2.03 fetch_gcm_slice                          │
    (one raw slice per member;                    │
     the ONLY remote read)                        │
              │                                   │
              ▼                                   │
    2.04 reduce_gcm_series ◄──────────────────────┤
    (one job per series key, full fan-out)        │
              │                                   │
STAGE 2 — PRODUCT                                 │
──────────────────────────────────────────────────────────────────
              ▼                                   │
    2.05 derive_change_factors ◄──────────────────┘
    (ONE job — the workflow's answer)
              │
              ├──► summary/*_change_factors_{annual,monthly}.csv
              │    composition.csv · provenance.json · report.md
              │    plots/*_change_factor_cloud.png
              ▼
STAGE 3 — FIGURES + RECORDS
──────────────────────────────────────────────────────────────────
    2.06 plot_gcm_timeseries   (reads 2.04's series, not 2.05)
              │
              ▼
    2.07 gather_logs · 2.08 gather_benchmarks
```

The region polygon feeds **three** rules — 2.03, 2.04 and 2.05 all declare it —
because stage B recomputes every expected digest, including the polygon
fingerprint. 2.06's edge from 2.05 is an **ordering edge only**; it plots the
per-member series from 2.04 and never opens the change-factor table.

| # | rule | in one line |
|---|---|---|
| 2.00 | `all` | Target aggregator. |
| 2.01 | `snapshot_config` | As WF1 1.01. |
| 2.02 | `delineate_region` | As WF1 1.02 — the same artifact. |
| 2.03 | `fetch_gcm_slice` | Acquires one raw CMIP6 slice. The only remote read. |
| 2.04 | `reduce_gcm_series` | Stage A: one local slice → one monthly series. |
| 2.05 | `derive_change_factors` | Stage B, one job. WF2's terminal product. |
| 2.06 | `plot_gcm_timeseries` | The eight projection figures. |
| 2.07 | `gather_logs` | Merges the log parts. |
| 2.08 | `gather_benchmarks` | Merges the timing parts. |

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

#### 2.03 · `fetch_gcm_slice`

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

#### 2.04 · `reduce_gcm_series`

**Does.** Stage A. Reduces one **local** raw slice to a monthly series over the
region polygon, for its (model, scenario, member) key. One job per key, no edges
between series, no network call.

**Writes.** `<proj>/scalar/<series_key>.nc` — persistent + `update()`, same
reason as 2.03.

#### 2.05 · `derive_change_factors`

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

#### 2.06 · `plot_gcm_timeseries`

**Does.** Plots the projected series — absolute levels and changes, annual and
monthly, for temperature and precipitation — from the per-member series of 2.04.
Its stage-B input is an **ordering edge only**; this rule never opens it.

**Writes.** Eight PNGs under `<proj>/plots/`, named
`<clim_project>_{precip,temp}_{annual,monthly}_{absolute,change}.png`. All eight
are declared; five used to be written but undeclared, and so were invisible to
Snakemake.

#### 2.07 · `gather_logs`

**Does.** As WF1 1.17, for WF2. Replaces two per-stage gathers that merged only
the fan-out rules, so following one run meant opening five files and knowing
their order.

**Writes.** `logs/wf2_climate_projections.log`.

#### 2.08 · `gather_benchmarks`

**Does.** As WF1 1.16, for WF2.

**Writes.** `benchmarks/wf2_benchmarks.md`.

---

# WF3 — climate experiment (`Snakefile_climate_experiment`)

The stress test itself. Generates stochastic weather realizations, perturbs each
across a temperature × precipitation grid, runs every member through Wflow, and
reduces the runs to the indicator tables that form the response surface.

Every climate artifact is generated **before** the model is used: 3.13 is the
first rule to put the model to work, and the whole stress-test ensemble already
exists by then.

```
STAGE 1 — GUARD + PROVENANCE   (config and hashes only)
──────────────────────────────────────────────────────────────────
   config ──► 3.01 check_project_consistency   (drift guard, fails loud)
                          │
        ┌─────────────────┼──────────────────┬──────────────────┐
        ▼                 ▼                  ▼                  ▼
  3.02 snapshot     3.03 delineate     3.04 write_model    3.06 write_
     _config           _region           _reference        experiment_config
        │                 │                    │
        │                 │                    ▼
        │                 │           3.05 check_model_reference
        │                 │            (verdict consumed by 3.13)
        │                 │
STAGE 2 — CLIMATE DATA   (the model is fingerprinted, never used)
──────────────────────────────────────────────────────────────────
        │                 ▼
        │      3.07 extract_historical_climate   (SHARED with WF1 1.03)
        │                 │  extract_historical.nc
        ▼                 │
  3.08 prepare_stress     │      3.09 prepare_weathergen_config
      _test_grid          │              │  weathergen_config.yml
        │  cst_1..N.csv   └──────────────┤
        │                                ▼
        │                 3.10 generate_weather_realizations
        │                                │  rlz_1..R_cst_0.nc  (unperturbed)
        └────────────────┐               │
                         ▼               ▼
                  3.11 perturb_climate_realization
                         │  rlz_<n>_cst_<m>.nc   (perturbed)
                         ▼
                  3.12 write_climate_data_catalog
                         │
STAGE 3 — MODEL RUN   (first use of the built model)
──────────────────────────────────────────────────────────────────
                         ▼
       3.13 downscale_climate_realization ◄── model + 3.05's verdict
                         │  inmaps + per-member TOML
                         ▼
                  3.14 run_wflow_batch_<b>   (B members per Julia session)
                         │  per-member run CSVs
                         ▼
STAGE 4 — PRODUCT + RECORDS
──────────────────────────────────────────────────────────────────
                  3.15 derive_wflow_indicators
                         │  q_indicators.csv · basin_indicators.csv
                         ▼
                  3.16 gather_benchmarks · 3.17 gather_logs
```

**The store feeds 3.10, not 3.08.** 3.08 enumerates the stress-test grid from
the config alone — it needs no climate data at all, and runs concurrently with
the extraction. The historical climate is what the *generator* resamples.

| # | rule | in one line |
|---|---|---|
| 3.00 | `all` | Target aggregator. |
| 3.01 | `check_project_consistency` | Startup drift guard against the wf1/wf2 snapshots. |
| 3.02 | `snapshot_config` | As WF1 1.01, kept inside the experiment. |
| 3.03 | `delineate_region` | As WF1 1.02 — the same artifact. |
| 3.04 | `write_model_reference` | Records which model state this experiment used. |
| 3.05 | `check_model_reference` | Refuses to simulate if that model has changed. |
| 3.06 | `write_experiment_config` | Records the experiment's own parameters. |
| 3.07 | `extract_historical_climate` | The shared climate store (= WF1 1.03). |
| 3.08 | `prepare_stress_test_grid` | **Creates** the stress test: one CSV per grid point. |
| 3.09 | `prepare_weathergen_config` | The one weather-generator config. |
| 3.10 | `generate_weather_realizations` | All `RLZ_NUM` unperturbed realizations, in one call. |
| 3.11 | `perturb_climate_realization` | **Applies** one grid point to one realization. |
| 3.12 | `write_climate_data_catalog` | Catalogs every generated climate file. |
| 3.13 | `downscale_climate_realization` | One member onto the Wflow grid: forcing + TOML. |
| 3.14 | `run_wflow_batch_<b>` | Runs Wflow.jl, `B` members per Julia session. |
| 3.15 | `derive_wflow_indicators` | The two indicator tables. WF3's terminal product. |
| 3.16 | `gather_benchmarks` | Merges the timing parts. |
| 3.17 | `gather_logs` | Merges the log parts. |

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

#### 3.04 · `write_model_reference`

**Does.** Records **which model state** this experiment used: the model's
relative path, a pointer-derived digest, and the per-input hashes behind it. Not
a copy — a hash answers the question a duplicated staticmaps would, and the
per-input hashes are kept so a later mismatch can *name* what changed. Its model
inputs are `ancient()` on purpose: if the reference were rewritten whenever the
model changed it would always match, and 3.05's comparison would be decorative.

**Writes.** `<exp>/config/model_reference.yml`.

#### 3.05 · `check_model_reference`

**Does.** The other half: recomputes the fingerprint and refuses to simulate if
the live model has changed since the experiment was recorded. Its sentinel is a
declared input of 3.13 — the first rule to touch the model — because a check
after the work is a post-mortem, not a guard.

**Writes.** `<exp>/.model_reference_ok` — `temp()`, and that is the trigger, not
an optimisation. A persisted sentinel would satisfy 3.13's edge with a **stale
verdict**: the check passed once, the file remains, and 3.13 is free to
re-simulate against a model that changed afterwards. Deleting it on consumption
forces the next invocation to re-evaluate. A guard evaluates; it does not cache
an answer.

**Do not merge 3.04 and 3.05.** They read as an obvious pair and merging them
destroys the guard — the `ancient()` / `temp()` asymmetry above *is* the
mechanism, not an accident.

#### 3.06 · `write_experiment_config`

**Does.** Records the experiment's own parameters, separately from the project
ones. Generated, never authored — a hand-written file here would be a second
source of truth competing with the `--configfile`. Immutable from the first
*successful* run, keyed off the merged workflow log's existence, since editing an
experiment's parameters before it has produced anything is ordinary work and
afterwards would silently redefine what the existing results mean.

**Writes.** `<exp>/config/experiment.yml`.

#### 3.07 · `extract_historical_climate`

**Does.** The shared historical-climate store producer — the same rule as WF1
1.03, byte-identical but for `message`/`log`/`benchmark`, with
`tests/test_climate_store_contract.py` failing on any other difference. Usually
already current when run in pipeline order.

**Writes.** `<store>/extract_historical.nc`, plus `<store>/orography.nc` on the
chirps branches.

#### 3.08 · `prepare_stress_test_grid`

**Does.** Enumerates the configured temperature × precipitation grid and writes
one file per stress-test point: twelve monthly rows of temperature delta,
precipitation mean factor and precipitation variance factor. **This is what
creates the stress test.**

**Writes.** `<wg>/_work/cst_1.csv` … `cst_<ST_NUM>.csv`.

#### 3.09 · `prepare_weathergen_config`

**Does.** Assembles the one weather-generator config from the shipped template
plus the project settings — the year arithmetic (middle year, simulation length)
and the two transient-change flags. The template is a **declared input**: until
2026-08-05 it was a params-only read, so editing it changed nothing until
something else forced a rerun, and 3.10 kept generating from superseded settings.

**Writes.** `<wg>/config/weathergen_config.yml`.

#### 3.10 · `generate_weather_realizations`

**Does.** Runs weathergenr **once** to produce all `RLZ_NUM` stochastic
realizations of the historical climate — the unperturbed `cst_0` baselines. The
plural is load-bearing: number carries meaning here, with 3.10 plural (all in one
job) against 3.13 singular (wildcarded, one job per member).

**Writes.** `<wg>/output/rlz_1_cst_0.nc` … `rlz_<RLZ_NUM>_cst_0.nc`, all
`temp()`.

**Writes (undeclared).** Four generator diagnostic figures moved into
`<wg>/plots/` (`obs_power_spectra.png`, `warm_annual_precip.png`,
`warm_annual_stats.png`, `warm_annual_wavelet.png`) and weathergenr's date CSVs
left in `<wg>/output/`.

#### 3.11 · `perturb_climate_realization`

**Does.** Takes one unperturbed realization and one stress-test point and
applies that perturbation — precipitation mean and variance factors, temperature
delta, transient flags, PET recompute. **It applies the stress test; 3.08 creates
it.** Its `st_num` wildcard is constrained to ≥ 1 so it can never become a second
producer of the reserved `cst_0` baseline, which would surface as a cyclic-graph
error.

**Writes.** `<wg>/output/rlz_<n>_cst_<m>.nc`, `temp()`.

#### 3.12 · `write_climate_data_catalog`

**Does.** Enumerates every generated climate file — perturbed and unperturbed —
into a hydromt data catalog the downscaling step reads, with the orography
sidecar path passed in explicitly rather than reconstructed by walking up from a
realization file.

**Writes.** `<exp>/config/catalogs/data_catalog_climate_experiment.yml`.

#### 3.13 · `downscale_climate_realization`

**Does.** Downscales one perturbed realization onto the Wflow grid via hydromt,
producing that member's forcing and its run TOML. The first rule to touch the
model, which is why 3.05's guard sentinel is a declared input here.

**Writes.** `<runs>/forcing/inmaps_rlz_<n>_cst_<m>.nc` (`temp()`) ·
`<runs>/config/rlz_<n>_cst_<m>.toml`.

#### 3.14 · `run_wflow_batch_<b>`

**Does.** Runs Wflow.jl for every member, `B` per Julia session to amortise
startup, through a parse-time loop of one anonymous rule per batch with static
per-member input/output lists. `B` defaults from `-c N` and is clamped by
`batch_size_max`; `batch_size: 1` restores one job per member. Rule identifiers
are per batch while the log label stays the singular `3.14_run_wflow` —
deliberately, so **this rule is exempt from the rename call-site rule**.

**Writes.** `<runs>/output/rlz_<n>_cst_<m>.csv` per member ·
`<runs>/output/outstates_rlz_<n>_cst_<m>.nc` per member (`temp()`).

#### 3.15 · `derive_wflow_indicators`

**Does.** Reduces every member's run to the two indicator tables that form the
response surface. WF3's terminal product. Takes the stress-test grid as a
declared input — it was an undeclared runtime read until R07 B6, and so invisible
to `--dry-run`.

**Writes.** `<exp>/results/q_indicators.csv` ·
`<exp>/results/basin_indicators.csv`.

#### 3.16 · `gather_benchmarks`

**Does.** As WF1 1.16, for WF3.

**Writes.** `<exp>/benchmarks/wf3_benchmarks.md`.

#### 3.17 · `gather_logs`

**Does.** As WF1 1.17, for WF3 — where the merge earns most: 3.11 and 3.13 write
one part per (rlz, cst) and 3.14 one per batch, so the experiment's `logs/` held
hundreds of files across several subdirectories. A clean full run leaves one.

**Writes.** `<exp>/logs/wf3_climate_experiment.log`.

---

## Consolidations — all four decided

The name audit raised four structural candidates. All were ruled on 2026-08-06;
none is an R10 item. The two accepted ones are already reflected above.

| # | candidate | verdict |
|---|---|---|
| M1 | merge the forcing-recipe rule into `add_climate_forcing` | **accepted** — `[R10-1]` |
| S1 | split `plot_results` into metrics + figures | **accepted** — `[R10-2]` |
| M2 | merge `write_outlet_index` into `declare_wflow_outputs` | **rejected.** Paired thematically, not structurally: `write_outlet_index` reads only `outlets.geojson` and `location_registry.csv`, so it runs in parallel with the waterbody and output-declaration rules. Merging would serialise a cheap pandas join behind a hydromt `r+` mutation — it *adds* an edge |
| M3 | merge `gather_benchmarks` + `gather_logs` per workflow | **rejected.** Both merge functions call `_remove_parts` — they delete the parts they consumed. In one rule, a failure in the second half strands the first half's already-deleted parts and the re-run degrades that artifact to "no part from this run". Today either succeeding independently means its output survives the other's failure |

**Also do not merge `write_model_reference` and `check_model_reference`** — the
`ancient()` / `temp()` asymmetry *is* the guard. See 3.05's section.

**The general lesson**, worth carrying into any future sweep: two rules being
small, adjacent and thematically similar is not an argument for merging them.
Check what each actually depends on, and whether either destroys its own inputs.

## Drift found in the Snakefile comments (not yet fixed)

Tracked as `[R10-4]`. Reported rather than patched, because these live in the
Snakefiles: two stale references to the rule C29 deleted in
`Snakefile_climate_experiment`, and all three `gather_benchmarks` comments
describing their output as `.tsv` when it is `.md`.

## Where the rules meet the artifacts

For what each rule reads and writes, rather than what it does:

- `dev/reference/workflows/model_creation.md`, `climate_experiment.md` — per-workflow detail.
- `dev/reference/contracts/weather-generator-seam.md`, `hydrological-model-seam.md` — the pinned interchange surfaces.
- `dev/milestones/r09/wf3-changes-proposal.md` appendix — the WF3 chain step by step, with the declared inputs of each stage.
- `dev/milestones/r10/rule-naming-design.md` — the verb vocabulary and the rename rationale.
