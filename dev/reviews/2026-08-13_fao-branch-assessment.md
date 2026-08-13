# Assessment — the upstream `fao` branch (Deltares/blueearth_cst)

*Written 2026-08-13. Read-only assessment; no code changed. Two owner rulings
taken the same day are recorded inline at §3 and §6.3.*

**Question asked:** the `fao` branch extends the workflow set and splits model
creation into historical climate + historical hydrology. That is the direction
we want. What is worth learning or reusing, especially from
`docs/notebooks/`?

**Short answer.** The *shape* is worth adopting and is cheaper than it looks —
main is already most of the way to the climate/hydrology split. The *code* is
largely not portable: `fao` is pinned to hydromt 0.9–0.10, hydromt_wflow 0.6 and
Wflow.jl v0, and main is on v1 of all three. The largest thing on `fao` that main
lacks entirely — the **delta-change future-hydrology workflow** — has been
**declined** (§3): it is top-down impact modelling in a repo whose method is
bottom-up. What is adopted is the workflow split, the forcing-selection layer
around it, and the notebook pattern; main's three notebooks are currently broken
and are the cheapest place to start.

**Rulings taken 2026-08-13** — §3 delta-change arm: *not adopted, stay strictly
bottom-up*. §6.3 notebook rot control: *commit outputs with a dated
"rendered against `<sha>`" banner*.

---

## 1. Provenance — what `fao` actually is

| | |
|---|---|
| Tip | `3d17df5` "update for dcrm and maintenance", 2026-04-17 |
| Merge-base with our `main` | `819e71d` "Merge pull request #28 from Deltares/add_docker", **2024-09-26** |
| Commits on `fao` since the base | 32 |
| Commits on our `main` since the same base | 1094 |

`fao` is a long-running Deltares development branch off the **pre-refactor**
tree. It is not behind main and it is not ahead of it — the two have been
developed independently for twenty months from the same ancestor, with different
goals. `fao` extended *scientific scope*; main rebuilt *engineering foundations*
(pixi lock, contracts, provenance, logging, baseline gate, spatial package,
~1900 tests). Neither tree can absorb the other by merge; the transfer is
selective and per item.

Layout also differs entirely: `fao` keeps `src/` + `snakemake/*.smk`, main keeps
`blueearth_cst/` + root `Snakefile_*`.

## 2. Workflow shape — five vs three

| `fao` | main | Relationship |
|---|---|---|
| `Snakefile_climate_historical` | *(part of WF1)* | **Split we want.** Climate-source extraction, station/subregion sampling, trends, comparison plots |
| `Snakefile_historical_hydrology` | `Snakefile_model_creation` (WF1) | Same job. `fao` runs the model once **per forcing dataset**; main runs one |
| `Snakefile_climate_projections` | `Snakefile_climate_projections` (WF2) | Same job; main is far more developed, but `fao` emits **gridded** monthly change factors, which main removed by ruling (§5.2) |
| `Snakefile_future_hydrology_delta_change` | *(absent)* | Top-down delta-change impact runs — **assessed and declined**, §3 |
| `Snakefile_climate_experiment` | `Snakefile_climate_experiment` (WF3) | Same job; main is far ahead (186 lines vs 1217) |

Sizes are worth stating plainly: every `fao` Snakefile is 167–208 lines; every
main Snakefile is 1139–1233. That gap is comment, contract checks, logging,
benchmarking and provenance, not rule count. Do not read `fao`'s brevity as
elegance to imitate — but do read its *rule decomposition* as a target shape.

### 2.1 The split is cheaper than a rewrite — main is already separable

This is the most useful finding in this document.

- WF1 rule **1.04 `extract_historical_climate` is already the shared store
  producer** — WF1 and WF3 both consume it.
- WF1 rule **1.05 `plot_climate_source`**'s subgraph builds with **neither**
  `models/hydrology/wflow/` **nor** `config/defaults/wflow_build_model.yml` on
  disk. Its docstring says so and `tests/test_plot_climate_source.py` pins it.

So the climate arm of WF1 is already model-independent by construction. Carving
`Snakefile_climate_historical` out of `Snakefile_model_creation` is a **Snakefile
partition plus a new observation-evaluation layer** — not a re-architecture.

`fao`'s structure is therefore a **target shape, not a migration path**. It also
matches the modularization direction recorded in `dev/roadmap.md` ("climate
analysis subworkflow" — run climate analysis model-independently from region +
catalog). The tension that item records against ADR 0002 is already closed: ADR
0002 was **superseded by 0006** on 2026-08-09, which retired the subcatchment
climate plots outright in favour of the canonical climate figure set. So the
`mod.forcing.data` coupling the roadmap item worried about is gone, and nothing
in that entry now argues against the split.

## 3. The method boundary — RULED: stay strictly bottom-up

> **Owner ruling, 2026-08-13 — the delta-change hydrology workflow is NOT
> adopted.** CST stays strictly bottom-up: WF2 remains a plausibility overlay,
> and no top-down GCM-scenario impact arm is added. `AGENTS.md` § Background
> stands unchanged and needs no amendment.
>
> Consequences: harvest items **9, 10 and 11 are closed as not-adopted**, the
> S8-08c gridded-branch removal (§5.2) **stands** and `save_grids: true` keeps
> raising, and `run_wflow_change_factors.jl` is not rewritten. §4.1 and §5.2 are
> kept as the record of what was assessed and why it was declined — not as
> pending work.

The reasoning behind the ruling, kept because a future reader will re-encounter
`fao` and ask the same question:

`AGENTS.md` § Background states that CST is bottom-up (decision-scaling / DMDU),
that stress-test scenarios come from the weather generator, and that CMIP6 output
is *"a plausibility overlay only … never drive a stress-test run."*

`Snakefile_future_hydrology_delta_change` is **top-down, scenario-driven impact
modelling**. It runs Wflow under GCM×SSP×horizon change factors and reports
projected change. That is structurally the thing the framing excludes.

The case *for* adopting it was that a delta-change arm is a standard CRIDA /
Decision Tree Framework companion and answers a question stakeholders always ask
("what do the models say for my basin?"). The case against, which won: it would
have to be walled off by an explicit non-coupling rule, and a wall is only as
good as the discipline maintaining it. The two arms drift into each other the
first time someone wants "the plausible corner of the response surface" — at
which point the repo's method section contradicts its own DAG. Declining is the
cheaper way to keep that from happening, and it costs no capability the
stress test itself needs.

## 4. Portability — the stack gap is real and verified

| | `fao` | main |
|---|---|---|
| Python | `>=3.9,<3.12` | `>=3.12,<3.13` |
| hydromt | `>=0.9.4,<=0.10` | `>=1.3,<2` |
| hydromt_wflow | `>=0.6.0` | `>=1.0,<2` |
| xarray | `<=2024.3.0` | unpinned (current) |
| Wflow.jl | v0.x | **1.0.2** |
| Julia | unpinned | `~1.11` |
| Lock file | none (`environment.yml` ranges) | `pixi.lock`, CI-enforced |

hydromt v1 is a rewrite, not a bump. Anything on `fao` that touches
`WflowModel`, `hydromt.flw`, catalogs or `setup_*` needs adaptation, not a copy.

### 4.1 The Julia delta-change driver is a rewrite, not a port

`src/wflow/run_wflow_change_factors.jl` is the clever part of `fao` — it applies
monthly gridded change factors **on the fly at each timestep**, so no perturbed
forcing files are ever written. Checked symbol by symbol against the Wflow.jl
1.0.2 source in the local depot:

| `fao` symbol | Status in Wflow.jl 1.0.2 |
|---|---|
| `Wflow.initialize_sbm_model(config)` | **gone** → `Wflow.Model(config)` |
| `model.vertical` | **gone** → `model.land` |
| `model.network` | **gone** → `model.domain` |
| `network.land.indices` | → `domain.land.network.indices` |
| `Wflow.load_fixed_forcing(model)` | → `load_fixed_forcing!(model)` |
| `Wflow.run_timestep(model)` | → `run_timestep!(model)` |
| `Wflow.param(model, "vertical.precipitation")` | → `get_param(model, "atmosphere_water__precipitation_volume_flux")` (CSDMS names) |

Every symbol it reaches for was renamed or removed. **The mechanism is sound and
the ~150 lines are a good specification; the file itself does not run here.**

Worth knowing before anyone proposes avoiding the custom driver: Wflow v1's
`scale`/`offset` config keys are applied **only to fixed (constant) forcing
values** (`io.jl:83`, inside `load_fixed_forcing!`). `update_forcing!` applies no
scaling to dynamic netCDF forcing. So v1 offers no native monthly-gridded
delta-change hook either — a custom driver remains necessary. Rewritten against
v1 it is arguably *cleaner*, since `get_param` + CSDMS names is a stabler seam
than reaching into `vertical.*`.

### 4.2 The design point worth keeping regardless of code

Two things in the delta-change workflow are methodologically right and should
survive any reimplementation:

1. **On-the-fly perturbation.** No materialized perturbed forcing. Note this is
   the *opposite* of what our WF3 does — WF3 writes per-realization perturbed
   netCDFs and wraps them in `temp()`. Both are defensible; the contrast is worth
   an explicit note when the delta-change arm is designed, because if the
   on-the-fly approach is adopted there, someone will reasonably ask why WF3 does
   not do the same.
2. **near → far state chaining.** The far-future run initialises from the
   near-future run's output states (`setup_toml_far` takes
   `outstates_{model}_{scenario}_near.nc` as an input). For glacier and deep
   groundwater storage this is the difference between a physically coherent
   trajectory and two disconnected snapshots. This is the single best idea in the
   whole `fao` DAG.

## 5. Harvest table

`ports` = adapt to our stack (pure pandas/xarray/matplotlib, no model API).
`idea only` = reimplement; the `fao` code does not run on v1.

| # | Item | On `fao` | Ports? | What main has | Recommend |
|---|---|---|---|---|---|
| 1 | Climate/hydrology workflow split | `Snakefile_climate_historical` + `Snakefile_historical_hydrology` | design | 1.04/1.05 already separable (§2.1) | **Yes** — highest value/cost ratio |
| 2 | Multi-forcing historical hydrology | `forcing_options:` map, one wflow run per source | design | one forcing per run | **Yes** — see §5.1 |
| 3 | Station-observation climate evaluation | `sample_climate_historical.py`, `plot_climate_location.py`, `plot_climate_basin.py` | ports (xarray/pandas) | absent — no `climate_locations` surface | **Yes** |
| 4 | SPI, dry-days, heat-days, frost-days | `plot_scalar_climate.py` (816 ln) | ports | **absent** (verified: no `spi` in `blueearth_cst/`) | Yes, selectively |
| 5 | Budyko screening | `plot_utils/plot_budyko.py` (239 ln) | idea only (uses `hydromt.flw`, v0 `WflowModel`) | **absent** (no `budyko`, no `aridity`) | **Yes** — cheap, high diagnostic value |
| 6 | MODIS snow-cover validation | `plot_results_grid.py`, `observations_snow:` | idea only | **absent** (`modis` in main is LAI only) | Defer — no ruling needed |
| 7 | Statistics heatmap tables | `plot_utils/plot_table_statistics.py` (293 ln) | **ports cleanly** — pure pandas/seaborn | absent as a heatmap; `shared/indicator_tables.py` is a different thing | **Yes** — cheapest useful win |
| 8 | Change-statistics engine | `compute_change_statistics.py` (662 ln) | ports partially (needs `wflow_utils`, xclim) | absent (WF3 has its own indicator path) | **No** — its consumer was #10 |
| 9 | Gridded monthly change factors | `save_grids: TRUE` branch in WF2 | n/a — see §5.2 | main **had** this and removed it by ruling; `save_grids: true` raises | **No** — ruled §3; S8-08c stands |
| 10 | Delta-change hydrology workflow | `Snakefile_future_hydrology_delta_change` + `run_wflow_change_factors.jl` | idea only (§4.1) | absent | **No** — ruled §3 |
| 11 | near→far state chaining | `setup_toml_far` | design | n/a | **No** — falls with #10, but see §4.2 |
| 12 | Notebook set | `docs/notebooks/` ×5 | design + prose | 3 broken notebooks (§6) | **Yes** — see §6 |
| 13 | Dry-month change-factor handling | `change_drymonth_threshold` / `_maxchange` (cap) | — | `projections/dry_month.py` (flag + NaN + status) | **No** — ours is better; see §5.2 |
| 14 | Return-period metrics | `metrics_definition.returninterval*` | — | `frequency_analysis` already in `export_wflow_results.py`; `metrics_definition.py` has the pulse/7-day family with a **water-year anchor** `fao` lacks | **No** |

### 5.1 Why multi-forcing historical hydrology matters here specifically

`AGENTS.md` frames CST as *rapid deployment, no local calibration*. That framing
makes forcing choice the dominant lever on the historical run, and `fao` handles
it directly: run the same uncalibrated model under N forcing datasets, then pick
using observed discharge, signatures and Budyko position. Their Piave example
carries the interpretation across two notebooks — ERA5 overestimates streamflow,
CHIRPS underestimates, ERA5 still wins on dynamics. That is exactly the decision
an uncalibrated rapid assessment needs support for, and main has no equivalent
step.

This is item 2 + 3 + 5 as one coherent capability, not three separate features.

### 5.2 The gridded change factors are a reversal, not a gap

Worth stating separately because it is the one place where "adopt from `fao`" is
the wrong verb.

Main's WF2 **had** a gridded branch and removed it — commit `fb0186c`
*"refactor(wf2)!: remove the gridded branch, fix the units attribute (S8-08)"*,
an owner ruling. `blueearth_cst/projections/gridded_outputs.py` now exists solely
to **reject** `save_grids: true` / `save_gridded: true` with a hard error (warning
on `false`).

The removal rationale is precise, and one clause of it is the whole point here:

> `raw/{series_key}.nc` already IS the basin slice on the source grid … the
> gridded series would have been a near-copy of a file every run already writes.
> **`grids/change/` was the only genuinely new artifact and no rule ever declared
> it to Snakemake.**

So the gridded *series* was correctly removed as a duplicate, and the gridded
*change factors* were removed as collateral — because at that moment nothing
consumed them. **A delta-change hydrology arm is exactly the consumer that did
not exist.** Three consequences:

1. Harvest #9 is not a port from `fao`. The implementation is in our own history
   (`fb0186c^`), written against our stack, and would be a **revert-and-adapt**.
2. It is a reversal of a recorded owner ruling, so it needs the ruling revisited
   with the new consumer as the evidence — not a quiet re-add.
3. It sequences *after* the §3 method ruling, not before: if the delta-change arm
   is not adopted, the S8-08c ruling stands unchanged and correct.

### 5.3 Where main is ahead — do not regress these

Recorded so a future adoption pass does not import a downgrade:

- **Dry months.** `fao` caps the change factor (`change_drymonth_maxchange: 50`).
  Main flags (`reference < min_reference` → `NaN` + `status`, keeping the
  absolute change). Capping silently fabricates a defensible-looking number;
  flagging says the ratio is undefined. Keep ours.
- **Water-year anchoring** in `metrics_definition.py` — `fao` hardcodes `YE`.
- **`interchange_contracts.py`**, provenance, `check_baseline.py`, log/benchmark
  reducers, `semantic_tree_diff.py` — `fao` has essentially none of this and
  negligible tests.
- **`shared/plot_evaluation.py`** — four evaluation sheets keyed by `wflow_id`,
  with three unit errors fixed relative to the `func_plot_signature.plot_hydro`
  ancestor that `fao` still ships (1196 lines).
- **Snakefile config parsing.** Main takes `workflow.configfiles[0]`; `fao` does
  `sys.argv[argv.index("--configfile") + 1]`, which breaks under a profile or
  `--configfiles`.

## 6. The notebooks

This is what was asked about specifically, so it gets its own treatment.

### 6.1 They are good, and the pattern is worth adopting wholesale

Five notebooks, one per workflow, indexed by `docs/notebooks/README.rst` (which
also carries a full dataset-provenance citation table — worth copying on its
own). The pattern each follows:

1. **Intro** naming the Snakefile and numbering what it does.
2. **`%%writefile ./config/my-project-settings.yml`** — the settings file *is*
   the tutorial. Every option is commented in place. This is the single best idea
   in the set: the config is not described, it is authored in front of the reader.
3. **Input-format cells** — `pd.read_csv(...)` on the station-locations and
   observation-timeseries files, so the required schema is shown rather than
   specified.
4. **DAG render before the run** (`--dag | dot -Tpng`), then a **rule-by-rule
   narrative where each rule names the config keys that tune it**. This is the
   part main's docs have no equivalent of anywhere.
5. `--unlock`, `--dryrun`, then the real run via a `runrealcmd` magic that
   streams output.
6. **DAG render after the run**, results-tree walk, then rendered figures **with
   interpretation** — not "here is the plot" but "ERA5 overestimates here, and
   this is what the Budyko position tells you about why".
7. **Forward link to the next notebook**, so the five read as one narrative:
   climate sources → hydrology → projections → impacts → stress test.

Point 6 is what makes them worth more than our docs: they teach the *reading* of
the output, which is the part a rapid-assessment tool most needs to transfer.

### 6.2 Main's three notebooks are currently broken

`docs/notebooks/` on main holds `Model building.ipynb`, `Climate
projections.ipynb`, `Climate Stress Test.ipynb` — ~10 KB each, outputs stripped,
last touched 2026-08-12 by a path-repointing commit. They are inherited skeletons
of the same pattern, without the results half. Verified defects:

- They `%%writefile ./config/my-project-settings.yml` and then run against it —
  but the repo's shipped seed configs live at `test_case/snake_config_*.yml`
  (`AGENTS.md` § Repo Map), so the notebook teaches a config location the repo
  does not use.
- DAG is written to `../../test_case/test_local/dag/dag_*.png` and then displayed
  from `./dag_*.png`. Mismatched paths, so the display fails. The directory does
  not exist, and the repo's actual DAG convention is
  `<project_dir>/logs/dag/...` via `scripts/plot_workflow_dag.py`.
- Result paths are pre-R9: `examples/myModel/hydrology_model/plots/basin_area.png`
  (now `data/spatial/plots/`) and `.../evaluation/plots/hydro_wflow_1.png` (now
  `models/hydrology/wflow/evaluation/plots/hydrograph_<wflow_id>.png`).

So this is not "adopt a new pattern" — it is "the pattern is already here,
half-built and rotted". That reframes the cost.

### 6.3 The cost to adopt — RULED: option C

> **Owner ruling, 2026-08-13 — rot control is option C below.** Notebooks commit
> their rendered outputs and carry a dated **"rendered against `<sha>`"** banner,
> with a periodic re-render as a board item. Staleness is made visible rather
> than prevented, which is the trade that fits a repo whose pipeline runs locally.

The options as assessed:

`fao`'s five notebooks total **15.6 MB** with outputs committed (largest: 6.4 MB).
They also hardcode `os.chdir(r'c:\repos\blueearth_cst')`, and nothing executes
them in CI — which is precisely how main's three got to their current state.

Committed outputs are what make them valuable (§6.1 point 6 needs rendered
figures) and are also what makes them rot invisibly. So adopting the pattern
means **choosing a rot-control mechanism**, and that is an owner decision, not a
default:

| Option | Cost | Rot behaviour |
|---|---|---|
| **A.** Execute in CI against the rapid config | Real CI minutes; needs data access | Cannot rot silently |
| **B.** Strip outputs, keep prose + a local run recipe | Free | Prose rots silently; loses the interpretation value |
| **C.** Commit outputs, add a dated "rendered against `<sha>`" banner + a periodic re-render task | Cheap; needs discipline | Rots visibly |
| **D.** Publish rendered notebooks as Artifacts, keep only sources in-repo | Moderate | Depends on re-render cadence |

Given main runs the pipeline locally rather than in CI (CI cannot reach
`test_case/test_local`), **C is the realistic default** and A is the honest ideal.

## 7. Anti-patterns not to import

Things on `fao` that would be regressions here, listed so an adoption pass does
not carry them across:

- **`ruleorder:` in `Snakefile_climate_projections`** (line 51). Main removed
  this deliberately and documents why in `AGENTS.md` § Conventions.
- **`julia --threads 4` hardcoded** in three rules. Main has
  `shared.julia_threads` ← `advanced_settings.yml`.
- **`julia ... "./src/wflow/run_wflow_change_factors.jl"`** — a relative path in a
  `shell:` body, correct only when cwd is the repo root.
- **`future_horizons` frozen to `near`/`far`.** The Snakefile carries duplicated
  `*_near` / `*_far` rule pairs (`downscale_..._near` and `downscale_..._far` are
  the same script), and the notebook says "assume period is near and far
  (fixed!!)". Three horizons means editing the Snakefile. A `{horizon}` wildcard
  with a chained-state dependency expressed as a function would generalise.
- **`.txt` sentinel outputs** for variable-cardinality figure rules
  (`basin_climate.txt`, `gridded_trends.txt`, `gridded_output.txt`,
  `plot_results.txt`). Snakemake then tracks a receipt rather than the artifacts;
  deleting a figure does not trigger a re-run.
- **No `temp()`** on delta-change intermediates — against `AGENTS.md` §
  Conventions.
- **`historical: 2000, 2010`** in the config — a bare string parsed downstream
  rather than a two-element list.
- **No lock file**, and `tabulate==0.8.10` pinned around a snakemake bug from
  2023.
- **`sample_climate_historical`** emits both `basin_*.nc` and `point_*.nc` from
  one rule, so a subregion-only change re-runs the station sampling.

## 8. Recommendation

With the §3 ruling taken, the expensive branch is closed and what remains is
three items, sequenced so each is independently useful. All three were admitted
to the board on 2026-08-13, **unranked** — their order relative to existing work
is the owner's call, not this document's:

| Item | Board note |
|---|---|
| 1 | `t2608131847` — repair and extend the workflow notebooks |
| 2 | `t2608131847a` — split the historical-climate workflow out of WF1 |
| 3 | `t2608131847b` — statistics heatmap tables |

1. **Fix and extend the notebooks** (`docs/**`, `lane/devmeta`). Repair the three
   broken ones against the current tree (§6.2), then adopt `fao`'s structure —
   config-first `%%writefile` pointed at a real seed config, input-schema cells,
   rule-by-rule narrative naming the config keys that tune each rule,
   results-with-interpretation, forward links — under §6.3's ruling: outputs
   committed, dated `rendered against <sha>` banner, re-render as a board item.
   Independent of everything else, immediately visible, and it forces a read of
   the current config surface that will inform item 2.
2. **Split `Snakefile_climate_historical` out of WF1** (`lane/pipeline`). Cheap
   per §2.1; this is the direction the assessment was asked about. Fold harvest
   items 2, 3 and 5 (multi-forcing runs, station-observation evaluation, Budyko)
   into the new workflow's evaluation layer — per §5.1 they are one capability,
   not three, and together they answer *which forcing dataset should this basin
   use?*, which an uncalibrated rapid assessment has no other support for.
3. **Port `plot_table_statistics.py`** (harvest #7, `lane/pipeline`). Pure
   pandas/seaborn, drops in, and the heatmap is useful to WF2 and WF3 whether or
   not item 2 has landed.

**Recommended starting point: 1** — this lane's territory, lowest cost, and its
output is effectively the specification for 2. Item 2 is the substantial one and
deserves a design pass before implementation.

Harvest #4 (SPI / dry-day / heat-day indices) and #6 (MODIS snow cover) are
genuine gaps that no ruling closes; they are cheap follow-ons to item 2 rather
than items in their own right.

## Refs

- `fao` tip `3d17df5`, merge-base `819e71d`; fetched as `upstream/fao`.
- Wflow.jl API checked against the local depot copy of **1.0.2**
  (`~/.julia/packages/Wflow/mJ7Ug`), specifically `src/Wflow.jl`, `src/io.jl`,
  `src/domain.jl`.
- Absence claims in §5 verified by grep over `blueearth_cst/`, `config/`,
  `tests/` — `budyko`, `aridity`, `snow_cover` return nothing; `spi` matches only
  `spines`; `modis` matches LAI sources only.
- Notebook staleness in §6.2 verified against `test_case/test_local` and
  `scripts/plot_workflow_dag.py`.
- §2.1's ADR status read from `dev/decisions/index.md` and ADR 0002's own
  `Superseded-by:` header, not inferred from `roadmap.md`.
- §5.2 quotes commit `fb0186c`'s message directly; the rejection lives in
  `blueearth_cst/projections/gridded_outputs.py`.
