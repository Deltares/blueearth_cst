# Rule index — every Snakemake rule, all three workflows

One page listing every rule in `Snakefile_model_creation`,
`Snakefile_climate_projections` and `Snakefile_climate_experiment`, what each
one does, and how they connect.

**Two name columns.** R10 renames twelve rules
(`dev/milestones/r10/rule-naming-design.md`, accepted and amended, **not yet
implemented**). Until that sweep lands, "current" is what you type on the command
line and "after R10" is the target. Once it lands, drop the first column.

**On the numbers.** `W.NN` is a **stable identifier assigned when a rule is
created**, not a position — it disambiguates log parts across workflows and gives
comments a short handle. It is not contiguous and not in definition order. Real
gaps: **2.05, 2.08, 2.09** (merged or removed rules) and **3.05** (deleted by C29,
2026-08-05). WF2 is also defined out of numeric order, and its
`gather_benchmarks` sits at 2.10 where its siblings are at 1.14 and 3.12.

---

# WF1 — model creation (`Snakefile_model_creation`)

Builds a distributed Wflow-SBM model from global datasets via hydromt and runs it
once on historical forcing. No calibration — rapid deployment.

```
                        config + data catalogs
                                 │
      1.01 snapshot_config ──────┤
                                 ▼
                      1.01b delineate_region  ──► spatial/geoms/region.geojson
                                 │                          │
                                 ▼                          │
                      1.02 prepare_spatial_maps             │
                                 │                          │
                                 ▼                          ▼
                      1.03 build_wflow_model      1.10 extract_climate_grid
                                 │                  (SHARED store, = WF3 3.02)
                                 ▼                          │
                 1.04 add_reservoirs_lakes_glaciers         │
                                 ▼                          │
                    1.05 add_gauges_and_outputs             │
                                 ▼                          │
                     1.06 write_outlet_index                │
                                 ▼                          │
                       1.07 setup_runtime                   │
                                 ▼                          │
                       1.08 add_forcing  ◄──────────────────┤
                                 │        inmaps_historical.nc
                                 ▼                          │
                         1.09 run_wflow                     │
                                 │                          │
             ┌───────────────────┼───────────────┐          │
             ▼                   ▼               ▼          ▼
     1.11 plot_results    1.12 plot_map   1.13 plot_forcing │
                                                  1.15 plot_climate_source
                                 │
                                 ▼
                1.14 gather_benchmarks · 1.16 gather_logs
```

| # | current | after R10 | what it does |
|---|---|---|---|
| 1.00 | `all` | — | Target aggregator: the full historical build plus the performance plots. |
| 1.01 | `snapshot_config` | — | Copies the config and every file it references into the project, plus an immutable content-addressed bundle of the effective settings. |
| 1.01b | `delineate_region` | — | Delineates the one project region polygon from `shared.basin` and the data catalog (ADR 0003). Every downstream extent comes from this artifact, not from a built model. |
| 1.02 | `prepare_spatial_maps` | — | Builds the engine-neutral spatial foundation — the maps a model is parameterised on, before any Wflow-specific step. |
| 1.03 | `build_wflow_model` | — | Parameterises Wflow-SBM on that spatial foundation via hydromt. |
| 1.04 | `add_reservoirs_lakes_glaciers` | — | Adds waterbodies to the built model (a hydromt update). |
| 1.05 | `add_gauges_and_outputs` | — | Adds gauge locations and the requested output variables, writing the `[output.csv]` block that decides which timeseries Wflow emits. |
| 1.06 | `write_outlet_index` | — | Writes the outlet-position → subcatchment-ID crosswalk that later joins model outputs to named stations. |
| 1.07 | `setup_runtime` | **`prepare_runtime_window`** | Computes the forcing time horizon and run window, and writes them into the model's TOML. |
| 1.08 | `add_forcing` | **`add_climate_forcing`** | Adds the historical climate forcing to the model (`inmaps_historical.nc`, a hydromt update). |
| 1.09 | `run_wflow` | — | Runs Wflow.jl once on that historical forcing. |
| 1.10 | `extract_climate_grid` | **`extract_historical_climate`** | The **shared** historical-climate store producer — the same rule WF3 declares as 3.02. Extracts climate data for the region and period into `data/climate/historical/<key>/`. |
| 1.11 | `plot_results` | **`plot_wflow_evaluation`** | Analyses and plots the Wflow run: hydrographs, and signature plots where observations exist. |
| 1.12 | `plot_map` | **`plot_basin_map`** | Plots basin, rivers, gauges and DEM on one map. |
| 1.13 | `plot_forcing` | — | The canonical climate figure set for the model's own forcing. |
| 1.14 | `gather_benchmarks` | — | Merges the per-rule timing parts into one benchmark table. |
| 1.15 | `plot_climate_source` | — | Climate figures on the **source** grid, from the shared store, before any regridding to the model. |
| 1.16 | `gather_logs` | — | Merges every WF1 log part into one workflow log. |

---

# WF2 — climate projections (`Snakefile_climate_projections`)

A plausibility overlay, not a driver. Computes monthly CMIP6 change factors that
situate the stress-test grid in projection space. **Nothing here feeds a
stress-test run.**

```
      CMIP6 store (gs://cmip6)          config + catalogs
              │                                 │
              ▼                     2.03 snapshot_config
    2.01 fetch_gcm_raw               2.03b delineate_region
    (one raw slice per member)                  │
              │                                 │
              ▼                                 │
    2.02 reduce_gcm_series  ◄───────────────────┘
    (stage A: one job per series key)
              │
              ▼
    2.04 derive_change_factors
    (stage B: ONE job — the workflow's answer)
              │
              ├──► summary tables (change factors, annual + monthly)
              │
              ▼
    2.06 plot_climate_proj_timeseries
              │
              ▼
    2.07 gather_logs · 2.10 gather_benchmarks
```

| # | current | after R10 | what it does |
|---|---|---|---|
| 2.00 | `all` | — | Target aggregator: the change-factor summaries plus the projection plots. |
| 2.01 | `fetch_gcm_raw` | **`fetch_gcm_slice`** | Acquires one raw CMIP6 slice. The only rule that reads the remote store. |
| 2.02 | `reduce_gcm_series` | — | Stage A: reduces one fetched slice to a monthly series for its (model, scenario, member) key. One job per key. |
| 2.03 | `snapshot_config` | — | As WF1 1.01. |
| 2.03b | `delineate_region` | — | As WF1 1.01b — the same one project region artifact. |
| 2.04 | `derive_change_factors` | — | Stage B, a single job: turns every reduced series into the monthly change factors per model, scenario and horizon. This is WF2's terminal product. |
| 2.06 | `plot_climate_proj_timeseries` | **`plot_gcm_timeseries`** | Plots the projected series — absolute levels and changes, annual and monthly, for temperature and precipitation. Consumes the per-member series from 2.02. |
| 2.07 | `gather_logs` | — | Merges every WF2 log part into one workflow log. |
| 2.10 | `gather_benchmarks` | — | Merges the per-rule timing parts into one benchmark table. |

---

# WF3 — climate experiment (`Snakefile_climate_experiment`)

The stress test itself. Generates stochastic weather realizations, perturbs each
across a temperature × precipitation grid, runs every member through Wflow, and
reduces the runs to the indicator tables that form the response surface.

```
   config ──► 3.00b check_project_consistency   (drift guard, fails loud)
                          │
        ┌─────────────────┼──────────────────┬──────────────────┐
        ▼                 ▼                  ▼                  ▼
  3.01 snapshot     3.01b delineate    3.01c write_model   3.01e write_
     _config           _region           _reference        experiment_config
                                              │
                                              ▼
                                    3.01d check_model_reference
                                     (sentinel consumed by 3.09)
                                              │
  3.02 extract_climate_grid  (SHARED with WF1 1.10)
        │                                     │
        ▼                                     │
  3.03 climate_stress_parameters               │
        │  cst_1..N.csv                        │
        │                3.04 prepare_weagen_config
        │                        │  weathergen_config.yml
        │                        ▼
        │                3.06 generate_weather_realization
        │                        │  rlz_1..R_cst_0.nc   (unperturbed)
        └────────────┐           │
                     ▼           ▼
              3.07 generate_climate_stress_test
                     │  rlz_<n>_cst_<m>.nc   (perturbed)
                     ▼
              3.08 climate_data_catalog
                     │
                     ▼
              3.09 downscale_climate_realization  ◄── model + guard sentinel
                     │  inmaps + per-member TOML
                     ▼
              3.10 run_wflow_batch_<b>   (B members per Julia session)
                     │  per-member run CSVs
                     ▼
              3.11 derive_wflow_indicators
                     │  q_indicators.csv · basin_indicators.csv
                     ▼
              3.12 gather_benchmarks · 3.13 gather_logs
```

| # | current | after R10 | what it does |
|---|---|---|---|
| 3.00 | `all` | — | Target aggregator: the two experiment indicator tables. |
| 3.00b | `check_project_consistency` | — | Startup drift guard. A WF3 config is a *full* config, so its project-level sections must describe the same project the built model came from; this fails loud on divergence, naming the diverging key, rather than letting the experiment silently reuse a model built under other settings. |
| 3.01 | `snapshot_config` | — | As WF1 1.01, but the snapshot stays inside the experiment. |
| 3.01b | `delineate_region` | — | As WF1 1.01b — the same one project region artifact. |
| 3.01c | `write_model_reference` | — | Records **which model state** this experiment used: the model's path, a pointer-derived digest, and the per-input hashes behind it. Not a copy — a hash answers the question a duplicated staticmaps would. The per-input hashes are kept so the guard can *name* what changed. |
| 3.01d | `check_model_reference` | — | The other half: recomputes the fingerprint and refuses to simulate if the live model has changed since the experiment was recorded. Its sentinel is an input of 3.09 — the first rule to touch the model — because a check after the work is a post-mortem, not a guard. |
| 3.01e | `write_experiment_config` | — | Records the experiment's own parameters, separately from the project ones. |
| 3.02 | `extract_climate_grid` | **`extract_historical_climate`** | The shared historical-climate store producer — the same rule as WF1 1.10. Usually already current when run in pipeline order. |
| 3.03 | `climate_stress_parameters` | **`prepare_stress_test_grid`** | Writes one `cst_<m>.csv` per stress-test point: twelve monthly rows of temperature delta, precipitation mean factor and precipitation variance factor, enumerated over the configured grid. **This is what creates the stress test.** |
| 3.04 | `prepare_weagen_config` | **`prepare_weathergen_config`** | Assembles the one weather-generator config from the template plus the project settings, including the year arithmetic and the two transient-change flags. |
| 3.06 | `generate_weather_realization` | **`generate_weather_realizations`** | Runs weathergenr once to produce **all** `RLZ_NUM` stochastic realizations of the historical climate — the unperturbed `cst_0` baselines. |
| 3.07 | `generate_climate_stress_test` | **`perturb_climate_realization`** | Takes one unperturbed realization and one stress-test point, and applies that perturbation — precipitation mean and variance factors, temperature delta, transient flags, PET recompute. **It applies the stress test; 3.03 creates it.** |
| 3.08 | `climate_data_catalog` | **`write_climate_data_catalog`** | Enumerates every generated climate file into a hydromt data catalog the downscaling step reads. |
| 3.09 | `downscale_climate_realization` | — | Downscales one perturbed realization onto the Wflow grid via hydromt, producing that member's forcing and its run TOML. |
| 3.10 | `run_wflow_batch_<b>` | — | Runs Wflow.jl for every member, `B` per Julia session to amortise startup. Rule identifiers are per batch while the log label stays the singular `3.10_run_wflow` — deliberately, so **this rule is exempt from the rename call-site rule**. |
| 3.11 | `derive_wflow_indicators` | — | Reduces every member's run to the two indicator tables that form the response surface. WF3's terminal product. |
| 3.12 | `gather_benchmarks` | — | Merges the per-rule timing parts into one benchmark table. |
| 3.13 | `gather_logs` | — | Merges every WF3 log part into one workflow log. |

**3.05 no longer exists.** It wrote one weather-generator config per member, and
nothing in that file varied except the output filename; C29 removed it and passes
the filename as an argument instead.

---

## Where the rules meet the artifacts

For what each rule reads and writes, rather than what it does:

- `dev/reference/workflows/model_creation.md`, `climate_experiment.md` — per-workflow detail.
- `dev/reference/contracts/weather-generator-seam.md`, `hydrological-model-seam.md` — the pinned interchange surfaces.
- `dev/milestones/r09/wf3-changes-proposal.md` appendix — the WF3 chain step by step, with the declared inputs of each stage.
