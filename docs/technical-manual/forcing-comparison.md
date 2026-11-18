---
title: "Forcing-set comparison"
format: html
---

## Named forcing sets {#named-recipes}

With [several forcing sets](configuration.md#named-historical-forcing-recipes), the workflow builds the model
once and simulates each set in `build_model.historical_forcings` separately. Each set owns one folder,
`historical/<id>/`, holding `forcing/inmaps_historical.nc`, `run/`, the evaluation `plots/` and `tables/`, and
`_engine/` (its build YAML, store catalog, forcing record and accompanying evaluation files), plus its TOML
`wflow_sbm_<id>.toml` at the model root. A single forcing set uses `historical/default/` and the model's own
`wflow_sbm.toml`. TOMLs stay at the model root because HydroMT-Wflow resolves model inputs from there. The
shared `wflow_sbm.toml`, `staticmaps.nc` and geometries are never rewritten per forcing set, and rerunning
one set leaves the others' files untouched. Two or more sets are compared automatically.

## Compare forcing sets {#compare-recipes}

When `build_model.historical_forcings` runs two or more named forcing sets, the workflow compares their
current runs (`compare_historical_forcings`). The comparison gives scores, discharge diagnostics and coverage
for every station, side by side. It is written to `models/hydrology/<model>/historical/forcing_comparison/`:
tables in `tables/`, figures in `plots/` and the run record in `_engine/comparison.json`.

### Requirements

- The runs must have the same fixed-model record, simulation dates, timestep, initialization and
  stations. The requested dates must all exist in every run.
- Only daily Gregorian responses are supported.
- Every input (TOMLs, forcing records, response tables, registry, observations) is hashed before and after
  the comparison, so a run that changes meanwhile is refused.

### Evaluation period

- The comparison evaluates every complete reporting year the runs share. Responses are labelled at interval
  end, so the first simulated year is never complete.
- The window expands inclusive integer reporting-year labels using `water_year_start`; no water-year trimming
  or endpoint shortening occurs. Water-year labels use their starting year, and seasonal months start at
  `water_year_start`.
- Warm-up matches historical model evaluation: when a run has more than 365 samples, timestamps before its
  first response timestamp plus one calendar year are excluded. This can exclude 366 samples across a leap
  year. Runs with at most 365 samples have no warm-up exclusion. Each run supplies its own simulated warm-up;
  the common exclusion uses the identical response calendar.
- Requested warm-up dates remain in the coverage tables but never enter diagnostics. A wholly excluded window
  is reported with unavailable results. Exact requested and warm-up boundaries are recorded in
  `_engine/comparison.json`.

### Dates and scores

- At each station, simulation-only diagnostics use the dates with finite nonnegative discharge in **all**
  runs. Observation-paired diagnostics additionally require finite nonnegative observations on those dates.
- Missing stations or dates are reported without infilling. Missing observation columns are allowed;
  duplicate or unknown observation station IDs, duplicate dates and non-midnight timestamps are errors.
- Daily and monthly KGE, NSE, NSElog, RMSE, MSE, Pbias and VE reuse the historical evaluation's calculations.
  Monthly means use the same available paired days, with no completeness threshold. Undefined scores keep
  their reasons; simulation-only results make no skill claim.
- Seasonal curves average available monthly means equally within each calendar month. Flow-duration curves
  rank available daily values using rank/(count+1), keeping zeros.

### Output tables

| File | Contents |
|---|---|
| `tables/inventory.csv` | One row per run: model and forcing hashes, forcing-set settings, start, end, sample count, toolbox and environment records |
| `tables/coverage.csv` | Per station: requested, warm-up, simulated-common, observation-paired and excluded day counts, first and last usable dates, and whether observations are available (with a reason when not) |
| `tables/performance.csv` | Per station, run, scale (daily or monthly) and metric: the score, the number of pairs and the reason for an unavailable score |
| `tables/daily.csv` | Daily discharge per station, support (simulated-common or observation-paired) and series (each run, plus the observations on the paired support) |
| `tables/monthly.csv` | Monthly mean discharge and its day count, per station, support and series |
| `tables/seasonal.csv` | Mean of the monthly means for each calendar month, per station, support and series |
| `tables/flow_duration.csv` | Exceedance (%) and discharge per station, support and series |
| `tables/stations.csv` | The first run's station registry, sorted by `wflow_id`; every run must have the same registry apart from `station_name` |
| `tables/supporting/support.csv` | Per-day support flags: every requested date with the warm-up, simulated-common and observation-paired masks, invalid-simulation count, observation validity and water year |

Figures in `plots/` cover every station on each applicable support, using the shared CST preset, palette,
legends and caption/title inventories.

### Regenerate figures

The numerical tables and their hashes are saved for independent inspection and rerendering, without current
model files:

```powershell
pixi run python scripts/render_figures.py historical-hydrology --project <comparison-dir> --out <figure-dir>
```

### Limits

Controlled checks establish arithmetic and artifact handling, not real-basin forcing suitability, local
calibration or empirical model adequacy. Review coverage before interpreting comparative performance.
