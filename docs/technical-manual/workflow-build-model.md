---
title: "Hydrological model building"
format: html
---

Model building prepares a Wflow-SBM basin model and runs it with historical weather. Maps, simulated
time series and comparisons with available discharge observations help assess its behavior before
stress testing. Construction uses global datasets and includes no local calibration.

## Inputs {#inputs}

- **Spatial data.** Terrain, hydrography, land cover, vegetation, soils and waterbody datasets for
  constructing the basin model.
- **Historical climate data.** Selected meteorological forcing for the historical simulation.
- **Build configuration files.** HydroMT settings, defaulting to `config/defaults/wflow_build_model.yml`
  and `config/defaults/wflow_update_waterbodies.yml`.
- **Reporting locations.** Optional CSV specifying locations for model outputs.
- **Discharge observations.** Optional CSV files for evaluation, with columns and units matching
  the modeled locations and variables.

See [input formats](data-formats-and-conventions.md) for file conventions and
[model preparation](model-preparation.md) for the role of each file and links to the upstream manuals.

## Configuration {#configuration}

**Project settings** define the basin, historical forcing and simulated variables.

| Setting | Purpose |
|---|---|
| `basin.region`, `basin.resolution` | Basin extent and model resolution. |
| `basin.output_locations` | Reporting locations. |
| `climate.forcings`, `climate.default_forcing`, `climate.window` | Forcing source and available record. |
| `model.outvars` | Variables saved for evaluation and later indicator calculations. |

**Workflow settings** control the historical simulation and evaluation. The annotated template is
`config/templates/example-basin/project_config_build_model.yml`.

| Setting | Purpose |
|---|---|
| `run_period.start`, `run_period.end` | Simulation years within the climate record. |
| `plotting.input_maps` | Spatial input maps; default `true`. |
| `plotting.forcing_figures` | Model-grid forcing maps and time series; default `false`. |
| `plotting.scales` | `[basin]` (default) or `[basin, locations]` for reporting-location and upstream-subbasin figures. |
| `plotting.view` | With several forcing sets: `comparison` (default) draws station discharge across runs only; `individual`/`both` keep the per-run station figures. |
| `hydromt.build_config_path` | Alternative HydroMT build configuration file. |
| `hydromt.waterbodies_config_path` | Alternative waterbody-update configuration file. |
| `output.netcdf_grid` | Optional native grids from this historical simulation. |
| `historical_forcings` | Named forcing sets to simulate on the fixed model; see [named forcing sets](#named-recipes). |

Wflow treats the starting date as the initial state time. With daily forcing, the first output is one
day later; forcing must cover that first update through the requested end date. See
[execution and run management](configuration-and-execution.md) for running the workflow.

## Processing sequence {#rule-flow}

![Main scientific processing steps and conditional paths described below.](assets/workflow-wf1.svg){width=100% fig-alt="Scientific processing from basin and climate inputs to the assessment products described in this section."}

The workflow prepares basin and thematic data, builds the model and adds historical forcing on the
model grid. It then runs the historical simulation, exports response tables and produces evaluation
figures. Basin and forcing maps can be produced before the simulation finishes; additional climate
sources may be extracted for shared map scales without changing the selected forcing.

## Outputs {#outputs}

Optional `output.netcdf_grid` in this workflow's settings enables native grids at
`models/hydrology/wflow/historical/default/run/output_grid.nc`. See the shared
[configuration reference](configuration.md#netcdf-grid). This request is independent of
ensemble simulation output and leaves scalar outputs unchanged.

Paths below are relative to `project_dir`. The main outputs are:

- **Spatial data (`data/spatial/`).** Basin vectors, reporting-location registry, thematic maps and
  the spatial catalog.
- **Historical climate (`data/climate/historical/`).** The selected climate dataset and shared map
  reference, with additional map candidates where needed.
- **Basin model (`models/hydrology/wflow/`; the folder is `model.name`, see
  [one model per folder](#model-versions)).** Model maps in `staticmaps.nc`, settings in
  `wflow_sbm.toml`, geometries and the HydroMT build configuration files used to build the model.
  `_engine/outlet_index.csv` links outlets to reporting locations.
- **Historical forcing (`models/hydrology/wflow/forcing/`).** Model-grid forcing in
  `inmaps_historical.nc` and optional forcing-check figures (`plotting.forcing_figures: true`).
- **Simulated time series (`models/hydrology/wflow/historical/default/run/`).** Per-variable `output_<code>.csv`
  tables and `_engine/response_tables.json`, with registry IDs linking series to reporting locations.
  Native Wflow `output.csv` is removed after successful conversion. Evaluation reads the retained tables.
- **Evaluation (`models/hydrology/wflow/historical/default/plots/` and `tables/`).** Model-response
  figures and scores where observations are available, including `tables/performance_indicators.csv`
  and station discharge summaries under `plots/stations/` when `plotting.scales` includes `locations`
  (with several forcing sets and `view: comparison`, these are drawn across runs in `forcing_comparison/`).

The completed model supplies subsequent hydrological simulations. Logs, benchmarks and configuration
archives are described in [execution records](run-records.md).

Spatial plots and unique basin-average response figures are generated by default. Model-forcing
plots and individual location/subbasin figures are optional. Enable them in the model-building YAML:

```yaml
plotting:
  input_maps: true
  forcing_figures: true
  scales: [basin, locations]
```

These switches affect figures only. Historical simulation, retained responses, per-variable time-series
exports and observation-based performance metrics remain enabled. `input_maps` defaults to `true`;
`forcing_figures` to `false` and `scales` to `[basin]`. `locations` controls both reporting-location and
associated upstream-subbasin figures; basin-scale figures always render. The forcing comparison
(`forcing_comparison/`) always renders with two or more forcing sets. Earlier keys are refused; see
[renamed settings](changes-and-migration.md#renamed-configuration-settings). Turning a family off leaves its
previously generated files intact; the workflow no longer requests them. Turning it on requests any missing
products through the usual run command.

Shared `plotting.annotation` remains in the project YAML. Legacy `forcing_plots`,
`evaluation.station_plots` and `evaluation.subbasin_plots` are accepted with deprecation warnings;
both evaluation switches map to the `locations` scale. Conflicting declarations, unknown plotting keys
and nonboolean switches are errors.

## Diagnostics and interpretation {#interpretation}

- **Basin and forcing maps.** Maps show the model's spatial inputs and historical forcing.
  They support inspection of the basin setup before interpreting simulated responses.
- **Historical response.** Time series describe model behavior under the selected historical weather.
  Per-variable exports complement the native Wflow output and do not replace it.
- **Observation comparisons.** Available observations support evaluation of simulated behavior at
  matching locations. Without observations, response figures describe the simulation but do not
  establish model skill; a successful run alone does not establish predictive adequacy.

See the [processing rule reference](workflow-rule-reference.md#build-model) for individual steps.

## Observed catchment water balance

Optional `evaluation.budyko: true` in the model-building YAML screens observed runoff against selected
prepared historical precipitation and PET. It requires the project file's
[`observations.river_discharge`](configuration.md#observations), whose `time_label` it uses:

```yaml
evaluation:
  budyko: true
```

Omitted, null or false disables the diagnostic. Enabling it asserts that the discharge file holds daily
mean m³/s over midnight-to-midnight UTC days. Naive timestamps are asserted
UTC; aware timestamps normalize to UTC and must land at midnight. Start labels shift one day in a derived
series; the input CSV and ordinary model evaluation retain their labels. The toolbox checks structure
and records the declaration as user-supplied support; it cannot establish from the values where the
measurements came from.

The diagnostic uses original snapped upstream catchments and hydrography cell areas. Prepared model
forcing is sampled at those cell centres using nearest cells, then area-weighted. The same contributing
area converts daily mean Q to depth with `Q * 86400 * 1000 / area_m2`. This discretization is a screening
approximation; inspect station areas and forcing coverage before interpreting departures.

Only complete matched water years of precipitation, PET and observed Q enter the ratios. The covered day
(interval end minus one day) determines window and water-year membership. Missing days remain missing;
partial edge years are excluded, and no spin-up year is removed. Aridity is total PET / total P and runoff
coefficient is total runoff depth / total P over the same retained years. Zero P or no eligible years
produces an unavailable result. These are ratios of totals, rather than averages of annual ratios.

Outputs live in each forcing-set folder, `models/hydrology/wflow/historical/<id>/`:
`tables/budyko_stations.csv`, `budyko_annual.csv` and `budyko_summary.csv`, the diagram
`plots/budyko_observed_runoff_coefficient_stations.png` (drawn by the evaluation rule), and daily evidence,
input hashes and support declarations in `_engine/budyko_daily.csv` and `_engine/budyko_methods.json`. The
target `evaluate_budyko` requires model/forcing preparation but no Wflow simulation. Retained tables support
`scripts/render_figures.py budyko --project <project_dir> --out <new_dir>`.

Water and energy limits describe long-term equilibrium assumptions. Short/discontinuous records,
storage changes, regulation, transfers, basin area and forcing/observation errors can explain departures.
The residual `1-Q/P` is not measured ET. Unusual points remain visible; no automatic pass/fail or source
ranking is supplied. Tables retain counts, actual years and exclusion reasons for inspection.

## One model per folder {#model-versions}

Each folder under `models/hydrology/` is one self-contained model: its static maps, TOMLs, every forcing
forcing set's run and their comparison. A model's runs change only when it is rerun with the same model
settings. `model.name` in the project file names the model folder that model building creates or runs and
hydrological simulation uses; the default is `wflow`. To try different model-building settings (another soil
dataset), set a new `model.name`: model building refuses to rebuild an existing model whose model-building
settings changed and names the setting that differs.

Every model records itself in `_engine/model.json`: its origin, parent, notes and the identity of its
static inputs (the `wflow_sbm.toml` pointer graph plus `staticgeoms/`, excluding everything the toolbox
writes per forcing set). The `write_model_record` rule writes this record for a model the toolbox builds
(`origin: built`) after every build.

A model prepared outside the toolbox is registered with `scripts/register_model.py register`
(`origin: user_managed`); see [Use a calibrated model](../guide/calibrated-model.md). Model building never
rebuilds or rewrites a registered model: the model-building rules (`build_wflow_model` to
`declare_gauges_and_outputs`) are not defined for it, and `publish_registered_model` checks it again and marks
it final as registered. Every forcing set, including a single one, uses the staged forcing update and writes
its own TOML, so a single forcing set writes `wflow_sbm_default.toml` beside the model's own `wflow_sbm.toml`.
Before planning, model building refuses a registered model whose files changed since registration and one
whose TOML lacks an output that `model.outvars` or `build_model.output` asks for, naming each.

Hydrological simulation uses `model.name`; with several models and no name it stops and lists them. It refuses
a registered model that changed since registration, and a retained experiment refuses a different model
(`--new-experiment`). Each experiment's `_engine/model_reference.yml` records the model folder and the digest
of its forcing-set inputs.

Comparisons across models (`scripts/register_model.py compare`) are written to
`models/hydrology/comparisons/<name>/` with the same tables and figures as a model's own
`historical/forcing_comparison/`. The runs must cover the same simulation period and initialization; the
evaluation window is every complete reporting year they share.

The former `historical_run` setting, which published named copies under `simulations/historical/`, is no
longer used. It is accepted with a deprecation warning and ignored; remove it from the configuration.

The model identity recorded with each comparison covers scientific TOML settings and referenced static,
cyclic, waterbody and initial-state inputs, including forcing variable bindings, scales and offsets. It
excludes the forcing file contents and pointer, simulation dates, output configuration and logging paths,
which have separate source and compatibility checks. NetCDF inputs use content hashes independent of
compression and container timestamps.
## Named forcing sets {#named-recipes}

With [several forcing sets](configuration.md#named-historical-forcing-recipes), the workflow builds the model
once and simulates each set in `build_model.historical_forcings` separately. Each set owns one folder,
`historical/<id>/`, holding `forcing/inmaps_historical.nc`, `run/`, the evaluation `plots/` and `tables/`, and
`_engine/` (its build YAML, store catalog, forcing record and evaluation sidecars), plus its TOML
`wflow_sbm_<id>.toml` at the model root. A single forcing set uses `historical/default/` and the model's own
`wflow_sbm.toml`. TOMLs stay at the model root because HydroMT-Wflow resolves model inputs from there. The
shared `wflow_sbm.toml`, `staticmaps.nc` and geometries are never rewritten per forcing set, and rerunning
one set leaves the others' files untouched. Two or more sets are compared automatically.

## Compare forcing sets {#compare-recipes}

When `build_model.historical_forcings` runs two or more named forcing sets, the workflow compares their
current runs (`compare_historical_forcings`) into `models/hydrology/<model>/historical/forcing_comparison/`:
`tables/` (performance, coverage, daily, monthly, seasonal, flow duration, stations, inventory),
`tables/supporting/support.csv` (per-day support flags), `plots/` and `_engine/comparison.json`. It evaluates
every complete reporting year the runs share: responses are labelled at interval end, so the first simulated
year is never complete. Every input (TOMLs, forcing records, response tables, registry, observations) is
hashed before and after the comparison, so a run that changes meanwhile is refused. The requested dates must
all exist in every run. The comparison requires identical fixed-model identity, simulation dates, timestep,
initialization and station identity; only daily Gregorian responses are supported. The inventory records each
forcing set and its run record. The window expands inclusive integer reporting-year labels using
`water_year_start`; no water-year trimming or endpoint shortening occurs. Water-year labels use their starting
year and seasonal months start at `water_year_start`. Exact requested and warm-up boundaries are recorded in
`_engine/comparison.json`.

Warm-up matches historical model evaluation: when a run has more than 365 samples, exclude timestamps before
its first response timestamp plus one calendar year. This can exclude 366 samples across a leap year.
Runs with at most 365 samples have no warm-up exclusion. Each run supplies its own simulated warm-up;
the common exclusion uses the identical response calendar. Requested warm-up dates remain in the coverage
tables but never enter diagnostics. A wholly excluded window is reported with unavailable results.

At each station, simulation-only diagnostics use dates with finite nonnegative discharge in **all** runs.
Observation-paired diagnostics additionally require finite nonnegative observations on those same dates.
Missing stations or dates are reported without infilling. Duplicate or unknown observation station IDs,
duplicate dates and non-midnight timestamps are errors. Missing observation columns are allowed. Daily and
monthly KGE, NSE, NSElog, RMSE, MSE, Pbias and VE reuse the historical evaluation's calculations. Monthly
means use the same available paired days, with no completeness threshold. Undefined scores retain reasons;
simulation-only results make no skill claim. Seasonal curves average available monthly means equally within
calendar month; flow-duration curves rank available daily values using rank/(count+1), retaining zeros.

Outputs include `inventory.csv`, `coverage.csv`, `performance.csv`, `support.csv`, `daily.csv`, `monthly.csv`,
`seasonal.csv` and `flow_duration.csv` under `tables/`. `support.csv` records every requested date and the
warm-up, simulated-common and observation-paired masks. Coverage counts and first/last usable dates are
station-specific; unavailable observations are explicit. Figures cover every station on each applicable
support, using the shared CST preset, palette, legends and caption/title inventories. Numerical tables and
their hashes are retained for independent inspection and rerendering, without current model files:

```powershell
pixi run python scripts/render_figures.py historical-hydrology --project <comparison-dir> --out <figure-dir>
```

Controlled checks establish arithmetic and artifact handling, not real-basin forcing suitability, local
calibration or empirical model adequacy. Review coverage before interpreting comparative performance.
