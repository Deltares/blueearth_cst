---
title: "Creating a project"
subtitle: "Create case files, choose settings and start an assessment"
---

Set up a basin assessment, check its data and run the workflows needed for your question. Keep
[case](glossary.md#case) files and results outside `blueearth_cst` so one installation can serve several
basins.

Complete the [installation guide](install.md) and try the [Quick start](quick-start.md). You will need:

- A basin outlet in longitude and latitude (decimal degrees).
- Spatial and historical climate datasets covering your basin and analysis period.
- Storage for the model, climate series and simulation outputs.
- An assessment question, such as how seasonal flows respond to warmer or drier conditions.

The supplied starter provides settings and input templates. You must arrange access to the datasets:
the default catalog uses the Deltares P-drive, and CMIP6 analysis also needs internet access.

## 1. Create the case files {#create-the-case-files}

Open PowerShell in the [toolbox folder](glossary.md#toolbox-folder), `blueearth_cst`. Choose a folder for your
cases, replacing `C:/cst-cases` below:

```powershell
New-Item -ItemType Directory -Force C:/cst-cases
```

Replace `my-basin`, the coordinates and purpose with your own values. Use a lowercase, hyphenated
case name; longitude comes before latitude:

```powershell
pixi run create-case my-basin --cases-root C:/cst-cases --longitude 12.5 --latitude -3 --purpose "Assess sensitivity of seasonal flows to climate change"
```

Append `--dry-run` to preview the files without writing them. Creation never runs workflows or overwrites an
existing [trial](glossary.md#trial).

You now have a case folder for shared input files and a first trial, `default`, whose settings files are ready
to edit. The trial starts with only historical climate analysis enabled, so your first run checks the climate
data before anything else is built:

```text
C:/cst-cases/
└── my-basin/
    ├── README.md                          # Minimal trial inventory
    ├── .gitignore                         # Trial outputs only
    ├── data/                              # Shared authored inputs
    └── default/
        ├── project_config.yml             # Basin, data and enabled workflows
        ├── project_config_<workflow>.yml  # Five workflow settings files
        ├── outputs/                       # Created by execution
        ├── INSTRUCTIONS.md                # Quick reference: commands and files
        └── notes.md                       # Purpose, revisions and evidence
```

Results go to `my-basin/default/outputs/`, set by `project.project_dir: outputs`. Put input files that every
trial uses in the case's `data/` folder, and files for one trial only in that trial's own `data/` folder.
In the settings files, a relative path starts from the folder of the file that contains it: the trial reaches
the shared outlet file as `../data/output_locations.csv`. A path that starts with `toolbox:`, such as
`toolbox:config/catalogs/deltares_data_pdrive.yml`, points into the toolbox folder.

### Add a trial

Use another trial to test an alternative, such as a different forcing dataset, without changing the first.
Start it from the maintained starter, reusing the case's outlet coordinates:

```powershell
pixi run create-case my-basin --cases-root C:/cst-cases --trial v2 --purpose "Compare alternative forcing" --dry-run
```

Or copy the scientific settings and enabled workflows of a named trial in the same case:

```powershell
pixi run create-case my-basin --cases-root C:/cst-cases --trial alternative-forcing --from-trial default --purpose "Compare alternative forcing" --dry-run
```

Remove `--dry-run` to create it, and give each trial its own purpose. A new trial starts with no outputs and
its own `notes.md`. Copying takes the source trial's own input files, keeps references to shared case data and
lists any files outside the case; it never copies outputs or run history. See
[Changes and migration](../technical-manual/changes-and-migration.md#copying-a-trial) for the inputs that
copying refuses and for the trial table in the case README.

**Edit your case files.** Shared settings belong in `project_config.yml`; workflow settings belong in the
files below. Pass `project_config.yml` to the [run command](glossary.md#run-command) with `--config`.

| File | Workflow and main choices |
|---|---|
| `project_config_analyze_climate.yml` | Historical climate diagnostics and coverage checks |
| `project_config_build_model.yml` | Historical model period, observations and build settings |
| `project_config_analyze_projections.yml` | CMIP6 models, SSPs and reference/future periods |
| `project_config_generate_scenarios.yml` | Weather realizations, perturbations, periods and seed |
| `project_config_simulate_system.yml` | Experiment, steps, indicator sets and resources |

The commented case files and [technical reference](../technical-manual/configuration.md) cover
settings beyond the key choices below.

## 2. Define the catchment and data sources {#define-the-catchment-and-data-sources}

Open `project_config.yml`. The case-creation command has filled in your outlet and paths.
Use `--catalog` during creation to select a different data catalog.

### Shared assessment choices {#set-the-shared-assessment-choices}

Each item below is one decision in `project_config.yml`.

1. **Output and data** (`project.project_dir`, `project.catalog`): keep `outputs` and point the catalog at
   datasets you can reach. A catalog locates data but does not download it, so
   check that its datasets cover your basin and years before running.
2. **Catchment** (`basin.region`, `basin.resolution`, `basin.sources`): give the outlet as longitude,
   then latitude. `uparea` in `basin.region` is a threshold in km² that the outlet's upstream area must
   reach, not a target catchment area. Choose the grid spacing in degrees; a finer grid adds no detail that
   the source datasets lack. Check that the names under `basin.sources` exist in your catalog.
3. **[Historical forcing](glossary.md#historical-forcing)** (`climate.forcings`, `climate.default_forcing`):
   declare one set per candidate dataset, compare them in [section 4](#check-historical-climate-data), then
   name your choice in `climate.default_forcing`. The choice also conditions weather generation, and the
   toolbox does no local calibration.
4. **[Climate window](glossary.md#climate-window)** (`climate.window`): the historical years to extract;
   see the [period guidance below](#keep-the-analysis-periods-consistent).
5. **Model outputs** (`model.outvars`): include every variable your indicators need
   ([section 5](#choose-indicators)).
6. **Workflow selection** (`workflows`, `workflow_configs`): set a workflow's switch to `true` to run it.
   `false` skips it without deleting its results; check that those results still suit your settings before
   reusing them.

For example, the workflow selection has this shape:

```{.yaml filename="project_config.yml (excerpt)"}
workflows:
  analyze_climate: true
  build_model: false
workflow_configs:
  analyze_climate: project_config_analyze_climate.yml
  build_model: project_config_build_model.yml
```

This is a fragment, not a complete case file. Use the maintained starter for all five workflow entries.

If the outlet falls on an isolated drainage cell, see
[Search for an outlet within an area](troubleshooting.md#search-for-an-outlet-within-an-area).

### Reporting locations {#output-locations}

List the points where the model should save results in the file named by `basin.output_locations`, the case's
`data/output_locations.csv`: `station_name`, longitude `x` and latitude `y` in decimal degrees, and optionally
`location_role` as `control` (default) or `observation`. Each point must drain to the basin outlet. The
toolbox moves each point downstream along the drainage to a river cell; after a run, compare the original and
resolved coordinates in `data/spatial/location_registry.csv` in the output directory before changing any
tolerance. See [location matching](../technical-manual/input-formats.md) for the matching rules and
observation records.

## 3. Set analysis periods {#keep-the-analysis-periods-consistent}

The historical climate record, model run, projection windows and generated scenarios serve different
purposes; they do not need identical dates.

- `climate.window`: historical calendar years, with at least 16 years for weather generation.
- Model building: `run_period` uses water years within the available historical forcing.
- Projection analysis: `reference_period` and `future_periods` use calendar years;
  historical runs end in 2014.
- Scenario generation: `future_period` sets the years of the scenario simulation.

Align projection and scenario periods when making a like-period plausibility comparison.

## 4. Check historical climate data

Run historical climate analysis, the only workflow the starter enables, to check coverage and seasonal
patterns before choosing the forcing for your assessment. From the `blueearth_cst` folder, run:

```powershell
$config = 'C:/cst-cases/my-basin/default/project_config.yml'
pixi run python scripts/run_workflows.py --config $config --cores 3
```

Append `-- --dry-run` to preview the tasks. The run command executes the enabled workflows and writes to your
configured [output directory](glossary.md#output-directory).

**Review the outputs:** inspect the basin boundary, river network,
[reporting locations](glossary.md#reporting-location) and climate diagnostics using
[Reading outputs](outputs.md#inspect-the-basin-and-historical-conditions). Then choose the forcing as below
and set `climate.default_forcing` to that set. You can stop here for a climate-only study.

### Choose historical forcing {#choose-historical-forcing}

The forcing you choose drives the historical model run and conditions the weather generator, so its biases
carry into both; see [Limitations](../technical-manual/limitations.md). Compare the candidates on each point
below, using each source's figures (examples in [Reading outputs](outputs.md#historical-climate)) and the
side-by-side comparison in `data/climate/historical/source_comparison/` in the output directory.

- **Coverage:** every source covers the climate window with at least 16 complete years. CHIRPS covers land
  between 50°S and 50°N only.
- **Seasonal cycle:** the seasonal precipitation figure shows the timing and size of the wet season you
  expect for the basin.
- **Annual totals and trend:** the annual precipitation figure shows similar year-to-year variability across
  sources; a large difference in mean totals is a bias that the model and generated weather will inherit.
- **Spatial pattern:** the precipitation and temperature maps resolve the gradients that matter in your basin.
  A finer grid alone does not make a source more accurate.
- **Extremes and dry spells:** if floods or droughts drive your question, compare annual maximum one- and
  five-day rainfall and dry-spell lengths; gridded products may smooth local extremes.
- **Local evidence:** where rain-gauge records exist, add them with `observations.met_stations` and
  [compare them with the gridded sources](../technical-manual/workflow-analyze-climate.md#rain-gauges).

See [Meteorological data](../technical-manual/historical-meteorological-data.md#checks-before-assessment) for
the checks per dataset.

## 5. Prepare the remaining workflows {#set-workflow-parameters}

Enable the workflows you need and review their settings below. Model building and scenario generation
must finish before running hydrological simulations with the generated scenarios.

### Hydrological model

Set `run_period` in `project_config_build_model.yml` to water years within the climate window. The forcing
and the model variables to save are set in the shared [project file](glossary.md#project-file)
([section 2](#set-the-shared-assessment-choices)).

### Climate projections

In `project_config_analyze_projections.yml`, choose `models` and `scenarios` for your ensemble, and
`reference_period` (default `climate.window`) and `future_periods` for the comparison. The small starter
ensemble is illustrative. Model names use `Institution/Source`; scenarios use SSP identifiers such as
`ssp245`.

### Climate scenarios

In `project_config_generate_scenarios.yml`, under `stochastic:`, choose:

- `future_period`: the years each generated weather series covers.
- `n_realizations`: the number of weather [realizations](glossary.md#realization) per grid point.
- `climate_perturbations`: for `temp` and `precip`, the `min` and `max` change and the number of steps
  `n_levels` between them. Temperature changes are added in °C; precipitation is multiplied, so `0.7` means
  30% less. One value applies to every month; a list of twelve sets January–December. `trajectory: constant`
  applies the full change from the start; `transient` ramps it up over the run.

Each realization also includes an [unperturbed reference](glossary.md#unperturbed-reference): generated
weather without imposed changes, not the historical record. Choose ranges and sample sizes for your question,
rather than copying the demonstration settings.

#### Stress-test design {#designing-the-stress-test}

##### Number of simulations

Each realization covers every point of the [perturbation grid](glossary.md#perturbation-grid) plus one
unperturbed reference:

```text
simulations = n_realizations × (temperature n_levels × precipitation n_levels + 1)
```

The quick-start example (`test_case/project_config_rapid_generate_scenarios.yml`) uses two realizations and a
2 × 2 grid: 2 × (2 × 2 + 1) = 10 simulations. The starter's three precipitation levels give
2 × (2 × 3 + 1) = 14. Each simulation covers the whole `future_period`, so more realizations, levels or years
need more computing time; see [Computation time](../technical-manual/runtime-and-resources.md) for measuring
your own runs. `include_unperturbed: false` skips the reference simulations; see
[Simulations and indicators](#choose-indicators).

##### Return-level indicators

Flood and low-flow return levels are fitted to annual extremes pooled across realizations, so each grid point
has about `future_period` years × `n_realizations` annual values. The fit needs at least ten values, or the
return period in years if that is larger. The rapid example's 9 years × 2 realizations give 18 values: enough
to run the fit, which is an operational minimum, not a reliability guarantee.

More realizations sample more generated weather but add no independent evidence about extremes beyond the
historical record, and with `trajectory: transient` the pooled years mix changing climate conditions. See
[Return-level estimation](../technical-manual/indicator-calculations.md#return-level-estimation-and-supporting-evidence).

##### Ranges and projections

Set the temperature and precipitation ranges over a span you can justify for the basin and your question; see
[From climate stresses to system vulnerabilities](../site/scientific-approach/system-vulnerabilities.qmd).
Projection analysis then shows where projected changes for the basin fall relative to the grid, as context for
judging plausibility. It does not set the grid, and the grid may deliberately extend beyond the projections.
For a like-period comparison, align the periods as in [section 3](#keep-the-analysis-periods-consistent).

### Simulations and indicators {#choose-indicators}

The simulation workflow uses the prepared model and generated scenarios to calculate indicators. Choose sets
for your assessment question and keep their required variables in `model.outvars` before simulation.

- [Default](../../blueearth_cst/indicators/sets/default/README.md): water balance, flow means and extremes.
  Use it for an initial comparison across climate scenarios.
- [ERFA](../../blueearth_cst/indicators/sets/eflow_erfa/README.md) (`eflow-erfa`): flow magnitude, timing,
  variability and intermittency. Use it to compare flow regimes, not classify ecological risk.
- [Crop-water](../../blueearth_cst/indicators/sets/crop_water/README.md): seasonal water availability,
  dry spells and low flows. Use it for catchment screening, not crop-yield or damage estimates.

In `project_config_simulate_system.yml`, select the steps and sets to calculate:

```{.yaml filename="project_config_simulate_system.yml (excerpt)"}
steps:
  simulations: true
  indicators: true
indicators:
  sets: [default, eflow-erfa]
include_unperturbed: true
```

Keep `include_unperturbed: true` when your indicators compare with the unperturbed reference, and supply any
required thresholds or observations. To calculate indicators from saved simulations, set
`steps.simulations: false`; see
[Recalculate indicators](running.md#calculate-different-indicators-from-existing-simulations). The
[stress-test results](results.md) page explains the result tables and unavailable values.

Rerun the command above after updating your settings. See [Running the workflows](running.md)
for execution options and recovery.

## 6. Review and record the results

- **Check the run:** open the summary named in the closing message and review outcomes and warnings.
- **Inspect the results:** use the [outputs guide](outputs.md) to find maps and diagnostics, and
  [Stress-test results](results.md) to read the indicator tables.
  Check basin delineation, forcing and historical model behavior before expanding the stress test.
- **Record your findings:** add data choices, model limits and links to
  [run summaries](glossary.md#run-summary) to `notes.md`.

For case-creation problems, unexpected basin boundaries or workflow failures, see
[Troubleshooting](troubleshooting.md).

To move a case created with an older toolbox layout, see
[Copy and verify an existing case](../technical-manual/changes-and-migration.md#copy-and-verify-an-existing-case).
