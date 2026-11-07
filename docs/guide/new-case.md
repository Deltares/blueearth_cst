---
title: "Creating a project"
subtitle: "Create case files, choose settings and start an assessment"
---

Set up a basin assessment, check its data and run the workflows needed for your question.
Keep case files and results outside `blueearth_cst` so one installation can serve several basins.

Complete the [installation guide](install.md) and try the [Quick start](quick-start.md). You will need:

- A basin outlet in longitude and latitude (decimal degrees).
- Spatial and historical climate datasets covering your basin and analysis period.
- Storage for the model, climate series and simulation outputs.
- An assessment question, such as how seasonal flows respond to warmer or drier conditions.

The supplied starter provides settings and input templates. You must arrange access to the datasets:
the default catalog uses the Deltares P-drive, and CMIP6 analysis also needs internet access.

## 1. Create the case files {#create-the-case-files}

Open PowerShell in `blueearth_cst`. Choose a location for your cases, replacing `C:/cst-cases` below:

```powershell
New-Item -ItemType Directory -Force C:/cst-cases/applications
```

Replace `my-basin`, the coordinates and purpose with your own values. Use a lowercase, hyphenated
case name; longitude comes before latitude:

```powershell
pixi run create-case my-basin --cases-root C:/cst-cases --longitude 12.5 --latitude -3 --purpose "Assess sensitivity of seasonal flows to climate change"
```

Append `--dry-run` to preview the files. Creating a case does not run workflows or overwrite an existing case.

Your case files are created under `C:/cst-cases/applications/my-basin/`:

```text
C:/cst-cases/
└── applications/
    └── my-basin/
        ├── project_config.yml             # Basin, data and enabled workflows
        ├── project_config_<workflow>.yml  # Five workflow settings files
        ├── data/                          # Output locations and observations
        ├── INSTRUCTIONS.md                # Case commands and documentation
        └── notes.md                       # Assessment notes and findings
```

Results go to `C:/cst-cases/runs/my-basin/`, set by `project.project_dir` in `project_config.yml`.
Keep `INSTRUCTIONS.md` with the case: it records the toolbox version and commands.

**Edit your case files.** Shared settings belong in `project_config.yml`; workflow settings belong
in the files below. Pass `project_config.yml` to the runner with `--config`.

| File | Workflow and main choices |
|---|---|
| `project_config_analyze_climate.yml` | Historical climate diagnostics and coverage checks |
| `project_config_build_model.yml` | Historical model period, observations and build settings |
| `project_config_analyze_projections.yml` | CMIP6 models, SSPs and reference/future periods |
| `project_config_generate_scenarios.yml` | Weather realizations, perturbations, periods and seed |
| `project_config_simulate_system.yml` | Experiment, operations, indicator sets and resources |

The commented case files and [technical reference](../technical-manual/configuration.md) cover
settings beyond the key choices below.

## 2. Define the catchment and data sources {#define-the-catchment-and-data-sources}

Open `project_config.yml`. The case-creation command has filled in your outlet and paths.
Use `--catalog` during creation to select a different data catalog.

### Shared assessment choices {#set-the-shared-assessment-choices}

1. **Outputs and data:** check `project.project_dir` and `project.catalog`.
   Catalogs locate data; they do not install it. Check the basin and time coverage before execution.
2. **Catchment:** review `basin.region`, `basin.resolution` and `basin.output_locations`.
   Coordinates are longitude then latitude; grid spacing is in degrees. The `uparea` delineation
   threshold is in km², not a target catchment area. Finer grids do not add detail to source datasets.
   Check that `basin.sources` names exist in your catalog.
3. **Historical forcing:** compare `climate.sources`, then choose `climate.forcing.precip_source`.
   Selected forcing also conditions weather generation; the toolbox does no local calibration.
4. **Historical period:** set `climate.window` using the
   [period guidance below](#keep-the-analysis-periods-consistent).
5. **Model outputs:** choose `model.outvars` so later indicators have their required responses.
6. **Workflow selection:** check each workflow's `config_path` and set `enabled: true` for those you need.
   Use `false` to skip a workflow without deleting its results; check their suitability before reuse.

For example, the workflow selection has this shape:

```yaml
workflows:
  analyze_climate:
    enabled: true
    config_path: project_config_analyze_climate.yml
  build_model:
    enabled: false
    config_path: project_config_build_model.yml
```

This is a fragment, not a complete case file. Use the maintained starter for all five workflow entries.

### Search for an outlet within an area

If an approximate outlet falls on an isolated drainage cell, use a `subbasin` search box.
HydroMT expects coordinates in west, south, east, north order:

```yaml
basin:
  region: "{'subbasin': [29.81375, 31.12208, 29.87375, 31.18208], 'uparea': 100}"
  sources:
    hydrography: merit_hydro_ihu
    basin_index: merit_hydro_ihu_index
```

Use a matching hydrography/index available in your catalog.
The box searches for outlets; upstream catchments can extend outside it and several rivers can select
several basins. It is distinct from the `bounds` argument limiting available hydrography.
Inspect the result before downstream runs. This search does not revise your output-location CSV.
Record any manual coordinate correction in case notes.

### Output locations

Provide `station_name`, longitude `x` and latitude `y` in decimal degrees. The optional `location_role`
is `control` (default) or `observation`.

Locations must lie in their declared parent basin and drain to its outlet. The toolbox follows drainage
to a river cell within `basin.delineation.snap_tolerance_m`; it does not snap across a divide, and it rejects
points draining into a pit. Inspect the original and resolved coordinates before changing tolerances.
See [location matching](../technical-manual/input-formats.md) for station identifiers and observation matching.

## 3. Set analysis periods {#keep-the-analysis-periods-consistent}

The historical climate record, model run, projection windows and generated scenarios serve different
purposes; they do not need identical dates.

- `climate.window`: historical calendar years, with at least 16 years for weather generation.
- Model building: `simulation_window` uses water years within the available historical forcing.
- Projection analysis: `reference_window` and `future_windows` use calendar years;
  historical runs end in 2014.
- Scenario generation: `simulation_window` sets the years of the scenario simulation.

Align projection and scenario periods when making a like-period plausibility comparison.

## 4. Check historical climate data

The starter enables only historical climate analysis. Check coverage and seasonal patterns
before choosing the forcing for your assessment.

From the `blueearth_cst` folder, run:

```powershell
$config = 'C:/cst-cases/applications/my-basin/project_config.yml'
pixi run python scripts/run_workflows.py --config $config --cores 3
```

Append `-- --dry-run` to preview tasks. The runner executes the enabled workflows and writes to
your configured output directory.

**Review the outputs:** inspect the basin boundary, river network, reporting locations and climate
diagnostics using
[Reading outputs](outputs.md#inspect-the-basin-and-historical-conditions). Then set `climate.forcing.precip_source`
to your chosen forcing. You can stop here for a climate-only study.

## 5. Prepare the remaining workflows {#set-workflow-parameters}

Enable the workflows you need and review their settings below. Model building and scenario generation
must finish before running hydrological simulations with the generated scenarios.
Projections provide context; they do not set the scenario grid.

### Hydrological model

Set `simulation_window` in
`project_config_build_model.yml` to water years within the historical forcing record.
Choose historical forcing and the model variables to save in the shared project file.

### Climate projections

In `project_config_analyze_projections.yml`, choose `models` and `scenarios` for your ensemble,
and `reference_window` and `future_windows` for the comparison. The small starter ensemble is
illustrative. Model names use `Institution/Source`; scenarios use SSP identifiers such as `ssp245`.

### Climate scenarios

In `project_config_generate_scenarios.yml`, choose:

- `simulation_window` and `n_realizations`: years and weather realizations per grid point.
- `climate_perturbations`: monthly temperature changes (°C) and precipitation multipliers, with
  `min` and `max` values in January–December order; `0.7` means 30% less precipitation.
- `n_levels` within each axis: two temperature and three precipitation levels give six combinations,
  plus an unperturbed reference.

Choose ranges and sample sizes for your question, rather than copying the demonstration settings.
The unperturbed reference is generated weather without imposed changes, not the historical record.

### Simulations and indicators {#choose-indicators}

The simulation workflow uses the prepared model and generated scenarios to calculate indicators.
Choose sets for your assessment question and retain their required variables in `model.outvars`
before simulation.

- [Default](../../blueearth_cst/indicators/sets/default/README.md): water balance, flow means and extremes.
  Use it for an initial comparison across climate scenarios.
- [ERFA](../../blueearth_cst/indicators/sets/eflow_erfa/README.md) (`eflow-erfa`): flow magnitude, timing,
  variability and intermittency. Use it to compare flow regimes, not classify ecological risk.
- [Crop-water](../../blueearth_cst/indicators/sets/crop_water/README.md): seasonal water availability,
  dry spells and low flows. Use it for catchment screening, not crop-yield or damage estimates.

In `project_config_simulate_system.yml`, select the operations and sets to calculate:

```yaml
operations: [simulations, indicators]
indicator_sets:
  - set: default
  - set: eflow-erfa
include_unperturbed: true
```

Keep `include_unperturbed: true` when your indicators require reference runs, and supply any required
thresholds or observations. To calculate indicators from saved simulations, use `operations: [indicators]`;
see [Recalculate indicators](running.md#calculate-different-indicators-from-existing-simulations).
The [outputs guide](outputs.md#reading-an-indicator-set) explains the result tables and unavailable values.

Rerun the command above after updating your settings. See [Running the workflows](running.md)
for execution options and recovery.

## 6. Review and record the results

- **Check the run:** open the summary named in the closing message and review outcomes and warnings.
- **Inspect the results:** use the [outputs guide](outputs.md) to find maps, diagnostics and indicator tables.
  Check basin delineation, forcing and historical model behavior before expanding the stress test.
- **Record your findings:** add data choices, model limits and links to run summaries to `notes.md`.

For case-creation problems, unexpected basin boundaries or workflow failures, see
[Troubleshooting](troubleshooting.md).
