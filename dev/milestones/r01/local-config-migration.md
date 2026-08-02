# Migrating a local snake config to R01 sectioned schema

If you have a `*_local.yml` config (gitignored, per-machine), it still
uses the old flat schema after R01 lands. Migrate it manually using
the mapping below. The Snakefiles read sectioned config only after
R01, so old configs will fail with `KeyError`.

## Mapping table

| Old (flat) key                    | New (sectioned) location                                     |
| --------------------------------- | ------------------------------------------------------------ |
| `project_dir`                     | `project.project_dir`                                        |
| `static_dir`                      | `project.static_dir`                                         |
| `data_sources`                    | `project.data_sources`                                       |
| `data_sources_climate`            | `project.data_sources_climate`                               |
| `model_region`                    | `shared.basin.region`                                        |
| `model_resolution`                | `shared.basin.resolution`                                    |
| `starttime` (ISO datetime)        | `shared.historical_window.starttime`                         |
| `endtime` (ISO datetime)          | `shared.historical_window.endtime`                           |
| `clim_historical`                 | `shared.clim_historical`                                     |
| `wflow_outvars`                   | `workflows.model_creation.wflow_outvars`                     |
| `model_build_config`              | `workflows.model_creation.model_build_config`                |
| `waterbodies_config`              | `workflows.model_creation.waterbodies_config`                |
| `output_locations`                | `workflows.model_creation.output_locations`                  |
| `observations_timeseries`         | `workflows.model_creation.observations_timeseries`           |
| `clim_project`                    | `workflows.climate_projections.clim_project`                 |
| `models`                          | `workflows.climate_projections.models`                       |
| `scenarios`                       | `workflows.climate_projections.scenarios`                    |
| `members`                         | `workflows.climate_projections.members`                      |
| `variables`                       | `workflows.climate_projections.variables`                    |
| `start_month_hyd_year`            | `workflows.climate_projections.start_month_hyd_year`         |
| `historical: 1980, 2010` (str)    | `workflows.climate_projections.historical_year_range: [1980, 2010]` (list) |
| `future_horizons.<h>: y, y` (str) | `workflows.climate_projections.future_horizons.<h>: [y, y]` (list) |
| `save_grids`                      | `workflows.climate_projections.save_grids`                   |
| `experiment_name`                 | `workflows.climate_experiment.experiment_name`               |
| `realizations_num`                | `workflows.climate_experiment.realizations_num`              |
| `horizontime_climate`             | `workflows.climate_experiment.horizontime_climate`           |
| `run_length`                      | `workflows.climate_experiment.run_length`                    |
| `run_historical`                  | `workflows.climate_experiment.run_historical`                |
| `temp.*`                          | `workflows.climate_experiment.stress_test.temp.*`            |
| `precip.*`                        | `workflows.climate_experiment.stress_test.precip.*`          |
| `Tlow`                            | `workflows.climate_experiment.Tlow`                          |
| `Tpeak`                           | `workflows.climate_experiment.Tpeak`                         |
| `aggregate_rlz`                   | `workflows.climate_experiment.aggregate_rlz`                 |

## New keys (no old equivalent)

For each workflow, add `enabled: true` (forward-compat marker;
documentary in R01). Default `true` for all three workflows.

## Format changes: string → list

Two projection keys change type, not just location:

- `historical: 1980, 2010` (a comma-separated string) →
  `historical_year_range: [1980, 2010]` (a YAML list of two integers).
- `future_horizons.<horizon>: 2030, 2060` (string) →
  `future_horizons.<horizon>: [2030, 2060]` (list).

The Snakefiles pass these through to `src/get_change_climate_proj.py`,
which now accepts either form, but new configs should use lists.

## Verification

After migrating your local config, dry-run all 3 Snakefiles to catch
typos:

```bash
pixi run snakemake all -c 1 -s Snakefile_model_creation     --configfile <your_local_config.yml> --dry-run
pixi run snakemake all -c 1 -s Snakefile_climate_projections --configfile <your_local_config.yml> --dry-run
pixi run snakemake all -c 1 -s Snakefile_climate_experiment  --configfile <your_local_config.yml> --dry-run
```

Any `KeyError` in the dry-run output means a key wasn't migrated —
trace it via the table above.

## Reference

For a clean sectioned config, see:
- `config/snake_config.template.yml` (annotated template)
- `config/snake_config_model_test.yml` (working example, tiny bbox)
- `tests/snake_config_model_test.yml` (test fixture)
