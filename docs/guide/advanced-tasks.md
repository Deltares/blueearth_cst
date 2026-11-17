---
title: "Advanced tasks"
subtitle: "Regenerate figures, rebuild outputs, transfer results and prune logs"
---

These tasks are occasional additions to the routine described in [Running the workflows](running.md). The
commands use `$config` and `$project_dir` as set in
[Run everything enabled in the project file](running.md#run-everything-enabled-in-the-project-file).

## Regenerate figures after visual changes

To refresh a figure from existing results, select its plotting target with the same configuration
and output directory as the original run. For example:

```powershell
pixi run python scripts/run_workflow.py analyze_projections --config $config --project-dir $project_dir --target plot_climate_projections --dry-run -- --force --notemp
```

Check the preview, then remove `--dry-run`. `--force` reruns the selected target and `--notemp`
keeps temporary inputs. Missing or outdated inputs may still require earlier tasks to run;
avoid `--forceall` when you only want new figures.

| Figures | Workflow argument | Target |
|---|---|---|
| Historical source diagnostics | `analyze_climate` | `plot_climate_diagnostics_<source>` |
| Climate-source comparison | `analyze_climate` | `plot_climate_comparison` |
| Basin maps | `build_model` | `plot_basin_map` |
| Model forcing | `build_model` | `plot_model_forcing` |
| Historical model evaluation | `build_model` | `plot_model_evaluation` |
| Projected climate changes | `analyze_projections` | `plot_climate_projections` |

Replace `<source>` with a configured source such as `era5`. Some plotting tasks also recalculate summaries or
indicators, so they are not all image-only operations. Keep the underlying climate, spatial and model-response
files. In particular, historical evaluation reads the per-variable tables `output_<code>.csv` and their record
`_engine/response_tables.json` in `historical/default/run/`.

Use `plotting.annotation` in the project file to select `none`, `caption` or `title`.
See [plot annotations](../technical-manual/shared-configuration.md#plot-annotations) for details.

## Rebuild outputs for testing

A rebuild is an intentional fresh start, not routine failure recovery. **It permanently deletes
generated spatial and climate data, models, climate scenarios and all experiments in the project,
including outputs of disabled workflows. No backup is made.**

Preview the deletion plan first:

```powershell
pixi run python scripts/run_workflows.py --config $config --rebuild -- --dry-run
```

To proceed, remove `-- --dry-run` and answer the confirmation prompt. Source configurations, external source
datasets, logs and attempt history are kept, but custom files inside generated output folders are treated as
outputs and deleted. Models the toolbox built and cross-model comparisons are deleted; a registered model
(`scripts/register_model.py register`) is kept.

Only enabled workflows run afterwards. To rebuild hydrological simulations, also enable model building and
scenario generation and set `steps.simulations: true`: their required results were deleted. Use
`--new-experiment` instead when you want to keep previous experiments. Read the
[reset boundaries](../technical-manual/execution-and-provenance.md#launcher-arguments-and-safety-boundaries)
before using rebuild.

## Transfer climate-projection results

Completed climate-projection results can be exported to a ZIP and imported into another project.
`$project_dir` is the source; replace the ZIP path and the destination output directory below:

```powershell
pixi run python scripts/transfer_results.py export --project-dir $project_dir --output "C:/exports/cmip6-results.zip"
pixi run python scripts/transfer_results.py import "C:/exports/cmip6-results.zip" --project-dir "C:/cst-cases/receiving-case/default/outputs"
```

Export writes a new ZIP. Before importing, check the
[destination requirements](../technical-manual/results-transfer.md#destination-checks-and-import): import
validates integrity, basin geometry and applicable configuration, and does not merge into a different existing
projection result. Check the command's result, then inspect the imported report and figures using
[Reading outputs](outputs.md#reading-projection-diagnostics).

This transfers projection results for reading and interpretation. It does not recreate the source environment
or guarantee reuse by a later projection-analysis run.

## Prune old log folders

Each launch keeps its log folder under `logs/`. To list old log folders, run
`pixi run python dev/scripts/prune_logs.py --project-dir <project_dir> --older-than 30`; add `--delete` to
remove them. Folders that saved projection results refer to are never listed.
