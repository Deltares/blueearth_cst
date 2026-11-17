---
title: "Running the workflows"
subtitle: "Run selected analyses, continue an assessment and reuse completed simulations"
---

Use this guide after [creating a case](new-case.md) and reviewing its configuration. It explains
how to run the analyses you need, check their completion and continue working with existing results.
For a first demonstration, follow the [Quick start](quick-start.md).
The Technical Manual explains
[execution behavior and reuse requirements](../technical-manual/configuration-and-execution.md).

Open PowerShell in the `blueearth_cst` folder used to install the toolbox. Run a [case](glossary.md#case) from
the same toolbox folder each time, with only one active command writing to its
[output directory](glossary.md#output-directory).

## Run everything enabled in the project file

Set the path to your case's [project file](glossary.md#project-file) and its output directory, replacing the
examples below with your own. Some commands on this page also use `$project_dir`:

```powershell
$config = 'C:/cst-cases/my-basin/default/project_config.yml'
$project_dir = 'C:/cst-cases/my-basin/default/outputs'
pixi run python scripts/run_workflows.py --config $config --cores 3
```

The [run command](glossary.md#run-command) executes workflows set to `true` under `workflows:` in this order: historical
climate analysis, model building, projection analysis, scenario generation, then hydrological simulations and
indicators.

**Check the opening summary:** confirm the configuration and output directory (`project.project_dir`).
The command uses up to three CPU cores; increasing `--cores` can also increase memory use.

Append `-- --dry-run` for a preview (dry run) of the planned tasks without executing them. On a first run,
some later tasks cannot be fully planned until earlier workflows have produced their inputs.

Run only the workflows you need. Hydrological simulations require a prepared model and generated
scenarios. Projections provide context without driving scenario generation. Historical climate analysis
can run on its own to help you choose forcing.

Disabling a workflow leaves its existing results in place. Before reusing them, check that they
represent the basin, forcing and model you intend to assess.

## Run one workflow

The simplest approach is to enable only the required workflow in `project_config.yml` and use
the command above. Missing prerequisites are not automatically supplied by disabled workflows.

For historical climate analysis, model building or projection analysis, you can also select a
workflow directly. This command previews model building:

```powershell
pixi run python scripts/run_workflow.py build_model --config $config --project-dir $project_dir --cores 3 --dry-run
```

Use your configured output directory for `--project-dir`; this option is required by `run_workflow.py`. Remove
`--dry-run` to execute. Replace `build_model` with `analyze_climate` or `analyze_projections` for those
analyses.

For scenario generation, use the all-workflow command with only `generate_scenarios` enabled. For hydrological
simulations, enable only `simulate_system`, or use the dedicated command below. These commands handle scenario
and experiment preparation; do not substitute a bare command for the workflow engine (Snakemake).

To preview the dedicated simulation command, use:

```powershell
pixi run python scripts/simulate_system.py --config $config --cores 3 --dry-run
```

Remove `--dry-run` to execute the configured `steps`. Each run command's `--help` lists supported
options; not every Snakemake option can be forwarded.

**When it finishes:** open the linked `summary.md` to check outcomes, warnings, detailed logs and available
timing reports. See [Reading outputs](outputs.md) for result checks and
[runtime and resource use](../technical-manual/runtime-and-resources.md) for interpreting benchmarks.

Each launch writes `logs/<YYYY-MM-DD_HH-MM-SSZ>__<scope>[__dry-run]/`. Executed runs are kept, failed ones
included; only the latest preview of each scope is kept. To list old log folders, run
`pixi run python dev/scripts/prune_logs.py --project-dir <project_dir> --older-than 30`; add `--delete` to
remove them. Folders that saved projection results refer to are never listed.

## Select, resume or create an experiment

An [experiment](glossary.md#experiment) groups hydrological simulations made with a particular model, set of
climate scenarios and scientific simulation settings. Its results are stored under `experiments/<name>/`.

**First experiment:** leave `experiment_name` unset in `project_config_simulate_system.yml`;
the toolbox starts with `stress_test_01`. Running the same command again selects the highest
experiment number already assigned and reuses completed work when its inputs still match.

**Interrupted run:** keep its configuration and rerun the command. Use the
[interruption and lock guidance](troubleshooting.md#a-project-is-locked-or-a-run-was-interrupted) if the run
command reports incomplete jobs or a stale lock.

**Changed inputs:** start a new experiment when you change the model, climate scenarios or scientific
simulation settings, so previous results remain available:

```powershell
pixi run python scripts/run_workflows.py --config $config --cores 3 --new-experiment --note "revised perturbation grid"
```

The run command refuses incompatible changes to an existing experiment. Forcing jobs to rerun does not resolve
that mismatch. Changing indicator choices or resource settings alone does not require a new scientific
experiment.

To use a different naming family, set `experiment_name: drought`; its experiments are numbered
`drought_01`, `drought_02`, and so on. An exact numbered name selects that experiment and cannot
be combined with `--new-experiment`. A new experiment can reuse previously generated climate scenarios
if they are complete and unchanged; it does not necessarily regenerate weather.

Keep `include_unperturbed: true` when your indicators need an
[unperturbed reference](glossary.md#unperturbed-reference) simulation. Setting it to `false` omits those
simulations, and changing this choice requires a new experiment. All generated climate scenarios remain
available.

For an interrupted experiment number allocation, consult the
[execution reference](../technical-manual/execution-and-provenance.md) before using `--recover-reservation`.
That option is not a repair tool for partial weather generation or damaged model-response files.

## Use a model prepared outside the toolbox

Build the default model `wflow` first. To use a model you changed elsewhere, for example by manual
calibration, give it its own folder beside the default and register it:

1. Copy the files you changed (for example `staticmaps.nc` and `wflow_sbm.toml`) into
   `models/hydrology/wflow_calibrated/` in the output directory. Start from the toolbox-built model so the
   gauges and output declarations stay the same.
2. Register the folder. `--copy-from-parent` copies the parent's other model inputs, never its runs:

   ```powershell
   pixi run python scripts/register_model.py register --project-dir <project_dir> `
       --name wflow_calibrated --parent wflow --copy-from-parent --notes "Manual calibration"
   ```

3. Select it in the project file and run `build_model`. For a
   [registered model](glossary.md#registered-model), the toolbox runs the historical simulations and
   evaluation without rebuilding or changing the model files:

   ```yaml
   model:
     name: wflow_calibrated
   ```

4. Run `simulate_system` for a new experiment (`--new-experiment` if earlier experiments used another
   model). The experiment records the model it used.

When the project holds more than one model, `simulate_system` needs `model.name` and lists the models
otherwise. A registered model whose files change afterwards is refused: register the edited version
under a new name. `scripts/register_model.py list` shows each model and whether it is unchanged;
`compare` compares historical runs of different models in `models/hydrology/comparisons/<name>/`:

```powershell
pixi run python scripts/register_model.py compare --project-dir <project_dir> `
    --run wflow/default --run wflow_calibrated/default --name calibration_effect
```

`prune` reports a model's size and the experiments using it, and deletes it only with `--delete` when
no experiment uses it. See [one model per folder](../technical-manual/workflow-build-model.md#model-versions).

## Calculate different indicators from existing simulations

You can calculate additional indicators without rerunning the hydrological model if the completed
experiment saved all required model variables. In `project_config_simulate_system.yml`, select the
existing experiment and request indicators only:

```yaml
experiment_name: stress_test_01
steps:
  simulations: false
indicators:
  sets: [default]
```

Replace the experiment name and indicator selection with your own, then run:

```powershell
pixi run python scripts/simulate_system.py --config $config --cores 3
```

The command reads completed model responses and writes each selected set to
`experiments/<name>/results/indicator_sets/<set-id>/`. No explicit target is needed. You can also use
`run_workflows.py` with the other workflows disabled.

The completed simulation record must exist, even for a preview. Indicators-only execution cannot
create missing model variables or complete unfinished simulations. For a new simulation, restore
`steps.simulations: true`. See [Choose indicators](new-case.md#choose-indicators) for input requirements.

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

## Next

- [Read the outputs](outputs.md) to connect indicators with their climate conditions.
- [Troubleshoot a failed run](troubleshooting.md) using its summary and logs.
