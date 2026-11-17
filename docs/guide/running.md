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
included; only the latest preview of each scope is kept. To remove old log folders, see
[Prune old log folders](advanced-tasks.md#prune-old-log-folders).

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

## Use a calibrated model

To run the stress test with a model you changed outside the toolbox, for example by manual calibration,
register it as a separate model. See [Use a calibrated model](calibrated-model.md).

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

## Other tasks

To regenerate figures, rebuild outputs for testing, transfer climate-projection results or prune old log
folders, see [Advanced tasks](advanced-tasks.md).

## Next

- [Read the outputs](outputs.md) to connect indicators with their climate conditions.
- [Troubleshoot a failed run](troubleshooting.md) using its summary and logs.
