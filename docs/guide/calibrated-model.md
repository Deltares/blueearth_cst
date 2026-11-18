---
title: "Use a calibrated model"
subtitle: "Run the stress test with a model prepared outside the toolbox"
---

Use these tasks to run the stress test with a model you changed outside the toolbox, for example by manual
calibration. The changed model gets its own folder beside the default model `wflow`, so both stay available
for comparison. Build `wflow` first, as described in [Running the workflows](running.md). In the commands
below, replace `<project_dir>` with your [output directory](glossary.md#output-directory).

## Register the model {#register-the-model}

1. Copy the files you changed, for example `staticmaps.nc` and `wflow_sbm.toml`, into
   `models/hydrology/wflow_calibrated/` in the output directory. Start from the toolbox-built model so the
   gauges and output declarations stay the same.
2. Register the folder. `--copy-from-parent` copies the parent's other model inputs, never its runs:

   ```powershell
   pixi run python scripts/register_model.py register --project-dir <project_dir> `
       --name wflow_calibrated --parent wflow --copy-from-parent --notes "Manual calibration"
   ```

**Check:** `pixi run python scripts/register_model.py list --project-dir <project_dir>` shows each model and
whether it is unchanged since registration.

## Run with it {#run-with-it}

Select the model in the [project file](glossary.md#project-file):

```{.yaml filename="project_config.yml (excerpt)"}
model:
  name: wflow_calibrated
```

Then run `build_model`. For a [registered model](glossary.md#registered-model), the toolbox runs the
historical simulations and evaluation without rebuilding or changing the model files. Next, run
`simulate_system` for a new experiment, adding `--new-experiment` if earlier experiments used another model;
the experiment records the model it used. While the project holds more than one model, keep `model.name` set:
hydrological simulation stops and lists the models otherwise.

To change a registered model later, copy it to a new folder and register that under a new name: the toolbox
refuses a registered model whose files changed after registration.

## Compare with the default {#compare-with-the-default}

Compare the historical runs of the two models:

```powershell
pixi run python scripts/register_model.py compare --project-dir <project_dir> `
    --run wflow/default --run wflow_calibrated/default --name calibration_effect
```

The tables and figures are written to `models/hydrology/comparisons/calibration_effect/`. Each `--run` names
`<model>/<recipe>`, with `default` for a single forcing recipe.

## Remove a model {#remove-a-model}

`prune` reports a model's size and the experiments that use it:

```powershell
pixi run python scripts/register_model.py prune --project-dir <project_dir> --name wflow_calibrated
```

Add `--delete` to remove the model; this is refused while any experiment uses it. See
[one model per folder](../technical-manual/workflow-build-model.md#model-versions) for the rules behind these
commands.
