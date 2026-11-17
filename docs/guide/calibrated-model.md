---
title: "Use a calibrated model"
subtitle: "Run the stress test with a model prepared outside the toolbox"
---

In the commands below, replace `<project_dir>` with your configured
[output directory](glossary.md#output-directory). Run each workflow as described in
[Running the workflows](running.md).

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
