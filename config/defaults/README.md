# Rule-read configs

These files are live engine settings, not case scaffolds. Changing a selected file changes a run.
Point case settings to these supplied files or an intentional edited copy; the copyable project starter
lives in [config/templates](../templates/README.md).

- `wflow_build_model.yml`: initial HydroMT-Wflow build recipe.
- `wflow_update_waterbodies.yml`: waterbody/glacier update recipe.
- `weathergen_config.yml`: weather-generator settings used by generation planning.

Changing selected generator settings changes its plan and can require a new collection/experiment.
Use the [configuration reference](../../docs/technical-manual/configuration.md) for selection and
[execution provenance](../../docs/technical-manual/execution-and-provenance.md) for generated captures.
Edit the source configuration, not the output archive. Shared appearance defaults live in
[config/plotting](../plotting/README.md).
