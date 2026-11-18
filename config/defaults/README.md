# Rule-read configs

These files are live engine settings, not case scaffolds. Changing a selected file changes a run.
Point case settings to these supplied files or an intentional edited copy; the copyable project starter
lives in [config/templates](../templates/README.md).

- `wflow_build_model.yml`: initial HydroMT-Wflow build configuration file.
- `wflow_update_waterbodies.yml`: waterbody/glacier update configuration file.
- `weathergenr_config.yml`: weathergenr settings used by generation planning; a project replaces it with
  `stochastic.generator_config_path`.

Changing selected generator settings changes its plan and can require a new collection/experiment.
Use the [configuration reference](../../docs/technical-manual/configuration.md) for selection and
[run records](../../docs/technical-manual/run-records.md) for generated captures.
Edit the source configuration, not the output archive. Shared appearance defaults live in
[config/plotting](../plotting/README.md).
