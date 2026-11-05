# Config templates

Copyable starters live here; live engine defaults are in [config/defaults](../defaults/README.md).
The single maintained project starter is [example-basin](example-basin/README.md).
Its YAML comments explain each setting, expected input and optional choices.

- `example-basin/`: project file, five workflow files, case data and notes, used by `scripts/create_case.py`.
- `indicator_set.template.yml`: optional named selection of registered indicators.
- `example-basin/data/output_locations.csv`: example outlet to replace.
- `example-basin/data/observed_daily_discharge_template.csv`: header only, not a usable observed series.

## Start a case

Use [Create your own case](../../docs/guide/new-case.md) or copy the whole starter and follow its README.
The starter uses the P-drive catalog with `hydro_rivers_lin` and `globcover` source names.
Another catalog requires review of source names and coverage. Copying configuration does not supply data.
Keep `config_path` links consistent with the copied filenames.

## Why the template's defaults are what they are

See the [configuration reference](../../docs/technical-manual/configuration.md) for the historical-window
minimum, projection member choices and compute defaults. The generator requires at least 16 years;
the historical model run is selected separately. Example grid/realization counts are not assessment advice.
Use the actual catalog for available models rather than maintaining a second model list here.

`wflow_sbm.reference.toml` is a reference copy, not a model-build input.
HydroMT generates the model's own TOML. Do not treat the reference copy as the complete current schema.

## Observation inputs

1. Replace the example location row with your intended coordinates.
2. Resolve locations and obtain station IDs from `data/spatial/location_registry.csv`.
3. Supply observed series only when available, using the required station IDs, dates and units.
   The comma-separated locations file and semicolon-separated observations have different purposes.
4. Configure optional observations in the build-model file. A selected observed variable must also be
   requested in `model.outvars`; missing observations do not establish model skill.

The [input-format reference](../../docs/technical-manual/input-formats.md) owns the exact CSV schemas,
matching rules, separators and provenance requirements. Keep local data with the case and record sources.
Unmaintained examples in `archive/` are outside the supported starting route.
