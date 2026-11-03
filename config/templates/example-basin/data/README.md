# Case data

Keep basin-specific sidecars here and record their provenance beside them.

1. Replace the example row in `output_locations.csv` with your intended location.
   Columns `station_name`, `x` and `y` use longitude/latitude in decimal degrees.
   Keep coordinates consistent with `basin.region` and inspect resolved locations.
2. If supplying observations, use the matching toolbox checkout's
   `docs/technical-manual/input-formats.md` for units, date/time format and resolved station IDs.
   Configure the series in `project_config_build_model.yml`; do not infer/convert units silently.
3. Replace the illustrative `time;1010;1020` header in `observed_daily_discharge_template.csv` with the
   resolved `wflow_id` columns and add real daily data before selecting it.
   The starter leaves observations disabled (`river discharge: null`); the header-only file is not data.
4. For each sidecar, record source, preparation/retrieval date, transformations and whether values are
   measured, derived or synthetic. Label synthetic examples and replace them before scientific interpretation.

The scaffolded case's `INSTRUCTIONS.md` links the version-identifiable local toolbox documentation.
For a manual copy, record the toolbox revision and use that checkout's guide, not repository-relative
links inherited from the original starter location.
