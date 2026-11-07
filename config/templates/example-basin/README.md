# Example basin starter

This is the toolbox-owned starter used by `scripts/create_case.py`.
It contains one project file, five workflow files, case-data placeholders and a blank run record.
External datasets are required. The sparse stress-test grid and small realization count are examples,
not assessment recommendations.

## Copy and configure

Prefer the toolbox's case scaffolder; see [Creating a project](../../../docs/guide/new-case.md).
It produces `INSTRUCTIONS.md` in the external case instead of this README.
For manual setup, copy this entire directory to `applications/<basin>/` in your cases repository.

In either case, review these replacements:

1. In `project_config.yml`, set `project.project_dir` to `../../runs/<basin>/` or another external root.
2. Set `basin.region` to your outlet and review `uparea` in km² and `basin.resolution`.
   The delineation threshold is not measured basin area. Inspect the produced basin geometry.
3. Replace `data/output_locations.csv` with intended locations consistent with that drainage network.
4. Check every selected catalog/source covers the basin and period.
   This starter expects `P:/wflow_global/hydromt`, `rivers: hydro_rivers_lin` and `lulc: globcover`.
5. Review `climate.sources`, choose `climate.forcing.precip_source` after source comparison, and keep at least
   16 years of historical record for weather generation. There is no local model calibration.
6. Set the required historical model `simulation_window` in `project_config_build_model.yml` within
   the extracted record. Review outputs and optional observations before running.
7. Review generator period, realizations, monthly perturbations, grid levels, trajectories and seed.
   Review projection selections/windows independently; projections never drive generation.
8. Start a fresh history in `notes.md`; do not inherit another case's success or adequacy claims.

Schema 3 resolves paths beside their declaring YAML; `toolbox:` selects checkout resources.
Keep the filenames or update their `config_path` references. Do not edit generated output archives.

## Run in stages

The starter enables WF0 alone. Run it, compare available historical sources and choose forcing.
Then disable WF0 before enabling downstream work. WF1 builds the model; WF3 independently generates
climate series. WF2 is optional plausibility context. WF4 requires the model and a ready collection.
Disabling a workflow leaves earlier products in place; check whether they still match your intended inputs.

For optional station rainfall comparisons, prepare the two CSVs described in the
[historical analysis reference](../../../docs/technical-manual/workflow-analyze-climate.md#rain-gauges).
Then enable `rain_gauges` in `project_config_analyze_climate.yml`. The supplied rain-gauge CSVs are
synthetic format examples; replace them before interpreting results. Rain gauges have no forcing role.

Use the [running guide](../../../docs/guide/running.md) from the toolbox checkout for commands,
preview/recovery and experiment selection. Generated cases retain local quick-reference commands.
Use [indicator selection](../../../docs/guide/new-case.md#choose-indicators) before simulation to retain
required variables.

## Files

- `project_config.yml`: shared case settings and workflow enablement.
- `project_config_<workflow>.yml`: one workflow's settings.
- `historical-comparison.yml`: optional retained-run comparison; review both directory paths and run names.
  See [historical hydrology](../../../docs/technical-manual/workflow-build-model.md#compare-retained-historical-runs).
- `project_config_delta.yml`: optional standalone projection-conditioned producer; see the
  [delta-change guide](../../../docs/technical-manual/delta-change-scenarios.md). It is not enabled by a workflow stanza.
- `data/`: case sidecars and their [local instructions](data/README.md).
- `notes.md`: retained findings and evidence pointers.

## Documentation after copying

The relative toolbox links above work in this source directory. After a manual copy, use the same
toolbox checkout's `docs/guide/new-case.md`, `running.md` and `outputs.md`.
Record its version/commit in case notes so the guidance is identifiable offline.
If you move the checkout, update your local references rather than substituting an unversioned online manual.
The [templates index](../README.md) is likewise a toolbox-source link, not a path in the copied case.
