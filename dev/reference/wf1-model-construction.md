# Model construction and historical simulation

Last updated: 2026-10-20

Current development contract for `build_model.smk` (WF1). Use this when changing model configuration,
build ordering or consumers of the mutable Wflow model. [The implementation map](toolbox-workflow-map.md) points to
rule definitions; [the user guide](../../docs/site/toolbox-reference/workflow-build-model.qmd) owns execution.

## Configuration ownership

The project file supplies `project`, `basin`, `climate` and `model` settings; WF1-specific settings are
composed from its `config_path` file into `workflows.build_model`.

- `project` owns the output root and data catalogs.
- `basin` owns region, resolution, controls and shared spatial settings.
- `climate.window` bounds the historical record and `climate.selected` selects forcing.
- `model.outvars` is shared with WF4 and belongs in the project file.
- WF1 owns its engine build/waterbody configs, observations and simulation window.

Use the closed schemas and resolvers in `blueearth_cst/shared/config_composition.py` and
`blueearth_cst/spatial/config.py` for exact admitted fields/defaults. `resolve_simulation_window` validates
the simulation period against the climate record. Do not copy an old `shared.*` configuration shape from
historical specifications. The current scaffolds are under `config/templates/`.

## Spatial and model ownership

The [spatial foundation](toolbox-spatial-foundation.md) owns model-neutral region and vector products.
WF1 adds thematic land/soil products and builds Wflow under `models/hydrology/wflow`.
Do not introduce a model-root dependency into the shared region or climate-extraction producers.

Use HydroMT/hydromt_wflow/Wflow names and setup methods directly. The build template
`config/defaults/wflow_build_model.yml` owns the constant-parameter values and CSDMS spellings.
The `setup_constant_pars` values land as `input.static.<name>.value` TOML scalars; do not assume they are
maps in `staticmaps.nc`. Preserve explicit reference choices until their scientific effect has been
assessed; template values are not evidence of local calibration.

The historical landing/precedence evidence is under `dev/records/validation/wflow-constant-restoration/`.
Inspect the resulting TOML and active upstream reader when changing those bindings; the old verification does
not certify a new build. The historical choice and its aggregate-sensitivity limits are summarized in
0001-restore-wflow-constant-parameters.md (private-checkout-only citation:
`../records/decision-history/0001-restore-wflow-constant-parameters.md`).

## Mutable model-root ordering

HydroMT build/update steps rewrite a model directory in place. Declaring a file written by the initial
build does not order a reader after later mutations. Downstream readers that require the completed model
must depend on the terminal `.model_final` sentinel produced by the final model-writing step.
Intermediate build/update steps instead depend on their preceding stage; making them depend on the
terminal sentinel would create a cycle.

`ancient()` suppresses timestamp-triggered reruns; it does not remove the dependency edge. If a later
rule begins mutating the model root, move the terminal completion point and update affected readers.
Keep this ordering concern separate from whether an artifact should trigger a rerun.

Waterbody setup, output declaration and forcing preparation retain separate failure/retry boundaries.
Avoid merging those steps based on one small-basin timing observation.

## Outputs and consumers

The persisted spatial registry owns stable location identifiers; `staticgeoms/outlet_index.csv` maps
model outlet positions to those identities. Positional `wflow_1` labels are model-output conveniences,
not replacements for basin/subbasin/location IDs.

`blueearth_cst/model/setup_gauges_and_outputs.py` and the shared Wflow output mappings own variable
translation and gauge configuration. Use [the indicator glossary](toolbox-indicator-glossary.md) to navigate
semantic names, CSDMS names, CSV codes and tokens; do not maintain another translation table here.

Observation admission checks registry membership and duplicate IDs before HydroMT reads the table.
The baseline discharge reader uses the outlet index to select its deterministic outlet when additional
gauges produce multiple discharge columns.

Historical climate diagnostics use their shared model-free producers. Model evaluation consumes model
responses and optional observations; do not restore a duplicate subcatchment climate-plotting pipeline.
WF4 consumes the validated model through its own runner and retained-response contract.

## Verification entry points

Use `tests/test_spatial_units_rule.py`, `tests/test_cross_workflow_inputs.py`,
`tests/test_check_baseline_discharge.py`, and the model/config tests for the affected contract.
The [validation ladder](dev-validation.md) selects DAG, consumer and numerical checks when behavior
changes; the old restoration measurements do not validate a newly changed build.
