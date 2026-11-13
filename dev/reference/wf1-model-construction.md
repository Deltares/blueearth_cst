# Model construction and historical simulation

Last updated: 2026-11-05

Use this reference when changing WF1 model configuration, Wflow build order or consumers of the
mutable model directory. The [spatial foundation](toolbox-spatial-foundation.md) owns model-free
products; the [workflow map](toolbox-workflow-map.md) locates WF1 rules. The
[workflow guide](../../docs/site/toolbox-reference/workflow-build-model.qmd) covers execution.

## Configuration ownership

The project YAML owns the output root and catalogs (`project`), region and controls (`basin`),
historical record and forcing choice (`climate`), and shared output variables (`model.outvars`).
WF1-specific build and waterbody configs, observations and simulation window come from its
`config_path` file into `workflows.build_model`. `climate.window` bounds the historical record;
`climate.selected` selects forcing. The simulation window must fit that record.

[`config_composition.py`](../../blueearth_cst/shared/config_composition.py) and
[`spatial/config.py`](../../blueearth_cst/spatial/config.py) own exact admitted fields and
composition. [`resolve_simulation_window`](../../blueearth_cst/shared/snake_utils.py) validates the
period. Use the tracked [configuration starter](../../config/templates/example-basin) when changing
the supported shape; do not restore a superseded `shared.*` section.

## Spatial and model construction

WF1 consumes the shared region and vector products, adds thematic land and soil rasters, and builds
Wflow under `models/hydrology/wflow`. Region and climate-extraction producers must remain independent
of the model root. Use HydroMT, hydromt_wflow and Wflow names, catalogs and `setup_*` methods directly.

The [build template](../../config/defaults/wflow_build_model.yml) owns constant-parameter values and
CSDMS spellings. `setup_constant_pars` writes these as `input.static.<name>.value` scalars in the
Wflow TOML; they are not maps in `staticmaps.nc`. Several explicit reference choices differ from
engine defaults, and template values do not imply local calibration. Check the resulting TOML and
the active upstream reader before changing a binding; preserve the scientific choice until its
effect is assessed. [`test_build_wflow_model.py`](../../tests/test_build_wflow_model.py) covers the
configured setup calls, but a new build still needs checks suited to the changed behavior.

## Mutable model-root ordering

HydroMT build and update steps write into one model directory. A file produced by the initial build
does not prove later mutations finished. Readers that need the completed model must depend on the
terminal `_engine/markers/.model_final` sentinel from the last model-writing step. Intermediate build and update
steps depend on their preceding stage; making them depend on the terminal sentinel would create a
cycle.

`ancient()` suppresses timestamp-triggered reruns but keeps the dependency edge. If a later rule
starts mutating the model root, move the terminal completion point and update affected readers.
Waterbody setup, output declaration and forcing preparation remain separate failure and retry
boundaries. Keep ordering distinct from the decision about whether an artifact should trigger a
rerun.

## Outputs and consumers

The spatial registry owns stable location identities. WF1 writes
`staticgeoms/outlet_index.csv` to connect model outlet positions to primary registry locations.
Positional `wflow_1` labels are conveniences for model outputs, not substitutes for basin, subbasin
or location IDs. [`write_outlet_index.py`](../../blueearth_cst/model/write_outlet_index.py) owns the
crosswalk.

[`setup_gauges_and_outputs.py`](../../blueearth_cst/model/setup_gauges_and_outputs.py) checks that
registry `wflow_id` values match the model gauge map and declares output columns. The shared
[`wflow_outputs.py`](../../blueearth_cst/shared/wflow_outputs.py) owns variable mappings; the
[indicator glossary](toolbox-indicator-glossary.md) connects semantic names, CSDMS names, CSV codes
and tokens. Do not maintain another translation table here.

When observations are configured,
[`observation_validation.py`](../../blueearth_cst/model/observation_validation.py) checks station
headers against the registry before HydroMT reads the table. User control and observation locations
need one series each; automatic outlets may have a series but do not require one. Duplicate or
unrecognized IDs fail. The baseline discharge reader uses the outlet index to select a deterministic
outlet when the model has multiple discharge columns.

WF1 converts temporary `run_default/output.csv` into per-variable rounded response tables and
`run_default/_engine/response_tables.json`, then removes the native CSV. Model evaluation reads these
responses and optional observations; historical climate diagnostics use the shared model-free
producers. Do not add a second subcatchment climate-plotting path under model evaluation. WF4
consumes the validated model through its own runner and retained-response contract.

## Verification entry points

Use [`test_spatial_units_rule.py`](../../tests/test_spatial_units_rule.py) and
[`test_cross_workflow_inputs.py`](../../tests/test_cross_workflow_inputs.py) for dependency boundaries;
[`test_check_baseline_discharge.py`](../../tests/test_check_baseline_discharge.py) for outlet selection;
and the model, observation and configuration tests for the changed contract. The
[validation ladder](dev-validation.md) selects DAG, consumer and numerical checks when behavior
changes. A passing build alone does not establish local calibration or predictive skill.
