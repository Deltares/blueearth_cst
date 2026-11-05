# Hydrological-model interface

Last updated: 2026-10-20

Use this reference when changing simulator preparation, replacing the hydrological engine or changing
retained responses. Preserve HM-1 through HM-7, including HM-6a and HM-6b, as stable identifiers.
The Wflow-specific artifacts below describe the current binding, not universal requirements for all models.

## Boundary and ownership

`blueearth_cst/experiment/simulator_adapter.py` owns current forcing admission, preparation and execution.
`wflow_response_reader.py` translates native results into declared response series;
`response_inventory.py` owns their retained inventory.
A replacement must implement preparation and execution appropriate to its engine and expose responses
matching the independent response request. It need not reproduce Wflow physics or every upstream file.

[Model construction](wf1-model-construction.md) owns WF1 build and mutable-root ordering.
[Spatial foundation](toolbox-spatial-foundation.md) owns model-neutral regions and identities.
[Retained handoffs](../../docs/site/toolbox-reference/retained-handoffs.qmd) owns publication paths,
ready inventories, named exports and reuse. Use those owners rather than copying their layouts here.

## HM-1 — static grid

Wflow's `staticmaps.nc` is the current model-grid reference.
`validate_hm1` in `blueearth_cst/shared/interchange_contracts.py` checks float64 spatial coordinates,
CRS and the consumed variable names in `_HM1_REFERENCED`. It does not validate the complete Wflow
parameter schema, physics or equality with another dataset's grid.
The preparation adapter must separately ensure forcing fits the model grid.

## HM-2 — prepared forcing

Historical WF1 forcing and per-run WF4 forcing share the current Wflow content requirements.
`validate_hm2` checks required dimensions and spatial coordinate dtypes, the required forcing variables
in `_HM2_VARS`, float32 variable dtypes, CRS and `grid_mapping` attributes.
These variable names connect to the run TOML's forcing mappings.

Unit attributes are checked against `_HM2_UNIT_ATTRS` only when present. Absence is accepted;
that is not evidence of correct physical units. [Forcing units](toolbox-forcing-units.md) owns source admission,
and the adapter owns prepared-value interpretation, calendar conversion and temporal compatibility.
The structural validator does not compare actual coordinate arrays with HM-1 or prove daily cadence.
WF1 forcing persists; WF4 prepared forcing is temporary and may be unavailable after a completed run.

## HM-3 — model geometries and outlet mapping

Wflow's `staticgeoms/` and the shared spatial foundation have different ownership.
Use the shared region for model-free extraction and maps; do not substitute the model's rectangular
grid extent for the delineated basin.

`validate_hm3` checks supplied region/outlet geometry types and CRS and the outlet-index column used for
subcatchment mapping. It does not establish full location membership or equality between the two trees.
`blueearth_cst/shared/spatial_geoms_parity.py` owns the applicable cross-tree relationships.
The [model-construction reference](wf1-model-construction.md#outputs-and-consumers) owns output mappings.

## HM-4 — run configuration

The current adapter prepares Wflow TOML configuration. `validate_hm4` checks the consumed configuration
fields and forcing mappings, not the full upstream schema or physics parameter values.
Use its implementation for exact required fields rather than a second key list.

WF1 requires a final-state pointer by default. WF4 removes the unused output-state pointer while retaining
initialization settings; `require_output_state=False` checks this variant.
Configuration-field presence does not prove any referenced file exists or that execution produced it.
Preparation code owns actual path rewriting and calendar settings.

## HM-5 — native response CSV

The Wflow binding publishes native columns determined by the model's output declarations.
`validate_hm5` checks only that a parsed table has `time` and at least one other column;
it does not establish daily cadence, numeric validity or complete requested coverage.
`validate_hm_gauge_column_identity` checks supplied TOML/native relationships.

The response reader maps native headers using the shared output vocabulary and an independent response
request. Retained inventories must represent the requested series; result tables cannot define their
own expected locations. Native header order and persisted ordinals preserve reference selection.
WF1's raw simulation CSV and WF4 native responses are retained for later reduction.
WF1 exports, evaluation and baseline checks reuse the full-precision native responses;
older trees where temporary-output cleanup removed them need one simulation to restore them.
The [indicator glossary](toolbox-indicator-glossary.md) owns variable/token terminology.

## HM-6a — WF1 final state

WF1 configures a final-state output whose location follows the model TOML.
There is no dedicated HM-6a state-file validator. HM-4 checks the pointer, not file existence or contents.
Treat output presence, usability as a warm state and numerical validity as separate claims requiring
appropriate evidence when needed.

## HM-6b — retired WF4 final state

The current WF4 binding does not emit an unused final state or chain states across scenarios.
`validate_hm6b` remains for captured older states and synthetic checks. Retaining temporary outputs cannot
recreate an output the adapter does not write. This ID is retained to interpret existing callers and evidence.

## HM-7 — retained indicator sets

The indicator planner determines requested run/group coverage from retained responses and collection
membership; publication writes complete tables and the ready inventory. Named exports are separate
post-processing views of immutable sets. The [retained-handoff guide][handoffs] owns their current paths.

`blueearth_cst/experiment/indicator_plan.py` owns set reading and publication;
`named_export.py` owns named views. `validate_hm7_v2` checks supplied markers, tables and group membership
against independently supplied response series and collection runs. Its implementation owns the accepted
schema versions and exact table headers. Empty scopes can legitimately have header-only tables when no
requested indicator applies; expected keys and coverage determine completeness.

The older `validate_hm7` rejects the retired interface. Its name must not be interpreted as validation of
current sets. Interchange checks do not establish scientific adequacy of an indicator or response series.

[handoffs]: ../../docs/site/toolbox-reference/retained-handoffs.qmd

## Verification and substitution

The interchange validators operate on parsed objects and return differences; they do not perform file I/O
or automatically run at every workflow boundary. Actual admission also depends on adapter and retained-reader
checks. Consult `tests/test_interchange_contracts.py` and the affected adapter, response-reader and publication
tests through the [validation ladder](dev-validation.md). Report fixture-dependent skips as unavailable
coverage, not compatibility passes.

Replacing an engine requires its adapter, native-response translation and producer/consumer checks. Passing
Wflow artifact-shape checks alone cannot establish compatibility of an alternative model. Use the supported
[WF4 guide](../../docs/site/toolbox-reference/workflow-simulate-system.qmd) for execution; P2 acceptance
(private-checkout-only citation: `../records/milestones/r12/implementation/evidence/p2/acceptance.md`) retains
historical preservation evidence with its original scope.
