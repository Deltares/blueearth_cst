# Hydrological-model interface

Last updated: 2026-11-05

Use this reference when changing simulator preparation, replacing the hydrological engine, or changing
retained responses. **HM** stands for **hydrological model**. HM-1 through HM-7, including HM-6a and HM-6b,
are stable interface identifiers. Wflow artifacts
describe the current binding; another engine need not reproduce its file schema or physics.

## Boundary and ownership

[`simulator_adapter.py`](../../blueearth_cst/experiment/simulator_adapter.py) owns current forcing admission,
preparation, and execution.
[`wflow_response_reader.py`](../../blueearth_cst/experiment/wflow_response_reader.py)
maps native output to neutral response series;
[`response_inventory.py`](../../blueearth_cst/experiment/response_inventory.py)
owns their retained inventory. [`native_locations.py`](../../blueearth_cst/shared/native_locations.py)
labels native output columns with the registry's `wflow_id`; response tokens and native units are fixed by
`shared/indicator_tables.py` and `experiment/wflow_response_reader.NATIVE_VARIABLES`. Indicator series
validation lives in `indicators/series.py` and `indicators/contract.py`, retained-table interpretation in
`shared/response_tables.py`. A replacement adapter must prepare and run its own engine and satisfy an
independent response request. [Model construction](wf1-model-construction.md) owns WF1 build and mutable
model roots; [spatial foundation](toolbox-spatial-foundation.md) owns model-neutral regions and IDs. See
[saved results](../../docs/technical-manual/saved-results.md) for publication and reuse.

## HM-1 — static grid

Wflow's `staticmaps.nc` is the model-grid reference. `validate_hm1` in
[`interchange_contracts.py`](../../blueearth_cst/shared/interchange_contracts.py) checks float64 spatial
coordinates, CRS, and only the variable names consumed by this toolbox (`_HM1_REFERENCED`). It does not
validate the full upstream parameter schema, physics, or equality with another dataset's grid. The adapter
must establish that prepared forcing fits this grid.

## HM-2 — prepared forcing

Historical WF1 forcing and per-run WF4 forcing share Wflow's content requirements. `validate_hm2` checks
time and spatial dimensions, float64 spatial coordinates, the required float32 variables in `_HM2_VARS`,
CRS, and `grid_mapping` attributes. The variable names connect to the run TOML's forcing mappings.

`_HM2_UNIT_ATTRS` is checked only when an attribute is present; absence does not prove correct physical
units. [Forcing-unit admission](toolbox-forcing-units.md) covers the historical source. The adapter owns
prepared-value interpretation, calendar conversion, and temporal compatibility. The structural validator
does not compare coordinate arrays with HM-1 or prove daily cadence. WF1 forcing persists; WF4 prepared
forcing is temporary and can be absent after a completed run.

## HM-3 — model geometries and outlet mapping

Wflow's `staticgeoms/` and the shared spatial foundation have different owners. Use the delineated shared
region for model-free extraction and maps, not the rectangular model grid extent. `validate_hm3` checks
supplied region and outlet geometry types, CRS, and the outlet-index column for subcatchment mapping. It
does not prove full location membership or equality of the geometry trees.
[`spatial_geoms_parity.py`](../../blueearth_cst/shared/spatial_geoms_parity.py) checks applicable cross-tree
relationships. The [WF1 output mapping](wf1-model-construction.md#outputs-and-consumers) owns model
location and output declarations.

## HM-4 — run configuration

The adapter prepares Wflow TOML. `validate_hm4` checks consumed fields and forcing mappings, not the full
upstream schema or physics values; the validator owns the exact field list. WF1 requires a final-state
output pointer by default. WF4 removes the unused output-state pointer while keeping initialization
settings; `require_output_state=False` checks that variant. Field presence does not prove that a referenced
file exists or that execution produced it. Preparation code owns path rewrites and calendar settings.

## HM-5 — native response CSV

Wflow's native columns follow the model's output declarations. `validate_hm5` checks only a parsed table's
`time` column and at least one other column; it does not prove daily cadence, numeric validity, or complete
requested coverage. `validate_hm_gauge_column_identity` checks supplied TOML and native-column relationships.

The response reader maps native headers through the shared output vocabulary and an independent response
request. Results cannot define their own expected locations: retained inventories must cover the request.
Native header order and persisted ordinals preserve reference selection. Native scalar CSVs are
temporary conversion inputs. WF1 evaluation/history/Qcrit and WF4 indicators read verified per-variable
tables with explicit metadata and configurable decimal precision (default 5). The shared
`response_tables.py` conversion verifies time, IDs, missing masks and rounding error before raw retirement.
New retained formats are historical-run/2 and response-inventory/5; old native-only publications must be
regenerated, with no compatibility mode. The independent development baseline comparator can inspect
temporary full-precision native output during its explicit `--notemp` verification run; this is not a
retained-response format. The [indicator glossary](toolbox-indicator-glossary.md) owns
variable and token names.

## HM-6a — WF1 final state

WF1 configures a final-state output at the location declared by its model TOML. There is no dedicated
HM-6a state-file validator. HM-4 checks the pointer, not file existence or contents. Output presence,
warm-state usability, and numerical validity require separate evidence when relevant.

## HM-6b — retired WF4 final state

Retired: WF4 writes no per-run final state; `validate_hm6b` remains for older captured states and synthetic
checks. History: the HM-6b row of the interchange-contracts design (private-checkout-only citation:
`dev/records/milestones/p32b/interchange-contracts-design.md`).

## HM-7 — retained indicator sets

The indicator planner derives requested run and group coverage from retained responses and collection
membership. Publication writes complete immutable tables and a ready inventory. Named exports are separate
views of those sets, with committed pointers; see
[saved results](../../docs/technical-manual/saved-results.md).
[`indicator_plan.py`](../../blueearth_cst/experiment/indicator_plan.py) owns set planning, reading, and
publication; [`named_export.py`](../../blueearth_cst/experiment/named_export.py) owns named views and the
dictionary columns (`DICTIONARY_COLUMNS`). The validator normalizes the run/group identity column to
`run_group_id` internally; that key is not an additional CSV column. Named indicator sets are selected in
`indicator_sets.py`, published by `indicator_export.py` and inspected by `indicator_inspection.py`, with
`indicator_registry.py` and `indicator_artifacts.py` (all in `blueearth_cst/experiment/`); tests include
`test_indicator_sets.py`, `test_partial_indicators.py` and `test_indicator_inspection.py`.

`validate_hm7_v2` checks supplied markers, tables, and group membership against independently supplied
response series and collection runs. Its implementation owns accepted schema versions and exact headers.
Header-only tables can be valid for an empty scope when no requested indicator applies; expected keys and
coverage still determine completeness. The older `validate_hm7` rejects the retired interface; its name
does not validate current sets. These checks do not establish scientific adequacy of an indicator or
response series.

## Verification and substitution

The interchange validators inspect parsed objects and return differences. They do not perform file I/O or
run automatically at every boundary. Adapter and retained-reader checks provide additional admission.
Use [`test_interchange_contracts.py`](../../tests/test_interchange_contracts.py),
[`test_wflow_response_reader.py`](../../tests/test_wflow_response_reader.py),
[`test_response_series.py`](../../tests/test_response_series.py),
[`test_response_tables.py`](../../tests/test_response_tables.py),
[`test_native_locations.py`](../../tests/test_native_locations.py),
[`test_response_inventory_v2.py`](../../tests/test_response_inventory_v2.py), and
[`test_hm7_v2.py`](../../tests/test_hm7_v2.py) for affected behavior. A fixture-dependent skip is
unavailable coverage. Replacing an engine requires adapter, response translation, and producer/consumer
checks; Wflow shape checks alone cannot establish compatibility. Apply the
[validation ladder](dev-validation.md) when behavior changes.
