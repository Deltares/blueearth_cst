---
title: "Saved results and indicator table structure"
format: html
---

Comparing indicator values requires knowing which model, climate scenarios and calculations produced
them. This reference describes the saved records that identify those inputs, the conditions for reuse
and the columns in scenario and indicator tables. The
[results chapter](results-and-interpretation.md) explains their meaning for an assessment.
Relative paths below start at `project_dir`, unless an experiment folder is named.

## Identifying and reusing saved results {#identities-and-reuse}

| Product | Immutable data and authoritative record |
|---|---|
| Scenario collection | `scenarios/_engine/data/<id>/`; marker `scenarios/_engine/collections/<id>/collection.json` |
| Simulation | `experiments/<name>/_engine/simulation.json`; per-variable `hydrology/wflow/output/<run_id>/` and `_engine/response_inventory.json` inside that experiment |
| Indicator set | Experiment `_engine/indicator_sets/<id>/indicators.json`; immutable tables in its `data/` |
| Named indicator view | Experiment `results/indicator_sets/<set-id>/`; committed by `_engine/indicator_sets/names/<set-id>/current.json` |

Collection identity covers generation settings, resolved seed, scientific sources and environment.
Simulation identity binds selected collection/runs, model and simulation settings. Indicator identity binds
saved model time series, declarations, water year and contributing execution sources/dependencies.
Changing the model, collection or frozen simulation inputs requires a new experiment. Changing a compatible
indicator request can create another immutable set without rerunning Wflow. A missing retained variable
requires a new simulation. Different simulations are never combined within one indicator set.

Ready markers are written after validation. A folder or CSV alone is not a validated result ready for use.
`scenario_collection_v2.read_collection_v2` verifies the marker, collection/revision identity, data root,
lookups and series descriptors. `response_inventory.py` verifies complete rounded-response coverage.
`indicator_artifacts.py` validates saved indicator schemas, hashes, tables and exact result keys.
Do not edit immutable data or marker bytes to make reuse succeed.

## Startup input checks

When hydrological simulation is enabled, `scripts/run_workflows.py` checks the experiment request before
upstream launches and
rechecks inputs before passing inputs to each enabled workflow. Checks include configuration, templates, local
input hashes
and scientific source identity. Catalog context includes parsed YAML and root selection: explicit
`meta.root`, the first existing `meta.roots` candidate, or the catalog directory.
A newly available preferred root can therefore invalidate a request without changing YAML bytes.

The initial input check does not load HydroMT or validate every adapter and unit conversion. Those checks
occur in the
workflow using those inputs. Different catalog roots or scientific sources are rejected rather than rewritten
to
fit a saved experiment. Records and existing results remain available for inspection.

## Scenario exports and lookups

Successful scenario generation generation or reuse mirrors the complete selected collection directly into
`scenarios/`:
`scenario_run_lookup.csv`, `perturbation_lookup.csv`, `series/`, `config/` and `weathergenr/`.
`scenarios/_engine/current.json` records the collection/revision, source marker and exported file hashes.
The immutable data root remains the hydrological simulation input. Generation snapshots are shared at
`scenarios/_engine/generation_inputs/<hash>/`; collections retain recorded references to them.

Exports require additional space. Matching files are retained. Publication updates files individually;
read after completion. Validation failures leave the prior export intact, and ordinary export I/O failures
roll back replaced files. Only previously exported files can be removed during refresh. Unrelated files
remain; untracked filename collisions fail. Dry runs do not refresh exports.

| Lookup | Exact columns and meaning |
|---|---|
| `scenario_run_lookup.csv` | `run_id,evaluate,type,rlz,st_id`; ordered stochastic run rows |
| `perturbation_lookup.csv` | `st_id,month,temp_change,precip_change,precip_variance_change`; monthly settings |

Read identifiers as strings to preserve zero padding. Unperturbed reference rows have an empty `st_id`;
they are not an ordinary design point. `month` runs January–December as 1–12. `temp_change` is an additive
temperature change in degrees Celsius; precipitation mean/variance changes are percentages, with 0 unchanged.
The configured precipitation bounds are multipliers; the lookup converts them to `(factor - 1) * 100`.
These are monthly perturbation settings, not annual response-surface coordinates.

## Optional gridded responses

Enabled grid requests are frozen in simulation settings and the creator configuration archive.
Their `response-inventory/5` records each NetCDF's hash, byte size, variables, units, native dimensions
and time coverage. The same inventory version verifies scalar table records and freezes their selected
decimal precision. Only the current table format is accepted; old native-only inventories require a new
experiment. Existing archived bytes are never rewritten to manufacture compatibility.

Enabled simulation completion requires both CSV and NetCDF for every selected run. Batch retries
retain verified pairs outside declared job outputs, check their hashes and native time coverage,
and recover interrupted publication from those copies. The complete simulation marker is written
after inventory validation; recovery copies are then removed. A missing or corrupted enabled grid
fails readiness, including indicators-only admission. Indicators-only never generates a missing grid.

Changing a frozen output request requires `--new-experiment`. Source/model identity guards still
apply across toolbox revisions; the ability to read old responses does not authorize re-simulation
with changed code or model inputs.

## Indicator tables and dictionary

New sets use `indicator-set/6`. On disk, immutable `data/` and named exports use the same split run/group
table headers below. The validator normalizes their identity column to `run_group_id` internally;
that internal key is not an additional CSV column. Named exports copy the immutable tables and add a
dictionary derived from declarations and coverage evidence.

| File | Columns or contents |
|---|---|
| `<token>_run_indicators.csv` | `indicator,location,run_id,value,status,reason` |
| `<token>_group_indicators.csv` | `indicator,location,group_id,value,status,reason` |
| `group_members.csv` | `group_id,run_id`, including singleton units |
| `group_metadata.csv` | `group_id,st_id,is_reference` |
| `indicators.csv` | Definitions, units, method/parameters, grain, group policy, temporal rules and availability |

Use `indicators.csv` for output units rather than inferring them from `<token>`. Its exact columns are
`named_export.DICTIONARY_COLUMNS`. `availability` is `available`, `partial` or `unavailable` across requested
locations; `unavailable_reason` explains missing inputs. It is distinct from each value row's `status`.

| Row status | Value cell | Reason cell |
|---|---|---|
| `available` | Numerical value | Empty |
| `undefined` | Empty, representing a calculated NaN | Empty |
| `unavailable` | Empty | Location-level reason, e.g. `missing_parameter:thresholds` |

Never turn either empty case into zero. Undefined results retain their method-specific interpretation;
for example, absent zero-flow events can leave ERFA timing undefined. An unexpected calculation error
propagates as an error rather than an undefined result. Rejected return-level fits also remain explicit.

A completely absent response excludes the affected calculation from value tables and records it under
manifest `unavailable_calculations`, for example with `missing_response:pet`. The dictionary retains the
declaration and its unavailable reason. Location-level omissions use `unavailable_results` and the rows
above. A set with no calculable declarations has no variable tables; dictionary and manifest inspection
remain necessary. See [evaluation mechanics](indicator-requests.md#inputs-evaluation-and-recovery).

## Combining indicator tables and identifying reference scenarios {#joins-and-reference-membership}

For a run table, join `run_id` to the selected collection's `scenario_run_lookup.csv`, then join its `st_id`
to `perturbation_lookup.csv`. The second join expands a design point to twelve monthly rows.
Filter/select a month deliberately when a single monthly setting is needed.

For a group table, join `group_id` directly to `group_metadata.csv`, then its `st_id` to the perturbation
lookup. `group_members.csv` explains contributing runs; it is not a shortcut to a unique coordinate.
Reference groups have `is_reference=true` and empty `st_id`. Preserve them separately or with a left join;
an inner coordinate join drops them. Run indicators usually average across members with equal weights and
NaN propagation, while qualified return-level bundles pool member-local blocks. The declaration and local
set README describe the precise aggregation. The [output guide](../guide/outputs.md) provides worked table
joins.

## Saving and checking named indicator sets {#named-publication-and-inspection}

Each named indicator set commits independently. A complete set may publish when another fails; the failed
set keeps its previous publication. The command exits nonzero when a required set fails. Thus named folders
can have different ages. The committed pointer and source manifest establish each view's identity.
Request-equivalent sets share calculation artifacts while retaining their own presentation metadata.

`inspect_indicators.py` checks integrity without importing the current calculation registry.
Numerical rechecking needs matching execution sources, inputs and dependencies. Historical check receipts
are not fresh validation. Hash consistency does not prove authenticity against coordinated artifact edits.
See [inspection and rechecking](indicator-requests.md#integrity-and-numerical-rechecking).

## Response-surface coordinates

When interpreting indicator tables or constructing response surfaces, preserve reference membership,
units and the configured monthly perturbation trajectories. State how those monthly perturbations are
represented on each axis so that readers can relate a plotted response to its climate inputs.

Implementation references: `experiment/generation_plan.py`, `scenario_collection_v2.py`, `scenario_export.py`,
`simulation_record.py`, `response_inventory.py`, `indicator_plan.py`, `indicator_artifacts.py` and
`named_export.py`. Relevant tests include `test_scenario_collection_v2.py`, `test_scenario_export.py`,
`test_response_inventory_v2.py`, `test_named_indicator_sets_workflow.py` and `test_partial_indicators.py`.
