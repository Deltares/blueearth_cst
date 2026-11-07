---
title: "Projection-conditioned delta-change scenarios"
format: html
---

Use the standalone delta-change producer to apply monthly projection changes to an existing historical
weather sequence, then simulate that collection with WF4. This optional projection-conditioned assessment
is separate from the scenario-neutral stochastic stress test. It does not change WF3's stress grid,
weather generator or projection-independent sampling.

## Inputs and configuration

Start with the optional
[delta configuration](../../config/templates/example-basin/project_config_delta.yml) in the basin starter.
The producer takes its own closed configuration, not a workflow stanza in the project file.
All listed settings are required; unknown settings are refused. Relative input paths resolve beside this YAML.
The initial profile supports `source_like: era5` only and requires the same selected source in WF4.

Supply a georeferenced historical NetCDF with daily time coordinates and `temp`, `temp_min`, `temp_max`
and `precip`. Retain pressure and radiation required by the selected WF4 preparation. Supply one pair of
monthly basin-scalar projection NetCDFs per explicitly listed model/scenario/member. Projection variables
are `temp` and `precip`, with one ordered sample per requested year and month. Singleton model/member
dimensions are allowed; unresolved spatial grids or multiple members in one variable are refused.

`reference_period` and `future_period` are inclusive projection years, common to all members in a collection.
They do not replace the historical timestamps. The template's 30-year windows are examples; the selected
files must cover every month of those periods. Missing months, duplicates, nonfinite samples, negative
precipitation and nonpositive reference monthly precipitation are errors. Temperature units must match
between projection files and be K or degrees Celsius; precipitation units must be explicit and match.
New collections accept projection precipitation in `mm/day` or `mm/month`. Reference monthly means below
the shared WF2 precipitation threshold (currently 0.1 mm/day) are refused, with affected months reported.
For this check, monthly totals are divided by each sample's calendar-specific number of days before
averaging. Equality with the threshold is accepted. Factors are never clipped or silently substituted.
`simulation_window` selects inclusive historical years for WF4, using its existing January 1 through
December 31 midnight convention. Historical forcing must cover that interval. This selection does not
perform or exclude a separate spin-up simulation.

For a shared historical warm start, prepare one historical state before the evaluation interval. Set
`model.cold_start__flag = false` and `state.path_input` in the source model's `wflow_sbm.toml`, retaining
its `state.variables` mapping. WF4 preserves that chosen path when writing each run configuration and
refuses a missing warm-state file or mapping. The model fingerprint covers the selected state bytes.
Use a path inside the model root; if `dir_input` is set, use an absolute state path inside that root so
the current fingerprint and runtime resolve the same file. Cold-start defaults remain unchanged.

Set `simulation_window` to the evaluation years only, with the warm-up performed separately. All
members and independent horizons then start from the same state; no future run supplies another
future run's initialization. Match the state epoch to the configured simulation start, remembering
Wflow uses the first forcing timestamp as the initial state time and begins output one timestep later.
The presence of a state file does not establish sufficient spin-up: compare storage and responses under
alternative warm-up lengths before treating initialization as qualified.

`reader.metadata.cst_unit_interpretation` declares the historical values' **effective physical units**.
Copy the evidenced binding from the historical extraction/preparation provenance, or establish an equivalent
binding for supplied data. Its variables must cover every data variable except `spatial_ref` exactly once.
The template shows the toolbox's extracted ERA5-style variable set; it is not permission to reinterpret
arbitrary raw files. Some historical files retain original unit labels after extraction converted values.
Do not apply those conversions a second time. This producer makes no unit conversion.
Historical effective units must be `degC` for all three temperature variables, `mm/day` for precipitation,
`hPa` for pressure and `W/m2` for radiation. Other declarations for those variables are refused because
WF4's extracted-ERA5 reader expects this binding. Projection temperature inputs may still use K or
degrees Celsius.

`unit_id_capacity` fixes run/group ID width and must accommodate all projection members plus two controls.
For example, 1000 gives `run_0001`. Changing capacity changes collection identity. Member ordering is
explicit and retained; duplicate model/scenario/member identities are refused.

## Generate and simulate

From the toolbox environment:

```powershell
python scripts/generate_delta_scenarios.py --config <case>/project_config_delta.yml --project-dir <run-root>
```

The command prints the ready collection manifest. Put that path in the existing WF4 configuration:

```yaml
scenario_collection:
  manifest_path: <run-root>/scenarios/_engine/collections/<identity>/collection.json
```

Keep the project's model, selected historical source and WF4 preparation consistent with the supplied
forcing interpretation. Then use the existing runner:

```powershell
python scripts/simulate_system.py --config <case>/project_config.yml --dry-run
python scripts/simulate_system.py --config <case>/project_config.yml --cores 2
```

WF4 owns PET preparation, initialization, simulation and indicator selection. The exercised path uses
De Bruin PET recomputed from adjusted temperature and historical radiation/pressure. No new PET algorithm,
spin-up exclusion or state chaining is supplied by the delta producer. Choose and qualify initialization
and evaluation periods for the intended assessment; a cold-start execution is not evidence of adequacy.

The producer always includes historical and neutral-factor controls. WF4's default includes both.
`include_unperturbed: false` selects only projection runs and excludes both controls; indicators requiring
a historical reference then require a new experiment including that control.

## Transformations and results

For calendar month *m*, the temperature offset is the future monthly mean minus the reference monthly mean.
The precipitation ratio is the future monthly mean divided by the reference monthly mean. Monthly samples
receive equal weight within each month. The offset is added to historical mean/minimum/maximum temperature;
the ratio multiplies historical precipitation. The same monthly factors apply to every historical grid cell.
All other variables, historical dates, leap days, event order and missing-value masks are preserved.

Per-member runs retain their model, scenario and member identities in `scenario_run_lookup.csv`.
Members are bundled only within the same model/scenario and collection-wide periods/method. Historical and
neutral controls each have separate bundles, including when the projection bundle has only one member.
Indicator exports retain per-run values and existing bundle reductions. Delta `group_metadata.csv` uses
`scenario_key` (a JSON array of role, model and scenario) instead of stochastic `st_id`; `is_reference`
identifies the historical control. Stochastic response-surface plotting does not accept this metadata.

The existing indicator declarations distinguish equal-weight member reductions from return levels fitted
to pooled member-local annual blocks. The latter remains a conditional pooled calculation: transformed
members share the historical sequence and do not supply independent additional years. Do not use member
count to claim a longer observed record or reduced uncertainty. Model spread, member spread and stochastic
weather variability are distinct quantities; no probabilities or cross-model ensemble reduction are supplied.

Monthly scaling does not independently prescribe future event sequencing or wet-day frequency. Threshold
crossings can change under scaling and rounding. Unchanged radiation/pressure and the historical event
sequence limit interpretation even when software and neutral-equivalence checks pass.

## Retention and recovery

Collections use the dedicated `delta-scenario-collection/1` profile. Existing stochastic collections retain
their profile and validation; experimental pilot collections are not converted in place.
The engine directory contains the frozen intent and ready marker. Its matching data directory contains
input snapshots, the producer source snapshot, scenario lookup and generated NetCDFs. The intent retains
full-precision monthly factors, resolved configuration, source hashes and numerical package versions.
New intents also retain the precipitation admission threshold and units. Older version-1 intents remain
readable under their original positive-denominator policy; reading them does not retroactively qualify
their dry-month factors. Regeneration applies the current policy and creates a new collection identity.
WF4 retains its own simulation and indicator provenance through the normal execution path.

### Optional projection-derived PET

With version 2, add the following optional mapping to use supplied PET change factors:

```yaml
pet:
  mode: projection_ratio
  method: makkink
```

Omitting `pet` keeps the existing De Bruin recomputation. The method string must match the `pet_method`
attribute on the `pet` variable in historical, reference and future NetCDFs. It identifies the upstream
calculation; the delta producer does not calculate PET from projection meteorology. Retain that calculation's
input and method provenance. Historical PET must have evidenced effective units `mm/day` in `reader`;
projection units must match and be `mm/day` or `mm/month`. Reference monthly PET must be strictly positive.
Missing PET, inconsistent methods/units, negative values and nonfinite mapped factors are errors.

Each monthly PET ratio multiplies historical PET once. The historical and neutral controls preserve that
baseline. WF4 prepares temperature with `skip_pet=True` and uses HydroMT's `setup_pet_forcing` for supplied
PET, so it does not additionally recompute PET from adjusted temperature. The preparation identity records
the provided-PET binding and reader, and the collection identity retains PET factors and method provenance.

For sensitivity comparisons, hold the model, shared initial state, evaluation interval, projection member
and temperature/precipitation inputs fixed. Report baseline PET differences when the supplied method differs
from the default De Bruin method: that comparison includes a method change, not only future meteorology.
PET calculated from monthly mean meteorology is an approximation to monthly mean PET because the formula
is nonlinear. Treat pressure/elevation choices and missing projection-member coverage explicitly.

### Gridded factors and member diagnostics

Use `schema_version: delta-change/2` with the required `factor_layout: grid` to retain spatial variation.
Alternatively, `factor_layout: basin_scalar` uses the existing scalar calculation. Version-1 configurations
remain scalar and do not accept this field. The other configuration fields have the same meanings.

Gridded reference and future files must have matching grids and explicit matching CRS, with temperature
and precipitation on monthly rasters. Historical forcing must have explicit CRS and `latitude` and
`longitude` coordinates. Monthly offsets and ratios are calculated on the projection grid, then mapped
to the historical forcing grid using HydroMT nearest-neighbour reprojection. Missing coverage and dry
reference cells in the active historical domain are errors. Outside that domain the factors are neutral
and historical missing values remain missing. This method does not add subgrid climate information.
The intent retains the applied arrays in month/latitude/longitude order, alongside their source inputs.

Plot a completed scalar or gridded collection with:

```powershell
python scripts/plot_delta_scenarios.py --manifest <collection.json> --output-dir <case>/plots/delta
```

The command validates the collection and writes monthly member comparisons, compact monthly basin-map panels
for each gridded member and variable, and `figure_captions.md`. Members are grouped by model and scenario
without pooling. Grid comparison lines are unweighted active-cell means; shaded spatial ranges are not
confidence intervals. Maps share each variable's colour scale across members and months, and use the
standard climate-map renderer with basin boundaries, subbasin divides, rivers and location markers.
Each member/variable figure places January–December in four columns and three rows, with narrow
gutters and one shared colourbar. Coordinate titles/tick labels, location labels and repeated scale bars
are omitted to keep panels compact; location markers remain. Geometry comes from
`<project>/models/hydrology/wflow/staticgeoms`; use
`--model-root <wflow-model>` for a different model location. Basin geometry with an explicit CRS is required
for gridded plots. Missing optional subbasin, river or location layers are reported in captions. The basin
viewport does not interpolate or refine the climate factors. Existing old grid-panel files are not deleted;
`figure_captions.md` lists the current exports.

Identity includes resolved settings, input bytes, producer revision and environment. A repeated identical
completed request verifies retained artifacts before reuse. Changed inputs produce a new identity. A partial
directory without a ready marker is refused; inspect and preserve it before explicitly removing task-owned
incomplete products or choosing a fresh project root. Corrupt retained bytes are errors, never overwritten.
