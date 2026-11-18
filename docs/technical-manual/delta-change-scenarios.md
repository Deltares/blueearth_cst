---
title: "Projection-conditioned delta-change scenarios"
format: html
---

The GCM delta method applies monthly changes from projection analysis to the case's historical weather, then
simulates the result in hydrological simulation. It is the projection-conditioned ("top-down") alternative to
the scenario-neutral stochastic stress test and is configured in the same scenario-generation file. Stochastic
generation never depends on projection analysis; only this method reads its gridded changes.

## Configure

In `project_config_generate_scenarios.yml`, replace `method: stochastic` and its `stochastic:` block with:

```yaml
method: delta_change
delta_change:
  future_periods: [mid]           # analyze_projections future_periods labels; default: all
```

The ensemble follows analyze_projections (CMIP6), and an omitted `schema_version` is `delta-change/3`.
Earlier delta-change keys are refused; see
[renamed settings](changes-and-migration.md#renamed-configuration-settings).

The basin starter carries this block, commented, at the end of its generation file. Everything else is
derived from the case:

| Derived | From |
|---|---|
| Members, windows, monthly changes | Projection analysis `analysis/gridded_changes/` (inventory schema ≥ 2) and `_engine/analysis_record.json` |
| Historical record | the case's climate-store extract (`climate` source, window and forcing set), as for stochastic generation |
| Historical period | Projection analysis's effective reference dates, including non-January water years |
| Units | supplied by the toolbox for supported sources (ERA5); extraction labels are not trusted |
| Run IDs | one run per member and window after the historical and neutral controls; capacity = runs + 2 |

Optional settings:

| Key | Meaning |
|---|---|
| `name` | Set folder name. Default `delta-<ensemble>-<window names>`. Never part of identity. |
| `dry_reference` | `{policy: refuse}` (default) or `{policy: cap, max_change: <fraction>}`; see below. |
| `scale_pet` | `true` scales PET by the projection analysis PET ratio; see below. Default `false`. |
| `unit_id_capacity` | Advanced. At least runs + 2; changes run-ID width and identity. |
| `historical_override` | Advanced. `period: {start, end}` replaces the projection-analysis reference years (reported as a mismatch). `path` with `source`, `units` and `units_note` replaces the historical file with a supported source; `units` must list every variable with that source's effective units. |

Unknown keys are refused. `analyze_projections` must use the same `ensemble`. In the owned runner projection
analysis runs before scenario generation; with projection analysis disabled its products must already exist. A
missing historical extract is admitted when historical climate analysis or model building runs earlier in the
same invocation, and refused otherwise with the expected path.

## Generate and simulate

```powershell
python scripts/run_workflows.py --config <case>/project_config.yml -- --dry-run
python scripts/run_workflows.py --config <case>/project_config.yml
```

A dry run computes the factors and reports the set it would publish, or reports it as pending when projection
analysis or the historical extract are produced by the same invocation. Delta generation accepts only the
dry-run flags after `--`; other forwarded options and `--rebuild` are refused. Run an upstream workflow alone
when it needs one.

Hydrological simulation finds the set from the current generation settings, as it does for stochastic sets; a
set produced in the same invocation is bound after scenario generation publishes it. A missing or stale set
stops with "run scenario generation". To pin a set explicitly, use
`scenario_collection: {name: <set or alias>}` or `{id: <first 12 hex characters>}` in the simulation file
(quote all-digit ids). Hydrological simulation checks that its climate source
and named historical forcing match the set's.

Hydrological simulation owns PET preparation, initialization, simulation and indicators. Without `pet`, it
recomputes PET from the adjusted temperature with unchanged radiation and pressure. Each run simulates the
set's historical period, preceded by the usual simulation warm-up; see
[initial conditions](hydrological-simulation.md#initial-conditions-and-warm-up). A cold-start run is not
evidence of adequate spin-up. Both controls are included by default; `include_unperturbed: false` simulates
projection runs only.

## Transformations

### Monthly factors

For calendar month *m* and each historical grid cell, the projection analysis monthly kernel supplies the
temperature offset (future minus reference mean) and the precipitation ratio (future divided by reference
mean); both are mapped to the historical forcing grid by HydroMT nearest neighbour.

The offset is added to mean, minimum and maximum temperature; the ratio multiplies precipitation. Outside the
active historical domain the factors are neutral. Dates, leap days, event order and missing-value masks are
preserved; the method adds no subgrid climate information, never changes rain-day frequency and never creates
rain on dry days.

### Dry reference months

Projection analysis withholds the ratio where the reference monthly mean is below its `min_denominator`. With
`policy: refuse` (default) generation stops, naming member, window and months. With `policy: cap`, the ratio
becomes future / reference limited to `1 ± max_change`; a zero reference or a missing future mean is still
refused. Capped members and months are recorded in the set and in plot captions; the policy and threshold are
part of identity.

### PET

With `scale_pet: true`, daily PET is computed on the historical source grid with the case's PET formula (De
Bruin or Makkink, HydroMT) from mean-sea-level pressure without elevation correction, then multiplied by the
projection analysis monthly PET ratio. Its `variables.pet.method` must equal the case's formula. Members
without a projection-analysis PET ratio are excluded and reported per window.

Every run in the set, controls included, uses this PET, which hydrological simulation takes as provided PET
instead of recomputing it. Because it skips the orographic temperature correction of the simulation's
model-grid PET, the set's historical control can differ slightly from model building's historical PET;
differences between runs carry the projected PET signal only. Offline PET formulas differ in how strongly they
respond to warming; treat the formula choice as part of the scenario.

## Results and interpretation

Each run keeps its model, scenario, member and `window_id` in `scenario_run_lookup.csv`. Members are grouped
only within one model, scenario and window; windows and scenarios are never pooled, and both controls form
their own groups. `group_metadata.csv` uses `scenario_key` (role, model, scenario, window) instead of the
stochastic `st_id`; stochastic response-surface plots do not accept it.

Return levels fitted to pooled member-local annual blocks remain conditional pooled calculations:
transformed members share the historical sequence and add no independent years. Model spread, member spread
and stochastic weather variability are distinct; the method supplies no probabilities or cross-model
reduction. Unchanged radiation and pressure (without `pet`) and the historical event sequence limit
interpretation even when software and neutral-control checks pass.

## Set layout and retention

Each set lives in `scenarios/<name>/` and is listed in `scenarios/collections.csv`:

| Path | Content |
|---|---|
| `series/run_*.nc` | Scenario forcing, self-contained |
| `delta/factors/<run_id>.nc` | Monthly offsets and ratios (and PET ratios) on the historical grid |
| `factor_lookup.csv`, `scenario_run_lookup.csv` | Run ↔ factor file, member and window |
| `config/` | Copies of the authored project and generation YAML (display only) |
| `_engine/` | Marker `delta-scenario-collection/2`, intent and producer snapshot |

The climate-store extract and projection-analysis grids are referenced in place by path and hash; a historical
override file outside the project is copied once into `scenarios/_engine/generation_inputs/`. Identity covers
the resolved settings (without `name`), input hashes, factor values, producer code and numerical package
versions. An identical rerun reuses the set; a renamed rerun adds an alias; new content under a used name gets
`-2`, `-3`. Reading a set checks its factor files and series, never the referenced inputs, so a rebuilt
climate store leaves it readable. Folders without a ready marker are listed as incomplete and never adopted or
removed automatically.

## Plots

```powershell
python scripts/plot_delta_scenarios.py --manifest <set>/_engine/collection.json --output-dir <case>/plots/delta
```

Each projection-analysis window renders into `<output>/<window_id>/`: monthly member comparisons per model and
scenario (grid lines are unweighted active-cell means; shaded ranges are spatial minima/maxima, not confidence
intervals) and monthly basin-map panels per run and variable on one colour scale, using the
[shared spatial-panel preset](../../blueearth_cst/plotting/docs/cst-style.md#spatial-map-panels).
`figure_captions.md` names the window, capped months and unavailable members. Basin geometry comes from
`<project>/models/hydrology/wflow/staticgeoms` (or `--model-root`). `--annual-matrix` adds the projection
analysis annual changes as a model × scenario map matrix (`--member-summary median|mean`); cells with withheld
dry-month ratios cannot be mapped there.

## Migrating from delta-change/1 and /2 {#migrating-from-delta-change1-and-2}

Versions 1 and 2 are read-only. See [delta-change/1 and /2](changes-and-migration.md#delta-change-1-and-2) to
regenerate a set with the current method.
