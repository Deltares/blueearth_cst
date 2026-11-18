---
title: "Location matching and daily responses"
format: html
---

This reference explains how reporting locations are matched to the drainage network and how saved
daily model responses enter indicator calculations. [Input formats](data-formats-and-conventions.md)
provides the location and observation CSV layouts, climate variables and calendar conventions.

## Output locations

The [reporting-location format](data-formats-and-conventions.md#reporting-locations) defines the
columns supplied through `basin.output_locations`. The following rules determine where each point
is represented in the model and which identifier links it to observations and results.

Controls define subbasins. Observations are tracked without controlling delineation. Each point must
start in active cells of a declared parent basin. Snapping follows downstream drainage to a river cell
within `basin.delineation.snap_tolerance_m`; it does not cross drainage divides to the nearest river.
A pit or an incompatible parent outlet causes an error. Successful adjustments report station and resolved
coordinates, including a move to a model-cell center.

Preparation writes `data/spatial/location_registry.csv`. IDs follow
`wflow_id = basin_id * 1000 + local_subbasin_number * 10 + m`, with `m=0` for the primary location and
`1`–`9` for additional points. `location_code`, such as `B001-S01-L01`, is a readable label.
If IDs are unknown, omit the pinned column and read the generated registry. A pinned mismatch fails with
the old-to-resolved crosswalk. Reconcile the observation headers after any spatial rebuild.

Native Wflow CSVs remain unchanged; derived table locations use resolved `wflow_id` labels where available.

## Observed time series

The project file's [`observations.river_discharge`](configuration.md#observations) supplies observed
discharge for model evaluation; `river discharge` must be in `model.outvars`.
Locations must exist when observations are supplied. The
[discharge-observation format](data-formats-and-conventions.md#discharge-observations) defines the
delimiter, timestamps, station columns and units. Every user-supplied control and observation location
needs one series; automatic outlets may have one but do not require it.

Empty, duplicate, noninteger and unknown station headers are rejected, as are missing required station
columns. Model evaluation reads observations with `;` as separator and compares finite, nonnegative
concurrent discharge pairs. An absent observation file means no observed performance claim; simulation-only
figures describe simulated behavior.

## Cataloged climate and spatial inputs

Catalogs keep upstream HydroMT schema, driver and adapter semantics. Supply the variables, units, geometry
and dates expected by the consuming build configuration file. Register source names rather than embedding
dataset paths in workflow rules. See [data requirements](data-requirements.md) and the configured build
configuration files. Catalog presence alone does not establish readable data, sufficient coverage or valid
forcing units.

Historical stores keep a rectangular basin extent with one native source-grid cell on each side.
`basin_cells.csv` identifies intersecting cells; equal weights over those cells determine the generator's
resampled dates. The full buffered rectangle remains available for forcing reprojection. This buffer does
not enlarge the modeled basin.

## Saved daily responses {#retained-daily-responses}

`model.outvars` uses these display names; the toolbox fixes their response tokens and native units.

| Display name | Token | Native units |
|---|---|---|
| `river discharge` | `q` | `m3 s-1` |
| `precipitation` | `precip` | `mm dt-1` |
| `potential evapotranspiration` | `pet` | `mm dt-1` |
| `actual evapotranspiration` | `aet` | `mm dt-1` |
| `groundwater recharge` | `gwr` | `mm dt-1` |
| `overland flow` | `overland_flow` | `m3 s-1` |
| `snow` | `snow` | `mm` |
| `temperature` | `temp` | `degC` |

`dt` is the model timestep; daily indicator calculations require 86400 seconds. These are input units,
not universal output units: each calculation declares its own output units in `indicators.csv`.
Discharge is a gauge/outlet series; other configured outputs are subcatchment means. `gwr` is signed net
recharge, and snow is snowpack liquid-water depth. A token alone does not establish agricultural or aquifer
support. The local set READMEs explain method-specific aggregation and limitations.

Indicator calculations accept daily series with explicit run, variable, location and ordinal, daily
time coordinate, calendar, timestep, time label, units, numeric values and a Boolean missing mask.
The mask must equal native NaN positions. Infinity is rejected. Duplicate keys, irregular/omitted dates,
ambiguous location order and incompatible paired series fail before reduction. Missing days belong on the
daily grid as NaNs. Supported calendars are `standard`, `gregorian` and `proleptic_gregorian`.

Hydrological simulation supplies daily interval-end series and handles simulation selection and warm-up.
Indicator calculations do not infer them from filenames or remove additional warm-up.
The [custom-indicator reference](indicator-extension.md) links the module documentation and calculation sets.
