---
title: "Input formats"
format: html
---

This page describes files supplied by the user, data prepared by the toolbox, and conventions for
reporting locations, observations, variables, units, dates and spatial grids.

## Files supplied by the user {#files-you-supply}

| Input | Format | Where it is selected |
|---|---|---|
| Project and workflow settings | YAML | Project configuration and its linked workflow files |
| Reporting locations | Comma-separated CSV | `basin.output_locations` |
| Discharge observations, when available | Semicolon-separated CSV | Model-building `observations` settings |
| Source datasets | Raster, vector, NetCDF or Zarr | Data catalogs and source selections |

HydroMT reads source rasters and vectors, and xarray handles gridded arrays. NetCDF carries climate
data between Python, R/weathergenr and Julia/Wflow. [Data catalogs](data-catalogs-and-access.md)
describe how the source files are read.

## Files prepared or read by the toolbox {#files-prepared-or-read-during-processing}

| File or input | Format and purpose |
|---|---|
| `data/spatial/geoms/region.geojson` | GeoJSON; shared extraction region |
| `data/spatial/spatial_maps.nc` | NetCDF; prepared model-grid spatial inputs |
| `data/spatial/location_registry.csv` | CSV; resolved reporting locations and identifiers |
| `extract_historical.nc` | NetCDF; daily climate with `time`, `latitude`, `longitude` dimensions |
| `basin_cells.csv` | CSV; climate-grid cells intersecting the basin |
| CMIP6 source stores | Zarr; monthly fields read remotely with HydroMT/xarray and GCS access |

The two historical-climate files reside together under
`data/climate/historical/<source>_<window>/`. The extractor keeps a rectangular extent with one
source-grid cell of padding on each side for forcing reprojection. `basin_cells.csv` selects the
intersecting cells used by the generator; the padding does not enlarge the basin.

## Reporting locations

`basin.output_locations` identifies a comma-separated CSV. Coordinates use EPSG:4326, in decimal degrees.

| Column | Requirement | Meaning |
|---|---|---|
| `station_name` | Required | Display name for the location |
| `x` | Required | Longitude |
| `y` | Required | Latitude |
| `location_role` | Optional | `control` defines subbasins (default); `observation` is reporting only |
| `wflow_id` | Optional | Must match the identifier resolved during preparation |

The starter's location file uses commas between fields:

```csv
station_name,x,y
outlet,9.6625,0.445833
```

**Prepared locations:** `data/spatial/location_registry.csv` records locations snapped downstream to
river cells within `basin.delineation.snap_tolerance_m`, together with their resolved `wflow_id` values.
The registry requires review after changes to the basin or locations. Observation columns use these
identifiers, not the display names.

Saved river discharge represents a gauge or outlet; the other configured model outputs are subcatchment means.
The [location-matching reference](input-formats.md#output-locations) explains drainage constraints,
identifier assignment and the checks applied after a spatial rebuild.

## Discharge observations

Historical model evaluation accepts discharge observations in a **semicolon-separated** CSV:

| Item | Required convention |
|---|---|
| First column | `time`, with ISO-8601 timestamps |
| Remaining column names | Integer `wflow_id` values from the location registry |
| Discharge values | m³/s |
| Missing observations | Empty fields |

This illustrative file uses semicolons between fields. Columns `101` and `102` represent resolved
`wflow_id` values; actual files must use the identifiers in their location registry. Values are in m³/s,
and the empty field on the second day represents a missing observation.

```csv
time;101;102
2000-01-01;12.4;8.1
2000-01-02;13.0;
2000-01-03;11.8;7.6
```

The validator rejects duplicate, unknown or missing required station columns. A header-only starter
file supplies the layout but contains no observations for evaluation.
The [observation-matching reference](input-formats.md#observed-time-series) describes required station
coverage and which measurements enter the comparison with simulated discharge.

## Variables, units and missing values

The historical-climate adapter provides the following canonical variables to the processing chain:

| Quantity | Variable | Units |
|---|---|---|
| Daily precipitation | `precip` | mm/day |
| Mean, minimum and maximum daily air temperature | `temp`, `temp_min`, `temp_max` | °C |
| Mean sea-level pressure | `press_msl` | hPa |
| Incoming surface and top-of-atmosphere shortwave radiation | `kin`, `kout` | W/m² |

**Unit conversion:** ERA5 conversions include kelvin to °C and pascals to hPa. Radiation conversions
depend on how the daily archive was prepared; changing a NetCDF `units` attribute does not convert values.
The catalog-preparation step checks the adapter arithmetic before accepting it for forcing.

**Missing values:** missing precipitation is distinct from a dry day, and missing discharge from zero flow.
Missing values must remain explicit, with coverage assessed after extraction. Diagnostics and indicators
apply different completeness rules, so a source accepted for a climate summary may still be unsuitable
for a calculation requiring uninterrupted daily values.

## Calendars, water years and time labels

| Data or calculation | Time convention |
|---|---|
| Historical extraction | Calendar-year bounds |
| Annual reporting | Configured water-year start, which may differ from January |
| Indicators with fixed seasons | Periods defined by the indicator set |
| Generated weather | No-leap calendar; time representation is converted for Wflow and the operations are recorded |
| Saved daily model responses | End-of-interval labels |
| Climate projections | Month lengths from each source model's calendar, including 360-day and no-leap calendars |

**Temporal alignment:** combining series requires consistent leap days, interval labels and included periods.
Matching start
and end years do not guarantee matching dates. The water-year setting changes annual reporting, not
historical extraction bounds. See [indicator periods](indicator-calculations.md#definitions-units-and-aggregation)
for set-specific definitions.

Projection analysis rejects unknown or unsupported calendars for weighted aggregation rather than
assuming Gregorian month lengths, avoiding calendar-dependent errors in climate-change comparisons.

**Daily continuity:** response readers require a regular time axis and explicit missing values.
A missing day must remain on that axis as NaN rather than being removed from the series.

## Spatial references, grids and output locations

For spatial references, grids, reprojection and model-input preparation, refer to the
[HydroMT manual](https://deltares.github.io/hydromt/stable/) and the
[HydroMT-Wflow manual](https://deltares.github.io/hydromt_wflow/latest/).
