---
title: "Projected PET, dew point and wind"
format: html
---

## PET {#pet}

List `pet` in `variables` and set `variable_options.pet: {method: makkink}` to estimate PET; use
`method: debruin` for de Bruin. PET adds analysis tables but no figures.

### Selecting PET

The method is explicit and applies to every model, member and period; omitting `pet` preserves the existing
output selection. A full supplied-variable declaration with `source`, `canonical`, `units` and `change`
remains supported and does not calculate PET. Bare `pet:` and mixing `method` with `source` are rejected.

### Required inputs

Both methods require matching monthly CMIP6 `tas` (K), `ps` (Pa) and `rsds` (W m-2); de Bruin additionally
requires `rsdt` (incident TOA shortwave radiation, W m-2). HydroMT catalog adapters convert temperature
to °C and surface pressure to hPa after source-unit and monthly-mean metadata admission. Sea-level
pressure (`psl`) is not substituted. No elevation, temperature or pressure correction is applied.

Inputs must agree in model, member, experiment, calendar, grid and monthly coordinates. Missing inputs
for a resolved member fail with the affected source; they do not silently reduce the PET ensemble or
select a different method. Deliberately revise the ensemble settings or supply matching inputs to recover.
The catalog currently certifies precipitation and temperature only. PET mechanism checks on controlled
fixtures do not qualify real CMIP6 dependency availability, transport or basin applicability.

### Calculation

HydroMT's `pet_makkink` or `pet_debruin` calculates an estimated mean daily rate for each native cell
from monthly mean meteorology, before spatial averaging. This approximation is not the monthly mean
of daily-derived PET: within-month nonlinear effects and meteorological covariances are unavailable.
The estimates describe atmospheric evaporative demand, not actual evapotranspiration or a full
aerodynamic/vegetation response. See the [HydroMT implementation](https://github.com/Deltares/hydromt/blob/v1.3.1/hydromt/model/processes/meteo.py).

### Spatial averaging

PET, precipitation and temperature use the same buffered bounding box with spherical cell-area weights
from native coordinate midpoint edges, including irregular rectilinear grids. Cells outside the basin
polygon but inside that box contribute. This is a spatial approximation, not a polygon integral; the
`basin` location token in figure names does not imply polygon weighting. Every PET dependency must be finite
across the entire selected box and all months. Missing inputs cannot become zero PET or change the represented
area.

### Change reporting

- Calculated PET has fixed `canonical: rate` and `units: mm/day`.
- Its default `change: relative` reports percent changes and uses `min_denominator: 0.1` mm/day, a reporting
  safeguard rather than a physical calibration. A finite positive override is accepted.
- Rates strictly below that threshold suppress relative changes while keeping absolute differences; equality
  is admitted. The same daily-rate threshold screens monthly, seasonal and annual references, never an
  unconverted period total.
- With `change: absolute`, omit `min_denominator`.
- Annual and seasonal diagnostic changes integrate the rates with native-calendar day counts and report mm;
  monthly changes keep mm/day.

### Saved products

PET's method (Makkink or de Bruin) and the approximation from monthly mean inputs are recorded in
`_engine/analysis_record.json`. The figure suite covers precipitation and temperature only. It remains an
analytical output: precipitation–temperature clouds and stress-test axes are unchanged.

| Location | Contents |
|---|---|
| `raw/` | Raw meteorological grids |
| `analysis/gridded_changes/` | Member-resolved annual/monthly diagnostic changes, including calculated PET when configured |
| `scalar/` | Monthly PET series, shared with the other variables |
| `analysis/` | Tidy tables, shared with the other variables |
| Summary CSVs | Existing summary CSVs include PET rows |

Source units, paths, calendar, method/library and support are recorded with the series and in
`_engine/analysis_record.json`.

### Reruns

Changing method refreshes current derived products in place; use separate project roots to keep both analyses.
Threshold-only edits do not reacquire raw inputs. Removing PET retires its previously managed figure paths
while keeping source raw data. The generated catalog's additional adapter mappings can invalidate earlier raw
identities, including P/T-only caches, under the existing catalog-integrity policy; unchanged output values do
not imply unchanged cache keys across toolbox revisions.

## Dew point and wind {#dew-point-and-wind}

In the projection workflow configuration, list `tdew` and/or `wind` in `variables` and set
`tdew: {method: specific_humidity}` or `wind: {method: sfcwind}` under `variable_options`. Omission produces
no corresponding derived result.

### Selecting the variables

These calculated forms accept only `method` and optional `change: absolute`; Celsius percentages and relative
wind changes are unsupported. Full supplied-variable declarations remain readable.

### Dew point

Dew point requires monthly `huss` (specific humidity, mass fraction) and `ps` (surface pressure).
For specific humidity q and pressure p in hPa, vapour pressure is
`e = q*p / (0.622 + 0.378*q)`. With `L = log10(e/6.11)`, dew point is
`Td = 237.3*L/(7.5-L)` in degC, inverting the
[NWS liquid-water relation](https://www.weather.gov/media/epz/wxcalc/vaporPressure.pdf).

This is liquid-water dew point even below freezing, distinct from frost point. Surface pressure
approximates pressure at humidity height. Inverting monthly inputs does not recover the monthly mean
of subdaily dew points. There is no relative-humidity fallback: CMIP6 `hurs` changes its saturation
reference to ice below zero and requires a separately specified conversion.

### Wind speed

- The default explicit wind method `sfcwind` requests `sfcWind`, the monthly mean of wind-speed magnitude.
- Alternatively, `wind: {method: components}` requests `uas` and `vas` and calculates
  `sqrt(u_mean^2 + v_mean^2)`: the magnitude of the monthly mean vector, a different statistic that can be
  smaller due to directional cancellation.
- Methods are never silently substituted across models.
- All wind inputs require documented 10 m height before source harmonisation. No 2 m conversion is applied.
- When separately read variables are merged, scalar height coordinates keep their source names as prefixes
  (for example, `tas_height` and `sfcWind_height`), preserving values and coordinate metadata. Different
  measurement heights do not relax the matching grid, time, calendar or CRS requirements.

See the [CMIP6 Amon definitions](https://raw.githubusercontent.com/PCMDI/cmip6-cmor-tables/main/Tables/CMIP6_Amon.json)
for native units, dimensions and temporal statistics.

### Spatial averaging and support

Both derivations run per native cell/month before spherical area weighting over the entire selected
buffered bounding box, including cells outside the basin polygon. Annual and seasonal results use
duration-weighted state means, with absolute changes in degC or m/s. Required support is fixed and
complete: invalid humidity, nonpositive pressure, negative scalar speed, nonfinite values, incompatible
grids/calendars/source identities or unqualified units/heights fail explicitly. Calm component wind is valid.
No clipping, filling, regridding or member fallback is performed.

### Saved products and records

Dependencies share `raw/`, selected monthly results share `scalar/`, and additional selected-variable rows
share `analysis/` and `summary/`. Method, statistic, height and source metadata remain in scalar attributes
and `_engine/analysis_record.json`; no figures are drawn for these variables. Analysis tables also carry
`method`, `definition` and `height_m`; these fields distinguish the two wind statistics when tables circulate
independently.

Concise captions give quantity, periods and counts; the suite note in `figure_captions.md` and the analysis
record carry methods and approximations. Derived annual/monthly native-grid changes are saved with the spatial
diagnostics; full derived monthly meteorological series remain basin averages in `scalar/`.

### Reruns and qualification

A rerender retires PNGs the previous caption manifest owned and the new one does not (renamed or removed
products and removed horizons); unlisted files are kept. Scientific choices refresh derived identities;
presentation changes do not reacquire inputs. Separate project roots keep alternative analyses. The P/T cloud
and stressor axes are unchanged. Controlled fixtures exercise these methods; matching live input availability
remains unqualified.
