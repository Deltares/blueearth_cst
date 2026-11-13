---
title: "Climate-projection analysis"
format: html
---

Projection analysis places the chosen stress-test range in the context of projected climate changes.
It compares basin-average temperature and precipitation between historical and future periods across
selected models and emissions scenarios. The results provide context for interpretation without
determining the weather scenarios used in the stress test.

The run-settings block lists analysis variables by short name, selected calculation methods and
area-weighted aggregation. It shows up to three climate model names with the total model count.
Ensemble counts use resolved model–member pairs without counting members again for each scenario;
larger ensembles show the member-count range per model. Scenario-dependent membership is flagged.

## Inputs {#inputs}

- **Basin definition.** Geographical extent used to calculate basin-average climate changes.
- **Climate projections.** Historical and future climate data for selected models, scenarios and members;
  see the [dataset reference](climate-projection-data.md) for current support.
- **Projection catalog.** Data-access information for the selected ensemble, currently provided through
  the [CMIP6 catalog](catalog-climate-projections.md).

A hydrological model and historical-forcing extraction are not required.

Missing period coverage is reported for each combination, and usable combinations can continue.
If none are usable, change-factor tables contain headers only; corrupt data and incompatible
calendars remain errors.

Climate-projection analysis excludes CMIP6 `THU/CIESM` completely, across all variables, experiments and members, because its
published precipitation has an invalid scale. Excluding the whole model keeps the ensemble consistent
across variables; it does not establish a defect in temperature. Requested CIESM combinations remain in
coverage and composition records as `source_quality_excluded`, with the reason and counts, but create no
acquisition jobs and contribute to no analysis or change-factor products. Member overrides cannot restore
the model. Existing source caches are preserved; rerunning projection analysis rebuilds affected derived products using
the reduced ensemble. No speculative rescaling is applied. See
[Monerie (2026), section 4](https://doi.org/10.1088/2515-7620/ae661c) for independent evidence of the
precipitation problem. A request containing only excluded models fails with no resolved combinations.

## Configuration {#configuration}

**Project settings** define the basin through `basin.region` and `basin.resolution`.

**Workflow settings** define the projection ensemble and comparison periods. The annotated template is
`config/templates/example-basin/project_config_analyze_projections.yml`.

| Setting | Purpose |
|---|---|
| `catalog`, `ensemble` | Projection catalog and ensemble name, currently `cmip6`. |
| `models`, `scenarios` | Climate models and emissions scenarios. |
| `members.preference` | Preferred ensemble members. |
| `members.selection`, `members.overrides` | Optional member-selection refinements. |
| `reference_window` | Historical reference period in inclusive calendar years. |
| `future_windows` | Named future periods in inclusive calendar years. |
| `variables` | Variables and their change conventions. |
| `stats` | Summary statistics. |

Each projection is compared with its own modeled historical reference, independently of the observed
record used for model building. Temperature changes are additive in °C; precipitation changes are
relative. See [execution and run management](configuration-and-execution.md) for running the workflow.

### Optional projected PET

Add `pet: {method: makkink}` to `variables` in the existing workflow configuration to estimate PET.
Use `method: debruin` for de Bruin. The method is explicit and applies to every model, member and period;
omitting `pet` preserves the existing output selection. A full supplied-variable declaration with
`source`, `canonical`, `units` and `change` remains supported and does not calculate PET. Bare `pet:`
and mixing `method` with `source` are rejected.

Both methods require matching monthly CMIP6 `tas` (K), `ps` (Pa) and `rsds` (W m-2); de Bruin additionally
requires `rsdt` (incident TOA shortwave radiation, W m-2). HydroMT catalog adapters convert temperature
to °C and surface pressure to hPa after source-unit and monthly-mean metadata admission. Sea-level
pressure (`psl`) is not substituted. No elevation, temperature or pressure correction is applied.
Inputs must agree in model, member, experiment, calendar, grid and monthly coordinates. Missing inputs
for a resolved member fail with the affected source; they do not silently reduce the PET ensemble or
select a different method. Deliberately revise the ensemble settings or supply matching inputs to recover.
The catalog currently certifies precipitation and temperature only. PET mechanism checks on controlled
fixtures do not qualify real CMIP6 dependency availability, transport or basin applicability.

HydroMT's `pet_makkink` or `pet_debruin` calculates an estimated mean daily rate for each native cell
from monthly mean meteorology, before spatial averaging. This approximation is not the monthly mean
of daily-derived PET: within-month nonlinear effects and meteorological covariances are unavailable.
The estimates describe atmospheric evaporative demand, not actual evapotranspiration or a full
aerodynamic/vegetation response. See the [HydroMT implementation](https://github.com/Deltares/hydromt/blob/v1.3.1/hydromt/model/processes/meteo.py).

PET, precipitation and temperature use the same buffered bounding box with spherical cell-area weights
from native coordinate midpoint edges, including irregular rectilinear grids. Cells outside the basin
polygon but inside that box contribute. This is a spatial approximation, not a polygon integral;
established `basin_avg` filenames remain unchanged. Every PET dependency must be finite across the
entire selected box and all months. Missing inputs cannot become zero PET or change the represented area.

Calculated PET has fixed `canonical: rate` and `units: mm/day`. Its default `change: relative` reports
percent changes and uses `min_denominator: 0.1` mm/day, a reporting safeguard rather than a physical
calibration. A finite positive override is accepted. Rates strictly below that threshold suppress
relative changes while retaining absolute differences; equality is admitted. The same daily-rate
threshold screens monthly, seasonal and annual references, never an unconverted period total.
With `change: absolute`, omit `min_denominator`. Annual and seasonal diagnostic changes integrate
the rates with native-calendar day counts and report mm; monthly changes retain mm/day.

PET adds annual anomalies (mm/year), monthly climatology (mm/day), and annual/seasonal change figures
(percent by default, mm with absolute change), with method and approximation in captions. It remains
an analytical output: precipitation–temperature clouds and stress-test axes are unchanged.
Raw meteorological grids remain in `raw/`; member-resolved annual/monthly diagnostic changes are
retained in `analysis/gridded_changes/`, including calculated PET when configured. Monthly PET
series share `scalar/`, tidy tables share `analysis/`, and the existing summary CSVs include PET rows.
Source units, paths, calendar, method/library and support are recorded with the series and in
`_engine/analysis_record.json`. Changing method refreshes current derived products in place; use
separate project roots to retain both analyses. Threshold-only edits do not reacquire raw inputs.
Removing PET retires its previously managed figure paths while retaining source raw data.
The generated catalog's additional adapter mappings can invalidate earlier raw identities, including
P/T-only caches, under the existing catalog-integrity policy; unchanged output values do not imply
unchanged cache keys across toolbox revisions.

## Optional projected dew point and wind

In the existing projection workflow configuration, select `tdew: {method: specific_humidity}`
and/or `wind: {method: sfcwind}` under `variables`. Omission produces no corresponding derived result.
These calculated forms accept only `method` and optional `change: absolute`; Celsius percentages
and relative wind changes are unsupported. Full supplied-variable declarations remain readable.

Dew point requires monthly `huss` (specific humidity, mass fraction) and `ps` (surface pressure).
For specific humidity q and pressure p in hPa, vapour pressure is
`e = q*p / (0.622 + 0.378*q)`. With `L = log10(e/6.11)`, dew point is
`Td = 237.3*L/(7.5-L)` in degC, inverting the
[NWS liquid-water relation](https://www.weather.gov/media/epz/wxcalc/vaporPressure.pdf).
This is liquid-water dew point even below freezing, distinct from frost point. Surface pressure
approximates pressure at humidity height. Inverting monthly inputs does not recover the monthly mean
of subdaily dew points. There is no relative-humidity fallback: CMIP6 `hurs` changes its saturation
reference to ice below zero and requires a separately specified conversion.

The default explicit wind method `sfcwind` requests `sfcWind`, the monthly mean of wind-speed
magnitude. Alternatively, `wind: {method: components}` requests `uas` and `vas` and calculates
`sqrt(u_mean^2 + v_mean^2)`: the magnitude of the monthly mean vector, a different statistic that
can be smaller due to directional cancellation. Methods are never silently substituted across models.
All wind inputs require documented 10 m height before source harmonisation. No 2 m conversion is applied.
When separately read variables are merged, scalar height coordinates retain their source names as
prefixes (for example, `tas_height` and `sfcWind_height`), preserving values and coordinate metadata.
Different measurement heights do not relax the matching grid, time, calendar or CRS requirements.
See the [CMIP6 Amon definitions](https://raw.githubusercontent.com/PCMDI/cmip6-cmor-tables/main/Tables/CMIP6_Amon.json)
for native units, dimensions and temporal statistics.

Both derivations run per native cell/month before spherical area weighting over the entire selected
buffered bounding box, including cells outside the basin polygon. Annual and seasonal results use
duration-weighted state means, with absolute changes in degC or m/s. Required support is fixed and
complete: invalid humidity, nonpositive pressure, negative scalar speed, nonfinite values, incompatible
grids/calendars/source identities or unqualified units/heights fail explicitly. Calm component wind is valid.
No clipping, filling, regridding or member fallback is performed.

Dependencies share `raw/`, selected monthly results share `scalar/`, and additional selected-variable
rows share `analysis/` and `summary/`. Method, statistic, height and source metadata remain in scalar
attributes and `_engine/analysis_record.json`. Figures add annual anomalies, monthly climatologies and
annual/seasonal mean differences. Analysis tables also carry `method`, `definition` and `height_m`;
these fields distinguish the two wind statistics when tables circulate independently. Figures use
`cmip6_tdew_*_basin_avg.png` and `cmip6_wind_*_basin_avg.png`.
Standalone captions explain methods, periods, counts and approximations. Derived annual/monthly native-grid
changes are retained with the spatial diagnostics; full derived monthly meteorological series remain
basin averages in `scalar/`. Removing opt-ins retires only their previously managed figures/captions, including
removed horizons. Scientific choices refresh derived identities; presentation changes do not reacquire
inputs. Separate project roots retain alternative analyses. The P/T cloud and stressor axes are unchanged.
Controlled fixtures exercise these methods; matching live input availability remains unqualified.

## Processing sequence {#rule-flow}

![Main scientific processing steps and conditional paths described below.](assets/workflow-wf2.svg){width=100% fig-alt="Scientific processing from basin and climate inputs to the assessment products described in this section."}

The workflow acquires available climate projections, calculates monthly basin averages and compares
historical and future periods. It then prepares diagnostic tables and figures; historical series can
serve several future scenarios. Data availability determines which comparisons can be produced.

## Outputs {#outputs}

Results are saved under `data/climate/projections/<ensemble>/`, relative to `project_dir`.
The main outputs are:

- **Climate data (`raw/` and `scalar/`).** Downloaded NetCDF grids and monthly basin-average series.
- **Change factors (`summary/`).** Annual and monthly changes in
  `<ensemble>_change_factors_annual.csv` and `<ensemble>_change_factors_monthly.csv`.
- **Coverage report (`report.md` and `summary/composition.csv`).** Available, partial and unavailable
  model–scenario–member comparisons; `summary/provenance.json` records their sources.
- **Diagnostic tables (`analysis/`).** Coverage, changes, seasonal cycles, model agreement and
  variability, with effective periods and availability in `_engine/analysis_record.json`.
- **Native-grid diagnostics (`analysis/gridded_changes/`).** Annual and monthly changes for each
  paired model–scenario–member and analysis window. NetCDFs preserve source-grid coordinates;
  `inventory.json` identifies members, sources, windows, methods and unavailable comparisons.
  Relative variables also carry `<name>_monthly_ratio` and the monthly reference and future means
  (`<name>_monthly_reference`, `<name>_monthly_future`). The ratio is missing where the reference mean
  is below the variable's `min_denominator`; the means remain, so dry cells can be reported.
- **Figures (`plots/full_period/` and `plots/period_<start>_<end>/`).** Available figures for the full
  record and analysis windows. `plots/figure_captions.md` and `_engine/plots/figure_captions.json` explain
  produced and unavailable figures; title inventories accompany them.

Figures use the diagnostic tables, whose effective periods may differ from the change-factor tables.
By default, each window also includes annual spatial matrices with model rows and scenario columns, and
twelve-month spatial sheets per model/scenario/variable. Basin-average monthly changes are summarized by
the existing box plots; individual `cmip6_delta_monthly_*.png` curve sheets are no longer produced.
Previously managed copies are removed when their figure directory is regenerated.
Spatial sheets summarize members by cell-wise median and retain
their counts in captions. Every available combination is shown, including ensembles smaller than ten models.
Maps use native GCM cells, clipped to the basin outline for display without interpolation. A small basin may
fall entirely inside one coarse cell. Annual temperature changes use day-weighted means; annual precipitation
changes compare duration integrals, rather than averaging monthly percentages. Monthly precipitation factors
are retained as future/reference ratios; display changes are `100 × (factor − 1)`. Each cell must have all
monthly inputs in both complete analysis windows; dry or incomplete cells remain unavailable.

These are projection diagnostics, independent of stochastic or delta scenario generation and requiring no
scenario collection or Wflow model. Native-grid changes reuse local raw slices and existing WF2 kernels.
Figure rerendering uses retained gridded changes when available, checking recorded member hashes.
For older runs without these products, it reconstructs spatial changes from their local raw slices and
scalar provenance in the review destination; those inputs must still be available.
Temperature and precipitation map ramps follow UNESCO draft report Figure 13, with explicit signed scales
and zero ticks. Maps share a scale across models/scenarios for each variable, window and temporal scope.
Project `plotting.annotation` selects a title, caption or neither. See
[execution records](execution-and-provenance.md#wf2-product-attribution) for production records and
[results transfer](results-transfer.md) for export and import.

### Analysis failures and retries

Analysis checks its declared scalar/raw input sets before calculating tables. It checks record metadata for
JSON serialization before spatial diagnostics, then serializes the complete record before publishing CSVs.
The record is written last. Metadata or spatial-preparation failures therefore stop before new CSVs or the
record are written by the analysis script.

This ordering does not provide an atomic bundle or preserve a previous publication across workflow failure.
Snakemake can remove declared products before execution and during failed-job cleanup; spatial diagnostics
can also be partially written before a failure. Correct the reported problem and rerun the workflow, reusing
compatible retained raw slices and scalar series. Individual remaining analysis files do not establish that
analysis finished successfully.

## Diagnostics and interpretation {#interpretation}

```{=html}
<span id="diagnostic-calculations-and-plot-reading"></span>
```

- **Reference and aggregation.** Temperature change is the future mean minus the reference mean;
  precipitation change is 100 × (future mean / reference mean − 1). Monthly averages account for
  month length in the model calendar; seasonal change diagnostics use precipitation rates, while
  seasonal totals are shown in mm.
- **Coverage.** Analysis retains complete water years within the calendar bounds, labeled by their
  starting year. Seasonal diagnostics require at least 90% of the window's water years, rounded up,
  and at least three complete seasons; each season requires three finite months.
- **Ensemble spread.** Change summaries average members within each model and then weight models
  equally; paired temperature–precipitation plots use one point per model and scenario. Median,
  interquartile and full-range summaries describe differences among models, while annual absolute-value
  plots show member medians and ranges. These ranges are not probabilities of future basin outcomes.
- **Interannual variability.** Standard deviation is calculated after removing a linear trend;
  precipitation CV divides detrended standard deviation by the original mean total. A future/reference
  ratio of one means unchanged variability, and annual member ratios are combined geometrically
  within models. Invalid ratios are excluded and recorded, rather than treated as no change.
- **Historical variability bands.** Grey bands show a 90% range of differences between window means
  expected from historical variability, assuming independent years. Positive serial correlation
  would widen the bands; they do not measure uncertainty in the projected change.
- **Sign agreement.** Agreement reports the fraction of models sharing the sign of the multi-model
  median after averaging members within models. It describes consistency of direction, not magnitude,
  significance or independent evidence, because related models are counted separately.
- **Dry periods and sparse ensembles.** Percentage changes exclude reference precipitation below
  the dry threshold (default 0.1 mm/day); CV also excludes dry future means. Hatching flags periods
  where at least half the models are excluded by this criterion, and affected summary plots omit
  scenario groups with fewer than ten models. Captions report eligible counts and omissions.

See the [processing rule reference](workflow-rule-reference.md#analyze-projections) for individual steps.
