# Figure naming conventions

Last updated: 2026-11-05

Use this reference when adding a figure family or changing figure filenames. It collects shared naming
rules and workflow profiles in one place. A profile defines the exact filename structure and vocabulary
for its figures; the owning producer implements it.

## Shared rules

- Use lowercase `snake_case` for toolbox-defined filename tokens and the file extension. Preserve
  identifiers according to their owning contract rather than normalizing them independently.
- Name the scientific variable, diagnostic and spatial support consistently. Reuse an existing token
  when its meaning matches; distinguish an area average from a map extent or a station observation.
- Use a stable comparison token for multi-source figures. List participating sources in legends,
  captions and provenance rather than making the filename depend on the number of sources.
- Keep project names, workflow IDs, units and analysis periods in directories or metadata unless a
  profile needs one to distinguish products in the same output directory.
- Keep presentation variants, such as captioned copies, identifiable without changing the scientific
  figure's identity. Each profile specifies supported formats and how variants are stored.
- Treat existing filenames as interfaces: workflow outputs, reports and downstream readers may depend
  on them. Upstream-controlled names can remain explicit exceptions to toolbox naming rules.

## Workflow profiles

The [historical climate profile](#historical-climate-diagnostics-wf0) below documents WF0. Other workflows
retain their current producer-defined filenames until their profiles are documented and checked here.
For example, WF1 forcing figures retain `forcing_<variable>_<kind>.png` under the
[CST plot-style rules](cst-style.md#figure-specific-decisions).
Moving this reference does not rename generated files or extend the WF0 builder to other workflows.

Add further profiles as sections in this document, using the shared rules and the
[extension procedure](#adding-or-changing-a-profile). Reuse the WF0 field order where it fits; document
only the additional distinctions or exceptions a figure family needs. Do not create a separate naming
rulebook for each workflow.

## Historical climate diagnostics (WF0)

The closed vocabulary and filename builder live in
[`climate_analysis/figure_naming.py`](../../climate_analysis/figure_naming.py). This profile applies to
WF0 source and comparison diagnostics; it is not a universal schema for every toolbox figure.

### Filename structure

WF0 figure filenames must use lowercase `snake_case` and follow:

```text
<dataset_scope>_<variable>_<plot_context>_<spatial_scope>.<extension>
```

The fields mean:

- `dataset_scope`: the dataset ID for a single-source figure, such as `era5`
  or `chirps`; use `comparison` for a figure containing multiple datasets.
- `variable`: the canonical, unabridged scientific variable name, such as
  `precip`, `temp`, `pet`, or `precip_temp` for a paired diagnostic.
- `plot_context`: the diagnostic meaning and, where needed, plot form, expressed with
  the controlled vocabulary below.
- `spatial_scope`: the spatial aggregation or coverage represented by the
  figure.

For comparison figures, the individual dataset names must appear in the figure
legend and run provenance, not in the filename. This keeps filenames stable as
the number of compared datasets changes.

### Controlled vocabulary

Use these abbreviations consistently:

| Meaning | Token |
| --- | --- |
| time series | `ts` |
| climatology | `clim` |
| average | `avg` |
| extent | `ext` |
| distribution box plot | `box` |

Canonical WF0 plot-context tokens:

```text
annual_clim_map
monthly_coverage
monthly_clim_band
annual_timing
monthly_anomaly
monthly_spi
spi_events
annual_rx1day_ts
annual_rx5day_ts
annual_sdii_ts
annual_wet_days_ts
dry_spell_exceedance
monthly_anomaly_acf
annual_trend_interval
annual_trend_ts
seasonal_anomaly
monthly_spi_fit
annual_trend_sensitivity
daily_agreement
```

The machine vocabulary and descriptions live in `figure_naming.py`. `annual_ts`, `monthly_box` and
`monthly_clim_line` remain registered for the WF1 source producer; WF0 does not emit them. WF0 emits
PNGs only. Optional captioned copies use the
same filename under `plots/captioned/`; captions never add a fifth field.

Recommended spatial-scope tokens:

```text
basin_avg
basin_ext
source_ext
subbasin_<id>_avg
subbasin_<id>_ext
station_<id>_avg
```

Do not abbreviate the canonical variables `precip`, `temp`, and `pet`. Avoid
`st` for station because `st` already identifies a stress-test member elsewhere
in the project.

### Which `<id>` each scope takes

`subbasin_<id>` and `station_<id>` use distinct identifier namespaces. The formulas are implemented
in [`identity.py`](../../spatial/identity.py):

| Scope | Column | Formula | Basin 1 example |
| --- | --- | --- | --- |
| `subbasin_<id>` | `subbasin_id` | `basin_id*100 + local_subbasin_number` | `101`–`104` |
| `station_<id>` | `wflow_id` | `basin_id*1000 + local_subbasin_number*10 + m` | `1010`, `1020`, … |

`m` is `0` for a subbasin's primary location and `1`–`9` for additional gauges. A subbasin figure
uses `101`; a station figure at its primary location uses `1010`. Their different digits keep a
subbasin area distinct from a model observation point.

Add new contexts or abbreviations to `figure_naming.py` and this reference, rather than spelling
them independently in plotting modules. The
[`figure-naming tests`](../../../tests/test_figure_naming.py) check admitted tokens and examples.
Rain-gauge comparisons emit `monthly_coverage`, `annual_ts`, `monthly_clim_line` and `daily_agreement`
with `station_<token>_avg`. Here the token maps to the literal user-supplied gauge ID in
`rain_gauges/tables/stations.csv`; it is independent of Wflow's `wflow_id` namespace.
The suffix identifies station comparison support; the sampled grid value is its native cell.

### Examples

```text
era5_precip_annual_trend_ts_basin_avg.png
era5_temp_annual_clim_map_basin_ext.png
chirps_precip_monthly_clim_band_basin_avg.png
era5_precip_annual_trend_ts_subbasin_101_avg.png
comparison_precip_annual_rx1day_ts_basin_avg.png
comparison_precip_temp_seasonal_anomaly_basin_avg.png
```

## Adding or changing a profile

1. Identify the figure families and their filename producers, output declarations and consumers.
   Check existing names before proposing a new grammar. Document an existing convention as implemented;
   describe any desired renaming separately.
2. Add a profile with its scope, field order, token meanings, spatial and run identifiers, formats,
   presentation variants and representative examples. Link to the owning builder or producer and
   relevant checks. Include only qualifiers needed to distinguish products, such as a scenario,
   model, ensemble or analysis window, and define their placement and identity source.
3. Reuse shared tokens and builders when their meanings and behavior fit. Keep workflow-specific
   vocabulary with its producer; add shared code only when multiple consumers need the same behavior.
4. Check that output declarations and renderers produce matching names and that names remain unique
   across the supported combinations. Update affected inventories, captions, reports, tests and links
   together. A documentation move alone must not change the generated names.
5. For an actual filename change, document the old-to-new mapping and how existing outputs will be
   read or regenerated. Follow the project's
   [naming and migration rules](../../../dev/reference/dev-naming-conventions.md); do not rename retained
   results in place as part of an editorial revision.
