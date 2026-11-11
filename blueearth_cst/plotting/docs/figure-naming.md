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
- Use the comparison directory for multi-source context. List participating sources in legends,
  captions and provenance rather than making the filename depend on the number of sources.
- Keep project names, workflow IDs, units and analysis periods in directories or metadata unless a
  profile needs one to distinguish products in the same output directory.
- Keep presentation variants, such as captioned copies, identifiable without changing the scientific
  figure's identity. Each profile specifies supported formats and how variants are stored.
- Treat existing filenames as interfaces: workflow outputs, reports and downstream readers may depend
  on them. Upstream-controlled names can remain explicit exceptions to toolbox naming rules.

## Workflow profiles

The observed water-balance profile uses `observed_runoff_coefficient_budyko_stations.png` under
model `evaluation/budyko/plots/`. `model/plot_budyko.py` renders this retained-table station comparison;
source recipe, period and station identities live in its tables and methods record. Annotation modes
use the same PNG path, with readable title/caption inventories and machine records in the root `_engine/`.

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
<dataset>_<subject>_<diagnostic>_<location>.<extension>
```

The fields mean:

- `dataset`: the dataset ID for a single-source figure, such as `era5`
  or `chirps`; use `comparison` for a figure containing multiple datasets.
- `subject`: the canonical variable (`precip`, `temp`, `pet`, `precip_temp`) or thematic index
  scope (`heavy_rainfall`, `heatwaves`, etc.). Avoid the redundant `climate_indices` prefix.
- `diagnostic`: concise meaning, such as `annual`, `annual_trend`, `monthly_cycle`, `timing`,
  `wet_spell_exceedance`, or `annual_mean_map`. Retain `map`/`box` when they distinguish products.
- `location`: `basin`, `source`, `sub<ID>`, `loc<ID>`, `station<ID>`, or a meaningful location name.
  Named locations use lowercase ASCII snake_case; missing, automatic or duplicate names fall back
  to their established IDs. Folder paths, captions and inventories retain the original IDs/support.

For comparison figures, the individual dataset names must appear in the figure
legend and run provenance, not in the filename. This keeps filenames stable as
the number of compared datasets changes.

### Controlled vocabulary

Logical API keys keep their established spellings; the filename builder shortens their exports:

| Logical key | Exported spelling |
| --- | --- |
| `annual_ts` | `annual` |
| `annual_trend_ts` | `annual_trend` |
| `monthly_clim_band`, `monthly_clim_line` | `monthly_cycle` |
| `annual_timing` | `timing` |
| `annual_clim_map` | `annual_mean_map` |
| `annual_rx1day_ts`, `annual_rx5day_ts` | `rx1day_annual`, `rx5day_annual` |
| `annual_sdii_ts`, `annual_wet_days_ts` | `sdii_annual`, `wet_days_annual` |
| `annual_intensity_ts` | `intensity_annual` |
| `monthly_spi`, `monthly_spi_fit` | `spi`, `spi_fit` |
| `monthly_anomaly_acf` | `anomaly_acf` |
| `annual_trend_interval`, `annual_trend_sensitivity` | `trend_intervals`, `trend_sensitivity` |

Stable logical plot-context keys used by producers (not literal exported filename tokens):

```text
annual_clim_map
annual_anomaly_map
monthly_clim_band
annual_timing
monthly_anomaly
monthly_spi
spi_events
annual_rx1day_ts
annual_rx5day_ts
annual_sdii_ts
annual_intensity_ts
annual_wet_days_ts
dry_spell_exceedance
monthly_anomaly_acf
annual_trend_interval
annual_trend_ts
seasonal_anomaly
monthly_spi_fit
annual_trend_sensitivity
daily_agreement
monthly_agreement
```

WF0 now groups Rx1day, Rx5day and SDII in `annual_intensity_ts`. The separate contexts and
`monthly_spi_fit` remain recognized for older exports; current WF0 does not emit those figures.

The machine vocabulary and descriptions live in `figure_naming.py`. `annual_ts` and `monthly_clim_line`
support annual overlays and monthly climatologies in location comparisons as well as the WF1 source
producer. `monthly_box` remains registered for the WF1 producer. `monthly_agreement` denotes complete
monthly observation/grid scatter panels. WF0 emits PNGs only. Optional captioned copies use the
same filename under `plots/captioned/`; captions never add a fifth field.

Optional annual spatial anomalies use `annual_anomaly_map` directly under the source's `plots/`.
If more than six year panels are requested, append `_page_01`, `_page_02`, etc. before `.png`;
a single figure has no suffix. `annual_anomaly_filename` owns this pagination exception. Exact dates
and panel membership remain in titles, captions and the source's retained anomaly record.

Climate-index sheets export `<dataset>_<scope>_<period>_<location>.png`, with scopes `spells`,
`heavy_rainfall`, `rainfall_characteristics`, `temperature_extremes`, and `heatwaves`. They replace
the combined `annual_indices_ts`/`seasonal_indices_ts` exports. Only available sheets and maps are
emitted in the owned `plots/indices/` directory; its captions and availability records list the set.

Logical support keys remain explicit in the API:

```text
basin_avg
basin_ext
source_ext
subbasin_<id>_avg
subbasin_<id>_ext
station_<id>_avg
location_<id>_point
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
Location comparisons emit `annual_ts`, `monthly_clim_line` and optional
`monthly_agreement` with `location_<id>_point`. Declared locations use the established Wflow ID as
`<id>`; gauge-only sites use `station_<safe-token>`. The comparison's `locations.csv` maps these tokens
to spatial location codes and literal gauge IDs. The logical `_point` support distinguishes native-cell
point samples from area averages; the exported location token carries the name or `loc<ID>`.
Earlier gauge outputs used `station_<token>_avg` and `daily_agreement`;
legacy retained products remain readable through the `rain-gauges` rerender family.

Comparison figures omit the `comparison_` prefix; their directory supplies the context.

### Examples

```text
era5_precip_annual_trend_basin.png
era5_temp_annual_mean_map_basin.png
chirps_precip_monthly_cycle_basin.png
era5_precip_annual_trend_sub101.png
precip_rx1day_annual_basin.png
precip_temp_seasonal_anomaly_basin.png
precip_annual_loc1010.png
precip_annual_mainstem_mid.png
chirps_precip_wet_spell_exceedance_basin.png
chirps_heavy_rainfall_annual_basin.png
```

### Migration to shorter names (2026-11-08)

New WF0 renders use the shorter names. Scientific calculations, folder ownership, logical context/support
keys and retained-table filenames are unchanged. Captions/title inventories use the new PNG paths.
Original run PNGs remain historical evidence; render into a separate destination or preserve them before
regenerating. Do not infer spatial reduction from a dropped `_avg`, `_ext` or `_point` suffix: use the
diagnostic, location token and the retained support metadata. WF1 forcing and WF2/WF3 naming stay unchanged.

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
