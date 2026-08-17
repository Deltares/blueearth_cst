# WF0 figure filename rule

> **Status:** Agreed working rule, 2026-08-17. Apply to WF0 first; consider
> extending the grammar to other workflows after the WF0 implementation is
> established.

## Rule

WF0 figure filenames must use lowercase `snake_case` and follow:

```text
<dataset_scope>_<variable>_<plot_context>_<spatial_scope>.<extension>
```

The fields mean:

- `dataset_scope`: the dataset ID for a single-source figure, such as `era5`
  or `chirps`; use `comparison` for a figure containing multiple datasets.
- `variable`: the canonical, unabridged scientific variable name, such as
  `precip`, `temp`, or `pet`.
- `plot_context`: the temporal interpretation and plot form, expressed with
  the controlled vocabulary below.
- `spatial_scope`: the spatial aggregation or coverage represented by the
  figure.

For comparison figures, the individual dataset names must appear in the figure
legend and run provenance, not in the filename. This keeps filenames stable as
the number of compared datasets changes.

Workflow ID, project name, units, and analysis period should remain outside the
filename unless one is required to prevent a real collision.

## Controlled vocabulary

Use these abbreviations consistently:

| Meaning | Token |
| --- | --- |
| time series | `ts` |
| climatology | `clim` |
| average | `avg` |
| extent | `ext` |
| distribution box plot | `box` |

Recommended plot-context tokens:

```text
annual_ts
monthly_box
annual_clim_map
monthly_clim_line
```

Recommended spatial-scope tokens:

```text
basin_avg
basin_ext
source_ext
subbasin_<id>_avg
subbasin_<id>_ext
station_<id>
```

Do not abbreviate the canonical variables `precip`, `temp`, and `pet`. Avoid
`st` for station because `st` already identifies a stress-test member elsewhere
in the project.

### Which `<id>` each scope takes

`subbasin_<id>` and `station_<id>` draw from **different identifier
namespaces**, and they are deliberately not the same number:

| Scope | Column | Formula (ADR 0003 §12) | Basin 1 |
| --- | --- | --- | --- |
| `subbasin_<id>` | `subbasin_id` | `basin_id*100 + local_subbasin_number` | `101`–`104` |
| `station_<id>` | `wflow_id` | `basin_id*1000 + local_subbasin_number*10 + m` | `1010`, `1020`, … |

`m` is `0` for a subbasin's primary location and `1`–`9` for additional gauges
inside it, so subbasin `101` holds stations `1010`, `1011`, `1012`, … The two
were EQUAL before 2026-08-06; §12 repealed that on purpose, having explicitly
rejected a near-aligned alternative for giving "two 3-digit namespaces with
different meanings and no visual tell".

So a subbasin figure takes `101`, never `1010` — the digits differ from the
station figure beside it (`hydrograph_1010.png`) because they identify
different things. This section exists because the example below said `1010`
until 2026-08-17, which is the station.

New contexts or abbreviations must be added to this controlled vocabulary
rather than introduced ad hoc in individual plotting modules.

## Examples

```text
era5_precip_annual_ts_basin_avg.png
era5_temp_annual_clim_map_basin_ext.png
chirps_precip_monthly_box_basin_avg.png
era5_precip_annual_ts_subbasin_101_avg.png
comparison_precip_annual_ts_basin_avg.png
comparison_temp_monthly_box_basin_avg.png
```
