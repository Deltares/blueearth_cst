# Guide illustration sources

Figures refreshed on 30 October 2026 with the plotting code at main revision `7931aab0`, including
its physical map-area correction. These illustrate toolbox products, not a validated basin assessment.
All thirteen examples were regenerated from the retained `wrt-egypt-v2` case. Ten changed; the monthly
climatology, annual precipitation trend and paired CMIP6 change cloud were byte-identical.

Spatial, historical-climate and projection figures use `scripts/render_figures.py` with families
`spatial`, `wf0` and `wf2`, `--annotation none`, the retained case as `--project`, and separate `--out`
folders. The hydrograph uses `shared.plot_evaluation.plot_hydrograph` with the exported daily
`run_default/output_q.csv` column `1050`, `Station(1050)` and no observations. This replaces the older
v3 example; its exported input values have the precision of that CSV.

Exact render paths, code revision and image SHA-256 hashes are in
[`new-case-sources.json`](new-case-sources.json). Render outputs are separate from the original run.
The latest seasonal precipitation example uses percentage changes and four sufficiently populated
scenarios; the agreement example uses annual and seasonal percentage classes. Guide captions describe
these conventions and exclusions.

Original run-output locations are listed below (relative to each case):

| Asset | Source |
|---|---|
| `basin.png` | `data/spatial/plots/subbasins.png` |
| `elevation-map.png` | `data/spatial/plots/elevation_basin.png` |
| `land-cover-map.png` | `data/spatial/plots/land_cover_basin.png` |
| `soil-clay-map.png` | `data/spatial/plots/soil_clay_basin.png` |
| `precip-map.png` | `data/climate/historical/era5_19510101_20201231/plots/era5_precip_annual_clim_map_basin_ext.png` |
| `temp-map.png` | `data/climate/historical/era5_19510101_20201231/plots/era5_temp_annual_clim_map_basin_ext.png` |
| `climate.png` | ERA5 monthly climatology; see the snapshot source manifest above. |
| `annual-precip-trend.png` | ERA5 annual precipitation trend; see the snapshot source manifest above. |
| `hydrograph.png` | Rendered from `models/hydrology/wflow/run_default/output_q.csv`, station 1050. |

CMIP6 paths are relative to `data/climate/projections/cmip6/plots/` in the v2 case:

| Asset | Source |
|---|---|
| `cmip6-cloud.png` | `period_2040_2060/cmip6_precip_temp_annual_change_cloud_basin_avg.png` |
| `cmip6-temperature.png` | `full_period/cmip6_temp_annual_anomaly_ts_basin_avg.png` |
| `cmip6-seasonal-precip.png` | `period_2040_2060/cmip6_precip_annual_seasonal_change_pct_box_basin_avg.png` |
| `cmip6-agreement.png` | `period_2040_2060/cmip6_precip_temp_annual_monthly_sign_agreement_basin_avg.png` |

Keep captions focused on how to read each output. The source figures retain their original labels,
including rounded trend slopes and reporting-location identifiers. Source climate records have different
coverage; CMIP6 period-change figures compare 2040–2060 with 1985–2004.
