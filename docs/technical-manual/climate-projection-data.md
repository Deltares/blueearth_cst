---
title: "Climate projections"
format: html
---

Climate projections describe possible changes in temperature and precipitation relative to a historical
reference period. They help interpret the conditions explored in the stress test without determining
the scenarios generated. The toolbox currently supports the
[Coupled Model Intercomparison Project Phase 6 (CMIP6)][cmip6], with scope to add other projection datasets
in future. See [projection analysis](workflow-analyze-projections.md) for data selection and calculations.

The current implementation reads from the [Pangeo CMIP6 archive hosted on Google Cloud][store].
This archive provides climate-model simulations of historical and future conditions, organized by
model, experiment, ensemble member and variable. Data are stored in Zarr format for efficient access
to selected portions of large datasets; the toolbox uses monthly precipitation and near-surface air
temperature. The [Pangeo archive documentation][pangeo] describes its holdings and organization.

Changing the CMIP6 ensemble or windows changes the plausibility comparison, not an already generated
scenario collection. Align relative precipitation changes and absolute temperature changes with the
stress-test axes before plotting them together. The projection settings template shows example selections;
it is not a prescribed ensemble for every basin. The number of available models and members, spatial
resolution, and variable coverage depend on the catalog. Ensemble spread describes differences among those
available projections, not a calibrated probability distribution for basin-scale outcomes.

[cmip6]: https://cds.climate.copernicus.eu/datasets/projections-cmip6?tab=overview
[store]: https://console.cloud.google.com/storage/browser/cmip6/CMIP6
[pangeo]: https://pangeo-data.github.io/pangeo-cmip6-cloud/overview.html#data-locations
