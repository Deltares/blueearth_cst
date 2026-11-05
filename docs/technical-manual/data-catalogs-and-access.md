---
title: "Data catalogs"
format: html
---

Data catalogs tell the toolbox where datasets are stored and how to read their variables and units.
They describe access to data; they do not install the datasets. Choose the catalog appropriate to your
data storage and analysis:

- **[HydroMT data catalog](catalog-deltares.md):** spatial data and historical meteorology for basin
  preparation, model building and weather generation.
- **[CMIP6 catalog](catalog-climate-projections.md):** the currently supported climate-projection
  archive, selected separately in the projection-analysis settings.

Each page explains when to use the catalog and shows a syntax-highlighted excerpt from its YAML file.
The excerpts illustrate selected fields; use the linked full files when configuring a case.

## Read a catalog entry

For catalog structure, dataset readers and data adapters, refer to the
[HydroMT manual](https://deltares.github.io/hydromt/stable/).

## Use your own data archive

Keep a project-owned catalog beside your case and select it through `project.catalog`. With schema 3,
project paths resolve beside the YAML that declares them; `toolbox:` refers to this toolbox checkout.
Dataset paths inside a catalog follow HydroMT's catalog-root rules.

Check source names, coverage, units and time conventions before substituting data. Raw provider files
may differ from the prepared daily archives expected by the toolbox. Historical forcing checks raise
`UnverifiedForcingUnits` if conversion rules differ from those reviewed in the
[forcing adapter](../../blueearth_cst/climate_analysis/prepare_climate_data_catalog.py).

## Catalogs created during preparation

The toolbox writes `data/spatial/spatial_catalog.yml` alongside `spatial_maps.nc` and
`location_registry.csv`. HydroMT-Wflow uses it to read the prepared basin data. To change those inputs,
update the project sources and repeat preparation; do not edit this generated catalog directly.
