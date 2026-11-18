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

## Choosing, replacing, and recording sources {#choosing-replacing-and-recording-sources}

HydroMT catalogs allow a project to name alternative data products without embedding file paths in the
workflow. A substitute must still provide the variables, units, geometry, temporal coverage, and
HydroMT-readable structure expected by the model setup. Changing a source can require rebuilding the model or
regenerating the weather collection before results are comparable.

Record the catalog and entry names, source product and version, spatial resolution, temporal windows,
any local preprocessing and observation series used for evaluation. Keep them with run records when
comparing basins or rerunning an assessment. See [configuration](configuration.md) for exact path rules.

The executing project's `basin.sources` and the selected build configuration files determine the actual source
names. The starter uses `deltares_data_pdrive.yml` and `lulc: globcover`; a local catalog's availability can
differ. Do not infer enabled data from the illustrative dataset pages. Generated CMIP6 catalog and store-index
files list the available models and ensemble members. Catalog YAML/root context is captured for reuse checks,
while workflows check the input formats, coverage and units. A catalog entry alone cannot establish scientific
suitability.

## Catalogs created during preparation

The toolbox writes `data/spatial/_engine/spatial_catalog.yml` for `spatial_maps.nc`,
`location_registry.csv` and `geoms/`; its paths are relative to `_engine/`. HydroMT-Wflow uses it to read the prepared basin data. To change those inputs,
update the project sources and repeat preparation; do not edit this generated catalog directly.
