---
title: "Datasets"
format: html
---

The toolbox reads external gridded data through HydroMT catalogs. The project names catalogs and source entries; the data files are not distributed with the workflow code. The [2025 CST technical note](https://github.com/tanerumit/blueearth_cst/blob/main/docs/references/cst-toolbox-technical-note-2025.md) describes the recommended global datasets and their scientific roles. The table below adapts that inventory to the [current project template](https://github.com/tanerumit/blueearth_cst/blob/main/config/templates/example-basin/project_config.yml), [data catalog](https://github.com/tanerumit/blueearth_cst/blob/main/config/catalogs/deltares_data.yml), and [projection template](https://github.com/tanerumit/blueearth_cst/blob/main/config/templates/example-basin/project_config_analyze_projections.yml). It is a source-role guide, not a claim that every listed source is enabled or locally available.

## Spatial data for the basin and model

| Data class and example source | Use in the toolbox | Assessment sensitivity |
|---|---|---|
| Hydrography and basin index — MERIT Hydro | Delineate the drainage area, establish flow directions and river cells, and place output points on the network | Basin extent, upstream area, routing, and location identity can change with the hydrography or snapping rules |
| River geometry — Lin et al. river data | Inform river width or related geometry in the model setup | Flow travel time and channel response can change |
| Land cover — configured `basin.sources.lulc` (GlobCover in the starter) | Classify land surface through matching HydroMT-Wflow mappings | Interception, evapotranspiration, infiltration and runoff partitioning can change |
| Leaf area index — MODIS LAI | Describe vegetation seasonality in the model maps | Evapotranspiration and interception behavior can change |
| Soil properties — SoilGrids | Derive spatial soil and storage parameters | Soil-water storage, recharge, and low-flow response can change |
| Lakes and reservoirs — HydroLakes and configured inventories | Represent waterbodies where the model setup includes them | Storage and routed flows can change, especially in regulated basins |
| Glaciers — Randolph Glacier Inventory (RGI), when configured | Define glacier areas for relevant basins | Snow and ice contributions to seasonal flow can change |

These products have different native spatial support, observation periods, and uncertainties. Resampling them
to one Wflow grid does not make their information equally precise. The model grid resolution is therefore a
computational choice, not a statement of the effective resolution of soil, land-cover, or river parameters.
Review basin boundaries, drainage structure, elevation, land cover, and known waterbodies after setup,
especially where a small spatial error could move an outlet or change upstream area.

## Historical meteorology and observations

| Source | Role | Checks before assessment |
|---|---|---|
| ERA5 daily forcing | Supplies the selected historical meteorology in the supplied example; drives the historical Wflow run and conditions weather generation | Coverage over the basin and time window, variable units, precipitation magnitude and seasonality, temperature, and missing values |
| CHIRPS precipitation | Alternative or candidate precipitation source for historical-climate comparison | Geographic coverage, rainfall seasonality and extremes, and consistency with the other forcing variables needed by the model and generator |
| Local observed discharge | Optional historical model evaluation at configured locations | Gauge identity, dates, units and known regulation |

The [2025 note](https://github.com/tanerumit/blueearth_cst/blob/main/docs/references/cst-toolbox-technical-note-2025.md) identifies historical meteorology as serving **two purposes**: forcing the hydrological model and conditioning the stochastic weather generator. This makes source selection unusually consequential in the current toolbox, which does not locally calibrate the model. Historical climate analysis can compare available climate variables across candidate sources before one is selected. In the current pipeline, CHIRPS supplies precipitation; when it is selected, ERA5 contributes the other variables needed for model building and weather generation. A structurally complete climate store is not, by itself, a scientific validation of the forcing.

The historical extraction window must cover the historical model run and provide enough complete years for the
weather generator. The project template requires at least 16 inclusive years. Compare source behavior with
local stations or hydrological evidence where available; gridded products may smooth local extremes or miss
topographic gradients even when basin-average statistics look plausible.

## Future climate information

Projection analysis reads configured **CMIP6** model, scenario, ensemble-member, reference-window, and
future-window
selections. It summarizes changes in precipitation and temperature for each combination. The results provide
an overlay on the stress-test response surface, not forcing for scenario generation. Changing the CMIP6
ensemble or windows
therefore changes the plausibility comparison, not an already generated scenario collection. Align its
relative precipitation changes and absolute temperature changes with the stress-test axes before plotting them
together.

The [projection settings template](https://github.com/tanerumit/blueearth_cst/blob/main/config/templates/example-basin/project_config_analyze_projections.yml) shows example selections; it is not a prescribed ensemble for every basin. The number of available models and members, spatial resolution, and variable coverage depend on the catalog. Ensemble spread describes differences among those available projections, not a calibrated probability distribution for basin-scale outcomes.

## Choosing, replacing, and recording sources

HydroMT catalogs allow a project to name alternative data products without embedding file paths in the
workflow. A substitute must still provide the variables, units, geometry, temporal coverage, and
HydroMT-readable structure expected by the model setup. Changing a source can require rebuilding the model or
regenerating the weather collection before results are comparable.

Record the catalog and entry names, source product and version, spatial resolution, temporal windows,
any local preprocessing and observation series used for evaluation. Keep them with run records when
comparing basins or rerunning an assessment. See [configuration](configuration.md) for exact path rules.

## Configured data sources and background maps {#source-authority-and-locator-asset}

The executing project's `basin.sources` and the selected engine recipes determine the actual source names.
The starter uses `deltares_data_pdrive.yml` and `lulc: globcover`; a local catalog's availability can differ.
Do not infer enabled data from the illustrative source-role inventory above. Generated CMIP6 catalog and
store-index files list the available models and ensemble members.

`config/basemap/natural_earth_50m.gpkg` is a vendored locator-map asset, not basin forcing or model data.
Its [local README](../../config/basemap/README.md) describes the map source and preparation.
Catalog YAML/root context is captured for reuse checks, while workflows check the input formats,
coverage and units. A catalog entry alone cannot establish scientific suitability.
