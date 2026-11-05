---
title: "Spatial data"
format: html
---

The toolbox combines global or near-global hydrography, land cover, vegetation, soil and waterbody
data to construct Wflow-SBM models. The table links to each source and summarizes its role and native
resolution. Distances for angular grids are approximate at the equator; static inventories have no
regular time step.

| Dataset | Model contribution | Spatial representation | Time representation |
|---|---|---|---|
| [MERIT Hydro v1.0][merit] | Terrain and drainage | 3 arc-seconds (~90 m) | Static |
| [MERIT Hydro IHU v1.0][ihu] | Upscaled hydrography | 30 arc-seconds (~1 km) | Static |
| [Lin river data v1][lin] | Channel geometry | River-reach polylines | Static; 2019 release |
| [GlobCover v2.3][globcover] | Land-cover parameters | 300 m raster | 2009 map |
| [MODIS MCD15A3H][modis] | Seasonal vegetation | 500 m native grid | 4-day composites |
| [SoilGrids 2017][soilgrids] | Soil parameters | 250 m maps | Static predictions |
| [HydroLAKES v1.0][lakes] | Lake extent and storage | Polygons; lakes ≥10 ha | Static inventory |
| [GRanD v1.1][grand] | Reservoir attributes | Dam points and polygons | Static inventory |
| [RGI v6.0][rgi] | Glacier extent | Glacier polygons | Static; 2017 release |

**Hydrography.** MERIT Hydro supplies terrain and drainage information for basin delineation, with
the configured IHU entry using the coarser 30 arc-second product. Lin's river data add width and
bankfull discharge estimates for channels wider than 30 m; this threshold is not a grid resolution.

**Land cover and vegetation.** GlobCover supplies the default land-cover classes, while MODIS LAI
provides seasonal vegetation information from Collection 6 observations extracted for 2003–2017.
The base catalog also includes [Copernicus Global Land Cover][vito], a 100 m alternative for 2015.

**Soils.** SoilGrids 2017 provides properties at seven depths from 0 to 200 cm for estimating soil
storage and hydraulic parameters. The prepared soil-thickness layer also uses the
[European Soil Database v2.0][esdac] for Eurasia.

**Waterbodies and glaciers.** HydroLAKES and GRanD describe lake and reservoir extent and storage;
the prepared reservoir data also use [JRC Global Surface Water][jrc] at 30 m for 1984–2015.
RGI provides static glacier outlines, and the model recipes apply area thresholds when selecting
lakes, reservoirs and glaciers.

Check the resulting drainage network, basin boundaries and major waterbodies against local information.
Resampling these sources to the model grid does not improve their underlying resolution or accuracy.

[merit]: https://global-hydrodynamics.github.io/MERIT_Hydro/
[ihu]: https://zenodo.org/records/5166932
[lin]: https://zenodo.org/records/3552776
[globcover]: https://due.esrin.esa.int/page_globcover.php
[modis]: https://ladsweb.modaps.eosdis.nasa.gov/missions-and-measurements/products/MCD15A3H
[soilgrids]: https://docs.isric.org/globaldata/soilgrids/SoilGrids_faqs_2017.html
[lakes]: https://www.hydrosheds.org/products/hydrolakes
[grand]: https://www.globaldamwatch.org/grand
[rgi]: https://nsidc.org/data/nsidc-0770/versions/6
[vito]: https://zenodo.org/records/3243509
[esdac]: https://esdac.jrc.ec.europa.eu/content/european-soil-database-v20-vector-and-attribute-data
[jrc]: https://global-surface-water.appspot.com/download
