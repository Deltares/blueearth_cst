---
title: model_processes
ingest-source: hydromt-user-guide
source: https://deltares.github.io/hydromt/latest/user_guide/models/model_processes.html
repo-source: https://github.com/Deltares/hydromt/blob/4b6ce64a7ed082740766e466a1861860f5150322/docs/user_guide/models/model_processes.rst
upstream-repo: Deltares/hydromt
upstream-commit: 4b6ce64a7ed082740766e466a1861860f5150322
pulled: 2026-05-04
doc-type: user-guide
license: MIT
---
<div class="currentmodule">

hydromt.model.processes

</div>

# Model processes

HydroMT uses **processes** functions to transform raw input data into model-ready inputs and parameters. These processes can be as simple as reprojecting a raster dataset to the model grid, or more complex operations like delineating a river network based on a digital elevation model (DEM).

Processes are only available through the Python API and are usually the backbone of the model and components `setup_` or `add_data_` methods. HydroMT proposes a collection of methods that can be re-used in your python scripts or when developing your own plugins.

## Grid

These methods allow to either create a grid or add data to an existing grid from different data sources.

| Processes | Description |
|----|----|
| `~grid.create_grid_from_region` | Create a 2D regular grid from a specified region. |
| `~grid.create_rotated_grid_from_geom` | Create a rotated grid based on a geometry. |
| `~grid.grid_from_constant` | Add data to a grid using a constant value. |
| `~grid.grid_from_rasterdataset` | Add data to a grid by resampling a raster dataset. |
| `~grid.grid_from_raster_reclass` | Reclassify raster data and add it to a grid. |
| `~grid.grid_from_geodataframe` | Add data to a grid by rasterizing a GeoDataFrame. |
| `~grid.rotated_grid` | Return the origin (x0, y0), shape (mmax, nmax) and rotation of the rotated grid. |

## Mesh

These methods allow to either create a mesh or add data to an existing mesh from different data sources.

| Processes | Description |
|----|----|
| `~mesh.create_mesh2d_from_region` | Create a 2D mesh from a specified region according to UGRID conventions. |
| `~mesh.create_mesh2d_from_mesh` | Create a 2D mesh based on an existing mesh. |
| `~mesh.create_mesh2d_from_geom` | Create a regular 2D mesh from a boundary geometry. |
| `~mesh.mesh2d_from_rasterdataset` | Add data to a mesh by resampling a raster dataset. |
| `~mesh.mesh2d_from_raster_reclass` | Reclassify raster data and add it to a mesh. |

## Region

These methods allow to parse different region definitions (from the HydroMT region dictionary) for setting up a model region.

| Processes | Description |
|----|----|
| `~region.parse_region_basin` | Parse a hydrographic basin region definition. |
| `~region.parse_region_bbox` | Parse a bounding box region definition. |
| `~region.parse_region_geom` | Parse a geometry file region definition. |
| `~region.parse_region_grid` | Parse a raster grid file region definition. |
| `~region.parse_region_other_model` | Parse a region definition based on another HydroMT model. |
| `~region.parse_region_mesh` | Parse a mesh file region definition. |

## Basin mask

This method allows to delineate hydrographic regions (basin, interbasin, subbasin) using the region dictionary and a hydrography (DEM, flow direction) dataset.

| Processes | Description |
|----|----|
| `~basin_mask.get_basin_geometry` | Return a geometry of the (sub)(inter)basin(s). |

## River bathymetry

These methods allow to estimate river channel dimensions based on either direct cross-section information or empirical relationships.

| Processes | Description |
|----|----|
| `~rivers.river_width` | Return average river width along a segment based on a river mask raster. |
| `~rivers.river_depth` | Derive river depth estimates from data or based on bankfull discharge. |

## Meteo

These methods allow to process meteorological data for use in gridded models. This includes time resampling, downscaling, and calculation of potential evapotranspiration.

| Processes | Description |
|----|----|
| `~meteo.precip` | Process precipitation data. |
| `~meteo.temp` | Process temperature data. |
| `~meteo.press` | Process atmospheric pressure data. |
| `~meteo.pet` | Determine reference evapotranspiration. |
| `~meteo.wind` | Process wind speed data. |
| `~meteo.press_correction` | Pressure correction based on elevation lapse_rate. |
| `~meteo.temp_correction` | Temperature correction based on elevation data. |
| `~meteo.resample_time` | Resample meteorological data to a different time frequency. |
| `~meteo.pet_debruin` | Calculate potential evapotranspiration using De Bruin method. |
| `~meteo.pet_makkink` | Calculate potential evapotranspiration using Makkink method. |
| `~meteo.pm_fao56` | Calculate potential evapotranspiration using Penman-Monteith FAO56 method. |
