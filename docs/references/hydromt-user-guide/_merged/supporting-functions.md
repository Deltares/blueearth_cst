---
title: HydroMT user guide — Supporting functionalities
ingest-source: hydromt-user-guide
upstream-repo: Deltares/hydromt
upstream-commit: 4b6ce64a7ed082740766e466a1861860f5150322
pulled: 2026-05-04
section: supporting-functions
doc-type: user-guide
license: MIT
sources:
  - https://deltares.github.io/hydromt/latest/user_guide/supporting_functions/methods_main.html
---

# HydroMT user guide — Supporting functionalities

_Merged from 1 upstream page(s) at commit `4b6ce64a`, pulled 2026-05-04. Page boundaries are preserved as `## ` headings._


---

## methods main

> **Source:** [supporting_functions/methods_main.md](https://deltares.github.io/hydromt/latest/user_guide/supporting_functions/methods_main.html) · [upstream `.rst`](https://github.com/Deltares/hydromt/blob/4b6ce64a7ed082740766e466a1861860f5150322/docs/user_guide/supporting_functions/methods_main.rst)

Supporting functions and (GIS) methods are the engine of HydroMT. Methods provide the low-level functionality, only accessible through the Python interface, to do the required processing of common data types such as grid and vector data.

HydroMT also provides higher-level processes that combine multiple methods to accomplish common tasks. These processes are used in the HydroMT model components to prepare model input data, but can also be used directly through the Python API. You can find which model processes are available in the `model processes documentation <model_processes>`

HydroMT provides a wide range of supporting functionalities, including:

- `RasterDataset accessor <raster_accessor>`
- `GeoDataset accessor <geodataset_accessor>`
- `Flow Directions methods <pyflwdir_wrappers>`
- `Statistical methods <stats>`
- `Extreme Value Analysis methods <extreme_value_analysis>`

## RasterDataset

Some powerful functionality that HydroMT uses is exposed in the `gis` module. In this module [xarray accessors](https://docs.xarray.dev/en/stable/internals/extending-xarray.html) are located. These allow for powerful new methods on top of xarray `Dataset` and `DataArray` classes.

HydroMT `~hydromt.gis.raster.RasterDataset` builds on top of these accessors to work with raster data. Available methods include reprojection, resampling, transforming, interpolating nodata or zonal statistics. And properties such as crs, bounds, or resolution .

To access these methods in python, first load a xarray Dataset or DataArray, for example using the HydroMT `~hydromt.data_catalog.DataCatalog.get_rasterdataset` method and use the `raster` accessor following by the method or attribute you want to use.

All available methods and properties are documented in the `raster API documentation <raster_api>`.

``` python
import hydromt

# Load a raster dataset from the data catalog
cat = hydromt.DataCatalog('path_to_your_catalog.yml')
ds = cat.get_rasterdataset('your_raster_key')

# Use the raster accessor to reproject the dataset
ds_reprojected = ds.raster.reproject(dst_crs='EPSG:4326', method="nearest")

# Get the bounds of the dataset
bounds = ds.raster.bounds
print(bounds)
```

For more example, you can check the [working with raster data notebook](../../_examples/working_with_raster.ipynb).

## GeoDataset

Some powerful functionality that HydroMT uses is exposed in the `gis` module. In this module [xarray accessors](https://docs.xarray.dev/en/stable/internals/extending-xarray.html) are located. These allow for powerful new methods on top of xarray `Dataset` and `DataArray` classes.

HydroMT `~hydromt.gis.vector.GeoDataset` builds on top of these accessors to work with geospatial vector data (N-dim point/line/polygon geometry). Available methods include reprojection, transforming, updating geometry or converting to geopandas.GeoDataFrame to access further GIS methods. And properties such as crs, bounds or geometry.

To access these methods in python, first load a xarray Dataset or DataArray, for example using the HydroMT `~hydromt.data_catalog.DataCatalog.get_geodataset` method and use the `vector` accessor following by the method or attribute you want to use.

All available methods and properties are documented in the `GeoDataset API documentation <geodataset_api>`.

``` python
import hydromt

# Load a geodataset from the data catalog
cat = hydromt.DataCatalog('path_to_your_catalog.yml')
gds = cat.get_geodataset('your_geodataset_key')

# Use the vector accessor to reproject the geodataset
gds_reprojected = gds.vector.to_crs(dst_crs='EPSG:4326')

# Convert to geopandas GeoDataFrame
gdf = gds.vector.geometry
print(gdf.head())
```

For more example, you can check the [working with geodataset notebook](../../_examples/working_with_geodatasets.ipynb).

## Flow directions

[PyFlwDir](https://deltares.github.io/pyflwdir/latest/index.html) contains a series of methods to work with gridded DEM and flow direction datasets. The `gis.flw` module builds on top of this and provides hydrological methods for raster DEM and flow direction data. For example, calculate flow direction, flow accumulation, stream network, catchments, or reproject hydrography.

Available methods are listed in the `flow directions API documentation <flw_api>`.

You can find examples of how to use these methods in the [working with flow direction notebook](../../_examples/working_with_flow_directions.ipynb).

## Statistical methods

HydroMT `stats.skills` module provides different statistical functions to compare model results with observations. They include:

- Absolute and percentual bias
- Nash-Sutcliffe model Efficiency (NSE) and log Nash-Sutcliffe model Efficiency (log-NSE)
- Various versions of the Kling-Gupta model Efficiency (KGE)
- Coefficient of determination (R-squared)
- Mean Squared Error (MSE) and Root Mean Squared Error (RMSE)

The full list is available in the `skills statistics API <statistics_skills>`.

As HydroMT provides methods to easily read the model results, applying a skill statistic just takes a few lines of code and can be applied directly across all observation locations in your model.

``` console
from hydromt.stats import nashsutcliffe
from hydromt_wflow import WflowModel
import xarray as xr
# read model results
# NOTE: the name of the results depends on the wflow run configuration (toml file)
mod = WflowModel(root=r'/path/to/wflow_model/root', mode='r')
sim = mod.results['Q_gauges_grdc']
# read observations
obs = xr.open_dataset(r'/path/to/grdc_obs.nc')
# calculate skill statistic
nse = nashsutcliffe(sim, obs)
```

## Extreme Value Analysis

HydroMT `stats.extremes` module provides different functions to perform Extreme Value Analysis (EVA) on hydrological data. They include full extreme value analysis or sub methods to get peaks, fit extremes or get return values.

The full list is available in the `extreme value analysis API <statistics_extremes>`.

You can find examples of how to use these methods in the [extreme value analysis notebook](../../_examples/doing_extreme_value_analysis.ipynb).
