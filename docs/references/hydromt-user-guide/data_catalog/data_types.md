---
title: data_types
ingest-source: hydromt-user-guide
source: https://deltares.github.io/hydromt/latest/user_guide/data_catalog/data_types.html
repo-source: https://github.com/Deltares/hydromt/blob/4b6ce64a7ed082740766e466a1861860f5150322/docs/user_guide/data_catalog/data_types.rst
upstream-repo: Deltares/hydromt
upstream-commit: 4b6ce64a7ed082740766e466a1861860f5150322
pulled: 2026-05-04
doc-type: user-guide
license: MIT
---
<div id="data_types">

<div class="currentmodule">

hydromt.data_catalog.drivers

</div>

</div>

# Supported data types

HydroMT currently supports the following data types:

- `RasterDataset <RasterDataset>`: static and dynamic raster (or gridded) data
- `GeoDataFrame <GeoDataFrame>`: static vector data
- `GeoDataset <GeoDataset>`: dynamic vector data
- `Dataset <Dataset>`: non-spatial n-dimensional data
- `DataFrame <DataFrame>`: 2D tabular data

Internally the RasterDataset, GeoDataset, and Dataset are represented by `xarray.Dataset` objects, the GeoDataFrame by `geopandas.GeoDataFrame`, and the DataFrame by `pandas.DataFrame`. We use drivers, typically from third-party packages and sometimes wrapped in HydroMT functions, to parse many different file formats to this standardized internal data representation.

> [!NOTE]
> It is also possible to create your own driver. See at `Custom Driver<custom_driver>`

## Recognized dimension names

- **time**: time or date stamp \["time"\].
- **x**: x coordinate \["x", "longitude", "lon", "long"\].
- **y**: y-coordinate \["y", "latitude", "lat"\].

## Raster data (RasterDataset)

- `Single variable GeoTiff raster <GeoTiff>`
- `Multi variable Virtual Raster Tileset (VRT) <VRT>`
- `Tiled raster dataset <Tile>`
- `Netcdf raster dataset <NC_raster>`

<div id="raster_formats">

| Driver | File formats | Method(s) | Comments |
|----|----|----|----|
| `rasterio <raster.rasterio_driver.RasterioDriver>` | GeoTIFF, ArcASCII, VRT, etc. (see [GDAL formats](http://www.gdal.org/formats_list.html)) | `~hydromt.readers.open_mfraster` | Based on `xarray.open_rasterio` and `rasterio.open` |
| `rasterio <raster.rasterio_driver.RasterioDriver>` with the `raster_tindex <hydromt.data_catalog.uri_resolvers.raster_tindex_resolver.RasterTindexResolver>` resolver | rasterio tile index file (see [gdaltindex](https://gdal.org/programs/gdaltindex.html)) | `~hydromt.readers.open_mfraster` | Options to merge tiles via `mosaic_kwargs` option. |
| `raster_xarray <raster.raster_xarray_driver.RasterDatasetXarrayDriver>` | NetCDF and Zarr | `xarray.open_mfdataset`, `xarray.open_zarr` | required y and x dimensions |

</div>

<div id="GeoTiff">

**Single variable GeoTiff raster**

</div>

Single raster files are parsed to a **RasterDataset** based on the **raster** driver. This driver supports 2D raster with "x" and "y" dimensions. A potential third dimension is called "dim0". The variable name is based on the filename, in this case <span class="title-ref">"GLOBCOVER_200901_200912_300x300m"</span>. The `chunks` key-word argument is passed to the underlying method and allows lazy reading of the data.

<div class="literalinclude" language="yaml">

../../assets/data_types/single_variable_geotiff_raster.yml

</div>

<div class="testsetup">

- 

from hydromt import DataCatalog

</div>

<div class="testcode" hide="">

geotiff

catalog_path = "docs/assets/data_types/single_variable_geotiff_raster.yml" catalog = DataCatalog(data_libs=\[catalog_path\])

</div>

### Multi-variable Virtual Raster Tileset (VRT)

Multiple raster layers from different files are parsed using the **raster** driver. Each raster becomes a variable in the resulting RasterDataset based on its filename. The path to multiple files can be set using a glob string or several keys, see description of the `uri` argument in the `yaml file description <data_yaml>`. Note that the rasters should have identical grids.

Here multiple .vrt files (dir.vrt, bas.vrt, etc.) are combined based on their variable name into a single dataset with variables flwdir, basins, etc. Other multiple file raster datasets (e.g. GeoTIFF files) can be read in the same way. VRT files are useful for large raster datasets which are often tiled and can be combined using [gdalbuildvrt.](https://gdal.org/programs/gdalbuildvrt.html)

<div class="literalinclude" language="yaml">

../../assets/data_types/vrt_raster_dataset.yml

</div>

<div class="testcode" hide="">

geotiff

catalog_path = "docs/assets/data_types/vrt_raster_dataset.yml" catalog = DataCatalog(data_libs=\[catalog_path\])

</div>

### Tiled raster dataset

Tiled index datasets are parsed using the `raster_tindex <hydromt.data_catalog.uri_resolvers.raster_tindex_resolver.RasterTindexResolver>` `~hydromt.data_catalog.uri_resolvers.uri_resolver.URIResolver`. This data format is used to combine raster tiles with different CRS projections. A polygon vector file (e.g. GeoPackage) is used to make a tile index with the spatial footprints of each tile. When reading a spatial slice of this data the files with intersecting footprints will be merged together in the CRS of the most central tile. Use [gdaltindex](https://gdal.org/programs/gdaltindex.html) to build an excepted tile index file.

Here a GeoPackage with the tile index referring to individual GeoTiff raster tiles is used. The `mosaic_kwargs` are passed to `hydromt.gis.merge` to set the resampling method. The name of the column in the tile index attribute table `tileindex` which contains the raster tile file names is set in the `driver.options`

<div class="literalinclude" language="yaml">

../../assets/data_types/tiled_raster_dataset.yml

</div>

<div class="testcode" hide="">

geotiff

catalog_path = "docs/assets/data_types/tiled_raster_dataset.yml" catalog = DataCatalog(data_libs=\[catalog_path\])

</div>

> [!NOTE]
> Tiled raster datasets are not read lazily as different tiles have to be merged together based on their values. For fast access to large raster datasets, other formats might be more suitable.

### Netcdf raster dataset

Netcdf and Zarr raster data are typically used for dynamic raster data and parsed using the **netcdf** and **zarr** drivers. A typical raster netcdf or zarr raster dataset has the following structure with two ("y" and "x") or three ("time", "y" and "x") dimensions. See list of recognized [dimensions](#dimensions) names.

``` console
Dimensions:      (latitude: NY, longitude: NX, time: NT)
Coordinates:
  * longitude    (longitude)
  * latitude     (latitude)
  * time         (time)
Data variables:
    temp         (time, latitude, longitude)
    precip       (time, latitude, longitude)
```

To read a raster dataset from a multiple file netcdf archive the following data entry is used, where the `options` are passed to `xarray.open_mfdataset` (or `xarray.open_zarr` for zarr data). In case the CRS cannot be inferred from the netcdf metadata it should be defined with the `crs` `metadata` here. The path to multiple files can be set using a glob string or several keys, see description of the `uri` argument in the `yaml file description <data_yaml>`. In this example additional renaming and unit conversion preprocessing steps are added to unify the data to match the HydroMT naming and unit `terminology <terminology>`.

<div class="literalinclude" language="yaml">

../../assets/data_types/netcdf_raster_dataset.yml

</div>

<div class="testcode" hide="">

geotiff

catalog_path = "docs/assets/data_types/netcdf_raster_dataset.yml" catalog = DataCatalog(data_libs=\[catalog_path\])

</div>

#### Preprocess functions when combining multiple files

In `xarray.open_mfdataset`, xarray allows for a **preprocess** function to be run before merging several netcdf files together. In hydroMT, some preprocess functions are available and can be passed through the options in the same way as any xr.open_mfdataset options. These preprocess functions are found at `hydromt.data_catalog.drivers.preprocessing.py`.

They include:

- **round_latlon**: round x and y dimensions to 5 decimals to avoid merging problems in xarray due to small differences in x, y values in the different netcdf files of the same data source.
- **to_datetimeindex**: Convert the time coordinate to a pandas DateTimeIndex.
- **remove_duplicates**: Remove duplicate time entries in the time coordinate.
- **harmonise_dims**: Harmonise the dimensions of all datasets to be the same before merging. This includes converting longitude from 0-360 to -180 to 180, having latitudes in N-S orientation, and convert time to datetimeindex format.

## Vector data (GeoDataFrame)

- `GeoPackage spatial vector data <GPKG_vector>`
- `Point vector from text delimited data <textdelimited_vector>`

<div id="vector_formats">

| Driver | File formats | Method(s) | Comments |
|----|----|----|----|
| `pyogrio <geodataframe.pyogrio_driver.PyogrioDriver>` | ESRI Shapefile, GeoPackage, GeoJSON, etc. | `~hydromt.readers.open_vector` | Point, Line and Polygon geometries. Uses `pyogrio.read_dataframe` |
| `geodataframe_table <geodataframe.table_driver.GeoDataFrameTableDriver>` | CSV, XY, PARQUET and EXCEL. | `~hydromt.readers.open_vector` | Point geometries only. |

</div>

### GeoPackage spatial vector data

Spatial vector data is parsed to a **GeoDataFrame** using the **vector** driver. For large spatial vector datasets we recommend the GeoPackage format as it includes a spatial index for fast filtering of the data based on spatial location. An example is shown below. Note that the `rename`, `unit_mult`, `unit_add` and `nodata` options refer to columns of the attribute table in case of a GeoDataFrame.

<div class="literalinclude" language="yaml">

../../assets/data_types/gpkg_geodataframe.yml

</div>

<div class="testsetup">

- 

from hydromt import DataCatalog

</div>

<div class="testcode" hide="">

geotiff

catalog_path = "docs/assets/data_types/gpkg_geodataframe.yml" catalog = DataCatalog(data_libs=\[catalog_path\])

</div>

### Point vector from text delimited data

Tabulated point vector data files can be parsed to a **GeoDataFrame** with the **vector_table** driver. This driver reads CSV (or similar delimited text files), EXCEL and XY (white-space delimited text file without headers) files. See this list of `dimension names <dimensions>` for recognized x and y column names.

A typical CSV point vector file is given below. A similar setup with headers can be used to read other text delimited files or excel files.

``` console
index, x, y, col1, col2
<ID1>, <X1>, <Y1>, <>, <>
<ID2>, <X2>, <Y2>, <>, <>
...
```

A XY files looks like the example below. As it does not contain headers or an index, the first column is assumed to contain the x-coordinates, the second column the y-coordinates and the index is a simple enumeration starting at 1. Any additional column is saved as column of the GeoDataFrame attribute table.

``` console
<X1>, <Y1>, <>, <>
<X2>, <Y2>, <>, <>
...
```

As the CRS of the coordinates cannot be inferred from the data it must be set in the data entry in the yaml file as shown in the example below.

<div class="literalinclude" language="yaml">

../../assets/data_types/csv_geodataframe.yml

</div>

<div class="testsetup">

- 

from hydromt import DataCatalog

</div>

<div class="testcode" hide="">

geotiff

catalog_path = "docs/assets/data_types/csv_geodataframe.yml" catalog = DataCatalog(data_libs=\[catalog_path\])

</div>

<div id="binary_vector">

HydroMT also supports reading and writing vector data in binary format. Currently only parquet is supported, but others could be added if desired. The structure of the files should be the same as the text format files described above but writing according to the parquet file spec. Since this is a binary format, not examples are provided, but for example pandas can write the same data structure to parquet as it can csv.

</div>

## Geospatial vector time-series (GeoDataset)

Geospatial vector time-series include time-series or n-dimensional data associated with a vector geometry dimension. Geometry can be of Point, Line or Polygon type.

- `Netcdf time-series dataset <NC_point>`
- `Vector with CSV time-series data <CSV_point>`

<div id="geo_formats">

| Driver | File formats | Method(s) | Comments |
|----|----|----|----|
| `geodataset_vector <geodataset.vector_driver.GeoDatasetVectorDriver>` | Combined vector location (e.g. CSV or GeoJSON) and text delimited time-series (e.g. CSV) data. | `~hydromt.readers.open_geodataset` | Uses `~hydromt.readers.open_vector`, `~hydromt.readers.open_timeseries_from_table` |
| `geodataset_xarray <geodataset.xarray_driver.GeoDatasetXarrayDriver>` | NetCDF and Zarr | `xarray.open_mfdataset`, `xarray.open_zarr` | required time and index [dimensions](#dimensions) and x- and y coordinates. |

</div>

### Netcdf time-series dataset

Netcdf and Zarr vector time-series data are parsed to **GeoDataset** using the **netcdf** and **zarr** drivers. A typical netcdf or zarr vector time-series dataset has the following structure with two ("time" and "index") dimensions, where the index dimension has x and y coordinates for points or geometry for polygons or lines. The time dimension and spatial coordinates are inferred from the data based on a list of recognized [dimensions](#dimensions) names.

``` console
Dimensions:      (stations: N, time: NT)
Coordinates:
  * time         (time)
  * stations     (stations)
    lon          (stations)
    lat          (stations)
Data variables:
    waterlevel   (time, stations)
```

To read a vector time-series dataset from a multiple file netcdf archive the following data entry is used, where the options are passed to `xarray.open_mfdataset` (or `xarray.open_zarr` for zarr data). In case the CRS cannot be inferred from the netcdf data it is defined here. The path to multiple files can be set using a sting glob or several keys, see description of the `uri` argument in the `yaml file description <data_yaml>`. In this example additional renaming and unit conversion preprocessing steps are added to unify the data to match the HydroMT naming and unit `terminology <terminology>`.

<div class="literalinclude" language="yaml">

../../assets/data_types/netcdf_geodataset.yml

</div>

<div class="testsetup">

- 

from hydromt import DataCatalog

</div>

<div class="testcode" hide="">

geotiff

catalog_path = "docs/assets/data_types/netcdf_geodataset.yml" catalog = DataCatalog(data_libs=\[catalog_path\])

</div>

### Vector with CSV time-series data

Vector with CSV time-series data where the geospatial vector geometries and time-series are saved in separate (text) files are parsed to **GeoDataset** using the **vector** driver. The GeoDataset must at least contain a location index with point geometries which is referred to by the `uri` argument. The path may refer to both GIS vector data such as GeoJSON with only Point geometries or tabulated point vector data such as csv files, see earlier examples for GeoDataFrame datasets. Finally, certain binary formats such as parquet are also supported. In addition a tabulated time-series text file can be passed to be used as a variable of the GeoDataset. This data is added by a second file which is referred to using the `data_path` option. The index of the time-series (in the columns header) and point locations must match.

<div class="literalinclude" language="yaml">

../../assets/data_types/csv_geodataset.yml

</div>

<div class="testsetup">

- 

from hydromt import DataCatalog

</div>

<div class="testcode" hide="">

geotiff

catalog_path = "docs/assets/data_types/csv_geodataset.yml" catalog = DataCatalog(data_libs=\[catalog_path\])

</div>

*Tabulated time series text file*

To read the time stamps, the `pandas.to_datetime` method is used.

``` console
time, <ID1>, <ID2>
<time1>, <value>, <value>
<time2>, <value>, <value>
...
```

## NetCDF time-series dataset (Dataset)

<div id="dataset_formats">

| Driver | File formats | Method(s) | Comments |
|----|----|----|----|
| `dataset_xarray <dataset.xarray_driver.DatasetXarrayDriver>` | NetCDF and Zarr | `xarray.open_mfdataset`, `xarray.open_zarr` | required time and index [dimensions](#dimensions). |

</div>

### Netcdf time-series dataset

NetCDF and zarr timeseries data are parsed to **Dataset** with the `~dataset.xarray_driver.DatasetXarrayDriver`. The resulting dataset is similar to the **GeoDataset** except that it lacks a spatial dimension.

<div class="literalinclude" language="yaml">

../../assets/data_types/netcdf_dataset.yml

</div>

<div class="testsetup">

- 

from hydromt import DataCatalog

</div>

<div class="testcode" hide="">

geotiff

catalog_path = "docs/assets/data_types/netcdf_dataset.yml" catalog = DataCatalog(data_libs=\[catalog_path\])

</div>

## 2D tabular data (DataFrame)

<div id="dataframe_formats">

| Driver | File formats | Method(s) | Comments |
|----|----|----|----|
| `pandas <dataframe.pandas_driver.PandasDriver>` | any file readable by pandas | `pandas.read_csv`, `pandas.read_excel`, `pandas.read_parquet`, `pandas.read_fwf` | Provide a sheet name or formatting through options |

</div>

> [!NOTE]
> Only 2-dimensional data tables are supported, please contact us through the issue list if you would like to have support for n-dimensional tables.

### Supported files

The DataFrameAdapter is quite flexible in supporting different types of tabular data formats. The driver allows for flexible reading of files: for example both mapping tables and time series data are supported. Please note that for timeseries, the `options` need to be used to set the correct column for indexing, and formatting and parsing of datetime-strings. See the relevant pandas function for which arguments can be used. Also note that the driver is not restricted to comma-separated files, as the delimiter can be given to the reader through the `options`.

<div class="literalinclude" language="yaml">

../../assets/data_types/csv_dataframe.yml

</div>

<div class="testsetup">

- 

from hydromt import DataCatalog

</div>

<div class="testcode" hide="">

geotiff

catalog_path = "docs/assets/data_types/csv_dataframe.yml" catalog = DataCatalog(data_libs=\[catalog_path\])

</div>

> [!NOTE]
> The yml-parser does not correctly parses <span class="title-ref">None</span> arguments. When this is required, the <span class="title-ref">null</span> argument should be used instead. This is parsed to the Python code as a <span class="title-ref">None</span>.
