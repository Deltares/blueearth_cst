---
title: HydroMT user guide — Data Catalog
source: "https://deltares.github.io/hydromt/latest/user_guide/data_catalog/data_overview.html"
reference: "HydroMT team. (n.d.). *Data Catalog*. HydroMT documentation. Retrieved May 10, 2026, from sources/tools/hydromt/hydromt-user-guide/04-data-catalog.md"
topic: tools
type: documentation
promoted: 2026-05-10
tags:
  - tools
  - hydromt
accessed: 2026-05-10
---

# 4 HydroMT user guide — Data Catalog

_Merged from 6 upstream page(s) at commit `4b6ce64a`, pulled 2026-05-04. Page boundaries are preserved as `## ` headings._

---

## 4.1 data overview

The best way to provide data to HydroMT is by using a **data catalog**. The goal of this data catalog is to provide simple and standardized access to (large) datasets. It supports many drivers to read different data formats and contains several pre-processing steps to unify the datasets. A data catalog can be initialized from one or more **yaml file(s)**, which contain all required information to read and pre-process a dataset, as well as meta data for reproducibility.

You can `explore and make use of pre-defined data catalogs <existing_catalog>` (primarily global data), `prepare your own data catalog <own_catalog>` (e.g. to include local data) or use a combination of both.

> [!TIP]
> If no yaml file is provided to the CLI build or update methods or to `~hydromt.data_catalog.DataCatalog`, HydroMT will use the data stored in the `artifact_data <existing_catalog>` which contains an extract of global data for a small region around the Piave river in Northern Italy.

> [!TIP]
> Tiles of tiled rasterdatasets which are described by a .vrt file can be cached locally. The requested data tiles will by default be stored to ~/.hydromt_data. To use this option from command line add `--cache` to the `hydromt build` or `hydromt update` commands. In Python the cache is a property of the DataCatalog and can be set at Initialization.

## 4.2 Using a data catalog

Command Line Interface (CLI)

When using the HydroMT command line interface (CLI), one can provide a data catalog by specifying the path to the yaml file with the `-d (--data)` option. Alternatively, you can also use names and versions of the `predefined data catalogs <existing_catalog>`. If no version is specified, the latest version available is used.

``` console
hydromt build MODEL -d /path/to/data-catalog.yml
```

Python API

Initialize a `~hydromt.data_catalog.DataCatalog` with references to user- or pre-defined data catalog yaml files

``` python
import hydromt
data_cat = hydromt.DataCatalog(data_libs=r'/path/to/data-catalog.yml')
```

You can find examples of how to read data from a catalog in Python in `this page <hydromt_data_read_python>`.

Pre-defined data catalogs  Preparing a data catalog  Supported data types  Data conventions  Cloud storage

---

## 4.3 data existing cat

This page contains a list of (global) datasets which can be used with various HydroMT models and workflows. Below are drop down lists with datasets per pre-defined data catalog for use with HydroMT. The summary per dataset contains links to the online source and available literature.

The `deltares_data` catalog is only available within the Deltares network. However a selection of this data for a the Piave basin (Northern Italy) is available online in the `artifact_data` archive and will be used if no data catalog is provided. Local or other datasets can also be included by extending the data catalog with new yaml `data catalog files <data_yaml>`. We plan to provide more data catalogs with open data sources in the (near) future. See the data catalog [changelog](https://github.com/Deltares/hydromt/blob/main/data/catalogs/changelog.rst) for recent updates on the pre-defined catalogs.

## 4.4 Using a predefined catalog

Command Line Interface (CLI)

To use a predefined catalog, you can specify the catalog name with the `-d` or `--data` option when running a HydroMT command. For example, to use the `deltares_data` catalog with the `hydromt build` command, you can run the following:

``` bash
hydromt build MODEL -d deltares_data ...
```

Alternatively, deltares_data can also be accessed with the `--dd` option:

``` bash
hydromt build MODEL --dd ...
```

You can specify a version of the catalog by adding the version number after the catalog name, e.g. `deltares_data=v1.0.0`.

``` bash
hydromt build MODEL -d deltares_data=v1.0.0 ...
```

Once you have set the data catalog you can specify the data source(s) for each method in the HydroMT `model workflow file <model_workflow>` as shown in the example below with the `setup_precip_forcing` method.

``` yaml
steps:
  - setup_region:
      region:
      bbox: [4.5, 51.5, 6.5, 53.5]

  - setup_maps_from_rasterdataset:
      raster_fn:
        source: 'eobs'
        version: 'v22.0e'
```

Python API

To use a predefined catalog in Python, you can specify the catalog name with the `data_libs` argument when initializing a `DataCatalog` class. You can specify a data catalog version by adding the version number after the catalog name. You can then get data from the catalog using the `DataCatalog.get_rasterdataset` or other :ref: `DataCatalog methods`.

``` python
from hydromt import DataCatalog
data_catalog = DataCatalog(data_libs=["deltares_data"])
# specify a data catalog version
data_catalog = DataCatalog(data_libs=["deltares_data=v1.0.0"])
# get data from the catalog
ds = data_catalog.get_rasterdataset("eobs") # get the most recently added
ds = data_catalog.get_rasterdataset("eobs", version="22.0e") # get a specific version
```

## 4.5 Available pre-defined data catalogs

### 4.5.1 Deltares data catalog

Data available for Deltares colleagues (p: drive). For non Deltares users, you can use it as inspiration to create your own. The catalog and it's different versions can be viewed here: <https://github.com/Deltares/hydromt/tree/main/data/catalogs/deltares_data>

Available data:

### 4.5.2 Artifact data catalog

Global data extract around the Piave basin in Northern Italy used for documentation, training and testing of HydroMT. The catalog and its different versions can be viewed here: <https://github.com/Deltares/hydromt/tree/main/data/catalogs/artifact_data>

Available data:

### 4.5.3 AWS data catalog

Data openly available in Amazon s3 bucket. The catalog and its different versions can be viewed here: <https://github.com/Deltares/hydromt/tree/main/data/catalogs/aws_data>

Available data:

### 4.5.4 GCS CMIP6 data catalog

CMIP6 dataset openly available and stored on a public Google Cloud Store. The catalog and its different versions can be viewed here: <https://github.com/Deltares/hydromt/tree/main/data/catalogs/gcs_cmip6_data>

Available data:

### 4.5.5 Earth Data Hub data catalog

Data stored in `Earth Data Hub ` (Destination Earth). In order to use this catalog, you need to setup credentials for accessing the data on Earth Data Hub. This includes creating an account on Earth Data Hub and setting up a .netrc file with your credentials. You can find more information on how to do this in the [Earth Data Hub documentation](https://earthdatahub.destine.eu/getting-started#configuring-netrc).

The catalog and its different versions can be viewed here: <https://github.com/Deltares/hydromt/tree/main/data/catalogs/earthdatahub_data>

Available data:

---

## 4.6 data prepare cat

**Steps in brief:**

1)  Have your (local) dataset ready in one of the supported format: `raster
    <raster_formats>`, `vector <vector_formats>`, `geospatial time-series
    <geo_formats>`, `time-series dataset <dataset_formats>` or `tabular data  <dataframe_formats>`.
2)  Create your own `yaml file <data_yaml>` with a reference to your prepared dataset following the HydroMT `data conventions <data_convention>`, see examples below.

A detailed description of the yaml file is given below. For more information see `~hydromt.data_catalog.DataCatalog.from_yml` and examples per `data type
<data_types>`

## 4.7 Data catalog yaml file

Each data source, is added to a data catalog yaml file with a user-defined name.

A blue print for a dataset called **my_dataset** is shown below. The `uri`, `data_type` and `driver` options are required and the `metadata` option with the shown keys is highly recommended. The `rename`, `nodata`, `unit_add` and `unit_mult` options are set per variable (or column for tables).

../../assets/example_catalog_simple.yml

-

from hydromt import DataCatalog

read_catalog

catalog = DataCatalog() catalog.from_yml("docs/assets/example_catalog_simple.yml")

The catalog file can start with an *optional* global **metadata** data section:

- **roots** (optional): root folders for all the data sources in the yaml file. If not provided the folder of where the yaml file is located will be used as root. This is used in combination with each data source **uri** argument to avoid repetition. The roots listed will be checked in the order they are provided. The first one to be found to exist will be used as the actual root. This should be used for cross platform and cross machine compatibility only, as can be seen above. Note that in the end only one of the roots will be used, so all data should still be located in the same folder tree.
- **version** (recommended): data catalog version
- **hydromt_version** (recommended): range of hydromt version that can read this catalog. Format should be acording to [PEP 440](https://peps.python.org/pep-0440/#version-specifiers).
- **category** (optional): used if all data source in catalog belong to the same category. Usual categories within HydroMT are *geography*, *topography*, *hydrography*, *meteo*, *landuse*, *ocean*, *socio-economic*, *observed data* but the user is free to define its own categories.

The following are **required arguments for each data source**:

- **data_type**: type of input data. Either *RasterDataset*, *GeoDataset*, *Dataset* *GeoDataFrame* or *DataFrame*.
- **driver**: data_type specific `Driver` to read a dataset. If the default settings of a driver are sufficient, then a string with the name of the driver is enough. Otherwise, a dictionary with the driver class properties can be used. Refer to the `Driver` `documentation <data_types>` to see which options are available.
- **uri**: URI pointing to where the data can be queried. Relative paths are combined with the global `root` option of the yaml file (if available) or the directory of the yaml file itself. To read multiple files in a single dataset (if supported by the driver) a string glob in the form of `"path/to/my/files/*.nc"` can be used. The filenames can be further specified with `{variable}`, `{year}` and `{month}` keys to limit which files are being read based on the get_data request in the form of `"path/to/my/files/{variable}_{year}_{month}.nc"`. Note that `month` is by default *not* zero-padded (e.g. January 2012 is stored as `"path/to/my/files/{variable}_2012_1.nc"`). Users can optionally add a formatting string to define how the key should be read. For example, in a path written as `"path/to/my/files/{variable}_{year}_{month:02d}.nc"`, the month always has two digits and is zero-padded for Jan-Sep (e.g. January 2012 is stored as `"path/to/my/files/{variable}_2012_01.nc"`).

A full list of **optional arguments for each data source** is given below

- **version** (recommended): data source version
- **metadata** (recommended): additional information on the dataset. In `SourceMetaData` there are many different metadata options available. Some metadata properties, like the `crs`, `nodata` or `temporal_extent` and `spatial_extent` can help HydroMT more efficiently read the data. Good meta data includes a *source_url*, *source_license*, *source_version*, *paper_ref*, *paper_doi*, *category*, etc. These are added to the data attributes. Usual categories within HydroMT are *geography*, *topography*, *hydrography*, *meteo*, *landuse*, *ocean*, *socio-economic*, *observed data* but the user is free to define its own categories.
- **data_adapter**: the data adapter harmonizes the data so that within HydroMT, there are strong conventions on for example variable naming, `HydroMT variable naming
  conventions <data_convention>`, dimension names. `recognized dimension names
  <dimensions>` and units.
- **placeholder** (optional): this argument can be used to generate multiple sources with a single entry in the data catalog file. If different files follow a logical nomenclature, multiple data sources can be defined by iterating through all possible combinations of the placeholders. The placeholder names should be given in the source name and the path and its values listed under the placeholder argument.

## 4.8 Data variants

Data variants are used to define multiple data sources with the same name, but from different providers or versions. Below, we show an example of a data catalog for a RasterDataset with multiple variants of the same data source (esa_worldcover), but this works identical for other data types. Here, the *metadata*, *data_type*, *driver* and are common arguments used for all variants. The variant arguments are used to extend and/or overwrite the common arguments, creating new sources.

``` yaml
esa_worldcover:
  metadata:
    crs: 4326
  data_type: RasterDataset
  driver:
    name: raster
    filesystem: local
  variants:
    - provider: local
      version: 2021
      uri: landuse/esa_worldcover_2021/esa-worldcover.vrt
    - provider: local
      version: 2020
      uri: landuse/esa_worldcover/esa-worldcover.vrt
    - provider: aws
      version: 2020
      uri: s3://esa-worldcover/v100/2020/ESA_WorldCover_10m_2020_v100_Map_AWS.vrt
      driver:
        name: raster
        filesystem: s3
```

Here is another example for meteo data (ERA5) in netcdf or zarr format:

../../assets/example_catalog.yml

To request a specific variant, the variant arguments can be used as keyword arguments to the `DataCatalog.get_rasterdataset` method, see code below. By default the newest version from the last provider is returned when requesting a data source with specific version or provider. Requesting a specific version from a HydroMT configuration file is also possible.

HydroMT worflow file (CLI)

Example 1:

``` yaml
steps:
  - setup_region:
      region:
      bbox: [4.5, 51.5, 6.5, 53.5]

  - setup_maps_from_rasterdataset:
      raster_fn:
        source: 'esa_worldcover'
        version: '2020'
```

Example 2:

``` yaml
steps:
  - setup_region:
      region:
      bbox: [4.5, 51.5, 6.5, 53.5]

  - setup_maps_from_rasterdataset:
      raster_fn:
        source: 'era5'
        provider: 'zarr'
```

Python API

Example 1:

``` python
from hydromt import DataCatalog
dc = DataCatalog().from_yml("data_catalog.yml")
# get the default version. This will return the latest (2020) version from the last
# provider (aws)
ds = dc.get_rasterdataset("esa_worldcover")
# get a 2020 version. This will return the 2020 version from the last provider (aws)
ds = dc.get_rasterdataset("esa_worldcover", version=2020)
# get a 2021 version. This will return the 2021 version from the local provider as
# this verion is not available from aws .
ds = dc.get_rasterdataset("esa_worldcover", version=2021)
# get the 2020 version from the local provider
ds = dc.get_rasterdataset("esa_worldcover", version=2020, provider="local")
```

Example 2:

``` python
from hydromt import DataCatalog
dc = DataCatalog().from_yml("data_catalog.yml")
# get the default provider. This will return the zarr provider
ds = dc.get_rasterdataset("era5")
# get the netcdf provider
ds = dc.get_rasterdataset("era5", provider="netcdf")
# get the zarr provider
ds = dc.get_rasterdataset("era5", provider="zarr")
```

> [!NOTE]
> The yml-parser does not correctly parses `None` arguments. When this is required, the `null` argument should be used instead. This is parsed to the Python code as a `None`.

---

## 4.9 data types

hydromt.data_catalog.drivers

# 4 Supported data types

HydroMT currently supports the following data types:

- `RasterDataset <RasterDataset>`: static and dynamic raster (or gridded) data
- `GeoDataFrame <GeoDataFrame>`: static vector data
- `GeoDataset <GeoDataset>`: dynamic vector data
- `Dataset <Dataset>`: non-spatial n-dimensional data
- `DataFrame <DataFrame>`: 2D tabular data

Internally the RasterDataset, GeoDataset, and Dataset are represented by `xarray.Dataset` objects, the GeoDataFrame by `geopandas.GeoDataFrame`, and the DataFrame by `pandas.DataFrame`. We use drivers, typically from third-party packages and sometimes wrapped in HydroMT functions, to parse many different file formats to this standardized internal data representation.

> [!NOTE]
> It is also possible to create your own driver. See at `Custom Driver<custom_driver>`

## 4.1 Recognized dimension names

- **time**: time or date stamp \["time"\].
- **x**: x coordinate \["x", "longitude", "lon", "long"\].
- **y**: y-coordinate \["y", "latitude", "lat"\].

## 4.2 Raster data (RasterDataset)

- `Single variable GeoTiff raster <GeoTiff>`
- `Multi variable Virtual Raster Tileset (VRT) <VRT>`
- `Tiled raster dataset <Tile>`
- `Netcdf raster dataset <NC_raster>`

| Driver | File formats | Method(s) | Comments |
|----|----|----|----|
| `rasterio <raster.rasterio_driver.RasterioDriver>` | GeoTIFF, ArcASCII, VRT, etc. (see [GDAL formats](http://www.gdal.org/formats_list.html)) | `~hydromt.readers.open_mfraster` | Based on `xarray.open_rasterio` and `rasterio.open` |
| `rasterio <raster.rasterio_driver.RasterioDriver>` with the `raster_tindex <hydromt.data_catalog.uri_resolvers.raster_tindex_resolver.RasterTindexResolver>` resolver | rasterio tile index file (see [gdaltindex](https://gdal.org/programs/gdaltindex.html)) | `~hydromt.readers.open_mfraster` | Options to merge tiles via `mosaic_kwargs` option. |
| `raster_xarray <raster.raster_xarray_driver.RasterDatasetXarrayDriver>` | NetCDF and Zarr | `xarray.open_mfdataset`, `xarray.open_zarr` | required y and x dimensions |

**Single variable GeoTiff raster**

Single raster files are parsed to a **RasterDataset** based on the **raster** driver. This driver supports 2D raster with "x" and "y" dimensions. A potential third dimension is called "dim0". The variable name is based on the filename, in this case `"GLOBCOVER_200901_200912_300x300m"`. The `chunks` key-word argument is passed to the underlying method and allows lazy reading of the data.

../../assets/data_types/single_variable_geotiff_raster.yml

-

from hydromt import DataCatalog

catalog_path = "docs/assets/data_types/single_variable_geotiff_raster.yml" catalog = DataCatalog(data_libs=\[catalog_path\])

### 4.2.1 Multi-variable Virtual Raster Tileset (VRT)

Multiple raster layers from different files are parsed using the **raster** driver. Each raster becomes a variable in the resulting RasterDataset based on its filename. The path to multiple files can be set using a glob string or several keys, see description of the `uri` argument in the `yaml file description <data_yaml>`. Note that the rasters should have identical grids.

Here multiple .vrt files (dir.vrt, bas.vrt, etc.) are combined based on their variable name into a single dataset with variables flwdir, basins, etc. Other multiple file raster datasets (e.g. GeoTIFF files) can be read in the same way. VRT files are useful for large raster datasets which are often tiled and can be combined using [gdalbuildvrt.](https://gdal.org/programs/gdalbuildvrt.html)

../../assets/data_types/vrt_raster_dataset.yml

catalog_path = "docs/assets/data_types/vrt_raster_dataset.yml" catalog = DataCatalog(data_libs=\[catalog_path\])

### 4.2.2 Tiled raster dataset

Tiled index datasets are parsed using the `raster_tindex <hydromt.data_catalog.uri_resolvers.raster_tindex_resolver.RasterTindexResolver>` `~hydromt.data_catalog.uri_resolvers.uri_resolver.URIResolver`. This data format is used to combine raster tiles with different CRS projections. A polygon vector file (e.g. GeoPackage) is used to make a tile index with the spatial footprints of each tile. When reading a spatial slice of this data the files with intersecting footprints will be merged together in the CRS of the most central tile. Use [gdaltindex](https://gdal.org/programs/gdaltindex.html) to build an excepted tile index file.

Here a GeoPackage with the tile index referring to individual GeoTiff raster tiles is used. The `mosaic_kwargs` are passed to `hydromt.gis.merge` to set the resampling method. The name of the column in the tile index attribute table `tileindex` which contains the raster tile file names is set in the `driver.options`

../../assets/data_types/tiled_raster_dataset.yml

catalog_path = "docs/assets/data_types/tiled_raster_dataset.yml" catalog = DataCatalog(data_libs=\[catalog_path\])

> [!NOTE]
> Tiled raster datasets are not read lazily as different tiles have to be merged together based on their values. For fast access to large raster datasets, other formats might be more suitable.

### 4.2.3 Netcdf raster dataset

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

../../assets/data_types/netcdf_raster_dataset.yml

catalog_path = "docs/assets/data_types/netcdf_raster_dataset.yml" catalog = DataCatalog(data_libs=\[catalog_path\])

#### 4.2.3.1 Preprocess functions when combining multiple files

In `xarray.open_mfdataset`, xarray allows for a **preprocess** function to be run before merging several netcdf files together. In hydroMT, some preprocess functions are available and can be passed through the options in the same way as any xr.open_mfdataset options. These preprocess functions are found at `hydromt.data_catalog.drivers.preprocessing.py`.

They include:

- **round_latlon**: round x and y dimensions to 5 decimals to avoid merging problems in xarray due to small differences in x, y values in the different netcdf files of the same data source.
- **to_datetimeindex**: Convert the time coordinate to a pandas DateTimeIndex.
- **remove_duplicates**: Remove duplicate time entries in the time coordinate.
- **harmonise_dims**: Harmonise the dimensions of all datasets to be the same before merging. This includes converting longitude from 0-360 to -180 to 180, having latitudes in N-S orientation, and convert time to datetimeindex format.

## 4.3 Vector data (GeoDataFrame)

- `GeoPackage spatial vector data <GPKG_vector>`
- `Point vector from text delimited data <textdelimited_vector>`

| Driver | File formats | Method(s) | Comments |
|----|----|----|----|
| `pyogrio <geodataframe.pyogrio_driver.PyogrioDriver>` | ESRI Shapefile, GeoPackage, GeoJSON, etc. | `~hydromt.readers.open_vector` | Point, Line and Polygon geometries. Uses `pyogrio.read_dataframe` |
| `geodataframe_table <geodataframe.table_driver.GeoDataFrameTableDriver>` | CSV, XY, PARQUET and EXCEL. | `~hydromt.readers.open_vector` | Point geometries only. |

### 4.3.1 GeoPackage spatial vector data

Spatial vector data is parsed to a **GeoDataFrame** using the **vector** driver. For large spatial vector datasets we recommend the GeoPackage format as it includes a spatial index for fast filtering of the data based on spatial location. An example is shown below. Note that the `rename`, `unit_mult`, `unit_add` and `nodata` options refer to columns of the attribute table in case of a GeoDataFrame.

../../assets/data_types/gpkg_geodataframe.yml

-

from hydromt import DataCatalog

catalog_path = "docs/assets/data_types/gpkg_geodataframe.yml" catalog = DataCatalog(data_libs=\[catalog_path\])

### 4.3.2 Point vector from text delimited data

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

../../assets/data_types/csv_geodataframe.yml

-

from hydromt import DataCatalog

catalog_path = "docs/assets/data_types/csv_geodataframe.yml" catalog = DataCatalog(data_libs=\[catalog_path\])

HydroMT also supports reading and writing vector data in binary format. Currently only parquet is supported, but others could be added if desired. The structure of the files should be the same as the text format files described above but writing according to the parquet file spec. Since this is a binary format, not examples are provided, but for example pandas can write the same data structure to parquet as it can csv.

## 4.4 Geospatial vector time-series (GeoDataset)

Geospatial vector time-series include time-series or n-dimensional data associated with a vector geometry dimension. Geometry can be of Point, Line or Polygon type.

- `Netcdf time-series dataset <NC_point>`
- `Vector with CSV time-series data <CSV_point>`

| Driver | File formats | Method(s) | Comments |
|----|----|----|----|
| `geodataset_vector <geodataset.vector_driver.GeoDatasetVectorDriver>` | Combined vector location (e.g. CSV or GeoJSON) and text delimited time-series (e.g. CSV) data. | `~hydromt.readers.open_geodataset` | Uses `~hydromt.readers.open_vector`, `~hydromt.readers.open_timeseries_from_table` |
| `geodataset_xarray <geodataset.xarray_driver.GeoDatasetXarrayDriver>` | NetCDF and Zarr | `xarray.open_mfdataset`, `xarray.open_zarr` | required time and index [dimensions](#dimensions) and x- and y coordinates. |

### 4.4.1 Netcdf time-series dataset

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

../../assets/data_types/netcdf_geodataset.yml

-

from hydromt import DataCatalog

catalog_path = "docs/assets/data_types/netcdf_geodataset.yml" catalog = DataCatalog(data_libs=\[catalog_path\])

### 4.4.2 Vector with CSV time-series data

Vector with CSV time-series data where the geospatial vector geometries and time-series are saved in separate (text) files are parsed to **GeoDataset** using the **vector** driver. The GeoDataset must at least contain a location index with point geometries which is referred to by the `uri` argument. The path may refer to both GIS vector data such as GeoJSON with only Point geometries or tabulated point vector data such as csv files, see earlier examples for GeoDataFrame datasets. Finally, certain binary formats such as parquet are also supported. In addition a tabulated time-series text file can be passed to be used as a variable of the GeoDataset. This data is added by a second file which is referred to using the `data_path` option. The index of the time-series (in the columns header) and point locations must match.

../../assets/data_types/csv_geodataset.yml

-

from hydromt import DataCatalog

catalog_path = "docs/assets/data_types/csv_geodataset.yml" catalog = DataCatalog(data_libs=\[catalog_path\])

*Tabulated time series text file*

To read the time stamps, the `pandas.to_datetime` method is used.

``` console
time, <ID1>, <ID2>
<time1>, <value>, <value>
<time2>, <value>, <value>
...
```

## 4.5 NetCDF time-series dataset (Dataset)

| Driver | File formats | Method(s) | Comments |
|----|----|----|----|
| `dataset_xarray <dataset.xarray_driver.DatasetXarrayDriver>` | NetCDF and Zarr | `xarray.open_mfdataset`, `xarray.open_zarr` | required time and index [dimensions](#dimensions). |

### 4.5.1 Netcdf time-series dataset

NetCDF and zarr timeseries data are parsed to **Dataset** with the `~dataset.xarray_driver.DatasetXarrayDriver`. The resulting dataset is similar to the **GeoDataset** except that it lacks a spatial dimension.

../../assets/data_types/netcdf_dataset.yml

-

from hydromt import DataCatalog

catalog_path = "docs/assets/data_types/netcdf_dataset.yml" catalog = DataCatalog(data_libs=\[catalog_path\])

## 4.6 2D tabular data (DataFrame)

| Driver | File formats | Method(s) | Comments |
|----|----|----|----|
| `pandas <dataframe.pandas_driver.PandasDriver>` | any file readable by pandas | `pandas.read_csv`, `pandas.read_excel`, `pandas.read_parquet`, `pandas.read_fwf` | Provide a sheet name or formatting through options |

> [!NOTE]
> Only 2-dimensional data tables are supported, please contact us through the issue list if you would like to have support for n-dimensional tables.

### 4.6.1 Supported files

The DataFrameAdapter is quite flexible in supporting different types of tabular data formats. The driver allows for flexible reading of files: for example both mapping tables and time series data are supported. Please note that for timeseries, the `options` need to be used to set the correct column for indexing, and formatting and parsing of datetime-strings. See the relevant pandas function for which arguments can be used. Also note that the driver is not restricted to comma-separated files, as the delimiter can be given to the reader through the `options`.

../../assets/data_types/csv_dataframe.yml

-

from hydromt import DataCatalog

catalog_path = "docs/assets/data_types/csv_dataframe.yml" catalog = DataCatalog(data_libs=\[catalog_path\])

> [!NOTE]
> The yml-parser does not correctly parses `None` arguments. When this is required, the `null` argument should be used instead. This is parsed to the Python code as a `None`.

---

## 4.7 data conventions

Names and units mentioned here are mandatory in order for the input data to be processed correctly and produce the right derived data. It is possible to use the `rename` option in the `data catalog yaml file <data_yaml>` so that data variables have hydroMT-compatible names and the `unit_mult` and `unit_add` options to convert units where necessary. This section lists the different variable naming and unit conventions of HydroMT by types. A list of recognized `dimension names <dimensions>` is found here.

## 4.8 Topography

| Name | Explanation | Unit |
|----|----|----|
| elevtn | altitude | \[m\] |
| mdt | mean dynamic topography | \[m\] |
| flwdir | flow direction. Format supported are ArcGIS D8, LDD, NEXTXY. The format is inferred from the data. |  |
| uparea | upstream area | \[km2\] |
| lndslp | slope | \[m/m\] |
| strord | Stralher streamorder | \[-\] |
| basins | basins ID mapping | \[-\] |

## 4.9 Surface waters

### 4.9.1 Rivers

| Name   | Explanation                             | Unit     |
|--------|-----------------------------------------|----------|
| rivlen | river length                            | \[m\]    |
| rivslp | river slope                             | \[m/m\]  |
| rivwth | river width                             | \[m\]    |
| rivmsk | mask of river cells (for raster models) | \[bool\] |

### 4.9.2 Reservoirs / lakes

| Name | Explanation | Unit |
|----|----|----|
| waterbody_id | reservoir/lake ID | \[-\] |
| Hylak_id | ID from the HydroLAKES database (to connect to the hydroengine library) | \[-\] |
| Area_avg | average waterbody area | \[m2\] |
| Vol_avg | average waterbody volume | \[m3\] |
| Depth_avg | average waterbody depth | \[m\] |
| Dis_avg | average waterbody discharge | \[m3/s\] |
| xout | longitude of the waterbody outlet | \[-\] |
| yout | latitude of the waterbody outlet | \[-\] |
| Capacity_max | maximum reservoir capacity volume | \[m3\] |
| Capacity_norm | normal/average reservoir capacity volume | \[m3\] |
| Capacity_min | minimum reservoir capacity volume | \[m3\] |
| Dam_height | height of the dam | \[m\] |

### 4.9.3 Glaciers

| Name      | Explanation                        | Unit  |
|-----------|------------------------------------|-------|
| simple_id | glacier ID in the current database | \[-\] |

## 4.10 Landuse and landcover

| Name    | Explanation            | Unit  |
|---------|------------------------|-------|
| landuse | landuse classification | \[-\] |
| LAI     | Leaf Area Index        | \[-\] |

## 4.11 Soil

| Name | Explanation | Unit |
|----|----|----|
| bd_sl\* | bulk density of the different soil layers (1 to 7 in soilgridsv2017) | \[g cm-3\] |
| clyppt_sl\* | clay content of the different soil layers (1 to 7 in soilgridsv2017) | \[%\] |
| oc_sl\* | organic carbon content of the different soil layers (1 to 7 in soilgridsv2017) | \[%\] |
| ph_sl\* | pH of the different soil layers (1 to 7 in soilgridsv2017) | \[-\] |
| sltppt_sl\* | silt content of the different soil layers (1 to 7 in soilgridsv2017) | \[%\] |
| sndppt_sl\* | sand content of the different soil layers (1 to 7 in soilgridsv2017) | \[%\] |
| soilthickness | soil thickness | \[cm\] |
| tax_usda | USDA soil classification | \[-\] |

## 4.12 Meteorology

| Name      | Explanation                       | Unit      |
|-----------|-----------------------------------|-----------|
| precip    | precipitation (rainfall+snowfall) | \[mm\]    |
| temp      | average temperature               | \[oC\]    |
| temp_min  | minimum temperature               | \[oC\]    |
| temp_max  | maximum temperature               | \[oC\]    |
| temp_dew  | dewpoint temperature              | \[oC\]    |
| press_msl | atmospheric pressure              | \[hPa\]   |
| kin       | shortwave incoming radiation      | \[W m-2\] |
| kout      | TOA incident solar radiation      | \[W m-2\] |
| ssr       | surface net solar radiation       | \[W m-2\] |
| wind10_u  | 10m wind U-component              | \[m s-1\] |
| wind10_v  | 10m wind V-component              | \[m s-1\] |
| tcc       | total cloud cover                 | \[-\]     |

## 4.13 Hydrology

| Name | Explanation | Unit |
|----|----|----|
| run | surface water runoff (overland flow + river discharge) | \[m3/s\] |
| vol | water volumes | \[m3\] |
| infilt | water infiltration in the soil | \[m3/s\] |
| runPav | excess infiltration runoff on paved areas | \[m3/s\] |
| runUnp | excess infiltration runoff on unpaved areas | \[m3/s\] |
| inwater | sum of all fluxes entering/leaving the surface waters (precipitation, evaporation, infiltration...) | \[m3/s\] |
| inwaterInternal | sum of all fluxes between the land and river surface waters (part of inwater) | \[m3/s\] |

---

## 4.14 data cloud storage

HydroMT can read data directly from cloud object stores — **Amazon S3**, **Google Cloud Storage**, and **Microsoft Azure Blob Storage / ADLS Gen2** — without downloading files manually. All cloud access is built on [fsspec](https://filesystem-spec.readthedocs.io), so any protocol that fsspec supports can be used.

Install the optional `io` dependencies to enable cloud storage access:

``` bash
pip install "hydromt[io]"
```

This installs `s3fs` (AWS), `gcsfs` (GCS), `adlfs` (Azure), and `azure-identity` / `azure-ai-ml` (Azure authentication and AzureML datastore support).

## 4.15 Quick comparison

| Provider | fsspec protocol | Required package | Example URI |
|----|----|----|----|
| Amazon S3 | `s3` | `s3fs` | `s3://bucket/path/file.tif` |
| Google Cloud Storage | `gcs` | `gcsfs` | `gs://bucket/path/file.zarr` |
| Azure Blob / ADLS Gen2 | `abfs` | `adlfs` | `abfs://container/path/file.nc` |

## 4.16 Simple cloud access (any provider)

The simplest way to read from any cloud store is to set the **filesystem** on the driver — exactly as you would for a local file, but with a cloud protocol. This works identically for S3, GCS, and Azure and uses the default `~hydromt.data_catalog.uri_resolvers.ConventionResolver`.

**AWS S3 (anonymous)**

``` yaml
esa_worldcover:
  data_type: RasterDataset
  uri: s3://esa-worldcover/v100/2020/ESA_WorldCover_10m_2020_v100_Map_AWS.vrt
  driver:
    name: rasterio
    filesystem:
      protocol: s3
      anon: true
```

**Google Cloud Storage**

``` yaml
cmip6_historical:
  data_type: RasterDataset
  uri: gs://cmip6/CMIP6/CMIP/MPI-ESM1-2-HR/historical/r1i1p1f1/day/tas/*/*
  driver:
    name: raster_xarray
    filesystem:
      protocol: gcs
```

**Azure Blob Storage (anonymous)**

``` yaml
noaa_isd:
  data_type: DataFrame
  uri: abfs://isdweatherdatacontainer/ISDWeather/year=2020/month=1/*.parquet
  driver:
    name: pandas
    filesystem:
      protocol: abfs
      account_name: azureopendatastorage
      anon: true
```

In all three cases, HydroMT:

1.  Creates an fsspec filesystem from the `filesystem:` block (e.g. `adlfs.AzureBlobFileSystem(account_name=..., anon=True)`).
2.  Passes the URI to `~hydromt.data_catalog.uri_resolvers.ConventionResolver`, which calls `fs.glob()` to resolve wildcards.
3.  Hands the resolved URIs to the driver for reading.

The Convention Resolver is **cloud-agnostic** — it doesn't know or care which provider is behind the filesystem. All provider-specific logic lives in the fsspec implementation (`s3fs`, `gcsfs`, `adlfs`).

**When to use this approach:**

- Public / anonymous containers
- Containers where you manage credentials via environment variables that the fsspec implementation picks up automatically
- Simple `abfs://` URIs without SAS tokens, HTTPS blob URLs, or AzureML datastore URIs

## 4.17 Azure Blob Resolver

For Azure-specific scenarios that go beyond what the generic approach offers, HydroMT provides a dedicated `~hydromt.data_catalog.uri_resolvers.AzureBlobResolver`.

Use it when you need any of:

- **URI normalisation** — `https://<account>.blob.core.windows.net/…` or `azureml://subscriptions/…/datastores/…/paths/…` URIs, which are automatically converted to `abfs://` internally.
- **Automatic SAS token fetching** — e.g. from the [Planetary Computer](https://planetarycomputer.microsoft.com) SAS API.
- **Azure credential chain** — the resolver walks explicit options → environment variables → `DefaultAzureCredential` (Managed Identity, Azure CLI, VS Code login, service principals) without manual configuration.
- **HTTPS output for GDAL / rasterio** — when a SAS token is available, `abfs://` URIs are transparently converted to signed HTTPS blob URLs that rasterio/GDAL can open directly (since those libraries do not understand the `abfs://` scheme).

See `choosing_resolver` for guidance on when the Convention Resolver is sufficient instead.

### 4.17.1 Configuration

Enable the resolver by adding `uri_resolver: name: azure_blob` to the data source. Options are passed under `uri_resolver.options`.

``` yaml
my_dataset:
  data_type: RasterDataset
  uri: abfs://container/path/to/data.tif
  driver:
    name: rasterio
  uri_resolver:
    name: azure_blob
    options:
      account_name: mystorageaccount
      # ... authentication options, see below
```

> [!NOTE]
> When a `uri_resolver` is specified, the resolver manages filesystem creation. You do **not** need to set `filesystem:` on the driver — the resolver will create an `abfs` filesystem and propagate it to the driver automatically.

### 4.17.2 Supported URI styles

| Style | Example |
|----|----|
| ADLS Gen2 (native) | `abfs://mycontainer/path/to/data.tif` |
| HTTPS Blob endpoint | `https://myaccount.blob.core.windows.net/mycontainer/path/data.tif` |
| AzureML datastore | `azureml://subscriptions/<sub>/resourcegroups/<rg>/workspaces/<ws>/datastores/<ds>/paths/<path>` |

All styles are normalised to `abfs://` internally. HTTPS blob URLs have their `account_name` extracted from the URL automatically, so you do not need to specify it again. AzureML URIs require the `azure-ai-ml` package and resolve the datastore to its underlying storage account / container via the AzureML SDK.

### 4.17.3 Authentication options

Credentials are resolved in the following order of precedence:

1.  **Explicit values** in `uri_resolver.options`:

    | Option | Description |
    |----|----|
    | `account_name` | Storage account name (required for most methods) |
    | `account_key` | Storage account access key |
    | `sas_token` | Shared Access Signature token string |
    | `connection_string` | Full connection string (takes highest precedence) |
    | `client_id`, `client_secret`, `tenant_id` | Service principal credentials |
    | `anon: true` | Anonymous access (public containers, skips all credential resolution) |
    | `sas_token_url` | URL returning JSON with a `"token"` key; a fresh SAS token is fetched automatically before each `resolve()` call |

2.  **Environment variables** (recognised by `adlfs` / `azure-storage-blob`):
    - `AZURE_STORAGE_ACCOUNT_NAME`
    - `AZURE_STORAGE_ACCOUNT_KEY`
    - `AZURE_STORAGE_SAS_TOKEN`
    - `AZURE_STORAGE_CONNECTION_STRING`
3.  **DefaultAzureCredential** (from `azure-identity`): covers Managed Identity, Azure CLI login, VS Code login, environment-variable service principals, and more. Activated automatically when `account_name` is set but no explicit key/token is provided.

### 4.17.4 Time-templated URIs

The resolver supports the same placeholder expansion as the Convention Resolver: `{year}`, `{month}`, `{day}`, and `{variable}`.

``` yaml
rainfall:
  data_type: RasterDataset
  uri: abfs://hydrodata/rainfall/{year}/{month}/precip.nc
  driver:
    name: raster_xarray
  uri_resolver:
    name: azure_blob
    options:
      account_name: mystorageaccount
      account_key: "..."
```

### 4.17.5 ABFS to HTTPS conversion

Internally, `AzureBlobResolver` normalises every URI to the `abfs://` scheme so that fsspec-based drivers (e.g. xarray with zarr) can open data via `adlfs` directly.

However, **rasterio and GDAL** do not understand the `abfs://` scheme. When a SAS token is available (either supplied explicitly or fetched from a `sas_token_url`), the resolver automatically rewrites the resolved `abfs://` URIs to HTTPS blob URLs of the form:

    https://<account>.blob.core.windows.net/<container>/<path>?<sas_token>

This allows rasterio / GDAL to open the data through their built-in HTTPS (`/vsicurl/`) handler without any additional configuration. The conversion only takes place when **both** an `account_name` **and** a `sas_token` are available; otherwise the `abfs://` URIs are returned as-is for fsspec-based drivers.

### 4.17.6 Examples

**Anonymous public container**

``` yaml
noaa_isd:
  data_type: DataFrame
  uri: abfs://isdweatherdatacontainer/ISDWeather/year=2020/month=1/*.parquet
  driver:
    name: pandas
  uri_resolver:
    name: azure_blob
    options:
      account_name: azureopendatastorage
      anon: true
```

**Planetary Computer with automatic SAS token fetching**

``` yaml
esa_worldcover:
  data_type: RasterDataset
  uri: abfs://esa-worldcover/v200/2021/map/ESA_WorldCover_10m_2021_v200_N51E003_Map.tif
  driver:
    name: rasterio
  uri_resolver:
    name: azure_blob
    options:
      account_name: ai4edataeuwest
      sas_token_url: https://planetarycomputer.microsoft.com/api/sas/v1/token/esa-worldcover
```

**AzureML datastore URI**

``` yaml
flood_hazard_maps:
  data_type: RasterDataset
  uri: azureml://subscriptions/00000000-aaaa-bbbb-cccc-123456789abc/resourcegroups/rg-my-project/workspaces/mlw-my-workspace/datastores/project_datastore/paths/hazard/flood_depth_100yr.tif
  driver:
    name: raster_xarray
  uri_resolver:
    name: azure_blob
```

The AzureML SDK will look up the datastore, extract the underlying storage account and container, and build an `abfs://` path automatically. Authentication is handled via `DefaultAzureCredential`.

For examples with explicit SAS tokens, see `azure_sas_quickstart`.

## 4.18 Choosing between resolvers

Start with the Convention Resolver for simple, public, or environment-variable-authenticated `abfs://` access. Switch to the Azure Blob Resolver the moment you need SAS tokens, non-`abfs://` URIs, or GDAL/rasterio compatibility with private data.

| Scenario | Convention Resolver | Azure Blob Resolver |
|----|----|----|
| Public `abfs://` container (anonymous) | Yes | Yes |
| Private `abfs://` container (key/SAS via env vars) | Yes | Yes |
| HTTPS blob URLs (`https://<acct>.blob.core.windows.net/…`) | No | **Yes** |
| AzureML datastore URIs (`azureml://…`) | No | **Yes** |
| Automatic SAS token fetching from a token API | No | **Yes** |
| Azure credential chain (DefaultAzureCredential) | No | **Yes** |
| rasterio / GDAL needs signed HTTPS URLs | No | **Yes** (automatic) |
| S3 or GCS data | **Yes** | No (Azure only) |

## 4.19 Step-by-step: accessing private Azure Blob Storage with a SAS token

This section walks through the steps to access data stored in a **private** Azure storage account. The workflow is: sign in to Azure, generate a SAS token with the required permissions, and reference that token in your data catalog.

### 4.19.1 Generate a SAS token

A SAS (Shared Access Signature) token grants time-limited, scoped access to a container or blob. You can create one from the Azure CLI, the Azure Portal, or Azure Storage Explorer.

**Azure Portal**

1.  Navigate to **Storage accounts** → your storage account.
2.  Open **Containers** → select the container → **Shared access tokens** (or use **Shared access signature** from the storage account menu for account-level tokens).
3.  Set **Allowed permissions** to *Read* and *List*, choose an expiry date/time, and click **Generate SAS token and URL**.
4.  Copy the **SAS token** value (starts with `sp=` or `sv=`).

### 4.19.2 Add the SAS token to your data catalog

Open your data catalog YAML file and add a source that uses the `azure_blob` resolver. Below are examples for each supported URI style.

**\`\`abfs://\`\` URI**

``` yaml
my_dataset:
  data_type: RasterDataset
  uri: abfs://<container>/<path-to-data>/*.tif
  driver:
    name: rasterio
  uri_resolver:
    name: azure_blob
    options:
      account_name: <storage-account-name>
      sas_token: <paste-your-sas-token-here>
```

**AzureML datastore URI**

``` yaml
my_dataset:
  data_type: RasterDataset
  uri: azureml://subscriptions/<sub-id>/resourcegroups/<rg>/workspaces/<ws>/datastores/<datastore>/paths/<path>.tif
  driver:
    name: rasterio
  uri_resolver:
    name: azure_blob
    options:
      sas_token: <paste-your-sas-token-here>
```

**HTTPS blob URL**

``` yaml
my_dataset:
  data_type: RasterDataset
  uri: https://<account>.blob.core.windows.net/<container>/<path>.tif
  driver:
    name: rasterio
  uri_resolver:
    name: azure_blob
    options:
      sas_token: "<paste-your-sas-token-here>"
```

With HTTPS blob URLs the `account_name` is extracted from the URL automatically, so you do not need to specify it separately. You can also append the SAS token directly to the URL as a query string (`https://…/<path>.tif?sp=rl&st=…`) and omit the `sas_token` option.

> [!TIP]
> To keep credentials out of version control, set the `AZURE_STORAGE_SAS_TOKEN` environment variable instead. The resolver picks it up automatically when no explicit `sas_token` is provided:
>
> ``` bash
> export AZURE_STORAGE_SAS_TOKEN="sp=rl&st=2026-03-30T09:00:00Z&se=..."
> ```
>
> On Windows (PowerShell):
>
> ``` powershell
> $env:AZURE_STORAGE_SAS_TOKEN = "sp=rl&st=2026-03-30T09:00:00Z&se=..."
> ```

### 4.19.3 Verify access

Test that HydroMT can read the data:

``` python
import hydromt

cat = hydromt.DataCatalog("path/to/my_catalog.yml")
ds = cat.get_rasterdataset("my_dataset")
print(ds)
```
