---
title: HydroMT — Architecture and conventions
source: "https://deltares.github.io/hydromt/latest/dev/architecture/index.html"
reference: "HydroMT team. (n.d.). *Architecture and conventions*. HydroMT documentation. Retrieved May 10, 2026, from sources/tools/hydromt/hydromt-architecture.md"
topic: tools
type: documentation
promoted: 2026-05-10
tags:
  - tools
  - hydromt
accessed: 2026-05-10
---

# HydroMT — Architecture and conventions

_Merged from 3 upstream page(s) at commit `4b6ce64a`, pulled 2026-05-04. Page boundaries are preserved as `## ` headings._


---

## index

Explore HydroMT's core architecture components and their relationships. Click on each card to jump to the detailed documentation.


Model


ModelComponent


DataCatalog


DataSource


URIResolver


Driver


DataAdapter


Extensibility


Conventions


architecture conventions


---

## architecture

HydroMT provides a modular and extensible framework for building and managing environmental and hydrological models. Its architecture is organized around a few core abstractions that define how models, data, and workflows interact.

At its core, HydroMT connects model components and data sources through a consistent API. This allows flexible model construction, reproducibility, and interoperability across a wide range of models and data systems.

The diagram below summarizes the relationships between these components.

<img src="https://deltares.github.io/hydromt/latest/_images/hydromt_architecture.jpeg" width="800" alt="HydroMT main architecture diagram" />

## Model

The `Model` represents the complete model setup and workflow. It defines the model domain, manages all `ModelComponent`'s, and coordinates data loading and transformations through the `DataCatalog`.

Models can be created interactively, through Python scripts, or from workflow definitions for full reproducibility.

See also:

- `model_component_architecture`
- `API <model_api>`
- `Implement your own <custom_model>`

## ModelComponent

A `ModelComponent` represents a modular building block of a model, such as a specific model file or model data (for example, static or forcing data). Each component defines how its data are read and written, and can interact with other components during setup.

Components make models composable, flexible, and easy to extend without modifying the HydroMT core. The below components are included in HydroMT by default but you can also create your own custom components.

| Model Component | Description |
|----|----|
| `~hydromt.model.components.ConfigComponent` | Component for managing model configuration |
| `~hydromt.model.components.GeomsComponent` | Component for managing 1D vector data |
| `~hydromt.model.components.TablesComponent` | Component for managing non-geospatial data |
| `~hydromt.model.components.DatasetsComponent` | Component for managing non-geospatial data |
| `~hydromt.model.components.SpatialDatasetsComponent` | Component for managing geospatial data |
| `~hydromt.model.components.GridComponent` | Component for managing regular gridded data |
| `~hydromt.model.components.MeshComponent` | Component for managing unstructured grids |
| `~hydromt.model.components.VectorComponent` | Component for managing geospatial vector data |

See also:

- `model_architecture`
- `API <model_components_api>`
- `Implement your own <custom_component>`

## DataCatalog

The `DataCatalog` is HydroMT's core data access and management layer. It provides a structured way to describe where datasets are located, how they can be accessed, and how they should be represented once loaded into memory.

Within the HydroMT architecture, the `DataCatalog` connects the `Model` and its components to both internal and external data sources. It achieves this by maintaining a registry of `DataSource` objects, each of which encapsulates the specific logic for accessing that specific dataset. It does not load or process data itself; instead, it delegates those responsibilities to `DataSource` objects.

The `DataCatalog` comes with built-in support for several data catalogs:

- **Predefined catalogs** - distributed with HydroMT or plugins (e.g., `deltares_data`, `aws_data`) that provide standardized and ready-to-use datasets.
- **Custom YAML catalogs** - defined by users to reference their own data sources and file structures.

The `DataCatalog` can be used through the `Model` class but also standalone in your own Python scripts for data access and processing outside of a model context.

See also:

- `data_source_architecture`
- `uri_resolver_architecture`
- `driver_architecture`
- `data_adapter_architecture`
- `API <data_catalog_api>`
- `Pre-defined catalogs <existing_catalog>`
- `Create your own <custom_data_catalog>`

## DataSource

Each entry in the `DataCatalog` is represented by a `DataSource`. A DataSource encapsulates all the logic required to retrieve and standardize a specific dataset, based on the catalog's attributes and metadata. This abstraction separates data definition (in the catalog) from data access and transformation (in the source).

When a model requests data by one of the DataCatalog's API functions (e.g. `get_rasterdataset`, `get_geodataframe` etc.), the `DataCatalog` looks up the matching DataSource and - along with some pre-processing of function parameters - calls its `read_data` method.

From there, the DataSource handles the complete workflow:

1.  **Resolve URIs** - Using a `URIResolver`, it determines the actual file paths or URLs for the requested data. This supports flexible backends, such as local directories, cloud storage, or web APIs.
2.  **Load data** - The resolved URIs are passed to a `Driver`, which reads the raw data into a Python object like an `xarray.Dataset` or `geopandas.GeoDataFrame`.
3.  **Transform and standardize** - The loaded data are passed through a `DataAdapter`, which applies consistent transformations (e.g., variable renaming, temporal/spatial slicing, or unit conversion) to produce a uniform HydroMT representation.

This layered design allows the DataCatalog to stay lightweight and declarative — it only stores attributes and metadata (e.g., names, paths, data types, parameters), while the DataSource performs the operational work needed to translate those definitions into usable model inputs.

See also:

- `data_catalog_architecture`
- `uri_resolver_architecture`
- `driver_architecture`
- `data_adapter_architecture`
- `API <data_source_api>`
- `Implement your own <custom_data_source>`

## URIResolver

The `URIResolver` locates data by resolving catalog references (URIs) into actual file paths or service endpoints. It takes query parameters such as spatial bounds or time ranges and returns one or more resolved URIs that can be read by a `Driver`.

Custom resolvers can be added to support specific naming conventions, APIs, or cloud storage systems.

See also:

- `data_catalog_architecture`
- `driver_architecture`
- `API <uri_resolver_api>`
- `Implement your own <custom_resolver>`


hydromt.data_catalog.drivers


## Driver

The `Driver` reads resolved data into memory as Python objects such as `xarray.Dataset` or `geopandas.GeoDataFrame`. Each HydroMT data type (e.g., raster, vector) has a dedicated driver interface.

Drivers handle the complexity of I/O operations, including merging multiple files and managing filesystem access through `fsspec`. New drivers can be added through HydroMT's plugin system to support custom formats.

Existing drivers in HydroMT core include:

| Driver | Data type | File formats |
|----|----|----|
| `rasterio <raster.rasterio_driver.RasterioDriver>` | RasterDataset | GeoTIFF, ArcASCII, VRT, tile index, etc. |
| `raster_xarray <raster.raster_xarray_driver.RasterDatasetXarrayDriver>` | RasterDataset | NetCDF and Zarr |
| `pyogrio <geodataframe.pyogrio_driver.PyogrioDriver>` | GeoDataFrame | ESRI Shapefile, GeoPackage, GeoJSON, etc. |
| `geodataframe_table <geodataframe.table_driver.GeoDataFrameTableDriver>` | GeoDataFrame | CSV, XY, PARQUET and EXCEL |
| `geodataset_vector <geodataset.vector_driver.GeoDatasetVectorDriver>` | GeoDataset | Vector location (e.g. CSV or GeoJSON) & text delimited time-series (e.g. CSV). |
| `geodataset_xarray <geodataset.xarray_driver.GeoDatasetXarrayDriver>` | GeoDataset | NetCDF and Zarr |
| `dataset_xarray <dataset.xarray_driver.DatasetXarrayDriver>` | Dataset | NetCDF and Zarr |
| `pandas <dataframe.pandas_driver.PandasDriver>` | DataFrame | Any file readable by pandas (CSV, EXCEL, PARQUET, etc.) |

See also:

- `uri_resolver_architecture`
- `data_adapter_architecture`
- `API <driver_api>`
- `Implement your own <custom_driver>`

## DataAdapter

The `DataAdapter` standardizes and transforms data after it has been read by a `Driver`. It performs operations such as variable selection, temporal/spatial slicing, renaming, or unit conversion to ensure consistency across datasets and models. Note that in some cases, some of these steps may also be performed in the `Driver` itself if the underlying xarray, pandas, or geopandas library supports it directly in their read function.

Each data type (raster, vector, etc.) has its own adapter interface responsible for transforming data into HydroMT's standardized representation.

See also:

- `driver_architecture`
- `data_catalog_architecture`
- `API <data_adapter_api>`
- `Implement your own <custom_data_adapter>`

## Extensibility

HydroMT's architecture is fully extensible. Developers can subclass models, components, drivers, adapters, or resolvers and register them through `the plugin system <register_plugins>`. This is only required if you want to use/support your custom classes through the HydroMT data catalog or model configuration yml files. If you are using your custom classes directly in Python code, you can simply instantiate and use them without registering as a plugin. This flexibility allows HydroMT to support new model types, data formats, and workflows without changing the core library.

See also:

- `register_plugins`


---

## conventions

## General

- HydroMT follows consistent `naming and unit conventions <data_convention>` for frequently used variables to ensure clarity and interoperability.
- Code and documentation should adhere to Pythonic naming standards (PEP8), and public API elements should have clear docstrings following the NumPy style.

## Data

- HydroMT supports a range of `data types <data_types>`, which can be extended as needed.
- Input data is defined in a `data catalog <data_yaml>` and parsed by HydroMT to its associated Python type through the `DataSource` class.
- The goal of the `DataAdapter` is to standardize internal data representation — including variable names, units, and structure — with minimal preprocessing.
- When accessing data from the catalog via any `DataCatalog.get_<data_type>` method, the adapter ensures a consistent and unified format.
- The `get_*` methods also support arguments to define spatial or temporal subsets of datasets, ensuring efficient and targeted data access.

## Model Class

The HydroMT `Model class <model_api>` defines the structure and behavior of models within the framework. To implement HydroMT for a specific model kernel or software, create a subclass named `<Name>Model` (e.g., `SfincsModel` for SFINCS) with model-specific readers, writers, and setup methods.

- `Model components <model_components>` are data attributes that together define a model instance. Each component represents a specific aspect (file/data) of the model and is parsed into a Python class and data object with predefined specifications. For example, the `GridComponent` data represents static regular grids of a model as an `xarray.Dataset`.
- Most model components include both `read` and `write` methods for handling model-specific formats. These methods may include optional keyword arguments but **must not** require positional arguments. Model outputs can also be handled through components but should not implement a `write` method.
- The `Model` should contain high level methods that go from raw data into model inputs and parameters. These methods are decorated with `@hydromt_step` to indicate they are part of the model workflow. Each method should have a clear purpose, and documented inputs and outputs.
- All public model methods, defined with `hydromt_step` may only accept arguments of basic Python types: `str`, `int`, `float`, `bool`, `None`, `list`, or `dict`. This restriction ensures methods can be fully defined in a `workflow YAML file <model_workflow>`.
- Model methods access data through the `Model.data_catalog` attribute — an instance of `hydromt.DataCatalog`. Any argument ending with `_fn` (short for *filename*) refers either to a source in the data catalog or to a file path. Inside the method, data can be read with any `DataCatalog.get_<data_type>` method, which handles both catalog entries and local file paths transparently.
- The Model class defines two high-level methods — `~hydromt.Model.build` and `~hydromt.Model.update` — which are available across all model plugins and exposed via the CLI. Additional high-level methods may be added in future releases.
- A model subclass can be exposed as a HydroMT plugin by declaring a `hydromt.models` [entry point](https://packaging.python.org/en/latest/specifications/entry-points/) in the package's `pyproject.toml`. For detailed instructions, refer to the `register_plugins` section.
- We strongly recommend writing integration and unit tests for all model classes and components to ensure correctness and maintain stability across releases.
