---
title: HydroMT user guide — Migration guide
ingest-source: hydromt-user-guide
upstream-repo: Deltares/hydromt
upstream-commit: 4b6ce64a7ed082740766e466a1861860f5150322
pulled: 2026-05-04
section: migration-guide
doc-type: user-guide
license: MIT
sources:
  - https://deltares.github.io/hydromt/latest/user_guide/migration_guide/index.html
  - https://deltares.github.io/hydromt/latest/user_guide/migration_guide/data_catalog.html
  - https://deltares.github.io/hydromt/latest/user_guide/migration_guide/model_workflow.html
  - https://deltares.github.io/hydromt/latest/user_guide/migration_guide/python_updates.html
---

# HydroMT user guide — Migration guide

_Merged from 4 upstream page(s) at commit `4b6ce64a`, pulled 2026-05-04. Page boundaries are preserved as `## ` headings._


---

## index

> **Source:** [migration_guide/index.md](https://deltares.github.io/hydromt/latest/user_guide/migration_guide/index.html) · [upstream `.rst`](https://github.com/Deltares/hydromt/blob/4b6ce64a7ed082740766e466a1861860f5150322/docs/user_guide/migration_guide/index.rst)

HydroMT is now at version 1.0.0 `sparkle-fill;1.5em`

This update introduces several significant changes to the model structure, configuration files, and data handling. The architecture has been redesigned to enhance flexibility, usability, and performance. HydroMT is now organized into a component-based architecture to replace the previous inheritance model. Instead of all model functionality being defined in a single `Model` class, a model is now composed of modular `ModelComponent` classes such as `GridComponent`, `VectorComponent`, or `ConfigComponent`. Similarly, the `DataCatalog` has been redesigned along `Driver` and `DataAdapter` classes to allow for more flexible reading of different data formats and sources and the harmonization of data to standard HydroMT data structures.

This section describes how to migrate HydroMT models and configurations to the newer version of the HydroMT core. It includes detailed steps, references to updated data structures, and example migration workflows.

It is divided into four main parts:

<div class="grid" gutter="2">

2

<div class="grid-item-card" text-align="center" link="data_catalog" link-type="doc">

`file-moved;5em;sd-text-icon blue-icon` +++ Migrating the Data Catalog

</div>

<div class="grid-item-card" text-align="center" link="model_workflow" link-type="doc">

`file-moved;5em;sd-text-icon blue-icon` +++ Migrating the model workflow file

</div>

<div class="grid-item-card" text-align="center" link="python_updates" link-type="doc">

`terminal;5em;sd-text-icon blue-icon` +++ Updates for python users

</div>

<div class="grid-item-card" text-align="center" link="migration_plugin" link-type="ref">

`database;5em;sd-text-icon blue-icon` +++ Migrating your HydroMT plugin

</div>

</div>

Users migrating from earlier versions of HydroMT should follow these general steps:

1.  Update their HydroMT YAML workflow file to match the v1 schema. (This includes converting <span class="title-ref">.ini</span> and <span class="title-ref">.toml</span> files to YAML format.)
2.  Migrate their data catalog following the updated v1 format.

For python users, you will have to review your scripts and some of the functions calls as some methods have been moved or renamed. The main changes to the HydroMT python API are documented in the `Python updates in version 1 <python_updates_v1>`.

For plugin developers, we include a more detailed guide about the architecture and how to change your `Model` plugin to the new component-based structure in the `Migrating your HydroMT plugin <migration_plugin>` section.

This guide provides the main updates and steps to migrate to HydroMT v1. All detailed changes can be found in the `changelog <changelog>`

<div class="toctree" hidden="" maxdepth="1">

Migrating the Data Catalog \<data_catalog\> Migrating the model workflow file \<model_workflow\> Updates for python users \<python_updates\>

</div>


---

## data catalog

> **Source:** [migration_guide/data_catalog.md](https://deltares.github.io/hydromt/latest/user_guide/migration_guide/data_catalog.html) · [upstream `.rst`](https://github.com/Deltares/hydromt/blob/4b6ce64a7ed082740766e466a1861860f5150322/docs/user_guide/migration_guide/data_catalog.rst)

## Overview

The data catalog structure has been refactored to introduce a more modular design and clearer separation of responsibilities across several new classes (`DataSource`, `Driver`, `URIResolver`, and `DataAdapter`):

- `URIResolver` is in charge of parsing the path or URI of the file (e.g if you are using some keywords like `{year}` or `{month}` in your paths or if you want to read tiled raster)
- `Driver` is in charge of reading the data from the source (e.g reading a netcdf file from a local disk or from cloud)
- `DataAdapter` is in charge of harmonizing the data to standard HydroMT data structures (e.g. renaming variables, setting attributes, units conversion, etc.)
- `DataSource` is the main class that ties everything together and is used by the `DataCatalog` to load data.

Key format changes:

- `path` is renamed to `uri`
- **driver**: `filesystem` or `driver_kwargs` moved under `driver`. `driver` can be a single string or a dictionnary with name and options (passed to underlying function that will read the data, e.g. xarray.open_mfdataset, etc.).
- **data_adapter**:`unit_add`, `unit_mult`, `rename`, etc. moved under `data_adapter`
- **uri_resolver**: can be specified mostly in the case of tiled rasters to pass required options.
- **metadata**: `crs` and `nodata` are moved under `metadata` (renamed from `meta`)
- A single catalog entry can now reference multiple data variants or versions

See more information about the current format in the `data catalog documentation <get_data>`.

## How to upgrade

All existing pre-defined catalogs have been updated to the new format. For your own catalogs, you can upgrade easily with the HydroMT `check` command:

``` bash
hydromt check -d /path/to/data_catalog.yml --format v0 --upgrade -v
```


---

## model workflow

> **Source:** [migration_guide/model_workflow.md](https://deltares.github.io/hydromt/latest/user_guide/migration_guide/model_workflow.html) · [upstream `.rst`](https://github.com/Deltares/hydromt/blob/4b6ce64a7ed082740766e466a1861860f5150322/docs/user_guide/migration_guide/model_workflow.rst)

## Overview

The HydroMT model configuration format has been redesigned. The root YAML file now includes three main keys: `modeltype`, `global`, and `steps`.

- `modeltype` (optional): Defines which model plugin is being used (e.g. `model`, `wflow_sbm`, `wflow_sfincs` etc.).
- `global`: Defines model-wide configuration, including data catalog(s), or model components if using the core `model` plugin.
- `steps`: Replaces the old numbered dictionary format with a sequential list of function calls.

Some of the functions (component specific read and write) are now explicitly mapped to model or component methods using the <span class="title-ref">\<component\>.\<method\></span> syntax. This is for example the case for reading and writing of individual model components (e.g. `config.read`, `grid.write` etc.).

Additionally, to keep a consistent experience for our users we believe it is best to offer a single format for configuring HydroMT, as well as reducing the maintenance burden on our side. We have decided that **YAML** suits this use case the best. Therefore we have decided to deprecate other config formats for configuring HydroMT including **ini** and **toml** formats.

Finally, the command line interface no longer supports a <span class="title-ref">--region</span> argument. The `region` should be specified under the appropriate section of the YAML file depending on the model plugin you are using.

See more information about the current format in the `data catalog documentation <model_workflow>`.

## How to upgrade

There is no automatic way to convert old HydroMT model workflow files to the new format. This is mainly because the file is highly dependent on the specific model plugin and methods being used. Some plugins may have changed the names of their methods or some of the arguments. We therefore advise that you check the documentation of the specific model plugin you are using. Usually they will provide templates or examples of the new YAML format as well to limit manual effort.

In general, you can follow these steps:

1.  If you are using an `.ini` or `.toml` file to configure HydroMT, convert this to a YAML format. You can refer to the `model workflow documentation <model_workflow>` for examples of how to structure the YAML file.
2.  Update the structure of the YAML file to include the `modeltype`, `global`, and `steps` keys.
3.  For each step in the old format, convert it to the new list format under the `steps` key. Use the <span class="title-ref">\<component\>.\<method\></span> syntax for component-specific methods. Be careful of indents.
4.  Move the `region` specification from the command line to the YAML file under the appropriate section. This will depending on the model plugin you are using.
5.  Review the function names and arguments to ensure they match the new API.
6.  Test the updated workflow file with your HydroMT model to ensure it works as expected using the `check` command:

``` bash
hydromt check -i /path/to/your_workflow.yml -v
```


---

## python updates

> **Source:** [migration_guide/python_updates.md](https://deltares.github.io/hydromt/latest/user_guide/migration_guide/python_updates.html) · [upstream `.rst`](https://github.com/Deltares/hydromt/blob/4b6ce64a7ed082740766e466a1861860f5150322/docs/user_guide/migration_guide/python_updates.rst)

In HydroMT v1, the internal data structure and API were redesigned to improve consistency and maintainability. Most changes affect how model components (such as `config` and `grid`) are accessed and how model data is read and written.

## Component Class

### Rationale

Prior to v1, the `Model` class was the only real place where developers could modify the behavior of Core. All parts of a model were implemented as class properties forcing every model to use the same terminology. While this was enough for some users, it was too restrictive for others. For example, the SFINCS plugin uses multiple grids for its computation, which was not possible in the setup pre-v1. There was also a lot of code duplication for the use of several parts of a model such as `maps`, `forcing` and `states`. To offer users more modularity and flexibility, as well as improve maintainability, we have decided to move the core to a component based architecture rather than an inheritance based one.

The model components are now **dedicated classes** rather than raw data objects (e.g., `xarray`, `dict`, or `geopandas`). Each component can be accessed via the `Model` instance and exposes its underlying data through the `.data` property.

For users of HydroMT, the main change is that the `Model` class is now a composition of `ModelComponent` classes. The core `Model` class does not contain any components by default, but these can be added by the user upon instantiation or through the yaml configuration file. Plugins will define their implementation of the `Model` class with the components they need.

For a model which is instantiated with a `GridComponent` component called `grid`, where previously you would call `model.setup_grid(...)`, you now call `model.grid.create(...)`. To access the grid component data you call `model.grid.data` instead of `model.grid`.

In the core of HydroMT, the available components are:

<table style="width:99%;">
<colgroup>
<col style="width: 34%" />
<col style="width: 37%" />
<col style="width: 27%" />
</colgroup>
<thead>
<tr>
<th>v0.x</th>
<th>v1.x</th>
<th>Description</th>
</tr>
</thead>
<tbody>
<tr>
<td>Model.config</td>
<td><code class="interpreted-text" role="py:class">~hydromt.model.components.ConfigComponent</code></td>
<td>Component for managing model configuration</td>
</tr>
<tr>
<td>Model.geoms</td>
<td><code class="interpreted-text" role="py:class">~hydromt.model.components.GeomsComponent</code></td>
<td>Component for managing 1D vector data</td>
</tr>
<tr>
<td>Model.tables</td>
<td><code class="interpreted-text" role="py:class">~hydromt.model.components.TablesComponent</code></td>
<td>Component for managing non-geospatial data</td>
</tr>
<tr>
<td><ul>
<li></li>
</ul></td>
<td><code class="interpreted-text" role="py:class">~hydromt.model.components.DatasetsComponent</code></td>
<td>Component for managing non-geospatial data</td>
</tr>
<tr>
<td>Model.maps / Model.forcing / Model.results / Model.states</td>
<td><code class="interpreted-text" role="py:class">~hydromt.model.components.SpatialDatasetsComponent</code></td>
<td>Component for managing geospatial data</td>
</tr>
<tr>
<td>GridModel.grid</td>
<td><code class="interpreted-text" role="py:class">~hydromt.model.components.GridComponent</code></td>
<td>Component for managing regular gridded data</td>
</tr>
<tr>
<td>MeshModel.mesh</td>
<td><code class="interpreted-text" role="py:class">~hydromt.model.components.MeshComponent</code></td>
<td>Component for managing unstructured grids</td>
</tr>
<tr>
<td>VectorModel.vector</td>
<td><code class="interpreted-text" role="py:class">~hydromt.model.components.VectorComponent</code></td>
<td>Component for managing geospatial vector data</td>
</tr>
</tbody>
</table>

### Example: Accessing Component Data

Each component provides structured access to its data via the `.data` property.

``` python
from hydromt import ExampleModel

model = ExampleModel(root="path/to/model", mode="r")

# Access xarray.Dataset of static grid
grid = model.grid.data

# Access configuration dictionary
config = model.config.data
```

### Example: Writing Components

Read and write operations are now handled at the **component level**.

``` python
# Write configuration file
model.config.write()

# Write updated grid to disk
model.grid.write()
```

These changes provide a clearer and more modular interface, making it easier to manipulate model components independently.

## DataCatalog

### Rationale

The data catalog structure has been refactored to introduce a more modular design and clearer separation of responsibilities across several new classes (`DataSource`, `Driver`, `URIResolver`, and `DataAdapter`):

- `URIResolver` is in charge of parsing the path or URI of the file (e.g if you are using some keywords like `{year}` or `{month}` in your paths or if you want to read tiled raster)
- `Driver` is in charge of reading the data from the source (e.g reading a netcdf file from a local disk or from cloud)
- `DataAdapter` is in charge of harmonizing the data to standard HydroMT data structures (e.g. renaming variables, setting attributes, units conversion, etc.)
- `DataSource` is the main class that ties everything together and is used by the `DataCatalog` to load data.

Additionally, to be able to support different version of the same data set (for example, data sets that get re-released frequently with updated data) or to be able to take the same data set from multiple data sources (e.g. local if you have it but AWS if you don't) the data catalog has undergone some changes. Now since a catalog entry no longer uniquely identifies one source, (since it can refer to any of the variants mentioned above) it becomes insufficient to request a data source by string only. Since the dictionary interface in python makes it impossible to add additional arguments when requesting a data source, we created a more extensive API for this. In order to make sure users' code remains working consistently and have a clear upgrade path when adding new variants we have decided to remove the old dictionary like interface. Dictionary like features such as <span class="title-ref">catalog\['source'\]</span>, <span class="title-ref">catalog\['source'\] = data</span>, <span class="title-ref">source in catalog</span> etc. should be removed for v1. Equivalent interfaces have been provided for each operation, so it should be fairly simple. Below is a small table with their equivalent functions.

### How to upgrade

The high levels functions of `DataCatalog` and the different `get_data` methods have not changed apart that the `Driver` and `DataAdapter` options have to be specified differently in the catalog yaml file or explicitly under `source_kwargs` when calling the `get_data` methods. However, the dictionary-like interface has been removed.

``` python
import hydromt
catalog = hydromt.DataCatalog('path_to_your_catalog.yml')

# Old v0.x way (removed)
gdf = catalog.get_geodataframe('path/to/locations.csv', driver_kwargs={'sep': ';'})

# New v1.x way
gdf = catalog.get_geodataframe(
    'path/to/locations.csv',
    source_kwargs={
      'driver': {'name': 'pandas', 'options': {'sep': ';'}}
    }
)
```

Below is a table with the equivalent functions for the removed dictionary-like interface:

| v0.x                     | v1                                   |
|--------------------------|--------------------------------------|
| if 'name' in catalog:    | if catalog.contains_source('name'):  |
| catalog\['name'\]        | catalog.get_source('name')           |
| for x in catalog.keys(): | for x in catalog.get_source_names(): |
| catalog\['name'\] = data | catalog.set_source('name',data)      |

## API and supporting functions

HydroMT provides a series of supporting functions either for GIS operations, statistical methods, or file specific I/O operations. These functions have been moved and/or grouped under specific submodules to improve discoverability and maintainability.

Main changes includes:

| v0.x              | v1                                     |
|-------------------|----------------------------------------|
| hydromt.config    | Removed                                |
| hydromt.log       | Removed (private: hydromt.\_utils.log) |
| hydromt.flw       | hydromt.gis.flw                        |
| hydromt.gis_utils | hydromt.gis.utils                      |
| hydromt.raster    | hydromt.gis.raster                     |
| hydromt.vector    | hydromt.gis.vector                     |
| hydromt.gis_utils | hydromt.gis.utils                      |
| hydromt.io        | hydromt.readers and hydromt.writers    |
