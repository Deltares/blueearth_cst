---
title: HydroMT user guide — Overview
ingest-source: hydromt-user-guide
upstream-repo: Deltares/hydromt
upstream-commit: 4b6ce64a7ed082740766e466a1861860f5150322
pulled: 2026-05-04
section: overview
doc-type: user-guide
license: MIT
sources:
  - https://deltares.github.io/hydromt/latest/user_guide/intro.html
  - https://deltares.github.io/hydromt/latest/user_guide/overview/index.html
  - https://deltares.github.io/hydromt/latest/user_guide/overview/hydromt_intro.html
  - https://deltares.github.io/hydromt/latest/user_guide/overview/hydromt_cli.html
  - https://deltares.github.io/hydromt/latest/user_guide/overview/hydromt_python.html
---

# HydroMT user guide — Overview

_Merged from 5 upstream page(s) at commit `4b6ce64a`, pulled 2026-05-04. Page boundaries are preserved as `## ` headings._


---

## intro

> **Source:** [intro.md](https://deltares.github.io/hydromt/latest/user_guide/intro.html) · [upstream `.rst`](https://github.com/Deltares/hydromt/blob/4b6ce64a7ed082740766e466a1861860f5150322/docs/user_guide/intro.rst)

In this user guide, you will find detailed descriptions and examples that describe many common tasks that you can accomplish with HydroMT. If you want to develop your own plugin, please refer to our `developer guide <intro_developer_guide>`.

<div class="grid" gutter="1">

2

<div class="grid-item-card" text-align="center" link="overview_index" link-type="ref">

`device-desktop;5em;sd-text-icon blue-icon` +++ **Overview**

Relatively new to HydroMT? Start here with an introduction about HydroMT, common usage and introduction on the CLI and Python interface.

</div>

<div class="grid-item-card" text-align="center" link="model_main" link-type="ref">

`graph;5em;sd-text-icon blue-icon` +++ **Working with models**

Want to know in more details how to work with models in HydroMT? Look here for details on model build/update, the structure of the workflow file or region options.

</div>

<div class="grid-item-card" text-align="center" link="get_data" link-type="ref">

`stack;5em;sd-text-icon blue-icon` +++ **Data Catalog**

Learn how to manage your data using the HydroMT Data Catalog, including existing pre-defined catalogs and how to create your own.

</div>

<div class="grid-item-card" text-align="center" link="methods_processes" link-type="ref">

`tools;5em;sd-text-icon blue-icon` +++ **Supporting (GIS) functionalities**

Interested in the supporting and GIS functionalities of HydroMT that you would like to use in your own plugin or python script? Check it out here.

</div>

<div class="grid-item-card" text-align="center" link="migration_guide" link-type="ref">

`file-moved;5em;sd-text-icon blue-icon` +++ **Migration Guide**

Already familiar with HydroMT but need help to update your code and workflow files for HydroMT version 1.0? Check out our migration guide.

</div>

<div class="grid-item-card" text-align="center" link="faq" link-type="ref">

`question;5em;sd-text-icon blue-icon` +++ **Frequently Asked Questions**

Find answers to the most commonly asked questions about HydroMT, as well as tips and tricks for using the software effectively in this page.

</div>

</div>

<div class="toctree" maxdepth="2" caption="User Guide" hidden="">

Overview \<overview/index\> Working with models \<models/model_overview\> Data Catalog \<data_catalog/data_overview\> Supporting functionalities \<supporting_functions/methods_main\> Migration guide \<migration_guide/index\> Terminology \<terminology\> Frequently Asked Questions \<faq\>

</div>


---

## index

> **Source:** [overview/index.md](https://deltares.github.io/hydromt/latest/user_guide/overview/index.html) · [upstream `.rst`](https://github.com/Deltares/hydromt/blob/4b6ce64a7ed082740766e466a1861860f5150322/docs/user_guide/overview/index.rst)

<div class="grid" gutter="1">

3

<div class="grid-item-card" text-align="center" link="detailed_intro" link-type="ref">

`book;5em;sd-text-icon blue-icon` +++ Introduction and common usage

</div>

<div class="grid-item-card" text-align="center" link="hydromt_cli" link-type="ref">

`terminal;5em;sd-text-icon blue-icon` +++ Command Line Interface

</div>

<div class="grid-item-card" text-align="center" link="hydromt_python" link-type="ref">

`file;5em;sd-text-icon blue-icon` +++ Python Interface

</div>

</div>

<div class="toctree" hidden="">

Introduction to HydroMT \<hydromt_intro\> HydroMT CLI interface \<hydromt_cli\> HydroMT Python API \<hydromt_python\>

</div>


---

## hydromt intro

> **Source:** [overview/hydromt_intro.md](https://deltares.github.io/hydromt/latest/user_guide/overview/hydromt_intro.html) · [upstream `.rst`](https://github.com/Deltares/hydromt/blob/4b6ce64a7ed082740766e466a1861860f5150322/docs/user_guide/overview/hydromt_intro.rst)

## Why HydroMT?

**HydroMT** (Hydro Model Tools) is an open-source Python package that facilitates the process of building and analyzing spatial geoscientific models with a focus on water system models. It does so by automating the workflow to go from raw data to a complete model instance which is ready to run and to analyze model results once the simulation has finished. As such it is an interface between *user*, *data* and hydro *models*. Furthermore it does so in a *fast*, *flexible*, *scalable*, *modular* and *reproducible* manner.

This process, before HydroMT is pictured below:

<figure>
<img src="../../_static/hydromt_before.jpg" alt="A sequence of a typical pre-analysis workflow, before HydroMT was invented" />
<figcaption aria-hidden="true">A sequence of a typical pre-analysis workflow, before HydroMT was invented</figcaption>
</figure>

In contrast to the image above, the same workflow using HydroMT is depicted in the next image:

<figure>
<img src="../../_static/hydromt_using.jpg" alt="A sequence describing how the pre-analysis goes using HydroMT" />
<figcaption aria-hidden="true">A sequence describing how the pre-analysis goes using HydroMT</figcaption>
</figure>

With HydroMT, users can focus on the science and analysis of their models, rather than the tedious and error-prone task of preparing input data and model files. HydroMT provides a standardized and efficient way to build, update and analyze hydrological models, making it easier for researchers and practitioners to work with complex water system models.

With its data-centered approach, users can with HydroMT:

- Rapidly build and update model instances anywhere in the world
- Prepare small to large-scale applications
- Use a wide range of input data formats and sources and combinations of global and local datasets thanks to the flexible data catalog and harmonization capabilities
- Focus on collecting and improving relevant input data rather than correcting model files
- Leverage pre-defined methods and workflows for common hydrological modeling tasks
- Reuse workflows and methods between different model software through a plugin architecture

## How to use HydroMT ?

![image](../../_static/core_and_plugins.png)

![image](../../_static/getting_started.png)

## How does HydroMT work?

The main use of HydroMT is to build or update a model instance from raw data. This includes reading and harmonizing the raw input data via the Data Catalog, and transforming the data through a series of methods and (GIS) processes to the model specific format and requirements.

Users can interact with HydroMT through a Command Line Interface (CLI) or directly through the Python API. They tell HydroMT information on the data to use in one or several data catalog files, and define the steps to go from raw data to a model instance in a model build configuration (or workflow) file (for example which DEM data to use, what is the spatial resolution, etc.). HydroMT then executes the defined steps to build or update the model instance.

<div id="arch_hydromt">

<figure>
<img src="../../_static/Architecture_model_data_input.png" alt="A diagram showing an overview of the architecture of HydroMT." />
<figcaption aria-hidden="true">A diagram showing an overview of the architecture of HydroMT.</figcaption>
</figure>

</div>

More concretely HydroMT is organized in the following way:

### Input Data

HydroMT is data-agnostic through the `Data Catalog`, which allows to read a wide range of data formats and unify the input data (e.g., on-the-fly renaming and unit conversion). Datasets are listed and passed to HydroMT in a user defined data catalog `yaml file <data_yaml>`. HydroMT also provides several `pre-defined data catalogs <existing_catalog>` with mostly global datasets that can be used as is, although not all datasets in these catalogs are openly accessible.

Currently, five different types of input data are supported by HydroMT and represented by a specific Python data object:

- gridded datasets such as DEMs or gridded spatially distributed rainfall datasets (represented by `RasterDataset <RasterDataset>` objects, a raster-specific extension of Xarray Datasets)
- tables or tabular data, that can be used to, for example, convert land use classes to model parameters (represented by Pandas `DataFrame <DataFrame>` objects)
- vector datasets such as administrative units or river center lines (represented by Geopandas `GeoDataFrame <GeoDataFrame>` objects)
- time series with associated geo-locations such as observations of discharge (represented by `GeoDataset <GeoDataset>` objects, a geo-specific extension of Xarray Datasets)
- non-spatial N-dimension data (represented by Xarray `Dataset <Dataset>` objects).

Under the hood, when HydroMT reads data from the catalog, it uses the `uri` provided to locate the data, a `Driver` to read the data from file using the appropriate python function for the specific format of the data (e.g. gridded datasets are read with rasterio for GeoTiff files or with xarray for NetCDF and zarr files), and a `Data Adapter` to read only the correct variables, slice the data for the required spatial region and time, and harmonize the data to HydroMT standards (e.g. renaming variables or converting units).

### Models

HydroMT defines any model instance through the model-agnostic `Model` class that is composed of several `Model Components`. These components typically represent a specific type of model data or file such as:

- a configuration component, for model settings, output options, etc. (usually represented as a dictionary)
- a static spatial component, for example static gridded data or mesh data depending on the model schematization (usually represented with a xarray object)
- a forcing component, for time-varying inputs such as rainfall or temperature (usually represented with a xarray object)
- etc.

The available model components and the underlying python object containing the data will vary depending on the model software being used. Model instances can be `built from scratch <model_build>`, and `existing models can be updated <model_update>` step-by-step based on a pipeline of methods defined in a `model build workflow` `.yaml file <model_workflow>`.

While HydroMT provides several general model components that can readily be used, each model software is unique in its own way either because it has specific data requirements or because its input file formats differ (netcdf or ascii or binary...). So HydroMT can easily be tailored to specific model software through a plugin infrastructure. These `plugins <plugins>` have the same interface, but with model-specific components, file readers, writers and workflows. In practice, you will most probably use HydroMT together with a specific plugin for your model software of choice that HydroMT core on its own.

### Methods and (GIS) processes

Most of the heavy work in HydroMT is done by `Methods and (GIS) processes <model_workflow>`, indicated by the gear wheels in the `architecture diagram <arch_hydromt>` above. `Methods` provide the low-level functionality such as rasterization, reprojection, or zonal statistics. `Processes` combine several methods to transform the raw input data to a model layer or parameters. Examples of processes include the delineation of hydrological basins (watersheds), conversion of landuse-landcover data to model parameter maps, and calculation of model skill statistics.

The list of processes to use as well as their order and options are defined in a `model build workflow` `.yaml file <model_workflow>`. HydroMT comes with a set of pre-defined methods and processes that can be used as is, but plugins will typically extend this list with model-specific methods and processes.

For advanced users or developers, HydroMT exposes all methods and processes through its Python API. Feel free to check the `API reference <api_reference>` for all available functions.

### Command Line Interface (CLI) and python interface (API)

Finally users can interact with HydroMT through the following interfaces:

- **Command Line Interface (CLI)**: the CLI is a high-level interface to HydroMT. It is used to run HydroMT commands such as `build <model_build>`, `update <model_update>`
- **Python Interface**: while most common functionalities can be called through the CLI, the Python interface offers more flexibility for advanced users. It allows you to e.g. interact directly with a model component Model API and apply the many methods and processes available.

Check the next page of this *Overview* section for a quickstart on how to use HydroMT through the `CLI <hydromt_cli>` and the `Python API <hydromt_python>`.

### Summary

To summarize, the functionality of HydroMT can be broken down into four components, which are around input data, model instances, methods and workflows. Users can interact with HydroMT through a high-level command line interface (CLI) to build model instances from scratch, update existing model instances or analyze model results. Furthermore, a Python interface is available that exposes all functionality for experienced users.

This page is an overview of the main architecture and concepts of HydroMT. If you would like to know more, please visit the `detailed architecture <architecture>` in the developer guide.


---

## hydromt cli

> **Source:** [overview/hydromt_cli.md](https://deltares.github.io/hydromt/latest/user_guide/overview/hydromt_cli.html) · [upstream `.rst`](https://github.com/Deltares/hydromt/blob/4b6ce64a7ed082740766e466a1861860f5150322/docs/user_guide/overview/hydromt_cli.rst)

The HydroMT command line interface (CLI) is a command line tool that allows you to run HydroMT commands from the terminal. It is installed as part of the HydroMT package.

To use the HydroMT CLI, open a terminal, (activate the environment where HydroMT is installed) and type `hydromt` followed by the command you want to run. You can also run `hydromt --help` to get an overview of which commands are available. The following commands are available:

<div class="dropdown">

**Information commands**

- `hydromt \-\-help <hydromt_help>`
- `hydromt \-\-version <hydromt_version>`
- `hydromt \-\-models <hydromt_models>`
- `hydromt \-\-plugins <hydromt_plugins>`

</div>

<div class="dropdown">

**Model commands**

- `hydromt build <hydromt_build>`
- `hydromt update <hydromt_update>`

</div>

<div class="dropdown">

**Data catalog commands**

- `hydromt export <hydromt_export>`

</div>

<div class="dropdown">

**Validation commands**

- `hydromt check <hydromt_check>`

</div>

## Information commands

The base commands or options are here to get some information about HydroMT like the help, installed version and available models.

### help

The `hydromt --help` command prints the help message for the HydroMT CLI. It shows the available commands and options that can be used with HydroMT. For example:

``` console
> hydromt --help

Usage: hydromt [OPTIONS] COMMAND [ARGS]...

Command line interface for hydromt models.

Options:
  --version     Show the version and exit.
  --models      Print available model plugins and exit.
  --components  Print available component plugins and exit.
  --plugins     Print available component plugins and exit.
  --help        Show this message and exit.

Commands:
  build   Build models
  check   Validate config / data catalog / region
  export  Export data
  update  Update models
```

### version

The `hydromt --version` command prints the installed version of HydroMT. For example:

``` console
> hydromt --version

HydroMT version: 1.0.0
```

### models

The `hydromt --models` command prints the available generic models from HydroMT core and the installed plugins together with their versions. For example:

``` console
> hydromt --models

Model plugins:
    - model (hydromt 1.3.0)
    - wflow_sbm (hydromt_wflow 1.0.0)
    - wflow_sediment (hydromt_wflow 1.0.0)
```

### plugins

The `hydromt --plugins` command prints the installed HydroMT plugins together with their versions. This includes the model plugins (e.g hydromt_wflow), available pre-defined data catalogs (e.g deltares_data), available drivers to read different types of data (e.g raster_xarray, geodataset_xarray). For plugin developers, it also includes available model components and URI resolvers. For example:

``` console
> hydromt --plugins

Model plugins:
    - model (hydromt 1.0.0)
Component plugins:
    - ConfigComponent (hydromt 1.0.0)
    - DatasetsComponent (hydromt 1.0.0)
    - GeomsComponent (hydromt 1.0.0)
    - GridComponent (hydromt 1.0.0)
    - ...
Driver plugins:
    - dataset_xarray (hydromt 1.0.0)
    - geodataframe_table (hydromt 1.0.0)
    - geodataset_vector (hydromt 1.0.0)
    - geodataset_xarray (hydromt 1.0.0)
    - ...
Catalog plugins:
    - deltares_data (hydromt_data 1.0.0)
    - artifact_data (hydromt_data 1.0.0)
    - aws_data (hydromt_data 1.0.0)
    - gcs_cmip6_data (hydromt_data 1.0.0)
Uri_resolver plugins:
    - convention (hydromt 1.0.0)
    - raster_tindex (hydromt 1.0.0)
```

## Model commands

### build

The `hydromt build` command is used to build models from scratch. It has two mandatory arguments:

- \`MODEL\`: The name of the model to build. The available models can be printed using the `hydromt --models` command.
- \`MODEL_ROOT\`: Absolute or relative path to the output folder of the model to build.

The `hydromt build` command has several options to specify the configuration file, the region, the data catalog, and other options. The most important ones are:

- \`-i, --config\`: Relative or absolute path to the HydroMT configuration file so that HydroMT knows what to prepare for our model (data to use, processing options etc.).
- \`-d, --data\`: Relative or absolute path to the local yaml data catalog file or name of a predefined data catalog. The data catalog is a yaml file that contains the paths to the data that will be used to build the model.

Here is an example of how to use the command:

``` console
> hydromt build wflow_sbm /path/to/model_root -i /path/to/wflow_build_config.yml  -d deltares_data  -v
```

You can find more information on the steps to build a model in the `Building a model <model_build>` section. In this section, you will also find how to `prepare a workflow file <model_workflow>`. To know more about the data catalog, you can refer to the `Working with data in HydroMT <get_data>` section.

Finally you can check the `hydromt build API <build_api>` for all the available options for the build command.

### update

The `hydromt update` command is used to update an existing model. It is quite similar to the build command and has two mandatory arguments:

- \`MODEL\`: The name of the model to update. The available models can be printed using the `hydromt --models` command.
- \`MODEL_ROOT\`: Absolute or relative path to the model to update.

The `hydromt update` command has several options to specify the configuration file, the the data catalog, and other options. The most important ones are:

- \`-i, --config\`: Relative or absolute path to the HydroMT configuration file so that HydroMT knows what to update for our model (data to use, processing options etc.).
- \`-d, --data\`: Relative or absolute path to the local yaml data catalog file or name of a predefined data catalog. The data catalog is a yaml file that contains the paths to the data that will be used to update the model.
- \`-o, --model-out\`: Relative or absolute path to the output folder of the updated model. If not provided, the current model will be overwritten.

Here is an example of how to use the command:

``` console
> hydromt update wflow_sbm /path/to/model_to_update -o /path/to/updated_model -i /path/to/wflow_update_config.yml -d /path/to/data_catalog.yml -v
```

You can find more information on the steps to update a model in the `Updating a model <model_update>` section. In this section, you will also find how to `prepare a workflow file <model_workflow>`. To know more about the data catalog, you can refer to the `Working with data in HydroMT <get_data>` section.

Finally you can check the `hydromt update API <update_api>` for all the available options for the update command.

## Data catalog commands

### export

The `hydromt export` command is used to export sample data from a data catalog for example to export global data for a specific region and time extent. It has one mandatory argument:

- \`EXPORT_DEST_PATH\`: Absolute or relative path to the output folder of the exported data.

The input data catalogs are specified using the <span class="title-ref">-d, --data</span> option as in the <span class="title-ref">build</span> or <span class="title-ref">update</span> commands.

There are two ways to specify the sources/extent of the data to export: either fully from the command line or by using a configuration file.

If you are using the command line, the main options are:

- \`-d, --data\`: Relative or absolute path to the local yaml data catalog file or name of a predefined data catalog. The data catalog is a yaml file that contains the paths to the data that will be used to export the data.
- \`s, --source\`: The data source to export. Only one can be specified from the command line.
- \`-t, --time\`: Set the time extent for which to export the data. The time extent is specified as a list with the start and end date.

Here is an example of how to use the command:

``` console
> hydromt export /path/to/exported_data -d /path/to/data_catalog.yml -s "era5" -t "['2010-01-01', '2010-01-31']" -v
```

If you want to export several sources or for more options, you can also use a configuration file instead. In that case, the main options are:

- \`-i, --config\`: Relative or absolute path to the export configuration file. The export configuration file is a yaml file that contains the sources, region, and time extent to export.

And the command line would look like:

``` console
> hydromt export /path/to/exported_data -i /path/to/export_config.yml -v
```

An example of the export file is:

``` yaml
export_data:
    data_libs:
        - /path/to/data_catalog.yml
    region:
        bbox: [4.68,53.19,4.69,53.20]
    time_range: ['2010-01-01', '2020-12-31']
    sources:
        - hydro_lakes
        - era5
    unit_conversion: False
    append: False
    meta:
        version: 0.1
```

You can find detailed document on the function in [hydromt.DataCatalog.export_data](../_generated/hydromt.data_catalog.DataCatalog.export_data.rst). For the region, only the `bbox` and `geom` types are supported, see the `region <region>` section for more information.

Finally you can check the `hydromt export API <export_api>` for all the available options for the export command.

## Validation commands

### check

The `hydromt check` command is used to validate the configuration file, and the data catalog. It can be useful to validate files before running other command lines to avoid errors. Please note that it will only check the syntax of the files provided. The actual data or calculations referenced will not be checked, loaded or performed.

The command does not have any required arguments but several options that you can choose from:

- `-m, --model`: The name of the model to validate. The available models can be printed using the `hydromt --models` command.
- `-i, --config`: Relative or absolute path to the HydroMT configuration file to validate. Note that hydromt v1 cannot validate v0 config files, and vice versa.
- `-d, --data`: Relative or absolute path to the local yaml data catalog file or name of a predefined data catalog to validate.
- `--format` specify which format of data catalog to validate. Accepted options are `v0` or `v1`
- `--upgrade` when validating `v0` data catalogs you can supply the `--upgrade` flag, and hydromt will convert the data catalog to the `v1` format and write it to a file with the same name but with the suffix `_v1` added to the file stem.

Here are some examples of how to use the command:

``` console
> hydromt check -m wflow -i /path/to/wflow_config.yml -d /path/to/data_catalog.yml -v

> hydromt check -d /path/to/data_catalog.yml --format v0 --upgrade -v

> hydromt check -m wflow -i /path/to/wflow_config.yml -v
```

The validation is so far limited:

- data catalog: only the format and options are validated but it does not try to load any of the data.
- configuration file: it will check if the methods exists and if the correct arguments are called. No validation is done on the content and type of the arguments themselves.

Finally you can check the `hydromt check API <check_api>` for all the available options for the check command.


---

## hydromt python

> **Source:** [overview/hydromt_python.md](https://deltares.github.io/hydromt/latest/user_guide/overview/hydromt_python.html) · [upstream `.rst`](https://github.com/Deltares/hydromt/blob/4b6ce64a7ed082740766e466a1861860f5150322/docs/user_guide/overview/hydromt_python.rst)

As HydroMT's architecture is modular, it is possible to use HydroMT as a python library without using the command line interface (CLI). Using the library, you have more of HydroMT's functionalities at your disposal. This can be useful if you want to for example:

- build / update models, check configurations or export data in Python instead of the CLI
- analyze model inputs or results
- analyze and compare input data without connecting to a specific model
- process input data for another reason than building a model
- create your own HydroMT plugin

So first, let's go deeper into the API of HydroMT. You have all available functions and their documentation in the [API reference](../api.rst).

HydroMT is here to read and harmonize **input data**, and to process it via its **methods and (GIS) processes** in order to prepare ready to run **models**. So HydroMT's methods are organized around these main objects.

## Logging configuration

When using HydroMT as a Python library, logging is not configured automatically. This is different from the CLI, where logging is initialized for you.

If you do not explicitly configure logging in your script, HydroMT will not emit log messages to the terminal, even if functions are executed successfully. This can make it difficult to understand what the library is doing internally or to debug issues.

To enable logging, you can call:

``` python
from hydromt import log

log.initialize_logging()
```

This sets up a default logging configuration and ensures that messages are printed to the console. If you need more advanced logging behavior (for example logging to a file or integrating with an existing application logger), review the module `hydromt.log` for your options or you should configure the Python `logging` module yourself before using HydroMT.

<div class="dropdown">

**Model functions**

- `build a model <hydromt_build_python>`
- `update a model <hydromt_update_python>`
- `loading a model <hydromt_load_python>`

</div>

<div class="dropdown">

**Data catalog functions**

- `reading data <hydromt_data_read_python>`
- `export data <hydromt_export_python>`

</div>

<div class="dropdown">

**Methods and processes**

- `methods <hydromt_methods_python>`
- `processes <hydromt_processes_python>`

</div>

## Model functions

You can use HydroMT to build or update a model in Python instead of the CLI. Additionally with Python, you can also load an existing model to read or analyze its inputs or results.

### Building a model

The `build` function is used to build models from scratch. In Python, you can also use the build function in combination with the build workflow file to build a model. Here is a small example of how to use the build function in Python:

``` python
from hydromt import ExampleModel, log
from hydromt.readers import read_workflow_yaml

# Configure logging
log.initialize_logging()

# Instantiate model
model = ExampleModel(
    root="./path/to/example_model",
    data_catalog=["./path/to/data_catalog.yml"],
)
# Read build options from yaml
_, _, build_options = read_workflow_yaml(
    "./path/to/build_options.yaml"
)
# Build model
model.build(steps=build_options)
```

Additionally, in Python, you can also build the model step-by step by calling each of the model steps as methods, instead of using a workflow file. For example:

``` python
from hydromt import ExampleModel, log

# Configure logging
log.initialize_logging()

# Instantiate model
model = ExampleModel(
    root="./path/to/example_model",
    data_catalog=["./path/to/data_catalog.yml"],
)
# Build model step by step
# Step 1: populate the config with some values
model.config.update(
    data = {'starttime': '2000-01-01', 'endtime': '2010-12-31'}
)
# Step 2: define the model grid
model.grid.create_from_region(
    region={"subbasin": [12.2051, 45.8331], "uparea": 50},
    res=1000,
    crs="utm",
    hydrography_fn="merit_hydro_1k",
    basin_index_fn="merit_hydro_index",
)
# Step 3: add DEM data to the model grid
model.grid.add_data_from_rasterdataset(
    raster_fn="merit_hydro_1k",
    variables="elevtn",
    fill_method=None,
    reproject_method="bilinear",
    rename={"elevtn": "DEM"},
)
# Write the model to disk
model.write()
```

### Updating a model

The `update` function is used to update an existing model. In Python, you can also use the update function in combination with the workflow file to update a model. Here is a small example of how to use the update function in Python:

``` python
from hydromt import ExampleModel, log
from hydromt.readers import read_workflow_yaml

# Configure logging
log.initialize_logging()

# Instantiate model
model = ExampleModel(
    root="./path/to/example_model_to_update",
    data_catalog=["./path/to/data_catalog.yml"],
    mode = "r+", # open model in read and write mode
)
# Read update options from yaml
_, _, update_options = read_workflow_yaml(
    "./path/to/update_options.yaml"
)
# If you want to save the model in a different folder
model.read()
model.root.set("./path/to/updated_example_model", mode="w")
# Update model
model.update(steps=update_options)
```

Similarly to build, you can also update the model step by step by calling each of the model steps as methods, instead of using a workflow file. For example:

``` python
from hydromt import ExampleModel, log

# Configure logging
log.initialize_logging()

# Instantiate model
model = ExampleModel(
    root="./path/to/example_model_to_update",
    data_catalog=["./path/to/data_catalog.yml"],
    mode = "r+", # open model in read and write mode
)
# If you want to save the model in a different folder
model.read()
model.root.set("./path/to/updated_example_model", mode="w")
# Update model step by step
# Step 1: update the config with new values
model.config.update(
    data = {'starttime': '2010-01-01', 'endtime': '2020-12-31'}
)
# Step 2: add landuse data in the model grid
model.grid.add_data_from_rasterdataset(
    raster_fn="vito",
    reproject_method="mode",
    rename={"vito": "landuse"},
)
# Write the updated model to disk
model.write()
```

### Loading and analyzing a model

You can also use HydroMT and its `~model.Model` and `~components.base.ModelComponent` classes to do some analysis on your model inputs or results. HydroMT views a model as a combination of different components to represent the different type of inputs of a model, like `config` for the model run configuration file, `forcing` for the dynamic forcing data of the model etc. For each component, there are methods to `<component>.set` (update or add a new data layer), `<component>.read` and `<component>.write`. The underlying data of each component is accessible via the `<component>.data` attribute (e.g dict, xarray or geopandas objects, etc.).

Here is a small example of how to use the `~model.Model` class in python to plot or analyze your model:

``` python
from hydromt import ExampleModel, log
# Configure logging
log.initialize_logging()
# create a ExampleModel instance for an existing model saved in "example_model" folder
model = hydromt.ExampleModel(root="example_model", mode="r")
# read/get the grid data
grid = model.grid.data
# plot the DEM
grid["DEM"].plot()
```

You can find more detailed examples on using the Model class in Python in:

- [Working with models in python](../../_examples/working_with_models.ipynb)

And feel free to visit some of the `plugins <plugins>` documentation to find even more examples!

## Data catalog functions

The DataCatalog is a core part of HydroMT to find, read, harmonize and transform input data. It is usually prepared from a yaml file that defines different data sources and their properties. You can find more information on the DataCatalog in the `Data Catalog documentation <get_data>`.

<div class="currentmodule">

hydromt.data_catalog

</div>

### Reading data

HydroMT supports reading five different data types:

- **rasterdataset**: raster (regular grid) data (e.g DEM, landuse, soil type, model grid etc.)
- **geodataframe**: vector data (e.g shapefiles, geojson, etc.)
- **geodataset**: time varying vector data (e.g point time-series, station data, etc.)
- **dataframe**: non-spatial tabular data (e.g csv, excel, etc.)
- **dataset**: non-spatial multi-dimensional data (e.g netcdf, hdf5, etc.)

All input data is accessible and be read through the `~data_catalog.DataCatalog` class via its `get_<data_type>` methods, for example: `get_rasterdataset`, `get_geodataframe`, `get_geodataset`, `get_dataframe`, `get_dataset`.

Here is a small example of how to use the DataCatalog to read data in Python:

``` python
from hydromt import DataCatalog, log

# Configure logging
log.initialize_logging()

# create a data catalog from a local data_catalog file
cat = DataCatalog("data_catalog.yml")
# read a raster dataset ("dem" source in the data catalog)
dem = cat.get_rasterdataset("dem")
# read a vector dataset ("catchments" source in the data catalog)
catchments = cat.get_geodataframe("catchments")
# read a geodataset with some time and space slicing
qobs = cat.get_geodataset(
  "qobs",
  time_range = ("2000-01-01", "2010-12-31"),
  bbox = [5.0, 50.0, 6.0, 51.0]
)
```

You can find more detailed examples on using the DataCatalog in Python in:

- [Reading raster data](../../_examples/reading_raster_data.ipynb)
- [Reading vector data](../../_examples/reading_vector_data.ipynb)
- [Reading geospatial point time-series data](../../_examples/reading_point_data.ipynb)
- [Reading tabular data](../../_examples/reading_tabular_data.ipynb)

In short, what happens in the bachkground of the `get_<data_type>` methods is:

1.  The DataCatalog finds the data source in the data catalog that matches the requested name and solves the data source path (e.g local file, remote url, database, etc.)
2.  The DataCatalog uses the appropriate data reader or `driver` to read the data from the source depending on its type and file format (e.g. tif, netcdf, shapefile, csv, etc.)
3.  The DataCatalog harmonizes (renaming, unit conversion) and slices (variables, time, space) the data according to the data source properties using the appropriate `data_adapter`.

HydroMT is flexible enough that you can add your own data types or readers if needed. You can find more information on how the DataCatalog works and how to implement your own data readers in the `Developer documentation <intro_developer_guide>`.

### Exporting data

HydroMT also supports exporting data from a DataCatalog for a specific region or time range. This can be useful to extract and export input data for a specific model or region of interest. For this, you can use the `~data_catalog.DataCatalog.export_data` method of the `~data_catalog.DataCatalog` class. Here is a small example of how to use the export_data method in Python:

``` python
from hydromt import DataCatalog

# create a data catalog from a local data_catalog file
cat = DataCatalog("data_catalog.yml")
# export data for a specific region and time range
cat.export_data(
    new_root = "./exported_data",
    bbox = [5.0, 50.0, 6.0, 51.0],
    time_range = ("2000-01-01", "2010-12-31"),
    source_names = ["dem", "landuse", "qobs"]
)
```

## Methods and processes

HydroMT provides a set of methods and (GIS) processes to process input data in order to prepare ready to run models. You can find more information on the available methods and processes in the `supporting functionnalities <methods_processes>`, `model processes <model_processes>` and in the `API reference <api_reference>`.

These methods and processes are only available via the Python API. You can use them directly in your Python scripts or Jupyter notebooks to process data for your models or for other purposes.

### Methods

Methods provide the low-level functionality to do the required processing of common data types such as grid and vector data.

HydroMT provides methods in different modules including:

- `gis.raster`: methods to work with raster (regular grid) data including reprojection, resampling, transforming, interpolating nodata or zonal statistics.
- `gis.vector`: methods to work with geodataset data (N-dimensional vector data). For example, reprojecting, transforming, updating geometry or converting to geopandas.GeoDataFrame to access further GIS methods.
- `gis.flw`: hydrological methods for raster DEM data. For example, calculate flow direction, flow accumulation, stream network, catchments, or reproject hydrography.
- `stats.skills`: statistical methods to compute skill scores of models (e.g. NSE, KGE, bias and many more).
- `stats.extremes`: methods to analyse extreme events (extract peaks or compute return values).

### Processes

Processes combine several methods to go from raw input data to a model component. Examples of processes include the delineation of hydrological basins (watersheds), conversion of landuse-landcover to model parameter maps, etc.

HydroMT provides processes in the `model.processes` module including:

- `model.processes.basin_mask`: processes to delineate (sub-)basins and create basin masks.
- `model.processes.grid`: processes to prepare regular gridded data from different data types.
- `model.processes.mesh`: processes to prepare unstructured mesh data from different data types.
- `model.processes.rivers`: processes to prepare river network data for hydrological or 1D hydraulic models.
- `model.processes.meteo`: processes to prepare gridded meteorological forcing data including downscaling methods.
