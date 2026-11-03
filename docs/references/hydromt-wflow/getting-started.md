---
title: HydroMT-Wflow getting started
source: "https://deltares.github.io/hydromt_wflow/stable/getting_started/"
reference: "HydroMT team. (n.d.). *HydroMT-Wflow getting started*. HydroMT-Wflow documentation. sources/tools/hydromt-wflow/getting-started.md"
topic: tools
type: documentation
promoted: 2026-05-17
tags:
  - tools
  - hydromt-wflow
accessed: 2026-05-17
---

# HydroMT-Wflow getting started

## 1. Getting started

HydroMT-Wflow is a model plugin for [HydroMT](https://deltares.github.io/hydromt), extending its core functionalities with Wflow-specific components and workflows. It can be installed as a standalone package or alongside other HydroMT model plugins (e.g. HydroMT-SFINCS, HydroMT-Fiat). We recommend installing HydroMT-Wflow in a dedicated Python environment to ensure dependency consistency.

## 2. Installation guide

### 2.1. Prerequisite: Python installation

You will need **Python 3.11 or newer** and a package/environment manager such as pip, conda, mamba or uv. These tools simplify installing packages and managing isolated environments.

If you do not yet have one installed, we recommend either:

- [Miniconda](https://docs.conda.io/en/latest/miniconda.html)
- [Miniforge (Mambaforge)](https://conda-forge.org/docs/)
- [uv](https://docs.astral.sh/uv/)

Both conda variants come preconfigured with the **conda-forge** channel, which provides free and open packages used by HydroMT.

### 2.2. Installing HydroMT-Wflow

HydroMT-Wflow is available from both **PyPI** and **conda-forge**. The simplest and most flexible approach is to install it using **pip** inside a new environment.

#### 2.2.1. Installation in a conda environment using uv / pip

We recommend creating a clean environment to avoid dependency conflicts. For example:

```
$ conda create -n hydromt-wflow uv python=3.13
$ conda activate hydromt-wflow
$ uv pip install hydromt_wflow
```

This will install HydroMT-Wflow along with HydroMT core and all required dependencies.

To verify the installation, you can list the installed HydroMT plugins:

```
$(hydromt-wflow) hydromt --plugins
    Model plugins:
        - model (hydromt 1.3.0)
        - wflow_sbm (hydromt_wflow 1.0.0)
        - wflow_sediment (hydromt_wflow 1.0.0)
    Component plugins:
        - ConfigComponent (hydromt 1.3.0)
        - DatasetsComponent (hydromt 1.3.0)
        - GeomsComponent (hydromt 1.3.0)
        - GridComponent (hydromt 1.3.0)
        - MeshComponent (hydromt 1.3.0)
        - SpatialDatasetsComponent (hydromt 1.3.0)
        - TablesComponent (hydromt 1.3.0)
        - VectorComponent (hydromt 1.3.0)
    Driver plugins:
        - dataset_xarray (hydromt 1.3.0)
        - geodataframe_table (hydromt 1.3.0)
        - geodataset_vector (hydromt 1.3.0)
        - geodataset_xarray (hydromt 1.3.0)
        - pandas (hydromt 1.3.0)
        - pyogrio (hydromt 1.3.0)
        - raster_xarray (hydromt 1.3.0)
        - rasterio (hydromt 1.3.0)
    Catalog plugins:
        - deltares_data (hydromt 1.3.0)
        - artifact_data (hydromt 1.3.0)
        - aws_data (hydromt 1.3.0)
        - gcs_cmip6_data (hydromt 1.3.0)
    Uri_resolver plugins:
        - convention (hydromt 1.3.0)
        - raster_tindex (hydromt 1.3.0)
```

#### 2.2.2. Installing optional dependencies

HydroMT-Wflow provides several optional dependencies that extend its capabilities, such as additional data sources or hydrological processing functions. You can install these easily using pip’s extras syntax:

```
$(hydromt-wflow) uv pip install "hydromt_wflow[extra]"
```

This will install optional packages such as:

- **gwwapi** - provides access to Global Water Watch reservoir datasets.
- **hydroengine** - enables integration with Google Earth Engine.
- **wradlib** - provides radar rainfall processing and interpolation tools.
- **pyet** - adds evapotranspiration computation support.

For a list of all the optional dependency groups and their contents, have a look at the pyproject.toml file. Use hydromt\_wflow\[all\] to install all optional dependencies.

#### 2.2.3. Installing via conda

HydroMT-Wflow is also available through the conda-forge channel. You can install it directly with:

```
$ conda create -n hydromt-wflow -c conda-forge hydromt_wflow
$ conda activate hydromt-wflow
```

Note that some optional dependencies (e.g. `gwwapi` or `hydroengine`) are only available through PyPI. You can install them afterwards with pip inside your conda environment:

```
$(hydromt-wflow) uv pip install "hydromt_wflow[extra]"
```

### 2.3. Developer installation

If you want to contribute to HydroMT-Wflow or modify its source code, see the [Developer installation guide](https://deltares.github.io/hydromt_wflow/stable/dev_guide/dev_install.html#dev-env).

For development work, you can use either a Conda-based setup or **Pixi**, which provides a fully reproducible project environment. Pixi should be used only in developer installations — not for general users — since it manages dependencies project-locally and is less suited for managing multiple plugins globally.

## 3. Frequently asked questions

This page contains some FAQ / tips and tricks to work with HydroMT-Wflow. For more general questions on how to work with data or the HydroMT config and command line, you can visit the [HydroMT core FAQ page](https://deltares.github.io/hydromt/latest/user_guide/faq.html).

### 3.1. Building a Wflow model

> **Q**: Can I use other region arguments than `basin` or `subbasin`?

To build a Wflow model, it is strongly recommended to use the `basin` or `subbasin` when defining your region of interest. This ensures that all upstream contributing cells are included. Using other options might cause your Wflow model run to fail. If you know exactly what you are doing, you can use the `geom` option and provide an exact shapefile of the basin or subbasins that was prepared using the same DEM source intended for [`setup_basemaps()`](https://deltares.github.io/hydromt_wflow/stable/api/_generated/hydromt_wflow.WflowSbmModel.setup_basemaps.html#hydromt_wflow.WflowSbmModel.setup_basemaps "hydromt_wflow.WflowSbmModel.setup_basemaps").

> **Q**: How to make sure sub-basins will be properly derived by HydroMT?

When deriving sub-basins, HydroMT needs to know how to snap outlet points to the stream network. You can specify a snapping argument such as stream order (*strord*) or upstream area (*uparea*) in the region definition:

- `{'subbasin': [xmin, ymin, xmax, ymax], 'strord': 3}`
- `{'subbasin': 'path/to/geometry.shp', 'uparea': 100000}`

This ensures sub-basin delineation is consistent with the river network derived from your DEM.

> **Q**: Can I use different datasets for precipitation, temperature, and PET forcing data?

Yes. That's exactly why the methods were separated. You can use different sources for precipitation, temperature, and PET forcing:

- [`setup_precip_forcing()`](https://deltares.github.io/hydromt_wflow/stable/api/_generated/hydromt_wflow.WflowSbmModel.setup_precip_forcing.html#hydromt_wflow.WflowSbmModel.setup_precip_forcing "hydromt_wflow.WflowSbmModel.setup_precip_forcing")
- [`setup_temp_pet_forcing()`](https://deltares.github.io/hydromt_wflow/stable/api/_generated/hydromt_wflow.WflowSbmModel.setup_temp_pet_forcing.html#hydromt_wflow.WflowSbmModel.setup_temp_pet_forcing "hydromt_wflow.WflowSbmModel.setup_temp_pet_forcing")
- [`setup_pet_forcing()`](https://deltares.github.io/hydromt_wflow/stable/api/_generated/hydromt_wflow.WflowSbmModel.setup_pet_forcing.html#hydromt_wflow.WflowSbmModel.setup_pet_forcing "hydromt_wflow.WflowSbmModel.setup_pet_forcing")

Each method allows specifying its own data source (e.g. *precip\_fn*, *temp\_pet\_fn*, or *pet\_fn*), and HydroMT will handle the resampling and temporal alignment automatically.

> **Q**: Can I add several gauges at the same time?

Yes. Since HydroMT v1, you can add multiple gauges directly in the configuration file using the **new list-based YAML format**. You no longer need to enumerate methods (e.g. *setup\_gauges2*). The example below shows how to define multiple gauge sources:

```yaml
steps:
  - setup_gauges:
      gauges_fn: my_gauges1
  - setup_gauges:
      gauges_fn: my_gauges2
```

### 3.2. Updating a Wflow model

> **Q**: Is there an easy way to update reservoir parameters in my Wflow model?

Yes. You can directly edit the `meta_reservoirs_simple_control.geojson` or `meta_reservoirs_no_control.geojson` files saved by HydroMT in the *staticgeoms* folder. Once you have updated the parameters, simply provide these edited GeoJSON files as your new local input data when rebuilding or updating your model.

> **Q**: Can I select a specific Wflow TOML config file when updating my model?

Yes. You can define this in the `global` section of your HydroMT configuration file using the `config_filename` argument:

```yaml
global:
  config_filename: path/to/my_wflow_config.toml
```

### 3.3. Others

> **Q**: Can I convert my old Wflow model to the new Wflow Julia version with HydroMT?

Conversion from old Python Wflow models to Wflow Julia is **no longer supported** since HydroMT-Wflow version 1.0. You can use an older HydroMT-Wflow version for that task and follow the steps described in its documentation.

HydroMT-Wflow **does** support upgrading Wflow Julia v0.x models to Wflow Julia v1.x using:

- [`upgrade_to_v1_wflow()`](https://deltares.github.io/hydromt_wflow/stable/api/_generated/hydromt_wflow.WflowSbmModel.upgrade_to_v1_wflow.html#hydromt_wflow.WflowSbmModel.upgrade_to_v1_wflow "hydromt_wflow.WflowSbmModel.upgrade_to_v1_wflow") for SBM models, and
- [`upgrade_to_v1_wflow()`](https://deltares.github.io/hydromt_wflow/stable/api/_generated/hydromt_wflow.WflowSedimentModel.upgrade_to_v1_wflow.html#hydromt_wflow.WflowSedimentModel.upgrade_to_v1_wflow "hydromt_wflow.WflowSedimentModel.upgrade_to_v1_wflow") for sediment models.

See the example notebook for details: [here](https://deltares.github.io/hydromt_wflow/stable/_examples/upgrade_to_wflow_v1.html#example-upgrade-to-wflow-v1).
