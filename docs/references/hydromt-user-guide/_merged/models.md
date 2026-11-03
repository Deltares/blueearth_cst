---
title: HydroMT user guide — Working with models
ingest-source: hydromt-user-guide
upstream-repo: Deltares/hydromt
upstream-commit: 4b6ce64a7ed082740766e466a1861860f5150322
pulled: 2026-05-04
section: models
doc-type: user-guide
license: MIT
sources:
  - https://deltares.github.io/hydromt/latest/user_guide/models/model_overview.html
  - https://deltares.github.io/hydromt/latest/user_guide/models/model_build.html
  - https://deltares.github.io/hydromt/latest/user_guide/models/model_update.html
  - https://deltares.github.io/hydromt/latest/user_guide/models/model_workflow.html
  - https://deltares.github.io/hydromt/latest/user_guide/models/model_region.html
  - https://deltares.github.io/hydromt/latest/user_guide/models/model_components.html
  - https://deltares.github.io/hydromt/latest/user_guide/models/model_processes.html
---

# HydroMT user guide — Working with models

_Merged from 7 upstream page(s) at commit `4b6ce64a`, pulled 2026-05-04. Page boundaries are preserved as `## ` headings._


---

## model overview

> **Source:** [models/model_overview.md](https://deltares.github.io/hydromt/latest/user_guide/models/model_overview.html) · [upstream `.rst`](https://github.com/Deltares/hydromt/blob/4b6ce64a7ed082740766e466a1861860f5150322/docs/user_guide/models/model_overview.rst)

## High level functionality

HydroMT has the following high-level functionality for setting up models from raw data or adjusting models:

- `building a model <model_build>`: building a model from scratch.
- `updating a model <model_update>`: adding or changing model components of an existing model.

The exact process of building or updating a model can be configured in a single configuration `.yaml file <model_workflow>`. This file describes the full pipeline of model methods and their arguments. The methods vary for the different model classes and `plugins`, as documented in this documentation or for each plugin documentation website.

<div class="toctree" hidden="">

Building a model \<model_build\> Updating a model \<model_update\> Model workflow file \<model_workflow\> Defining a region \<model_region\> Model components (advanced) \<model_components\> Model processes (advanced) \<model_processes\>

</div>


---

## model build

> **Source:** [models/model_build.md](https://deltares.github.io/hydromt/latest/user_guide/models/model_build.html) · [upstream `.rst`](https://github.com/Deltares/hydromt/blob/4b6ce64a7ed082740766e466a1861860f5150322/docs/user_guide/models/model_build.rst)

To build a complete model from scratch using available data the `build` method can be used. The build method is identical for all `HydroMT model plugins <plugins>`, but the model methods (i.e. sections and options in the .yaml configuration file) are different for each model.

**Steps in brief:**

1)  Prepare or use a pre-defined **data catalog** with all the required data sources, see `working with data <get_data>`
2)  Prepare a **model workflow** which describes the complete pipeline to build your model: see `model workflow <model_workflow>`. This workflow file will also contain the **region** definition for your model (e.g. bounding box, point coordinates, polygon etc.): see `region definition <region>`.
3)  **Build** you model using the CLI or Python interface

The `hydromt build` method can be run from the command line or Python after the right conda environment is activated. The HydroMT core package contain implementation for generalized model classes. Specific model implementation for softwares have to be built from associated `HydroMT plugin <plugins>` that needs to be installed to your Python environment.

If you work from command line, you can check which models are available by running:

``` console
$ hydromt --models
```

Here is how to build a model from:

<div class="tab-set">

<div class="tab-item">

Command Line Interface (CLI)

For the hydromt core `example_model` plugin using the pre-defined `artifact_data` catalog:

``` console
$ hydromt build example_model "./path/to/example_model" -d "artifact_data" -i "./path/to/build_options.yaml" -v
```

For HydroMT Wflow SBM plugin `wflow_sbm` with a user-defined data catalog:

``` console
$ hydromt build wflow_sbm "./path/to/wflow_model" -d "./path/to/data_catalog.yml" -i "./path/to/build_options.yaml" -v
```

</div>

<div class="tab-item">

Python API

For the hydromt core `example_model` plugin using the pre-defined `artifact_data` catalog:

``` python
from hydromt import ExampleModel, log
from hydromt.readers import read_workflow_yaml

# Configure logging
log.initialize_logging()

# Instantiate model
model = ExampleModel(
    root="./path/to/example_model",
    data_catalog=["artifact_data"],
)
# Read build options from yaml
_, _, build_options = read_workflow_yaml(
    "./path/to/build_options.yaml"
)
# Build model
model.build(steps=build_options)
```

For HydroMT Wflow SBM plugin `wflow_sbm` with a user-defined data catalog:

``` python
from hydromt import log
from hydromt_wflow import WflowSbmModel
from hydromt.readers import read_workflow_yaml

# Configure logging
log.initialize_logging()

# Instantiate model
model = WflowSbmModel(
    root="./path/to/wflow_model",
    data_catalog=["./path/to/data_catalog.yml"],
)
# Read build options from yaml
_, _, build_options = read_workflow_yaml(
    "./path/to/build_options.yaml"
)
# Build model
model.build(steps=build_options)
```

Additionally, in Python, you can also build the model step-by step by calling each of the model steps as methods instead of using a workflow file. For example:

``` python
from hydromt import ExampleModel, log

# Configure logging
log.initialize_logging()

# Instantiate model
model = ExampleModel(
    root="./path/to/example_model",
    data_catalog=["artifact_data"],
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

</div>

</div>


---

## model update

> **Source:** [models/model_update.md](https://deltares.github.io/hydromt/latest/user_guide/models/model_update.html) · [upstream `.rst`](https://github.com/Deltares/hydromt/blob/4b6ce64a7ed082740766e466a1861860f5150322/docs/user_guide/models/model_update.rst)

To add or change one or more components of an existing model the `update` method can be used. The update method works identical for all `HydroMT model plugins <plugins>`, but the model methods (i.e. sections and options in the `.yaml workflow file <model_workflow>`) are different for each model.

**Steps in brief:**

1)  You have an **existing model** schematization. This model does not have to be complete.
2)  Prepare or use a pre-defined **data catalog** with all the required data sources, see `working with data <get_data>`
3)  Prepare a **model workflow** with the methods that you want to use to add or change components of your model: see `model workflow <model_workflow>`.
4)  **Update** your model using the CLI or Python interface

The `hydromt update` method can be run from the command line or Python after the right conda environment is activated. By default, the model is updated in place, overwriting the existing model schematization. To save a copy of the model provide a new output model root directory (using the `-o` option when working from command line). All model methods in the .yaml configuration file will be updated.

Here is how to update a model from:

<div class="tab-set">

<div class="tab-item">

Command Line Interface (CLI)

For the hydromt core `example_model` plugin using the pre-defined `artifact_data` catalog and saving the updated model in place:

``` console
$ hydromt update example_model "./path/to/example_model_to_update" -d "artifact_data" -i "./path/to/update_options.yaml" -v
```

For the hydromt core `example_model` plugin using the pre-defined `artifact_data` catalog and saving the updated model to a new location:

``` console
$ hydromt update example_model "./path/to/example_model_to_update" -o "./path/to/updated_example_model" -d "artifact_data" -i "./path/to/update_options.yaml" -v
```

For HydroMT Wflow SBM plugin `wflow_sbm` with a user-defined data catalog:

``` console
$ hydromt update wflow_sbm "./path/to/wflow_model" -d "./path/to/data_catalog.yml" -i "./path/to/update_options.yaml" -v
```

</div>

<div class="tab-item">

Python API

For the hydromt core `example_model` plugin using the pre-defined `artifact_data` catalog and saving the updated model in place:

``` python
from hydromt import ExampleModel, log
from hydromt.readers import read_workflow_yaml

# Configure logging
log.initialize_logging()

# Instantiate model
model = ExampleModel(
    root="./path/to/example_model_to_update",
    data_catalog=["artifact_data"],
    mode = "r+", # open model in read and write mode
)
# Read update options from yaml
_, _, update_options = read_workflow_yaml(
    "./path/to/update_options.yaml"
)
# Update model
model.update(steps=update_options)
```

For hydromt core `example_model` plugin using the pre-defined `artifact_data` catalog and saving the updated model to a new location:

``` python
from hydromt import ExampleModel, log
from hydromt.readers import read_workflow_yaml

# Configure logging
log.initialize_logging()

# Instantiate model
model = ExampleModel(
    root="./path/to/example_model_to_update",
    data_catalog=["artifact_data"],
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

For HydroMT Wflow SBM plugin `wflow_sbm` with a user-defined data catalog:

``` python
from hydromt import log
from hydromt_wflow import WflowSbmModel
from hydromt.readers import read_workflow_yaml

# Configure logging
log.initialize_logging()

# Instantiate model
model = WflowSbmModel(
    root="./path/to/wflow_model",
    data_catalog=["./path/to/data_catalog.yml"],
    mode = "r+", # open model in read and write mode
)
# Read update options from yaml
_, _, update_options = read_workflow_yaml(
    "./path/to/update_options.yaml"
)
# Update model
model.update(steps=update_options)
```

Additionally, in Python, you can also update the model step-by step by calling each of the model steps as methods instead of using a workflow file. For example:

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

</div>

</div>


---

## model workflow

> **Source:** [models/model_workflow.md](https://deltares.github.io/hydromt/latest/user_guide/models/model_workflow.html) · [upstream `.rst`](https://github.com/Deltares/hydromt/blob/4b6ce64a7ed082740766e466a1861860f5150322/docs/user_guide/models/model_workflow.rst)

A model workflow (the `.yaml` file that tells hydromt what to do and in which orders) consists of three main sections:

1.  `modeltype`
2.  `global`
3.  `steps`

The `modeltype` tells hydromt which model to use. In the case of using hydromt core, `model` and `example_model` are the only options but if you have plugins installed those will probably provide other options as well (e.g `sfincs`, `wflow_sbm`, etc.). You can discover which options you have installed with the CLI `hydromt --models`.

The `global` is where you provide any configuration that the model will need at initialization. This is where you for example, can list which data catalogs to use (if you do not want to specify it in your CLI or python script), or other options depending on the plugin you are using (e.g. name of the configuration file for Wflow, crs for Delft3D etc.).

For users that wish to use the core `model` class, this is also where you define which components the model should have, if they are spatial components and which components the model should use to define its region. If you are using a plugin, the plugin will have done this for you and you do not need to define the components. To know more about defining components, please check the `model components <model_components>` page.

Generally if you use a plugin, the `global` part may look something like this:

``` yaml
modeltype: wflow_sbm
global:
    data_libs:
        - artifact_data
        - local_data.yml
    config_filename: wflow_sbm.toml
    ...
```

Finally there is the `steps` part of the workflow. This should be a list, where each list item should be a name of a function you want to run, followed by any arguments you want to pass to that function. You can use the `.` syntax to call functions on components, or omit this if the function you want to call is defined on the model.

For example, using the core `example_model` plugin:

``` yaml
steps:
    - config.update: # update lines in the model config file
        data:
          'starttime': '2000-01-01'
          'endtime': '2010-12-31'
    - grid.create_from_region: # create the model grid from a region
        region:
          subbasin: [12.2051, 45.8331]
          uparea: 50
        res: 1000
        crs: utm
        hydrography_fn: merit_hydro_1k
        basin_index_fn: merit_hydro_index
    - grid.add_data_from_rasterdataset: # add DEM data to the model grid
        raster_fn: merit_hydro_1k
        variables: elevtn
        fill_method: null
        reproject_method: bilinear
        rename:
          elevtn: DEM
```

In the example above `config` and `grid` are the name of the components and `update`, `create_from_region`, and `add_data_from_rasterdataset` are the functions on it that you want to call. Please check for each specific component which functions are available to call. This should be well documented in the documentation of the plugin you are using.

In general, at the end of the steps, HydroMT will end with a last hidden step to `write` the whole model. If you only wish to write specific components (e.g. when updating) or change some of the write options, you can add a final step to call `component.write` on the components you wish to write.

``` yaml
steps:
    ...
    - grid.write: # write only the grid component to disk and change the filename
        filename: grid.nc
    - config.write: # write only the config component to disk
```

Finally, some plugins like Wflow SBM may have defined their methods at the model and not the component level. In this case, you would not need to specify the component name before the method name. For example it could be that a specific method actually updates several components at once (e.g. `setup_basemaps` in Wflow SBM updates `config`, `staticmaps` and `geoms`).

Here is an example of a Wflow SBM workflow:

``` yaml
modeltype: wflow_sbm
global:
    data_libs:
        - artifact_data
    config_filename: wflow_sbm.toml
steps:
    - setup_basemaps: # create the model grid, basin mask and flow directions from a region
        region:
          subbasin: [12.2051, 45.8331]
          uparea: 50
        res: 0.008333
        hydrography_fn: merit_hydro_ihu
        basin_index_fn: merit_hydro_index
    - setup_rivers: # add river network
        hydrography_fn: merit_hydro_ihu
        river_geom_fn: hydro_rivers_lin
        river_upa: 30
    - staticmaps.write:
    - geoms.write:
```

As mentioned in `building <model_build>` and `updating <model_update>` a model pages, if you are using hydromt in Python, you can also use the worfklow file and the `build` and `update` methods, or just prepare your model step by step by calling the methods directly.

For example:

``` python
from hydromt import ExampleModel, log

# Configure logging
log.initialize_logging()

# Instantiate model
model = ExampleModel(
    root="./path/to/example_model",
    data_catalog=["artifact_data"],
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


---

## model region

> **Source:** [models/model_region.md](https://deltares.github.io/hydromt/latest/user_guide/models/model_region.html) · [upstream `.rst`](https://github.com/Deltares/hydromt/blob/4b6ce64a7ed082740766e466a1861860f5150322/docs/user_guide/models/model_region.rst)

To setup a model for a specific region, you can use several geospatial or hydrographic region definitions which are explained in this chapter. The syntax is interpreted using the `~hydromt.workflows.basin_mask.parse_region` method. For hydrographic regions see also the [delineate basins](../../_examples/delineate_basin.ipynb) example and `~hydromt.workflows.basin_mask.get_basin_geometry` method.

<figure class="align-center">
<img src="../../_static/region.png" alt="Examples of region types supported in HydroMT" />
<figcaption aria-hidden="true"><strong>Examples of region types supported in HydroMT</strong></figcaption>
</figure>

> [!NOTE]
> All x and y coordinates in the *point* and *bbox* are in the EPSG:4326 (WGS84) coordinate reference system. For hydrographic regions, the coordinates should be in the same CRS as the hydrography dataset used for delineation. Note that this can depend per HydroMT plugin so please check the documentation of the plugin you are using.

## Geospatial region

> Bounding box (bbox): `{'bbox': [xmin, ymin, xmax, ymax]}`
>
> Geometry file (geom): `{'geom': '/path/to/geometry_file'}`
>
> Based on another model: `{'<model_name>': '/path/to/model_root'}`
>
> Based on a raster file: `{'grid': '/path/to/raster_file'}`
>
> Based on a mesh file: `{'mesh': '/path/to/mesh_file'}`

## Hydrographic region

The following hydrographic regions are supported:

- basin
- subbasin
- interbasin

**Basin**: is defined by the entire area which drains to the sea or an inland depression. To delineate the basin(s) touching a region or point location, users can supply the following:

> - One point location: `{'basin': [x, y]}`
> - More point locations: `{'basin': [[x1, x2, ..], [y1, y2, ..]]}`
> - Bounding box: `{'basin': [xmin, ymin, xmax, ymax]}`
> - Geometry file: `{'basin': '/path/to/geometry_file'}`
> - Single unique basin ID: `{'basin': [ID1]}`
> - Several unique basin ID: `{'basin': [ID1, ID2, ..]}`
>
> To filter basins within a bounding box or geometry file, variable-threshold pairs to define streams can be used, e.g.: `'uparea':30` to filter based on streams with a minimum drainage area of 30 km2 or `'strord':8` to filter basins based on streams with a minimal stream order of 8. The variables should be available in the dataset on which the delineation is based, e.g. Hydro MERIT.
>
> - `{'basin': [xmin, ymin, xmax, ymax], '<variable>': threshold}`
>
> To only select basins with their outlet location use `'outlets': true` in combination with a bounding box or geometry file
>
> - `{'basin': [xmin, ymin, xmax, ymax], 'outlets': true}`

**Subbasin**: is defined by the area that drains into an outlet, stream or region. Users can supply the following:

> - One point location: `{'subbasin': [x, y]}`
> - More point locations: `{'subbasin': [[x1, x2, ..], [y1, y2, ..]]}`
> - Bounding box: `{'subbasin': [xmin, ymin, xmax, ymax]}`
> - Geometry file: `{'subbasin': '/path/to/geometry_file'}`
>
> Where x, y coordinates are those of the outlet(s) of the sub-basin to derive. To speed up the delineation process users can supply an estimated initial bounding box in combination with all the options mentioned above. A warning will be raised if the bounding box does not contain all upstream area.
>
> - `{'subbasin': [x, y], 'bounds': [xmin, ymin, xmax, ymax]}`
>
> The sub-basins can further be refined based on one (or more) variable-threshold pair(s) to define streams, as described above for basins. If used in combination with point outlet locations, these are snapped to the nearest stream which meets the threshold criteria.
>
> - `{'subbasin': [x, y], '<variable>': threshold}`

**Interbasin**: is defined by the area that drains into an outlet or stream and bounded by a region and therefore does not necessarily include all upstream area. Users should supply a bounding region in combination with stream and/or outlet arguments. The bounding region is defined by a bounding box or a geometry file; streams by a (or more) variable-threshold pair(s) and outlet by point location coordinates. Similar to sub-basins, point locations are snapped to nearest downstream stream if combined with stream arguments.

> - `{'interbasin': [xmin, ymin, xmax, ymax], '<variable>': threshold}`
> - `{'interbasin': /path/to/geometry_file, '<variable>': threshold}`
> - `{'interbasin': [xmin, ymin, xmax, ymax], '<variable>': threshold, 'xy': [x, y]}`
>
> To only select interbasins based on the outlet location of the entire basins use `'outlets': true`
>
> `{'interbasin': [xmin, ymin, xmax, ymax], 'outlets': true}`


---

## model components

> **Source:** [models/model_components.md](https://deltares.github.io/hydromt/latest/user_guide/models/model_components.html) · [upstream `.rst`](https://github.com/Deltares/hydromt/blob/4b6ce64a7ed082740766e466a1861860f5150322/docs/user_guide/models/model_components.rst)

> [!NOTE]
> This page is only about how to use components, not how to create a custom one. If you want to create a custom component or add default components to your custom model please refer to `custom_component` or `custom_model`

Model components are how HydroMT specifies a lot of its behaviors. Basically For HydroMT, a model is several by several components or files that you can then build and update step by step.

Each specific model and plugins will have its own set of components that it uses. Some common components could be `config`, `grid`, `forcing`, `geoms`, `states`, etc. Always visit the documentation of the specific plugin you are using to see which components it supports and what they do.

## The anatomy of a components

Components in hydromt receive a reference back to the model they are a part of, so that they can access more global properties of the model such as the root, the data catalog and the write permissions as well as other components. This means that components can't effectively exist outside of a model.

In general a component will have the following properties and functions:

- `data`: the main data object of the component. This could be a dictionary, a xarray object, a geopandas dataframe, etc. depending on the component.
- `read()`: function to read the component from disk into memory.
- `write()`: function to write the component from memory to disk.
- `set()`: function to add or update values in the component data.
- Other functions that are specific to the component and the model plugin.

## Adding components to a model

If you are using a plugin, the components should be created automatically for you. However if you are using the hydromt core standalone and its `Model` class, you will need to create and add components yourself.

There are basically two ways to add a component to a model: using the workflow yaml and using the python interface:

<div class="tab-set">

<div class="tab-item">

Workflow Yaml (CLI or Python)

In the `global` section of the workflow yaml, you can define which components the model should have, if they are spatial components and which components the model should use to define its region:

``` yaml
modeltype: model
global:
    components:
        grid:
            type: GridComponent
        config:
            type: ConfigComponent
        forcing:
            type: GridComponent
            region_component: grid
    region_component: grid
```

</div>

<div class="tab-item">

Python API

Below is an example of how to construct a components and how to add it to the model:

``` python
from hydromt.model.component import ConfigComponent, GridComponent
from hydromt.model import Model

# Prepare your components
components = {
    "config": {
        "type": ConfigComponent,
        "filename": "config.yaml",
    },
    "grid": {
        "type": GridComponent,
    },
    "forcing": {
        "type": GridComponent,
        "region_component": "grid",
    },
}

# Instantiate the model
model = Model(
    root=str("tmp"),
    data_catalog=["artifact_data"],
    mode="w",
    components=components,
    region_component="grid",
)
```

</div>

</div>

In the above examples, you can see that `components` should take a mapping where the keys are the name the component will have (e.g. `grid`). These must then again take a mapping that specifies at least the type of component. The name of the component type should correspond to the python class name (e.g. `GridComponent`).

An additional point of note is that spatial components (such as `forcing` and `grid`) in the examples above, can either define their own region (`grid`) or derive their region from another component (`forcing`). This can be done by specifying the `region_component` key, and should refer to the name of the spatial component you wish to use. You can also specify which spatial component the model should derive it's region from.


---

## model processes

> **Source:** [models/model_processes.md](https://deltares.github.io/hydromt/latest/user_guide/models/model_processes.html) · [upstream `.rst`](https://github.com/Deltares/hydromt/blob/4b6ce64a7ed082740766e466a1861860f5150322/docs/user_guide/models/model_processes.rst)

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
