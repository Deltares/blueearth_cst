---
title: model_build
ingest-source: hydromt-user-guide
source: https://deltares.github.io/hydromt/latest/user_guide/models/model_build.html
repo-source: https://github.com/Deltares/hydromt/blob/4b6ce64a7ed082740766e466a1861860f5150322/docs/user_guide/models/model_build.rst
upstream-repo: Deltares/hydromt
upstream-commit: 4b6ce64a7ed082740766e466a1861860f5150322
pulled: 2026-05-04
doc-type: user-guide
license: MIT
---
# Building a model

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
