---
title: model_update
ingest-source: hydromt-user-guide
source: https://deltares.github.io/hydromt/latest/user_guide/models/model_update.html
repo-source: https://github.com/Deltares/hydromt/blob/4b6ce64a7ed082740766e466a1861860f5150322/docs/user_guide/models/model_update.rst
upstream-repo: Deltares/hydromt
upstream-commit: 4b6ce64a7ed082740766e466a1861860f5150322
pulled: 2026-05-04
doc-type: user-guide
license: MIT
---
# Updating a model

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
