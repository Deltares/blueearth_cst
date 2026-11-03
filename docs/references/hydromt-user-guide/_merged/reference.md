---
title: HydroMT user guide — Terminology and FAQ
ingest-source: hydromt-user-guide
upstream-repo: Deltares/hydromt
upstream-commit: 4b6ce64a7ed082740766e466a1861860f5150322
pulled: 2026-05-04
section: reference
doc-type: user-guide
license: MIT
sources:
  - https://deltares.github.io/hydromt/latest/user_guide/terminology.html
  - https://deltares.github.io/hydromt/latest/user_guide/faq.html
---

# HydroMT user guide — Terminology and FAQ

_Merged from 2 upstream page(s) at commit `4b6ce64a`, pulled 2026-05-04. Page boundaries are preserved as `## ` headings._


---

## terminology

> **Source:** [terminology.md](https://deltares.github.io/hydromt/latest/user_guide/terminology.html) · [upstream `.rst`](https://github.com/Deltares/hydromt/blob/4b6ce64a7ed082740766e466a1861860f5150322/docs/user_guide/terminology.rst)

HydroMT and this documentation use a specific terminology to describe specific objects or processes.

| Term | Explanation |
|----|----|
| Command Line Interface (CLI) | high-level interface to HydroMT *build*, *update*, *check* and *export* methods. |
| Workflow file (HydroMT) | (.yaml) file describing the complete pipeline with all methods and their arguments to *build* or *update* a model. |
| Data catalog | A set of data sources available for HydroMT. It is build up from *yaml* files containing one or more data sources with information about how to read and optionally preprocess the data and contains meta-data about the data source. |
| Data source | Input data to be processed by HydroMT. Data sources are listed in yaml files. |
| Model | A set of files describing the schematization, forcing, states, simulation configuration and results for any supported model kernel and model classes. The final set of files is dependent on the model type (grid, vector or mesh model for examples) or the model plugin. |
| Model class | A model instance can be instantiated from different model schematization concepts. Generalized model classes currently supported within HydroMT are Grid Model (distributed models), vector Model (semi-distributed), Mesh Model (unstructured) and in the future Network Model (relational model). Specific model classes for specific softwares have been implemented as plugins, see Model plugin. |
| Model attributes | Direct properties of a model, such as the model root. They can be called when using HydroMT from python. |
| Model component | A building block of a model. Model components are usually linked to a specific model file or input data. Examples are config, grid, vector, mesh, forcing, states, parameters, etc. HydroMT provides generic model component classes but plugins can define their own. |
| Model plugin | (Plugin) Package that links the HydroMT Model class to a specific model software so that HydroMT can build and update models and analyze its simulation results. For example *HydroMT-Wflow*, *HydroMT-SFINCS* etc. Plugins are installed separately from HydroMT and are not part of the HydroMT core package. Plugins are the most common way of using HydroMT to build and update specific models. |
| Model kernel | The model software to execute a model simulation. This is *not* part of any HydroMT plugin. |
| Region | Argument of the one of the model setup methods that specifies the region of interest where the model should be prepared / which spatial subregion should be clipped. |


---

## faq

> **Source:** [faq.md](https://deltares.github.io/hydromt/latest/user_guide/faq.html) · [upstream `.rst`](https://github.com/Deltares/hydromt/blob/4b6ce64a7ed082740766e466a1861860f5150322/docs/user_guide/faq.rst)

This page contains some FAQ / tips and tricks to work with HydroMT.

## Working with models in HydroMT

> **Q**: Does HydroMT contain any model kernels/software to run model simulations?

HydroMT focusses on the setup of models and analysis of model simulations, but does not contain the model software itself. In between the setup and analysis the model software needs to be executed to run a model simulation.

> **Q**: Can I re-use the same method when building / updating a model from the command line interface with an .yaml configuration file.

Yes, that is possible. In the YAML file, each setup or update method is listed under the steps section. So you can easily repeat the same method multiple times with different arguments. For example:

``` yaml
steps:
  - config.update:
      forcing_file: era5_2010.nc
  - setup_precip_forcing:
      precip_data: "era5"
      start: 2010-01-01
      end: 2010-12-31
  - forcing.write:
  - config.update:
      forcing_file: chirps_2010.nc
  - setup_precip_forcing:
      precip_data: "chirps"
      start: 2010-01-01
      end: 2010-12-31
  - forcing.write:
```

Here `config.update`, `setup_precip_forcing` and `forcing.write` are called several times to create several forcing files in one go.

> **Q**: How can I just write specific model data component (i.e.: grid, geoms, forcing, config or states) instead of the all model data when updating?

Each model plugin implements a combined `write()` method that writes the entire model and is called by default at the end of a `build` or `update`. If you however add a write method (e.g. `grid.write` for a Grid model, `forcing.write`, `config.write`, etc.) to the .yaml file the call to the general write method is disabled and only the selected model data attributes are written.

> **Q**: Why is there no logging output in my terminal when I run my Python hydromt scripts?

HydroMT does not configure logging automatically when used as a Python library. This is different from the command line interface (CLI), where logging is initialized for you. If you run HydroMT from a Python script without configuring logging, no log messages will be shown in the terminal, even though the code is executing correctly. To enable logging output, you need to explicitly initialize logging in your script:

``` python
from hydromt import log

log.initialize_logging()
```

This sets up a default logging configuration and ensures that messages are printed to the console. If you need more advanced logging behavior (for example logging to a file or integrating with an existing application logger), review the module hydromt.log for your options or you should configure the Python logging module yourself before using HydroMT.

## Working with data in HydroMT

> **Q**: Does HydroMT contain (global) datasets which can be used to build/update models?

HydroMT does not contain any datasets. A small spatial subset for the Piave basin in northern Italy of some data that is often used in combination with HydroMT is made available for testing purposes. The data will automatically be downloaded to the "~/.hydromt" folder on your machine if no other data catalogs are provided. See also `Working with data in HydroMT <get_data>` page. We are working on creating more data catalogs from (cloud optimized analysis read) open data sources.

> **Q**: Can I supply my own data to HydroMT?

Yes, absolutely! Checkout the `Preparing a data catalog <own_catalog>` page in the user guide.
