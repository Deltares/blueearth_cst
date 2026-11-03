---
title: data_existing_cat
ingest-source: hydromt-user-guide
source: https://deltares.github.io/hydromt/latest/user_guide/data_catalog/data_existing_cat.html
repo-source: https://github.com/Deltares/hydromt/blob/4b6ce64a7ed082740766e466a1861860f5150322/docs/user_guide/data_catalog/data_existing_cat.rst
upstream-repo: Deltares/hydromt
upstream-commit: 4b6ce64a7ed082740766e466a1861860f5150322
pulled: 2026-05-04
doc-type: user-guide
license: MIT
---
# Pre-defined data catalogs

This page contains a list of (global) datasets which can be used with various HydroMT models and workflows. Below are drop down lists with datasets per pre-defined data catalog for use with HydroMT. The summary per dataset contains links to the online source and available literature.

The `deltares_data` catalog is only available within the Deltares network. However a selection of this data for a the Piave basin (Northern Italy) is available online in the `artifact_data` archive and will be used if no data catalog is provided. Local or other datasets can also be included by extending the data catalog with new yaml `data catalog files <data_yaml>`. We plan to provide more data catalogs with open data sources in the (near) future. See the data catalog [changelog](https://github.com/Deltares/hydromt/blob/main/data/catalogs/changelog.rst) for recent updates on the pre-defined catalogs.

## Using a predefined catalog

<div class="tab-set">

<div class="tab-item">

Command Line Interface (CLI)

To use a predefined catalog, you can specify the catalog name with the `-d` or `--data` option when running a HydroMT command. For example, to use the `deltares_data` catalog with the <span class="title-ref">hydromt build</span> command, you can run the following:

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

Once you have set the data catalog you can specify the data source(s) for each method in the HydroMT `model workflow file <model_workflow>` as shown in the example below with the <span class="title-ref">setup_precip_forcing</span> method.

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

</div>

<div class="tab-item">

Python API

To use a predefined catalog in Python, you can specify the catalog name with the `data_libs` argument when initializing a `DataCatalog` class. You can specify a data catalog version by adding the version number after the catalog name. You can then get data from the catalog using the `DataCatalog.get_rasterdataset` or other :ref: <span class="title-ref">DataCatalog methods</span>.

``` python
from hydromt import DataCatalog
data_catalog = DataCatalog(data_libs=["deltares_data"])
# specify a data catalog version
data_catalog = DataCatalog(data_libs=["deltares_data=v1.0.0"])
# get data from the catalog
ds = data_catalog.get_rasterdataset("eobs") # get the most recently added
ds = data_catalog.get_rasterdataset("eobs", version="22.0e") # get a specific version
```

</div>

</div>

## Available pre-defined data catalogs

### Deltares data catalog

Data available for Deltares colleagues (p: drive). For non Deltares users, you can use it as inspiration to create your own. The catalog and it's different versions can be viewed here: <https://github.com/Deltares/hydromt/tree/main/data/catalogs/deltares_data>

Available data:

### Artifact data catalog

Global data extract around the Piave basin in Northern Italy used for documentation, training and testing of HydroMT. The catalog and its different versions can be viewed here: <https://github.com/Deltares/hydromt/tree/main/data/catalogs/artifact_data>

Available data:

### AWS data catalog

Data openly available in Amazon s3 bucket. The catalog and its different versions can be viewed here: <https://github.com/Deltares/hydromt/tree/main/data/catalogs/aws_data>

Available data:

### GCS CMIP6 data catalog

CMIP6 dataset openly available and stored on a public Google Cloud Store. The catalog and its different versions can be viewed here: <https://github.com/Deltares/hydromt/tree/main/data/catalogs/gcs_cmip6_data>

Available data:

### Earth Data Hub data catalog

Data stored in <span class="title-ref">Earth Data Hub \<https://earthdatahub.destine.eu/\></span> (Destination Earth). In order to use this catalog, you need to setup credentials for accessing the data on Earth Data Hub. This includes creating an account on Earth Data Hub and setting up a .netrc file with your credentials. You can find more information on how to do this in the [Earth Data Hub documentation](https://earthdatahub.destine.eu/getting-started#configuring-netrc).

The catalog and its different versions can be viewed here: <https://github.com/Deltares/hydromt/tree/main/data/catalogs/earthdatahub_data>

Available data:
