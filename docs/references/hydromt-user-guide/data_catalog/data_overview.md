---
title: data_overview
ingest-source: hydromt-user-guide
source: https://deltares.github.io/hydromt/latest/user_guide/data_catalog/data_overview.html
repo-source: https://github.com/Deltares/hydromt/blob/4b6ce64a7ed082740766e466a1861860f5150322/docs/user_guide/data_catalog/data_overview.rst
upstream-repo: Deltares/hydromt
upstream-commit: 4b6ce64a7ed082740766e466a1861860f5150322
pulled: 2026-05-04
doc-type: user-guide
license: MIT
---
# Overview data

The best way to provide data to HydroMT is by using a **data catalog**. The goal of this data catalog is to provide simple and standardized access to (large) datasets. It supports many drivers to read different data formats and contains several pre-processing steps to unify the datasets. A data catalog can be initialized from one or more **yaml file(s)**, which contain all required information to read and pre-process a dataset, as well as meta data for reproducibility.

You can `explore and make use of pre-defined data catalogs <existing_catalog>` (primarily global data), `prepare your own data catalog <own_catalog>` (e.g. to include local data) or use a combination of both.

> [!TIP]
> If no yaml file is provided to the CLI build or update methods or to `~hydromt.data_catalog.DataCatalog`, HydroMT will use the data stored in the `artifact_data <existing_catalog>` which contains an extract of global data for a small region around the Piave river in Northern Italy.

> [!TIP]
> Tiles of tiled rasterdatasets which are described by a .vrt file can be cached locally. The requested data tiles will by default be stored to ~/.hydromt_data. To use this option from command line add <span class="title-ref">--cache</span> to the <span class="title-ref">hydromt build</span> or <span class="title-ref">hydromt update</span> commands. In Python the cache is a property of the DataCatalog and can be set at Initialization.

## Using a data catalog

<div class="tab-set">

<div class="tab-item">

Command Line Interface (CLI)

When using the HydroMT command line interface (CLI), one can provide a data catalog by specifying the path to the yaml file with the `-d (--data)` option. Alternatively, you can also use names and versions of the `predefined data catalogs <existing_catalog>`. If no version is specified, the latest version available is used.

``` console
hydromt build MODEL -d /path/to/data-catalog.yml
```

</div>

<div class="tab-item">

Python API

Initialize a `~hydromt.data_catalog.DataCatalog` with references to user- or pre-defined data catalog yaml files

``` python
import hydromt
data_cat = hydromt.DataCatalog(data_libs=r'/path/to/data-catalog.yml')
```

You can find examples of how to read data from a catalog in Python in `this page <hydromt_data_read_python>`.

</div>

</div>

<div class="toctree" hidden="">

Pre-defined data catalogs \<data_existing_cat\> Preparing a data catalog \<data_prepare_cat\> Supported data types \<data_types\> Data conventions \<data_conventions\> Cloud storage \<data_cloud_storage\>

</div>
