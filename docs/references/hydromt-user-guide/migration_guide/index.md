---
title: index
ingest-source: hydromt-user-guide
source: https://deltares.github.io/hydromt/latest/user_guide/migration_guide/index.html
repo-source: https://github.com/Deltares/hydromt/blob/4b6ce64a7ed082740766e466a1861860f5150322/docs/user_guide/migration_guide/index.rst
upstream-repo: Deltares/hydromt
upstream-commit: 4b6ce64a7ed082740766e466a1861860f5150322
pulled: 2026-05-04
doc-type: user-guide
license: MIT
---
# Migration Guide

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
