---
title: "HydroMT data catalog"
format: html
---

The HydroMT data catalog provides access to spatial datasets and historical meteorology used to prepare
the basin model and generate weather. The supplied project starter selects
`toolbox:config/catalogs/deltares_data_pdrive.yml` through `project.catalog`.
Access to the underlying Deltares archive is required.

## Locate the archive

This excerpt from [deltares_data_pdrive.yml](../../config/catalogs/deltares_data_pdrive.yml) lists
alternative mount paths for the same archive. HydroMT uses the first available root.

```yaml
meta:
  roots:
    - p:/wflow_global/hydromt
    - /mnt/p/wflow_global/hydromt
    - /p/wflow_global/hydromt
  version: v1.2.0
  hydromt_version: ">=1.0a, <2"
  name: deltares_data
```

## Read a dataset definition

The beginning of the `globcover` entry identifies the land-cover version, file and reader:

```yaml
globcover:
  data_type: RasterDataset
  version: 2.3
  uri: landuse/globcover/GLOBCOVER_L4_200901_200912_V2.3.tif
  driver:
    name: rasterio
```

The full entry also supplies reader options and metadata. In the project file, `basin.sources`
selects spatial entries, `climate.sources` lists candidates, and `climate.forcing.precip_source` chooses
forcing for historical simulation and weather generation. These names must match catalog entries.

## Other bundled variants

- [deltares_data.yml](../../config/catalogs/deltares_data.yml) describes a local archive rooted at
  `C:\data\wflow_global\hydromt`. It is a different catalog version, with some different source names;
  changing catalogs therefore requires checking the project's selections, not just the storage path.
- [deltares_data_linux.yml](../../config/catalogs/deltares_data_linux.yml) contains an older schema
  using `path`, `kwargs` and a singular `meta.root`. Do not select it merely because you use Linux:
  the P-drive catalog above already lists Linux mount paths and uses the current catalog structure.

For example, the local catalog uses `rivers_lin2019_v1`, while the P-drive catalog uses
`hydro_rivers_lin`. For an archive you maintain yourself, follow
[Use your own data archive](data-catalogs-and-access.md#use-your-own-data-archive).
