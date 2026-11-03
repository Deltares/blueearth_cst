---
title: "CMIP6 catalog"
format: html
---

Projection analysis uses its own catalog, selected by `catalog` in the projection settings file.
The current catalog, `toolbox:config/catalogs/cmip6_data.yml`, describes CMIP6 data in the
[Pangeo archive on Google Cloud](https://console.cloud.google.com/storage/browser/cmip6/CMIP6).
It is separate from the project catalog used for spatial data and historical meteorology.

## Read a projection entry

The beginning of an entry in [cmip6_data.yml](../../config/catalogs/cmip6_data.yml) identifies the
model, experiment and cloud location:

```yaml
cmip6_AS-RCEC/TaiESM1_historical_{member}: &cmip6_amon
  data_type: RasterDataset
  uri: gs://cmip6/CMIP6/CMIP/AS-RCEC/TaiESM1/historical/{member}/Amon/{variable}/*/*
  driver:
    name: raster_xarray
```

`{member}` and `{variable}` are filled from the selected ensemble member and climate variable.
`Amon` identifies monthly atmospheric output. The full entry provides the available members,
reader options and conversions, including precipitation to daily rates and temperature to degrees Celsius.

## Availability and updates

The catalog is generated together with
[cmip6_store_index.json](../../config/catalogs/cmip6_store_index.json), which records available members
and grid/version locations. Listed members have both monthly precipitation (`pr`) and near-surface
air temperature (`tas`); other variables are not guaranteed. Check coverage for the selected models
and periods rather than assuming every combination is available.

Both files must describe the same crawl of the archive. Regenerate them together using
`dev/scripts/generate_cmip6_catalog.py`; do not edit either file by hand. See
[projection analysis](workflow-analyze-projections.md) for selection and calculation details.
