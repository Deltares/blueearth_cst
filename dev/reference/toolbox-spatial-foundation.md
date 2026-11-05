# Shared spatial foundation

Last updated: 2026-11-05

Use this reference when changing the shared region, spatial products or location identities. The
[architecture reference](toolbox-architecture.md) explains workflow responsibilities; the
[workflow map](toolbox-workflow-map.md) locates rules and consumers.

## Shared region and product ownership

The project region comes from `basin` settings and the project data catalog. Model-free climate
extraction, projection analysis and spatial products use that region; they must not obtain their
extent from a built Wflow model. [`delineate_region.py`](../../blueearth_cst/spatial/delineate_region.py)
owns region creation. Its rule parameters must track the spatial inputs it actually reads, without
adding unrelated workflow selection or model settings.

The vector stage produces basins, subbasins, rivers and a location registry. WF1 adds the thematic
land and soil rasters needed to parameterize Wflow. Keep the shared region and vector producers
independent of thematic inputs they do not consume. The implementations are
[`delineate_spatial_units.py`](../../blueearth_cst/spatial/delineate_spatial_units.py) and
[`products.py`](../../blueearth_cst/spatial/products.py); the [WF1 reference](wf1-model-construction.md)
owns model construction.

`hydrography.nc` carries the grid stack between vector and thematic stages. Preserve its identifier
types and nodata semantics when reading it. It is deliberately absent from the model-input
`spatial_catalog.yml`: the thematic writer alone publishes the complete catalog, so the vector and
thematic stages never write competing versions. The writer is
[`prepare_spatial_maps.py`](../../blueearth_cst/spatial/prepare_spatial_maps.py), using `products.py`.

## Spatial identities and controls

[`identity.py`](../../blueearth_cst/spatial/identity.py) assigns stable basin, subbasin and location
identifiers and checks their capacity. Basin IDs follow area and outlet-grid ordering; subbasin and
location IDs remain deterministic under input row reordering. Do not duplicate the encoding in another
registry or substitute positional model labels for these persisted IDs. Changing capacity requires an
identity-format migration: the current format allows at most 99 local subbasins per parent basin and
nine additional, non-primary locations per subbasin.

The foundation resolves each parent basin independently and snaps configured controls to the analysis
river network. Automatic partitioning supplies a fallback where controls do not prescribe a split.
`basin.delineation.max_subbasins` limits each automatic parent partition separately; it is not a
project-wide total. [`spatial/config.py`](../../blueearth_cst/spatial/config.py) validates that ceiling,
snap tolerance and river threshold. The project config and
[`advanced_settings.yml`](../../config/advanced_settings.yml) own admitted values and defaults.
The [WF1 output mapping](wf1-model-construction.md#outputs-and-consumers) explains how model gauge
positions connect back to these identifiers.

## Basin maps

Draw the basin-area map from the shared spatial foundation, not from `staticmaps.nc` or a
model-dependent geometry copy. [`products.py`](../../blueearth_cst/spatial/products.py) clips physical
waterbody layers from catalog sources and handles empty or missing layers. Wflow retains its own
waterbody filtering and setup. [`plot_spatial_maps.py`](../../blueearth_cst/shared/plot_spatial_maps.py)
reads the spatial products; [plot style](toolbox-plot-style.md) owns layout and export conventions.

## Verification entry points

Use [`test_spatial_identity.py`](../../tests/test_spatial_identity.py) for stable identifiers and
capacity, [`test_spatial_config.py`](../../tests/test_spatial_config.py) for configuration bounds,
[`test_spatial_products.py`](../../tests/test_spatial_products.py) for products, and
[`test_spatial_units_rule.py`](../../tests/test_spatial_units_rule.py) for shared rule inputs.
[`test_cross_workflow_inputs.py`](../../tests/test_cross_workflow_inputs.py) checks dependency direction.
Choose affected checks through the [validation ladder](dev-validation.md); these tests do not by
themselves establish performance or adequacy for a new basin.
