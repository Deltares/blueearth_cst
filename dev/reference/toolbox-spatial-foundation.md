# Shared spatial foundation

Last updated: 2026-10-20

Use this reference when changing spatial ownership, region reuse, location identities or basin maps.
`blueearth_cst/spatial/` owns the implementation; the [implementation map](toolbox-workflow-map.md) points
to rule definitions and consumer checks.

## Ownership and dependency direction

Delineate the project region once from basin settings and the catalog. Model-free climate extraction,
projection analysis and spatial products consume that same region artifact. They must not read a built
Wflow model to obtain their extent.

The shared vector foundation supplies basins, subbasins, rivers and the location registry. WF1 alone
prepares thematic raster products needed to parameterize Wflow. Keep model-neutral producers independent
of land/soil inputs they do not consume; changing a model build must not manufacture a new region.

`delineate_region.py`, `delineate_spatial_units.py`, `products.py` and `config.py` in
`blueearth_cst/spatial/` own the producer contracts. Their shared rule parameters must describe their
actual spatial inputs, not unrelated workflow selection or model settings.

`hydrography.nc` carries the grid stack between the vector and thematic stages. Preserve its identifier
types and nodata semantics when reading it. It is deliberately absent from the model-input catalog:
the thematic stage alone writes the complete `spatial_catalog.yml`, avoiding two writers of one catalog.
`products.py` and `prepare_spatial_maps.py` own that boundary.

## Spatial identities and controls

`blueearth_cst/spatial/identity.py` owns stable basin, subbasin and location identifiers and capacity
validation. Do not copy its encoding tables into another registry. `spatial/config.py` owns configuration
translation, while the project schema and `config/advanced_settings.yml` own admitted settings/defaults.

The foundation resolves each parent feature independently and snaps configured controls to the analysis
river network. Automatic partitioning supplies fallback parents when controls do not prescribe a split.
`basin.delineation.max_subbasins` is a ceiling applied independently to each automatic parent partition,
not a project-wide budget. Configuration and identity code enforce its bounds and location capacities;
read those owners for exact defaults and limits rather than historical configuration names.

The persisted location registry preserves stable spatial identifiers. Model-specific gauge positions and
outlet-index mappings are downstream representations; do not substitute positional model output labels
for the foundation's identity. The [WF1 contract](wf1-model-construction.md) describes those mappings.

## Basin maps

Draw the basin-area map from the shared spatial foundation, not from `staticmaps.nc` or a model-dependent
copy of the geometry. Physical waterbody layers are clipped from catalog sources by
`spatial.products.clip_waterbodies`; the hydrological model keeps its own filtering and setup behavior.
Empty layers and missing-source warnings follow that producer's implementation.

`blueearth_cst/shared/plot_spatial_maps.py` owns the map reader. Figure layout and export conventions belong in
[plot style](toolbox-plot-style.md), rather than being repeated here.

## Verification entry points

Use `tests/test_spatial_identity.py`, `tests/test_spatial_config.py`, `tests/test_spatial_products.py`
and `tests/test_spatial_units_rule.py` for their respective contracts.
`tests/test_cross_workflow_inputs.py` checks workflow dependency boundaries.
Select the applicable checks through the [validation ladder](dev-validation.md).

For historical rationale and the original bounded timing/identity measurements, see
0003-one-shared-region-artifact.md (private-checkout-only citation:
`../records/decision-history/0003-one-shared-region-artifact.md`). Those observations do not set today's
configuration or certify a new basin.
