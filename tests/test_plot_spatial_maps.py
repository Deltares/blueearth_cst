# -*- coding: utf-8 -*-
"""Unit tests for the spatial-map figure family and the categorical raster path.

The categorical drawing these figures rely on (present-only keys, unclassified
codes, nodata) is tested against the template in ``tests/test_prepared_maps.py``.

The REGISTRY in ``shared.plot_spatial_maps`` is a table of decisions — which
layer, which palette, which unit — and what is tested is the invariants that
would make a figure wrong while still looking right: a label whose unit
contradicts its scale factor, a nominal layer routed onto a ramp, a duplicate
output stem quietly overwriting another figure.

The rendering itself is not asserted on here. A figure is verified by looking at
it (AGENTS.md, "Figures are terminal artifacts"); what these tests protect is
everything that decides WHAT gets drawn.
"""

import numpy as np
import pytest
import xarray as xr

from blueearth_cst.shared import plot_spatial_maps as family
from blueearth_cst.shared.map_frames import map_caption
from blueearth_cst.shared.raster_style import RasterStyle


def _codes(values, nx=4):
    """A small nominal raster carrying ``values``, on a geographic grid."""
    array = np.array(values, dtype="float64").reshape(-1, nx)
    return xr.DataArray(
        array,
        dims=("latitude", "longitude"),
        coords={
            "latitude": np.arange(array.shape[0]) * 0.01,
            "longitude": np.arange(nx) * 0.01,
        },
    )


# --- the registry -------------------------------------------------------------


def test_output_stems_are_unique():
    """Two entries sharing a stem would silently overwrite one another."""
    stems = [figure.stem for figure in family.SPATIAL_MAP_FIGURES]
    assert len(stems) == len(set(stems))


def test_no_stem_collides_with_the_basin_map_in_the_same_folder():
    """One rule writes both into ``data/spatial/plots``."""
    assert "elevation_basin" not in {
        figure.stem for figure in family.SPATIAL_MAP_FIGURES
    }


def test_only_the_surface_soil_slice_is_registered():
    """Soil stems carry no depth token, so a second slice would be ambiguous."""
    for figure in family.SPATIAL_MAP_FIGURES:
        if figure.variable.startswith("soil_") and "_sl" in figure.variable:
            assert figure.variable.endswith("_sl1"), figure.variable
            assert figure.stem.endswith("_basin"), figure.stem


def test_the_figure_set_declares_exactly_what_the_rule_promises():
    """The Snakefile builds rule 1.12's outputs from this list, so a registry
    edit that is not matched by a re-run leaves the rule promising files that
    are never written. Pinning the count makes the edit visible in review."""
    assert len(family.SPATIAL_MAP_FIGURES) == 10


def test_only_source_stable_variables_are_declared_as_rule_outputs():
    """A declared output the data cannot produce fails the RULE, not the figure.

    ``soil_BDTICM_M_250m_ll`` is soilgrids v1.0's own filename -- the catalog
    does not rename it, and soilgrids_2020 has no equivalent entry -- so a
    project on the other soil source would stop with "missing output files".
    """
    ungiven = {f.variable for f in family.SPATIAL_MAP_FIGURES if not f.guaranteed}
    assert ungiven == {"soil_BDTICM_M_250m_ll"}


def test_the_rule_output_list_is_one_png_per_declared_figure():
    """PNG only since 2026-08-10 — the vector copy went unread. `formats` is
    still a parameter, so a caller preparing a manuscript can ask for a PDF."""
    paths = family.figure_paths("P/plots")
    declared = [f for f in family.SPATIAL_MAP_FIGURES if f.guaranteed]
    assert len(paths) == len(declared)
    assert "P/plots/land_cover_basin.png" in paths
    assert not any(path.endswith(".pdf") for path in paths)
    assert "P/plots/soil_depth_to_bedrock_basin.png" not in paths


def test_a_caller_can_still_ask_for_a_pdf():
    paths = family.figure_paths("P/plots", formats=("png", "pdf"))
    assert "P/plots/land_cover_basin.pdf" in paths


def test_the_undeclared_figures_are_still_drawn():
    """`guaranteed=False` keeps a figure out of the rule's promise, not out of
    the run -- it is the difference between a lost figure and a failed workflow."""
    everything = family.figure_paths("P/plots", declared_only=False)
    assert "P/plots/soil_depth_to_bedrock_basin.png" in everything


def test_a_constant_layer_is_noted_but_still_drawn():
    """Skipping it would make the rule's promise conditional on the data: a
    project whose bedrock depth is uniform would fail on a missing output."""
    maps = _dataset()
    maps["land_cover"] = (("latitude", "longitude"), np.full((2, 2), 30.0))
    figure = family.SpatialFigure("land_cover", "s", classes=())
    layer = family.prepare_layer(maps, figure, None)
    assert family._is_degenerate(layer)[0] is True  # reported...
    assert np.isfinite(layer.values).all()  # ...and still a drawable field


def test_too_many_subbasins_drops_the_key_but_keeps_the_fill():
    """A forty-row legend beside a 40 mm panel explains nothing a reader wanted;
    the colours still separate the units, which is what the figure is for."""
    codes = list(range(101, 101 + family._MAX_LEGEND_SUBBASINS + 5))
    classes = family.subbasin_classes(_codes(codes, nx=1))
    assert len(classes) == len(codes)
    assert all(label is None for _, _, label in classes)


def test_variables_are_unique():
    stems = [figure.variable for figure in family.SPATIAL_MAP_FIGURES]
    assert len(stems) == len(set(stems))


def test_every_entry_is_either_continuous_or_nominal_never_both():
    """A ramp AND a class table would leave which one wins to argument order."""
    for figure in family.SPATIAL_MAP_FIGURES:
        assert (figure.style is None) != (figure.classes is None), figure.variable


def test_every_continuous_style_names_a_resolvable_palette():
    """A typo in a colormap name would surface as a KeyError mid-render."""
    from blueearth_cst.shared.map_classes import _style_colormap

    for figure in family.SPATIAL_MAP_FIGURES:
        if figure.style is not None:
            assert _style_colormap(figure.style, 5).N == 5, figure.variable


def test_a_rescaled_layer_states_the_rescaled_unit_on_its_label():
    """A scale factor and a unit string that disagree look right and are wrong."""
    depth = next(
        figure
        for figure in family.SPATIAL_MAP_FIGURES
        if figure.variable == "soil_BDTICM_M_250m_ll"
    )
    assert depth.scale == 0.01
    assert "(m)" in depth.style.label and "cm" not in depth.style.label


def test_the_land_cover_table_is_the_products_own_legend():
    """Codes and colours are Copernicus'. A drifted colour un-recognises the map."""
    table = dict((code, colour) for code, colour, _ in family.LAND_COVER_CLASSES)
    assert table[50] == "#fa0000"  # urban / built-up
    assert table[80] == "#0032c8"  # permanent water
    assert table[112] == "#009900"  # closed forest, evergreen broadleaf
    assert len(family.LAND_COVER_CLASSES) == 23


def test_the_land_cover_codes_are_unique_and_ordered():
    for table in (family.LAND_COVER_CLASSES, family.GLOBCOVER_CLASSES):
        codes = [code for code, _, _ in table]
        assert codes == sorted(set(codes))


def test_the_globcover_table_is_the_products_own_legend():
    table = {code: colour for code, colour, _ in family.GLOBCOVER_CLASSES}
    assert table[40] == "#006400"  # broadleaved evergreen forest
    assert table[210] == "#0046c8"  # water bodies
    assert len(family.GLOBCOVER_CLASSES) == 23


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("globcover", "GLOBCOVER_CLASSES"),
        ("globcover_2009_v2.3", "GLOBCOVER_CLASSES"),
        ("vito", "LAND_COVER_CLASSES"),
        ("vito_2019_v3.0.1", "LAND_COVER_CLASSES"),
    ],
)
def test_the_land_cover_table_follows_the_layer_source(source, expected):
    assert family.land_cover_classes(_layer(source)) is getattr(family, expected)


def test_globcover_code_40_is_forest_not_cropland():
    """The defect this table exists for: GlobCover drawn with Copernicus codes."""
    labels = {
        code: label for code, _, label in family.land_cover_classes(_layer("globcover"))
    }
    assert "evergreen" in labels[40] and "Cropland" not in labels[40]


def test_an_unknown_land_cover_source_is_drawn_unclassified_with_a_warning():
    with pytest.warns(RuntimeWarning, match="no classification table is registered"):
        table = family.land_cover_classes(_layer("esa_worldcover"))
    assert [code for code, _, _ in table] == [-1]


# --- the caption skeleton and source credit ------------------------------------


def _layer(source):
    return xr.DataArray(np.zeros((2, 2)), dims=("y", "x"), attrs={"source": source})


def test_the_caption_credits_the_product_not_the_catalog_key():
    """ "vito" is our plumbing; a figure credits Copernicus."""
    assert map_caption("Lead", "soilgrids") == "Lead from SoilGrids (ISRIC, 2017)."
    assert "Copernicus" in map_caption("Lead", "vito")


def test_every_source_the_maps_can_meet_has_a_credit():
    """A layer whose catalog key is unmapped falls back to the key itself."""
    for key in (
        "merit_hydro_ihu",
        "vito",
        "globcover",
        "modis_lai",
        "soilgrids",
        "era5",
        "chirps",
    ):
        assert f"from {key}." not in map_caption("Lead", key)


def test_an_unmapped_source_still_gets_a_line():
    """Ugly and visible beats silently unattributed -- the first gets fixed."""
    assert map_caption("Lead", "some_new_catalog_entry") == (
        "Lead from some_new_catalog_entry."
    )


def test_a_map_caption_is_one_sentence_plus_needed_notes():
    assert family._spatial_caption("Basin topsoil pH (H₂O)", _layer("soilgrids")) == (
        "Basin topsoil pH (H₂O) from SoilGrids (ISRIC, 2017)."
    )
    unsourced = xr.DataArray(np.zeros((2, 2)), dims=("y", "x"))
    assert family._spatial_caption("Lead", unsourced, "A note") == "Lead. A note."
    assert map_caption("Lead", notes=["", "Caveat."]) == "Lead. Caveat."


# --- layer preparation --------------------------------------------------------


def _dataset():
    """A 2x2 stand-in for ``spatial_maps.nc``: one masked layer, one seasonal."""
    coords = {"latitude": [0.0, 0.01], "longitude": [0.0, 0.01]}
    return xr.Dataset(
        {
            "subbasin_id": (("latitude", "longitude"), np.array([[1, 1], [0, 2]])),
            "land_cover": (
                ("latitude", "longitude"),
                np.array([[30.0, 40.0], [50.0, 60.0]]),
            ),
            "leaf_area_index": (
                ("month", "latitude", "longitude"),
                np.stack([np.full((2, 2), value) for value in (1.0, 3.0)]),
            ),
        },
        coords={**coords, "month": [1, 2]},
    )


def test_a_thematic_layer_is_clipped_to_the_basin():
    """The thematic rasters arrive clipped to the BOUNDING BOX, not the basin.

    Unmasked, a land-cover map paints a full rectangle and the basin outline
    reads as a box drawn over a larger dataset.
    """
    maps = _dataset()
    figure = family.SpatialFigure("land_cover", "s", classes=())
    layer = family.prepare_layer(maps, figure, family._basin_mask(maps))
    assert np.isnan(layer.values[1, 0])  # the cell where subbasin_id == 0
    assert layer.values[0, 0] == 30.0


def test_a_hydrography_layer_is_left_alone():
    maps = _dataset()
    figure = family.SpatialFigure("land_cover", "s", classes=(), mask_to_basin=False)
    layer = family.prepare_layer(maps, figure, family._basin_mask(maps))
    assert np.isfinite(layer.values).all()


def test_a_seasonal_layer_is_reduced_to_its_annual_mean():
    """Stated in the FILENAME, because a silent seasonal average is not assumable."""
    maps = _dataset()
    figure = family.SpatialFigure("leaf_area_index", "s", RasterStyle("x", "YlGn"))
    layer = family.prepare_layer(maps, figure, None)
    assert layer.dims == ("latitude", "longitude")
    assert layer.values.max() == pytest.approx(2.0)


def test_a_constant_layer_is_reported_as_degenerate():
    """Which layers are flat depends on the basin, so this is decided per render."""
    flat = xr.DataArray(np.full((2, 2), 7.0), dims=("y", "x"))
    degenerate, reason = family._is_degenerate(flat)
    assert degenerate and "constant" in reason


def test_an_empty_layer_is_reported_as_degenerate():
    empty = xr.DataArray(np.full((2, 2), np.nan), dims=("y", "x"))
    degenerate, reason = family._is_degenerate(empty)
    assert degenerate and "no valid" in reason


def test_a_varying_layer_is_not_degenerate():
    varied = xr.DataArray(np.array([[1.0, 2.0], [3.0, np.nan]]), dims=("y", "x"))
    assert family._is_degenerate(varied)[0] is False


# --- derived class tables -----------------------------------------------------


def test_subbasin_classes_are_derived_from_the_raster_not_declared():
    """Identifiers are assigned per project; no constant could list them."""
    classes = family.subbasin_classes(_codes([101, 101, 102, 103]))
    assert [code for code, _, _ in classes] == [101, 102, 103]
    assert [label for _, _, label in classes] == [
        "Subbasin 101",
        "Subbasin 102",
        "Subbasin 103",
    ]


def test_subbasin_colours_reuse_rather_than_run_out():
    """More subbasins than palette colours still renders, every unit coloured."""
    codes = list(range(101, 101 + len(family.SUBBASIN_COLORS) + 2))
    classes = family.subbasin_classes(_codes(codes, nx=1))
    assert len(classes) == len(codes)
    assert {colour for _, colour, _ in classes} <= set(family.SUBBASIN_COLORS)


def test_touching_subbasins_never_share_a_colour():
    """A row of units: each touches its neighbours, so adjacent colours differ."""
    codes = list(range(101, 101 + 3 * len(family.SUBBASIN_COLORS)))
    classes = family.subbasin_classes(_codes(codes, nx=1))
    colour = {code: c for code, c, _ in classes}
    assert all(colour[a] != colour[b] for a, b in zip(codes, codes[1:]))


def test_an_unknown_variable_is_refused_rather_than_drawn_on_a_default_ramp(tmp_path):
    """A layer with no style would be plotted on a ramp that means nothing."""
    with pytest.raises(KeyError, match="not in the spatial map registry"):
        family.plot_spatial_maps(tmp_path, variables=["not_a_layer"])
