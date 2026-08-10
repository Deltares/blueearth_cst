# -*- coding: utf-8 -*-
"""Unit tests for the spatial-map figure family and the categorical raster path.

Two things are covered, and they are different in kind.

The CATEGORICAL PATH in ``shared.cartographic_map`` is new template behaviour
and is tested for the properties that make a nominal map honest rather than
merely drawn: classes absent from the basin do not appear in the legend, classes
present but undeclared are drawn and warned about instead of vanishing, and the
codes are remapped onto contiguous indices so a legend's numeric gaps cannot
become colour gaps.

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
from blueearth_cst.shared.cartographic_map import (
    COLOR_UNCLASSIFIED,
    LABEL_UNCLASSIFIED,
    RasterStyle,
    _categorical_overlay_contrast,
    _category_handles,
    category_entries,
)


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


_THREE_CLASSES = (
    (10, "#ff0000", "ten"),
    (20, "#00ff00", "twenty"),
    (30, "#0000ff", "thirty"),
)


# --- the categorical path -----------------------------------------------------


def test_only_the_classes_the_basin_carries_reach_the_legend():
    """A global legend declares 23 land-cover classes; a basin has a handful.

    Listing the absent ones would describe ground the map does not show.
    """
    style = RasterStyle(label="x", palette=None, categories=_THREE_CLASSES)
    entries = category_entries(_codes([10, 10, 30, 30]), style)
    assert [label for _, _, label in entries] == ["ten", "thirty"]


def test_declared_order_is_kept_not_the_order_the_codes_appear_in():
    """The legend reads in the product's order, not in raster-scan order."""
    style = RasterStyle(label="x", palette=None, categories=_THREE_CLASSES)
    entries = category_entries(_codes([30, 20, 10, 10]), style)
    assert [codes[0] for codes, _, _ in entries] == [10, 20, 30]


def test_an_undeclared_code_is_drawn_and_warned_about_never_dropped():
    """Dropping it would render real ground transparent -- i.e. as nodata."""
    style = RasterStyle(label="x", palette=None, categories=_THREE_CLASSES)
    with pytest.warns(RuntimeWarning, match="does not declare"):
        entries = category_entries(_codes([10, 10, 99, 99]), style)
    codes, colour, label = entries[-1]
    assert codes == (99,)
    assert colour == COLOR_UNCLASSIFIED
    assert LABEL_UNCLASSIFIED in label and "99" in label


def test_several_undeclared_codes_collapse_into_one_legend_entry():
    """Two greys that are the same grey are worse than one entry naming both."""
    style = RasterStyle(label="x", palette=None, categories=_THREE_CLASSES)
    with pytest.warns(RuntimeWarning):
        entries = category_entries(_codes([10, 98, 99, 99]), style)
    assert len([e for e in entries if e[1] == COLOR_UNCLASSIFIED]) == 1
    assert entries[-1][0] == (98, 99)


def test_nodata_does_not_become_a_class():
    """NaN cells are absent, not a category -- they draw transparent."""
    style = RasterStyle(label="x", palette=None, categories=_THREE_CLASSES)
    entries = category_entries(_codes([10, np.nan, np.nan, 20]), style)
    assert [codes[0] for codes, _, _ in entries] == [10, 20]


def test_every_present_class_gets_exactly_one_legend_handle():
    style = RasterStyle(label="x", palette=None, categories=_THREE_CLASSES)
    entries = category_entries(_codes([10, 20, 30, 30]), style)
    handles = _category_handles(entries)
    assert len(handles) == 3
    assert [handle.get_label() for handle in handles] == ["ten", "twenty", "thirty"]


def test_divides_flip_to_a_pale_line_over_dark_classes():
    """The contrast heuristic has to work off class colours, not off a ramp.

    ``_overlay_contrast`` classifies the raster to answer this and would return
    nonsense for codes; the categorical branch measures the painted colours.
    """
    raster = _codes([10, 10, 10, 10])

    def contrast(colour):
        style = RasterStyle(label="x", palette=None, categories=((10, colour, "c"),))
        return _categorical_overlay_contrast(raster, category_entries(raster, style))

    assert contrast("#000000") != contrast("#ffffff")


# --- the registry -------------------------------------------------------------


def test_output_stems_are_unique():
    """Two entries sharing a stem would silently overwrite one another."""
    stems = [figure.stem for figure in family.SPATIAL_MAP_FIGURES]
    assert len(stems) == len(set(stems))


def test_no_stem_collides_with_the_basin_map_in_the_same_folder():
    """One rule writes both into ``data/spatial/plots``."""
    assert "basin_area" not in {figure.stem for figure in family.SPATIAL_MAP_FIGURES}


def test_every_soil_stem_records_its_depth_slice():
    """The filename is the only place the slice is written down now that the
    figures carry no title, and ``sl1`` is one of seven the source ships."""
    for figure in family.SPATIAL_MAP_FIGURES:
        if figure.variable.startswith("soil_") and "sl1" in figure.variable:
            assert figure.stem.endswith("_topsoil"), figure.stem


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
    assert "P/plots/land_cover.png" in paths
    assert not any(path.endswith(".pdf") for path in paths)
    assert "P/plots/soil_depth_to_bedrock.png" not in paths


def test_a_caller_can_still_ask_for_a_pdf():
    paths = family.figure_paths("P/plots", formats=("png", "pdf"))
    assert "P/plots/land_cover.pdf" in paths


def test_the_undeclared_figures_are_still_drawn():
    """`guaranteed=False` keeps a figure out of the rule's promise, not out of
    the run -- it is the difference between a lost figure and a failed workflow."""
    everything = family.figure_paths("P/plots", declared_only=False)
    assert "P/plots/soil_depth_to_bedrock.png" in everything


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
    entries = category_entries(
        _codes(codes, nx=1),
        RasterStyle(label="x", palette=None, categories=classes),
    )
    assert _category_handles(entries) == []


def test_variables_are_unique():
    stems = [figure.variable for figure in family.SPATIAL_MAP_FIGURES]
    assert len(stems) == len(set(stems))


def test_every_entry_is_either_continuous_or_nominal_never_both():
    """A ramp AND a class table would leave which one wins to argument order."""
    for figure in family.SPATIAL_MAP_FIGURES:
        assert (figure.style is None) != (figure.classes is None), figure.variable


def test_every_continuous_style_names_a_resolvable_palette():
    """A typo in a colormap name would surface as a KeyError mid-render."""
    from blueearth_cst.shared.cartographic_map import _style_colormap

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
    codes = [code for code, _, _ in family.LAND_COVER_CLASSES]
    assert codes == sorted(set(codes))


# --- the source footnote ------------------------------------------------------


def _layer(source):
    return xr.DataArray(np.zeros((2, 2)), dims=("y", "x"), attrs={"source": source})


def test_the_footnote_credits_the_product_not_the_catalog_key():
    """ "vito" is our plumbing; a figure credits Copernicus."""
    assert (
        family.source_caveat(_layer("soilgrids")) == "Source: SoilGrids (ISRIC, 2017)."
    )
    assert "Copernicus" in family.source_caveat(_layer("vito"))


def test_every_source_the_family_can_meet_has_a_credit():
    """A layer whose catalog key is unmapped falls back to the key itself."""
    for key in ("merit_hydro_ihu", "vito", "modis_lai", "soilgrids"):
        assert key not in family.source_caveat(_layer(key))


def test_an_unmapped_source_still_gets_a_line():
    """Ugly and visible beats silently unattributed -- the first gets fixed."""
    assert family.source_caveat(_layer("some_new_catalog_entry")) == (
        "Source: some_new_catalog_entry."
    )


def test_a_layer_with_no_source_gets_no_footnote():
    assert family.source_caveat(xr.DataArray(np.zeros((2, 2)), dims=("y", "x"))) is None


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


def test_subbasin_colours_cycle_rather_than_run_out():
    """A 30-subbasin project still renders; the divides keep them apart."""
    codes = list(range(101, 101 + len(family._QUALITATIVE_COLORS) + 2))
    classes = family.subbasin_classes(_codes(codes, nx=1))
    assert len(classes) == len(codes)
    assert classes[0][1] == classes[len(family._QUALITATIVE_COLORS)][1]


def test_an_unknown_variable_is_refused_rather_than_drawn_on_a_default_ramp(tmp_path):
    """A layer with no style would be plotted on a ramp that means nothing."""
    with pytest.raises(KeyError, match="not in the spatial map registry"):
        family.plot_spatial_maps(tmp_path, variables=["not_a_layer"])
