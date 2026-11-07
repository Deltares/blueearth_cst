"""Behaviour guards for the map template and its shared adapter helpers.

Carried over from ``test_plot_map.py`` when the legacy cartographic module was
retired (2026-10-29): the behaviours still matter, the code under test moved.
"""

import numpy as np
import pytest
from shapely.geometry import Point, box, shape

from blueearth_cst.plotting.prepared_maps import (
    MapClasses,
    MapFrame,
    RasterMap,
    _class_shares,
    _draw_overlays,
    unkeyed_classes,
)
from blueearth_cst.shared.map_frames import overlay_features

_FRAME = MapFrame((0.0, 4.0, 0.0, 1.0), (0.0, 2.0, 4.0), (0.0, 1.0), 1.0)


def test_invalid_frame_rejects_before_figure_allocation():
    import matplotlib.pyplot as plt

    from blueearth_cst.plotting.prepared_maps import (
        draw_raster_map,
        draw_raster_panels_detailed,
    )

    before = plt.get_fignums()
    frame = MapFrame((0.0, 0.0, 0.0, 1.0), (), (), 1.0)
    with pytest.raises(ValueError):
        draw_raster_map(_raster([10, 20, 30, 10]), frame, _classes(), {})
    with pytest.raises(ValueError):
        draw_raster_panels_detailed(
            [_raster([10, 20, 30, 10])], ["one"], frame, _classes(), {}
        )
    assert plt.get_fignums() == before


@pytest.mark.parametrize("count", [1, 4, 6])
def test_map_panel_geometry_empty_slots_and_one_shared_missing_key(count):
    import matplotlib.pyplot as plt

    from blueearth_cst.plotting.prepared_maps import (
        MapScale,
        draw_raster_panels_detailed,
    )

    scale = MapScale(
        "mm",
        (-2.0, 0.0, 2.0),
        (-2.0, 0.0, 2.0),
        ("brown", "teal"),
        "brown",
        "teal",
        "gray",
        "available",
        "neither",
    )
    rasters = [_raster([np.nan, -1.0, 0.0, 1.0])] + [_raster([-1.0, 0.0, 1.0, 2.0])] * (
        count - 1
    )
    fig = draw_raster_panels_detailed(
        rasters, [str(i) for i in range(count)], _FRAME, scale, {}
    )
    try:
        fig.canvas.draw()
        assert len(fig.axes) == count  # trailing cells have no axes or duplicated data
        sizes = [
            (ax.bbox.width / fig.dpi * 25.4, ax.bbox.height / fig.dpi * 25.4)
            for ax in fig.axes
        ]
        np.testing.assert_allclose(
            sizes, np.tile([65.0, 16.25], (count, 1)), atol=0.001
        )
        assert sum(len(ax.child_axes) for ax in fig.axes) == 1
        assert any(
            t.get_text() == "No data" for t in fig.axes[-1].get_legend().get_texts()
        )
        bounds = fig.get_tightbbox(fig.canvas.get_renderer())
        assert bounds.x0 >= 0 and bounds.y0 >= 0
        assert bounds.x1 <= fig.get_figwidth() and bounds.y1 <= fig.get_figheight()
        if count > 1:
            assert fig.axes[0].bbox.x0 < fig.axes[1].bbox.x0
        if count > 3:
            assert fig.axes[3].bbox.y0 < fig.axes[0].bbox.y0
    finally:
        plt.close(fig)


def _classes(**kwargs):
    return MapClasses(
        "x",
        codes=(10, 20, 30),
        colors=("#ff0000", "#00ff00", "#0000ff"),
        labels=("ten", "twenty", "thirty"),
        missing_color="#a0a0a0",
        **kwargs,
    )


def test_matrix_pagination_preserves_row_column_semantics_and_map_sizes():
    import matplotlib.pyplot as plt

    from blueearth_cst.plotting.prepared_maps import MapScale, draw_raster_matrix

    scale = MapScale(
        "%", (-2, 0, 2), (-2, 0, 2), ("blue", "red"), "blue", "red", "gray", "available"
    )
    fields = [[_raster([1, 0, -1, 1]) for _ in range(5)] for _ in range(5)]
    titles = [[f"{r}:{c}" for c in range(5)] for r in range(5)]
    figures = draw_raster_matrix(
        fields,
        [f"Model {i}" for i in range(5)],
        [f"SSP {i}" for i in range(5)],
        titles,
        _FRAME,
        scale,
        {},
    )
    try:
        assert [len(fig.axes) - 1 for fig in figures] == [12, 3, 8, 2]
        drawn, widths = [], []
        for fig in figures:
            fig.canvas.draw()
            drawn.extend(ax.get_title(loc="left") for ax in fig.axes[:-1])
            widths.extend(ax.bbox.width / fig.dpi for ax in fig.axes[:-1])
            assert any("Model" in text.get_text() for text in fig.texts)
            assert any("SSP" in text.get_text() for text in fig.texts)
        assert sorted(drawn) == sorted(title for row in titles for title in row)
        np.testing.assert_allclose(widths, widths[0])
    finally:
        for fig in figures:
            plt.close(fig)


def _raster(values):
    return RasterMap(
        np.array([values], dtype=float), np.arange(5.0), np.array([0.0, 1.0])
    )


def test_compact_map_keeps_callers_canvas_and_omits_repeated_decorations():
    import matplotlib.pyplot as plt

    from blueearth_cst.plotting.prepared_maps import draw_raster_map

    fig, axes = plt.subplots(1, 2, figsize=(6, 2))
    try:
        original_size = fig.get_size_inches().copy()
        overlays = {
            "locations": [
                {
                    "geometry": {"type": "Point", "coordinates": [2, 0.5]},
                    "properties": {"station_name": "Control"},
                }
            ]
        }
        for ax in axes:
            returned = draw_raster_map(
                _raster([10, 20, 30, 10]),
                _FRAME,
                _classes(),
                overlays,
                ax=ax,
                compact=True,
            )
            assert returned is fig
            assert len(ax.collections) == 1
            assert not ax.get_xlabel() and not ax.get_ylabel()
            assert len(ax.get_xticks()) == len(ax.get_yticks()) == 0
            assert not ax.texts and not ax.child_axes
            assert 3 <= ax.lines[0].get_markersize() < 6
            assert ax.lines[0].get_markeredgewidth() < 0.6
        np.testing.assert_array_equal(fig.get_size_inches(), original_size)
        assert len(fig.axes) == 2
    finally:
        plt.close(fig)


@pytest.mark.parametrize("columns", [3, 4])
def test_shared_map_panels_follow_preset_and_keep_every_field(columns):
    from dataclasses import replace

    import matplotlib.pyplot as plt

    from blueearth_cst.plotting.prepared_maps import MapScale, draw_raster_panels
    from blueearth_cst.plotting.preset import NEUTRAL, use_preset

    preset = replace(NEUTRAL, map_panels={**NEUTRAL.map_panels, "columns": columns})
    scale = MapScale(
        "%", (-2, 0, 2), (-2, 0, 2), ("blue", "red"), "blue", "red", "gray", "available"
    )
    with use_preset(preset):
        (fig,) = draw_raster_panels(
            [_raster([1, -1, 0, 1])] * 11,
            [str(i) for i in range(11)],
            _FRAME,
            scale,
            {},
        )
        try:
            fig.canvas.draw()
            assert len(fig.axes) == 12  # eleven supplied maps, one shared key
            assert fig.get_figwidth() * 25.4 == pytest.approx(190)
            axes, key = fig.axes[:-1], fig.axes[-1]
            np.testing.assert_allclose(
                [ax.bbox.width for ax in axes], axes[0].bbox.width
            )
            assert [ax.get_title(loc="left") for ax in axes] == [
                str(i) for i in range(11)
            ]
            assert axes[columns].bbox.y0 < axes[0].bbox.y0
            assert key.spines["outline"].get_window_extent().y0 == pytest.approx(
                axes[-1].bbox.y0
            )
            assert all(not ax.get_xticks().size for ax in axes)
        finally:
            plt.close(fig)


@pytest.mark.parametrize(
    "count,maximum,expected",
    [(12, 12, [12]), (13, 12, [12, 1]), (25, 12, [12, 12, 1]), (12, 6, [6, 6])],
)
def test_map_panels_paginate_without_loss_or_resizing(count, maximum, expected):
    from dataclasses import replace

    import matplotlib.pyplot as plt

    from blueearth_cst.plotting.prepared_maps import MapScale, draw_raster_panels
    from blueearth_cst.plotting.preset import NEUTRAL, use_preset

    preset = replace(
        NEUTRAL, map_panels={**NEUTRAL.map_panels, "max_panels_per_figure": maximum}
    )
    scale = MapScale(
        "%", (-2, 0, 2), (-2, 0, 2), ("blue", "red"), "blue", "red", "gray", "available"
    )
    with use_preset(preset):
        figures = draw_raster_panels(
            [_raster([1, -1, 0, 1])] * count,
            [str(i) for i in range(count)],
            _FRAME,
            scale,
            {},
        )
        try:
            assert [len(fig.axes) - 1 for fig in figures] == expected
            titles, widths = [], []
            for fig in figures:
                fig.canvas.draw()
                for ax in fig.axes[:-1]:
                    titles.append(ax.get_title(loc="left"))
                    widths.append(ax.bbox.width / fig.dpi)
                    np.testing.assert_array_equal(
                        ax.collections[0].norm.boundaries, scale.boundaries
                    )
            assert titles == [str(i) for i in range(count)]
            np.testing.assert_allclose(widths, widths[0])
        finally:
            for fig in figures:
                plt.close(fig)


@pytest.mark.parametrize("extent", [(0, 4, 0, 1), (0, 1, 0, 4), (0, 2, 0, 2)])
def test_keys_extend_canvas_without_resizing_map(extent):
    import matplotlib.pyplot as plt

    from blueearth_cst.plotting.prepared_maps import MapScale, draw_raster_map

    frame = MapFrame(extent, (), (), 1.0)
    raster = _raster([10, 20, 30, 10])
    scale = MapScale(
        "mm/year",
        (0, 20, 40),
        (0, 20, 40),
        ("blue", "red"),
        "white",
        "black",
        "gray",
        "available",
    )
    figures = [
        draw_raster_map(raster, frame, key, {}, show_key=show)
        for key, show in [(_classes(), False), (_classes(), True), (scale, True)]
    ]
    try:
        sizes = []
        for fig in figures:
            fig.canvas.draw()
            ax = fig.axes[0]
            sizes.append(np.array([ax.bbox.width, ax.bbox.height]) / fig.dpi)
            bounds = fig.get_tightbbox(fig.canvas.get_renderer())
            assert bounds.x0 >= 0
            assert bounds.y0 >= 0
            assert bounds.x1 <= fig.get_figwidth()
            assert bounds.y1 <= fig.get_figheight()
        np.testing.assert_allclose(sizes, np.tile(sizes[0], (3, 1)), atol=1e-10)
        assert figures[1].get_figwidth() > figures[0].get_figwidth()
        assert figures[2].get_figwidth() > figures[0].get_figwidth()
    finally:
        for fig in figures:
            plt.close(fig)


def test_only_the_classes_the_basin_carries_are_keyed():
    """Listing absent classes would describe ground the map does not show."""
    _, shares = _class_shares(_raster([10, 10, 30, 30]), _FRAME, _classes())
    assert sorted(shares) == [0, 2]


def test_an_undeclared_code_is_drawn_as_unclassified_never_dropped():
    """Dropping it would render real ground as nodata."""
    _, shares = _class_shares(_raster([10, 10, 99, 98]), _FRAME, _classes())
    assert shares == {0: 0.5, 3: 0.5}  # index len(codes): one unclassified entry


def test_nodata_does_not_become_a_class():
    _, shares = _class_shares(_raster([10, np.nan, np.nan, 20]), _FRAME, _classes())
    assert sorted(shares) == [0, 1]


def test_a_class_below_the_key_share_is_drawn_but_named_as_unkeyed():
    raster = _raster([10, 10, 10, 30])
    assert unkeyed_classes(raster, _FRAME, _classes(key_min_share=0.3)) == ["thirty"]
    assert unkeyed_classes(raster, _FRAME, _classes()) == []


def _feature(geometry, **properties):
    return {
        "type": "Feature",
        "geometry": geometry.__geo_interface__,
        "properties": properties,
    }


def test_divides_are_merged_so_a_shared_edge_is_drawn_once():
    """Two dashed strokes at one place interleave into an apparent solid line."""
    touching = [_feature(box(0, 0, 1, 1)), _feature(box(0, 1, 1, 2))]
    divides = overlay_features({"subbasins": touching})["divides"]
    assert len(divides) == 1
    # Outer ring of the union (6) plus the internal divide counted ONCE (1).
    # Two un-merged rings would total 8.
    assert shape(divides[0]["geometry"]).length == pytest.approx(7.0)


@pytest.mark.parametrize("name", ["Alexandria", None, float("nan"), "", "   "])
def test_locations_prefer_names_with_id_fallback(name):
    import matplotlib.pyplot as plt

    expected = "Alexandria" if name == "Alexandria" else "101"
    location = _feature(Point(0.05, -0.05), station_name=name, wflow_id=101)
    fig, ax = plt.subplots()
    try:
        labels = _draw_overlays(ax, {"locations": [location]})
        assert [label for _, _, label in labels] == [expected]
    finally:
        plt.close(fig)
