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


def _scale(extend="both"):
    from blueearth_cst.plotting.prepared_maps import MapScale

    return MapScale(
        "mm",
        (0, 10, 20),
        (0, 10, 20),
        ("blue", "red"),
        "white",
        "black",
        "gray",
        "available",
        extend,
    )


@pytest.mark.parametrize(
    "values,permitted,expected",
    [
        ([5, 10, 15, 20], "both", "neither"),  # the closed top boundary is in scale
        ([-1, 10, 15, 20], "both", "min"),
        ([5, 10, 15, 21], "both", "max"),
        ([-1, 10, 15, 21], "both", "both"),
        ([-1, 10, 15, 21], "max", "max"),
        ([-1, 10, 15, 21], "neither", "neither"),
        ([np.nan, 10, 15, 20], "both", "neither"),
    ],
)
def test_key_extensions_follow_drawn_off_scale_cells(values, permitted, expected):
    from blueearth_cst.plotting.prepared_maps import key_extend

    assert key_extend([_raster(values)], _FRAME, _scale(permitted)) == expected


def test_key_extensions_ignore_hidden_and_out_of_frame_cells():
    from blueearth_cst.plotting.prepared_maps import key_extend

    inside = np.array([[True, True, True, False]])
    hidden = RasterMap(
        np.array([[5.0, 10.0, 15.0, 99.0]]),
        np.arange(5.0),
        np.array([0.0, 1.0]),
        inside,
    )
    beyond = RasterMap(
        np.array([[5.0, 10.0, 15.0, 99.0]]), np.arange(5.0) + 4.5, np.array([0.0, 1.0])
    )
    masked = RasterMap(
        np.ma.masked_greater(np.array([[5.0, 10.0, 15.0, 99.0]]), 50),
        np.arange(5.0),
        np.array([0.0, 1.0]),
    )
    for raster in (hidden, masked):
        assert key_extend([raster], _FRAME, _scale()) == "neither"
    # Only the first column of ``beyond`` intersects the 0-4 frame.
    assert key_extend([beyond], _FRAME, _scale()) == "neither"


def test_paginated_panels_share_one_pointed_key():
    """A late page's off-scale cell extends every page's key, not only its own."""
    from dataclasses import replace

    import matplotlib.pyplot as plt

    from blueearth_cst.plotting.prepared_maps import draw_raster_panels
    from blueearth_cst.plotting.preset import NEUTRAL, use_preset

    preset = replace(
        NEUTRAL, map_panels={**NEUTRAL.map_panels, "max_panels_per_figure": 2}
    )
    rasters = [_raster([5, 10, 15, 20])] * 3 + [_raster([5, 10, 15, 25])]
    with use_preset(preset):
        figures = draw_raster_panels(
            rasters, [str(i) for i in range(4)], _FRAME, _scale(), {}
        )
    try:
        keys = [fig.axes[-1]._colorbar for fig in figures]
        assert [key.extend for key in keys] == ["max", "max"]
        assert not any(key.extendrect for key in keys)
        # Off-scale colours are unchanged: the norm keeps the permitted sides.
        cmap = figures[1].axes[1].collections[0].cmap
        assert cmap.get_over() == pytest.approx((0, 0, 0, 1))
    finally:
        for fig in figures:
            plt.close(fig)


@pytest.mark.parametrize(
    "value,text",
    [(0.0, "0"), (-0.0, "0"), (0.4, "0.4"), (-600.0, "−600"), (2.5e-1, "0.25")],
)
def test_key_ticks_carry_no_padded_zeros(value, text):
    from blueearth_cst.plotting.prepared_maps import _tick_text

    assert _tick_text(value) == text


@pytest.mark.parametrize(
    "ticks,expected",
    [
        ([0.35, 0.4, 0.45], ["0.35", "0.40", "0.45"]),
        ([9.7, 9.8], ["9.7", "9.8"]),
        ([-0.5, 0.0, 0.25], ["−0.50", "0.00", "0.25"]),
        ([10.0, 11.0], ["10", "11"]),
        ([0.1 + 0.2, 0.4], ["0.3", "0.4"]),
    ],
)
def test_coordinate_ticks_share_one_precision_per_axis(ticks, expected):
    from blueearth_cst.plotting.prepared_maps import _coordinate_formatter

    formatter = _coordinate_formatter(ticks)
    assert [formatter(value) for value in ticks] == expected


@pytest.mark.parametrize("height,thinned", [(0.9, False), (0.12, True)])
def test_crowded_key_keeps_every_second_tick_with_ends_and_zero(height, thinned):
    import matplotlib.pyplot as plt
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import BoundaryNorm, ListedColormap

    from blueearth_cst.plotting.prepared_maps import _thin_crowded_key_ticks

    boundaries = [-600, -450, -300, -150, 0, 150, 300, 450, 600]
    fig = plt.figure(figsize=(2, 3))
    cax = fig.add_axes((0.4, 0.05, 0.08, height))
    mappable = ScalarMappable(BoundaryNorm(boundaries, 8), ListedColormap(["0.5"] * 8))
    colorbar = fig.colorbar(mappable, cax=cax, ticks=boundaries)
    try:
        _thin_crowded_key_ticks(fig, colorbar)
        ticks = list(colorbar.get_ticks())
        assert ticks == (boundaries[::2] if thinned else boundaries)
        assert {-600, 0, 600} <= set(ticks)
    finally:
        plt.close(fig)


@pytest.mark.parametrize(
    "name,expected",
    [
        ("north_tributary", "north tributary"),
        ("auto_01", "auto 01"),
        ("A__B", "A B"),
        ("Alexandria", "Alexandria"),
    ],
)
def test_map_labels_read_underscores_as_spaces(name, expected):
    from blueearth_cst.plotting.location_labels import map_display_label, wrap_label

    assert map_display_label(name) == expected
    assert not wrap_label(map_display_label(name), 6).endswith("_")


def test_point_labels_avoid_rivers_when_a_clear_offset_exists():
    """IDs stay raw in data; the drawn label is the display form and leaves the river."""
    import matplotlib.pyplot as plt
    from matplotlib.text import Text

    from blueearth_cst.plotting.prepared_maps import draw_raster_map

    frame = MapFrame((0.0, 4.0, 0.0, 4.0), (), (), 1.0)
    raster = RasterMap(
        np.full((1, 1), 15.0), np.array([0.0, 4.0]), np.array([0.0, 4.0])
    )
    # A river leaving the gauge up and to the right covers the default offset.
    river = {
        "geometry": {"type": "LineString", "coordinates": [[2, 2], [4, 4]]},
        "properties": {},
    }
    gauge = {
        "geometry": {"type": "Point", "coordinates": [2, 2]},
        "properties": {"station_name": "north_tributary"},
    }
    fig = draw_raster_map(
        raster, frame, _scale(), {"rivers": [river], "locations": [gauge]}
    )
    try:
        ax = fig.axes[0]
        (label,) = [t for t in ax.texts if "tributary" in t.get_text()]
        assert label.get_text() == "north\ntributary"
        renderer = fig.canvas.get_renderer()
        box = Text.get_window_extent(label, renderer)
        line = next(line for line in ax.lines if line.get_label() == "_cst_river")
        path = line.get_transform().transform_path(line.get_path())
        assert not path.intersects_bbox(box, filled=False)
    finally:
        plt.close(fig)
