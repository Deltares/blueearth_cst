"""The reviewed CST chart preset, built from ``config/plotting/``.

Render entry points draw inside ``use_preset(CST_PRESET)``. The module-level
tokens repeat the preset's values for CST figure code that composes its own
artists; the YAML remains the single source for both.
"""

from blueearth_cst.plotting.preset import Preset, chart_rcparams
from blueearth_cst.shared import plot_style
from blueearth_cst.shared.plot_colors import (
    MAP_FEATURE_COLORS,
    NEUTRAL_COLORS,
    RIVER_ORDER_COLORS,
)
from blueearth_cst.shared.plot_setup import STYLE_SETTINGS

_CHARTS = STYLE_SETTINGS["charts"]
W_COL1, W_PANEL, W_MID, W_FULL = (
    _CHARTS["widths"][key] for key in ("col1", "panel", "mid", "full")
)


H_SINGLE, H_STACK2, H_SIDE2, H_ROW3, H_GRID4 = (
    _CHARTS["heights"][key] for key in ("single", "stack2", "side2", "row3", "grid4")
)


OUTER_PAD_X_MM, OUTER_PAD_Y_MM = (_CHARTS["padding"][key] for key in ("x", "y"))


FS_LABEL, FS_TICK, FS_NOTE = (
    _CHARTS["fonts"][key] for key in ("label", "tick", "note")
)


INK = NEUTRAL_COLORS["ink"]


MUTED = NEUTRAL_COLORS["muted"]


AXIS = NEUTRAL_COLORS["axis"]


GRID = NEUTRAL_COLORS["grid"]


BOUNDARY = NEUTRAL_COLORS["boundary"]


LW_LINE, LW_MEAN, LW_REF, LW_EDGE, LW_RANGE, LW_IQR, LW_STEM = (
    _CHARTS["lines"][key]
    for key in ("line", "mean", "ref", "edge", "range", "iqr", "stem")
)


LW_BOUNDARY = _CHARTS["lines"]["boundary"]


MS_LINE, MS_POINT, S_SCATTER = (
    _CHARTS["markers"][key] for key in ("line", "point", "scatter")
)


BAND_ALPHA = _CHARTS["band_alpha"]


LABEL_BOX = dict(_CHARTS["label_box"])


_COLORS = {
    key: NEUTRAL_COLORS[key] for key in ("ink", "muted", "axis", "grid", "boundary")
}


STYLE = {
    **_CHARTS["rcparams"],
    "axes.prop_cycle": plot_style.rcparams()["axes.prop_cycle"],
    **chart_rcparams(_COLORS, _CHARTS["fonts"], _CHARTS["lines"]),
    "savefig.dpi": plot_style.RASTER_DPI,
}


_MAP_STYLE = STYLE_SETTINGS["maps"]
if len(_MAP_STYLE["river_order_widths"]) != len(RIVER_ORDER_COLORS):
    raise ValueError(
        "style.yml maps.river_order_widths and colors.yml river_order need one "
        "entry per stream-order step"
    )
MAPS = {
    "river_order": tuple(zip(_MAP_STYLE["river_order_widths"], RIVER_ORDER_COLORS)),
    "river": MAP_FEATURE_COLORS["river"],
    "river_width": _MAP_STYLE["river_width"],
    "river_halo_width": _MAP_STYLE["river_halo_width"],
    "basin_outline": MAP_FEATURE_COLORS["basin_outline"],
    "basin_outline_width": _MAP_STYLE["basin_outline_width"],
    "divide": MAP_FEATURE_COLORS["subcatchment"],
    "divide_width": _MAP_STYLE["divide_width"],
    "divide_dash": tuple(_MAP_STYLE["divide_dash"]),
    "location": MAP_FEATURE_COLORS["gauge"],
    "location_marker": _MAP_STYLE["location_marker"],
    "location_size": _MAP_STYLE["location_size"],
    "location_label_size": _MAP_STYLE["location_label_size"],
    "location_label_wrap": int(_MAP_STYLE["location_label_wrap"]),
    "marker_edge": MAP_FEATURE_COLORS["marker_edge"],
    "halo": MAP_FEATURE_COLORS["halo"],
    "scale_bar_segments": int(_MAP_STYLE["scale_bar_segments"]),
    **{
        f"locator_{name}": MAP_FEATURE_COLORS[f"locator_{name}"]
        for name in ("ocean", "land", "coast", "border", "city")
    },
    "locator_mark": MAP_FEATURE_COLORS["locator_basin"],
}


CST_PRESET = Preset(
    widths=dict(_CHARTS["widths"]),
    heights=dict(_CHARTS["heights"]),
    padding=dict(_CHARTS["padding"]),
    fonts=dict(_CHARTS["fonts"]),
    colors=_COLORS,
    lines=dict(_CHARTS["lines"]),
    markers=dict(_CHARTS["markers"]),
    band_alpha=BAND_ALPHA,
    label_box=LABEL_BOX,
    rcparams=STYLE,
    maps=MAPS,
)
