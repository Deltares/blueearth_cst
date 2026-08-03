# -*- coding: utf-8 -*-
"""Render a basin map. Edit the values below, then run:

    pixi run python dev/scripts/basin_map_example.py
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from blueearth_cst.shared import plot_map  # noqa: E402
from blueearth_cst.shared.plot_map import (  # noqa: E402
    _basin_outline,
    load_basin_layers,
    plot_basin_map,
)

# --- inputs ----------------------------------------------------------------

MODEL_DIR = REPO_ROOT / "test_case" / "basin_map_fixture" / "hydrology_model"
OUT_PATH = REPO_ROOT / ".tmp" / "basin_map.png"
DPI = 200

# --- plotting parameters ---------------------------------------------------
# Any constant from plot_map.py's TUNABLE CONSTANTS block.

plot_map.FIGURE_WIDTH_MM = 180.0
plot_map.FONT_SIZE_BASE = 8.0
plot_map.COLOR_RIVER = "#2c6fad"
plot_map.RIVER_WIDTH_MIN = 0.2
plot_map.RIVER_WIDTH_MAX = 1.2
plot_map.MARKER_SIZE = 18

# --- render ----------------------------------------------------------------

dem, rivers, basins, geoms = load_basin_layers(MODEL_DIR)

fig, ax = plot_basin_map(
    dem,
    rivers,
    _basin_outline(basins),
    subbasins=basins,
    gauges=geoms.get("gauges_locations"),
    outlets=geoms.get("outlets"),
    lakes=geoms.get("lakes"),
    reservoirs=geoms.get("reservoirs"),
    glaciers=geoms.get("glaciers"),
    extent=None,  # [lon_min, lon_max, lat_min, lat_max]
    gauge_label_column="wflow_id",
    river_order_column="strord",
    elevation_label="elevation [m a.s.l.]",
)

OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT_PATH, dpi=DPI)
print(f"wrote {OUT_PATH}")
