# -*- coding: utf-8 -*-
"""Render a basin map from files on disk. Edit the values below, then run:

    pixi run python dev/scripts/basin_map_example.py

Every layer is read from its own file — the DEM from a netCDF, the vectors from
GeoJSON — so any of them can be swapped for a file that never came from wflow.
"""

import sys
from pathlib import Path

import geopandas as gpd
import xarray as xr

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from blueearth_cst.shared import plot_map  # noqa: E402
from blueearth_cst.shared.plot_map import plot_basin_map  # noqa: E402

# --- input files -----------------------------------------------------------
# Set any of the optional vector paths to None to leave that layer off the map.

MODEL_DIR = REPO_ROOT / "test_case" / "basin_map_fixture" / "hydrology_model"
GEOMS_DIR = MODEL_DIR / "staticgeoms"

STATICMAPS_PATH = MODEL_DIR / "staticmaps.nc"
DEM_VARIABLE = "land_elevation"

RIVERS_PATH = GEOMS_DIR / "rivers.geojson"
BASINS_PATH = GEOMS_DIR / "basins.geojson"  # one polygon per subcatchment
SUBBASINS_PATH = GEOMS_DIR / "subbasins.geojson"
GAUGES_PATH = GEOMS_DIR / "gauges_locations.geojson"
OUTLETS_PATH = GEOMS_DIR / "outlets.geojson"
LAKES_PATH = None
RESERVOIRS_PATH = None
GLACIERS_PATH = None

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

# --- read the files --------------------------------------------------------


def read_vector(path):
    return gpd.read_file(path) if path is not None else None


# load() before the file closes -- the render touches the values repeatedly.
with xr.open_dataset(STATICMAPS_PATH) as dataset:
    dem = dataset[DEM_VARIABLE].load()

rivers = read_vector(RIVERS_PATH)
# The map wants ONE outer boundary at its heaviest weight; basins.geojson holds
# one polygon per subcatchment, so dissolve them. The divides come in
# separately, lighter and dashed.
basin = read_vector(BASINS_PATH).dissolve()

# --- render ----------------------------------------------------------------

fig, ax = plot_basin_map(
    dem,
    rivers,
    basin,
    subbasins=read_vector(SUBBASINS_PATH),
    gauges=read_vector(GAUGES_PATH),
    outlets=read_vector(OUTLETS_PATH),
    lakes=read_vector(LAKES_PATH),
    reservoirs=read_vector(RESERVOIRS_PATH),
    glaciers=read_vector(GLACIERS_PATH),
    extent=None,  # [lon_min, lon_max, lat_min, lat_max]
    gauge_label_column="wflow_id",
    river_order_column="strord",
    elevation_label="elevation [m a.s.l.]",
)

OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT_PATH, dpi=DPI)
print(f"wrote {OUT_PATH}")
