# -*- coding: utf-8 -*-
"""A worked, EDITABLE example of calling ``plot_basin_map`` layer by layer.

    pixi run python dev/scripts/basin_map_example.py --open

Everything you are meant to change sits in the three numbered blocks below —
which layers go on the map, how their columns are read, and what the figure
looks like. Edit, re-run, look. Each render takes a few seconds and lands in a
gitignored scratch directory, never in a project's ``plots/``.

This is the companion to ``preview_basin_map.py``, not a replacement:

* ``preview_basin_map.py`` renders what the WORKFLOW renders, from a model
  directory, and drives the style constants from the COMMAND LINE (``--set``,
  ``--sweep``). Reach for it to tune a value or sweep a range.
* This script calls the layer-in ``plot_basin_map`` DIRECTLY, so it is where you
  see what the function actually takes: a DEM, a river network, a basin
  outline, and whatever optional layers you hand it. Reach for it to try
  dropping a layer, re-framing the map, or plotting something that did not come
  from wflow at all.

Things worth trying, each a one-line edit:

    DRAW["subbasins"] = False       # outline only, no internal divides
    DRAW["gauges"] = False          # the markers and their labels both go
    GAUGE_LABEL_COLUMN = None       # keep the markers, drop the 101/102 labels
    RIVER_ORDER_COLUMN = None       # every reach at one width
    EXTENT = [9.70, 9.86, 0.38, 0.50]   # crop to the eastern half
    STYLE["COLOR_RIVER"] = "#1b7f5f"
    STYLE["FIGURE_WIDTH_MM"] = 90.0      # a single-column figure
    STYLE["_LEGEND_TITLE"] = None        # drop the legend's title row

Plotting a basin that is NOT a wflow model is the same call — build the four
arguments yourself and skip ``load_basin_layers``:

    fig, ax = plot_basin_map(my_dem, my_rivers, my_catchment)

``my_dem`` is a 2-D geographic ``xarray.DataArray`` (EPSG:4326, NaN outside the
basin); the rest are GeoDataFrames in the same CRS.
"""

from __future__ import annotations

import argparse
import difflib
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
for entry in (str(REPO_ROOT), str(Path(__file__).resolve().parent)):
    if entry not in sys.path:
        sys.path.insert(0, entry)

from blueearth_cst.shared import plot_map  # noqa: E402
from blueearth_cst.shared.plot_map import (  # noqa: E402
    _basin_outline,
    load_basin_layers,
    plot_basin_map,
)
from preview_basin_map import gauges_fn_for, resolve_project_dir  # noqa: E402

# ===========================================================================
# 1. WHICH LAYERS GO ON THE MAP
# ===========================================================================
# Every one of these is OPTIONAL — the DEM, the rivers and the basin outline
# are the only required arguments. Switch one off and it disappears from the
# drawing AND from the legend; nothing else moves.
#
# A layer the model does not have is simply absent, so switching it on does
# nothing rather than failing. `basin_map_fixture` carries subcatchments,
# gauges and an outlet, and no waterbodies at all.
# ---------------------------------------------------------------------------

DRAW = {
    "subbasins": True,  # internal divides, dashed — the basin outline stays either way
    "gauges": True,  # output locations, blue, labelled
    "outlets": True,  # the model's own outlet(s), black
    "lakes": True,
    "reservoirs": True,
    "glaciers": True,
}

# ===========================================================================
# 2. HOW THE LAYERS ARE READ
# ===========================================================================
# These are the parameters that make the function work on data that did not
# come from wflow: nothing here assumes a hydromt column name, it is only the
# default.
# ---------------------------------------------------------------------------

#: Column annotated beside each gauge. ``None`` leaves the markers unlabelled.
#: Try ``"station_name"`` on a model that carries one.
GAUGE_LABEL_COLUMN = "wflow_id"

#: Numeric column scaling river line weights (wflow: Strahler order). ``None``,
#: or a column the layer does not have, draws every reach at one width.
RIVER_ORDER_COLUMN = "strord"

#: Colourbar label — change it with the DEM's units.
ELEVATION_LABEL = "elevation [m a.s.l.]"

#: ``[lon_min, lon_max, lat_min, lat_max]``, or ``None`` for the DEM's own
#: bounding box plus a small margin. Set it to crop, or to frame two basins
#: identically so they can be compared side by side.
EXTENT = None

# ===========================================================================
# 3. STYLE
# ===========================================================================
# Any constant from plot_map.py's TUNABLE CONSTANTS block, by name. Run
#     pixi run python dev/scripts/preview_basin_map.py --list
# to see all of them with their current values and their own comments.
# ---------------------------------------------------------------------------

STYLE = {
    # "FIGURE_WIDTH_MM": 180.0,
    # "COLOR_RIVER": "#2c6fad",
    # "RIVER_WIDTH_MAX": 1.2,
    # "MARKER_SIZE": 18,
    # "FONT_SIZE_BASE": 8.0,
}

#: Where the render goes, and at what resolution. Never a project's plots/.
OUT_DIR = REPO_ROOT / ".tmp" / "basin_map_example"
OUT_NAME = "basin_map_example.png"
DPI = 200


# ---------------------------------------------------------------------------
# Below here is the plumbing. You should not need to touch it.
# ---------------------------------------------------------------------------


def build_layers(project_dir: Path) -> dict:
    """Turn a wflow model directory into the arguments ``plot_basin_map`` takes.

    This is the whole of what ``plot_basin_map_from_model`` does before it
    draws, spelled out: the wflow-specific knowledge lives HERE, and what it
    hands on is plain DEM + GeoDataFrames.
    """
    dem, rivers, basins, geoms = load_basin_layers(project_dir / "hydrology_model")

    # `basins` holds one polygon PER SUBCATCHMENT once gauges are burned into
    # the subcatchment map, and a single polygon otherwise. The figure wants it
    # in two roles: a dissolved outer boundary, and the divides inside it.
    layers = {
        "dem": dem,
        "rivers": rivers,
        "basin": _basin_outline(basins),
        "subbasins": basins if len(basins) > 1 else None,
        "outlets": geoms.get("outlets"),
        "lakes": geoms.get("lakes"),
        "reservoirs": geoms.get("reservoirs"),
        "glaciers": geoms.get("glaciers"),
    }

    # hydromt_wflow renames `output_locations` to `output-locations`, so the
    # gauge layer is resolved from what the model HOLDS, never from a filename.
    from blueearth_cst.shared.gauges import gauges_layer_name

    gauges_name = gauges_layer_name(geoms, gauges_fn_for(project_dir))
    layers["gauges"] = geoms.get(gauges_name) if gauges_name is not None else None
    return layers


def render(project_dir: Path, out_dir: Path, open_it: bool = False) -> Path:
    layers = build_layers(project_dir)
    optional = {
        name: (layers[name] if DRAW.get(name, True) else None)
        for name in ("subbasins", "gauges", "outlets", "lakes", "reservoirs", "glaciers")
    }

    drawn = sorted(name for name, layer in optional.items() if layer is not None)
    skipped = sorted(name for name, layer in optional.items() if layer is None)
    print(f"drawing : dem, rivers, basin, {', '.join(drawn) if drawn else '(nothing optional)'}")
    print(f"omitted : {', '.join(skipped) if skipped else '(nothing)'}")
    if STYLE:
        print(f"style   : {', '.join(f'{k}={v!r}' for k, v in STYLE.items())}")

    # Check every name BEFORE snapshotting any: reading an unknown one raises a
    # bare AttributeError from inside the snapshot, which says nothing useful.
    for name in STYLE:
        if hasattr(plot_map, name):
            continue
        close = difflib.get_close_matches(name, dir(plot_map), n=3, cutoff=0.5)
        hint = f" Did you mean {', '.join(close)}?" if close else ""
        raise SystemExit(
            f"{name!r} is not a constant in plot_map.py.{hint} Run "
            "`preview_basin_map.py --list` to see every name with its comment."
        )

    previous = {name: getattr(plot_map, name) for name in STYLE}
    for name, value in STYLE.items():
        setattr(plot_map, name, value)
    try:
        fig, _ = plot_basin_map(
            layers["dem"],
            layers["rivers"],
            layers["basin"],
            extent=EXTENT,
            gauge_label_column=GAUGE_LABEL_COLUMN,
            river_order_column=RIVER_ORDER_COLUMN,
            elevation_label=ELEVATION_LABEL,
            **optional,
        )
    finally:
        for name, value in previous.items():
            setattr(plot_map, name, value)

    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / OUT_NAME
    # The figure comes back already laid out, so ONE savefig is enough — and no
    # bbox_inches="tight", which would discard the declared millimetre width.
    fig.savefig(path, dpi=DPI)
    print(f"wrote   : {path}")

    if open_it:
        if sys.platform == "win32":
            os.startfile(path)  # noqa: S606 — the point of --open
        else:
            subprocess.run(["open" if sys.platform == "darwin" else "xdg-open", str(path)])
    return path


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        epilog="Edit the three numbered blocks at the top of this file, then re-run.",
    )
    parser.add_argument(
        "--project-dir",
        help="the folder holding hydrology_model/ (default: $BASIN_MAP_PROJECT_DIR, "
        "then test_case/basin_map_fixture)",
    )
    parser.add_argument("--out-dir", default=str(OUT_DIR), help="default: %(default)s")
    parser.add_argument("--open", action="store_true", help="open the render when done")
    args = parser.parse_args(argv)

    project_dir = resolve_project_dir(args.project_dir)
    print(f"project : {project_dir}")
    render(project_dir, Path(args.out_dir).expanduser().resolve(), args.open)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
