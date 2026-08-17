"""Report — and optionally write — the bounding box of the project's region.

**Why this exists rather than reading the model.** The built Wflow tree already
carries a rectangle at ``models/hydrology/wflow/staticgeoms/region.geojson``,
but that file is hydromt's, not ours: ``WflowStaticmapsComponent`` hardcodes
``region_component=None``, so ``Model.region`` resolves to
``GridComponent._region_data`` — ``box(*self.bounds)`` — and the value is
whatever a hydromt release decides "region" means for a grid model. Depending on
it also means depending on a MODEL BUILD, which WF0 deliberately does not do.

Measured on both fixtures, the Wflow grid's bounds ARE the basin's
``total_bounds`` to floating-point equality (max delta 3.3e-16), so deriving the
box here loses nothing. It gains a little: the grid in ``staticmaps.nc`` is
exact, but the ``staticgeoms/region.geojson`` written beside it carries
6-decimal coordinates — ~3.3e-7 degrees off the true corner. This script's own
write round-trips at zero, and ``hydromt.writers.write_region`` is a plain
``gdf.to_file`` with no precision argument, so that residual is the coordinate
precision of the file as written rather than a different rectangle.

Reads ``data/spatial/geoms/region.geojson`` — the delineated basin, and the one
project region artifact (ADR 0003) — never the model's ``staticgeoms/``. That
direction is the repo's stance, not this script's preference:
``blueearth_cst/spatial/delineate_region.py`` states the same rule for the
extraction ("never from a built model's ``staticmaps.nc`` or
``staticgeoms/region.geojson``").

**Prints by default; writes only with ``--out``.** The box is a derived value,
so seeing it is usually the whole need and nothing is written unless asked.
Where a built model happens to be present its rectangle is compared, so the
agreement is visible — but the model is optional and its absence is not an
error, which is what keeps this usable on a model-free WF0 project.

Usage (from the repo root, inside pixi)::

    python dev/scripts/region_bbox.py --config test_case/snake_config_rapid.yml
    python dev/scripts/region_bbox.py --config <cfg> --out
    python dev/scripts/region_bbox.py --config <cfg> --out some/where/bbox.geojson

Not part of a run: this inspects a project tree
(see AGENTS.md, "Three homes for executables").
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import geopandas as gpd  # noqa: E402
import yaml  # noqa: E402
from shapely.geometry import box  # noqa: E402

from blueearth_cst.spatial.delineate_region import read_region  # noqa: E402

#: The delineated basin, project-root-relative (ADR 0003's one region artifact).
REGION_PATH = "data/spatial/geoms/region.geojson"

#: Where ``--out`` puts the box when given no value: beside the basin it came
#: from, under a name that says which shape it is. The collision this whole
#: exercise is about — two directories holding a ``region.geojson`` that mean
#: different things — is not worth reproducing inside one directory.
DEFAULT_OUT = "data/spatial/geoms/region_bbox.geojson"

#: hydromt's own rectangle, for comparison only. Optional: a model-free project
#: has no model tree, and that is a supported state rather than a failure.
MODEL_REGION_PATH = "models/hydrology/wflow/staticgeoms/region.geojson"


def region_bbox(region_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """The axis-aligned bounding box of a region, in the region's own CRS."""
    return gpd.GeoDataFrame(
        {"name": ["region_bbox"]},
        geometry=[box(*region_gdf.total_bounds)],
        crs=region_gdf.crs,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="the --configfile the project was run with",
    )
    parser.add_argument(
        "--out",
        nargs="?",
        const=DEFAULT_OUT,
        default=None,
        metavar="PATH",
        help=(
            f"write the box as GeoJSON (bare flag: <project_dir>/{DEFAULT_OUT}); "
            "default is to report only"
        ),
    )
    args = parser.parse_args(argv)

    with open(args.config, encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    project_dir = Path(config["project"]["project_dir"])
    region_path = project_dir / REGION_PATH

    print(f"project        : {project_dir}")
    print(f"region         : {region_path}")
    if not region_path.is_file():
        print("\nregion.geojson not found - run any workflow's delineate_region rule")
        return 1

    region_gdf = read_region(region_path)
    bbox_gdf = region_bbox(region_gdf)
    # Plain floats, not the numpy scalars `total_bounds` returns: these lines
    # get copied into configs and issue reports, and `np.float64(9.65...)` is
    # not something you can paste anywhere.
    west, south, east, north = (float(v) for v in bbox_gdf.total_bounds)

    print(f"crs            : {region_gdf.crs}")
    print()
    print(f"west           : {west!r}")
    print(f"south          : {south!r}")
    print(f"east           : {east!r}")
    print(f"north          : {north!r}")
    print(f"width x height : {east - west} x {north - south}")
    print(f"basin area     : {float(region_gdf.geometry.area.sum())} (crs units^2)")
    print(f"bbox area      : {float(bbox_gdf.geometry.area.sum())} (crs units^2)")
    print(f"stage_data bbox: [{west}, {south}, {east}, {north}]")

    model_region_path = project_dir / MODEL_REGION_PATH
    print()
    if model_region_path.is_file():
        model_bounds = gpd.read_file(model_region_path).total_bounds
        deltas = [abs(a - b) for a, b in zip(bbox_gdf.total_bounds, model_bounds)]
        print(f"vs hydromt     : {model_region_path}")
        print(f"  max |delta|  : {max(deltas):.3e} deg")
        print(
            "  (a residual of ~3e-7 is the coordinate precision of hydromt's "
            "written GeoJSON, not a different rectangle)"
        )
    else:
        print(f"vs hydromt     : no model at {model_region_path} - skipped")

    if args.out is not None:
        out_path = Path(args.out)
        if not out_path.is_absolute():
            out_path = project_dir / out_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        bbox_gdf.to_file(out_path, driver="GeoJSON")
        written = gpd.read_file(out_path).total_bounds
        drift = max(abs(a - b) for a, b in zip(bbox_gdf.total_bounds, written))
        print()
        print(f"wrote          : {out_path}")
        print(f"  round-trip   : max |delta| {drift:.3e} deg")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
