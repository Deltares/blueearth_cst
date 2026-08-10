# -*- coding: utf-8 -*-
"""Render the spatial-map family from a project on disk, without a WF1 run.

The sibling of ``preview_basin_map.py`` for the rest of the figure set that
``data/spatial/spatial_maps.nc`` supports. It calls the PRODUCTION entry point
(``shared.plot_spatial_maps.plot_spatial_maps``), so what it renders is what the
workflow renders — this script only chooses the project and the output folder.

    # the whole family, into the project's own data/spatial/plots
    pixi run python dev/scripts/preview_spatial_maps.py

    # one layer, somewhere else, at screen resolution
    pixi run python dev/scripts/preview_spatial_maps.py \
        --variable land_cover --out-dir .tmp/spatial_preview --dpi 150

Unlike ``preview_basin_map.py`` this does NOT refuse to write into the project
directory, and the difference is deliberate: that script drives hand-overridden
TUNABLES, so its renders must never be mistaken for a run product. This one
overrides nothing, so its output IS the run product — which is why the default
output folder is the same ``data/spatial/plots`` the rule writes to.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from blueearth_cst.shared.plot_spatial_maps import (  # noqa: E402
    SPATIAL_DIRNAME,
    SPATIAL_MAP_FIGURES,
    SPATIAL_MAPS_FILENAME,
    plot_spatial_maps,
)

#: Project directories tried, in order, when ``--project-dir`` is not given.
#: Only ``test_local`` qualifies: ``basin_map_fixture`` carries a wflow model and
#: no ``data/spatial/`` at all, so it cannot render this family. That is the one
#: place this family departs from the repo's usual "render the layer-rich
#: fixture" rule, and it departs because the layer-rich fixture has no thematic
#: rasters — not because the rule was overlooked. ``test_local`` is layer-rich
#: where it counts here: five subbasins and five locations.
_CANDIDATE_PROJECT_DIRS = ("test_case/test_local",)


def _primary_checkout() -> Path:
    """The main working tree, which is where ``test_case/`` normally lives."""
    try:
        common = subprocess.run(
            ["git", "rev-parse", "--git-common-dir"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return REPO_ROOT
    return (REPO_ROOT / common).resolve().parent


def resolve_spatial_dir(requested: str | None) -> Path:
    """``--project-dir``, then the env var, then the first candidate present.

    ``$BASIN_MAP_PROJECT_DIR`` is a HINT here and falls through when the project
    it names has no ``data/spatial/``. It is shared with
    ``preview_basin_map.py``, whose default project is ``basin_map_fixture`` — a
    wflow model with no spatial foundation at all — so honouring it strictly
    would leave this script dead in exactly the shell where the other one is set
    up. ``--project-dir`` stays authoritative: an explicit path that does not
    hold the file is an error, not a reason to render something else.
    """
    if requested:
        candidates = [Path(requested).expanduser()]
    else:
        hint = os.environ.get("BASIN_MAP_PROJECT_DIR")
        candidates = [Path(hint).expanduser()] if hint else []
        candidates += [
            root / candidate
            for root in dict.fromkeys((REPO_ROOT, _primary_checkout()))
            for candidate in _CANDIDATE_PROJECT_DIRS
        ]
    for candidate in candidates:
        spatial_dir = candidate.resolve() / SPATIAL_DIRNAME
        if (spatial_dir / SPATIAL_MAPS_FILENAME).is_file():
            return spatial_dir
    raise SystemExit(
        f"no {SPATIAL_DIRNAME}/{SPATIAL_MAPS_FILENAME} found. Pass --project-dir "
        "<dir>, or set $BASIN_MAP_PROJECT_DIR. Tried: "
        + ", ".join(str(c / SPATIAL_DIRNAME) for c in candidates)
    )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="See the module docstring for worked examples.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="print the registry — variable, output stem, whether declared — then exit",
    )
    parser.add_argument(
        "--project-dir",
        help="the folder holding data/spatial/ (default: $BASIN_MAP_PROJECT_DIR, "
        "then the first known test project present)",
    )
    parser.add_argument(
        "--variable",
        action="append",
        default=[],
        help="render only this variable; repeatable (default: the whole family)",
    )
    parser.add_argument(
        "--out-dir",
        help="where the figures go (default: the project's own data/spatial/plots)",
    )
    parser.add_argument(
        "--dpi", type=int, help="PNG resolution (default: the export dpi)"
    )
    parser.add_argument(
        "--png-only",
        action="store_true",
        help="skip the PDF, which is most of the render time on a large family",
    )
    args = parser.parse_args(argv)

    if args.list:
        for figure in SPATIAL_MAP_FIGURES:
            declared = (
                "" if figure.guaranteed else "  (drawn, not a declared rule output)"
            )
            print(f"{figure.variable:24s} -> {figure.stem}{declared}")
        return 0

    spatial_dir = resolve_spatial_dir(args.project_dir)
    print(f"spatial : {spatial_dir}")
    written = plot_spatial_maps(
        spatial_dir,
        plot_dir=args.out_dir,
        variables=args.variable or None,
        dpi=args.dpi,
        formats=("png",) if args.png_only else ("png", "pdf"),
    )
    print(f"wrote   : {len(written)} file(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
