# Current guidance: dev/reference/wf1-model-construction.md
# -*- coding: utf-8 -*-
"""Render toolbox figures from a project tree on disk, without running a workflow.

The figure families this toolbox ships are produced deep inside Snakemake rules,
so the normal way to look at one is to run the workflow that writes it. That is
minutes-to-hours per look, which makes iterating on a figure's *design*
impractical. This script rebuilds each family's inputs from artefacts a finished
run already left behind, and calls the same plotting function the rule calls.

    # what can be rendered, and is its input present?
    pixi run python dev/scripts/preview_plots.py --list

    # render everything available, into a scratch dir
    pixi run python dev/scripts/preview_plots.py --all

    # render one family and open the output folder
    pixi run python dev/scripts/preview_plots.py hydro --open

Renders land in ``--out-dir`` (a gitignored scratch tree by default) and NEVER
in a project's own ``plots/``: a preview must not be able to take the place of a
run product that the baseline fingerprints. Map figures (WF0/WF1 climate maps
and the rule 1.11 spatial set) are re-rendered by ``scripts/render_figures.py``
and are not duplicated here.

**Coverage is deliberately partial.** A family appears below only when its
inputs can be rebuilt HONESTLY from a finished run. Three are excluded for
stated reasons rather than forgotten:

* ``signatures`` needs observed discharge, and observations live outside the
  repository by design (``config/templates/observations/`` holds header-only
  schemas). Passing simulated discharge as if it were observed would render a
  figure that looks right and means nothing.
* the WF0 source and WF1 forcing climate figures, which
  ``scripts/render_figures.py`` re-renders through their rules' own functions.
* ``clim-year`` / ``clim-month`` were the per-subcatchment climate signatures.
  The consolidation retired that family along with its producer
  (``climate_analysis/subcatchment_climate.py``) and ``plot_clim``, so there is
  no longer a rule-side function for a preview to call. The climate a reader
  wants is the map/series family under ``forcing/plots/``, which is the second
  exclusion above.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_FIXTURE = REPO_ROOT / "test_case" / "test_local"
DEFAULT_OUT = REPO_ROOT / ".tmp" / "preview-plots"

MODEL_SUBDIR = Path("models") / "hydrology" / "wflow"


def _model_dir(project_dir: Path) -> Path:
    return project_dir / MODEL_SUBDIR


def _render_hydro(project_dir: Path, out_dir: Path) -> list[Path]:
    """Hydrograph from a finished run's discharge, simulated only.

    ``qobs`` is left None deliberately — see the module docstring on why observed
    discharge is not substituted.
    """
    import xarray as xr

    from blueearth_cst.experiment.wflow_response_reader import ResponseRequest
    from blueearth_cst.shared.plot_evaluation import Station, plot_hydrograph
    from blueearth_cst.shared.response_tables import open_table_responses

    record = (
        _model_dir(project_dir) / "historical/default/run/_engine/response_tables.json"
    )
    response = open_table_responses("preview", record, ResponseRequest(("q",)))[0]
    qsim = xr.DataArray(
        response.values,
        coords={"time": response.time.to_numpy()},
        dims=("time",),
        name="Q",
    )
    station = Station(int(response.location_id))
    plot_hydrograph(qsim, station, str(out_dir))
    return sorted(out_dir.glob("hydrograph*.png"))


#: name -> (one-line description, input probe, renderer)
#:
#: The probe is what ``--list`` reports on, so a missing fixture is named as a
#: missing FILE rather than surfacing later as an exception inside a plot call.
RENDERERS = {
    "hydro": (
        "Simulated hydrograph at the first discharge column",
        lambda p: _model_dir(p) / "historical/default/run/_engine/response_tables.json",
        _render_hydro,
    ),
}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("names", nargs="*", help="families to render (default: none)")
    parser.add_argument(
        "--all", action="store_true", help="render every available family"
    )
    parser.add_argument(
        "--list", action="store_true", help="show families and input status"
    )
    parser.add_argument(
        "--project-dir",
        type=Path,
        default=DEFAULT_FIXTURE,
        help=f"project tree to read (default: {DEFAULT_FIXTURE})",
    )
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--open", action="store_true", help="open the output folder when done"
    )
    args = parser.parse_args(argv)

    project_dir = args.project_dir.resolve()

    if args.list or not (args.names or args.all):
        print(f"project-dir: {project_dir}\n")
        for name, (desc, probe, _) in RENDERERS.items():
            path = probe(project_dir)
            status = "ok     " if path.exists() else "MISSING"
            print(f"  {status}  {name:<12} {desc}")
            if not path.exists():
                print(f"{'':>13}needs {path.relative_to(project_dir)}")
        print("\nMap figures: scripts/render_figures.py (wf0, wf1-forcing, spatial)")
        return 0

    unknown = [n for n in args.names if n not in RENDERERS]
    if unknown:
        parser.error(f"unknown: {', '.join(unknown)}. Known: {', '.join(RENDERERS)}")

    import matplotlib

    matplotlib.use("Agg")

    selected = list(RENDERERS) if args.all else args.names
    args.out_dir.mkdir(parents=True, exist_ok=True)

    failures = 0
    for name in selected:
        _, probe, render = RENDERERS[name]
        if not probe(project_dir).exists():
            print(f"skip {name}: missing {probe(project_dir)}")
            failures += 1
            continue
        written = render(project_dir, args.out_dir)
        for path in written:
            print(f"{name}: {path}")
        if not written:
            print(f"{name}: rendered, but wrote no file matching its expected name")
            failures += 1

    if args.open and sys.platform == "win32":
        subprocess.run(["explorer", str(args.out_dir)], check=False)

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
