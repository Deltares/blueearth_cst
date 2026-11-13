"""Re-render toolbox figures from a finished run's retained products.

Each figure family calls the same rendering function its workflow rule calls,
fed from files the run already wrote, so a plotting revision can be reviewed
without rerunning the workflow. Output goes to ``--out``, never into the run.

    pixi run python scripts/render_figures.py --list
    pixi run python scripts/render_figures.py wf0 --project <project_dir> --out <dir>
    pixi run python scripts/render_figures.py wf2 --project <project_dir> --out <dir>         --scenario ssp126 --scenario ssp585
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib  # noqa: E402

matplotlib.use("Agg")


def _historical_hydrology(args):
    from blueearth_cst.model.plot_historical_comparison import render

    if args.source or args.scenario:
        raise ValueError("historical-hydrology uses the retained comparison selection")
    render(args.project, args.out, args.annotation)
    return {"historical-hydrology": args.out / "plots"}


def _wf0(args):
    from blueearth_cst.climate_analysis.diagnostic_rerender import rerender_sources

    return rerender_sources(
        args.project, args.out, sources=args.source, annotation=args.annotation
    )


def _wf1_forcing(args):
    from blueearth_cst.model.forcing_rerender import rerender_forcing

    if args.source:
        raise ValueError("--source does not apply to wf1-forcing")
    return rerender_forcing(args.project, args.out, annotation=args.annotation)


def _rain_gauges(args):
    from blueearth_cst.climate_analysis.plot_rain_gauges import render

    if args.source:
        raise ValueError("--source does not apply to rain-gauges (common source mask)")
    from blueearth_cst.climate_analysis.diagnostic_outputs import comparison_root

    historical = args.project / "data/climate/historical"
    comparison = comparison_root(historical)
    if (comparison / "locations.csv").is_file():
        from blueearth_cst.climate_analysis.plot_location_comparison import (
            render as render_locations,
        )

        render_locations(comparison, args.out, args.annotation)
        return {"locations": args.out / "plots/locations"}
    root = comparison / "stations"
    if not root.exists():
        # Retained runs created before station comparisons joined comparison/.
        root = historical / "rain_gauges"
    render(root, args.out, args.annotation)
    return {"rain_gauges": args.out / "plots"}


def _wf0_comparison(args):
    from blueearth_cst.climate_analysis.diagnostic_rerender import rerender_comparisons

    if args.source or args.scenario:
        raise ValueError("wf0-comparison uses the retained complete source selection")
    return rerender_comparisons(args.project, args.out, args.annotation)


def _budyko(args):
    from blueearth_cst.model.plot_budyko import render

    if args.source:
        raise ValueError("--source does not apply to budyko")
    render(
        args.project / "models/hydrology/wflow/evaluation/budyko",
        args.out,
        args.annotation,
    )
    return {"budyko": args.out / "plots"}


def _spatial(args):
    from blueearth_cst.shared.plot_spatial_maps import (
        SPATIAL_DIRNAME,
        plot_spatial_figure_set,
    )

    if args.source:
        raise ValueError("--source does not apply to spatial")
    plot_dir = args.out / "spatial" / "plots"
    plot_spatial_figure_set(
        args.project / SPATIAL_DIRNAME, plot_dir=plot_dir, annotation=args.annotation
    )
    return {"spatial": plot_dir}


def _wf2(args):
    from blueearth_cst.projections.projection_rerender import rerender_projections

    if args.source:
        raise ValueError("--source does not apply to wf2")
    return rerender_projections(
        args.project, args.out, annotation=args.annotation, scenarios=args.scenario
    )


#: Family name -> (what it renders, renderer). Add a family by pointing it at a
#: re-render function that sits beside its rule adapter in ``blueearth_cst``.
FAMILIES = {
    "historical-hydrology": (
        "Fixed-model historical discharge from retained comparison tables",
        _historical_hydrology,
    ),
    "budyko": ("WF1 observed water balance from retained tables", _budyko),
    "rain-gauges": (
        "Rain-gauge comparisons from retained station evidence",
        _rain_gauges,
    ),
    "wf0": (
        "WF0 per-source diagnostics, enabled subbasin figures and maps (comparisons use wf0-comparison)",
        _wf0,
    ),
    "wf0-comparison": (
        "WF0 basin-average and per-location comparisons from retained products",
        _wf0_comparison,
    ),
    "wf1-forcing": (
        "WF1 rule 1.12 forcing maps, annual series and monthly cycles (re-reduces forcing; minutes)",
        _wf1_forcing,
    ),
    "spatial": (
        "WF1 rule 1.11 spatial maps: elevation, subbasins, land cover, LAI, soils",
        _spatial,
    ),
    "wf2": (
        "WF2 rule 2.06 projection suite from the retained analysis tables (minutes)",
        _wf2,
    ),
}


def main(argv=None):
    """Return nonzero when a family's retained inputs are missing or invalid."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("family", nargs="?", choices=sorted(FAMILIES))
    parser.add_argument("--list", action="store_true", help="list figure families")
    parser.add_argument("--project", type=Path, help="finished run's project_dir")
    parser.add_argument("--out", type=Path, help="output directory (outside the run)")
    parser.add_argument(
        "--source", action="append", help="limit to a source (repeatable)"
    )
    parser.add_argument(
        "--scenario",
        action="append",
        help="wf2: keep only this SSP, as a rerun with it configured would (repeatable)",
    )
    parser.add_argument(
        "--annotation", choices=("none", "caption", "title"), default="none"
    )
    parser.add_argument("--traceback", action="store_true")
    args = parser.parse_args(argv)
    if args.list:
        for name, (summary, _) in FAMILIES.items():
            print(f"{name:6} {summary}")
        return 0
    if args.family is None or args.project is None or args.out is None:
        parser.error("family, --project and --out are required unless --list is used")
    if args.out.resolve().is_relative_to(args.project.resolve()):
        parser.error("--out must lie outside the run so retained figures stay intact")
    if args.scenario and args.family != "wf2":
        parser.error("--scenario applies to wf2 only")
    try:
        written = FAMILIES[args.family][1](args)
    except (OSError, KeyError, ValueError) as exc:
        if args.traceback:
            raise
        print(f"Rendering {args.family} failed: {exc}", file=sys.stderr)
        return 1
    for name, path in written.items():
        print(f"{name}: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
