"""Render WF0 results with the plotting core alone, outside CST execution.

Proves the portable boundary of ``blueearth_cst/plotting/``: only that package
and ``blueearth_cst/__init__.py`` are copied next to the output, then imported
under ``python -I`` with nothing but numpy and matplotlib. Inputs are plain
files: WF0 CSV tables, the historical map-reference JSON, basin/river GeoJSON,
and one ``.npz`` holding a native annual-climatology field.

Two steps. ``export`` needs the Pixi environment (xarray) and runs once::

    pixi run python dev/scripts/plotting_standalone_demo.py export <project_dir> <out_dir>

``render`` needs only numpy and matplotlib; run it isolated, with any Python::

    python -I dev/scripts/plotting_standalone_demo.py render <project_dir> <out_dir>

``render`` writes three figures (annual series, monthly cycle, precipitation
map) with the neutral default and again inside ``use_preset`` with an
explicitly supplied preset, and fails if a CST module, pandas or PyYAML was
imported or the Matplotlib backend changed at import.

Not part of a run: this is a maintenance proof (AGENTS.md, Repo Map).
"""

import argparse
import csv
import dataclasses
import json
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SOURCES = {"era5": ("ERA5", "#0D38E0", "o"), "chirps": ("CHIRPS", "#CC593D", "s")}
FORBIDDEN = ("blueearth_cst.shared", "pandas", "yaml", "xarray", "snakemake", "hydromt")


def _historical(project: Path) -> Path:
    return project / "data/climate/historical"


def _source_dirs(project: Path) -> dict[str, Path]:
    found = {}
    for source in SOURCES:
        matches = sorted(_historical(project).glob(f"{source}_*"))
        if matches:
            found[source] = matches[0]
    if not found:
        raise SystemExit(f"no WF0 source directories under {_historical(project)}")
    return found


def export(project: Path, out: Path) -> None:
    """Write native precipitation fields and cell edges for the selected source."""
    import numpy as np
    import xarray as xr

    reference = json.loads(
        (_historical(project) / "_engine/map_reference.json").read_text(
            encoding="utf-8"
        )
    )
    source = reference["frame"]["selected_source"]
    path = _source_dirs(project)[source] / "tables/annual_climatology.nc"
    with xr.open_dataset(path) as fields:
        field = fields["precip"].load()

    def edges(centres):
        step = float(centres[1] - centres[0])
        return np.concatenate([centres - step / 2, [centres[-1] + step / 2]])

    out.mkdir(parents=True, exist_ok=True)
    np.savez(
        out / "precip_field.npz",
        values=field.values,
        longitude_edges=edges(field["longitude"].values.astype(float)),
        latitude_edges=edges(field["latitude"].values.astype(float)),
        source=source,
    )
    print(f"exported {path.relative_to(project)} -> {out / 'precip_field.npz'}")


def _isolated_core(out: Path) -> None:
    """Copy only the core next to the outputs and import it from there."""
    core = out / "core"
    shutil.rmtree(core, ignore_errors=True)
    (core / "blueearth_cst").mkdir(parents=True)
    shutil.copy2(REPO / "blueearth_cst/__init__.py", core / "blueearth_cst")
    shutil.copytree(
        REPO / "blueearth_cst/plotting",
        core / "blueearth_cst/plotting",
        ignore=shutil.ignore_patterns("__pycache__"),
    )
    sys.path.insert(0, str(core))


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def render(project: Path, out: Path) -> None:
    if not sys.flags.isolated:
        raise SystemExit("run render with python -I so the checkout is not importable")
    _isolated_core(out)
    import matplotlib

    backend = matplotlib.get_backend()
    import numpy as np
    from matplotlib import pyplot as plt

    from blueearth_cst.plotting import prepared_charts as pc
    from blueearth_cst.plotting.fundamentals import MONTH_ABBR
    from blueearth_cst.plotting.prepared_maps import (
        MapFrame,
        MapScale,
        RasterMap,
        draw_raster_map,
    )
    from blueearth_cst.plotting.preset import NEUTRAL, use_preset

    leaked = sorted(
        name for name in sys.modules if name.startswith(FORBIDDEN) or name == "yaml"
    )
    if leaked or matplotlib.get_backend() != backend:
        raise SystemExit(f"portable boundary broken: {leaked or 'backend changed'}")

    sources = _source_dirs(project)
    annual = {s: _rows(d / "tables/annual_indices.csv") for s, d in sources.items()}
    years = np.array(
        sorted({int(r["reporting_year"]) for v in annual.values() for r in v})
    )
    lines = []
    for source, rows in annual.items():
        by_year = {int(r["reporting_year"]): float(r["total"] or "nan") for r in rows}
        label, color, marker = SOURCES[source]
        values = np.array([by_year.get(int(y), np.nan) for y in years])
        lines.append(pc.AnnualLine(label, color, marker, values))
    bands = []
    for source, directory in sources.items():
        rows = sorted(
            _rows(directory / "tables/precip_climatology.csv"),
            key=lambda r: int(r["calendar_month"]),
        )
        label, color, marker = SOURCES[source]
        mean, low, high = (
            np.array([float(r[name]) for r in rows]) for name in ("mean", "p10", "p90")
        )
        bands.append(pc.MonthlyBand(label, color, marker, mean, low, high))

    reference = json.loads(
        (_historical(project) / "_engine/map_reference.json").read_text(
            encoding="utf-8"
        )
    )
    frame_record, scale_record = reference["frame"], reference["variables"]["precip"]
    field = np.load(out / "precip_field.npz")
    raster = RasterMap(
        field["values"], field["longitude_edges"], field["latitude_edges"]
    )
    frame = MapFrame(
        tuple(frame_record["extent"]),
        frame_record["longitude_ticks"],
        frame_record["latitude_ticks"],
        frame_record["aspect"],
    )
    scale = MapScale(
        scale_record["units"],
        scale_record["boundaries"] or [],
        scale_record["ticks"] or [],
        [tuple(c) for c in scale_record["colors"] or []],
        scale_record["under_color"],
        scale_record["over_color"],
        scale_record["missing_color"],
        scale_record["status"],
    )
    overlays = {}
    for key in ("basins", "rivers"):
        path = project / f"data/spatial/geoms/{key}.geojson"
        if path.is_file():
            overlays[key] = json.loads(path.read_text(encoding="utf-8"))["features"]

    # An explicitly supplied preset: same layout, a warmer palette and larger type.
    supplied = dataclasses.replace(
        NEUTRAL,
        fonts={"label": 8, "tick": 7, "note": 7},
        colors={**NEUTRAL.colors, "ink": "#3b1f0e", "muted": "#7a5230"},
        rcparams={**NEUTRAL.rcparams, "font.size": 8, "axes.labelcolor": "#3b1f0e"},
    )

    def draw(tag: str) -> None:
        figures = {
            "annual": pc.annual_series(years, lines, ylabel="Precipitation (mm yr⁻¹)"),
            "monthly": pc.monthly_cycle(
                bands,
                month_labels=MONTH_ABBR,
                ylabel="Precipitation (mm month⁻¹)",
                band_label="10th–90th percentile",
                zero=True,
            ),
            "map": draw_raster_map(raster, frame, scale, overlays),
        }
        for name, figure in figures.items():
            path = out / f"{tag}_{name}.png"
            figure.savefig(path, dpi=200)
            plt.close(figure)
            print(f"wrote {path.name}")

    with use_preset(NEUTRAL):
        draw("neutral")
    with use_preset(supplied):
        draw("supplied")
    print("portable boundary held: no CST, pandas or PyYAML module was imported")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("step", choices=("export", "render"))
    parser.add_argument("project_dir", type=Path)
    parser.add_argument("out_dir", type=Path)
    args = parser.parse_args()
    step = export if args.step == "export" else render
    step(args.project_dir.resolve(), args.out_dir.resolve())


if __name__ == "__main__":
    main()
