"""Re-render WF0 source figures from a finished run's retained products.

The same ``diagnostic_render.render`` that rule 0.04b calls, fed from files the
run already wrote instead of Snakemake params, so a figure revision can be
reviewed without rerunning the workflow. Front end: ``scripts/render_figures.py``.

Everything the rule passes is recovered from the run: settings and water-year
start from each store's ``_engine/diagnostics.json``, the display reference via
``shared.retained_reference`` and overlays from ``data/spatial/geoms``. Output
goes to a separate directory, never the run's ``plots/``. Not covered:
per-subbasin figures and the rule 0.05 comparison set.
"""

from __future__ import annotations

import json
from pathlib import Path

from blueearth_cst.climate_analysis.diagnostic_outputs import (
    diagnostic_source_outputs,
    retarget_render,
)
from blueearth_cst.climate_analysis.diagnostic_render import render
from blueearth_cst.shared.retained_reference import load_retained_reference

#: Overlay layers the climate map rules read from the spatial units.
OVERLAY_LAYERS = ("basins", "subbasins", "rivers", "locations")


def overlay_paths(project_dir: str | Path) -> dict[str, str]:
    """The run's spatial overlay GeoJSON files, as rules 0.04b and 1.12 pass them."""
    geoms = Path(project_dir) / "data" / "spatial" / "geoms"
    return {key: str(geoms / f"{key}.geojson") for key in OVERLAY_LAYERS}


def source_stores(project_dir: str | Path, reference: dict) -> dict[str, Path]:
    """``{source: store_dir}`` for the run's candidate sources, in reference order.

    Candidates come from the reference, not from whatever folders exist: an
    orphaned store from an earlier period would otherwise join the pooled
    anomaly bounds silently. Two stores claiming one source is an error.
    """
    historical = Path(project_dir) / "data" / "climate" / "historical"
    candidates = [entry["source"] for entry in reference["identity"]["sources"]]
    stores = {}
    for metadata in sorted(historical.glob("*/_engine/diagnostics.json")):
        sources = json.loads(metadata.read_text(encoding="utf-8"))["sources"]
        if len(sources) != 1 or sources[0] not in candidates:
            continue  # the comparison folder, or a store this run did not use
        if sources[0] in stores:
            raise ValueError(
                f"Two stores hold {sources[0]} diagnostics: {stores[sources[0]]} "
                f"and {metadata.parents[1]}; remove the orphan or re-run WF0"
            )
        stores[sources[0]] = metadata.parents[1]
    return {source: stores[source] for source in candidates if source in stores}


def rerender_sources(
    project_dir: str | Path,
    out_dir: str | Path,
    sources: list[str] | None = None,
    annotation: str = "none",
) -> dict[str, Path]:
    """Render each source's WF0 figure set into ``out_dir/<store name>/plots``.

    Returns ``{source: plot_dir}``. Raises when a requested source has no
    retained diagnostics, rather than rendering a partial set silently.
    """
    project_dir, out_dir = Path(project_dir), Path(out_dir)
    reference = load_retained_reference(project_dir)
    stores = source_stores(project_dir, reference)
    if not stores:
        raise FileNotFoundError(f"{project_dir}: no retained WF0 source diagnostics")
    missing = sorted(set(sources or ()) - set(stores))
    if missing:
        raise ValueError(
            f"No retained WF0 diagnostics for {missing}; available: {sorted(stores)}"
        )
    pooled_roots = [str(store) for store in stores.values()]
    written = {}
    for source in sources or list(stores):
        store = stores[source]
        metadata = json.loads(
            (store / "_engine" / "diagnostics.json").read_text(encoding="utf-8")
        )
        settings = metadata["settings"]
        paths = retarget_render(
            diagnostic_source_outputs(store, source, settings), out_dir / store.name
        )
        render(
            paths,
            settings,
            int(metadata["water_year_start"]),
            [source],
            pooled_roots,
            overlay_paths(project_dir),
            annotation=annotation,
            reference_record=reference,
        )
        written[source] = Path(paths["render_root"]) / "plots"
    return written
