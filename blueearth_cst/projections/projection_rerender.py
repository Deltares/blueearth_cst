"""Re-render WF2 projection figures (rule 2.06) from a finished run.

Calls ``plot_proj_timeseries.render_suite`` on the run's retained analysis
tables and record, writing to a separate directory: a few minutes at 600 dpi,
not a WF2 rerun.
Scenario order falls back to the IPCC colour order rather than the run config.
``scenarios`` drops the other SSPs' rows from every table first, previewing a
run whose config has since narrowed its scenario list.
Front end: ``scripts/render_figures.py``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

#: Where WF2 writes each ensemble's products, relative to ``project_dir``.
PROJECTIONS_DIR = Path("data") / "climate" / "projections"


def rerender_projections(
    project_dir: str | Path,
    out_dir: str | Path,
    annotation: str = "none",
    scenarios: list[str] | None = None,
) -> dict[str, Path]:
    """Write each ensemble's suite into ``out_dir/projections/<ensemble>/plots``."""
    from blueearth_cst.projections.plot_proj_timeseries import render_suite
    from blueearth_cst.projections.projection_analysis import (
        ensemble_summary,
        product_availability,
        seasonal_values,
        with_variability_band,
    )
    from blueearth_cst.projections.projection_figures import FIGURE_KEYS

    records = sorted(
        (Path(project_dir) / PROJECTIONS_DIR).glob("*/_engine/analysis_record.json")
    )
    if not records:
        raise FileNotFoundError(f"{project_dir}: no retained WF2 analysis record")
    written = {}
    for record_path in records:
        ensemble_dir = record_path.parents[1]
        tables = {
            path.stem: pd.read_csv(path)
            for path in (ensemble_dir / "analysis").glob("*.csv")
        }
        if scenarios:
            keep = {*scenarios, "historical"}
            tables = {
                name: table.loc[table.scenario.isna() | table.scenario.isin(keep)]
                if "scenario" in table
                else table
                for name, table in tables.items()
            }
        # A run made before the product list changed has a stale availability
        # table; rebuild it from the retained tables with the analysis step's
        # own function.
        availability = tables["availability"]
        if set(availability.filename) != set(FIGURE_KEYS):
            availability = product_availability(
                {
                    "coverage": tables["coverage"],
                    "trajectories": tables["member_anomalies"],
                    "models": tables["model_changes"],
                    "climatology": tables["climatology"],
                    "member_ratios": tables["variability_member_ratios"],
                    "model_ratios": tables["variability_model_ratios"],
                },
                sorted(availability.window_id.dropna().unique()),
            )
        availability["window_id"] = availability.window_id.fillna("")
        tables["availability"] = availability
        # Runs from before the variability band get it from their own
        # per-member changes, by the analysis step's function.
        for name in ("change_summary", "monthly_change_summary"):
            if "noise_halfwidth" not in tables[name]:
                tables[name] = with_variability_band(tables[name], tables["changes"])
        if "seasonal_values" not in tables:
            tables["seasonal_values"] = seasonal_values(
                tables["climatology"], tables["changes"]
            )
            tables["seasonal_values_summary"] = ensemble_summary(
                tables["seasonal_values"]
            )
        plot_dir = Path(out_dir) / "projections" / ensemble_dir.name / "plots"
        render_suite(
            tables,
            json.loads(record_path.read_text(encoding="utf-8")),
            plot_dir,
            annotation=annotation,
            scenarios=scenarios,
            captions_json=plot_dir.parent / "_engine" / "figure_captions.json",
        )
        written[ensemble_dir.name] = plot_dir
    return written
