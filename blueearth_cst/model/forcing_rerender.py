"""Re-render WF1 forcing figures (rule 1.12) from a finished run.

Calls ``plot_map_forcing.plot_forcing`` on the run's built Wflow model with the
retained historical reference, writing to a separate directory. The forcing is
re-reduced from the model's own ``inmaps`` file, so this takes as long as the
rule's figure step (minutes), not seconds. Front end: ``scripts/render_figures.py``.
"""

from __future__ import annotations

import calendar
from pathlib import Path

from blueearth_cst.climate_analysis.diagnostic_rerender import overlay_paths
from blueearth_cst.model.plot_map_forcing import plot_forcing
from blueearth_cst.shared.retained_reference import load_retained_reference
from blueearth_cst.shared.snake_utils import water_year_end_anchor

#: The model root rule 1.12 passes, relative to ``project_dir``.
MODEL_DIR = Path("models") / "hydrology" / "wflow"


def rerender_forcing(
    project_dir: str | Path, out_dir: str | Path, annotation: str = "none"
) -> dict[str, Path]:
    """Write the nine forcing figures into ``out_dir/forcing/plots``."""
    project_dir = Path(project_dir)
    model_dir = project_dir / MODEL_DIR
    if not (model_dir / "staticmaps.nc").is_file():
        raise FileNotFoundError(f"{model_dir}: no built Wflow model to re-render")
    reference = load_retained_reference(project_dir)
    # The reference records the reporting month the rules derived from
    # water_year_start, so the forcing years align with WF0's.
    month = calendar.month_abbr[int(reference["identity"]["reporting_month"])]
    plot_dir = Path(out_dir) / "forcing" / "plots"
    plot_forcing(
        wflow_root=model_dir,
        plot_dir=plot_dir,
        geoms_dir=Path(overlay_paths(project_dir)["basins"]).parent,
        anchor=water_year_end_anchor(month),
        annotation=annotation,
        reference_record=reference,
    )
    return {"forcing": plot_dir}
