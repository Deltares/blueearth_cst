"""The declared climate figure names: WF1 model-forcing and WF0 source sets.

WF1 writes model-grid forcing QA under
``models/hydrology/wflow/forcing/plots/`` (rule 1.12, drawn by
``model/forcing_figures.py``); WF0 source figures are drawn by
``diagnostic_render``. This module only declares names: Snakefiles import it at
parse time, so it stays pure Python over the registries below. Its rendering
half retired with the legacy cartographic module (2026-10-29).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Optional

from blueearth_cst.climate_analysis.figure_naming import (
    figure_filename,
    map_spatial_scope,
)
from blueearth_cst.climate_analysis.source_capabilities import (
    PRECIP_ONLY_SOURCES as PRECIP_ONLY_SOURCES,
)
from blueearth_cst.climate_analysis.source_capabilities import (
    source_climate_vars as source_climate_vars,
)
from blueearth_cst.shared.variable_registry import CLIMATE_VARS as _CLIMATE_VARS

#: One entry per variable: label and unit for the axes, and how the variable
#: aggregates in time. ``sum`` is a flux that accumulates (a yearly TOTAL);
#: ``mean`` is a state that averages. Getting this wrong is not cosmetic -- a
#: summed temperature is meaningless and a meaned rainfall understates by ~365x.
#:
#: **Re-exported, not defined here** (R14 `C-57`). The same three variables were
#: also described by every project's `variables:` block, in a second vocabulary
#: that knew nothing about this one; `shared/variable_registry.py` is now the
#: single record and this is its presentation view. The name and the value are
#: unchanged, including insertion order, so every reader of this module is
#: untouched. `C-49` -- getting the table out of Python altogether -- is a
#: separate row and stays outside this phase.
CLIMATE_VARS = _CLIMATE_VARS

#: One entry per figure kind. ``map`` is the spatial view (climatological
#: field), ``annual`` and ``monthly`` are the temporal views (domain mean).
FIGURE_KINDS = ("map", "annual", "monthly")

#: Datasets this set is applied to. The key is the filename prefix AND the
#: title, so a figure copied out of its directory still says what it is -- the
#: reason the source figures carried a ``source_`` prefix before this module,
#: now applied to both sides so the two directories are directly comparable.
DATASETS = {
    "source": "source grid (raw extraction)",
    "forcing": "model grid (wflow forcing)",
}

#: How each family FRAMES its maps, and it is not the same answer for both.
#:
#: ``"basin"`` crops to the basin's bounding box; ``"raster"`` shows the field's
#: own full footprint. The forcing is masked to the basin, so for it the two are
#: nearly the same box and ``"basin"`` is simply the tidier one.
#:
#: The SOURCE side has been both, and is BACK on ``"basin"`` as of 2026-08-17
#: (owner's ruling), reversing the 2026-08-16 ruling that had moved it to
#: ``"raster"`` the day before. Both directions are recorded here because the
#: trade is real and neither answer is free.
#:
#: ``"raster"`` was chosen because a raw extraction is a handful of reanalysis
#: cells reaching well past the catchment — at ERA5's 0.25 degrees a small basin
#: sits inside one or two of them — so cropping to the basin shows a single flat
#: class and calls it a map of the climate. Measured on the rapid fixture: the
#: whole era5 extraction is 4x5 cells and the basin is 0.200 x 0.133 degrees,
#: under one cell wide.
#:
#: ``"basin"`` is chosen anyway, because the question the figure pair answers is
#: what downscaling changed, and that comparison needs both families framed on
#: the same footprint. A flat source panel is then INFORMATION — it says the
#: source grid does not resolve this basin — rather than a defect of the figure.
#: On a fine source (CHIRPS at 0.05 degrees) the same frame is genuinely
#: informative.
#:
#: ``_raster_within`` crops before classifying the colourbar, so source-local
#: scales describe the drawn cells. Explicit pooled scales supplied by direct
#: callers can cover a wider footprint; canonical WF0/WF1 plots omit them.
MAP_EXTENT = {
    "source": "basin",
    "forcing": "basin",
}

#: A year is plotted as a TOTAL only if it is essentially complete; below this
#: fraction of the modal timestep count it is dropped instead. A truncated first
#: or last year otherwise draws a dip that looks like climate and is calendar.
_COMPLETE_YEAR_FRACTION = 0.9


#: Which ``plot_context`` token of the WF0 grammar each figure kind carries.
#: ``monthly`` is ``monthly_box`` HERE because this family draws the
#: year-to-year distribution as boxes; the comparison family draws the same
#: months as lines and names itself ``monthly_clim_line`` accordingly.
KIND_PLOT_CONTEXTS = {
    "map": "annual_clim_map",
    "annual": "annual_ts",
    "monthly": "monthly_box",
}

#: The kinds that REDUCE the field to a series over an area, and therefore exist
#: once per spatial scope. ``map`` is not one of them: it draws the field itself,
#: so a per-subbasin map would be the same raster cropped twelve ways.
AGGREGATED_KINDS = tuple(k for k in FIGURE_KINDS if k != "map")


def _legacy_figure_name(dataset: str, var: str, kind: str) -> str:
    """``<dataset>_<variable>_<kind>.png`` — the pre-grammar spelling.

    Retained for the FORCING family only. ``wf0-figure-naming.md`` stages
    the new grammar at WF0 first, and the forcing figures have their own
    consumers and their own directory; migrating them is a separate step with
    its own rename sweep.
    """
    return f"{dataset}_{var}_{kind}.png"


def source_figure_names(
    clim_source: str,
    variables: Optional[Sequence[str]] = None,
    spatial_scopes: Sequence[str] = ("basin_avg",),
) -> list[str]:
    """Every filename the SOURCE family writes for one source, in a stable order.

    The WF0 grammar (``climate_analysis.figure_naming``), so the
    ``dataset_scope`` is the source id itself — ``era5_precip_annual_ts_basin_avg.png``
    — rather than the literal word ``source``. A figure copied out of its
    directory then still says which dataset it came from, which the old
    ``source_precip_annual.png`` did not.

    ``spatial_scopes`` lists the areas the AGGREGATED kinds are drawn for; the
    map is drawn once, framed per :data:`MAP_EXTENT`. Pass only the scopes whose
    count is knowable at DAG-parse time — the per-subbasin scopes are not, and
    the Snakefile declares those through a ``directory()`` instead.

    It must be the SAME list the drawing rule writes: narrow only the
    declaration and the extra files are undeclared, narrow only the drawing and
    the job ends in ``MissingOutputException``.
    """
    map_scope = map_spatial_scope(MAP_EXTENT["source"])
    names = []
    for var in _resolve_variables(variables):
        for kind in FIGURE_KINDS:
            context = KIND_PLOT_CONTEXTS[kind]
            if kind == "map":
                names.append(figure_filename(clim_source, var, context, map_scope))
                continue
            names.extend(
                figure_filename(clim_source, var, context, scope)
                for scope in spatial_scopes
            )
    return names


def figure_names(dataset: str, variables: Optional[Sequence[str]] = None) -> list[str]:
    """Every filename this module writes for ``dataset``, in a stable order.

    The FORCING family's declaration (rule 1.12), on the legacy spelling. The
    source family is named by :func:`source_figure_names` under the WF0 grammar;
    the two spellings coexist deliberately while that migration is staged.
    """
    if dataset not in DATASETS:
        raise ValueError(
            f"unknown dataset {dataset!r}; expected one of {sorted(DATASETS)}"
        )
    return [
        _legacy_figure_name(dataset, var, kind)
        for var in _resolve_variables(variables)
        for kind in FIGURE_KINDS
    ]


def _resolve_variables(variables: Optional[Sequence[str]]) -> tuple[str, ...]:
    """Validate a variable subset, defaulting to the full canonical set.

    Order follows :data:`CLIMATE_VARS`, not the caller's, so two callers passing
    the same variables in a different order still declare the same filenames in
    the same sequence.
    """
    if variables is None:
        return tuple(CLIMATE_VARS)
    requested = set(variables)
    unknown = sorted(requested - set(CLIMATE_VARS))
    if unknown:
        raise ValueError(
            f"unknown climate variables {unknown}; expected a subset of "
            f"{list(CLIMATE_VARS)}"
        )
    if not requested:
        raise ValueError("variables must name at least one climate variable")
    return tuple(var for var in CLIMATE_VARS if var in requested)
