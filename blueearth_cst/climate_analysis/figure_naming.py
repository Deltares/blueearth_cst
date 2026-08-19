"""The WF0 figure filename grammar and its controlled vocabulary.

``dev/reference/wf0-figure-filename-rule.md`` (agreed 2026-08-17) fixes the shape::

    <dataset_scope>_<variable>_<plot_context>_<spatial_scope>.<extension>

with each field drawn from a controlled vocabulary rather than spelled ad hoc at
each call site. This module IS that vocabulary: every WF0 figure name is built
here, so a new context or scope is added in one place and a typo is a raised
error rather than a file nobody finds.

Four fields, and what each answers:

``dataset_scope``
    WHICH dataset — the source id (``era5``, ``chirps``) for a single-source
    figure, :data:`COMPARISON_SCOPE` for one carrying several. Comparison
    figures deliberately do NOT list their datasets in the filename: the legend
    and the run record carry them, so the name stays stable as the compared set
    changes.
``variable``
    The canonical unabridged name — ``precip``, ``temp``, ``pet``. Never
    abbreviated; the rule says so explicitly.
``plot_context``
    The temporal interpretation AND the plot form together, because the two are
    not independent: a monthly climatology drawn as boxes and drawn as lines are
    different figures answering different questions.
``spatial_scope``
    What area the figure reduces to, or frames on.

What is deliberately ABSENT: workflow id, project name, units and analysis
period. Those belong to the run, not to the figure, and putting them in the name
makes every filename churn when a window moves.

**Scope of this grammar, today.** WF0's figures only -- the per-source set
(rule 0.05) and the cross-source comparison (rule 0.06). The wflow FORCING
family (rule 1.13) keeps its ``forcing_<var>_<kind>.png`` names, because the
rule stages WF0 first and extending it is a separate migration with its own
consumers. ``climate_figures`` therefore carries both spellings; see its
``_legacy_figure_name``.
"""

# NO `from __future__ import annotations`: imported by `script:` modules.
from typing import Optional

#: ``dataset_scope`` for a figure carrying more than one dataset.
COMPARISON_SCOPE = "comparison"

#: Every ``plot_context`` token, mapped to what it means. A figure kind that is
#: not in here cannot be named, which is the point: the vocabulary is closed and
#: extended deliberately.
#:
#: Built from the abbreviations the rule fixes -- ``ts`` time series, ``clim``
#: climatology, ``box`` distribution box plot -- so a reader who knows four
#: tokens can read every name.
PLOT_CONTEXTS = {
    "annual_ts": "one value per year, as a time series",
    "annual_clim_map": "per-year aggregate averaged over years, as a map",
    "monthly_box": "per-calendar-month distribution across years, as boxes",
    "monthly_clim_line": "per-calendar-month mean across years, as lines",
}

#: ``spatial_scope`` tokens that name a fixed area.
#:
#: ``_avg`` REDUCES the field to one series over that area; ``_ext`` FRAMES a
#: map on it and reduces nothing. The suffix is the difference between a number
#: and a picture, so the two are never interchangeable.
FIXED_SPATIAL_SCOPES = {
    "basin_avg": "domain mean over the basin's cells",
    "basin_ext": "framed on the basin's extent",
    "source_ext": "framed on the source grid's own extent",
}

#: Prefixes for the scopes that carry an id. ``station_<id>`` is reserved for
#: the evaluation family and is not produced here yet.
#:
#: ``st`` is deliberately NOT used for station: it already identifies a
#: stress-test member elsewhere in the project, and one token meaning two things
#: across a project is how a filename stops being self-describing.
ID_SPATIAL_SCOPES = {
    "subbasin": ("avg", "ext"),
    "station": ("avg",),
}


def subbasin_scope(subbasin_id, kind: str = "avg") -> str:
    """The ``spatial_scope`` token for one subbasin, e.g. ``subbasin_1010_avg``."""
    if kind not in ID_SPATIAL_SCOPES["subbasin"]:
        raise ValueError(
            f"subbasin scope kind {kind!r}; expected one of "
            f"{sorted(ID_SPATIAL_SCOPES['subbasin'])}"
        )
    token = str(subbasin_id).strip().lower().replace(" ", "_")
    if not token:
        raise ValueError("subbasin_scope: the id is empty")
    return f"subbasin_{token}_{kind}"


def _validate_spatial_scope(spatial_scope: str) -> str:
    if spatial_scope in FIXED_SPATIAL_SCOPES:
        return spatial_scope
    for prefix, kinds in ID_SPATIAL_SCOPES.items():
        if spatial_scope.startswith(f"{prefix}_") and spatial_scope.endswith(
            tuple(f"_{kind}" for kind in kinds)
        ):
            return spatial_scope
    raise ValueError(
        f"unknown spatial_scope {spatial_scope!r}; expected one of "
        f"{sorted(FIXED_SPATIAL_SCOPES)} or an id scope "
        f"({', '.join(f'{p}_<id>_{k}' for p, ks in ID_SPATIAL_SCOPES.items() for k in ks)}). "
        "Add it to figure_naming rather than spelling it at the call site."
    )


def figure_filename(
    dataset_scope: str,
    variable: str,
    plot_context: str,
    spatial_scope: str,
    extension: str = "png",
) -> str:
    """One figure filename, validated against the controlled vocabulary.

    Raises
    ------
    ValueError
        For an unknown ``plot_context`` or ``spatial_scope``, or an empty
        ``dataset_scope``/``variable``. Loud on purpose: the Snakefile declares
        its outputs through this function and the plotter writes through it, so
        a token that silently passed here would surface as a
        ``MissingOutputException`` at the end of the job instead.
    """
    if plot_context not in PLOT_CONTEXTS:
        raise ValueError(
            f"unknown plot_context {plot_context!r}; expected one of "
            f"{sorted(PLOT_CONTEXTS)}. Add it to figure_naming rather than "
            "spelling it at the call site."
        )
    _validate_spatial_scope(spatial_scope)
    for field, value in (("dataset_scope", dataset_scope), ("variable", variable)):
        if not str(value).strip():
            raise ValueError(f"figure_filename: {field} is empty")
    return f"{dataset_scope}_{variable}_{plot_context}_{spatial_scope}.{extension}"


def map_spatial_scope(extent_policy: Optional[str]) -> str:
    """The ``spatial_scope`` a map figure takes, from how it is FRAMED.

    Derived rather than hardcoded, because the framing has already flipped once:
    ``climate_figures.MAP_EXTENT["source"]`` was ``raster`` on 2026-08-16 and
    ``basin`` on 2026-08-17. Both ``basin_ext`` and ``source_ext`` are in the
    vocabulary precisely so the name can follow the decision; a frozen token
    would mislabel the figure the next time it moves.
    """
    return "basin_ext" if extent_policy == "basin" else "source_ext"
