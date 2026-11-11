"""The WF0 figure filename grammar and its controlled vocabulary.

The WF0 profile in ``blueearth_cst/plotting/docs/figure-naming.md`` defines the shape::

    <dataset>_<variable-or-scope>_<diagnostic>_<location>.<extension>

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
    Stable logical diagnostic key, translated to its shorter exported spelling.
    Plot-form words remain where they distinguish products, such as maps and boxes.
``spatial_scope``
    Stable support key, exported as basin, source, sub<ID>, loc<ID>, station<ID>,
    or a meaningful unique location name. Captions retain the precise support.

What is deliberately ABSENT: workflow id, project name, units and analysis
period. Those belong to the run, not to the figure, and putting them in the name
makes every filename churn when a window moves.

**Scope of this grammar, today.** WF0's figures only -- the per-source set
(rule 0.04) and the cross-source comparison (rule 0.05). The wflow FORCING
family (rule 1.12) keeps its ``forcing_<var>_<kind>.png`` names, because the
rule stages WF0 first and extending it is a separate migration with its own
consumers. ``climate_figures`` therefore carries both spellings; see its
``_legacy_figure_name``.
"""

# NO `from __future__ import annotations`: imported by `script:` modules.
import re
import unicodedata
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
    "annual_anomaly_map": "annual native-cell departure from the reference mean",
    "daily_agreement": "paired observed and native-cell daily precipitation",
    "monthly_agreement": "complete paired monthly observations and native-cell values",
    "annual_indices_ts": "annual climate-index series with explicit units",
    "seasonal_indices_ts": "configured-season climate-index series",
    "wet_spell_exceedance": "complete wet-spell empirical exceedance",
    "monthly_coverage": "legacy monthly completeness view; no longer emitted by WF0",
    "monthly_clim_band": "monthly climatology and 10th–90th percentile band",
    "annual_timing": "cumulative annual precipitation timing",
    "monthly_anomaly": "monthly departure calendar",
    "monthly_spi": "multiscale SPI calendar",
    "spi_events": "SPI drought duration and severity",
    "annual_rx1day_ts": "annual maximum one-day precipitation",
    "annual_rx5day_ts": "annual maximum five-day precipitation",
    "annual_sdii_ts": "annual observed wet-day intensity",
    "annual_intensity_ts": "annual Rx1day, Rx5day and wet-day intensity panels",
    "annual_wet_days_ts": "annual scaled wet-day count",
    "dry_spell_exceedance": "complete dry-spell exceedance",
    "monthly_anomaly_acf": "monthly anomaly autocorrelation",
    "annual_trend_interval": "annual Theil–Sen trends with intervals",
    "annual_trend_ts": "annual precipitation with Theil–Sen trend",
    "seasonal_anomaly": "seasonal precipitation–temperature anomaly scatter",
    "monthly_spi_fit": "calendar-month SPI distribution checks",
    "annual_trend_sensitivity": "start-year trend sensitivity",
    "annual_ts": "one value per year, as a time series",
    "annual_clim_map": "per-year aggregate averaged over years, as a map",
    "monthly_box": "per-calendar-month distribution across years, as boxes",
    "monthly_clim_line": "per-calendar-month mean across years, as lines",
}
THEMATIC_SCOPES = (
    "spells",
    "heavy_rainfall",
    "rainfall_characteristics",
    "temperature_extremes",
    "heatwaves",
)
PLOT_CONTEXTS.update(
    {
        f"{period}_{scope}_ts": f"{period} {scope.replace('_', ' ')} climate-index series"
        for period in ("annual", "seasonal")
        for scope in THEMATIC_SCOPES
    }
)

#: ``spatial_scope`` tokens that name a fixed area.
#:
#: Logical support retains avg/ext even though exported filenames shorten both
#: to the area name. A map diagnostic explicitly retains its map token.
FIXED_SPATIAL_SCOPES = {
    "basin_avg": "domain mean over the basin's cells",
    "basin_ext": "framed on the basin's extent",
    "source_ext": "framed on the source grid's own extent",
}

#: Prefixes for scopes carrying an id. Rain-gauge figures map station tokens
#: back to literal input IDs in their retained station table.
#:
#: ``st`` is deliberately NOT used for station: it already identifies a
#: stress-test member elsewhere in the project, and one token meaning two things
#: across a project is how a filename stops being self-describing.
ID_SPATIAL_SCOPES = {
    "subbasin": ("avg", "ext"),
    "station": ("avg",),
    "location": ("point",),
}

# Logical contexts stay stable for callers and retained records. Only their
# exported spelling is shortened; map and trend distinctions remain explicit.
SHORT_CONTEXTS = {
    "annual_ts": "annual",
    "annual_trend_ts": "annual_trend",
    "annual_clim_map": "annual_mean_map",
    "monthly_clim_band": "monthly_cycle",
    "monthly_clim_line": "monthly_cycle",
    "annual_timing": "timing",
    "annual_rx1day_ts": "rx1day_annual",
    "annual_rx5day_ts": "rx5day_annual",
    "annual_sdii_ts": "sdii_annual",
    "annual_intensity_ts": "intensity_annual",
    "annual_wet_days_ts": "wet_days_annual",
    "monthly_anomaly_acf": "anomaly_acf",
    "monthly_spi": "spi",
    "monthly_spi_fit": "spi_fit",
    "annual_trend_interval": "trend_intervals",
    "annual_trend_sensitivity": "trend_sensitivity",
    "annual_indices_ts": "annual",
    "seasonal_indices_ts": "seasonal",
}


def location_name_token(name: str | None) -> str:
    """Return a portable lowercase filename token for a meaningful location name."""
    if not isinstance(name, str):
        return ""
    ascii_name = (
        unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode()
    )
    return re.sub(r"[^a-z0-9]+", "_", ascii_name.lower()).strip("_")


def _short_spatial_scope(spatial_scope: str, location_name: str | None) -> str:
    if spatial_scope in FIXED_SPATIAL_SCOPES:
        return "source" if spatial_scope == "source_ext" else "basin"
    for prefix, kinds in ID_SPATIAL_SCOPES.items():
        for kind in kinds:
            if spatial_scope.startswith(f"{prefix}_") and spatial_scope.endswith(
                f"_{kind}"
            ):
                identifier = spatial_scope[len(prefix) + 1 : -(len(kind) + 1)]
                if prefix in {"station", "location"} and location_name:
                    token = location_name_token(location_name)
                    if token:
                        return token
                stem = {"subbasin": "sub", "location": "loc", "station": "station"}[
                    prefix
                ]
                return f"{stem}{identifier}"
    raise ValueError(f"Cannot shorten spatial scope {spatial_scope!r}")


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
    *,
    location_name: str | None = None,
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
    subject = variable
    diagnostic = SHORT_CONTEXTS.get(plot_context, plot_context)
    if variable == "climate_indices":
        if plot_context == "wet_spell_exceedance":
            subject = "precip"
        elif plot_context in {"annual_indices_ts", "seasonal_indices_ts"}:
            subject = "indices"
        elif plot_context in {
            f"{period}_{scope}_ts"
            for period in ("annual", "seasonal")
            for scope in THEMATIC_SCOPES
        }:
            # Thematic sheets: annual_heavy_rainfall_ts -> heavy_rainfall_annual.
            period, scope = plot_context.removesuffix("_ts").split("_", 1)
            subject, diagnostic = scope, period
        else:
            subject = "indices"
    place = _short_spatial_scope(spatial_scope, location_name)
    return f"{dataset_scope}_{subject}_{diagnostic}_{place}.{extension}"


def annual_anomaly_filename(
    source: str, variable: str, page: int, page_count: int
) -> str:
    """Name one multi-panel anomaly figure, suffixing only split figures."""
    if not 1 <= page <= page_count:
        raise ValueError("figure page must be between one and the total page count")
    filename = figure_filename(source, variable, "annual_anomaly_map", "basin_ext")
    if page_count == 1:
        return filename
    width = max(2, len(str(page_count)))
    return f"{filename[:-4]}_page_{page:0{width}d}.png"


def map_spatial_scope(extent_policy: Optional[str]) -> str:
    """The ``spatial_scope`` a map figure takes, from how it is FRAMED.

    Derived rather than hardcoded, because the framing has already flipped once:
    ``climate_figures.MAP_EXTENT["source"]`` was ``raster`` on 2026-08-16 and
    ``basin`` on 2026-08-17. Both ``basin_ext`` and ``source_ext`` are in the
    vocabulary precisely so the name can follow the decision; a frozen token
    would mislabel the figure the next time it moves.
    """
    return "basin_ext" if extent_policy == "basin" else "source_ext"
