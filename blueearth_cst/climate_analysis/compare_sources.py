"""One table and one figure per variable, COMPARING the candidate climate sources.

Rule ``compare_climate_sources``'s script (``analyze_climate.smk`` 0.06). WF0
already draws the canonical figure set once per candidate source, each in its own
directory, and rule 0.04b pins them to a shared scale so two directories can be
read against each other. This module is the step that stops asking the reader to
do that: every source on ONE axis, plus the summary table that says what each of
them is.

The shape follows the CRIDA Zimbabwe report (``UNESCO_FinalReport_Draft_v4``,
Table 1 / Figures 2-3) — a gridded-dataset summary table, an annual series per
dataset, and a monthly climatology per dataset — narrowed to the datasets a
project actually extracted.

Three boundaries, all deliberate:

* **Figures only for a variable at least two sources carry** (owner ruling
  2026-08-17). A single-carrier variable drawn as a "comparison" is one line and
  a legend, which asserts a comparison that was not made. The TABLE still lists
  every candidate — that is what makes it a summary rather than a second copy of
  the figure's contents.
* **``pet`` is not compared.** It is not in a climate store; it is derived per
  source from that source's orography by the model-parity machinery
  (``plot_climate_source.source_grid_climate``). Comparing it would make this
  rule depend on an orography input per source, and it cannot arise today
  anyway: era5 is the only supported source that is not precipitation-only, so
  ``pet`` can never have two carriers. Extending to it is wiring the orography
  inputs, not a redesign — see :data:`COMPARABLE_VARS`.
* **No shared-scale input.** ``climate_levels.json`` (rule 0.04b) exists so
  SEPARATE per-source figures are comparable. Every figure here already carries
  every source on one axis, so the edge would buy nothing and re-fire this rule
  whenever the scale moved.

Values come from ``climate_figures``' own derivations
(:data:`~blueearth_cst.climate_analysis.climate_figures.VALUE_DERIVATIONS`), on
the same water-year anchor, so "annual precipitation" means here exactly what it
means in each source's own ``source_precip_annual.png``. Deriving it
independently is the defect that indirection exists to prevent — the same reason
``climate_levels`` reaches for it.
"""

# NO `from __future__ import annotations`: this module is imported by a
# `script:` module, whose Snakemake preamble displaces the first statement.
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.ticker import MaxNLocator

from blueearth_cst.climate_analysis.climate_figures import (
    CLIMATE_VARS,
    MONTH_LABELS,
    _series_axes,
    _style_series_axes,
    annual_series,
    monthly_spread,
    source_climate_vars,
)
from blueearth_cst.shared.plot_style import RASTER_DPI
from blueearth_cst.shared.snake_utils import (
    DEFAULT_WATER_YEAR_ANCHOR,
    PRECIP_ONLY_SOURCES,
    log_row,
    save_figure,
)

#: The variables a comparison can be drawn for: those a climate store CARRIES.
#: ``pet`` is absent by construction — it is derived per source rather than
#: extracted (see the module docstring). Ordered as :data:`CLIMATE_VARS` is, so
#: the figure set reads in the same order as every per-source directory.
COMPARABLE_VARS = ("precip", "temp")

#: The figure kinds this module renders. Deliberately its OWN tuple rather than
#: ``climate_figures.FIGURE_KINDS``: there is no ``map`` kind here — two grids at
#: different resolutions cannot share one raster panel — and a future entry added
#: there must not silently demand a renderer this module does not have.
COMPARISON_KINDS = ("annual", "monthly")

#: Filename prefix, mirroring ``<dataset>_<variable>_<kind>.png``. A figure
#: copied out of its directory still says what it is.
FIGURE_PREFIX = "comparison"

#: The summary table, written in both machine- and human-readable form. The CSV
#: is the one a GUI or a notebook reads; the Markdown is the one that goes in a
#: report next to the figures.
TABLE_STEM = "dataset_comparison"

#: Qualitative line colours, one per source, in declaration order — the primary
#: ``shared.clim_historical`` therefore always takes the first.
#:
#: Okabe-Ito, which is distinguishable under all three common dichromacies. It
#: lives HERE rather than in ``shared/plot_style.py`` because this is the only
#: figure family in the toolbox that encodes a CATEGORY by colour; everything
#: else colours by quantity through ``RASTER_STYLES``, and editing
#: ``plot_style.py`` would escalate a figure change to the full validation
#: ladder. Hoist it if a second caller appears — that is the Trigger on
#: ``dev/tasks/t2608171130-hoist-the-categorical-palette-out-of-compare-sources.md``.
SOURCE_COLORS = (
    "#0072B2",  # blue
    "#D55E00",  # vermillion
    "#009E73",  # bluish green
    "#CC79A7",  # reddish purple
    "#E69F00",  # orange
    "#56B4E9",  # sky blue
    "#000000",  # black
)


def comparison_variables(sources: Sequence[str]) -> tuple:
    """The variables at least two of ``sources`` can honestly be compared on.

    Carriage is judged by
    :func:`~blueearth_cst.climate_analysis.climate_figures.source_climate_vars`
    — the same function that decides which figures each source gets on its own —
    intersected with :data:`COMPARABLE_VARS`. A precipitation-only source
    therefore contributes to ``precip`` and to nothing else, which keeps its
    borrowed era5 temperature out of a comparison figure exactly as it keeps it
    out of that source's own figure set.

    Called at DAG-parse time to DECLARE the figures, so it must not read a file:
    the answer is a pure function of the configured source names.
    """
    carriers = {var: 0 for var in COMPARABLE_VARS}
    for source in sources:
        for var in source_climate_vars(source):
            if var in carriers:
                carriers[var] += 1
    return tuple(var for var in COMPARABLE_VARS if carriers[var] >= 2)


def comparison_figure_names(variables: Sequence[str]) -> list:
    """Every figure filename this module writes, in a stable order.

    The Snakefile declares its outputs from this, so it is the single source of
    the ``comparison_<variable>_<kind>.png`` scheme. Nothing else may build
    those names by hand.
    """
    unknown = sorted(set(variables) - set(COMPARABLE_VARS))
    if unknown:
        raise ValueError(
            f"comparison_figure_names: unknown variables {unknown}; expected a "
            f"subset of {list(COMPARABLE_VARS)}"
        )
    ordered = [var for var in COMPARABLE_VARS if var in set(variables)]
    return [
        f"{FIGURE_PREFIX}_{var}_{kind}.png"
        for var in ordered
        for kind in COMPARISON_KINDS
    ]


def comparison_table_names() -> list:
    """The two summary-table filenames, machine-readable first."""
    return [f"{TABLE_STEM}.csv", f"{TABLE_STEM}.md"]


def comparison_outputs(sources: Sequence[str]) -> list:
    """Every file rule 0.06 writes for ``sources``, table first."""
    return comparison_table_names() + comparison_figure_names(
        comparison_variables(sources)
    )


# --- the summary table --------------------------------------------------------
# Column keys are snake_case (the CSV's contract); DISPLAY_HEADERS renders them
# for the Markdown twin. Two spellings of one column set, never two column sets.

#: The table's columns, in order, with their Markdown headers.
#:
#: FIVE columns and no more (owner ruling 2026-08-17). The report's Table 1 is
#: read to answer "what are these datasets?", and the first draft answered a
#: dozen questions instead — DOI, cell counts, a period mean, the delivered
#: length in years. Each was defensible and together they made the table wider
#: than the page. What went, and why it is not lost:
#:
#: * ``years`` — derivable from the window, which is now one column;
#: * ``compared_variables`` — the figure set already states it: a variable with
#:   one carrier has no figure;
#: * ``mean_annual_precip`` — it is the annual figure's legend, and it belongs
#:   beside the series it summarises rather than in a table of DATASETS;
#: * ``doi`` / ``source_url`` — the reference names the paper; a reader who
#:   wants the identifier reads the catalog entry, which is where it lives.
DISPLAY_HEADERS = {
    "source": "Dataset",
    "temporal_resolution": "Time step",
    "time_window": "Time window",
    "spatial_resolution": "Grid size",
    "reference": "Reference",
}

#: Free-text column kept in the CSV but rendered BELOW the Markdown table rather
#: than inside it. It is a sentence, and a sentence in a grid cell is what makes
#: a compact table wide again — while dropping it would lose the one fact a
#: reader most needs about a precipitation-only source.
FOOTNOTE_COLUMN = "remarks"

#: Rendered for a provenance field neither the store nor the catalog supplies.
#: Absent keys render rather than raise: a summary table with one blank cell is
#: worth more than no table, and a locally staged catalog entry legitimately
#: carries no DOI.
MISSING = "—"


def _catalog_metadata(source: str, data_sources) -> dict:
    """The catalog entry's ``metadata:`` block for ``source``, or ``{}``.

    **The fallback is load-bearing, not belt-and-braces.** hydromt attaches an
    entry's metadata to what it returns, and ``extract_historical_climate``
    writes that through — but only on the branch that fetches a whole Dataset.
    The chirps branch fetches ONE variable and calls ``.to_dataset()`` on it,
    and the metadata does not survive: measured 2026-08-17 on a real extraction,
    that store's only attribute is ``region_bbox``. Reading the store alone
    would therefore blank the Reference, Version and DOI columns for exactly the
    precipitation-only sources a comparison is usually run to judge.

    Resolved through ``DataCatalog.to_dict()`` rather than by parsing the YAML,
    so an ``alias:`` entry resolves to the target it points at. Never raises: an
    unreachable catalog costs the provenance columns, not the table.
    """
    if not data_sources:
        return {}
    entry = _catalog_entries(data_sources).get(source, {})
    found = entry.get("metadata") or entry.get("meta") or {}
    return found if isinstance(found, dict) else {}


def _catalog_entries(data_sources) -> dict:
    """Every entry in the catalog library, parsed ONCE per process.

    Keyed on the library rather than on ``(library, source)``, which is what the
    first version did — and `to_dict()` parses the WHOLE library either way, so
    that cached the answer while re-paying the cost for every row. Measured on
    the rapid fixture: rule 0.06 took 18 s, of which 12 s was parsing the same
    catalog twice for two sources.
    """
    key = str(data_sources)
    if key in _CATALOG_CACHE:
        return _CATALOG_CACHE[key]
    entries: dict = {}
    try:
        import hydromt

        entries = hydromt.DataCatalog(data_libs=data_sources).to_dict()
    except Exception as exc:  # noqa: BLE001 -- provenance is not worth a failed rule
        log_row(
            f"could not read catalog metadata from {data_sources}: {exc}",
            module="compare",
            level="WARNING",
        )
    _CATALOG_CACHE[key] = entries
    return entries


#: One parse per catalog LIBRARY per process, keyed by the library path.
_CATALOG_CACHE: dict = {}


def _attr(ds: xr.Dataset, name: str, fallback: Optional[Mapping] = None) -> str:
    """A provenance field: the store's own attribute, else the catalog entry's.

    Store first, because that is the record of what was actually extracted; the
    catalog is what the entry says TODAY and may have been edited since.
    """
    value = ds.attrs.get(name)
    if value is None or (isinstance(value, str) and not value.strip()):
        value = (fallback or {}).get(name)
    if value is None or (isinstance(value, str) and not value.strip()):
        return MISSING
    return str(value).strip()


def _temporal_resolution(ds: xr.Dataset) -> str:
    """The store's time step, named rather than printed as a timedelta."""
    time = pd.DatetimeIndex(ds["time"].values)
    if time.size < 2:
        return MISSING
    step = pd.Series(time).diff().dropna().mode()
    if step.empty:
        return MISSING
    days = pd.Timedelta(step.iloc[0]).total_seconds() / 86400.0
    if np.isclose(days, 1.0):
        return "daily"
    if days < 1.0:
        hours = days * 24.0
        return f"{hours:g}-hourly"
    if 28.0 <= days <= 31.0:
        return "monthly"
    return f"{days:g}-daily"


#: How a gridded store may spell its y/x dimensions, in the order they are
#: tried. The store contract is ``latitude``/``longitude`` (CHIRPS's ``lat``/
#: ``lon`` are normalised at extraction), but the plotting derivations this
#: module reuses are grid-agnostic, so the table stays that way too rather than
#: being the one part that silently reports ``—`` on another spelling.
_SPATIAL_DIM_PAIRS = (("latitude", "longitude"), ("y", "x"), ("lat", "lon"))


def _spatial_dims(ds: xr.Dataset) -> Optional[tuple]:
    for pair in _SPATIAL_DIM_PAIRS:
        if all(dim in ds.dims for dim in pair):
            return pair
    return None


def _grid_step(ds: xr.Dataset, dim: str) -> Optional[float]:
    if dim not in ds.coords or ds[dim].size < 2:
        return None
    return float(np.median(np.abs(np.diff(ds[dim].values))))


def _spatial_resolution(ds: xr.Dataset) -> str:
    """Cell size in degrees, as one number when the cells are square."""
    dims = _spatial_dims(ds)
    if dims is None:
        return MISSING
    lat, lon = (_grid_step(ds, dim) for dim in dims)
    if lat is None or lon is None:
        return MISSING
    if np.isclose(lat, lon, rtol=1e-3):
        return f"{lat:g}°"
    return f"{lat:g}° × {lon:g}°"


def _grid_shape(ds: xr.Dataset) -> str:
    """Extraction footprint in cells — the BUFFERED bbox, not the basin."""
    dims = _spatial_dims(ds)
    if dims is None:
        return MISSING
    return " × ".join(str(ds.sizes[dim]) for dim in dims)


def _time_window(ds: xr.Dataset) -> str:
    """The extracted span as ONE cell — ``2000-01-01 → 2016-12-31``.

    One column rather than a from/to pair: the two were always read together,
    and splitting them spent two columns of a five-column table on one fact.
    """
    time = pd.DatetimeIndex(ds["time"].values)
    if not time.size:
        return MISSING
    return f"{time.min().date().isoformat()} → {time.max().date().isoformat()}"


def _remarks(source: str, ds: xr.Dataset, metadata: Optional[Mapping] = None) -> str:
    """The catalog's note, plus what a precipitation-only store really holds."""
    parts = []
    note = _attr(ds, "notes", metadata)
    if note != MISSING:
        parts.append(note)
    if source in PRECIP_ONLY_SOURCES:
        parts.append(
            "precipitation only; temperature, radiation and pressure in this "
            "store are era5's, regridded so the model can be forced"
        )
    return "; ".join(parts) if parts else MISSING


def summarize_sources(stores: Mapping, data_sources=None) -> pd.DataFrame:
    """One row per candidate source: what it is, and what was extracted from it.

    Parameters
    ----------
    stores : mapping
        ``{source_name: extract_historical.nc}``, in the order the rows are to
        read — declaration order, so the project's own ``clim_historical``
        leads.
    data_sources : str | Path | list, optional
        hydromt data catalog(s), used ONLY to fill provenance a store did not
        keep (see :func:`_catalog_metadata`). Omitted, the table reports what
        the extractions themselves carry.

    Notes
    -----
    **The window column is the EXTRACTED span, not the dataset's record.** A
    store holds what ``shared.historical_window`` asked for narrowed to what the
    source could deliver (``shared/climate_window.py``), so the record CHIRPS
    publishes from 1981 is not what this column reports. The dataset's own
    coverage is a property of the catalog entry, not of the extraction.

    It is also PER SOURCE, and therefore not the window the figures draw: those
    restrict every source to the span they all share (:func:`mutual_window`), so
    a difference between two lines is climate rather than calendar. The two
    numbers differing is the point — this column is where a reader sees that one
    source was extracted over a shorter record than another.
    """
    rows = []
    for source, path in stores.items():
        metadata = _catalog_metadata(source, data_sources)
        with xr.open_dataset(path) as ds:
            rows.append(
                {
                    "source": source,
                    "temporal_resolution": _temporal_resolution(ds),
                    "time_window": _time_window(ds),
                    "spatial_resolution": _spatial_resolution(ds),
                    "reference": _attr(ds, "paper_ref", metadata),
                    FOOTNOTE_COLUMN: _remarks(source, ds, metadata),
                }
            )
    return pd.DataFrame(rows)


def write_comparison_table(table: pd.DataFrame, out_dir: Union[str, Path]) -> list:
    """Write the summary table as CSV and as Markdown, and return both paths.

    The CSV carries every column including :data:`FOOTNOTE_COLUMN`; the Markdown
    renders the grid columns as the table and the free-text one as notes beneath
    it.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"{TABLE_STEM}.csv"
    md_path = out_dir / f"{TABLE_STEM}.md"

    table.to_csv(csv_path, index=False)

    columns = [name for name in DISPLAY_HEADERS if name in table.columns]
    rendered = table[columns].copy()
    rendered.columns = [DISPLAY_HEADERS[name] for name in columns]
    notes = ""
    if FOOTNOTE_COLUMN in table.columns:
        lines = [
            f"- **{row['source']}** — {row[FOOTNOTE_COLUMN]}"
            for _, row in table.iterrows()
            if row[FOOTNOTE_COLUMN] and row[FOOTNOTE_COLUMN] != MISSING
        ]
        notes = "\n" + "\n".join(lines) + "\n" if lines else ""
    md_path.write_text(
        "# Gridded climate datasets compared\n\n"
        + rendered.to_markdown(index=False)
        + "\n"
        + notes,
        encoding="utf-8",
    )
    log_row(
        f"Wrote the dataset comparison table ({len(table)} sources) -> "
        f"{csv_path.name}, {md_path.name}",
        module="compare",
    )
    return [csv_path, md_path]


# --- putting the sources on common ground -------------------------------------
# Two corrections, both applied before any value is derived. Without them the
# figures compare two different things and read as a difference between the
# DATASETS.


#: Coordinate rounding used to match a store's grid against ``basin_cells.csv``.
#: Six decimals, matching ``weathergen/generate_weather.R``'s ``mask_key`` — the
#: other consumer of the same file — so the two cannot disagree about which
#: cells the basin touches.
_CELL_KEY_DECIMALS = 6


def basin_cell_mask(ds: xr.Dataset, cells_csv: Union[str, Path]):
    """Boolean mask of the store cells the basin touches, or ``None``.

    **The domains differ before this, and not slightly.** A store is a bbox read
    plus ``BUFFER_CELLS``, and the buffer is counted in CELLS — so ERA5's
    0.25-degree store reaches physically further past the basin than CHIRPS's
    0.05-degree one, and a plain domain mean over each extraction averages two
    different areas. Measured on the rapid fixture: 4x5 ERA5 cells against 7x8
    CHIRPS cells, the same basin under both. A difference between the two lines
    would then partly be the neighbouring climate each grid happened to include.

    The mask comes from the store's OWN ``basin_cells.csv``, the file rule 0.04
    already writes and weathergenr already averages over
    (``extract_historical_climate.write_basin_cell_mask``). Reusing it rather
    than re-deriving from the basin polygon means the comparison, the weather
    generator and the stress test all agree on what "the basin" is — and it
    inherits that function's INTERSECTS rule, which is what keeps a basin
    smaller than one ERA5 cell from selecting no cells at all.

    Returns ``None`` when the file is absent, when the store does not spell its
    grid ``latitude``/``longitude``, or when nothing matches — the figure is
    then drawn over the full extraction and says so, which beats no figure.
    """
    if cells_csv is None or not Path(cells_csv).is_file():
        return None
    if not {"latitude", "longitude"} <= set(ds.dims):
        log_row(
            "store does not spell its grid latitude/longitude; the basin mask "
            "cannot be matched and the full extraction is used",
            module="compare",
            level="WARNING",
        )
        return None
    frame = pd.read_csv(cells_csv)
    keys = {
        (
            round(float(lat), _CELL_KEY_DECIMALS),
            round(float(lon), _CELL_KEY_DECIMALS),
        )
        for lat, lon in zip(frame["latitude"], frame["longitude"])
    }
    lats = [round(float(v), _CELL_KEY_DECIMALS) for v in ds["latitude"].values]
    lons = [round(float(v), _CELL_KEY_DECIMALS) for v in ds["longitude"].values]
    values = np.array([[(la, lo) in keys for lo in lons] for la in lats])
    if not values.any():
        log_row(
            f"{Path(cells_csv).name} matched no cell in the store; the full "
            "extraction is used instead",
            module="compare",
            level="WARNING",
        )
        return None
    return xr.DataArray(
        values,
        dims=("latitude", "longitude"),
        coords={"latitude": ds["latitude"], "longitude": ds["longitude"]},
    )


def mutual_window(datasets: Sequence) -> Optional[tuple]:
    """The time span every dataset covers, or ``None`` if they do not overlap.

    Each candidate is extracted over the widest span it holds inside
    ``shared.historical_window``, so two sources routinely deliver different
    records (``shared/climate_window.py``). Averaging each over its own record
    and drawing the results as a comparison attributes a calendar difference to
    the datasets — and on a monthly climatology it is invisible, because the
    x-axis has no year on it.
    """
    starts, ends = [], []
    for ds in datasets:
        time = pd.DatetimeIndex(ds["time"].values)
        if not time.size:
            return None
        starts.append(time.min())
        ends.append(time.max())
    if not starts:
        return None
    lower, upper = max(starts), min(ends)
    return (lower, upper) if lower <= upper else None


def _on_common_ground(ds: xr.Dataset, mask, window) -> xr.Dataset:
    """One store restricted to the basin cells and the shared period."""
    if mask is not None:
        ds = ds.where(mask)
    if window is not None:
        ds = ds.sel(time=slice(*window))
    return ds


def _window_note(window) -> str:
    if window is None:
        return "each source over its own extracted period"
    lower, upper = window
    return f"common period {lower.date().isoformat()} to {upper.date().isoformat()}"


def comparison_caveat(window, masked: bool) -> str:
    """The footnote every comparison figure carries, stating what was compared.

    Rendered on the figure rather than left to a caption, for the same reason
    the per-source set carries its own: the file outlives its directory. It
    names the period because a reader cannot otherwise tell whether two lines
    differ by climate or by calendar.
    """
    domain = (
        "Basin cells only — each grid's cells touching the basin"
        if masked
        else "Full extraction grids, which differ in footprint"
    )
    return (
        f"{domain}; {_window_note(window)}. The grids differ in resolution, "
        "which damps extremes on the coarser one."
    )


# --- the comparison figures ---------------------------------------------------


def _axis_unit(spec: Mapping, per: str) -> str:
    """``mm y^-1`` for a flux, the bare unit for a state."""
    return f"{spec['unit']} {per}$^{{-1}}$" if spec["how"] == "sum" else spec["unit"]


def _source_color(index: int) -> str:
    return SOURCE_COLORS[index % len(SOURCE_COLORS)]


def _render_annual_comparison(datasets: Mapping, var: str, anchor: str, caveat: str):
    """Every source's annual series on one axis, with its period mean in the key.

    The report's Figure 2. The mean goes in the LEGEND rather than as a line per
    source, because with more than two datasets a horizontal line each is the
    part of the figure that stops being readable first — and the mean is the
    number the surrounding text quotes.
    """
    spec = CLIMATE_VARS[var]
    fig, ax = _series_axes(caveat)
    for index, (source, ds) in enumerate(datasets.items()):
        series = annual_series(ds[var], spec, anchor)
        years = series["time"].dt.year.values.astype(float)
        values = series.values.astype(float)
        finite = values[np.isfinite(values)]
        mean = float(np.mean(finite)) if finite.size else float("nan")
        ax.plot(
            years,
            values,
            color=_source_color(index),
            marker="o",
            lw=1.1,
            ms=3.2,
            zorder=3,
            label=f"{source} (mean {mean:,.0f})",
        )
    ax.set_xlabel("Year")
    ax.set_ylabel(f"{spec['label'].capitalize()} ({_axis_unit(spec, 'y')})")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.legend(frameon=False, fontsize=6.5, loc="best")
    _style_series_axes(ax)
    return fig


def _render_monthly_comparison(datasets: Mapping, var: str, anchor: str, caveat: str):
    """Every source's monthly climatology on one axis.

    The report's Figure 3. A LINE per source, not the box-whiskers each source's
    own ``*_monthly.png`` draws: the boxes describe one dataset's interannual
    spread, and overplotting several sets of them hides the between-source
    difference this figure exists to show. The mean of each month's
    distribution is the same quantity the diamonds mark there.
    """
    spec = CLIMATE_VARS[var]
    months = np.arange(1, 13)
    fig, ax = _series_axes(caveat, aspect=0.40)
    for index, (source, ds) in enumerate(datasets.items()):
        spread = monthly_spread(ds[var], spec)
        climatology = [
            float(np.mean(values)) if values.size else np.nan for values in spread
        ]
        ax.plot(
            months,
            climatology,
            color=_source_color(index),
            marker="o",
            lw=1.3,
            ms=3.2,
            zorder=3,
            label=source,
        )
    ax.set_xticks(months)
    ax.set_xticklabels(MONTH_LABELS)
    ax.set_xlim(0.4, 12.6)
    ax.set_xlabel("Month")
    ax.set_ylabel(f"{spec['label'].capitalize()} ({_axis_unit(spec, 'month')})")
    ax.legend(frameon=False, fontsize=6.5, loc="best")
    _style_series_axes(ax)
    return fig


_RENDERERS = {
    "annual": _render_annual_comparison,
    "monthly": _render_monthly_comparison,
}


def plot_comparison_figures(
    stores: Mapping,
    out_dir: Union[str, Path],
    variables: Sequence[str],
    anchor: str = DEFAULT_WATER_YEAR_ANCHOR,
    basin_cells: Optional[Mapping] = None,
) -> list:
    """Write one figure per ``(variable, kind)``, every source on one axis.

    Every source is put on common ground first — masked to the basin cells its
    own grid contributes (:func:`basin_cell_mask`) and clipped to the period all
    of them cover (:func:`mutual_window`) — so a difference between two lines is
    the datasets rather than their footprints or their calendars.

    ``variables`` MUST be what :func:`comparison_variables` returned for the
    same source set, since that is what the rule declared its outputs from:
    narrow only the declaration and the extra files are undeclared, narrow only
    the drawing and the job ends in ``MissingOutputException``.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    basin_cells = basin_cells or {}
    opened = {name: xr.open_dataset(path) for name, path in stores.items()}
    written = []
    try:
        window = mutual_window(list(opened.values()))
        if window is None and len(opened) > 1:
            log_row(
                "the sources' extracted periods do not overlap; each is drawn "
                "over its own record and the figures say so",
                module="compare",
                level="WARNING",
            )
        masks = {
            name: basin_cell_mask(ds, basin_cells.get(name))
            for name, ds in opened.items()
        }
        prepared = {
            name: _on_common_ground(ds, masks[name], window)
            for name, ds in opened.items()
        }
        masked = bool(masks) and all(mask is not None for mask in masks.values())
        caveat = comparison_caveat(window, masked)
        log_row(
            f"Common ground: {_window_note(window)}; "
            + (
                "basin cells only ("
                + ", ".join(
                    f"{name} {int(mask.values.sum())}"
                    for name, mask in masks.items()
                    if mask is not None
                )
                + " cells)"
                if masked
                else "full extraction grids (no basin mask available)"
            ),
            module="compare",
        )
        for var in [v for v in COMPARABLE_VARS if v in set(variables)]:
            # A source that does not carry the variable is left OUT of the
            # figure rather than drawn from a borrowed field -- the same ruling
            # `source_climate_vars` applies to the per-source set.
            carriers = {
                name: ds
                for name, ds in prepared.items()
                if var in source_climate_vars(name) and var in ds
            }
            for kind in COMPARISON_KINDS:
                fig = _RENDERERS[kind](carriers, var, anchor, caveat)
                out_path = out_dir / f"{FIGURE_PREFIX}_{var}_{kind}.png"
                save_figure(out_path, dpi=RASTER_DPI)
                plt.close(fig)
                written.append(out_path)
            log_row(
                f"Compared {var} across {len(carriers)} source(s): "
                f"{', '.join(carriers)}",
                module="compare",
            )
    finally:
        for ds in opened.values():
            ds.close()
    return written


def compare_climate_sources(
    stores: Mapping,
    out_dir: Union[str, Path],
    anchor: str = DEFAULT_WATER_YEAR_ANCHOR,
    data_sources=None,
    basin_cells: Optional[Mapping] = None,
) -> list:
    """The rule's whole job: the summary table, then the comparison figures.

    ``data_sources`` is the hydromt catalog, read only to fill provenance the
    stores did not keep; ``basin_cells`` is ``{source: basin_cells.csv}``, the
    domain the figures average over. Returns every path written, table first —
    the order :func:`comparison_outputs` declares them in.
    """
    variables = comparison_variables(list(stores))
    log_row(
        f"Comparing {len(stores)} climate sources ({', '.join(stores)}); "
        f"variables with more than one source: "
        f"{', '.join(variables) if variables else 'none'}",
        module="compare",
    )
    os.makedirs(out_dir, exist_ok=True)
    written = write_comparison_table(summarize_sources(stores, data_sources), out_dir)
    written += plot_comparison_figures(stores, out_dir, variables, anchor, basin_cells)
    return written


if __name__ == "__main__":
    if "snakemake" in globals():
        sm = globals()["snakemake"]
        from blueearth_cst.shared.snake_utils import tee_to_log, water_year_end_anchor

        with tee_to_log(sm.log[0]):
            compare_climate_sources(
                dict(zip(sm.params.sources, sm.input.climate_ncs)),
                sm.params.out_dir,
                anchor=water_year_end_anchor(sm.params.water_year_start),
                # In `params`, not `input`, for the same reason rule 0.05 keeps
                # its catalog there: the freshness boundary is rule 0.04's
                # catalog edge (ext2-01), and duplicating it here would rebuild
                # the table on every catalog touch with no extraction change.
                data_sources=sm.params.data_sources,
                # A real `input`, unlike the catalog: it is rule 0.04's own
                # output and changes with the extraction it describes.
                basin_cells=dict(zip(sm.params.sources, sm.input.basin_cells)),
            )
