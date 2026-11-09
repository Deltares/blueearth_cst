"""Draw prepared annual series and monthly bands without scientific preparation.

Callers own ordering, units, statistical meaning, style context and export.
Arrays are read without mutation; masked values and NaNs retain their gaps.
"""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.text import Annotation
from numpy.typing import NDArray

from blueearth_cst.plotting import fundamentals as f
from blueearth_cst.plotting.preset import active

TREND_DASH = (0, (5, 2.5))


@dataclass(frozen=True)
class PreparedLine:
    """A labelled numeric or datetime series; scientific masks are caller-owned."""

    label: str
    x: object
    y: object
    color: str
    linestyle: object = "-"
    marker: str = ""


def series_panel(
    lines: Sequence[PreparedLine],
    *,
    xlabel: str,
    ylabel: str,
    ticks: tuple[Sequence, Sequence[str]] | None = None,
    xlim: tuple | None = None,
    width_mm: float = 127,
    height_mm: float = 65,
) -> Figure:
    """Draw one prepared comparison panel with a measured, nontruncated legend.

    Missing values retain gaps. Legend rows grow the canvas without shrinking
    the data area or fonts. The caller owns units, support, ordering and export.
    """
    if not lines:
        raise ValueError("supply at least one prepared line")
    for line in lines:
        _vector(line.y, "line values")
        if np.asanyarray(line.x).ndim != 1 or len(line.x) != len(line.y):
            raise ValueError("line coordinates must have matching vector lengths")
    # One entry per row also accommodates long retained-run labels without loss.
    fig, axes = f.new_fig(height_mm=height_mm + 4 * len(lines), width_mm=width_mm)
    ax = axes[0, 0]
    handles = []
    for line in lines:
        (artist,) = ax.plot(
            line.x,
            line.y,
            label=line.label,
            marker=line.marker,
            markevery=max(1, len(line.y) // 12),
            **f.time_series_style(line.label, line.color),
        )
        handles.append(artist)
    ax.set(xlabel=xlabel, ylabel=ylabel, ylim=(0, None))
    if xlim is not None:
        ax.set_xlim(*xlim)
    if np.issubdtype(np.asanyarray(lines[0].x).dtype, np.datetime64):
        from matplotlib.dates import AutoDateLocator, ConciseDateFormatter

        locator = AutoDateLocator(minticks=3, maxticks=5)
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(ConciseDateFormatter(locator))
        f.date_year_ticks(ax)
    if ticks is not None:
        ax.set_xticks(ticks[0], ticks[1])
        ax.set_xticks([], minor=True)
    if not any(np.isfinite(np.asanyarray(line.y)).any() for line in lines):
        ax.set_yticks([])
        ax.text(
            0.5,
            0.5,
            "No common support",
            transform=ax.transAxes,
            ha="center",
            va="center",
        )
    f.time_series_grid(ax)
    f.top_legend(fig, *[[handle] for handle in handles])
    return fig


@dataclass(frozen=True)
class AnnualLine:
    """One source's values, already aligned with the supplied display years."""

    label: str
    color: str
    marker: str
    values: NDArray[np.number]


@dataclass(frozen=True)
class MonthlyBand:
    """A supplied mean and envelope in the caller's twelve-month label order."""

    label: str
    color: str
    marker: str
    mean: NDArray[np.number]
    lower: NDArray[np.number]
    upper: NDArray[np.number]


@dataclass(frozen=True)
class TrendSegment:
    """Already evaluated fitted endpoints and their caller-formatted annotation."""

    color: str
    years: tuple[float, float]
    values: tuple[float, float]
    label: str


def _vector(value: object, name: str, length: int | None = None) -> None:
    array = np.asanyarray(value)
    if array.ndim != 1 or not np.issubdtype(array.dtype, np.number):
        raise ValueError(f"{name} must be a one-dimensional numeric array")
    if length is not None and len(array) != length:
        raise ValueError(f"{name} must contain {length} values")


def _annual_inputs(years: NDArray[np.number], lines: Sequence[AnnualLine]) -> None:
    _vector(years, "years")
    if not len(years):
        raise ValueError("years must not be empty")
    for line in lines:
        _vector(line.values, "annual values", len(years))


def _trend_inputs(trends: Sequence[TrendSegment]) -> None:
    for trend in trends:
        _vector(trend.years, "trend years", 2)
        _vector(trend.values, "trend values", 2)


def _identity_handles(sources: Sequence[AnnualLine | MonthlyBand]) -> list[Line2D]:
    return [
        Line2D(
            [],
            [],
            marker=source.marker,
            label=source.label,
            **f.time_series_style(source.label, source.color),
        )
        for source in sources
    ]


def draw_annual_lines(
    ax: Axes,
    years: NDArray[np.number],
    lines: Sequence[AnnualLine],
) -> list[Line2D]:
    """Draw source lines without changing ticks, legends or margins."""
    _annual_inputs(years, lines)
    artists = []
    for line in lines:
        artists.extend(
            ax.plot(
                years,
                line.values,
                marker=line.marker,
                label=line.label,
                **f.time_series_style(line.label, line.color),
            )
        )
    return artists


def draw_annual_trends(
    ax: Axes,
    trends: Sequence[TrendSegment],
    *,
    label_x: float,
) -> tuple[list[Line2D], list[Annotation]]:
    """Draw supplied trend segments and separate their end labels vertically."""
    _trend_inputs(trends)
    lines = []
    labels = []
    for trend in trends:
        lines.extend(
            ax.plot(
                trend.years,
                trend.values,
                color=trend.color,
                lw=active().lines["ref"],
                ls=TREND_DASH,
                alpha=0.6,
            )
        )
        labels.append([trend.values[-1], trend.label, trend.color])
    ax.margins(y=0.08)
    lo, hi = ax.get_ylim()
    gap = (hi - lo) * 0.07
    labels.sort(key=lambda row: row[0])
    for i in range(1, len(labels)):
        labels[i][0] = max(labels[i][0], labels[i - 1][0] + gap)
    annotations = [
        ax.annotate(
            text,
            (label_x, y),
            xytext=(-3, 8),
            textcoords="offset points",
            ha="right",
            va="bottom",
            fontsize=active().fonts["tick"],
            color=color,
            bbox=dict(active().label_box),
        )
        for y, text, color in labels
    ]
    return lines, annotations


def annual_series(
    years: NDArray[np.number],
    lines: Sequence[AnnualLine],
    *,
    ylabel: str,
    trends: Sequence[TrendSegment] = (),
) -> Figure:
    """Draw the standard annual chart from aligned values and retained fits."""
    _annual_inputs(years, lines)
    _trend_inputs(trends)
    fig, axes = f.new_fig(active().heights["single"])
    ax = axes[0, 0]
    draw_annual_lines(ax, years, lines)
    displayed = years
    if len(lines) == 1:
        finite = np.isfinite(lines[0].values)
        if finite.any():
            displayed = years[finite]
    f.year_ticks(ax, displayed)
    ax.margins(y=0.08)
    draw_annual_trends(ax, trends, label_x=years[-1] - 0.4)
    ax.set_ylabel(ylabel)
    handles, _ = ax.get_legend_handles_labels()
    f.top_legend(fig, handles if len(lines) > 1 else [])
    f.time_series_grid(ax)
    return fig


def monthly_cycle(
    bands: Sequence[MonthlyBand],
    *,
    month_labels: Sequence[str],
    ylabel: str,
    band_label: str,
    zero: bool,
) -> Figure:
    """Draw twelve supplied means/envelopes; labels define their display order."""
    if len(month_labels) != 12:
        raise ValueError("month_labels must contain twelve labels")
    for band in bands:
        for name in ("mean", "lower", "upper"):
            _vector(getattr(band, name), f"monthly {name}", 12)
    fig, axes = f.new_fig(active().heights["single"])
    ax = axes[0, 0]
    x = np.arange(12)
    top = 0.0
    fills = []
    lines = []
    for band in bands:
        fills.append(
            ax.fill_between(
                x,
                band.lower,
                band.upper,
                color=band.color,
                alpha=active().band_alpha,
                lw=0,
            )
        )
        lines.extend(
            ax.plot(
                x,
                band.mean,
                marker=band.marker,
                **f.time_series_style(band.label, band.color),
            )
        )
        finite = np.ma.asarray(band.upper).compressed()
        finite = finite[np.isfinite(finite)]
        if finite.size:
            top = max(top, float(finite.max()))
    ax.set_xticks(x, month_labels)
    ax.set_xlim(-0.5, 11.5)
    if zero:
        ax.set_ylim(0, max(top * 1.08, 1))
    else:
        ax.margins(y=0.08)
    ax.set_ylabel(ylabel)
    f.time_series_grid(ax)
    spread = Patch(
        facecolor=active().colors["muted"],
        alpha=active().band_alpha,
        edgecolor="none",
        label=band_label,
    )
    f.inside_legend(fig, ax, [*_identity_handles(bands), spread], fills, lines)
    return fig
