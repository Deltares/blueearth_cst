"""Selected reviewed diagnostic artists, fed by retained toolbox tables.

No lab input, page, runner, or output-path machinery is used here.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import date, timedelta

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import BoundaryNorm, ListedColormap, Normalize, to_hex, to_rgb
from matplotlib.figure import Figure
from matplotlib.legend_handler import HandlerTuple
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FixedLocator, MaxNLocator, StrMethodFormatter

from blueearth_cst.climate_analysis.diagnostics import SPI_SCALES, month_order
from blueearth_cst.plotting import prepared_charts as pc
from blueearth_cst.plotting.fundamentals import MM as MM
from blueearth_cst.plotting.fundamentals import MONTH_ABBR as MONTH_ABBR
from blueearth_cst.plotting.fundamentals import add_caption as add_caption
from blueearth_cst.plotting.fundamentals import blank as blank
from blueearth_cst.plotting.fundamentals import colourbar as colourbar
from blueearth_cst.plotting.fundamentals import five_year_ticks as five_year_ticks
from blueearth_cst.plotting.fundamentals import heatmap_frame as heatmap_frame
from blueearth_cst.plotting.fundamentals import inside_legend as inside_legend
from blueearth_cst.plotting.fundamentals import new_fig as new_fig
from blueearth_cst.plotting.fundamentals import top_legend as top_legend
from blueearth_cst.plotting.fundamentals import year_ticks as year_ticks
from blueearth_cst.plotting.fundamentals import ygrid as ygrid
from blueearth_cst.plotting.prepared_charts import TREND_DASH as TREND_DASH
from blueearth_cst.shared.plot_colors import (
    DATASET_COLORS,
    DEFAULT_COLOR_CYCLE,
    NEUTRAL_COLORS,
    PRECIP_ANOMALY_COLORMAP,
)
from blueearth_cst.shared.plot_preset import AXIS as AXIS
from blueearth_cst.shared.plot_preset import BAND_ALPHA as BAND_ALPHA
from blueearth_cst.shared.plot_preset import BOUNDARY as BOUNDARY
from blueearth_cst.shared.plot_preset import FS_LABEL as FS_LABEL
from blueearth_cst.shared.plot_preset import FS_NOTE as FS_NOTE
from blueearth_cst.shared.plot_preset import FS_TICK as FS_TICK
from blueearth_cst.shared.plot_preset import GRID as GRID
from blueearth_cst.shared.plot_preset import H_GRID4 as H_GRID4
from blueearth_cst.shared.plot_preset import H_ROW3 as H_ROW3
from blueearth_cst.shared.plot_preset import H_SIDE2 as H_SIDE2
from blueearth_cst.shared.plot_preset import H_SINGLE as H_SINGLE
from blueearth_cst.shared.plot_preset import H_STACK2 as H_STACK2
from blueearth_cst.shared.plot_preset import INK as INK
from blueearth_cst.shared.plot_preset import LABEL_BOX as LABEL_BOX
from blueearth_cst.shared.plot_preset import LW_BOUNDARY as LW_BOUNDARY
from blueearth_cst.shared.plot_preset import LW_EDGE as LW_EDGE
from blueearth_cst.shared.plot_preset import LW_IQR as LW_IQR
from blueearth_cst.shared.plot_preset import LW_LINE as LW_LINE
from blueearth_cst.shared.plot_preset import LW_MEAN as LW_MEAN
from blueearth_cst.shared.plot_preset import LW_RANGE as LW_RANGE
from blueearth_cst.shared.plot_preset import LW_REF as LW_REF
from blueearth_cst.shared.plot_preset import LW_STEM as LW_STEM
from blueearth_cst.shared.plot_preset import MS_LINE as MS_LINE
from blueearth_cst.shared.plot_preset import MS_POINT as MS_POINT
from blueearth_cst.shared.plot_preset import MUTED as MUTED
from blueearth_cst.shared.plot_preset import OUTER_PAD_X_MM as OUTER_PAD_X_MM
from blueearth_cst.shared.plot_preset import OUTER_PAD_Y_MM as OUTER_PAD_Y_MM
from blueearth_cst.shared.plot_preset import S_SCATTER as S_SCATTER
from blueearth_cst.shared.plot_preset import STYLE as STYLE
from blueearth_cst.shared.plot_preset import W_COL1 as W_COL1
from blueearth_cst.shared.plot_preset import W_FULL as W_FULL
from blueearth_cst.shared.plot_preset import W_MID as W_MID
from blueearth_cst.shared.plot_preset import W_PANEL as W_PANEL
from blueearth_cst.shared.plot_setup import STYLE_SETTINGS

MISSING = NEUTRAL_COLORS["missing"]


QA = NEUTRAL_COLORS["qa"]


SOURCE_STYLE = {
    name: (color, STYLE_SETTINGS["charts"]["dataset_marker"])
    for name, color in DATASET_COLORS.items()
}


DODGE = 0.2


SPI_BOUNDS = [-3, -2, -1, 1, 2, 3]


UNITS = {
    "mm_month": "mm month⁻¹",
    "mm_year": "mm yr⁻¹",
    "mm_decade": "mm decade⁻¹",
    "days_year": "days yr⁻¹",
    "days_decade": "days decade⁻¹",
    "mm_day": "mm d⁻¹",
    "degc": "°C",
}


def source_styles(keys: Sequence[str]) -> dict[str, tuple[str, str]]:
    """Colour and marker per source id: registry first, then free slots in config order."""
    reserved = {SOURCE_STYLE[k.lower()][0] for k in keys if k.lower() in SOURCE_STYLE}
    available = [color for color in DEFAULT_COLOR_CYCLE if color not in reserved]
    available = available or list(DEFAULT_COLOR_CYCLE)
    free_index = 0
    out = {}
    for k in keys:
        if k.lower() in SOURCE_STYLE:
            out[k] = SOURCE_STYLE[k.lower()]
        else:
            markers = STYLE_SETTINGS["charts"]["fallback_markers"]
            out[k] = (
                available[free_index % len(available)],
                markers[(free_index // len(available)) % len(markers)],
            )
            free_index += 1
    return out


def tint(color: str) -> str:
    """Partial / incomplete: the identity colour 55 % toward white."""
    return to_hex(0.45 * np.array(to_rgb(color)) + 0.55)


def anomaly_bounds(values: np.ndarray) -> list[float]:
    """Symmetric continuous scale spanning all finite pooled monthly anomalies."""
    a = np.abs(values[np.isfinite(values)])
    limit = float(a.max()) if a.size and a.max() > 0 else 1.0
    return [-limit, limit]


def fmt_signed(x: float, digits: int = 0) -> str:
    return f"{x:+.{digits}f}".replace("-", "−")


def fmt_p(p: float) -> str:
    return "$p$ < 0.01" if p < 0.01 else f"$p$ = {p:.2f}"


def plural(n: int, word: str) -> str:
    return f"{n} {word}" if n == 1 else f"{n} {word}s"


@dataclass
class View:
    sources: list[str]
    labels: dict[str, str]
    styles: dict[str, tuple[str, str]]  # source -> (colour, marker)
    results: dict[str, dict]  # source -> restored plot-ready tables
    years: np.ndarray  # reporting years of the display period
    m0: int
    reference: tuple[int, int]
    wet_mm: float
    agreement: pd.DataFrame
    common_years: np.ndarray
    support: str = "basin mean"
    status: str = "prototype-tested, not scientifically qualified"
    anom_bounds: list[float] | None = None  # fixed per case
    max_missing_month: int = 0
    max_missing_year: int = 0

    def color(self, s: str) -> str:
        return self.styles[s][0]

    def marker(self, s: str) -> str:
        return self.styles[s][1]


def dodge(j: int, n: int) -> float:
    return (j - (n - 1) / 2) * DODGE


def month_axis(ax: plt.Axes, m0: int) -> None:
    ax.set_xticks(np.arange(12), [MONTH_ABBR[m - 1] for m in month_order(m0)])
    ax.set_xlim(-0.5, 11.5)


def heatmap_months(ax: plt.Axes, m0: int) -> None:
    ax.set_ylim(12.5, 0.5)
    ax.set_yticks(range(1, 13), [MONTH_ABBR[m - 1] for m in month_order(m0)])


def id_handles(
    v: View, sources: Sequence[str] | None = None, line: bool = True, label=None
) -> list:
    sources = v.sources if sources is None else sources
    return [
        Line2D(
            [],
            [],
            color=v.color(s),
            marker=v.marker(s),
            ms=MS_LINE + 1,
            lw=LW_LINE if line else 0,
            label=label(s) if label else v.labels[s],
        )
        for s in sources
    ]


def identity_row(v: View, sources: Sequence[str] | None = None, **kw) -> list:
    """Identity entries, or none for a single source named by the page caption."""
    sources = v.sources if sources is None else sources
    return id_handles(v, sources, **kw) if len(sources) > 1 else []


MISSING_KEY = Patch(facecolor=MISSING, edgecolor="none", label="Missing or unavailable")


def notice(text: str) -> str:
    """Plain availability notice; the exporter draws and records it."""
    return text


def coverage(v: View) -> Figure:
    n = len(v.sources)
    fig, axs = new_fig(24 + 8 * n, width_mm=W_FULL if n > 1 else W_MID)
    ax = axs[0, 0]
    ax.set_facecolor(MISSING)
    edges = np.arange(len(v.years) * 12 + 1) / 12 + v.years[0]
    for i, s in enumerate(v.sources):
        cls = (
            v.results[s]["coverage"]
            .reindex(index=month_order(v.m0), columns=v.years)
            .to_numpy(dtype=float, copy=True)
        )
        cls[cls == 0] = np.nan
        pale = to_hex(0.3 * np.array(to_rgb(v.color(s))) + 0.7)
        gap_color = to_hex(0.7 * np.array(to_rgb(v.color(s))))
        ax.pcolormesh(
            edges,
            [i - 0.5, i + 0.5],
            cls.T.reshape(1, -1),
            cmap=ListedColormap([gap_color, pale]),
            vmin=0.5,
            vmax=2.5,
            edgecolors=(*to_rgb(GRID), 0.3),
            linewidth=0.15,
            rasterized=True,
        )
    ax.set_yticks(range(n), [v.labels[s] for s in v.sources])
    ax.set_ylim(n - 0.5, -0.5)
    ax.set_xlim(edges[0], edges[-1])
    year_ticks(ax, v.years, continuous=True)
    ax.tick_params(axis="y", length=3)
    full = tuple(
        Patch(facecolor=to_hex(0.3 * np.array(to_rgb(v.color(s))) + 0.7))
        for s in v.sources
    )
    part = tuple(
        Patch(facecolor=to_hex(0.7 * np.array(to_rgb(v.color(s))))) for s in v.sources
    )
    tup = HandlerTuple(ndivide=None, pad=0)
    legend = fig.legend(
        handles=[full, part, MISSING_KEY],
        labels=["Complete", "Usable with gaps", "Missing or unusable"],
        loc="outside right upper",
        ncols=1,
        handler_map={tuple: tup},
        frameon=True,
        facecolor="white",
        edgecolor="none",
        framealpha=0.88,
    )
    fig.canvas.draw()
    legend_width = legend.get_window_extent().width / fig.bbox.width
    legend.set_loc("upper right")
    legend.borderaxespad = 0
    fig.get_layout_engine().set(rect=(0, 0, 1 - legend_width - 0.02, 1))
    fig.canvas.draw()
    legend.set_bbox_to_anchor((1, ax.get_position().y1), transform=fig.transFigure)
    return fig


def _climatology(v: View, key: str, ylabel: str, zero: bool) -> Figure:
    have = [s for s in v.sources if v.results[s][key] is not None]
    return pc.monthly_cycle(
        [
            pc.MonthlyBand(
                v.labels[s],
                v.color(s),
                v.marker(s),
                v.results[s][key]["mean"].to_numpy(),
                v.results[s][key]["p10"].to_numpy(),
                v.results[s][key]["p90"].to_numpy(),
            )
            for s in have
        ],
        month_labels=[MONTH_ABBR[m - 1] for m in month_order(v.m0)],
        ylabel=ylabel,
        band_label="10–90th percentile",
        zero=zero,
    )


def seasonal(v: View) -> Figure:
    return _climatology(v, "clim", UNITS["mm_month"], zero=True)


def temperature(v: View) -> Figure | str:
    if not any(v.results[s]["tclim"] is not None for s in v.sources):
        return notice("No temperature variable is declared for the selected source(s).")
    return _climatology(v, "tclim", UNITS["degc"], zero=False)


def timing(v: View) -> Figure:
    fig, axs = new_fig(H_SINGLE)
    ax = axs[0, 0]
    fr = (0.25, 0.5, 0.75)
    n = len(v.sources)
    for j, s in enumerate(v.sources):
        t = v.results[s]["timing"]
        for k, f in enumerate(fr):
            d = t[f].dropna()
            if d.empty:
                continue
            y = k + dodge(j, n)
            lo, med, hi = np.percentile(d, [25, 50, 75])
            mean = float(d.mean())
            half_height = 0.06
            ax.plot([d.min(), lo], [y, y], color=v.color(s), lw=0.6)
            ax.plot([hi, d.max()], [y, y], color=v.color(s), lw=0.6)
            ax.add_patch(
                plt.Rectangle(
                    (lo, y - half_height),
                    hi - lo,
                    2 * half_height,
                    facecolor="none",
                    edgecolor=v.color(s),
                    lw=0.6,
                )
            )
            ax.plot(
                [med, med],
                [y - half_height, y + half_height],
                color=v.color(s),
                lw=0.6,
            )
            label_offset = 5 if j == 0 else -5
            ax.annotate(
                f"{mean:.0f}",
                (mean, y),
                xytext=(0, label_offset),
                textcoords="offset points",
                ha="center",
                va="bottom" if j == 0 else "top",
                fontsize=FS_NOTE,
                color=v.color(s),
                bbox=LABEL_BOX,
            )
    start = date(2001, v.m0, 1)
    starts = [
        (date(2001 + (v.m0 + k - 1) // 12, (v.m0 + k - 1) % 12 + 1, 1) - start).days + 1
        for k in range(12)
    ]
    ax.set_xticks(
        starts, [(start + timedelta(days=d - 1)).strftime("%b") for d in starts]
    )
    ax.set_xlim(1, 366)
    top = ax.secondary_xaxis("top")
    top.set_xticks([1, 91, 182, 274, 366])
    top.tick_params(labelsize=FS_TICK, labelcolor=MUTED, color=AXIS)
    top.spines["top"].set_color(AXIS)
    ax.set_yticks(range(3), ["25", "50", "75"])
    ax.set_ylim(2.6, -0.6)
    ax.set_xlabel("Month")
    ax.set_ylabel("Cumulative (%)")
    ax.grid(axis="x")
    ax.tick_params(axis="y", length=3)
    identities = (
        [
            Line2D(
                [],
                [],
                color=v.color(s),
                marker="o",
                ms=MS_LINE + 1,
                lw=LW_LINE,
                label=v.labels[s],
            )
            for s in v.sources
        ]
        if n > 1
        else []
    )
    inside_legend(fig, ax, identities, lines=ax.lines)
    return fig


def _stack(v: View):
    n = len(v.sources)
    height = H_SINGLE if n == 1 else H_STACK2 if n == 2 else H_STACK2 + 48 * (n - 2)
    fig, axs = new_fig(height, n, 1, width_mm=W_FULL if n > 1 else W_MID, sharex=True)
    return fig, axs, height


def anomaly(v: View) -> Figure:
    """Absolute monthly departures on a continuous, zero-centred shared scale."""
    bounds = v.anom_bounds or [-1.0, 1.0]
    limit = max(abs(bounds[0]), abs(bounds[-1]))
    norm = Normalize(-limit, limit)
    cmap = plt.get_cmap(PRECIP_ANOMALY_COLORMAP).with_extremes(bad=MISSING)
    fig, axs, stack = _stack(v)
    mesh = None
    for i, (ax, s) in enumerate(zip(axs[:, 0], v.sources)):
        ax.set_facecolor(MISSING)
        heatmap_frame(ax)
        grid = v.results[s]["anomaly"].reindex(columns=v.years)
        mesh = ax.pcolormesh(
            v.years,
            np.arange(1, 13),
            grid,
            cmap=cmap,
            norm=norm,
            edgecolors="none",
            rasterized=True,
        )
        heatmap_months(ax, v.m0)
        if len(v.sources) > 1:
            ax.set_title(f"{chr(97 + i)}) {v.labels[s]}")
    axs[-1, 0].set_xlim(v.years[0] - 0.5, v.years[-1] + 0.5)
    five_year_ticks(axs[-1, 0], v.years)
    ticks = MaxNLocator(nbins=4).tick_values(-limit, limit)
    ticks = [tick for tick in ticks if norm.vmin <= tick <= norm.vmax]
    label = "Precipitation\nanomaly (mm)"
    colourbar(fig, mesh, axs[:, 0].tolist(), label, ticks, stack)
    fig.get_layout_engine().set(rect=(0, 0, 0.94, 1))
    return fig


def spi(v: View) -> Figure:
    cmap = plt.get_cmap("RdBu", len(SPI_BOUNDS) + 1)
    norm = BoundaryNorm(SPI_BOUNDS, cmap.N, extend="both")
    fig, axs, stack = _stack(v)
    stack *= 0.56
    fig.set_size_inches(fig.get_figwidth(), stack * MM)
    mesh = None
    for i, (ax, s) in enumerate(zip(axs[:, 0], v.sources)):
        ax.set_facecolor(MISSING)
        heatmap_frame(ax)
        for row, k in enumerate(SPI_SCALES):
            ser = v.results[s]["spi"][k]
            x = (
                ser.index.year
                - (ser.index.month < v.m0).astype(int)
                + ((ser.index.month - v.m0) % 12 + 0.5) / 12
            )
            edges = np.append(x - 1 / 24, x[-1] + 1 / 24) if len(x) else x
            mesh = ax.pcolormesh(
                edges,
                [row - 0.5, row + 0.5],
                ser.to_numpy()[None, :],
                cmap=cmap,
                norm=norm,
                rasterized=True,
            )
        ax.hlines(
            [0.5, 1.5], v.years[0], v.years[-1] + 1, colors=GRID, linewidth=LW_EDGE
        )
        ax.set_ylim(2.5, -0.5)
        ax.set_yticks(range(3), [f"SPI-{k}" for k in SPI_SCALES])
        if len(v.sources) > 1:
            ax.set_title(f"{chr(97 + i)}) {v.labels[s]}")
    axs[-1, 0].set_xlim(v.years[0], v.years[-1] + 1)
    five_year_ticks(axs[-1, 0], v.years, rotation=0)
    colourbar(
        fig,
        mesh,
        axs[:, 0].tolist(),
        "SPI",
        SPI_BOUNDS,
        stack,
        shrink=0.8,
    )
    return fig


def drought_events(v: View) -> Figure:
    """Duration-severity scatter per SPI scale; tinted = censored by a gap or the record edge."""
    scales = (3, 12)
    fig, axs = new_fig(H_SIDE2, 1, 2, width_mm=W_FULL)
    for i, (ax, k) in enumerate(zip(axs[0], scales)):
        drawn = False
        for s in v.sources:
            ev = v.results[s]["drought"][k]
            if ev.empty:
                continue
            drawn = True
            done, cens = (
                ev[~ev["censored"].astype(bool)],
                ev[ev["censored"].astype(bool)],
            )
            edge = to_hex(0.65 * np.array(to_rgb(v.color(s))))
            ax.scatter(
                done["duration"],
                done["severity"],
                marker=v.marker(s),
                s=S_SCATTER * 2.25,
                alpha=0.7,
                facecolor=v.color(s),
                edgecolor=edge,
                linewidth=0.7,
                zorder=3,
            )
            ax.scatter(
                cens["duration"],
                cens["severity"],
                marker=v.marker(s),
                s=S_SCATTER * 2.25,
                alpha=0.7,
                facecolor=tint(v.color(s)),
                edgecolor=edge,
                linewidth=0.9,
                hatch="////////",
                hatch_linewidth=0.5,
                zorder=3,
            )
        if not drawn:
            ax.text(
                0.5,
                0.5,
                "no events",
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize=FS_TICK,
                color=MUTED,
            )
        ax.set_title(f"{chr(97 + i)}) SPI-{k}")
        ax.set_xlabel("Duration (months)")
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0)
        ax.grid()
    axs[0, 0].set_ylabel("Severity (dimensionless)")
    event_handles = [
        Line2D(
            [],
            [],
            color=MUTED,
            linewidth=1.4,
            alpha=0.7,
            label="Complete",
        ),
        Patch(
            facecolor=tint(MUTED),
            edgecolor=MUTED,
            hatch="////////",
            hatch_linewidth=0.5,
            linewidth=0.7,
            alpha=0.7,
            label="Truncated",
        ),
    ]
    inside_legend(fig, axs[0, 0], [*identity_row(v, line=False), *event_handles])
    axs[0, 0].get_legend().set_loc("upper left")
    return fig


def spi_checks(v: View) -> Figure:
    """Shapiro-Wilk p-values of reference SPI per calendar month and scale, gamma vs Pearson III."""
    ramps = {
        s: [
            to_hex(0.65 * np.array(to_rgb(v.color(s)))),
            tint(v.color(s)),
            to_hex(0.1 * np.array(to_rgb(v.color(s))) + 0.9),
        ]
        for s in v.sources
    }
    norm = BoundaryNorm([0, 0.01, 0.05, 1], 3)
    rows = [(s, d) for s in v.sources for d in ("gamma", "pearson3")]
    fig, axs = new_fig(
        14 + 24 * len(rows),
        len(rows),
        1,
        width_mm=W_MID if len(rows) == 2 else W_FULL,
        sharex=True,
    )
    for i, (ax, (s, d)) in enumerate(zip(axs[:, 0], rows)):
        cmap = ListedColormap(ramps[s])
        tab = np.array(
            [v.results[s]["spi_checks"][k][f"sw_{d}"].to_numpy() for k in SPI_SCALES],
            dtype=float,
        )
        ax.set_facecolor(MISSING)
        ax.pcolormesh(
            np.arange(13) - 0.5,
            np.arange(4) - 0.5,
            tab,
            cmap=cmap,
            norm=norm,
            edgecolors=(*to_rgb(GRID), 0.5),
            linewidth=0.4,
        )
        for (r, c), val in np.ndenumerate(tab):
            if np.isfinite(val):
                ax.text(
                    c,
                    r,
                    f"{val:.2f}" if val >= 0.01 else "<0.01",
                    ha="center",
                    va="center",
                    fontsize=FS_NOTE,
                    color=INK,
                    bbox=LABEL_BOX,
                )
        ax.set_ylim(2.5, -0.5)
        ax.set_yticks(range(3), [f"SPI-{k}" for k in SPI_SCALES])
        name = "gamma" if d == "gamma" else "Pearson III"
        ax.set_title(f"{chr(97 + i)}) {v.labels[s]} · {name}")
        heatmap_frame(ax)
        ax.tick_params(axis="both", length=3)
    month_axis(axs[-1, 0], v.m0)
    keys = [tuple(Patch(facecolor=ramps[s][i]) for s in v.sources) for i in range(3)]
    legend = fig.legend(
        handles=keys,
        labels=["$p$ < 0.01", "0.01 ≤ $p$ < 0.05", "$p$ ≥ 0.05"],
        handler_map={tuple: HandlerTuple(ndivide=None, pad=0.2)},
        loc="outside right upper",
        ncols=1,
        frameon=True,
        facecolor="white",
        edgecolor="none",
        framealpha=0.88,
        borderpad=0.4,
        labelspacing=0.5,
        handletextpad=0.5,
    )
    legend.set_loc("upper right")
    legend.borderaxespad = 0
    fig.get_layout_engine().set(rect=(0, 0, 0.84, 1))
    fig.canvas.draw()
    top = axs[0, 0].get_position().y1
    legend.set_bbox_to_anchor((1, top), transform=fig.transFigure)
    return fig


EXTREME_LABELS = {
    "Rx1day": "mm",
    "Rx5day": "mm",
    "SDII": UNITS["mm_day"],
    "Wet days": UNITS["days_year"],
}


def extremes(indicator: str) -> Callable[[View], Figure]:
    def chart(v: View) -> Figure:
        fig, axs = new_fig(H_SINGLE)
        ax = axs[0, 0]
        for s in v.sources:
            y = v.results[s]["extremes"][indicator].reindex(v.years)
            ax.plot(
                v.years, y, color=v.color(s), lw=LW_LINE, marker=v.marker(s), ms=MS_LINE
            )  # NaN breaks the line
        ax.set_ylabel(EXTREME_LABELS[indicator])
        ax.margins(y=0.08)
        trend_key = {"SDII": "sdii", "Wet days": "wet_days"}.get(indicator, indicator)
        unit = (
            "mm d⁻¹ decade⁻¹"
            if indicator == "SDII"
            else UNITS["days_decade"]
            if indicator == "Wet days"
            else UNITS["mm_decade"]
        )
        _annual_trends(
            ax,
            v,
            "extremes",
            indicator,
            trend_key,
            unit,
            digits=2 if indicator == "SDII" else 1,
        )
        year_ticks(ax, v.years)
        ygrid(ax)
        top_legend(fig, identity_row(v))
        return fig

    return chart


def spell(v: View) -> Figure:
    fig, axs = new_fig(H_SINGLE, width_mm=W_COL1)
    ax = axs[0, 0]
    ax.set_box_aspect(1)
    xmax = 2
    for s in v.sources:
        ex = v.results[s]["spells"]
        ex = ex[ex >= 0.002]
        if ex.empty:
            continue
        ax.plot(
            ex.index,
            ex.to_numpy(),
            color=v.color(s),
            lw=LW_LINE,
            marker=v.marker(s),
            ms=MS_LINE,
            markevery=max(1, ex.size // 12),
        )  # shape stays as identity without crowding the curve
        xmax = max(xmax, int(ex.index.max()))
    ax.set_yscale("log")
    ticks = [0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1]
    ax.set_yticks(ticks, [f"{t:g}" for t in ticks])
    ax.minorticks_off()
    ax.set_ylim(0.0015, 1.2)
    ax.set_xlim(0.5, xmax + 0.5)
    ax.set_xlabel("Dry-spell length (days)")
    ax.set_ylabel("Exceedance Pr. (log)")
    ax.grid()
    inside_legend(fig, ax, identity_row(v), lines=ax.lines)
    return fig


def acf(v: View) -> Figure:
    fig, axs = new_fig(H_SINGLE)
    ax = axs[0, 0]
    n = len(v.sources)
    rmax = 0.0
    bands = []
    for j, s in enumerate(v.sources):
        a = v.results[s]["acf"]
        lags = a.index.to_numpy() + dodge(j, n)
        bound = 1.96 / np.sqrt(a["pairs"].where(a["pairs"] > 0))
        bands.append(
            ax.fill_between(
                a.index.to_numpy(),
                -bound,
                bound,
                color=v.color(s),
                alpha=BAND_ALPHA,
                linewidth=0,
                zorder=1,
            )
        )
        ax.vlines(lags, 0, a["r"], color=v.color(s), lw=LW_STEM)
        edge = to_hex(0.65 * np.array(to_rgb(v.color(s))))
        ax.plot(
            lags,
            a["r"],
            lw=0,
            marker=v.marker(s),
            color=v.color(s),
            ms=MS_LINE,
            markeredgecolor=edge,
            markeredgewidth=0.7,
        )
        finite = a["r"].dropna()
        if not finite.empty:
            rmax = max(rmax, float(np.abs(finite).max()))
        finite_bounds = bound.loc[np.isfinite(bound)]
        if not finite_bounds.empty:
            rmax = max(rmax, float(finite_bounds.max()))
    lim = max(0.5, np.ceil(rmax * 10) / 10)
    ax.axhline(0, color=MUTED, lw=LW_REF)
    ax.set_xticks(range(1, 13))
    ax.set_xlim(0.4, 12.6)
    ax.set_ylim(-lim, lim)
    ax.set_xlabel("Lag (months)")
    ax.set_ylabel("Autocorrelation (r)")
    ygrid(ax)

    confidence = Patch(
        facecolor=MUTED,
        alpha=BAND_ALPHA,
        edgecolor="none",
        label="95% confidence limits",
    )
    inside_legend(fig, ax, [*identity_row(v, line=False), confidence], bands)
    return fig


TREND_INDICATORS = (
    ("total", "Annual precipitation", UNITS["mm_decade"]),
    ("wet_days", "Wet days", UNITS["days_decade"]),
    ("Rx1day", "Rx1day", UNITS["mm_decade"]),
)


def slopes(v: View) -> Figure:
    n = len(v.sources)
    fig, axs = new_fig(48 + 8 * max(0, n - 2), 1, 3, width_mm=W_FULL)
    fig.get_layout_engine().set(w_pad=1.5 * MM, wspace=0.02)
    for i, (ax, (key, title, unit)) in enumerate(zip(axs[0], TREND_INDICATORS)):
        span = 0.0
        for k, s in enumerate(v.sources):
            t = v.results[s]["trends"][key]
            if not np.isfinite(t["slope"]):
                continue
            sig = t["p"] < 0.05
            ax.errorbar(
                t["slope"],
                k * 0.6,
                xerr=[[t["slope"] - t["lo"]], [t["hi"] - t["slope"]]],
                fmt=v.marker(s),
                color=v.color(s),
                mfc=v.color(s) if sig else "white",
                mec=v.color(s),
                mew=1.0,
                ms=MS_POINT,
                lw=LW_LINE * 1.5,
                capsize=3,
                capthick=1.2,
                zorder=3,
            )
            span = max(span, abs(t["lo"]), abs(t["hi"]))
        ax.axvline(0, color=MUTED, lw=LW_REF, zorder=1)
        span = span * 1.15 or 1
        ax.set_xlim(-span, span)
        ax.set_ylim((n - 1) * 0.6 + 0.6, -0.6)
        ax.set_yticks([])
        ax.set_title(f"{chr(97 + i)}) {title}")
        ax.set_xlabel(unit)
        ax.locator_params(axis="x", nbins=6)
        ax.grid(axis="x", alpha=0.5)
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color(BOUNDARY)
            spine.set_linewidth(LW_BOUNDARY)
        ax.tick_params(axis="y", length=3)
    significance = [
        Line2D(
            [],
            [],
            linestyle="none",
            marker="o",
            color=MUTED,
            markerfacecolor=MUTED,
            markersize=MS_POINT,
            label="p < 0.05",
        ),
        Line2D(
            [],
            [],
            linestyle="none",
            marker="o",
            color=MUTED,
            markerfacecolor="white",
            markersize=MS_POINT,
            label="p ≥ 0.05",
        ),
    ]
    fig.legend(
        handles=[*identity_row(v, line=False), *significance],
        loc="outside right upper",
        ncols=1,
        frameon=True,
        facecolor="white",
        edgecolor="none",
        framealpha=0.88,
        borderpad=0.4,
        labelspacing=0.5,
        handletextpad=0.5,
    )
    for ax in axs[0]:
        ax.title.set_fontsize(FS_NOTE)
        ax.xaxis.label.set_fontsize(FS_NOTE)
    return fig


def series(
    v: View,
    ylabel: str = UNITS["mm_year"],
    trend_unit: str = UNITS["mm_decade"],
    digits: int = 0,
) -> Figure:
    """Annual basin series; defaults retain the reviewed precipitation figure."""
    return pc.annual_series(
        v.years,
        [
            pc.AnnualLine(
                v.labels[s],
                v.color(s),
                v.marker(s),
                v.results[s]["annual"]["total"].reindex(v.years).to_numpy(),
            )
            for s in v.sources
        ],
        ylabel=ylabel,
        trends=_prepared_trends(v, "annual", "total", "total", trend_unit, digits),
    )


def _prepared_trends(
    v: View,
    table: str,
    column: str,
    indicator: str,
    unit: str,
    digits: int,
) -> list[pc.TrendSegment]:
    """Translate retained estimates using the original eligible endpoint rule."""
    segments = []
    for s in v.sources:
        y = v.results[s][table][column].reindex(v.years)
        t = v.results[s]["trends"][indicator]
        ok = y.dropna().index.to_numpy()
        if np.isfinite(t["slope"]) and ok.size:
            span = np.array([ok[0], ok[-1]], dtype=float)
            line = t["intercept"] + t["slope"] / 10 * span
            segments.append(
                pc.TrendSegment(
                    v.color(s),
                    tuple(span),
                    tuple(line),
                    f"{fmt_signed(t['slope'], digits)} {unit}",
                )
            )
    return segments


def _annual_trends(
    ax: plt.Axes,
    v: View,
    table: str,
    column: str,
    indicator: str,
    unit: str,
    digits: int = 0,
) -> None:
    """Draw retained Sen estimates with separated, source-colored slope labels."""
    pc.draw_annual_trends(
        ax,
        _prepared_trends(v, table, column, indicator, unit, digits),
        label_x=v.years[-1] - 0.4,
    )


def sensitivity(v: View) -> Figure:
    """Sen slope with dependence-adjusted interval for each start year (end fixed); filled = p < 0.05."""
    fig, axs = new_fig(H_SIDE2, 1, 2, width_mm=W_FULL, sharex=True)
    starts = sorted(
        {
            year
            for s in v.sources
            for key in ("total", "Rx1day")
            for year in v.results[s]["sensitivity"][key].index
        }
    )
    for i, (ax, (key, title)) in enumerate(
        zip(axs[0], (("total", "Annual precipitation"), ("Rx1day", "Rx1day")))
    ):
        for j, s in enumerate(v.sources):
            t = v.results[s]["sensitivity"][key]
            if t.empty:
                continue
            x = t.index.to_numpy() + dodge(j, len(v.sources))
            c = v.color(s)
            edge = to_hex(0.65 * np.array(to_rgb(c)))
            sig = (t["p"] < 0.05).to_numpy()
            for selected, fill_color in ((sig, c), (~sig, "white")):
                values = t.loc[selected]
                ax.errorbar(
                    x[selected],
                    values["slope"],
                    yerr=[
                        values["slope"] - values["lo"],
                        values["hi"] - values["slope"],
                    ],
                    fmt=v.marker(s),
                    linestyle="none",
                    color=c,
                    mfc=fill_color,
                    mec=edge,
                    mew=0.7,
                    ms=MS_POINT,
                    elinewidth=LW_LINE * 1.5,
                    capsize=3,
                    capthick=1.2,
                    zorder=3,
                )
        ax.xaxis.set_major_locator(
            MaxNLocator(nbins=5, integer=True, steps=[1, 2, 5, 10])
        )
        ax.xaxis.set_major_formatter(StrMethodFormatter("{x:.0f}"))
        ax.xaxis.set_minor_locator(FixedLocator(starts))
        ax.tick_params(axis="x", which="minor", length=2, width=LW_EDGE, color=AXIS)
        if starts:
            ax.set_xlim(min(starts) - 0.5, max(starts) + 0.5)
        ax.axhline(0, color=MUTED, lw=LW_REF)
        ax.set_title(f"{chr(97 + i)}) {title}")
        ax.set_ylabel(UNITS["mm_decade"])
        ax.set_xlabel(f"Trend start year (fixed end: {v.years[-1]})")
        ygrid(ax)
    sig = [
        Line2D(
            [], [], lw=0, marker="o", ms=MS_LINE + 1, color=MUTED, label="$p$ < 0.05"
        ),
        Line2D(
            [],
            [],
            lw=0,
            marker="o",
            ms=MS_LINE + 1,
            mfc="white",
            mec=MUTED,
            mew=1.0,
            label="$p$ ≥ 0.05",
        ),
    ]
    edge = Line2D([], [], color=MUTED, lw=LW_LINE * 1.5, label="95% interval")
    legend = fig.legend(
        handles=[*identity_row(v, line=False), *sig, edge],
        loc="outside right upper",
        ncols=1,
        frameon=True,
        facecolor="white",
        edgecolor="none",
        framealpha=0.88,
    )
    legend.set_loc("upper right")
    legend.borderaxespad = 0
    fig.get_layout_engine().set(rect=(0, 0, 0.84, 1))
    fig.canvas.draw()
    legend.set_bbox_to_anchor(
        (1, axs[0, 0].get_position().y1), transform=fig.transFigure
    )
    return fig


def pt_scatter(v: View) -> Figure | str:
    """Seasonal P anomaly (%) against T anomaly; P and T sources are labelled separately."""
    have = [
        s
        for s in v.sources
        if v.results[s]["pt"] is not None and not v.results[s]["pt"].empty
    ]
    if not have:
        return notice(
            "No temperature variable is declared in this case, so the P–T scatter is unavailable."
        )
    # Own y-scale per season: dry-season totals are small, so their % anomalies are much larger.
    fig, axs = new_fig(H_GRID4, 2, 2, width_mm=W_FULL, sharex=True)
    xlim = (
        max(float(np.nanmax(np.abs(v.results[s]["pt"]["t_anom"]))) for s in have) * 1.08
    )
    for i, (ax, season) in enumerate(zip(axs.flat, ("DJF", "MAM", "JJA", "SON"))):
        ax.set_box_aspect(1)
        for s in have:
            d = v.results[s]["pt"]
            d = d[d["season"] == season]
            ax.scatter(
                d["t_anom"],
                d["p_anom_pct"],
                s=S_SCATTER * 2.25,
                alpha=0.7,
                marker=v.marker(s),
                color=v.color(s),
                edgecolor=to_hex(0.65 * np.array(to_rgb(v.color(s)))),
                linewidth=0.7,
                zorder=3,
            )
        ax.axhline(0, color=MUTED, lw=LW_REF)
        ax.axvline(0, color=MUTED, lw=LW_REF)
        ax.set_xlim(-xlim, xlim)
        ax.set_title(f"{chr(97 + i)}) {season}")
        ax.grid()
        ax.tick_params(axis="both", length=3)
    for ax in axs[-1]:
        ax.set_xlabel(f"Temperature anomaly ({UNITS['degc']})")
    fig.supylabel("Precipitation anomaly (%)", fontsize=FS_LABEL, color=INK)
    inside_legend(
        fig,
        axs[0, 0],
        identity_row(v, have, line=False, label=lambda s: f"Precip.: {v.labels[s]}"),
    )
    return fig


CHARTS: dict[str, Callable[[View], Figure | str]] = {
    "coverage": coverage,
    "seasonal": seasonal,
    "temperature": temperature,
    "timing": timing,
    "anomaly": anomaly,
    "spi": spi,
    "drought-events": drought_events,
    **{f"extreme-{m}": extremes(m) for m in EXTREME_LABELS},
    "spell": spell,
    "acf": acf,
    "slopes": slopes,
    "series": series,
    "pt": pt_scatter,
    "spi-checks": spi_checks,
    "sensitivity": sensitivity,
}
