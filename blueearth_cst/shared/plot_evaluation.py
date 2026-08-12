# -*- coding: utf-8 -*-
"""The four evaluation sheets rule 1.15 draws for each modelled station.

One station, four figures, split by the question a reader is asking:

* ``hydrograph_<wflow_id>`` — what the model produces. Daily series, annual
  means, the mean annual cycle and both flow-duration curves. The ONLY sheet
  that needs no observations, so it is drawn for every station on every run.
* ``signatures_peaks_<wflow_id>`` — the high end. Flood frequency, the
  year-by-year annual maximum against observed, and the wettest year.
* ``signatures_lows_<wflow_id>`` — the dry end, in the same three places, so the
  two sheets read against each other without re-learning the layout.
* ``performance_<wflow_id>`` — goodness of fit: the daily scatter, the
  cumulative volume that shows WHEN a bias accrues, and the metrics table.

This replaces ``func_plot_signature.plot_hydro`` (five stacked panels) and
``plot_signatures`` (ten untitled ones). What changed beyond the layout:

* **Keyed by wflow_id.** The old figures were named ``wflow_1`` — a 1..N counter
  invented in ``plot_results`` — or by a station_name the delineation generated.
  Every other artifact of a run names a point by its ``wflow_id``: the
  ``Q_1010`` column in ``output.csv``, the registry, the labels on
  ``basin_area``. So do these.
* **Three unit errors fixed.** The annual and monthly panels plotted
  ``resample(...).sum()`` of a series in m3 s-1 and labelled the result
  ``m3 yr-1`` / ``m3 month-1`` — wrong by 86400, since a sum of daily mean
  discharges is not a volume. They are means now, in m3 s-1, which is also the
  unit of the daily panel above them. Cumulative discharge, labelled
  ``m3 s-1``, is a real volume here.
* **The performance panel can show failure.** Its predecessor pinned the y axis
  to [0, 1], so a model with NSE = -3 plotted off the bottom and looked
  identical to one scoring 0. It is a table now, and the table prints what the
  number is.
* **The log flow-duration curve is a log AXIS**, not ``np.log(Q)`` on a linear
  one. This basin has zero-flow days, and those became ``-inf``.

Nothing here reads or writes a file except the figures themselves;
``model/plot_results.py`` resolves the model, the registry and the observations.
"""

import os

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
from matplotlib import gridspec, rc_context
from matplotlib.ticker import MaxNLocator

from blueearth_cst.shared import plot_style
from blueearth_cst.shared.snake_utils import save_figure

# ===========================================================================
# TUNABLE CONSTANTS
# ===========================================================================

#: Observed and simulated, in that order. Okabe-Ito blue and vermillion: the
#: pair stays separable under all three dichromacies AND differs in lightness,
#: so the dashed/solid distinction below is a greyscale BACKUP rather than the
#: only cue carrying the difference. Owner's call 2026-08-10 that observed is
#: blue and simulated red; the specific hexes are the accessible version of it.
COLOR_OBSERVED = "#0072B2"
COLOR_SIMULATED = "#D55E00"
#: Marks that belong to neither series — the period mean, a 1:1 line's anchor.
COLOR_NEUTRAL = "#1a1a1a"
COLOR_GRID = "#dcdcdc"
#: The 1:1 reference line on every scatter panel.
COLOR_REFERENCE = "0.55"

#: Observed dashes, simulated solid. Simulated is solid because it is the dense
#: series — a dashed line over 7000 daily values reads as a smear.
DASH_OBSERVED = (0, (3, 1.6))
LINE_WIDTH = 0.7

#: Sheet heights in millimetres. The width is ``plot_style.FIGURE_WIDTH_MM`` for
#: all four, so they stack in a report at one size.
HEIGHT_HYDROGRAPH_MM = 148.0
HEIGHT_EXTREMES_MM = 122.0
HEIGHT_PERFORMANCE_MM = 118.0

#: Return periods marked on the frequency panels.
RETURN_PERIODS = (2.0, 5.0, 10.0, 30.0)
#: Gringorten plotting-position constants, as the previous implementation used.
_PLOTTING_POSITION_A = 0.3

#: A log axis cannot show a zero, and this toolbox runs on basins that dry out.
#: The floor is the 1st percentile of the POSITIVE simulated values, never below
#: this, so one 1e-9 cell cannot set the range for the whole curve.
_LOG_FDC_FLOOR = 1e-3

#: Metric rows on the performance table: the ideal value a reader should compare
#: against, and how many decimals the number is worth. Grouped because "higher
#: is better" and "zero is better" are different readings and a reader should
#: not have to remember which is which per row.
METRIC_FORMAT = {
    "KGE": ("1", "{:.3f}"),
    "NSE": ("1", "{:.3f}"),
    "NSElog": ("1", "{:.3f}"),
    "VE": ("1", "{:.3f}"),
    "RMSE": ("0", "{:.2f}"),
    "MSE": ("0", "{:.2f}"),
    "Pbias": ("0 %", "{:+.1f} %"),
}
METRIC_GROUPS = (
    ("Efficiency", ("KGE", "NSE", "NSElog", "VE")),
    ("Error", ("RMSE", "MSE", "Pbias")),
)

#: Seconds per day: the daily mean discharge to volume conversion, and the
#: reason the cumulative panel is in m3 rather than in the "cumulative m3 s-1"
#: its predecessor printed.
SECONDS_PER_DAY = 86400.0

_MONTH_INITIALS = list("JFMAMJJASOND")


class Station:
    """Who a sheet is about: the wflow_id, and the context around it.

    ``wflow_id`` is the key — it names the ``output.csv`` column, the registry
    row and the point on ``basin_area``, and it is what the filenames use. The
    other two are shown on the sheet because a user-supplied station name is
    real information when a project has one, and because the subbasin is how a
    reader finds this point on the delineation map.
    """

    def __init__(self, wflow_id, subbasin_id=None, station_name=None):
        self.wflow_id = int(wflow_id)
        self.subbasin_id = subbasin_id
        self.station_name = station_name

    @property
    def caption(self):
        """The identity line at the head of every sheet.

        The wflow_id is set off by the wider separator because it is the KEY;
        what follows it is context, and a project without a location registry or
        without user station names still gets a correct, shorter line.
        """
        context = []
        if self.subbasin_id is not None:
            context.append(f"subbasin {self.subbasin_id}")
        if self.station_name:
            context.append(str(self.station_name))
        head = f"wflow_id {self.wflow_id}"
        return f"{head}  ·  {' · '.join(context)}" if context else head


# ===========================================================================
# DERIVED VALUES
# ===========================================================================
# Functions, never module constants: a constant assembled from the tunables
# above snapshots them at import, which would make any override silently
# ineffective. Same rule the cartographic template documents.
# ---------------------------------------------------------------------------


def _rc():
    """The rcParams every evaluation sheet is drawn under.

    The page settings come from ``plot_style`` — the same 180 mm and 600 dpi the
    maps and the climate figures use. What is added here is picture-level: a
    grid (a hydrograph is read by value, unlike a map), and left-aligned bold
    panel titles, which is what makes a lettered panel scannable.
    """
    params = plot_style.rcparams(axes_linewidth=0.6)
    params.update(
        {
            "axes.grid": True,
            "grid.color": COLOR_GRID,
            "grid.linewidth": 0.4,
            "axes.edgecolor": "0.3",
            "axes.titlesize": plot_style.FONT_SIZE_BASE,
            "axes.titleweight": "bold",
            "axes.titlelocation": "left",
            "axes.titlepad": 3.0,
            "figure.facecolor": "white",
        }
    )
    return params


def _figure_size(height_mm):
    return (
        plot_style.FIGURE_WIDTH_MM / plot_style.MM_PER_INCH,
        height_mm / plot_style.MM_PER_INCH,
    )


def _panel(ax, letter, title):
    """Letter and title one panel. Every panel gets both.

    The predecessor cleared all ten of its titles (``ax.set_title("")``), which
    left the reader decoding each panel from its axis labels.
    """
    ax.set_title(f"{letter}  {title}")


def _identity(fig, station, subject, legend_ax=None):
    """The header line, the footnote, and one figure-level legend.

    A single legend rather than one per panel: every panel on a sheet draws the
    same two series, so repeating the key is noise. ``legend_ax`` is whichever
    axes carries both handles.
    """
    fig.suptitle(
        f"{station.caption} · {subject}",
        fontsize=plot_style.FONT_SIZE_TITLE,
        fontweight="bold",
        x=0.006,
        ha="left",
    )
    if legend_ax is not None:
        handles, labels = legend_ax.get_legend_handles_labels()
        if handles:
            fig.legend(
                handles,
                labels,
                loc="outside upper right",
                ncols=2,
                frameon=False,
                fontsize=plot_style.FONT_SIZE_LEGEND,
            )


def _caveat(fig, text):
    if text:
        fig.supxlabel(
            text, fontsize=plot_style.FONT_SIZE_CAVEAT, color=plot_style.COLOR_CAVEAT
        )


def _draw_series(ax, simulated, observed, x_sim=None, x_obs=None):
    """Observed then simulated, so the dense simulated line reads on top."""
    if observed is not None:
        ax.plot(
            x_obs if x_obs is not None else observed["time"],
            observed,
            color=COLOR_OBSERVED,
            lw=LINE_WIDTH,
            ls=DASH_OBSERVED,
            label="observed",
            zorder=3,
        )
    ax.plot(
        x_sim if x_sim is not None else simulated["time"],
        simulated,
        color=COLOR_SIMULATED,
        lw=LINE_WIDTH,
        label="simulated",
        zorder=4,
    )


def _reference_line(ax, high):
    """A 1:1 line and matching limits, WITHOUT locking the aspect ratio.

    A locked aspect shrinks the axes inside its layout cell and centres it, so
    two panels sharing a row come out different heights with their edges out of
    line — measured on the first draft, and the reason this uses equal LIMITS
    instead. The line is still correct in data coordinates, which is what a
    reader compares points against.
    """
    ax.plot(
        [0, high], [0, high], color=COLOR_REFERENCE, ls=(0, (4, 2)), lw=0.6, zorder=1
    )
    ax.set_xlim(0, high)
    ax.set_ylim(0, high)


def _score_box(ax, text):
    """A boxed corner label, rather than text floating at a fixed fraction.

    Its predecessor placed scores at hard-coded axes fractions (0.2, 0.7) and
    (0.5, 0.05), which lands on the data as soon as the data changes shape.
    """
    ax.text(
        0.04,
        0.96,
        text,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=plot_style.FONT_SIZE_LEGEND,
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.75", lw=0.4),
    )


def _month_axis(ax):
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=(1, 3, 5, 7, 9, 11)))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    ax.set_xlabel("")


def _draw_fdc(ax, simulated, observed, log_scale):
    """A flow-duration curve, linear or log, in real discharge units."""
    for data, color, style in (
        (observed, COLOR_OBSERVED, DASH_OBSERVED),
        (simulated, COLOR_SIMULATED, "-"),
    ):
        if data is None:
            continue
        probability = np.arange(1, data.time.size + 1) / (data.time.size + 1)
        ax.plot(
            probability,
            data.sortby(data, ascending=False),
            color=color,
            lw=LINE_WIDTH,
            ls=style,
            label="observed" if color == COLOR_OBSERVED else "simulated",
        )
    ax.set_xlabel("exceedance probability (–)")
    ax.set_ylabel("Q (m$^3$ s$^{-1}$)")
    if log_scale:
        ax.set_yscale("log")
        positive = simulated.where(simulated > 0)
        floor = (
            float(np.nanpercentile(positive, 1)) if positive.notnull().any() else 0.0
        )
        ax.set_ylim(bottom=max(_LOG_FDC_FLOOR, floor))


def _draw_frequency(ax, observed, simulated, ylabel, ascending):
    """Extremes on a Gumbel reduced variate, with return periods at the frame.

    The labels sit AT the top of the frame with a halo, not rotated 45 degrees
    across the middle of the panel as they were — where they crossed the points
    they were meant to annotate.
    """
    count = simulated.time.size
    b = 1.0 - 2.0 * _PLOTTING_POSITION_A
    probability = (np.arange(1, count + 1.0) - _PLOTTING_POSITION_A) / (count + b)
    reduced = -np.log(-np.log(probability))
    if observed is not None:
        ax.plot(
            reduced,
            observed.sortby(observed, ascending=ascending),
            "+",
            color=COLOR_OBSERVED,
            ms=4,
            label="observed",
        )
    ax.plot(
        reduced,
        simulated.sortby(simulated, ascending=ascending),
        "o",
        ms=2.8,
        mfc=COLOR_SIMULATED,
        mec="none",
        label="simulated",
    )
    top = ax.get_ylim()[1]
    for period in RETURN_PERIODS:
        position = -np.log(-np.log(1 - 1.0 / period))
        ax.axvline(position, color="0.75", lw=0.5, zorder=1)
        ax.text(
            position,
            top,
            f"T={period:.0f}y",
            rotation=90,
            va="top",
            ha="right",
            fontsize=plot_style.FONT_SIZE_CAVEAT,
            color="0.45",
            bbox=dict(boxstyle="square,pad=0.12", fc="white", ec="none", alpha=0.85),
        )
    ax.set_xlabel("Gumbel reduced variate (–)")
    ax.set_ylabel(ylabel)


def annual_extremes(simulated, observed):
    """``(sim_max, obs_max, sim_nm7q, obs_nm7q)`` for the two extremes sheets.

    Annual maxima are taken on a SEPTEMBER water year, as the previous
    implementation did, so one flood season is never split across two "years".
    NM7Q is the annual minimum of the 7-day rolling mean, on calendar years.
    """
    start = f"{int(simulated['time.year'][0])}-09-01"
    end = f"{int(simulated['time.year'][-1])}-08-31"

    def maxima(data):
        return data.sel(time=slice(start, end)).resample(time="YS-SEP").max("time")

    def nm7q(data):
        return data.rolling(time=7).mean().resample(time="YE").min("time")

    return (
        maxima(simulated),
        maxima(observed) if observed is not None else None,
        nm7q(simulated),
        nm7q(observed) if observed is not None else None,
    )


def wettest_and_driest_year(simulated):
    """The years of the highest and lowest MEAN discharge.

    Ranked on the mean rather than on ``resample(...).sum()``: on a series with
    missing days a sum ranks by how much data a year has as much as by how wet
    it was.
    """
    annual = simulated.groupby("time.year").mean()
    return int(annual.idxmax().values), int(annual.idxmin().values)


def _save(fig, plot_dir, stem):
    """Write one sheet and return its path.

    PNG only since 2026-08-10 (owner's call, applied across the deliverable).
    The PDF was the vector, embedded-font publication copy and nothing read it;
    600 dpi at 180 mm carries the figure everywhere it is used, and not
    serialising each sheet twice halves the rule's render time.

    The metadata scrub stays: the default embeds the matplotlib version, which
    would move a fingerprint on every environment bump.
    """
    png = os.path.join(str(plot_dir), f"{stem}.png")
    save_figure(png, fig=fig, dpi=plot_style.RASTER_DPI, metadata={"Software": None})
    plt.close(fig)
    return png


# ===========================================================================
# THE FOUR SHEETS
# ===========================================================================


def plot_hydrograph(simulated, station, plot_dir, observed=None, caveat=None):
    """What the model produces: daily, annual, seasonal, and both FDCs.

    The one sheet drawn for EVERY station on every run — every panel is
    computable from the simulation alone, so ``observed`` only adds a second
    line where one exists.
    """
    years = np.unique(simulated["time.year"].values)
    annual = simulated.groupby("time.year").mean()
    cycle = simulated.groupby("time.month").mean()
    annual_obs = observed.groupby("time.year").mean() if observed is not None else None
    cycle_obs = observed.groupby("time.month").mean() if observed is not None else None

    with rc_context(_rc()):
        fig = plt.figure(
            figsize=_figure_size(HEIGHT_HYDROGRAPH_MM), layout="constrained"
        )
        grid = gridspec.GridSpec(3, 2, figure=fig, height_ratios=[1.15, 1.0, 1.0])
        ax_daily = fig.add_subplot(grid[0, :])
        ax_annual = fig.add_subplot(grid[1, 0])
        ax_cycle = fig.add_subplot(grid[1, 1])
        ax_fdc = fig.add_subplot(grid[2, 0])
        ax_fdc_log = fig.add_subplot(grid[2, 1])

        _draw_series(ax_daily, simulated, observed)
        _panel(ax_daily, "a", f"Daily discharge · {years[0]}–{years[-1]}")
        ax_daily.set_ylabel("Q (m$^3$ s$^{-1}$)")

        _draw_series(
            ax_annual,
            annual,
            annual_obs,
            x_sim=annual["year"],
            x_obs=None if annual_obs is None else annual_obs["year"],
        )
        _panel(ax_annual, "b", "Annual mean discharge")
        ax_annual.set_ylabel("Q (m$^3$ s$^{-1}$)")
        ax_annual.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=6))

        _draw_series(
            ax_cycle,
            cycle,
            cycle_obs,
            x_sim=cycle["month"],
            x_obs=None if cycle_obs is None else cycle_obs["month"],
        )
        _panel(ax_cycle, "c", "Mean annual cycle")
        ax_cycle.set_ylabel("Q (m$^3$ s$^{-1}$)")
        ax_cycle.set_xticks(np.arange(1, 13))
        ax_cycle.set_xticklabels(_MONTH_INITIALS)

        _draw_fdc(ax_fdc, simulated, observed, log_scale=False)
        _panel(ax_fdc, "d", "Flow-duration curve")
        _draw_fdc(ax_fdc_log, simulated, observed, log_scale=True)
        _panel(ax_fdc_log, "e", "Flow-duration curve (log)")

        for ax in (ax_daily, ax_annual, ax_cycle):
            ax.margins(x=0.01)
        _identity(fig, station, "discharge", ax_daily)
        _caveat(fig, caveat)
        return _save(fig, plot_dir, f"hydrograph_{station.wflow_id}")


def plot_extremes(simulated, observed, station, plot_dir, kind, caveat=None):
    """One end of the distribution: frequency, year-by-year, and one year.

    ``kind`` is ``"peaks"`` or ``"lows"``. The two sheets are deliberately
    identical in shape — analysis on the top row, the illustrating year full
    width below — so a reader moving between them re-reads the layout for free.

    Both need observations: two of the three panels compare against them.
    """
    if kind not in ("peaks", "lows"):
        raise ValueError(f"kind={kind!r}; expected 'peaks' or 'lows'")
    peaks = kind == "peaks"
    sim_max, obs_max, sim_nm7q, obs_nm7q = annual_extremes(simulated, observed)
    sim_extreme, obs_extreme = (sim_max, obs_max) if peaks else (sim_nm7q, obs_nm7q)
    wet, dry = wettest_and_driest_year(simulated)
    year = wet if peaks else dry

    with rc_context(_rc()):
        fig = plt.figure(figsize=_figure_size(HEIGHT_EXTREMES_MM), layout="constrained")
        grid = gridspec.GridSpec(2, 2, figure=fig, height_ratios=[1.32, 1.0])
        ax_frequency = fig.add_subplot(grid[0, 0])
        ax_scatter = fig.add_subplot(grid[0, 1])
        ax_year = fig.add_subplot(grid[1, :])

        _draw_frequency(
            ax_frequency,
            obs_extreme,
            sim_extreme,
            "annual max Q (m$^3$ s$^{-1}$)" if peaks else "NM7Q (m$^3$ s$^{-1}$)",
            ascending=peaks,
        )
        _panel(ax_frequency, "a", "Flood frequency" if peaks else "Low-flow frequency")

        high = float(max(obs_extreme.max(), sim_extreme.max()) * 1.08)
        ax_scatter.plot(
            obs_extreme,
            sim_extreme,
            "o",
            ms=3.2,
            mfc=COLOR_SIMULATED,
            mec="white",
            mew=0.4,
            ls="none",
            zorder=2,
        )
        if peaks:
            # The period mean, so a systematic under- or over-estimate of flood
            # magnitude reads at a glance rather than out of the cloud.
            ax_scatter.plot(
                obs_extreme.mean(),
                sim_extreme.mean(),
                marker="D",
                ms=4.5,
                mfc=COLOR_NEUTRAL,
                mec="white",
                mew=0.5,
                ls="none",
                zorder=3,
                label="period mean",
            )
            ax_scatter.legend(
                loc="lower right",
                frameon=False,
                fontsize=plot_style.FONT_SIZE_CAVEAT,
                handletextpad=0.3,
            )
        _reference_line(ax_scatter, high)
        _panel(ax_scatter, "b", "Annual maximum" if peaks else "Annual 7-day minimum")
        ax_scatter.set_xlabel("observed (m$^3$ s$^{-1}$)")
        ax_scatter.set_ylabel("simulated (m$^3$ s$^{-1}$)")
        _score_box(ax_scatter, f"R$^2$ = {r_squared(obs_extreme, sim_extreme):.2f}")

        _draw_series(
            ax_year, simulated.sel(time=str(year)), observed.sel(time=str(year))
        )
        _panel(ax_year, "c", f"{'Wettest' if peaks else 'Driest'} year · {year}")
        ax_year.set_ylabel("Q (m$^3$ s$^{-1}$)")
        ax_year.margins(x=0.01)
        _month_axis(ax_year)

        _identity(fig, station, "high flows" if peaks else "low flows", ax_year)
        _caveat(fig, caveat)
        stem = f"signatures_{'peaks' if peaks else 'lows'}_{station.wflow_id}"
        return _save(fig, plot_dir, stem)


#: Row positions of the metrics table, in axes fractions: the group band, the
#: metric names, then daily / monthly / ideal.
_TABLE_ROWS = {
    "group": 0.92,
    "header": 0.68,
    "daily": 0.44,
    "monthly": 0.25,
    "ideal": 0.06,
}
#: Where the row labels sit, and where the metric columns start.
_TABLE_LABEL_X = 0.0
_TABLE_FIRST_X = 0.135


def _draw_metrics_table(ax, metrics):
    """The goodness-of-fit table, laid out as text on a blank axes.

    A blank axes rather than ``ax.table``: matplotlib's table sizes its cells to
    the axes and ignores the figure's type scale, so it comes out at a different
    size from every other label on the page.

    METRICS ACROSS, aggregations down. The first version had it the other way —
    seven rows in a narrow column — which was right while the table shared a row
    with the scatter. It sits on a FULL-WIDTH band now, and seven rows of four
    narrow columns across 180 mm is mostly whitespace. Transposed, the same
    numbers fit three short rows, and the daily/monthly pair reads as the
    comparison it is rather than as two columns to track down the page.

    The ``ideal`` row is not decoration. A reader should not have to remember
    whether Pbias wants 0 or 1, and the group band says which direction each
    block is read in.
    """
    ax.axis("off")
    names = [
        name for _, group in METRIC_GROUPS for name in group if name in metrics.index
    ]
    if not names:
        return
    span = (1.0 - _TABLE_FIRST_X) / len(names)
    centres = {
        name: _TABLE_FIRST_X + span * (position + 0.5)
        for position, name in enumerate(names)
    }

    # Group band: which columns are read "higher is better" and which "zero is".
    for group, members in METRIC_GROUPS:
        present = [name for name in members if name in centres]
        if not present:
            continue
        left = centres[present[0]] - span / 2
        right = centres[present[-1]] + span / 2
        ax.text(
            (left + right) / 2,
            _TABLE_ROWS["group"],
            group.upper(),
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=plot_style.FONT_SIZE_CAVEAT,
            color="0.45",
            fontweight="bold",
        )
        ax.plot(
            [left + 0.006, right - 0.006],
            [_TABLE_ROWS["group"] - 0.085] * 2,
            transform=ax.transAxes,
            color="0.65",
            lw=0.6,
            clip_on=False,
        )

    for name in names:
        ax.text(
            centres[name],
            _TABLE_ROWS["header"],
            name,
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=plot_style.FONT_SIZE_LEGEND,
            color="0.2",
            fontweight="bold",
        )
    ax.plot(
        [_TABLE_LABEL_X, 1.0],
        [(_TABLE_ROWS["header"] + _TABLE_ROWS["daily"]) / 2] * 2,
        transform=ax.transAxes,
        color="0.3",
        lw=0.7,
        clip_on=False,
    )

    for row in ("daily", "monthly", "ideal"):
        muted = row == "ideal"
        ax.text(
            _TABLE_LABEL_X,
            _TABLE_ROWS[row],
            row,
            transform=ax.transAxes,
            ha="left",
            va="center",
            fontsize=plot_style.FONT_SIZE_LEGEND,
            color="0.45" if muted else "0.2",
            fontweight="normal" if muted else "bold",
        )
        for name in names:
            ideal, template = METRIC_FORMAT[name]
            ax.text(
                centres[name],
                _TABLE_ROWS[row],
                ideal if muted else template.format(metrics.loc[name, row]),
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize=plot_style.FONT_SIZE_BASE,
                color="0.55" if muted else "black",
                family="monospace",
            )
    ax.plot(
        [_TABLE_LABEL_X, 1.0],
        [(_TABLE_ROWS["monthly"] + _TABLE_ROWS["ideal"]) / 2] * 2,
        transform=ax.transAxes,
        color=COLOR_GRID,
        lw=0.5,
        clip_on=False,
    )


def plot_performance(simulated, observed, station, plot_dir, metrics, caveat=None):
    """Goodness of fit: the daily scatter, the cumulative volume, the table.

    ``metrics`` is this station's block of ``performance_metrics.csv``, indexed
    by metric with ``daily`` and ``monthly`` columns — rendered, not recomputed,
    so the figure and the table cannot disagree.

    The cumulative panel is the one that says WHEN a bias accrues. Pbias and VE
    in the table give its size; a curve that separates in one season and tracks
    in another is telling a different story from one that drifts throughout, and
    no single number carries that.
    """
    with rc_context(_rc()):
        fig = plt.figure(
            figsize=_figure_size(HEIGHT_PERFORMANCE_MM), layout="constrained"
        )
        # The two pictures first, side by side; the table full width beneath
        # them. A reader meets the scatter and the volume curve, then reads the
        # numbers that summarise them — rather than the other way round.
        grid = gridspec.GridSpec(2, 2, figure=fig, height_ratios=[2.0, 1.0])
        ax_scatter = fig.add_subplot(grid[0, 0])
        ax_volume = fig.add_subplot(grid[0, 1])
        ax_table = fig.add_subplot(grid[1, :])

        high = float(np.ceil(max(observed.max(), simulated.max())))
        ax_scatter.plot(
            observed,
            simulated,
            "o",
            ms=1.6,
            mfc=COLOR_SIMULATED,
            mec="none",
            alpha=0.5,
            ls="none",
            zorder=2,
        )
        _reference_line(ax_scatter, high)
        _panel(ax_scatter, "a", "Daily discharge")
        ax_scatter.set_xlabel("observed (m$^3$ s$^{-1}$)")
        ax_scatter.set_ylabel("simulated (m$^3$ s$^{-1}$)")
        _score_box(
            ax_scatter,
            f"R$^2$ = {r_squared(observed, simulated):.2f}\nn = {simulated.time.size:,}",
        )

        # A real volume: cumulative daily mean discharge times a day, in 10^6 m3.
        # Its predecessor plotted the bare cumsum and labelled it m3 s-1.
        scale = SECONDS_PER_DAY / 1e6
        _draw_series(
            ax_volume,
            simulated.cumsum("time") * scale,
            observed.cumsum("time") * scale,
            x_sim=simulated["time"],
            x_obs=observed["time"],
        )
        _panel(ax_volume, "b", "Cumulative volume")
        ax_volume.set_ylabel("V (10$^6$ m$^3$)")
        ax_volume.margins(x=0.01)

        _panel(ax_table, "c", "Goodness of fit")
        _draw_metrics_table(ax_table, metrics)

        _identity(fig, station, "performance", ax_volume)
        _caveat(fig, caveat)
        return _save(fig, plot_dir, f"performance_{station.wflow_id}")


def r_squared(observed, simulated):
    """Coefficient of determination between two aligned series.

    Kept here rather than imported from ``func_plot_signature`` so this module
    depends on nothing that is only there for the retired figures. Non-finite
    pairs are dropped rather than propagated: NM7Q on a basin that dries out
    produces them, and one NaN would blank the whole score.
    """
    left = np.asarray(observed, dtype="float64").ravel()
    right = np.asarray(simulated, dtype="float64").ravel()
    finite = np.isfinite(left) & np.isfinite(right)
    if finite.sum() < 2:
        return float("nan")
    return float(stats.linregress(left[finite], right[finite]).rvalue ** 2)


def plot_station_evaluation(
    simulated,
    station,
    plot_dir,
    observed=None,
    metrics=None,
    signatures=False,
    caveat=None,
    log=print,
):
    """Every sheet this station's data supports, in reading order.

    The hydrograph is always drawn. The other three need observations, and the
    two extremes sheets additionally need enough years for annual statistics to
    mean anything — which is what ``signatures`` carries from the caller, so the
    "more than a year of record" rule stays where the record is read.

    A sheet that cannot be drawn is REPORTED, never silently absent: "there are
    no observations" and "the figure failed" look identical in an empty folder.
    """
    written = [plot_hydrograph(simulated, station, plot_dir, observed, caveat)]
    if observed is None:
        log(f"wflow_id {station.wflow_id}: no observations; drew the hydrograph only.")
        return written
    if metrics is not None:
        written.append(
            plot_performance(simulated, observed, station, plot_dir, metrics, caveat)
        )
    if signatures:
        for kind in ("peaks", "lows"):
            written.append(
                plot_extremes(simulated, observed, station, plot_dir, kind, caveat)
            )
    else:
        log(
            f"wflow_id {station.wflow_id}: record too short for annual statistics; "
            "no peaks/lows sheets."
        )
    return written


__all__ = [
    "Station",
    "annual_extremes",
    "plot_extremes",
    "plot_hydrograph",
    "plot_performance",
    "plot_station_evaluation",
    "r_squared",
    "wettest_and_driest_year",
]
