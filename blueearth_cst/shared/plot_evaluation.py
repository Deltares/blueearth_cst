# -*- coding: utf-8 -*-
"""The four evaluation sheets rule 1.15 draws for each modelled station.

One station, four figures, split by the question a reader is asking:

* ``hydrograph_<wflow_id>`` — what the model produces. Daily series, annual
  means, the mean annual cycle and both flow-duration curves. The ONLY sheet
  that needs no observations, so it is drawn for every station on every run.
* ``signatures_peaks_<wflow_id>`` — the high end. Flood frequency, the
  year-by-year annual maximum against observed, and the full record with its wettest year highlighted.
* ``signatures_lows_<wflow_id>`` — the dry end, in the same three places, so the
  two sheets read against each other without re-learning the layout.
* ``performance_<wflow_id>`` — goodness of fit: the daily scatter, the
  cumulative volume that shows WHEN a bias accrues, and the indicators table.

This replaces ``func_plot_signature.plot_hydro`` (five stacked panels) and
``plot_signatures`` (ten untitled ones). What changed beyond the layout:

* **Keyed by wflow_id.** The old figures were named ``wflow_1`` — a 1..N counter
  invented in ``plot_results`` — or by a station_name the delineation generated.
  Every other artifact of a run names a point by its ``wflow_id``: the
  ``Q_1010`` column in ``output.csv``, the registry, the labels on
  ``elevation_basin``. So do these.
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

# matplotlib, numpy and scipy are DEFERRED into the eight functions that draw.
# `build_model.smk` imports this module at PARSE time for exactly one name --
# STATION_PLOT_DIRNAME, the string below -- and a module-level import therefore
# bought matplotlib + scipy (~4.7s) on every WF1 dry-run and every real run in
# order to learn the word "stations". The constant stays HERE, beside the code
# that draws into the bin, for the reason its own comment gives; it is the
# imports that move.
from blueearth_cst.plotting.figure_titles import write_figure_titles
from blueearth_cst.shared import plot_style
from blueearth_cst.shared.plot_colors import EVALUATION_COLORS, NEUTRAL_COLORS
from blueearth_cst.shared.plot_setup import STYLE_SETTINGS
from blueearth_cst.shared.snake_utils import save_figure

#: The bin every per-station sheet is written into, relative to
#: ``evaluation/plots/``. Defined HERE, beside the code that draws them,
#: because two unrelated readers need the same string: ``plot_results.py``
#: to write into it, and ``build_model.smk`` rule 1.15 to declare it as a
#: ``directory()`` output. A bin rather than flat files because the sheets
#: are named for wflow_ids, a product of the model BUILD -- so their count
#: is unknowable when the Snakefile is parsed, and only a directory can be
#: declared. That is what lets ``--delete-all-output`` reach them
#: (R7-5 / t2608071206); WF0's ``subbasins/`` bin is the same device for
#: the same reason.
STATION_PLOT_DIRNAME = "stations"

# ===========================================================================
# TUNABLE CONSTANTS
# ===========================================================================

#: Expanded original blue for observed, Terracotta for simulated. Preserve
#: dashed/solid distinctions; this palette is not certified for colour vision.
COLOR_OBSERVED = EVALUATION_COLORS["observed"]
COLOR_SIMULATED = EVALUATION_COLORS["simulated"]
#: Marks that belong to neither series — the period mean, a 1:1 line's anchor.
COLOR_NEUTRAL = NEUTRAL_COLORS["evaluation_neutral"]
COLOR_GRID = NEUTRAL_COLORS["evaluation_grid"]
#: The 1:1 reference line on every scatter panel.
COLOR_REFERENCE = NEUTRAL_COLORS["evaluation_reference"]

#: Observed dashes, simulated solid. Simulated is solid because it is the dense
#: series — a dashed line over 7000 daily values reads as a smear.
DASH_OBSERVED = (
    STYLE_SETTINGS["evaluation"]["observed_dash"][0],
    tuple(STYLE_SETTINGS["evaluation"]["observed_dash"][1]),
)
LINE_WIDTH = STYLE_SETTINGS["evaluation"]["line_width"]

#: Sheet heights in millimetres. The width is ``plot_style.FIGURE_WIDTH_MM`` for
#: all four, so they stack in a report at one size.
HEIGHT_HYDROGRAPH_MM = STYLE_SETTINGS["evaluation"]["heights"]["hydrograph"]
HEIGHT_EXTREMES_MM = STYLE_SETTINGS["evaluation"]["heights"]["extremes"]
HEIGHT_PERFORMANCE_MM = STYLE_SETTINGS["evaluation"]["heights"]["performance"]

#: Return periods marked on the frequency panels.
RETURN_PERIODS = (2.0, 5.0, 10.0, 30.0)
#: Gringorten plotting-position constants, as the previous implementation used.
_PLOTTING_POSITION_A = 0.3

#: A log axis cannot show a zero, and this toolbox runs on basins that dry out.
#: The floor is the 1st percentile of the POSITIVE simulated values, never below
#: this, so one 1e-9 cell cannot set the range for the whole curve.
_LOG_FDC_FLOOR = 1e-3

#: Indicator rows on the performance table: the ideal value a reader should compare
#: against, and how many decimals the number is worth. Grouped because "higher
#: is better" and "zero is better" are different readings and a reader should
#: not have to remember which is which per row.
INDICATOR_FORMAT = {
    "KGE": ("1", "{:.3f}"),
    "NSE": ("1", "{:.3f}"),
    "NSElog": ("1", "{:.3f}"),
    "VE": ("1", "{:.3f}"),
    "RMSE": ("0", "{:.2f}"),
    "MSE": ("0", "{:.2f}"),
    "Pbias": ("0 %", "{:+.1f} %"),
}
INDICATOR_GROUPS = (
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
    row and the point on ``elevation_basin``, and it is what the filenames use. The
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


def _figure_size(height_mm):
    return (
        plot_style.FIGURE_WIDTH_MM / plot_style.MM_PER_INCH,
        height_mm / plot_style.MM_PER_INCH,
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


def _draw_fdc(ax, simulated, observed, log_scale):
    """A flow-duration curve, linear or log, in real discharge units."""
    import numpy as np

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
    import numpy as np

    from blueearth_cst.shared import plot_preset as cst

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
        ms=cst.MS_POINT,
        mfc=COLOR_SIMULATED,
        mec=COLOR_NEUTRAL,
        mew=0.5,
        alpha=0.7,
        label="simulated",
    )
    # Reserve headroom for the return-period labels without covering the largest mark.
    bottom, top = ax.get_ylim()
    ax.set_ylim(bottom, top + 0.22 * (top - bottom))
    top = ax.get_ylim()[1]
    for period in RETURN_PERIODS:
        position = -np.log(-np.log(1 - 1.0 / period))
        ax.axvline(position, color=cst.BOUNDARY, lw=0.5, alpha=0.7, zorder=1)
        ax.text(
            position,
            top,
            f"T={period:.0f}y",
            rotation=90,
            va="top",
            ha="right",
            fontsize=cst.FS_NOTE,
            color=cst.MUTED,
            bbox=cst.LABEL_BOX,
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


def _save(fig, plot_dir, stem, annotation="none", record=None):
    """Write one sheet and return its path.

    PNG only since 2026-08-10 (owner's call, applied across the deliverable).
    The PDF was the vector, embedded-font publication copy and nothing read it;
    600 dpi at 180 mm carries the figure everywhere it is used, and not
    serialising each sheet twice halves the rule's render time.

    The metadata scrub stays: the default embeds the matplotlib version, which
    would move a fingerprint on every environment bump.
    """
    import matplotlib.pyplot as plt

    png = os.path.join(str(plot_dir), f"{stem}.png")
    record = record or {}
    save_figure(
        png,
        fig=fig,
        annotation=annotation,
        title=record.get("title", ""),
        caption=record.get("caption", ""),
        dpi=plot_style.RASTER_DPI,
        metadata={"Software": None},
    )
    plt.close(fig)
    return png


# ===========================================================================
# THE FOUR SHEETS
# ===========================================================================


def plot_hydrograph(
    simulated, station, plot_dir, observed=None, caveat=None, *, annotation="none"
):
    """What the model produces: daily, annual, seasonal, and both FDCs.

    The one sheet drawn for EVERY station on every run — every panel is
    computable from the simulation alone, so ``observed`` only adds a second
    line where one exists.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib import gridspec
    from matplotlib.ticker import PercentFormatter

    from blueearth_cst.plotting import fundamentals as artists
    from blueearth_cst.plotting.preset import use_preset
    from blueearth_cst.shared import plot_preset as cst

    prepared = _hydrograph_data(simulated, observed)
    years = prepared["years"]
    annual, cycle = prepared["simulated"]["annual"], prepared["simulated"]["cycle"]
    annual_obs = prepared["observed"]["annual"] if observed is not None else None
    cycle_obs = prepared["observed"]["cycle"] if observed is not None else None

    with use_preset(cst.CST_PRESET):
        fig = plt.figure(
            figsize=_figure_size(HEIGHT_HYDROGRAPH_MM), layout="constrained"
        )
        fig.get_layout_engine().set(
            w_pad=cst.OUTER_PAD_X_MM * artists.MM,
            h_pad=cst.OUTER_PAD_Y_MM * artists.MM,
        )
        grid = gridspec.GridSpec(3, 2, figure=fig, height_ratios=[1.15, 1.0, 1.0])
        ax_daily = fig.add_subplot(grid[0, :])
        ax_annual = fig.add_subplot(grid[1, 0])
        ax_cycle = fig.add_subplot(grid[1, 1])
        ax_fdc = fig.add_subplot(grid[2, 0])
        ax_fdc_log = fig.add_subplot(grid[2, 1])

        _draw_series(ax_daily, prepared["daily_simulated"], prepared["daily_observed"])
        ax_daily.set_title("a) Daily discharge", loc="left", y=1.0)

        _draw_series(
            ax_annual,
            annual,
            annual_obs,
            x_sim=annual["year"],
            x_obs=None if annual_obs is None else annual_obs["year"],
        )
        ax_annual.set_title("b) Annual mean", loc="left", y=1.0)
        tick_step = 10 if years.size > 35 else 5
        first_tick = int(years[0] // tick_step * tick_step)
        major = np.arange(first_tick, years[-1] + 1, tick_step)
        ax_annual.set_xticks(major)
        ax_annual.set_xticks(
            np.setdiff1d(np.arange(first_tick, years[-1] + 1), major), minor=True
        )
        ax_annual.set_xlim(first_tick - 0.5, years[-1] + 0.5)

        _draw_series(
            ax_cycle,
            cycle,
            cycle_obs,
            x_sim=cycle["month"],
            x_obs=None if cycle_obs is None else cycle_obs["month"],
        )
        ax_cycle.set_title("c) Seasonal cycle", loc="left", y=1.0)
        ax_cycle.set_xticks(np.arange(1, 13))
        ax_cycle.set_xticklabels(artists.MONTH_ABBR)

        _hydrograph_fdc(ax_fdc, prepared, log_scale=False)
        ax_fdc.set_title("d) Flow duration", loc="left", y=1.0)
        _hydrograph_fdc(ax_fdc_log, prepared, log_scale=True)
        ax_fdc_log.set_title("e) Flow duration (log)", loc="left", y=1.0)

        for ax in fig.axes:
            ax.set_ylabel("m³ s⁻¹")
            ax.grid(axis="y", alpha=0.7)
        for ax in (ax_fdc, ax_fdc_log):
            ax.set_xlabel("Exceedance probability (%)")
            ax.xaxis.set_major_formatter(
                PercentFormatter(xmax=1, decimals=0, symbol="")
            )

        for ax in (ax_daily, ax_annual, ax_cycle):
            ax.margins(x=0.01)
        handles, _ = ax_daily.get_legend_handles_labels()
        for handle, label in zip(
            handles,
            ("Observed", "Simulated") if observed is not None else ("Simulated",),
        ):
            handle.set_label(label)
        artists.inside_legend(fig, ax_daily, handles, lines=ax_daily.lines)
        # Finalize layout after the adaptive legend chooses its corner.
        fig.canvas.draw()
        record = _hydrograph_caption(prepared, station, plot_dir, caveat)
        filename = _save(
            fig, plot_dir, f"hydrograph_{station.wflow_id}", annotation, record
        )
    return filename


def _hydrograph_data(simulated, observed=None):
    """Regular daily context and complete-period summaries on matched support.

    Discharge is never infilled: eligible annual and monthly means use observed
    days. When observations exist, comparison panels use only concurrent,
    finite, nonnegative pairs within the intersection of the input spans.
    """
    import numpy as np
    import pandas as pd
    import xarray as xr

    from blueearth_cst.climate_analysis import diagnostics as dg
    from blueearth_cst.climate_analysis.diagnostic_settings import DEFAULTS

    tol = dg.Completeness(
        DEFAULTS["max_missing_days_month"], DEFAULTS["max_missing_days_year"]
    )

    def clean(data):
        if data.dims != ("time",) or not data.time.size:
            raise ValueError(
                "Hydrograph discharge must be a nonempty one-dimensional daily time series"
            )
        index = pd.DatetimeIndex(data.time.values)
        if (
            index.hasnans
            or not index.is_unique
            or not index.is_monotonic_increasing
            or not index.equals(index.normalize())
        ):
            raise ValueError(
                "Hydrograph dates must be increasing, unique, midnight daily timestamps"
            )
        values = np.asarray(data.values, dtype=float)
        calendar = pd.date_range(index[0], index[-1], freq="D")
        series = (
            pd.Series(values, index=index)
            .where(np.isfinite(values) & (values >= 0))
            .reindex(calendar)
        )
        return series, {
            "input_period": [str(index[0].date()), str(index[-1].date())],
            "input_days": len(index),
            "omitted_dates": len(calendar) - len(index),
            "negative_values": int((np.isfinite(values) & (values < 0)).sum()),
            "nonfinite_values": int((~np.isfinite(values)).sum()),
        }

    sim, sim_input = clean(simulated)
    obs, obs_input = clean(observed) if observed is not None else (None, None)
    context = pd.date_range(
        min(sim.index[0], obs.index[0]) if obs is not None else sim.index[0],
        max(sim.index[-1], obs.index[-1]) if obs is not None else sim.index[-1],
        freq="D",
    )
    daily_sim = sim.reindex(context)
    daily_obs = obs.reindex(context) if obs is not None else None
    if obs is not None:
        start, end = max(sim.index[0], obs.index[0]), min(sim.index[-1], obs.index[-1])
        concurrent = (
            pd.date_range(start, end, freq="D")
            if start <= end
            else pd.DatetimeIndex([])
        )
        sim_comparison, obs_comparison = (
            sim.reindex(concurrent),
            obs.reindex(concurrent),
        )
        paired = sim_comparison.notna() & obs_comparison.notna()
        sim_comparison, obs_comparison = (
            sim_comparison.where(paired),
            obs_comparison.where(paired),
        )
    else:
        concurrent, sim_comparison, obs_comparison = sim.index, sim, None

    def array(series, dimension):
        return xr.DataArray(
            series.to_numpy(),
            dims=dimension,
            coords={dimension: series.index.to_numpy()},
        )

    def summaries(series, inputs):
        if series.empty:
            annual = pd.Series(dtype=float, index=pd.Index([], dtype=int))
            monthly = pd.Series(dtype=float, index=pd.DatetimeIndex([]))
            full_years = np.array([], dtype=int)
            partial_years = []
            invalid_years = []
            excluded_months = 0
        else:
            full_years = dg.full_years(series, 1)
            partial_years = [
                int(year)
                for year in np.unique(series.index.year)
                if year not in full_years
            ]
            groups = series.resample("MS")
            months = groups.mean().index
            missing = pd.Series(months.days_in_month, index=months) - groups.count()
            full_month = (months >= series.index[0]) & (
                (months + pd.offsets.MonthEnd()) <= series.index[-1]
            )
            monthly = groups.mean().where(full_month & (missing <= tol.max_month))
            invalid_month = missing > tol.max_month
            valid_annual = (~invalid_month).groupby(months.year).all() & (
                missing.groupby(months.year).sum() <= tol.max_year
            )
            annual = series.groupby(series.index.year).mean().reindex(full_years)
            annual = annual.where(valid_annual.reindex(full_years, fill_value=False))
            invalid_years = [int(year) for year in annual.index[annual.isna()]]
            excluded_months = int(monthly.isna().sum())
        cycle = monthly.groupby(monthly.index.month).mean().reindex(range(1, 13))
        ranked = np.sort(series.dropna().to_numpy())[::-1]
        positions = np.arange(1, len(ranked) + 1) / (len(ranked) + 1)
        return {
            "annual": array(annual, "year"),
            "cycle": array(cycle, "month"),
            "fdc": ranked,
            "probability": positions,
            "metadata": {
                **inputs,
                "valid_days": len(ranked),
                "missing_days": len(series) - len(ranked),
                "zero_days": int((ranked == 0).sum()),
                "zero_percent": float((ranked == 0).mean() * 100)
                if len(ranked)
                else None,
                "annual_valid_years": [
                    int(year) for year in annual.index[annual.notna()]
                ],
                "annual_partial_years_excluded": partial_years,
                "annual_invalid_years": invalid_years,
                "monthly_valid_periods": int(monthly.notna().sum()),
                "monthly_excluded_periods": excluded_months,
            },
        }

    return {
        "daily_simulated": array(daily_sim, "time"),
        "daily_observed": array(daily_obs, "time") if daily_obs is not None else None,
        "simulated": summaries(sim_comparison, sim_input),
        "observed": summaries(obs_comparison, obs_input)
        if obs_comparison is not None
        else None,
        "years": np.unique(concurrent.year)
        if len(concurrent)
        else np.unique(context.year),
        "context_period": [str(context[0].date()), str(context[-1].date())],
        "comparison_period": [str(concurrent[0].date()), str(concurrent[-1].date())]
        if len(concurrent)
        else None,
        "calendar_days": len(concurrent),
        "methods": {
            "max_missing_days_month": tol.max_month,
            "max_missing_days_year": tol.max_year,
        },
    }


def _hydrograph_fdc(ax, prepared, log_scale):
    """Finite-only Weibull positions, retaining zero mass on the log panel."""
    import numpy as np

    from blueearth_cst.shared.plot_preset import FS_NOTE, LABEL_BOX, MUTED

    positive = []
    for key, color, style in (
        ("observed", COLOR_OBSERVED, DASH_OBSERVED),
        ("simulated", COLOR_SIMULATED, "-"),
    ):
        result = prepared[key]
        if result is None:
            continue
        values, probability = result["fdc"], result["probability"]
        selected = values > 0 if log_scale else np.ones(len(values), dtype=bool)
        if selected.any():
            ax.plot(
                probability[selected],
                values[selected],
                color=color,
                lw=LINE_WIDTH,
                ls=style,
            )
        positive.extend(values[values > 0].tolist())
    if log_scale:
        ax.set_yscale("log")
        if positive:
            ax.set_ylim(bottom=min(positive))
            zero_labels = []
            for key, label in (("simulated", "sim"), ("observed", "obs")):
                if prepared[key] is not None:
                    share = prepared[key]["metadata"]["zero_percent"]
                    zero_labels.append(
                        f"{label} {share:.1f}%"
                        if share is not None
                        else f"{label} unavailable"
                    )
            ax.text(
                0.96,
                0.96,
                "Zero flow: " + "; ".join(zero_labels),
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=FS_NOTE,
                color=MUTED,
                bbox=LABEL_BOX,
            )
        else:
            ax.set_yticks([])
            ax.text(
                0.5,
                0.5,
                "No positive discharge",
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize=FS_NOTE,
                color=MUTED,
            )
    elif not len(prepared["simulated"]["fdc"]):
        ax.text(
            0.5,
            0.5,
            "No valid discharge",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=FS_NOTE,
            color=MUTED,
        )
    ax.set_xlim(0, 1)


def _hydrograph_caption(prepared, station, plot_dir, caveat):
    """Update the station caption inventory without touching other figure types."""
    import json
    from pathlib import Path

    root = Path(plot_dir)
    name = f"hydrograph_{station.wflow_id}.png"
    start, end = prepared["context_period"]
    observed = prepared["observed"]
    caption = (
        f"Discharge at {station.caption}, {start}–{end}. "
        "a) Daily discharge; b) annual mean; c) mean seasonal cycle; "
        "d) flow-duration curve; e) flow-duration curve on a logarithmic discharge axis. "
        "Discharge is in m³ s⁻¹. The vermillion solid line shows simulated discharge"
        + (
            " and the blue dashed line shows observed discharge. "
            if observed is not None
            else "; observations were not supplied. "
        )
        + "The daily panel retains each series' available context; "
        + (
            "panels b–e use only concurrent finite, nonnegative observed–simulated pairs. "
            if observed is not None
            else "panels b–e use finite, nonnegative simulation values. "
        )
        + "Omitted dates are restored as missing days on a regular daily calendar. "
        + f"Annual means require full calendar-year boundaries, at most {prepared['methods']['max_missing_days_month']} missing days "
        f"in every month and {prepared['methods']['max_missing_days_year']} in the year. Monthly means require full calendar-month "
        "boundaries and the same monthly gap limit. Means use observed days without infilling. "
        "The seasonal cycle gives each eligible monthly mean equal weight across years; a month can be eligible "
        "even when its year is not. FDCs rank valid days with Weibull positions r/(n+1), including zeros. "
        "The log panel shows positive flows at those original positions, retains all positive tails down to "
        "the minimum across both series and reports zero-flow shares separately."
        + (f" {caveat}" if caveat else "")
    )
    details = []
    for key, label in (("simulated", "Simulation"), ("observed", "Observations")):
        if prepared[key] is None:
            continue
        record = prepared[key]["metadata"]
        zero = (
            f"{record['zero_percent']:.2f}%"
            if record["zero_percent"] is not None
            else "unavailable"
        )
        details.append(
            f"{label}: {record['valid_days']} valid comparison days and {record['missing_days']} missing days; "
            f"zero flow {record['zero_days']} days ({zero}); {record['omitted_dates']} omitted input dates; "
            f"{record['negative_values']} negative and {record['nonfinite_values']} nonfinite input values excluded. "
            f"Partial years excluded: {', '.join(map(str, record['annual_partial_years_excluded'])) or 'none'}; "
            f"full years failing completeness: {', '.join(map(str, record['annual_invalid_years'])) or 'none'}. "
            f"Eligible months: {record['monthly_valid_periods']}; excluded months: {record['monthly_excluded_periods']}."
        )
    if prepared["comparison_period"]:
        caption += f" Comparison span: {'–'.join(prepared['comparison_period'])}."
    else:
        caption += " No overlapping comparison span."
    caption += " " + " ".join(details)
    engine = root / "_engine"
    engine.mkdir(parents=True, exist_ok=True)
    path = engine / "figure_captions.json"
    inventory = (
        json.loads(path.read_text(encoding="utf-8"))
        if path.is_file()
        else {"schema_version": 1, "figures": {}}
    )
    inventory["figures"][name] = {
        "title": f"Discharge hydrograph, seasonal cycle and flow duration — {station.caption}; {start}–{end}",
        "caption": caption,
        "sources": ["simulated", "observed"] if observed is not None else ["simulated"],
        "period": [start, end],
        "support": f"wflow_id {station.wflow_id}",
        "encoding": "hydrograph",
        "availability": "available"
        if prepared["simulated"]["metadata"]["valid_days"]
        else "unavailable: no valid comparison discharge",
        "captioned_path": None,
        "comparison_period": prepared["comparison_period"],
        "calendar_days": prepared["calendar_days"],
        "methods": prepared["methods"]
        | {
            "discharge_infill": False,
            "seasonal_weighting": "equal monthly means across years",
            "fdc_positions": "r/(n+1) valid days",
        },
        "series": {
            key: prepared[key]["metadata"]
            for key in ("simulated", "observed")
            if prepared[key] is not None
        },
    }
    write_figure_titles(root, inventory["figures"])
    inventory["produced_paths"] = list(inventory["figures"]) + [
        "figure_titles.md",
        "_engine/figure_titles.json",
        "figure_captions.md",
        "_engine/figure_captions.json",
    ]
    path.write_text(
        json.dumps(inventory, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (root / "figure_captions.md").write_text(
        "# Figure captions\n\n"
        + "\n\n".join(
            f"## {key}\n\n{record['caption']}"
            for key, record in inventory["figures"].items()
        )
        + "\n",
        encoding="utf-8",
    )

    return inventory["figures"][name]


def plot_extremes(
    simulated, observed, station, plot_dir, kind, caveat=None, *, annotation="none"
):
    """One end of the distribution: frequency, year-by-year, and one year.

    ``kind`` is ``"peaks"`` or ``"lows"``. The two sheets are deliberately
    identical in shape — analysis on the top row, the full daily record with
    the selected year highlighted below.

    Both need observations: two of the three panels compare against them.
    """
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt
    import pandas as pd
    from matplotlib import gridspec

    from blueearth_cst.plotting import fundamentals as artists
    from blueearth_cst.plotting.preset import use_preset
    from blueearth_cst.shared import plot_preset as cst

    if kind not in ("peaks", "lows"):
        raise ValueError(f"kind={kind!r}; expected 'peaks' or 'lows'")
    peaks = kind == "peaks"
    sim_max, obs_max, sim_nm7q, obs_nm7q = annual_extremes(simulated, observed)
    sim_extreme, obs_extreme = (sim_max, obs_max) if peaks else (sim_nm7q, obs_nm7q)
    wet, dry = wettest_and_driest_year(simulated)
    year = wet if peaks else dry

    with use_preset(cst.CST_PRESET):
        fig = plt.figure(figsize=_figure_size(HEIGHT_EXTREMES_MM), layout="constrained")
        fig.get_layout_engine().set(
            w_pad=cst.OUTER_PAD_X_MM * artists.MM,
            h_pad=cst.OUTER_PAD_Y_MM * artists.MM,
        )
        grid = gridspec.GridSpec(2, 2, figure=fig, height_ratios=[1.32, 1.0])
        ax_frequency = fig.add_subplot(grid[0, 0])
        ax_scatter = fig.add_subplot(grid[0, 1])
        ax_year = fig.add_subplot(grid[1, :])

        _draw_frequency(
            ax_frequency,
            obs_extreme,
            sim_extreme,
            "m³ s⁻¹",
            ascending=peaks,
        )

        high = float(max(obs_extreme.max(), sim_extreme.max()) * 1.2)
        ax_scatter.plot(
            obs_extreme,
            sim_extreme,
            "o",
            ms=cst.MS_POINT,
            mfc=COLOR_SIMULATED,
            mec=COLOR_NEUTRAL,
            mew=0.5,
            alpha=0.7,
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
                loc="upper right",
                frameon=True,
                facecolor="white",
                edgecolor="none",
                framealpha=0.88,
                handletextpad=0.3,
            )
        _reference_line(ax_scatter, high)
        # Keep zero-flow points fully visible at the frame.
        ax_scatter.set_xlim(-0.025 * high, high)
        ax_scatter.set_ylim(-0.025 * high, high)
        ax_scatter.set_xlabel("Observed (m³ s⁻¹)")
        ax_scatter.set_ylabel("Simulated (m³ s⁻¹)")
        ax_scatter.text(
            0.975,
            0.84 if peaks else 0.975,
            f"R$^2$ = {r_squared(obs_extreme, sim_extreme):.2f}",
            transform=ax_scatter.transAxes,
            ha="right",
            va="top",
            fontsize=cst.FS_NOTE,
            bbox=cst.LABEL_BOX,
        )

        _draw_series(ax_year, simulated, observed)
        ax_year.axvspan(
            pd.Timestamp(year, 1, 1),
            pd.Timestamp(year + 1, 1, 1),
            facecolor=cst.MUTED,
            edgecolor=cst.MUTED,
            alpha=0.16,
            linewidth=0.6,
            label=f"{'Wettest' if peaks else 'Driest'} year ({year})",
            zorder=1,
        )
        ax_year.set_ylabel("m³ s⁻¹")
        start, end = (
            pd.Timestamp(simulated.time.values[0]),
            pd.Timestamp(simulated.time.values[-1]),
        )
        ax_year.set_xlim(start, end)
        span = end.year - start.year
        step = 10 if span > 35 else 5 if span > 12 else 2 if span > 5 else 1
        ax_year.xaxis.set_major_locator(mdates.YearLocator(step))
        ax_year.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax_year.xaxis.set_minor_locator(mdates.YearLocator())
        ax_year.tick_params(axis="x", which="minor", length=2)

        for ax, letter in ((ax_frequency, "a)"), (ax_scatter, "b)"), (ax_year, "c)")):
            ax.grid(axis="y", alpha=0.7)
            ax.annotate(
                letter,
                xy=(0, 1),
                xycoords="axes fraction",
                xytext=(5, -5),
                textcoords="offset points",
                ha="left",
                va="top",
                fontsize=cst.FS_LABEL,
                fontweight="normal",
                bbox=cst.LABEL_BOX,
                zorder=5,
            )
        handles, _ = ax_year.get_legend_handles_labels()
        artists.inside_legend(
            fig,
            ax_year,
            handles=handles,
            lines=ax_year.lines,
        )
        legend = ax_year.get_legend()
        if (
            legend.get_window_extent().x0
            < ax_year.get_window_extent().x0 + ax_year.bbox.width / 2
        ):
            # Leave room for c) only when the less occupied corner is on the left.
            legend.set_bbox_to_anchor((0, 0, 1, 0.88))
        stem = f"signatures_{'peaks' if peaks else 'lows'}_{station.wflow_id}"
        record = _extremes_caption(simulated, station, plot_dir, kind, year, caveat)
        return _save(fig, plot_dir, stem, annotation, record)


def _extremes_caption(simulated, station, plot_dir, kind, year, caveat):
    """Keep signature context and methods in the station caption inventory."""
    import json
    from pathlib import Path

    peaks = kind == "peaks"
    period = [str(value)[:10] for value in simulated.time.values[[0, -1]]]
    extreme = "annual maximum" if peaks else "annual minimum seven-day mean"
    caption = (
        f"{'High' if peaks else 'Low'} flows at {station.caption}, {'–'.join(period)}. "
        f"a) {extreme.capitalize()} discharge on a Gumbel reduced-variate axis; "
        "T marks return periods in years. Blue crosses denote observations and "
        "vermillion circles simulations. "
        f"b) Simulated versus observed {extreme} discharge; the dashed line denotes "
        "1:1 agreement and R² is squared Pearson correlation. "
        + ("The black diamond marks the period mean. " if peaks else "")
        + "c) Daily discharge over the full supplied simulation period; the shaded vertical band "
        f"marks the {'wettest' if peaks else 'driest'} simulated calendar year ({year}), "
        "selected by annual mean discharge. "
        "Observations are blue and dashed; simulations are vermillion and solid. "
        "Annual maxima use September–August water years; seven-day minima use calendar years. "
        "Frequency positions use (rank − 0.3)/(n + 0.4), without fitting an extreme-value distribution. "
        + (caveat or "")
    ).strip()
    root = Path(plot_dir)
    engine = root / "_engine"
    engine.mkdir(parents=True, exist_ok=True)
    path = engine / "figure_captions.json"
    inventory = (
        json.loads(path.read_text(encoding="utf-8"))
        if path.exists()
        else {"schema_version": 1, "figures": {}}
    )
    inventory["figures"][f"signatures_{kind}_{station.wflow_id}.png"] = {
        "title": f"{'High' if peaks else 'Low'}-flow diagnostics — {station.caption}; {'–'.join(period)}",
        "caption": caption,
        "sources": ["simulated", "observed"],
        "period": period,
        "support": f"wflow_id {station.wflow_id}",
        "encoding": f"signatures_{kind}",
        "availability": "available",
        "captioned_path": None,
        "highlighted_year": int(year),
    }
    write_figure_titles(root, inventory["figures"])
    inventory["produced_paths"] = list(inventory["figures"]) + [
        "figure_titles.md",
        "_engine/figure_titles.json",
        "figure_captions.md",
        "_engine/figure_captions.json",
    ]
    path.write_text(
        json.dumps(inventory, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (root / "figure_captions.md").write_text(
        "# Figure captions\n\n"
        + "\n\n".join(
            f"## {name}\n\n{record['caption']}"
            for name, record in inventory["figures"].items()
        )
        + "\n",
        encoding="utf-8",
    )

    return inventory["figures"][f"signatures_{kind}_{station.wflow_id}.png"]


#: Row positions of the indicators table, in axes fractions: the group band, the
#: indicator names, then daily / monthly / ideal.
_TABLE_ROWS = {
    "group": 0.92,
    "header": 0.68,
    "daily": 0.44,
    "monthly": 0.25,
    "ideal": 0.06,
}
#: Where the row labels sit, and where the indicator columns start.
_TABLE_LABEL_X = 0.0
_TABLE_FIRST_X = 0.135


def _draw_indicators_table(ax, indicators):
    """The goodness-of-fit table, laid out as text on a blank axes.

    A blank axes rather than ``ax.table``: matplotlib's table sizes its cells to
    the axes and ignores the figure's type scale, so it comes out at a different
    size from every other label on the page.

    INDICATORS ACROSS, aggregations down. The first version had it the other way —
    seven rows in a narrow column — which was right while the table shared a row
    with the scatter. It sits on a FULL-WIDTH band now, and seven rows of four
    narrow columns across 180 mm is mostly whitespace. Transposed, the same
    numbers fit three short rows, and the daily/monthly pair reads as the
    comparison it is rather than as two columns to track down the page.

    The ``ideal`` row is not decoration. A reader should not have to remember
    whether Pbias wants 0 or 1, and the group band says which direction each
    block is read in.
    """
    import numpy as np

    ax.axis("off")
    names = [
        name
        for _, group in INDICATOR_GROUPS
        for name in group
        if name in indicators.index
    ]
    if not names:
        return
    span = (1.0 - _TABLE_FIRST_X) / len(names)
    centres = {
        name: _TABLE_FIRST_X + span * (position + 0.5)
        for position, name in enumerate(names)
    }

    # Group band: which columns are read "higher is better" and which "zero is".
    for group, members in INDICATOR_GROUPS:
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
            ideal, template = INDICATOR_FORMAT[name]
            ax.text(
                centres[name],
                _TABLE_ROWS[row],
                ideal
                if muted
                else template.format(indicators.loc[name, row])
                if np.isfinite(indicators.loc[name, row])
                else "—",
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


def _performance_pairs(simulated, observed):
    """Regular daily calendar with one shared finite, nonnegative pair mask."""
    import numpy as np
    import pandas as pd

    def clean(data):
        if data.dims != ("time",) or not data.time.size:
            raise ValueError(
                "Performance discharge requires a nonempty one-dimensional daily series"
            )
        dates = pd.DatetimeIndex(data.time.values)
        if (
            dates.hasnans
            or not dates.is_unique
            or not dates.is_monotonic_increasing
            or not dates.equals(dates.normalize())
        ):
            raise ValueError(
                "Performance dates require increasing, unique midnight timestamps"
            )
        values = np.asarray(data.values, dtype=float)
        calendar = pd.date_range(dates[0], dates[-1], freq="D")
        return pd.Series(values, index=dates), {
            "input_period": [str(dates[0].date()), str(dates[-1].date())],
            "input_days": len(dates),
            "omitted_dates": len(calendar) - len(dates),
            "negative_values": int((np.isfinite(values) & (values < 0)).sum()),
            "nonfinite_values": int((~np.isfinite(values)).sum()),
        }

    sim, sim_info = clean(simulated)
    obs, obs_info = clean(observed)
    start, end = max(sim.index[0], obs.index[0]), min(sim.index[-1], obs.index[-1])
    calendar = (
        pd.date_range(start, end, freq="D") if start <= end else pd.DatetimeIndex([])
    )
    pairs = pd.DataFrame(
        {"simulated": sim.reindex(calendar), "observed": obs.reindex(calendar)}
    )
    valid = np.isfinite(pairs).all(axis=1) & (pairs >= 0).all(axis=1)
    pairs = pairs.where(valid, np.nan)
    monthly = pairs.resample("ME").mean().dropna() if len(calendar) else pairs.copy()
    return pairs, {
        "comparison_period": [str(start.date()), str(end.date())]
        if len(calendar)
        else None,
        "calendar_days": len(calendar),
        "paired_days": int(valid.sum()),
        "missing_comparison_days": int((~valid).sum()),
        "paired_months": len(monthly),
        "simulated": sim_info,
        "observed": obs_info,
    }


def _performance_r_squared(pairs):
    """Squared Pearson correlation on pairs, undefined for constant operands."""
    import numpy as np

    valid = pairs.dropna()
    if len(valid) < 2 or (valid.nunique() < 2).any():
        return float("nan")
    return float(np.corrcoef(valid["observed"], valid["simulated"])[0, 1] ** 2)


def plot_performance(
    simulated,
    observed,
    station,
    plot_dir,
    indicators,
    caveat=None,
    *,
    annotation="none",
):
    """Paired-day scatter and cumulative volume, with the supplied CSV indicators."""
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib import gridspec

    from blueearth_cst.plotting import fundamentals as artists
    from blueearth_cst.plotting.preset import use_preset
    from blueearth_cst.shared import plot_preset as cst

    pairs, support = _performance_pairs(simulated, observed)
    finite = pairs.dropna()
    with use_preset(cst.CST_PRESET):
        fig = plt.figure(
            figsize=_figure_size(HEIGHT_PERFORMANCE_MM), layout="constrained"
        )
        fig.get_layout_engine().set(
            w_pad=cst.OUTER_PAD_X_MM * artists.MM,
            h_pad=cst.OUTER_PAD_Y_MM * artists.MM,
        )
        grid = gridspec.GridSpec(2, 2, figure=fig, height_ratios=[2.0, 1.0])
        ax_scatter = fig.add_subplot(grid[0, 0])
        ax_volume = fig.add_subplot(grid[0, 1])
        ax_table = fig.add_subplot(grid[1, :])

        high = 1.04 * (
            max(float(np.ceil(finite.to_numpy().max())), 1.0) if len(finite) else 1.0
        )
        ax_scatter.plot(
            finite["observed"],
            finite["simulated"],
            "o",
            ms=cst.MS_POINT,
            mfc=COLOR_SIMULATED,
            mec="none",
            alpha=0.45,
            ls="none",
            zorder=2,
        )
        _reference_line(ax_scatter, high)
        ax_scatter.set_xlabel("Observed (m³ s⁻¹)")
        ax_scatter.set_ylabel("Simulated (m³ s⁻¹)")
        for ax, label in ((ax_scatter, "a)"), (ax_volume, "b)")):
            ax.text(
                0.025,
                0.975,
                label,
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=cst.FS_LABEL,
                fontweight="bold",
                bbox=cst.LABEL_BOX,
                zorder=5,
            )
        ax_scatter.grid(alpha=0.7)

        # Pandas cumsum resumes after missing dates while retaining NaN gaps.
        # Both curves integrate the exact same valid days; no infilling occurs.
        cumulative = pairs.cumsum() * (SECONDS_PER_DAY / 1e6)
        lines = []
        for key, color, style, label in (
            ("observed", COLOR_OBSERVED, DASH_OBSERVED, "Observed"),
            ("simulated", COLOR_SIMULATED, "-", "Simulated"),
        ):
            (line,) = ax_volume.plot(
                cumulative.index,
                cumulative[key],
                color=color,
                ls=style,
                lw=cst.LW_LINE,
                label=label,
            )
            lines.append(line)
        ax_volume.set_ylabel("10⁶ m³")
        volumes = cumulative.to_numpy()
        volumes = volumes[np.isfinite(volumes)]
        top = float(volumes.max()) if volumes.size else 0.0
        ax_volume.set_ylim(0, top * 1.08 if top > 0 else 1.0)
        ax_volume.set_xlabel("")
        ax_volume.yaxis.get_major_formatter().set_useOffset(False)
        ax_volume.xaxis.set_major_locator(
            mdates.AutoDateLocator(minticks=3, maxticks=6)
        )
        ax_volume.xaxis.set_major_formatter(
            mdates.ConciseDateFormatter(ax_volume.xaxis.get_major_locator())
        )
        if len(pairs) and pairs.index[-1].year - pairs.index[0].year >= 3:
            import pandas as pd

            step = 10 if pairs.index[-1].year - pairs.index[0].year > 35 else 5
            first = pairs.index[0].year // step * step
            ax_volume.set_xlim(pd.Timestamp(first, 1, 1), pairs.index[-1])
            ax_volume.xaxis.set_major_locator(mdates.YearLocator(step))
            ax_volume.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
            ax_volume.xaxis.set_minor_locator(mdates.YearLocator())
        elif not len(pairs):
            from matplotlib.ticker import NullFormatter

            ax_volume.set_xticks([])
            ax_volume.xaxis.set_major_formatter(NullFormatter())
        ax_volume.grid(axis="y", alpha=0.7)
        ax_volume.margins(x=0.01)
        ax_scatter.legend(
            handles=lines,
            loc="upper left",
            bbox_to_anchor=(0.025, 0.89),
            borderaxespad=0,
            ncols=1,
            frameon=True,
            facecolor="white",
            edgecolor="none",
            framealpha=0.88,
            borderpad=0.6,
            labelspacing=0.5,
            handletextpad=0.5,
        )
        if not len(finite):
            for ax in (ax_scatter, ax_volume):
                ax.text(
                    0.5,
                    0.5,
                    "No valid concurrent discharge",
                    transform=ax.transAxes,
                    ha="center",
                    va="center",
                    fontsize=cst.FS_NOTE,
                    color=cst.MUTED,
                )
        _draw_indicators_table(ax_table, indicators)
        for text in ax_table.texts:
            text.set_fontsize(
                cst.FS_TICK
                if text.get_text() in ("EFFICIENCY", "ERROR")
                else cst.FS_LABEL
            )
        score = _performance_r_squared(pairs)
        support["squared_pearson_r"] = float(score) if np.isfinite(score) else None
        record = _performance_caption(support, station, plot_dir, caveat)
        return _save(
            fig, plot_dir, f"performance_{station.wflow_id}", annotation, record
        )


def _performance_caption(support, station, plot_dir, caveat):
    """External performance caption, preserving the other station figures."""
    import json
    from pathlib import Path

    root = Path(plot_dir)
    root.mkdir(parents=True, exist_ok=True)
    period = support["comparison_period"]
    span = "–".join(period) if period else "no overlapping record"
    caption = (
        f"Discharge performance at {station.caption}, {span}. "
        "(a) Simulated versus observed daily discharge; dashed line denotes 1:1 agreement. "
        "(b) Cumulative volume over paired valid days, retaining data gaps. "
        "The table summarises daily and monthly performance; dashes denote undefined values. "
        + (f"{caveat}" if caveat else "")
    )
    engine = root / "_engine"
    engine.mkdir(parents=True, exist_ok=True)
    path = engine / "figure_captions.json"
    inventory = (
        json.loads(path.read_text(encoding="utf-8"))
        if path.exists()
        else {"schema_version": 1, "figures": {}}
    )
    inventory["figures"][f"performance_{station.wflow_id}.png"] = {
        "title": f"Discharge performance — {station.caption}; {span}",
        "caption": caption.strip(),
        "sources": ["simulated", "observed"],
        "period": period,
        "support": f"wflow_id {station.wflow_id}",
        "encoding": "performance",
        "availability": "available"
        if support["paired_days"]
        else "unavailable: no valid concurrent discharge",
        "captioned_path": None,
        "comparison": support,
        "methods": {
            "paired_nonnegative_discharge": True,
            "volume": "paired days × 86400 s",
            "monthly": "available paired-day means",
            "infill": False,
            "correlation": "squared Pearson r",
            "volume_gap_handling": "missing paired days remain gaps; cumulative paired-only totals resume without infilling",
            "undefined_scores": "constant/fewer than two pairs yield undefined correlation; zero observed denominators yield undefined efficiencies",
            "caveat": caveat,
        },
    }
    write_figure_titles(root, inventory["figures"])
    inventory["produced_paths"] = list(inventory["figures"]) + [
        "figure_titles.md",
        "_engine/figure_titles.json",
        "figure_captions.md",
        "_engine/figure_captions.json",
    ]
    path.write_text(
        json.dumps(inventory, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (root / "figure_captions.md").write_text(
        "# Figure captions\n\n"
        + "\n\n".join(
            f"## {name}\n\n{record['caption']}"
            for name, record in inventory["figures"].items()
        )
        + "\n",
        encoding="utf-8",
    )

    return inventory["figures"][f"performance_{station.wflow_id}.png"]


def r_squared(observed, simulated):
    """Coefficient of determination between two aligned series.

    Kept here rather than imported from ``func_plot_signature`` so this module
    depends on nothing that is only there for the retired figures. Non-finite
    pairs are dropped rather than propagated: NM7Q on a basin that dries out
    produces them, and one NaN would blank the whole score.
    """
    import numpy as np
    import scipy.stats as stats

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
    indicators=None,
    signatures=False,
    caveat=None,
    log=print,
    annotation="none",
):
    """Every sheet this station's data supports, in reading order.

    The hydrograph is always drawn. The other three need observations, and the
    two extremes sheets additionally need enough years for annual statistics to
    mean anything — which is what ``signatures`` carries from the caller, so the
    "more than a year of record" rule stays where the record is read.

    A sheet that cannot be drawn is REPORTED, never silently absent: "there are
    no observations" and "the figure failed" look identical in an empty folder.
    """
    written = [
        plot_hydrograph(
            simulated, station, plot_dir, observed, caveat, annotation=annotation
        )
    ]
    if observed is None:
        log(
            f"wflow_id {station.wflow_id}: simulation hydrograph written; observation-based evaluation omitted because no observations are available"
        )
        return written
    if indicators is not None:
        written.append(
            plot_performance(
                simulated,
                observed,
                station,
                plot_dir,
                indicators,
                caveat,
                annotation=annotation,
            )
        )
    if signatures:
        for kind in ("peaks", "lows"):
            written.append(
                plot_extremes(
                    simulated,
                    observed,
                    station,
                    plot_dir,
                    kind,
                    caveat,
                    annotation=annotation,
                )
            )
    else:
        log(
            f"wflow_id {station.wflow_id}: annual peak/low-flow figures omitted because the evaluation record is too short; "
            "other discharge figures are retained."
        )
    return written


__all__ = [
    "STATION_PLOT_DIRNAME",
    "Station",
    "annual_extremes",
    "plot_extremes",
    "plot_hydrograph",
    "plot_performance",
    "plot_station_evaluation",
    "r_squared",
    "wettest_and_driest_year",
]
