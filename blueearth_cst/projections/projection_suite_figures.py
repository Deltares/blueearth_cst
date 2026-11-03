"""WF2 suite artists: plot prepared tables, never recompute climate statistics."""

import matplotlib

matplotlib.use("Agg")
import zlib

import numpy as np
import pandas as pd
from matplotlib.colors import to_rgb
from matplotlib.legend_handler import HandlerTuple
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
from matplotlib.ticker import MaxNLocator, MultipleLocator

from blueearth_cst.plotting.fundamentals import (
    inside_legend,
    new_fig,
    year_ticks,
    ygrid,
)
from blueearth_cst.plotting.preset import use_preset
from blueearth_cst.projections.projection_analysis import (
    MIN_MODELS_IQR,
    MONTH_DAYS,
    SEASON_MONTHS,
)
from blueearth_cst.projections.projection_figures import SCENARIO_COLORS, figure_key
from blueearth_cst.shared.plot_preset import (
    CST_PRESET,
    FS_NOTE,
    FS_TICK,
    H_SINGLE,
    H_STACK2,
    INK,
    LW_RANGE,
    LW_REF,
    MUTED,
    W_FULL,
    W_MID,
)

STRIP_PERIODS = ("annual", "DJF", "MAM", "JJA", "SON")
MONTHS = (
    "Jan",
    "Feb",
    "Mar",
    "Apr",
    "May",
    "Jun",
    "Jul",
    "Aug",
    "Sep",
    "Oct",
    "Nov",
    "Dec",
)


def _color(scenario):
    return SCENARIO_COLORS.get(scenario, MUTED)


def _label(scenario):
    if scenario == "ssp534-over":
        return "SSP5-3.4-OS"
    if scenario == "ssp370-lowNTCF":
        return "SSP3-7.0-LowNTCF"
    return (
        f"SSP{scenario[3]}-{int(scenario[4:]) / 10:.1f}"
        if scenario in SCENARIO_COLORS
        else scenario
    )


def _scenarios(table, order=None):
    found = set(table.scenario.dropna())
    return [s for s in (order or SCENARIO_COLORS) if s in found] + sorted(
        found - set(order or SCENARIO_COLORS)
    )


def _panels(periods, ylabel):
    fig, axes = new_fig(H_SINGLE, ncols=len(periods), width_mm=W_FULL, sharey=True)
    for i, (ax, period) in enumerate(zip(axes.flat, periods, strict=True)):
        ax.text(
            0.02,
            1.02,
            f"{chr(97 + i)})",
            transform=ax.transAxes,
            va="bottom",
            color=INK,
            fontsize=FS_NOTE,
        )
        ax.set_xlabel("Annual" if period == "annual" else period)
        ygrid(ax)
    axes.flat[0].set_ylabel(ylabel)
    return fig, axes.flat


def _legend(fig, scenarios):
    fig.legend(
        handles=[
            Line2D([], [], color=_color(s), marker="o", label=_label(s))
            for s in scenarios
        ],
        loc="outside upper center",
        ncols=max(1, len(scenarios)),
    )


#: The historical-variability band: the mean change the models' own
#: year-to-year variability gives on its own (``noise_halfwidth``).
NOISE_FILL = "#dfe4e7"

#: The Historical group of the seasonal-values grids.
HISTORICAL_COLOR = "#5b6770"


def _dry_cell(xy, width, height, **kw):
    """A hatched mostly-dry area: excluded sign cells, dry panels and months.

    Hatched rather than flat gray, which would read like the palest agreement
    class or the historical-variability band.
    """
    return Rectangle(
        xy,
        width,
        height,
        facecolor="white",
        edgecolor="#cfd6da",
        hatch="///",
        lw=0,
        **kw,
    )


def _mostly_dry(rows):
    """Whether at most half the models in prepared summary ``rows`` remain.

    For precipitation the exclusions are the analysis dry-reference rule in
    practice; the figures shade rather than drop these periods.
    """
    if rows.empty or "n_models_excluded" not in rows:
        return False
    kept = pd.to_numeric(rows.n_models, errors="coerce").sum()
    excluded = pd.to_numeric(rows.n_models_excluded, errors="coerce").sum()
    return excluded > 0 and kept <= excluded


def dry_shading(filename, tables, record, window_id=""):
    """Whether a product shades any mostly-dry period (the artists' own rules)."""
    key = figure_key(filename)
    if "precipitation" not in key and key != "sign-agreement.png":
        return False
    if key == "sign-agreement.png":
        data = tables.get("sign_agreement", pd.DataFrame())
        if data.empty:
            return False
        data = data.loc[
            (data.window_id == window_id)
            & (data.statistic == "mean")
            & (data.variable == "precip")
            & (data.quantity == "change_pct")
        ]
        support = data.groupby("scenario").n.transform("max")
        return bool((data.n <= 0.5 * support).any())
    if key in GRID_PRODUCTS:
        name, variable, quantity, statistic = GRID_PRODUCTS[key]
        if variable != "precip" or not (
            quantity == "change_pct" or statistic in ("sd", "cv")
        ):
            return False
        data = tables.get(SUMMARIES[name], pd.DataFrame())
        periods, by = SEASONS, ["period"]
    elif key.startswith("monthly-change-"):
        data = tables.get("change_summary", pd.DataFrame())
        quantity, statistic, periods, by = (
            "change_pct",
            "mean",
            MONTHS,
            ["period", "scenario"],
        )
    else:
        return False
    if data.empty:
        return False
    data = data.loc[
        (data.window_id == window_id)
        & (data.variable == "precip")
        & (data.quantity == quantity)
        & (data.statistic == statistic)
        & data.period.isin(periods)
    ]
    source = GRID_PRODUCTS[key][0] if key in GRID_PRODUCTS else "model_changes"
    if source in tables:  # as drawn
        data = data.loc[
            data.scenario.isin(
                grid_support(filename, tables, window_id).get("models", {})
            )
        ]
    return any(_mostly_dry(rows) for _, rows in data.groupby(by))


SEASONS = tuple(SEASON_MONTHS)


def reference_amounts(climatology, scenarios=None):
    """Reference precipitation per period (mm), from the prepared climatology.

    The ensemble-median monthly reference rate times each month's days, summed
    over the period: the seasonal-cycle figure's historical line as an amount,
    so a percentage change can be read against how much falls. Uses the first
    displayed scenario's models, as the seasonal-cycle figure does.
    """
    data = climatology.loc[
        (climatology.variable == "precip") & (climatology.side == "reference")
    ]
    if data.empty:
        return {}
    rates = (
        data.loc[data.scenario == _scenarios(data, scenarios)[0]]
        .set_index("period")["median"]
        .reindex(MONTHS)
    )
    return {
        period: sum(rates[m] * MONTH_DAYS[m] for m in months)
        for period, months in {"annual": MONTHS, **SEASON_MONTHS}.items()
    }


def shown_scenarios(table, scenarios=None):
    """Scenarios with at least ``MIN_MODELS_IQR`` models with a value, if any.

    A handful of points reads as a distribution it is not, so sparser
    scenarios are left out of the seasonal grids; captions name them.
    Returns (shown, dropped) with dropped as {scenario: model count}.
    """
    support = (
        table.loc[np.isfinite(table.value.astype(float))]
        .groupby("scenario")
        .model.nunique()
    )
    order = [s for s in _scenarios(table, scenarios) if s != "historical"]
    shown = [s for s in order if support.get(s, 0) >= MIN_MODELS_IQR]
    # A small ensemble where no scenario reaches the minimum keeps them all:
    # the rule removes a sparse scenario beside fuller ones, not the figure.
    if not shown:
        shown = [s for s in order if support.get(s, 0)]
    return shown, {s: int(support.get(s, 0)) for s in order if s not in shown}


#: Seasonal-grid products: (model table, variable, quantity, statistic). The
#: summary table is the model table's summary twin.
GRID_PRODUCTS = {
    "change-temperature.png": ("model_changes", "temp", "change", "mean"),
    "change-precipitation-pct.png": ("model_changes", "precip", "change_pct", "mean"),
    "values-precipitation-total.png": ("seasonal_values", "precip", "value", "total"),
    "values-temperature-mean.png": ("seasonal_values", "temp", "value", "mean"),
    "values-precipitation-sd.png": ("seasonal_values", "precip", "value", "sd"),
    "values-temperature-sd.png": ("seasonal_values", "temp", "value", "sd"),
    "values-precipitation-cv.png": ("seasonal_values", "precip", "value", "cv"),
}
#: Monthly-change products: (variable, quantity) of the model changes.
MONTHLY_PRODUCTS = {
    "monthly-change-temperature.png": ("temp", "change"),
    "monthly-change-precipitation.png": ("precip", "change_pct"),
}
SUMMARIES = {
    "model_changes": "change_summary",
    "seasonal_values": "seasonal_values_summary",
}
#: Y-axis unit per (variable, quantity, statistic); the title says the rest.
GRID_UNITS = {
    ("temp", "change", "mean"): "°C",
    ("precip", "change_pct", "mean"): "%",
    ("precip", "value", "total"): "mm (season total)",
    ("temp", "value", "mean"): "°C",
    ("precip", "value", "sd"): "mm (season total)",
    ("temp", "value", "sd"): "°C",
    ("precip", "value", "cv"): "%",
}


def grid_support(filename, tables, window_id="", scenarios=None):
    """Models per shown group (min and max over seasons) and dropped scenarios.

    For the seasonal-grid captions, which report model counts instead of
    printing them in the panels. Empty for other products.
    """
    key = figure_key(filename)
    if key in GRID_PRODUCTS:
        name, variable, quantity, statistic = GRID_PRODUCTS[key]
        periods = SEASONS
    elif key in MONTHLY_PRODUCTS:
        variable, quantity = MONTHLY_PRODUCTS[key]
        name, statistic, periods = "model_changes", "mean", MONTHS
    else:
        return {}
    data = tables[name]
    data = data.loc[
        (data.variable == variable)
        & (data.quantity == quantity)
        & (data.statistic == statistic)
        & data.period.isin(periods)
    ]
    if window_id and "window_id" in data:
        data = data.loc[data.window_id == window_id]
    shown, dropped = shown_scenarios(data, scenarios)
    finite = data.loc[np.isfinite(data.value.astype(float))]
    counts = finite.groupby(["scenario", "period"]).model.nunique()
    groups = (["historical"] if quantity == "value" else []) + shown
    return {
        "models": {
            s: [int(counts[s].min()), int(counts[s].max())]  # JSON-stable
            for s in groups
            if s in counts.index.get_level_values(0)
        },
        "dropped": dropped,
    }


def _round_axis(ax, low, high, floor=None):
    """Y limits and ticks on 1-2-5 steps, at most 8 labels, with a margin.

    A limit moves one step further out when the data sit within 3% of the
    range from it, so points stay clear of the frame; ``floor`` pins the lower
    end (0 for amounts) with a sliver below it.
    """
    ticks = MaxNLocator(nbins=7, steps=[1, 2, 5, 10]).tick_values(low, high)
    step = ticks[1] - ticks[0]
    top, bottom = ticks[ticks >= high].min(), ticks[ticks <= low].max()
    if top - high < 0.03 * (top - bottom):
        top += step
    if floor is not None:
        bottom = floor - 0.03 * (top - floor)
    elif low - bottom < 0.03 * (top - bottom):
        bottom -= step
    ax.yaxis.set_major_locator(MultipleLocator(step))
    ax.set_ylim(bottom, top)


def _group(ax, x, values, color, row, seed):
    """Model points, the prepared median tick and IQR band of one group."""
    values = np.sort(values[np.isfinite(values)])
    rng = np.random.default_rng(zlib.crc32(seed.encode()))
    if row is not None and row.iqr_shown:
        ax.add_patch(
            Rectangle(
                (x - 0.3, row.q25),
                0.6,
                row.q75 - row.q25,
                facecolor=color,
                alpha=0.22,
                lw=0,
                zorder=2,
            )
        )
    ax.scatter(
        x + rng.uniform(-0.16, 0.16, len(values)),
        values,
        color=tuple(0.8 * c for c in to_rgb(color)),
        edgecolors="white",
        linewidths=0.3,
        s=10,
        alpha=0.7,
        zorder=4,
    )
    if row is not None and np.isfinite(row["median"]):
        ax.plot(
            [x - 0.3, x + 0.3],
            [row["median"]] * 2,
            color=color,
            lw=1.8,
            solid_capstyle="butt",
            zorder=5,
        )


def season_grid(
    models,
    summary,
    *,
    variable,
    quantity,
    statistic="mean",
    scenarios=None,
    amounts=None,
):
    """Seasonal 2x2 grid of model points, median and IQR per scenario.

    Changes (``quantity`` "change"/"change_pct") share one y-axis across
    seasons, with the historical-variability band. Seasonal values
    (``"value"``: total, mean, sd, cv) add a Historical group and give each
    season its own axis, so dry seasons are not squeezed. Precipitation %
    changes, SD and CV hatch mostly-dry seasons. The y-axis shows the unit
    only; periods go in the title and caption.
    """
    pick = lambda t: t.loc[  # noqa: E731
        (t.variable == variable)
        & (t.quantity == quantity)
        & t.period.isin(SEASONS)
        & (t.statistic == statistic)
    ]
    models, sums = pick(models), pick(summary)
    scen, _ = shown_scenarios(models, scenarios)
    totals = quantity == "value"
    groups = (["historical"] if totals else []) + scen
    unit = GRID_UNITS[(variable, quantity, statistic)]
    # Amounts, spreads and CV start at zero; mean temperature does not.
    floor = 0 if totals and statistic != "mean" else None
    fig, grid = new_fig(2 * 65, nrows=2, ncols=2, width_mm=W_FULL, sharey=not totals)
    values = models.loc[models.scenario.isin(groups), "value"].to_numpy(float)
    values = np.r_[0, values[np.isfinite(values)]]
    for i, (ax, season) in enumerate(zip(grid.flat, SEASONS, strict=True)):
        panel = sums.loc[sums.period == season]
        title = f"{chr(97 + i)}) {season}"
        if amounts and np.isfinite(amounts.get(season, np.nan)):
            title += f" · ref. ≈{amounts[season]:.0f} mm"
        dry = (
            variable == "precip"
            and (quantity == "change_pct" or statistic in ("sd", "cv"))
            and _mostly_dry(panel.loc[panel.scenario.isin(scen)])
        )
        if dry:
            ax.add_patch(_dry_cell((0, 0), 1, 1, transform=ax.transAxes, zorder=0))
            title += " · dry"
        ax.set_title(title, loc="left")
        if not scen:
            ax.text(
                0.5,
                0.5,
                "No eligible\nmodel estimates",
                transform=ax.transAxes,
                ha="center",
                va="center",
                color=MUTED,
                fontsize=FS_NOTE,
            )
            ax.set_xticks([])
            if i % 2 == 0:
                ax.set_ylabel(unit)
            continue
        if totals:
            local = models.loc[
                (models.period == season) & models.scenario.isin(groups), "value"
            ].to_numpy(float)
            local = local[np.isfinite(local)]
            if not len(local):
                local = np.array([0.0, 1.0])
            low = 0 if floor is not None else local.min()
            _round_axis(ax, low, local.max(), floor=floor)
        else:
            _round_axis(ax, values.min(), values.max())
            if not dry and "noise_halfwidth" in panel:
                half = panel.noise_halfwidth.astype(float).median()
                if np.isfinite(half):
                    ax.axhspan(-half, half, color=NOISE_FILL, lw=0, zorder=0.5)
        ygrid(ax)
        if floor is not None or not totals:
            ax.axhline(0, color=MUTED, lw=LW_REF)
        # The Historical group sits left of a divider, set back from the
        # scenarios so its label does not crowd theirs.
        first = int(totals)
        xs = ([-0.35] if totals else []) + [first + j for j in range(len(scen))]
        for x, s in zip(xs, groups, strict=True):
            rows = panel.loc[panel.scenario == s]
            _group(
                ax,
                x,
                models.loc[
                    (models.period == season) & (models.scenario == s), "value"
                ].to_numpy(float),
                HISTORICAL_COLOR if s == "historical" else _color(s),
                rows.iloc[0] if len(rows) else None,
                f"{season}:{s}",
            )
        if totals:
            ax.axvline(0.4, color=MUTED, lw=0.5, ls=":")
        ax.set_xticks(
            xs, ["Historical" if s == "historical" else _label(s) for s in groups]
        )
        ax.tick_params(axis="x", length=0, pad=2)
        ax.set_xlim(xs[0] - 0.65, xs[-1] + 0.6)
        if i % 2 == 0:
            ax.set_ylabel(unit)
    return fig


def seasonal_figure(climatology, changes, summary, *, variable, scenarios=None):
    """Monthly median climatologies, with only the historical model range shaded."""
    data = climatology.loc[climatology.variable == variable]
    fig, axes = new_fig(H_SINGLE, width_mm=W_MID)
    ax = axes[0, 0]
    scen = _scenarios(data, scenarios)
    unit = "°C" if variable == "temp" else "mm d⁻¹"
    historical = (
        data.loc[(data.scenario == scen[0]) & (data.side == "reference")]
        .set_index("period")
        .reindex(MONTHS)
    )
    ax.fill_between(
        range(12), historical["min"], historical["max"], color=MUTED, alpha=0.18, lw=0
    )
    ax.plot(range(12), historical["median"], color=INK, lw=1.5)
    handles = [Line2D([], [], color=INK, lw=1.5, label="Historical")]
    for s in scen:
        future = (
            data.loc[(data.scenario == s) & (data.side == "future")]
            .set_index("period")
            .reindex(MONTHS)
        )
        ax.plot(range(12), future["median"], color=_color(s), lw=1.0)
        handles.append(Line2D([], [], color=_color(s), lw=1.0, label=_label(s)))
    ax.set_xticks(range(12), MONTHS)
    ax.set_xlim(-0.4, 11.4)
    ax.set_ylabel(unit)
    ax.tick_params(axis="both", length=3)
    if variable == "precip":
        ax.set_ylim(bottom=0)
    ygrid(ax)
    # Above the axes: the historical range spans both top corners, so an
    # inside legend would cover it.
    fig.legend(handles=handles, loc="outside upper center", ncols=len(handles))
    return fig


def _tint(color, amount):
    """``color`` mixed towards white by ``amount`` (0 keeps it, 1 is white)."""
    return tuple(c + (1 - c) * amount for c in to_rgb(color))


def monthly_change_figure(models, summary, *, variable, scenarios=None):
    """Monthly box plots of model changes, one panel per scenario in two columns.

    Tinted IQR boxes with a median bar and light whiskers to the inter-model
    range, over a continuous historical-variability band. Scenarios with fewer
    than ``MIN_MODELS_IQR`` models are left out, as on the seasonal grids.
    """
    quantity = "change_pct" if variable == "precip" else "change"
    pick = lambda t: t.loc[  # noqa: E731
        (t.variable == variable)
        & (t.statistic == "mean")
        & (t.quantity == quantity)
        & t.period.isin(MONTHS)
    ]
    data = pick(summary)
    scen, _ = shown_scenarios(pick(models), scenarios)
    ncols = min(2, max(1, len(scen)))
    nrows = max(1, (len(scen) + ncols - 1) // ncols)
    fig, axes = new_fig(
        65 * nrows, nrows=nrows, ncols=ncols, width_mm=W_FULL, sharey=True
    )
    shown = data.loc[data.scenario.isin(scen)]
    bounds = np.r_[
        0,
        shown[["min", "max"]].to_numpy(float).ravel(),
        shown.get("noise_halfwidth", pd.Series(dtype=float)).to_numpy(float),
    ]
    bounds = bounds[np.isfinite(bounds)]
    for j, s in enumerate(scen):
        ax = axes.flat[j]
        color = _color(s)
        rows = data.loc[data.scenario == s].set_index("period").reindex(MONTHS)
        for x, (_, row) in enumerate(rows.iterrows()):
            if variable == "precip" and _mostly_dry(rows.iloc[[x]]):
                ax.add_patch(
                    _dry_cell(
                        (x - 0.5, 0), 1, 1, transform=ax.get_xaxis_transform(), zorder=0
                    )
                )
            # Full-width month cells join into one stepped band.
            elif np.isfinite(half := float(row.get("noise_halfwidth", np.nan))):
                ax.add_patch(
                    Rectangle(
                        (x - 0.5, -half),
                        1,
                        2 * half,
                        color=NOISE_FILL,
                        lw=0,
                        zorder=0.5,
                    )
                )
            if not np.isfinite(row["median"]):
                continue
            ax.plot(
                [x] * 2,
                [row["min"], row["max"]],
                color=_tint(color, 0.35),
                lw=LW_RANGE,
                solid_capstyle="butt",
                zorder=2,
            )
            if row.iqr_shown:
                ax.add_patch(
                    Rectangle(
                        (x - 0.3, row.q25),
                        0.6,
                        row.q75 - row.q25,
                        facecolor=_tint(color, 0.7),
                        edgecolor=color,
                        lw=0.6,
                        zorder=3,
                    )
                )
            ax.plot(
                [x - 0.3, x + 0.3],
                [row["median"]] * 2,
                color=tuple(0.8 * c for c in to_rgb(color)),
                lw=1.6,
                solid_capstyle="butt",
                zorder=4,
            )
        ax.set_title(f"{chr(97 + j)}) {_label(s)}", loc="left")
        ax.set_xticks(range(12), MONTHS)
        ax.set_xlim(-0.6, 11.6)
        ax.tick_params(axis="x", length=0, pad=2)
        ax.axhline(0, color=MUTED, lw=LW_REF, zorder=1)
        if len(bounds):
            _round_axis(ax, bounds.min(), bounds.max())
        if j % ncols == 0:
            ax.set_ylabel("%" if variable == "precip" else "°C")
        ygrid(ax)
    if not scen:
        axes.flat[0].text(
            0.5,
            0.5,
            "No eligible\nmodel estimates",
            transform=axes.flat[0].transAxes,
            ha="center",
            va="center",
            color=MUTED,
            fontsize=FS_NOTE,
        )
    for ax in list(axes.flat)[max(1, len(scen)) :]:
        ax.set_visible(False)
    return fig


def ipcc_trajectory(
    envelope,
    *,
    variable,
    scenarios=None,
    members=None,
    windows=(),
    reference=(1985, 2014),
):
    """Prepared annual model medians and ranges, with distinct historical traces."""
    data = envelope.loc[envelope.variable == variable]
    fig, axes = new_fig(H_SINGLE, width_mm=W_MID)
    ax = axes[0, 0]
    scen = _scenarios(data, scenarios)
    handles = []
    if members is not None:
        traces = members.loc[members.variable == variable]
        historical = traces.loc[traces.year <= 2014].drop_duplicates(
            ["model", "member", "year"]
        )
        historical_models = historical.groupby(["model", "year"]).anomaly.mean()
        historical_range = historical_models.groupby("year").agg(
            ["median", "min", "max"]
        )
        years = np.arange(int(data.year.min()), 2015)
        historical_range = historical_range.reindex(years)
        ax.fill_between(
            years,
            historical_range["min"],
            historical_range["max"],
            color=MUTED,
            alpha=0.12,
            lw=0,
        )
        ax.plot(years, historical_range["median"], color=INK, lw=1.8)
        handles.append(Line2D([], [], color=INK, lw=1.4, label="Historical"))
        for _, member in historical.groupby(["model", "member"]):
            member = member.set_index("year").sort_index()
            years = np.arange(member.index.min(), member.index.max() + 1)
            ax.plot(
                years,
                member.reindex(years).anomaly,
                color=MUTED,
                lw=0.5,
                alpha=0.13,
                zorder=0.5,
            )
        for (scenario, _, _), member in traces.loc[traces.year >= 2014].groupby(
            ["scenario", "model", "member"]
        ):
            member = member.set_index("year").sort_index()
            years = np.arange(member.index.min(), member.index.max() + 1)
            ax.plot(
                years,
                member.reindex(years).anomaly,
                color=_color(scenario),
                lw=0.5,
                alpha=0.13,
                zorder=0.5,
            )
    for s in scen:
        sub = data.loc[data.scenario == s].sort_values("year")
        if members is not None:
            sub = sub.loc[sub.year >= 2014]
        if sub.empty:
            continue
        years = np.arange(sub.year.min(), sub.year.max() + 1)
        sub = sub.set_index("year").reindex(years)
        ax.plot(years, sub["median"], color=_color(s), lw=1.5)
        handles.append(Line2D([], [], color=_color(s), lw=1.4, label=_label(s)))
        ax.fill_between(
            years,
            sub["min"].astype(float),
            sub["max"].astype(float),
            color=_color(s),
            alpha=0.12,
            lw=0,
        )
    for window in windows:
        a, b = window["used"]
        ax.axvspan(a - 0.5, b + 0.5, color="#d9d6d4", lw=0, zorder=0)
    ax.axvline(2014.5, color=MUTED, lw=LW_REF, ls="--", gid="scenario-handover")
    ax.axhline(0, color=MUTED, lw=LW_REF)
    ax.set(
        xlabel="Year",
        ylabel=f"Relative to {reference[0]}–{reference[1]} ({'%' if variable == 'precip' else '°C'})",
    )
    year_ticks(ax, np.arange(int(data.year.min()), int(data.year.max()) + 1))
    if data.year.max() - data.year.min() > 100:
        ax.set_xticks(
            np.arange(
                int(data.year.min()) // 20 * 20 + 20, int(data.year.max()) + 1, 20
            )
        )
        ax.set_xticks(
            np.arange(int(data.year.min()) // 5 * 5, int(data.year.max()) + 1, 5),
            minor=True,
        )
        ax.set_xlim(data.year.min() - 0.5, data.year.max() + 0.5)
    ax.tick_params(axis="y", length=3)
    ygrid(ax)
    inside_legend(
        fig,
        ax,
        handles=handles,
        bands=ax.collections,
        lines=ax.lines,
    )
    return fig


def annual_overview(
    members, *, variable, scenarios=None, reference=(1985, 2014), windows=()
):
    """Unsmoothed absolute values and anomalies; scenarios wrap in column pairs.

    Gray historical traces are deduplicated across scenarios, with their black
    member median. Scenario envelopes are member ranges and medians, never
    claimed to be equal-model reductions. Missing years remain gaps.
    """
    data = members.loc[members.variable == variable]
    scen = _scenarios(data, scenarios)
    ncols = min(2, len(scen))
    blocks = (len(scen) + ncols - 1) // ncols
    fig, axes = new_fig(
        H_STACK2 * blocks,
        nrows=2 * blocks,
        ncols=ncols,
        width_mm=W_FULL,
        sharex=True,
        sharey="row",
    )
    for block in range(1, blocks):
        for r in range(2):
            axes[2 * block + r, 0].sharey(axes[r, 0])
    for j in range(len(scen), blocks * ncols):
        for r in range(2):
            axes[2 * (j // ncols) + r, j % ncols].set_visible(False)
    hist = data.loc[data.year <= 2014].drop_duplicates(["model", "member", "year"])
    future = data.loc[data.year >= 2014]
    historical_alpha = 0.13
    labels = (
        "mm yr⁻¹" if variable == "precip" else "°C",
        f"Relative to {reference[0]}–{reference[1]} ({'%' if variable == 'precip' else '°C'})",
    )
    for r, column in enumerate(("value", "anomaly")):
        for j, s in enumerate(scen):
            ax = axes[2 * (j // ncols) + r, j % ncols]
            for _, member in hist.groupby(["model", "member"]):
                member = member.set_index("year").sort_index()
                years = np.arange(member.index.min(), member.index.max() + 1)
                ax.plot(
                    years,
                    member.reindex(years)[column],
                    color=MUTED,
                    lw=0.5,
                    alpha=historical_alpha,
                )
            median = hist.groupby("year")[column].median()
            if len(median):
                years = np.arange(median.index.min(), median.index.max() + 1)
                ax.plot(
                    years,
                    median.reindex(years),
                    color="black",
                    lw=1.2,
                    gid="historical-median",
                )
            sub = future.loc[future.scenario == s]
            member_alpha = 0.13
            for _, member in sub.groupby(["model", "member"]):
                member = member.set_index("year").sort_index()
                years = np.arange(member.index.min(), member.index.max() + 1)
                ax.plot(
                    years,
                    member.reindex(years)[column],
                    color=_color(s),
                    alpha=member_alpha,
                    lw=0.25,
                )
            env = sub.groupby("year")[column].agg(["min", "max", "median"])
            if not env.empty:
                years = np.arange(env.index.min(), env.index.max() + 1)
                env = env.reindex(years)
                ax.fill_between(
                    years, env["min"], env["max"], color=_color(s), alpha=0.25, lw=0
                )
                ax.plot(years, env["median"], color=_color(s), lw=1.0)
            ax.axvline(2014.5, color=MUTED, lw=LW_REF, ls="--", gid="scenario-handover")
            for window in windows:
                a, b = window["used"]
                ax.axvspan(
                    a - 0.5,
                    b + 0.5,
                    color="#d9d6d4",
                    zorder=0,
                    gid=f"window:{window['window_id']}",
                )
            # Letters run in reading order over visible panels only, so a
            # short last block does not skip the hidden slot's letters.
            block, width = j // ncols, min(ncols, len(scen) - j // ncols * ncols)
            letter = chr(97 + 2 * ncols * block + r * width + j % ncols)
            ax.text(
                0.0,
                1.02,
                f"{letter}) {_label(s)} · {'Anomaly' if r else 'Annual'}",
                transform=ax.transAxes,
                ha="left",
                va="bottom",
                fontsize=FS_NOTE,
            )
            if r:
                ax.axhline(0, color=MUTED, lw=LW_REF)
                ax.set_xlabel("Year")
            year_ticks(ax, np.arange(int(data.year.min()), int(data.year.max()) + 1))
            if data.year.max() - data.year.min() > 100:
                ax.set_xticks(
                    np.arange(
                        int(data.year.min()) // 20 * 20 + 20,
                        int(data.year.max()) + 1,
                        20,
                    )
                )
                ax.set_xticks(
                    np.arange(
                        int(data.year.min()) // 5 * 5, int(data.year.max()) + 1, 5
                    ),
                    minor=True,
                )
                ax.set_xlim(data.year.min() - 0.5, data.year.max() + 0.5)
            ax.tick_params(axis="y", length=3)
            ax.tick_params(axis="x", labelbottom=bool(r))
            if j % ncols == 0:
                ax.set_ylabel(labels[r])
            ygrid(ax)
    return fig


def pooled_ratio_figure(members, summary, scenarios=None):
    """Member points with prepared geometric-member, equal-model summaries."""
    indicators = (
        ("temp", "std_detrended", "Temp. SD"),
        ("precip", "std_detrended", "Precip. SD"),
        ("precip", "cv_detrended", "Precip. CV"),
    )
    data = members.loc[(members.period == "annual") & np.isfinite(members.ratio)]
    if "variability_eligible" in data:
        data = data.loc[data.variability_eligible]
    scen = _scenarios(data, scenarios)
    fig, axes = new_fig(H_SINGLE, ncols=3, width_mm=W_FULL, sharey=True)
    shown = data.loc[
        ((data.variable == "temp") & (data.statistic == "std_detrended"))
        | (
            (data.variable == "precip")
            & data.statistic.isin(["std_detrended", "cv_detrended"])
        )
    ]
    values = np.r_[1.0, shown.ratio.to_numpy(float)]
    low, high = values.min(), values.max()
    pad = 0.06 * (high - low or 1)
    for i, (ax, (variable, statistic, title)) in enumerate(
        zip(axes.flat, indicators, strict=True)
    ):
        ax.set_title(f"{chr(97 + i)}) {title}", loc="left")
        ax.set_ylim(low - pad, high + pad)
        ax.axhline(1, color=MUTED, lw=LW_REF)
        ax.tick_params(axis="both", length=3)
        ygrid(ax)
        for j, scenario in enumerate(scen):
            color = _color(scenario)
            vals = np.sort(
                data.loc[
                    (data.variable == variable)
                    & (data.statistic == statistic)
                    & (data.scenario == scenario),
                    "ratio",
                ].to_numpy(float)
            )
            rng = np.random.default_rng(
                zlib.crc32(f"pooled:{variable}:{statistic}:{scenario}".encode())
            )
            jitter = rng.choice([-1.0, 1.0], len(vals)) * rng.uniform(
                0.09, 0.2, len(vals)
            )
            ax.scatter(
                j + jitter,
                vals,
                s=10,
                color=tuple(0.65 * c for c in to_rgb(color)),
                edgecolor="white",
                linewidth=0.4,
                alpha=0.4,
                zorder=5,
            )
            rows = summary.loc[
                (summary.period == "annual")
                & (summary.variable == variable)
                & (summary.statistic == statistic)
                & (summary.scenario == scenario)
            ]
            if rows.empty:
                continue
            row = rows.iloc[0]
            ax.plot(
                [j, j], [row["min"], row["max"]], color=color, lw=LW_RANGE, zorder=2
            )
            for end in (row["min"], row["max"]):
                ax.plot(
                    [j - 0.07, j + 0.07],
                    [end, end],
                    color=color,
                    lw=LW_RANGE,
                    solid_capstyle="butt",
                    zorder=2,
                )
            if row.iqr_shown:
                ax.add_patch(
                    Rectangle(
                        (j - 0.25, row.q25),
                        0.5,
                        row.q75 - row.q25,
                        facecolor="white",
                        edgecolor=color,
                        lw=0.9,
                        zorder=3,
                    )
                )
            ax.plot(
                [j - 0.25, j + 0.25], [row["median"]] * 2, color=color, lw=1.5, zorder=4
            )
        ax.set_xticks(range(len(scen)), [""] * len(scen))
        ax.set_xlim(-0.6, len(scen) - 0.4)
    axes.flat[0].set_ylabel("Future / reference")
    legend = fig.legend(
        handles=[Line2D([], [], color=_color(s), label=_label(s)) for s in scen],
        loc="upper right",
        bbox_to_anchor=(0.98, 0.99),
        ncols=max(1, len(scen)),
        frameon=False,
        borderaxespad=0,
    )
    legend.set_in_layout(False)
    fig.get_layout_engine().set(rect=(0, 0, 1, 0.94))
    return fig


def change_cloud(models, *, combined=False, scenarios=None):
    """Paired model mean changes; scenario colors share a circular marker."""
    data = models.loc[(models.period == "annual") & (models.statistic == "mean")]
    keys = ["model", "scenario", "window_id"]
    temp = data.loc[
        (data.variable == "temp") & (data.quantity == "change"), [*keys, "value"]
    ]
    precip = data.loc[
        (data.variable == "precip") & (data.quantity == "change_pct"), [*keys, "value"]
    ]
    paired = temp.merge(precip, on=keys, suffixes=("_temp", "_precip")).dropna()
    fig, axes = new_fig(120, width_mm=120)
    ax = axes[0, 0]
    scen = _scenarios(paired, scenarios)
    handles = []
    for s in scen:
        sub = paired.loc[paired.scenario == s]
        color = _color(s)
        edge = tuple(0.6 * c for c in to_rgb(color))
        ax.scatter(
            sub.value_precip,
            sub.value_temp,
            color=color,
            s=28,
            alpha=0.75,
            marker="o",
            edgecolor=edge,
            linewidth=0.55,
            zorder=3,
        )
        handles.append(
            Line2D(
                [],
                [],
                linestyle="",
                marker="o",
                markersize=4,
                markerfacecolor=color,
                markeredgecolor=edge,
                alpha=0.75,
                label=_label(s),
            )
        )
        if combined:
            for _, group in sub.groupby("model"):
                group = group.sort_values("window_id")
                ax.plot(
                    group.value_precip,
                    group.value_temp,
                    color=color,
                    alpha=0.25,
                    lw=0.6,
                )
    p = paired.value_precip.to_numpy(float)
    t = paired.value_temp.to_numpy(float)
    pad_x = max((p.max() - p.min()) * 0.12, 1.0)
    pad_y = max((t.max() - t.min()) * 0.12, 0.2)
    ax.set(
        xlim=(min(p.min(), 0) - pad_x, max(p.max(), 0) + pad_x),
        ylim=(min(t.min(), 0) - pad_y, max(t.max(), 0) + pad_y),
        xlabel="Mean precipitation change (%)",
        ylabel="Mean temperature change (°C)",
        box_aspect=1,
    )
    ax.axhline(0, color=MUTED, lw=LW_REF, zorder=1)
    ax.axvline(0, color=MUTED, lw=LW_REF, zorder=1)
    ax.grid(axis="both")
    ax.tick_params(axis="both", length=3)
    inside_legend(fig, ax, handles)
    return fig


def agreement_matrix(agreement, scenarios=None):
    """Prepared model sign counts, with discrete direction/agreement encoding."""
    data = agreement.loc[agreement.statistic == "mean"]
    scen = _scenarios(data, scenarios)
    quantities = (
        ("temp", "change", "Temperature"),
        ("precip", "change_pct", "Precipitation"),
    )
    periods = [p for p in STRIP_PERIODS if p in set(data.period)]
    positive = ("#eef3f8", "#c6d6e8", "#8fb0d3", "#3d6fa5")
    negative = ("#f8eeee", "#ecc9c9", "#d98f8f", "#a8403f")
    # Compact cells (about 7 x 3.6 mm) carry only the median sign, under
    # two-line scenario labels ("SSP1" / "2.6"), annual then seasons. Text is
    # one point below the preset so it stays in proportion to the cells.
    small = FS_NOTE - 1
    fig, axes = new_fig(
        24 + 3.6 * len(periods),
        ncols=2,
        width_mm=max(64, 24 + 2 * 7 * len(scen)),  # at least the legend
    )
    ys = [
        i + (0.45 if i and periods[0] == "annual" else 0) for i in range(len(periods))
    ]
    any_dry = False
    for b, (ax, (variable, quantity, title)) in enumerate(
        zip(axes.flat, quantities, strict=True)
    ):
        support = (
            data.loc[(data.variable == variable) & (data.quantity == quantity)]
            .groupby("scenario")
            .n.max()
        )
        for period, y in zip(periods, ys, strict=True):
            for j, s in enumerate(scen):
                rows = data.loc[
                    (data.variable == variable)
                    & (data.quantity == quantity)
                    & (data.period == period)
                    & (data.scenario == s)
                ]
                if rows.empty:
                    continue
                row = rows.iloc[0]
                median = row["median"]
                sign = "+" if median > 0 else "−" if median < 0 else ""
                k = int(
                    np.clip(
                        np.digitize(row.fraction, [0, 0.6, 0.8, 0.9, 1.0001]) - 1, 0, 3
                    )
                )
                # At most half the scenario's largest support: the dry rule left
                # too few precipitation models for agreement to mean much.
                sparse = variable == "precip" and row.n <= 0.5 * support[s]
                any_dry |= sparse
                fill = (
                    "white"
                    if sparse
                    else (positive if sign == "+" else negative)[k]
                    if sign and np.isfinite(row.fraction)
                    else "#eeeeee"
                )
                ax.add_patch(
                    Rectangle((j, y), 1, 1, facecolor=fill, edgecolor="white", lw=1)
                )
                if sparse:
                    # Excluded: hatched with no sign, so no direction is read
                    # into it, and not flat gray, which reads like the palest
                    # agreement class.
                    ax.add_patch(_dry_cell((j + 0.04, y + 0.06), 0.92, 0.88))
                    continue
                rgb = to_rgb(fill)
                linear = [
                    c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4
                    for c in rgb
                ]
                luminance = sum(
                    c * w for c, w in zip(linear, (0.2126, 0.7152, 0.0722), strict=True)
                )
                ax.text(
                    j + 0.5,
                    y + 0.5,
                    sign,
                    ha="center",
                    va="center",
                    fontsize=small,
                    color="white" if luminance < 0.179 else "#173442",
                )
        ax.set(xlim=(0, len(scen)), ylim=(ys[-1] + 1, 0))
        ax.set_xticks(
            np.arange(len(scen)) + 0.5,
            [_label(s).replace("-", "\n", 1) for s in scen],
            linespacing=1.0,
        )
        ax.xaxis.tick_top()
        ax.tick_params(axis="x", bottom=False, labelbottom=False)
        ax.set_yticks(
            np.array(ys) + 0.5,
            ["Annual" if p == "annual" else p for p in periods] if b == 0 else [],
        )
        ax.tick_params(axis="both", labelsize=small, pad=1.5)
        ax.tick_params(axis="x", length=2, width=0.6)
        ax.tick_params(axis="y", length=2 if b == 0 else 0, width=0.6)
        ax.set_title(f"{chr(97 + b)}) {title}", loc="left", fontsize=FS_TICK)
        ax.grid(False)
        for spine in ax.spines.values():
            spine.set_visible(False)
    handles = [
        (
            Patch(facecolor=positive[k], edgecolor=MUTED if k == 0 else "none"),
            Patch(facecolor=negative[k], edgecolor=MUTED if k == 0 else "none"),
        )
        for k in range(4)
    ]
    labels = ["<60%", "60–80%", "80–90%", "≥90%"]
    if any_dry:
        handles.append(_dry_cell((0, 0), 1, 1))
        labels.append("Excluded (too few models)")
    legend = fig.legend(
        handles,
        labels,
        handler_map={tuple: HandlerTuple(ndivide=None, pad=0.3)},
        loc="upper left",
        bbox_to_anchor=(0.01, 0.995),
        ncols=3,
        frameon=False,
        alignment="left",
        fontsize=small,
        title_fontsize=small,
        title="Models sharing the median sign (+ blue, − red)",
    )
    # Reserve exactly the legend's height: the "outside" legend slot leaves a
    # wide band above titles that sit over top-side tick labels.
    legend.set_in_layout(False)
    fig.canvas.draw()
    top = legend.get_window_extent().y0 / fig.bbox.height - 0.01
    fig.get_layout_engine().set(rect=(0, 0, 1, top))
    return fig


def availability_matrix(coverage, scenarios=None):
    """Resolved scenario member counts and unique loaded historical references."""
    identities = coverage[
        ["model", "source_id", "institution", "family", "family_source", "country"]
    ].drop_duplicates("model")
    identities = identities.sort_values("source_id", key=lambda s: s.str.lower())
    scen = _scenarios(coverage, scenarios)
    runs = ["historical", *scen]

    def counts_of(model):
        available = coverage.loc[(coverage.model == model) & coverage.available]
        return [
            available.reference_series_key.replace("", np.nan).nunique(),
            *(available.loc[available.scenario == s, "member"].nunique() for s in scen),
        ]

    counts_by_model = {model: counts_of(model) for model in identities.model}
    # Models with nothing loaded are left out instead of filling
    # rows with zeros; the caption gives their number and coverage.csv
    # names them.
    empty = identities.loc[[not any(counts_by_model[m]) for m in identities.model]]
    identities = identities.loc[~identities.model.isin(empty.model)]
    # One line per model (x in mm): model, family, institution, then a 9 mm
    # column per run whose filled tile marks loaded members.
    xs = [0, 25, 38, *[68 + 9 * j for j in range(len(runs))]]
    total = 68 + 9 * len(runs)
    n = len(identities)
    fig, axes = new_fig(12 + 4.2 * n, width_mm=total + 4)
    ax = axes[0, 0]
    ax.set_axis_off()
    ax.set(xlim=(0, total), ylim=(n + 0.5, -1.0))
    titles = ["Model", "Family", "Institution", "Hist.", *[_label(s) for s in scen]]
    for j, (x, title) in enumerate(zip(xs, titles, strict=True)):
        ax.text(
            x + (4.5 if j >= 3 else 0),
            -0.25,
            title.replace("-", "\n", 1) if j >= 3 else title,
            ha="center" if j >= 3 else "left",
            va="center",
            fontsize=FS_NOTE,
            fontweight="bold",
            color=INK,
            linespacing=1.0,
        )
    ax.axhline(0.5, color=INK, lw=0.8)
    for i, row in enumerate(identities.itertuples(index=False), start=1):
        if i % 2 == 0:
            ax.axhspan(i - 0.5, i + 0.5, color="#f1f5f6", lw=0, zorder=0)
        family = (
            row.family
            if row.family_source == "listed"
            else (f"{row.family}*" if row.family_source == "provisional" else "–")
        )
        where = (
            f"{row.institution}, {row.country}"
            if pd.notna(row.country) and row.country
            else row.institution
        )
        for x, text in zip(xs[:3], (row.source_id, family, where), strict=True):
            ax.text(x, i, text, ha="left", va="center", fontsize=FS_NOTE, color=INK)
        for j, count in enumerate(counts_by_model[row.model]):
            if not count:
                continue
            ax.add_patch(
                Rectangle(
                    (xs[j + 3] + 1.5, i - 0.32), 6, 0.64, facecolor="#c6d6e8", lw=0
                )
            )
            if count > 1:
                ax.text(
                    xs[j + 3] + 4.5,
                    i,
                    str(count),
                    ha="center",
                    va="center",
                    fontsize=FS_NOTE,
                    color=INK,
                )
    ax.axhline(n + 0.5, color=INK, lw=0.8)
    return fig


def render_product(filename, tables, record, *, window_id="", scenarios=None):
    """Dispatch one available product inside the shared WF0 style context."""

    filename = figure_key(filename)

    def table(name):
        data = tables[name]
        return (
            data.loc[data.window_id == window_id]
            if window_id and "window_id" in data
            else data
        )

    with use_preset(CST_PRESET):
        if filename == "availability.png":
            return availability_matrix(table("coverage"), scenarios)
        variable = "temp" if "temperature" in filename else "precip"
        if filename.startswith("trajectory-"):
            return ipcc_trajectory(
                table("trajectory_range"),
                variable=variable,
                scenarios=scenarios,
                members=table("member_anomalies"),
                windows=record["windows"],
                reference=record["reference"],
            )
        if filename.startswith("annual-"):
            return annual_overview(
                table("member_anomalies"),
                variable=variable,
                scenarios=scenarios,
                reference=record["reference"],
                windows=record["windows"],
            )
        if filename.startswith("seasonal-cycle-"):
            return seasonal_figure(
                table("climatology_summary"),
                table("model_changes"),
                table("change_summary"),
                variable=variable,
                scenarios=scenarios,
            )
        if filename.startswith("monthly-change-"):
            return monthly_change_figure(
                table("model_changes"),
                table("change_summary"),
                variable=variable,
                scenarios=scenarios,
            )
        if filename == "variability-member-ratios.png":
            return pooled_ratio_figure(
                table("variability_member_ratios"),
                pd.DataFrame(
                    [
                        row
                        for row in record["variability_model_summary"]
                        if row["window_id"] == window_id
                    ]
                ),
                scenarios,
            )
        if filename == "sign-agreement.png":
            return agreement_matrix(table("sign_agreement"), scenarios)
        if filename.startswith("change-factor-cloud"):
            return change_cloud(
                table("model_changes"),
                combined="combined" in filename,
                scenarios=scenarios,
            )
        if filename in GRID_PRODUCTS:
            name, variable, quantity, statistic = GRID_PRODUCTS[filename]
            with_amounts = (quantity, statistic) in (
                ("change_pct", "mean"),
                ("value", "total"),
            )
            return season_grid(
                table(name),
                table(SUMMARIES[name]),
                variable=variable,
                quantity=quantity,
                statistic=statistic,
                scenarios=scenarios,
                amounts=reference_amounts(table("climatology_summary"), scenarios)
                if with_amounts
                else None,
            )
    raise ValueError(f"unknown suite product: {filename}")
