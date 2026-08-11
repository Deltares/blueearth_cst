# -*- coding: utf-8 -*-
"""Render the PROPOSED WF2 projection figure set, without running the workflow.

This is a **prototype**, not a preview of what the toolbox ships. It exists so
the figure design can be judged from images before any producer, Snakefile,
report, test or output-contract change lands. Brief:
``dev/wf2-plot-standardization-task-brief.md``.

    # what is present, and what would be drawn from it?
    pixi run python dev/scripts/preview_wf2_projection_plots.py --list

    # the single-horizon case the fixture config declares
    pixi run python dev/scripts/preview_wf2_projection_plots.py

    # the two-horizon case, which is the discriminating one
    pixi run python dev/scripts/preview_wf2_projection_plots.py \
        --horizon near=2040-2060 --horizon far=2070-2090 --open

Two rules inherited from ``preview_plots.py``
---------------------------------------------
Renders land in a gitignored scratch tree (``.tmp/`` by default) and NEVER in a
project's own ``plots/``. A preview must not be able to take the place of a run
product that the baseline fingerprints. Inputs are rebuilt from artefacts a
finished run already left behind — ``scalar/*.nc`` for the series and
``summary/*_change_factors_{annual,monthly}.csv`` for the authoritative change
factors.

One rule it BREAKS, deliberately
--------------------------------
It is not registered as a ``preview_plots.py`` family and it does not call the
rule-side plotting functions. It cannot: the whole point of the prototype is
that ``plot_proj_timeseries.py`` and ``get_change_climate_proj_summary.py`` do
not implement the design being proposed. Reaching for that registry would
re-couple the prototype to the code it exists to bypass. Everything below draws
with its own matplotlib calls, on the shared page contract imported from
``shared/plot_style.py`` (read, never written).

What the proposal changes
-------------------------
1. Nine independently styled figures become four, on one page contract:
   two annual overviews, one change-factor cloud, and one monthly figure per
   configured horizon.
2. Monthly change is computed against the CORRESPONDING HISTORICAL CALENDAR
   MONTH, over that horizon's years only. The shipped figures compare a future
   month with the historical ANNUAL mean, and average the full 2015-2100 series
   instead of the horizon (``plot_proj_timeseries.py`` lines 216-223 and 221).
   The change-factor TABLES already use the corrected definition, which is what
   makes them the cross-check here rather than a second opinion: every render
   reproduces them from the series and reports the largest disagreement.
3. Only scenario carries a visual identity. Models and members are separate
   traces and points but are never in a legend and never get their own colour,
   marker or line style.

The two-horizon case
--------------------
``test_case/test_local`` declares ONE horizon (``far: [2070, 2090]``), so a
second window is prototype-declared rather than shipped: the scalar series run
to 2100, so ``--horizon near=2040-2060`` is real CMIP6 data over a window the
fixture config does not name. Only ``far`` is cross-checkable against the
tables, and the run says which horizons are prototype-declared rather than
silently checking whichever one it can.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402 -- must follow the Agg backend selection
import pandas as pd  # noqa: E402
import xarray as xr  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from blueearth_cst.shared import plot_style  # noqa: E402 -- needs the sys.path above

DEFAULT_PROJECT_DIR = REPO_ROOT / "test_case" / "test_local"
DEFAULT_OUT = REPO_ROOT / ".tmp" / "preview-wf2-projection-plots"
CLIM_SUBDIR = Path("data") / "climate" / "projections"

#: Scenario ink. Carried over from the shipped producers verbatim so the
#: prototype is judged on layout and semantics rather than on a palette change
#: nobody asked for. Scenario is the ONLY visual identity in the whole set.
SCENARIO_COLORS = {
    "ssp126": "#003466",
    "ssp245": "#f69320",
    "ssp370": "#df0000",
    "ssp585": "#980002",
}
SCENARIO_LABELS = {
    "ssp126": "SSP1-2.6",
    "ssp245": "SSP2-4.5",
    "ssp370": "SSP3-7.0",
    "ssp585": "SSP5-8.5",
}
COLOR_HISTORICAL = "0.55"
MONTH_LABELS = [
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
]

#: Variables, in the order they appear on every two-panel figure.
VARIABLES = {
    "precip": {
        "name": "Precipitation",
        "absolute_units": "mm/day",
        "change_units": "%",
        "relative": True,
    },
    "temp": {
        "name": "Temperature",
        "absolute_units": "°C",
        "change_units": "°C",
        "relative": False,
    },
}


# ===========================================================================
# INPUTS
# ===========================================================================


def normalise_model(name: str) -> str:
    """Drop the institute prefix from a model coordinate.

    ``scalar/*.nc`` carries ``NOAA-GFDL/GFDL-ESM4``; the change-factor tables
    carry ``GFDL-ESM4``. Without this every join between series-derived and
    table-derived values matches nothing, and an empty agreement check reads as
    a broken calculation rather than a spelling difference.
    """
    return str(name).split("/")[-1]


def load_scalar_series(scalar_dir: Path) -> pd.DataFrame:
    """Every ``scalar/*.nc`` as one long frame.

    Returns columns ``model, scenario, member, year, month, precip, temp``.

    Year and month are read off the time index element-wise rather than through
    a datetime coercion: these series are monthly and declare a ``noleap``
    calendar (``cst_calendar``), so they decode to ``cftime`` objects on which
    ``.resample`` and ``.dt`` behave differently than on a ``DatetimeIndex``.
    Both object types answer ``.year`` and ``.month``, which is all this needs.
    """
    files = sorted(scalar_dir.glob("*.nc"))
    if not files:
        raise SystemExit(f"no scalar series under {scalar_dir}")

    frames = []
    for path in files:
        with xr.open_dataset(path) as ds:
            times = ds.indexes["time"]
            frame = pd.DataFrame(
                {
                    "year": [t.year for t in times],
                    "month": [t.month for t in times],
                    "precip": ds["precip"].squeeze(drop=True).values,
                    "temp": ds["temp"].squeeze(drop=True).values,
                }
            )
            frame["model"] = normalise_model(ds["model"].values.item())
            frame["scenario"] = str(ds["scenario"].values.item())
            frame["member"] = str(ds["member"].values.item())
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def load_change_factors(
    summary_dir: Path, clim_project: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The authoritative annual and monthly change-factor tables, ``mean`` only.

    Both tables carry ``mean`` and ``median`` rows. Filtering is not cosmetic:
    an unfiltered table doubles every point in the cloud, and a naive "six
    combinations" check passes at twelve.
    """
    annual = pd.read_csv(summary_dir / f"{clim_project}_change_factors_annual.csv")
    monthly = pd.read_csv(summary_dir / f"{clim_project}_change_factors_monthly.csv")
    annual = annual[annual["statistic"] == "mean"].copy()
    monthly = monthly[monthly["statistic"] == "mean"].copy()
    return annual, monthly


def reference_window(annual: pd.DataFrame) -> tuple[int, int]:
    """The reference window, read off the table rather than guessed from config.

    The table states what was actually differenced (``1990-01-01 / 2010-12-01``),
    which is the only definition an agreement check can be held to. Both
    endpoints are inclusive years, so the fixture's window is 21 years — the
    ``n_reference_years`` column in ``composition.csv`` agrees.
    """
    windows = annual["reference_window"].unique()
    if len(windows) != 1:
        raise SystemExit(f"expected one reference window, found {sorted(windows)}")
    start, end = str(windows[0]).split("/")
    return int(start.strip()[:4]), int(end.strip()[:4])


def parse_horizon(text: str) -> tuple[str, int, int]:
    """``name=start-end`` into its three parts."""
    try:
        name, years = text.split("=")
        start, end = years.split("-")
        return name.strip(), int(start), int(end)
    except ValueError as exc:  # pragma: no cover -- argparse surface
        raise SystemExit(f"bad --horizon {text!r}; expected name=START-END") from exc


# ===========================================================================
# THE PROPOSED CALCULATION
# ===========================================================================


def combination_key(frame: pd.DataFrame) -> pd.Series:
    return frame["model"] + " " + frame["scenario"] + " " + frame["member"]


def monthly_reference(series: pd.DataFrame, ref: tuple[int, int]) -> pd.DataFrame:
    """Historical mean per (model, member, calendar month) over the reference window.

    This is the baseline the proposal differences against, and the one the
    shipped figures do not use: they difference a future month against a single
    historical ANNUAL mean, which erases the seasonal cycle from the comparison.
    """
    hist = series[(series["scenario"] == "historical") & series["year"].between(*ref)]
    return hist.groupby(["model", "member", "month"], as_index=False)[
        ["precip", "temp"]
    ].mean()


def annual_reference(series: pd.DataFrame, ref: tuple[int, int]) -> pd.DataFrame:
    """Historical mean per (model, member) over the reference window."""
    hist = series[(series["scenario"] == "historical") & series["year"].between(*ref)]
    return hist.groupby(["model", "member"], as_index=False)[["precip", "temp"]].mean()


def monthly_change(
    series: pd.DataFrame, ref: tuple[int, int], horizon: tuple[int, int]
) -> pd.DataFrame:
    """Per-combination monthly change, the corrected definition.

    Future calendar month over THIS HORIZON's years, against the corresponding
    historical calendar month. Precipitation as a percentage, temperature as a
    difference in degrees.
    """
    future = series[
        (series["scenario"] != "historical") & series["year"].between(*horizon)
    ]
    future = future.groupby(["model", "scenario", "member", "month"], as_index=False)[
        ["precip", "temp"]
    ].mean()
    merged = future.merge(
        monthly_reference(series, ref),
        on=["model", "member", "month"],
        suffixes=("", "_ref"),
    )
    merged["precip_change"] = (
        (merged["precip"] - merged["precip_ref"]) / merged["precip_ref"] * 100.0
    )
    merged["temp_change"] = merged["temp"] - merged["temp_ref"]
    return merged


def annual_change(
    series: pd.DataFrame, ref: tuple[int, int], horizon: tuple[int, int]
) -> pd.DataFrame:
    """Per-combination annual change over a horizon — one point in the cloud."""
    future = series[
        (series["scenario"] != "historical") & series["year"].between(*horizon)
    ]
    future = future.groupby(["model", "scenario", "member"], as_index=False)[
        ["precip", "temp"]
    ].mean()
    merged = future.merge(
        annual_reference(series, ref), on=["model", "member"], suffixes=("", "_ref")
    )
    merged["precip_change"] = (
        (merged["precip"] - merged["precip_ref"]) / merged["precip_ref"] * 100.0
    )
    merged["temp_change"] = merged["temp"] - merged["temp_ref"]
    return merged


def annual_series(series: pd.DataFrame, ref: tuple[int, int]) -> pd.DataFrame:
    """Annual means per combination per year, plus the anomaly against the reference.

    Every trace on the two annual overview figures comes from here — historical
    and future alike, each differenced against its OWN model's historical
    reference window so a future trace is continuous with the historical one it
    follows.
    """
    annual = series.groupby(["model", "scenario", "member", "year"], as_index=False)[
        ["precip", "temp"]
    ].mean()
    merged = annual.merge(
        annual_reference(series, ref), on=["model", "member"], suffixes=("", "_ref")
    )
    merged["precip_anomaly"] = (
        (merged["precip"] - merged["precip_ref"]) / merged["precip_ref"] * 100.0
    )
    merged["temp_anomaly"] = merged["temp"] - merged["temp_ref"]
    return merged


def shipped_monthly_change(series: pd.DataFrame, ref: tuple[int, int]) -> pd.DataFrame:
    """What the SHIPPED figures plot, reproduced here for the falsifier only.

    Two departures from the definition above, both read off
    ``plot_proj_timeseries.py``: the future monthly mean is taken over the FULL
    future series rather than the horizon (line 221), and it is differenced
    against the historical ANNUAL mean rather than the same calendar month
    (lines 216-223). Reproduced rather than described so the artifact can show
    the gap as a number.
    """
    future = series[series["scenario"] != "historical"]
    future = future.groupby(["model", "scenario", "member", "month"], as_index=False)[
        ["precip", "temp"]
    ].mean()
    merged = future.merge(
        annual_reference(series, ref), on=["model", "member"], suffixes=("", "_ref")
    )
    merged["precip_change"] = (
        (merged["precip"] - merged["precip_ref"]) / merged["precip_ref"] * 100.0
    )
    merged["temp_change"] = merged["temp"] - merged["temp_ref"]
    return merged


# ===========================================================================
# THE FIGURES
# ===========================================================================


def scenario_legend(ax, scenarios, *, with_historical=True) -> None:
    """One compact legend: historical plus scenarios, and nothing else.

    Built from proxy handles rather than from the drawn artists, because the
    drawn artists are one per COMBINATION — labelling those is how "one trace
    per combination" becomes a legend naming every model, which is the contract
    this figure set is here to hold.
    """
    handles = []
    if with_historical:
        handles.append(
            plt.Line2D([], [], color=COLOR_HISTORICAL, lw=1.2, label="Historical")
        )
    for scenario in scenarios:
        handles.append(
            plt.Line2D(
                [],
                [],
                color=SCENARIO_COLORS.get(scenario, "0.2"),
                lw=1.2,
                label=SCENARIO_LABELS.get(scenario, scenario),
            )
        )
    ax.legend(handles=handles, loc="upper left", frameon=False, ncols=len(handles))


def footnote(fig, text: str, *, width: int = 138) -> float:
    """Wrap a caveat under the panels and return the figure fraction it occupies.

    Wrapped rather than trusted to fit: ``fig.text`` does not clip, it draws
    straight off the canvas, so an unwrapped caveat SILENTLY loses its tail —
    the first render of this set lost "CMIP6 is a plausibility overlay, not a
    stress-test driver", which is the one sentence on the figure that says what
    the figure is not for.

    The return value is the ``bottom`` a caller passes to ``tight_layout``'s
    ``rect``, so the reservation follows the number of wrapped lines instead of
    being a constant that a longer caveat quietly overruns. Call this BEFORE
    ``tight_layout``; matplotlib's layout engines ignore ``fig.text``.
    """
    wrapped = textwrap.fill(text, width=width)
    fig.text(
        0.006,
        0.006,
        wrapped,
        ha="left",
        va="bottom",
        fontsize=plot_style.FONT_SIZE_CAVEAT,
        color=plot_style.COLOR_CAVEAT,
    )
    line_height_points = plot_style.FONT_SIZE_CAVEAT * 1.4
    lines = wrapped.count("\n") + 1
    return min(lines * line_height_points / (fig.get_figheight() * 72.0) + 0.012, 0.3)


def figure_annual(
    annual: pd.DataFrame, variable: str, ref: tuple[int, int], out_path: Path, dpi: int
) -> int:
    """Absolute and anomaly panels over the full historical/future series."""
    meta = VARIABLES[variable]
    scenarios = sorted(s for s in annual["scenario"].unique() if s != "historical")
    width = plot_style.figure_width_inches()

    with plt.rc_context(plot_style.rcparams()):
        fig, axes = plt.subplots(2, 1, figsize=(width, width * 0.62), sharex=True)
        traces = 0
        for panel, column, units in (
            (axes[0], variable, meta["absolute_units"]),
            (axes[1], f"{variable}_anomaly", meta["change_units"]),
        ):
            for _, group in annual.groupby(["model", "scenario", "member"], sort=True):
                scenario = group["scenario"].iloc[0]
                color = (
                    COLOR_HISTORICAL
                    if scenario == "historical"
                    else SCENARIO_COLORS.get(scenario, "0.2")
                )
                panel.plot(
                    group["year"],
                    group[column],
                    color=color,
                    lw=0.7,
                    alpha=0.85,
                    zorder=2,
                )
                traces += 1
            panel.grid(True, lw=0.4, color="0.85", zorder=0)
            panel.set_axisbelow(True)
            panel.set_ylabel(units)

        axes[1].axhline(0.0, color="0.3", lw=0.7, zorder=1)
        # The historical/future transition, labelled rather than left to be
        # inferred from where the grey stops.
        transition = annual.loc[annual["scenario"] == "historical", "year"].max() + 0.5
        for panel in axes:
            panel.axvline(transition, color="0.4", lw=0.7, ls=(0, (4, 3)), zorder=1)
        axes[0].annotate(
            "historical | future",
            xy=(transition, 1.0),
            xytext=(3, -9),
            textcoords="offset points",
            xycoords=("data", "axes fraction"),
            fontsize=plot_style.FONT_SIZE_CAVEAT,
            color="0.4",
        )
        axes[0].set_title(f"{meta['name']}, annual mean", loc="left")
        axes[1].set_title(
            f"Anomaly against {ref[0]}–{ref[1]}, each model against its own historical",
            loc="left",
        )
        axes[1].set_xlabel("Year")
        scenario_legend(axes[0], scenarios)
        bottom = footnote(
            fig,
            f"One trace per (model, scenario, member); {traces // 2} traces per panel. "
            "Colour encodes scenario only — models and members are not distinguished. "
            "CMIP6 is a plausibility overlay, not a stress-test driver.",
        )
        fig.tight_layout(rect=(0, bottom, 1, 1))
        fig.savefig(out_path, dpi=dpi)
        plt.close(fig)
    return traces // 2


def figure_cloud(
    changes: dict[str, pd.DataFrame],
    horizons: dict[str, tuple[int, int]],
    out_path: Path,
    dpi: int,
) -> int:
    """The change-factor cloud, faceted by horizon on identical axes.

    No marginal KDEs: a kernel density over six points asserts a distribution
    the design does not construct, and the brief's first non-goal is exactly
    that. One panel when only one horizon exists.
    """
    names = list(horizons)
    scenarios = sorted(
        {s for frame in changes.values() for s in frame["scenario"].unique()}
    )
    width = plot_style.figure_width_inches()
    panel_width = width / max(len(names), 1)

    with plt.rc_context(plot_style.rcparams()):
        fig, axes = plt.subplots(
            1,
            len(names),
            figsize=(width, min(panel_width * 1.05 + 0.5, width * 0.55)),
            squeeze=False,
        )
        all_precip = pd.concat(frame["precip_change"] for frame in changes.values())
        all_temp = pd.concat(frame["temp_change"] for frame in changes.values())
        # Identical axes across facets: a horizon that looks calmer must BE
        # calmer, not merely be drawn on a kinder scale.
        pad_x = max((all_precip.max() - all_precip.min()) * 0.15, 0.5)
        pad_y = max((all_temp.max() - all_temp.min()) * 0.15, 0.1)
        xlim = (min(all_precip.min(), 0) - pad_x, max(all_precip.max(), 0) + pad_x)
        ylim = (min(all_temp.min(), 0) - pad_y, max(all_temp.max(), 0) + pad_y)

        points = 0
        for axis, name in zip(axes[0], names):
            frame = changes[name]
            for scenario in scenarios:
                subset = frame[frame["scenario"] == scenario]
                axis.scatter(
                    subset["precip_change"],
                    subset["temp_change"],
                    s=26,
                    color=SCENARIO_COLORS.get(scenario, "0.2"),
                    alpha=0.85,
                    edgecolor="white",
                    linewidth=0.4,
                    zorder=3,
                )
                points += len(subset)
            axis.axhline(0.0, color="0.35", lw=0.7, zorder=1)
            axis.axvline(0.0, color="0.35", lw=0.7, zorder=1)
            axis.grid(True, lw=0.4, color="0.88", zorder=0)
            axis.set_axisbelow(True)
            axis.set_xlim(*xlim)
            axis.set_ylim(*ylim)
            start, end = horizons[name]
            axis.set_title(f"{name} ({start}–{end})", loc="left")
            axis.set_xlabel("Change in mean precipitation (%)")
        axes[0][0].set_ylabel("Change in mean temperature (°C)")
        for axis in axes[0][1:]:
            axis.tick_params(labelleft=False)
        scenario_legend(axes[0][0], scenarios, with_historical=False)
        bottom = footnote(
            fig,
            f"One point per (model, scenario, member) per horizon; {points} points drawn. "
            "Axes are shared across horizons, so a calmer-looking horizon is a calmer one. "
            "Marginal densities removed: these points are not a distribution.",
        )
        fig.tight_layout(rect=(0, bottom, 1, 1))
        fig.savefig(out_path, dpi=dpi)
        plt.close(fig)
    return points


def figure_monthly(
    changes: pd.DataFrame,
    name: str,
    horizon: tuple[int, int],
    ref: tuple[int, int],
    out_path: Path,
    dpi: int,
) -> int:
    """Precipitation (%) and temperature (degC) change by calendar month."""
    scenarios = sorted(changes["scenario"].unique())
    width = plot_style.figure_width_inches()

    with plt.rc_context(plot_style.rcparams()):
        fig, axes = plt.subplots(1, 2, figsize=(width, width * 0.36))
        traces = 0
        for axis, column, label in (
            (axes[0], "precip_change", "Change in precipitation (%)"),
            (axes[1], "temp_change", "Change in temperature (°C)"),
        ):
            for _, group in changes.groupby(["model", "scenario", "member"], sort=True):
                group = group.sort_values("month")
                axis.plot(
                    group["month"],
                    group[column],
                    color=SCENARIO_COLORS.get(group["scenario"].iloc[0], "0.2"),
                    lw=0.9,
                    alpha=0.85,
                    marker="o",
                    markersize=2.2,
                    zorder=2,
                )
                traces += 1
            axis.axhline(0.0, color="0.35", lw=0.7, zorder=1)
            axis.grid(True, lw=0.4, color="0.88", zorder=0)
            axis.set_axisbelow(True)
            axis.set_xticks(range(1, 13), MONTH_LABELS)
            axis.set_ylabel(label)
        scenario_legend(axes[0], scenarios, with_historical=False)
        fig.suptitle(
            f"Monthly change factors — {name} ({horizon[0]}–{horizon[1]}) "
            f"against {ref[0]}–{ref[1]}",
            fontsize=plot_style.FONT_SIZE_TITLE,
            x=0.006,
            y=0.995,
            va="top",
            ha="left",
        )
        bottom = footnote(
            fig,
            f"One trace per (model, scenario, member); {traces // 2} traces per panel. Each future "
            f"calendar month is differenced against the SAME historical calendar month, using only "
            f"{horizon[0]}–{horizon[1]}.",
        )
        fig.tight_layout(rect=(0, bottom, 1, 1))
        fig.savefig(out_path, dpi=dpi)
        plt.close(fig)
    return traces // 2


def figure_falsifier(
    proposed: pd.DataFrame,
    shipped: pd.DataFrame,
    table: pd.DataFrame,
    combination: tuple[str, str, str],
    horizon_name: str,
    horizon: tuple[int, int],
    out_path: Path,
    dpi: int,
) -> None:
    """One combination, three definitions: proposal, shipped figure, and the table.

    Not part of the proposed set. It exists so the monthly-semantics claim can
    be judged as a picture and a number rather than taken on trust — the
    proposal must sit ON the table's markers, and the shipped definition must
    visibly not.
    """
    model, scenario, member = combination
    width = plot_style.figure_width_inches()

    def pick(frame):
        return frame[
            (frame["model"] == model)
            & (frame["scenario"] == scenario)
            & (frame["member"] == member)
        ].sort_values("month")

    prop, ship = pick(proposed), pick(shipped)
    tab = table[
        (table["model"] == model)
        & (table["scenario"] == scenario)
        & (table["member"] == member)
        & (table["horizon"] == horizon_name)
    ].sort_values("month")

    with plt.rc_context(plot_style.rcparams()):
        fig, axes = plt.subplots(1, 2, figsize=(width, width * 0.36))
        panels = (
            (axes[0], "precip", "Change in precipitation (%)"),
            (axes[1], "temp", "Change in temperature (°C)"),
        )
        for axis, variable, label in panels:
            column = f"{variable}_change"
            # The table states precipitation change as `relative_value` but
            # carries no temperature DELTA column — that one is derived from the
            # two absolutes, the same subtraction the agreement check makes.
            rows = tab[tab["variable"] == variable].sort_values("month")
            table_values = (
                rows["relative_value"]
                if VARIABLES[variable]["relative"]
                else rows["absolute_value"] - rows["reference_value"]
            )
            axis.plot(
                ship["month"],
                ship[column],
                color="0.55",
                lw=1.0,
                ls=(0, (4, 3)),
                label="Shipped figure definition",
                zorder=2,
            )
            axis.plot(
                prop["month"],
                prop[column],
                color=SCENARIO_COLORS.get(scenario, "0.2"),
                lw=1.2,
                label="Proposed definition",
                zorder=3,
            )
            axis.scatter(
                rows["month"],
                table_values,
                s=30,
                facecolor="none",
                edgecolor="black",
                linewidth=0.8,
                label="Change-factor table",
                zorder=4,
            )
            axis.axhline(0.0, color="0.35", lw=0.7, zorder=1)
            axis.grid(True, lw=0.4, color="0.88", zorder=0)
            axis.set_axisbelow(True)
            axis.set_xticks(range(1, 13), MONTH_LABELS)
            axis.set_ylabel(label)
        axes[0].legend(loc="best", frameon=False)
        fig.suptitle(
            f"Monthly semantics — {model} {SCENARIO_LABELS.get(scenario, scenario)} "
            f"{member}, {horizon_name} ({horizon[0]}–{horizon[1]})",
            fontsize=plot_style.FONT_SIZE_TITLE,
            x=0.006,
            y=0.995,
            va="top",
            ha="left",
        )
        bottom = footnote(
            fig,
            "The proposal reproduces the authoritative table. The shipped figure differences "
            "each future month against the historical ANNUAL mean and averages 2015–2100 "
            "instead of the horizon, so it disagrees with the table it sits beside.",
        )
        fig.tight_layout(rect=(0, bottom, 1, 1))
        fig.savefig(out_path, dpi=dpi)
        plt.close(fig)


# ===========================================================================
# THE AGREEMENT CHECK
# ===========================================================================


def check_against_tables(
    proposed: pd.DataFrame, table: pd.DataFrame, horizon_name: str
) -> pd.DataFrame | None:
    """Reproduce the change-factor table from the series, and report the worst gap.

    The tables already implement the corrected monthly definition, so this is a
    real-data agreement check rather than a hand calculation: if the proposal is
    right, it lands on them to within the tables' 3-decimal rounding. A failure
    here is a defect in this prototype — the tables are authoritative.
    """
    subset = table[table["horizon"] == horizon_name]
    if subset.empty:
        return None
    merged = proposed.merge(
        subset[
            [
                "model",
                "scenario",
                "member",
                "month",
                "variable",
                "relative_value",
                "reference_value",
                "absolute_value",
            ]
        ],
        on=["model", "scenario", "member", "month"],
    )
    rows = []
    for variable, meta in VARIABLES.items():
        part = merged[merged["variable"] == variable].copy()
        if part.empty:
            continue
        if meta["relative"]:
            part["table_value"] = part["relative_value"]
        else:
            part["table_value"] = part["absolute_value"] - part["reference_value"]
        part["prototype_value"] = part[f"{variable}_change"]
        part["difference"] = (part["prototype_value"] - part["table_value"]).abs()
        rows.append(
            {
                "variable": variable,
                "units": meta["change_units"],
                "rows_compared": len(part),
                "max_abs_difference": part["difference"].max(),
                "worst": part.loc[
                    part["difference"].idxmax(), ["model", "scenario", "month"]
                ].to_dict(),
            }
        )
    return pd.DataFrame(rows)


# ===========================================================================
# DRIVER
# ===========================================================================


def render(args: argparse.Namespace) -> int:
    clim_dir = Path(args.project_dir) / CLIM_SUBDIR / args.clim_project
    scalar_dir, summary_dir = clim_dir / "scalar", clim_dir / "summary"
    for path in (scalar_dir, summary_dir):
        if not path.is_dir():
            raise SystemExit(f"missing input directory: {path}")

    series = load_scalar_series(scalar_dir)
    annual_table, monthly_table = load_change_factors(summary_dir, args.clim_project)
    ref = reference_window(annual_table)
    horizons = {
        name: (start, end) for name, start, end in map(parse_horizon, args.horizon)
    }

    future = series[series["scenario"] != "historical"]
    resolved = future.groupby(["model", "scenario", "member"]).ngroups
    historical = (
        series[series["scenario"] == "historical"].groupby(["model", "member"]).ngroups
    )
    print(
        f"reference window     : {ref[0]}-{ref[1]} ({ref[1] - ref[0] + 1} years, inclusive)"
    )
    print(f"resolved combinations: {resolved} future, {historical} historical")
    print(
        f"horizons             : {', '.join(f'{k} {v[0]}-{v[1]}' for k, v in horizons.items())}"
    )

    if args.list:
        for name, frame in (
            ("annual table", annual_table),
            ("monthly table", monthly_table),
        ):
            print(f"{name:<21}: {len(frame)} mean rows")
        print(f"series rows          : {len(series)}")
        return 0

    out_dir = Path(args.out_dir)
    (out_dir / "overview").mkdir(parents=True, exist_ok=True)

    annual = annual_series(series, ref)
    counts = {}
    for variable in VARIABLES:
        target = (
            out_dir / "overview" / f"annual-{VARIABLES[variable]['name'].lower()}.png"
        )
        counts[target.name] = figure_annual(annual, variable, ref, target, args.dpi)
        print(
            f"wrote {target.relative_to(out_dir)}  ({counts[target.name]} traces/panel)"
        )

    changes = {
        name: annual_change(series, ref, span) for name, span in horizons.items()
    }
    cloud = out_dir / "overview" / "change-factor-cloud.png"
    counts[cloud.name] = figure_cloud(changes, horizons, cloud, args.dpi)
    print(f"wrote {cloud.relative_to(out_dir)}  ({counts[cloud.name]} points)")

    proposed = {}
    for name, span in horizons.items():
        proposed[name] = monthly_change(series, ref, span)
        window_dir = out_dir / "windows" / f"{name}-{span[0]}-{span[1]}"
        window_dir.mkdir(parents=True, exist_ok=True)
        target = window_dir / "monthly-change-factors.png"
        counts[target.name] = figure_monthly(
            proposed[name], name, span, ref, target, args.dpi
        )
        print(
            f"wrote {target.relative_to(out_dir)}  ({counts[target.name]} traces/panel)"
        )

    # The falsifier, drawn for whichever horizon the tables can actually check.
    checkable = [n for n in horizons if n in set(monthly_table["horizon"])]
    if checkable:
        name = checkable[0]
        shipped = shipped_monthly_change(series, ref)
        first = proposed[name].sort_values(["model", "scenario", "member"]).iloc[0]
        figure_falsifier(
            proposed[name],
            shipped,
            monthly_table,
            (first["model"], first["scenario"], first["member"]),
            name,
            horizons[name],
            out_dir / "falsifier-monthly-semantics.png",
            args.dpi,
        )
        print("wrote falsifier-monthly-semantics.png")

        report = check_against_tables(proposed[name], monthly_table, name)
        print(
            f"\nagreement with {args.clim_project}_change_factors_monthly.csv, horizon {name!r}:"
        )
        print(report.to_string(index=False))
    else:
        print(
            f"\nno horizon in {sorted(horizons)} appears in the change-factor tables "
            f"({sorted(set(monthly_table['horizon']))}); agreement check SKIPPED"
        )
    for name in horizons:
        if name not in set(monthly_table["horizon"]):
            print(
                f"note: horizon {name!r} is prototype-declared, not present in the tables"
            )

    print(f"\nrenders in {out_dir}")
    if args.open:
        subprocess.run(["explorer", str(out_dir)], check=False)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--project-dir", default=str(DEFAULT_PROJECT_DIR))
    parser.add_argument("--clim-project", default="cmip6")
    parser.add_argument(
        "--horizon",
        action="append",
        metavar="NAME=START-END",
        help="repeatable; defaults to far=2070-2090 (the fixture's configured horizon)",
    )
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT))
    parser.add_argument("--dpi", type=int, default=plot_style.RASTER_DPI)
    parser.add_argument("--list", action="store_true", help="report inputs and exit")
    parser.add_argument(
        "--open", action="store_true", help="open the output folder when done"
    )
    args = parser.parse_args(argv)
    args.horizon = args.horizon or ["far=2070-2090"]
    return render(args)


if __name__ == "__main__":
    raise SystemExit(main())
