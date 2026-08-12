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
re-couple the prototype to the code it exists to bypass.

The WF1 page contract, reused rather than re-derived
----------------------------------------------------
Everything below draws under ``cartographic_map._publication_rc()`` at
``series_figure_size(...)``, in CONSTRAINED layout, with the caveat carried by
``fig.supxlabel(..., wrap=True)`` — the same four decisions
``climate_analysis/climate_figures.py`` makes for the WF1 series figures. That
is not incidental tidiness:

* ``layout="constrained"`` rather than ``tight_layout`` because the WF1 maps are
  built on it, and a figure family that mixes the two cannot be made to agree on
  its margins.
* ``supxlabel`` rather than a hand-placed ``fig.text`` because it is part of the
  layout, so a long caveat re-flows instead of being drawn off the canvas.
  ``fig.text`` does not clip — it silently loses the tail, which is how the
  first version of this script dropped "CMIP6 is a plausibility overlay, not a
  stress-test driver" off the right-hand edge of every figure.

Both helpers are called THROUGH the module (``cartographic_map._publication_rc()``)
rather than imported by name, because ``dev/scripts/preview_basin_map.py --set``
rebinds those globals at runtime; a ``from ... import`` would snapshot them at
import and silently ignore an override.

No titles anywhere
------------------
Owner ruling, 2026-08-11, and it applies to every figure this toolbox draws:
panels carry ``a)``/``b)`` labels instead of titles. What a title used to say is
routed to the places a journal figure keeps it — the variable and its unit go in
the y-label (WF1's ``f"{label.capitalize()} ({axis_unit})"`` convention), and the
horizon, reference window and trace counts go in the caveat line.

What the proposal changes
-------------------------
1. Nine independently styled figures become five, on one page contract: two
   annual overviews, two views of the change-factor cloud, and one monthly
   figure per configured horizon.
2. Monthly change is computed against the CORRESPONDING HISTORICAL CALENDAR
   MONTH, over that horizon's years only. The shipped figures compare a future
   month with the historical ANNUAL mean, and average the full 2015-2100 series
   instead of the horizon (``plot_proj_timeseries.py`` lines 216-223 and 221).
   The change-factor TABLES already use the corrected definition, which is what
   makes them the cross-check here rather than a second opinion: every render
   reproduces them from the series and reports the largest disagreement.
3. Scenario is the only visual ENCODING — no model or member gets its own
   colour, marker or line style, and no legend names one. The cloud additionally
   labels each point with its model name (owner ruling, 2026-08-11): a direct
   annotation identifies a point without making model a visual channel.

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
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402 -- must follow the Agg backend selection
import pandas as pd  # noqa: E402
import xarray as xr  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from blueearth_cst.shared import cartographic_map, plot_style  # noqa: E402

DEFAULT_PROJECT_DIR = REPO_ROOT / "test_case" / "test_local"
DEFAULT_OUT = REPO_ROOT / ".tmp" / "preview-wf2-projection-plots"
CLIM_SUBDIR = Path("data") / "climate" / "projections"

#: Scenario ink. Carried over from the shipped producers verbatim so the
#: prototype is judged on layout and semantics rather than on a palette change
#: nobody asked for. Scenario is the only visual ENCODING in the whole set.
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

#: Marker per horizon, for the combined cloud only. Horizon is not model or
#: member, so encoding it costs nothing the scenario-only rule protects.
HORIZON_MARKERS = ["o", "s", "^", "D", "v"]

MONTH_LABELS = ["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"]

#: Point size of a panel's ``a)`` label and of the model annotations on the
#: cloud. ``climate_figures.FONT_SIZE_ANNOTATION`` is the WF1 value for text
#: that sits inside the axes rather than labelling them.
FONT_SIZE_ANNOTATION = 6.0

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
# PAGE FURNITURE
# ===========================================================================


def new_figure(aspect: float, nrows: int = 1, ncols: int = 1, **kwargs):
    """A figure at the shared page width, in the WF1 rc and constrained layout.

    ``aspect`` is chosen per figure SHAPE rather than left at
    ``series_figure_size``'s 0.42 default, which sizes a single-axes series: a
    stacked pair needs most of double that height, a side-by-side pair needs
    about the single height with squarer panels.
    """
    size = cartographic_map.series_figure_size(aspect)
    return plt.subplots(nrows, ncols, figsize=size, layout="constrained", **kwargs)


def style_series_axes(ax) -> None:
    """The WF1 series treatment: an L-frame with a horizontal-only grid.

    Mirrors ``climate_figures._style_series_axes``. Deliberately re-stated here
    rather than imported: that one is private to the WF1 module, and this is a
    prototype that must not reach into it. Keep the two in step by eye — the
    values are three lines, and coupling a prototype to a private helper costs
    more than restating it.
    """
    ax.grid(axis="y", alpha=0.25, lw=0.5)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def style_scatter_axes(ax) -> None:
    """As above, but gridded on BOTH axes.

    A deliberate departure from the series treatment: on the change-factor cloud
    both coordinates carry meaning and both zero lines are drawn, so a
    horizontal-only grid would imply the x position is the approximate one.
    """
    ax.grid(alpha=0.25, lw=0.5)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def panel_label(ax, letter: str) -> None:
    """``a)``, ``b)`` … at the panel's top-left corner.

    Owner ruling, 2026-08-11: no titles above figures anywhere in this toolbox.
    The letter sits just outside the axes so it cannot collide with data or with
    the legend, and it carries no descriptive text — what the title used to say
    is in the y-label and the caveat.
    """
    ax.annotate(
        f"{letter})",
        xy=(0, 1),
        xycoords="axes fraction",
        xytext=(0, 4),
        textcoords="offset points",
        ha="left",
        va="bottom",
        fontsize=plot_style.FONT_SIZE_BASE,
        fontweight="bold",
    )


def caveat(fig, text: str) -> None:
    """The provenance line under the panels, as part of the layout.

    ``supxlabel`` with ``wrap=True``, the WF1 mechanism: constrained layout
    reserves room for it and long text re-flows. A ``fig.text`` at a hand-picked
    position does neither — it draws off the canvas and loses its tail without
    raising anything.
    """
    fig.supxlabel(
        text,
        fontsize=plot_style.FONT_SIZE_CAVEAT,
        color=plot_style.COLOR_CAVEAT,
        wrap=True,
    )


def scenario_handles(scenarios, *, with_historical=True):
    """Legend proxies: historical plus scenarios, and nothing else.

    Proxies rather than the drawn artists, because the drawn artists are one per
    COMBINATION — labelling those is how "one trace per combination" becomes a
    legend naming every model, which is the contract this figure set holds.
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
    return handles


#: Where a model label may sit relative to its point, in points, best first.
#: Tried in order until one lands clear of the labels already placed.
LABEL_OFFSETS = [
    (5, 3),
    (-5, 3),
    (5, -9),
    (-5, -9),
    (0, 8),
    (0, -12),
    (10, -3),
    (-10, -3),
]


def label_points(ax, frame, blockers=()) -> None:
    """Annotate every cloud point with its model name, avoiding overprints.

    Owner ruling, 2026-08-11. This is the one place a model name appears on a
    figure, and it is a DIRECT ANNOTATION rather than a visual channel: the
    point keeps its scenario colour and the shared marker, so nothing about the
    ink encodes which model it is.

    Placement is greedy against the real rendered extents rather than a fixed
    rotation of offsets. A rotation looks like it works and does not: it keys on
    the row's ORDER, while collisions are a fact about the row's POSITION, so
    the two ``far``-panel models that differ by one percentage point drew their
    labels on top of each other while models at opposite corners were carefully
    given different offsets. Each label is drawn, measured, and moved to the next
    candidate if it overlaps one already placed; the first candidate is kept when
    none is clear, which is better than dropping a label silently.

    Three things count as occupied and are seeded into the blocker list before
    any label is placed: the markers themselves, the labels already placed, and
    whatever artists the caller passes in ``blockers`` — in practice the legend,
    which a label will otherwise happily print underneath.

    **Call this last, after the axis limits are final.** Extents are measured in
    display space, so a later ``set_xlim`` moves every point out from under the
    labels this placed.
    """
    figure = ax.figure
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()

    placed = [artist.get_window_extent(renderer) for artist in blockers]
    # The markers. A label overprinting the point it names is the one collision
    # that makes the figure actively misleading rather than merely crowded.
    marker_pad = 6.0 * figure.dpi / 72.0
    for _, row in frame.iterrows():
        x, y = ax.transData.transform((row["precip_change"], row["temp_change"]))
        placed.append(
            matplotlib.transforms.Bbox.from_extents(
                x - marker_pad, y - marker_pad, x + marker_pad, y + marker_pad
            )
        )

    for _, row in frame.iterrows():
        point = (row["precip_change"], row["temp_change"])
        text = None
        for dx, dy in LABEL_OFFSETS:
            text = ax.annotate(
                row["model"],
                xy=point,
                xytext=(dx, dy),
                textcoords="offset points",
                ha="left" if dx > 0 else ("right" if dx < 0 else "center"),
                va="bottom" if dy > 0 else "top",
                fontsize=FONT_SIZE_ANNOTATION,
                color="0.25",
            )
            extent = text.get_window_extent(renderer).expanded(1.04, 1.2)
            if not any(extent.overlaps(other) for other in placed):
                placed.append(extent)
                break
            text.remove()
            text = None
        if text is None:
            # Every candidate collided — keep the first rather than lose the
            # label, and let the overlap be visible instead of the name absent.
            dx, dy = LABEL_OFFSETS[0]
            text = ax.annotate(
                row["model"],
                xy=point,
                xytext=(dx, dy),
                textcoords="offset points",
                ha="left",
                va="bottom",
                fontsize=FONT_SIZE_ANNOTATION,
                color="0.25",
            )
            placed.append(text.get_window_extent(renderer).expanded(1.04, 1.2))


# ===========================================================================
# THE FIGURES
# ===========================================================================


def figure_annual(
    annual: pd.DataFrame, variable: str, ref: tuple[int, int], out_path: Path, dpi: int
) -> int:
    """Absolute (a) and anomaly (b) panels over the full historical/future series."""
    meta = VARIABLES[variable]
    scenarios = sorted(s for s in annual["scenario"].unique() if s != "historical")

    with plt.rc_context(cartographic_map._publication_rc()):
        # 0.78 rather than 2x0.42: the panels share an x axis, so the pair needs
        # one set of tick labels rather than two.
        fig, axes = new_figure(0.78, nrows=2, sharex=True)
        traces = 0
        panels = (
            (axes[0], variable, f"{meta['name']} ({meta['absolute_units']})", "a"),
            (
                axes[1],
                f"{variable}_anomaly",
                f"{meta['name']} anomaly ({meta['change_units']})",
                "b",
            ),
        )
        for panel, column, ylabel, letter in panels:
            for _, group in annual.groupby(["model", "scenario", "member"], sort=True):
                scenario = group["scenario"].iloc[0]
                color = (
                    COLOR_HISTORICAL
                    if scenario == "historical"
                    else SCENARIO_COLORS.get(scenario, "0.2")
                )
                panel.plot(
                    group["year"], group[column], color=color, lw=0.7, alpha=0.85
                )
                traces += 1
            style_series_axes(panel)
            panel.set_ylabel(ylabel)
            panel_label(panel, letter)

        axes[1].axhline(0.0, color="0.3", lw=0.6)
        # The historical/future handover, marked rather than left to be inferred
        # from where the grey stops.
        transition = annual.loc[annual["scenario"] == "historical", "year"].max() + 0.5
        for panel in axes:
            panel.axvline(transition, color="0.45", lw=0.6, ls=(0, (4, 3)))
        axes[1].set_xlabel("Year")
        axes[1].xaxis.set_major_locator(MaxNLocator(integer=True))
        axes[0].legend(
            handles=scenario_handles(scenarios),
            loc="upper left",
            frameon=False,
            ncols=len(scenarios) + 1,
        )
        caveat(
            fig,
            f"a) annual mean. b) anomaly against {ref[0]}–{ref[1]}, each model differenced "
            f"against its own historical run; the dashed rule marks the historical/future "
            f"handover. One trace per (model, scenario, member): {traces // 2} per panel. "
            "Colour encodes scenario only — models and members are not distinguished. "
            "CMIP6 is a plausibility overlay, not a stress-test driver.",
        )
        fig.savefig(out_path, dpi=dpi)
        plt.close(fig)
    return traces // 2


def figure_cloud_faceted(
    changes: dict[str, pd.DataFrame],
    horizons: dict[str, tuple[int, int]],
    out_path: Path,
    dpi: int,
) -> int:
    """The change-factor cloud, one panel per horizon on identical axes.

    No marginal KDEs: a kernel density over six points asserts a distribution the
    design does not construct. One panel when only one horizon exists.
    """
    names = list(horizons)
    scenarios = sorted(
        {s for frame in changes.values() for s in frame["scenario"].unique()}
    )

    with plt.rc_context(cartographic_map._publication_rc()):
        # Squarer than a series figure: both coordinates are changes in the same
        # sense, so the panel should not privilege one axis by stretching it.
        fig, axes = new_figure(
            0.50 if len(names) > 1 else 0.62, ncols=len(names), squeeze=False
        )
        all_precip = pd.concat(frame["precip_change"] for frame in changes.values())
        all_temp = pd.concat(frame["temp_change"] for frame in changes.values())
        # Identical axes across facets: a horizon that looks calmer must BE
        # calmer, not merely be drawn on a kinder scale. The padding is generous
        # because every point now carries a text label beside it.
        pad_x = max((all_precip.max() - all_precip.min()) * 0.22, 1.0)
        pad_y = max((all_temp.max() - all_temp.min()) * 0.22, 0.2)
        xlim = (min(all_precip.min(), 0) - pad_x, max(all_precip.max(), 0) + pad_x)
        ylim = (min(all_temp.min(), 0) - pad_y, max(all_temp.max(), 0) + pad_y)

        points = 0
        for index, (axis, name) in enumerate(zip(axes[0], names)):
            frame = changes[name]
            for scenario in scenarios:
                subset = frame[frame["scenario"] == scenario]
                axis.scatter(
                    subset["precip_change"],
                    subset["temp_change"],
                    s=24,
                    color=SCENARIO_COLORS.get(scenario, "0.2"),
                    alpha=0.9,
                    edgecolor="white",
                    linewidth=0.4,
                    zorder=3,
                )
                points += len(subset)
            axis.axhline(0.0, color="0.35", lw=0.6)
            axis.axvline(0.0, color="0.35", lw=0.6)
            style_scatter_axes(axis)
            axis.set_xlim(*xlim)
            axis.set_ylim(*ylim)
            axis.set_xlabel("Change in mean precipitation (%)")
            panel_label(axis, "abcde"[index])
        axes[0][0].set_ylabel("Change in mean temperature (°C)")
        for axis in axes[0][1:]:
            axis.tick_params(labelleft=False)
        legend = axes[0][0].legend(
            handles=scenario_handles(scenarios, with_historical=False),
            loc="upper left",
            frameon=False,
        )
        # Last, and after every limit is final: label placement is measured in
        # display space, so anything that moves the points invalidates it.
        for index, (axis, name) in enumerate(zip(axes[0], names)):
            label_points(axis, changes[name], blockers=[legend] if index == 0 else ())
        panels = ", ".join(
            f"{letter}) {name} {horizons[name][0]}–{horizons[name][1]}"
            for letter, name in zip("abcde", names)
        )
        caveat(
            fig,
            f"Panels: {panels}. One point per (model, scenario, member) per horizon; "
            f"{points} points drawn. Axes are shared across panels, so a calmer-looking "
            "horizon is a calmer one. Each point is annotated with its model; colour still "
            "encodes scenario alone. Marginal densities removed: these points are not a "
            "distribution.",
        )
        fig.savefig(out_path, dpi=dpi)
        plt.close(fig)
    return points


def figure_cloud_combined(
    changes: dict[str, pd.DataFrame],
    horizons: dict[str, tuple[int, int]],
    out_path: Path,
    dpi: int,
) -> int:
    """Every horizon on ONE pair of axes, horizon by marker shape.

    Retained at the owner's request alongside the faceted view: the faceted one
    answers "what does this horizon look like", this one answers "how far does
    the cloud travel between horizons", and the second question is the reason
    the overlay existed in the shipped figure. Marker encodes horizon — which is
    neither model nor member, so the scenario-only rule is untouched.
    """
    names = list(horizons)
    scenarios = sorted(
        {s for frame in changes.values() for s in frame["scenario"].unique()}
    )

    with plt.rc_context(cartographic_map._publication_rc()):
        fig, axis = new_figure(0.62)
        points = 0
        for index, name in enumerate(names):
            frame = changes[name]
            marker = HORIZON_MARKERS[index % len(HORIZON_MARKERS)]
            for scenario in scenarios:
                subset = frame[frame["scenario"] == scenario]
                axis.scatter(
                    subset["precip_change"],
                    subset["temp_change"],
                    s=26,
                    marker=marker,
                    color=SCENARIO_COLORS.get(scenario, "0.2"),
                    alpha=0.9,
                    edgecolor="white",
                    linewidth=0.4,
                    zorder=3,
                )
                points += len(subset)
        axis.axhline(0.0, color="0.35", lw=0.6)
        axis.axvline(0.0, color="0.35", lw=0.6)
        style_scatter_axes(axis)
        axis.set_xlabel("Change in mean precipitation (%)")
        axis.set_ylabel("Change in mean temperature (°C)")
        # Margins widened before labelling: the annotations sit outside the data
        # extent, and autoscale does not know they exist.
        axis.margins(0.14)

        handles = scenario_handles(scenarios, with_historical=False)
        for index, name in enumerate(names):
            start, end = horizons[name]
            handles.append(
                plt.Line2D(
                    [],
                    [],
                    color="0.35",
                    lw=0,
                    marker=HORIZON_MARKERS[index % len(HORIZON_MARKERS)],
                    markersize=4,
                    label=f"{name} {start}–{end}",
                )
            )
        legend = axis.legend(handles=handles, loc="upper left", frameon=False, ncols=2)
        # One call over ALL horizons, not one per horizon: every point on these
        # axes has to be visible to the placer, or a `near` label lands on a
        # `far` point that the placer for `near` never knew about.
        label_points(axis, pd.concat(changes.values()), blockers=[legend])
        caveat(
            fig,
            f"All horizons on one pair of axes; {points} points, one per (model, scenario, "
            "member) per horizon. Colour encodes scenario and marker encodes horizon — "
            "neither encodes model, which is annotated directly instead. Companion to the "
            "faceted cloud: this view shows how far the cloud travels between horizons.",
        )
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
    """Precipitation (a) and temperature (b) change by calendar month."""
    scenarios = sorted(changes["scenario"].unique())

    with plt.rc_context(cartographic_map._publication_rc()):
        fig, axes = new_figure(0.38, ncols=2)
        traces = 0
        panels = (
            (axes[0], "precip_change", "Precipitation change (%)", "a"),
            (axes[1], "temp_change", "Temperature change (°C)", "b"),
        )
        for axis, column, ylabel, letter in panels:
            for _, group in changes.groupby(["model", "scenario", "member"], sort=True):
                group = group.sort_values("month")
                axis.plot(
                    group["month"],
                    group[column],
                    color=SCENARIO_COLORS.get(group["scenario"].iloc[0], "0.2"),
                    lw=0.9,
                    alpha=0.9,
                    marker="o",
                    markersize=2.0,
                )
                traces += 1
            axis.axhline(0.0, color="0.35", lw=0.6)
            style_series_axes(axis)
            axis.set_xticks(range(1, 13), MONTH_LABELS)
            axis.set_xlabel("Month")
            axis.set_ylabel(ylabel)
            panel_label(axis, letter)
        axes[0].legend(
            handles=scenario_handles(scenarios, with_historical=False),
            loc="upper left",
            frameon=False,
            ncols=len(scenarios),
        )
        caveat(
            fig,
            f"Horizon {name} ({horizon[0]}–{horizon[1]}) against {ref[0]}–{ref[1]}. Each "
            "future calendar month is differenced against the SAME historical calendar "
            f"month, using only {horizon[0]}–{horizon[1]}. One trace per (model, scenario, "
            f"member): {traces // 2} per panel, colour encoding scenario alone.",
        )
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

    Not part of the proposed set. It exists so the monthly-semantics claim can be
    judged as a picture and a number rather than taken on trust — the proposal
    must sit ON the table's markers, and the shipped definition must visibly not.
    """
    model, scenario, member = combination

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
    ]

    with plt.rc_context(cartographic_map._publication_rc()):
        fig, axes = new_figure(0.38, ncols=2)
        panels = (
            (axes[0], "precip", "Precipitation change (%)", "a"),
            (axes[1], "temp", "Temperature change (°C)", "b"),
        )
        for axis, variable, ylabel, letter in panels:
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
            )
            axis.plot(
                prop["month"],
                prop[column],
                color=SCENARIO_COLORS.get(scenario, "0.2"),
                lw=1.2,
                label="Proposed definition",
            )
            axis.scatter(
                rows["month"],
                table_values,
                s=26,
                facecolor="none",
                edgecolor="black",
                linewidth=0.7,
                label="Change-factor table",
                zorder=4,
            )
            axis.axhline(0.0, color="0.35", lw=0.6)
            style_series_axes(axis)
            axis.set_xticks(range(1, 13), MONTH_LABELS)
            axis.set_xlabel("Month")
            axis.set_ylabel(ylabel)
            panel_label(axis, letter)
        axes[0].legend(loc="lower left", frameon=False)
        caveat(
            fig,
            f"{model} {SCENARIO_LABELS.get(scenario, scenario)} {member}, horizon "
            f"{horizon_name} ({horizon[0]}–{horizon[1]}). The proposal reproduces the "
            "authoritative table. The shipped figure differences each future month against "
            "the historical ANNUAL mean and averages 2015–2100 instead of the horizon, so it "
            "disagrees with the table it sits beside.",
        )
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
    for variable in VARIABLES:
        target = (
            out_dir / "overview" / f"annual-{VARIABLES[variable]['name'].lower()}.png"
        )
        count = figure_annual(annual, variable, ref, target, args.dpi)
        print(f"wrote {target.relative_to(out_dir)}  ({count} traces/panel)")

    changes = {
        name: annual_change(series, ref, span) for name, span in horizons.items()
    }
    for target, draw in (
        (out_dir / "overview" / "change-factor-cloud.png", figure_cloud_faceted),
        (
            out_dir / "overview" / "change-factor-cloud-combined.png",
            figure_cloud_combined,
        ),
    ):
        count = draw(changes, horizons, target, args.dpi)
        print(f"wrote {target.relative_to(out_dir)}  ({count} points)")

    proposed = {}
    for name, span in horizons.items():
        proposed[name] = monthly_change(series, ref, span)
        window_dir = out_dir / "windows" / f"{name}-{span[0]}-{span[1]}"
        window_dir.mkdir(parents=True, exist_ok=True)
        target = window_dir / "monthly-change-factors.png"
        count = figure_monthly(proposed[name], name, span, ref, target, args.dpi)
        print(f"wrote {target.relative_to(out_dir)}  ({count} traces/panel)")

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
