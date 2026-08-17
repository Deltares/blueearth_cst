# -*- coding: utf-8 -*-
"""How the WF2 projection figures are DRAWN.

Split from :mod:`blueearth_cst.projections.projection_figures`, which declares
WHAT the family contains. The split is not tidiness: the Snakefile imports the
contract at parse time to build its target map, and ``snake_utils`` deliberately
keeps matplotlib out of that path (0.14 s to import, no pyplot). Putting the
drawing in the same module would drag matplotlib into every WF2 parse, including
the four dry-runs ``tests/test_cli.py`` performs. Contract: pure. Drawing: here.

Both WF2 producers import from this module — ``get_change_climate_proj_summary``
for the two cloud views off the stage-B merge, ``plot_proj_timeseries`` for the
annual overviews and the monthly figures — which is why the palette and the page
furniture live here rather than in either of them.

The page contract, reused rather than re-derived
------------------------------------------------
Everything draws under ``_publication_rc()`` at ``series_figure_size(...)``, in
CONSTRAINED layout, with the caveat carried by ``fig.supxlabel(..., wrap=True)``
— the same four decisions ``climate_analysis/climate_figures.py`` makes for the
WF1 series figures, imported from the same place it imports them from.

* ``layout="constrained"`` rather than ``tight_layout``, because the WF1 maps
  are built on it and a figure family that mixes the two cannot be made to agree
  on its margins.
* ``supxlabel`` rather than a hand-placed ``fig.text``, because it is part of the
  layout, so a long caveat re-flows instead of being drawn off the canvas.
  ``fig.text`` does not clip — it silently loses the tail, which is how the
  first draft of this design dropped "CMIP6 is a plausibility overlay, not a
  stress-test driver" off the right-hand edge of every figure.
"""

from __future__ import annotations

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import MaxNLocator

from blueearth_cst.shared import plot_style
from blueearth_cst.shared.cartographic_map import _publication_rc, series_figure_size

#: Scenario ink. Carried over from the producers this design replaces, verbatim:
#: the set is judged on layout and semantics, not on a palette change nobody
#: asked for. Scenario is the ONLY visual encoding in the whole family.
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

#: Marker per horizon, for the COMBINED cloud only. Horizon is neither model nor
#: member, so encoding it costs nothing the scenario-only rule protects.
HORIZON_MARKERS = ("o", "s", "^", "D", "v")

MONTH_LABELS = ("J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D")

#: Point size of a panel's ``a)`` label and of the model annotations on the
#: cloud. Matches WF1's value for text that sits INSIDE the axes rather than
#: labelling them.
FONT_SIZE_ANNOTATION = 6.0

#: Variables, in the order they appear on every two-panel figure.
VARIABLES = {
    "precip": {
        "name": "Precipitation",
        "absolute_units": "mm/day",
        "change_units": "%",
    },
    "temp": {
        "name": "Temperature",
        "absolute_units": "°C",
        "change_units": "°C",
    },
}


# ===========================================================================
# PAGE FURNITURE
# ===========================================================================


def new_figure(aspect, nrows=1, ncols=1, **kwargs):
    """A figure at the shared page width, in the WF1 rc and constrained layout.

    ``aspect`` is chosen per figure SHAPE rather than left at
    ``series_figure_size``'s 0.42 default, which sizes a single-axes series: a
    stacked pair needs most of double that height, a side-by-side pair needs
    about the single height with squarer panels.
    """
    size = series_figure_size(aspect)
    return plt.subplots(nrows, ncols, figsize=size, layout="constrained", **kwargs)


def style_series_axes(ax):
    """The WF1 series treatment: an L-frame with a horizontal-only grid."""
    ax.grid(axis="y", alpha=0.25, lw=0.5)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def style_scatter_axes(ax):
    """As above, but gridded on BOTH axes.

    A deliberate departure from the series treatment: on the change-factor cloud
    both coordinates carry meaning and both zero lines are drawn, so a
    horizontal-only grid would imply the x position is the approximate one.
    """
    ax.grid(alpha=0.25, lw=0.5)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def panel_label(ax, letter):
    """``a)``, ``b)`` … at the panel's top-left corner.

    Owner ruling, 2026-08-11, toolbox-wide: no titles above figures anywhere.
    The letter sits just outside the axes so it cannot collide with data or with
    the legend, and it carries no descriptive text — what the title used to say
    is in the y-label and in the caveat.
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


def caveat(fig, text):
    """The provenance line under the panels, as part of the layout."""
    fig.supxlabel(
        text,
        fontsize=plot_style.FONT_SIZE_CAVEAT,
        color=plot_style.COLOR_CAVEAT,
        wrap=True,
    )


def scenario_handles(scenarios, with_historical=True):
    """Legend proxies: historical plus scenarios, and nothing else.

    Proxies rather than the drawn artists, because the drawn artists are one per
    COMBINATION — labelling those is how "one trace per combination" becomes a
    legend naming every model, which is the contract this family holds.
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
LABEL_OFFSETS = (
    (5, 3),
    (-5, 3),
    (5, -9),
    (-5, -9),
    (0, 8),
    (0, -12),
    (10, -3),
    (-10, -3),
)


def label_points(ax, frame, blockers=()):
    """Annotate every cloud point with its model name, avoiding overprints.

    Owner ruling, 2026-08-11. This is the ONE place a model name appears on a
    figure, and it is a direct annotation rather than a visual channel: the point
    keeps its scenario colour and the shared marker, so nothing about the ink
    encodes which model it is.

    Placement is greedy against the real rendered extents rather than a fixed
    rotation of offsets. A rotation looks like it works and does not: it keys on
    the row's ORDER, while collisions are a fact about the row's POSITION, so two
    models differing by one percentage point drew their labels on top of each
    other while models at opposite corners were carefully given different
    offsets. Each label is drawn, measured, and moved to the next candidate if it
    overlaps one already placed; the first candidate is kept when none is clear,
    which is better than dropping a label silently.

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


def draw_annual_overview(annual, variable, reference, out_path, dpi=None):
    """Absolute (a) and anomaly (b) panels over the full historical/future series.

    ``annual`` is :func:`plot_proj_timeseries.annual_series`' frame. Returns the
    number of traces per panel, so a caller can assert it equals the number of
    resolved combinations.
    """
    dpi = plot_style.RASTER_DPI if dpi is None else dpi
    meta = VARIABLES[variable]
    scenarios = sorted(s for s in annual["scenario"].unique() if s != "historical")

    with plt.rc_context(_publication_rc()):
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
        historical_years = annual.loc[annual["scenario"] == "historical", "year"]
        if not historical_years.empty:
            transition = historical_years.max() + 0.5
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
            f"a) annual mean. b) anomaly against {reference[0]}–{reference[1]}, each "
            "model differenced against its own historical run; the dashed rule marks "
            "the historical/future handover. One trace per (model, scenario, member): "
            f"{traces // 2} per panel. Colour encodes scenario only — models and "
            "members are not distinguished. CMIP6 is a plausibility overlay, not a "
            "stress-test driver.",
        )
        fig.savefig(out_path, dpi=dpi)
        plt.close(fig)
    return traces // 2


def draw_cloud_faceted(changes, horizons, out_path, dpi=None):
    """The change-factor cloud, one panel per horizon on identical axes.

    No marginal KDEs: a kernel density over six points asserts a distribution the
    design does not construct. That is the one substantive removal from the
    seaborn ``JointGrid`` this replaces.
    """
    dpi = plot_style.RASTER_DPI if dpi is None else dpi
    names = list(horizons)
    scenarios = sorted(
        {s for frame in changes.values() for s in frame["scenario"].unique()}
    )

    with plt.rc_context(_publication_rc()):
        # Squarer than a series figure: both coordinates are changes in the same
        # sense, so the panel should not privilege one axis by stretching it.
        fig, axes = new_figure(
            0.50 if len(names) > 1 else 0.62, ncols=len(names), squeeze=False
        )
        all_precip = pd.concat(frame["precip_change"] for frame in changes.values())
        all_temp = pd.concat(frame["temp_change"] for frame in changes.values())
        # Identical axes across facets: a horizon that looks calmer must BE
        # calmer, not merely be drawn on a kinder scale. The padding is generous
        # because every point carries a text label beside it.
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
            f"{points} points drawn. Axes are shared across panels, so a "
            "calmer-looking horizon is a calmer one. Each point is annotated with its "
            "model; colour still encodes scenario alone. Marginal densities removed: "
            "these points are not a distribution.",
        )
        fig.savefig(out_path, dpi=dpi)
        plt.close(fig)
    return points


def draw_cloud_combined(changes, horizons, out_path, dpi=None):
    """Every horizon on ONE pair of axes, horizon by marker shape.

    Kept beside the faceted view at the owner's request (2026-08-11, reaffirmed
    2026-08-17): the faceted one answers "what does this horizon look like", this
    one answers "how far does the cloud travel between horizons", and the second
    question is the reason the overlay existed in the figure this replaces.
    Marker encodes horizon — neither model nor member, so the scenario-only rule
    is untouched.
    """
    dpi = plot_style.RASTER_DPI if dpi is None else dpi
    names = list(horizons)
    scenarios = sorted(
        {s for frame in changes.values() for s in frame["scenario"].unique()}
    )

    with plt.rc_context(_publication_rc()):
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
            f"All horizons on one pair of axes; {points} points, one per (model, "
            "scenario, member) per horizon. Colour encodes scenario and marker encodes "
            "horizon — neither encodes model, which is annotated directly instead. "
            "Companion to the faceted cloud: this view shows how far the cloud travels "
            "between horizons.",
        )
        fig.savefig(out_path, dpi=dpi)
        plt.close(fig)
    return points


def draw_monthly_change(changes, name, horizon, reference, out_path, dpi=None):
    """Precipitation (a) and temperature (b) change by calendar month.

    ``changes`` comes from ``plot_proj_timeseries.monthly_change_from_table`` —
    the authoritative table, not a recomputation. The caveat says so, because the
    definition is the thing that was wrong before and a reader has no other way
    to tell which one a figure used.
    """
    dpi = plot_style.RASTER_DPI if dpi is None else dpi
    scenarios = sorted(changes["scenario"].unique())

    with plt.rc_context(_publication_rc()):
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
            f"Horizon {name} ({horizon[0]}–{horizon[1]}) against "
            f"{reference[0]}–{reference[1]}, read from the change-factor table: each "
            "future calendar month is differenced against the SAME historical calendar "
            f"month, over {horizon[0]}–{horizon[1]} only. One trace per (model, "
            f"scenario, member): {traces // 2} per panel, colour encoding scenario "
            "alone. CMIP6 is a plausibility overlay, not a stress-test driver.",
        )
        fig.savefig(out_path, dpi=dpi)
        plt.close(fig)
    return traces // 2
