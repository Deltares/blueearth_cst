"""Shared Matplotlib drawing fundamentals.

Helpers draw with the active preset (``blueearth_cst.plotting.preset``); wrap
rendering in ``use_preset`` to choose one. Backend selection belongs to callers.
"""

from collections.abc import Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.transforms import Affine2D, Bbox

from blueearth_cst.plotting.preset import active

MM = 1 / 25.4


MONTH_ABBR = (
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


def new_fig(
    height_mm: float,
    nrows: int = 1,
    ncols: int = 1,
    width_mm: float | None = None,
    **kw,
):
    """Constrained-layout figure at its printed size; width defaults to ``mid``."""
    preset = active()
    if width_mm is None:
        width_mm = preset.widths["mid"]
    fig, axs = plt.subplots(
        nrows,
        ncols,
        figsize=(width_mm * MM, height_mm * MM),
        layout="constrained",
        squeeze=False,
        **kw,
    )
    fig.get_layout_engine().set(
        w_pad=preset.padding["x"] * MM, h_pad=preset.padding["y"] * MM
    )
    return fig, axs


def ygrid(ax: plt.Axes) -> None:
    ax.grid(axis="y")


def year_ticks(ax: plt.Axes, years: np.ndarray, continuous: bool = False) -> None:
    """Sparse year labels with finer ticks, thinning ticks for long records."""
    if continuous:
        ax.set_xlim(years[0], years[-1] + 1)
    else:
        ax.set_xlim(years[0] - 0.5, years[-1] + 0.5)
    step = 10 if years.size > 35 else 5 if years.size > 12 else 2
    major = np.arange(years[0] + (-years[0]) % step, years[-1] + 1, step)
    ax.set_xticks(major)
    tick_step = max(1, int(np.ceil((years[-1] - years[0]) / 60)))
    ticks = np.arange(years[0] + (-years[0]) % tick_step, years[-1] + 1, tick_step)
    ax.set_xticks(np.setdiff1d(ticks, major), minor=True)
    edge, axis = active().lines["edge"], active().colors["axis"]
    ax.tick_params(axis="x", which="major", length=3, width=edge, color=axis)
    ax.tick_params(axis="x", which="minor", length=2, width=edge, color=axis)


def five_year_ticks(ax: plt.Axes, years: np.ndarray, rotation: int = 0) -> None:
    """Label five-year marks and show intervening years as short minor ticks."""
    major = np.arange(years[0] + (-years[0]) % 5, years[-1] + 1, 5)
    ax.set_xticks(major)
    ax.set_xticks(np.setdiff1d(years, major), minor=True)
    preset = active()
    edge, axis = preset.lines["edge"], preset.colors["axis"]
    ax.tick_params(
        axis="x",
        which="major",
        length=3,
        width=edge,
        color=axis,
        labelrotation=rotation,
        labelsize=preset.fonts["tick"],
    )
    ax.tick_params(axis="x", which="minor", length=2, width=edge, color=axis)


def heatmap_frame(ax: plt.Axes) -> None:
    """Keep the shared soft plot boundary while suppressing heatmap tick marks."""
    ax.tick_params(length=0)


def blank() -> Line2D:
    return Line2D([], [], lw=0, alpha=0, label=" ")


def top_legend(fig: Figure, *rows: list, handler_map: dict | None = None) -> None:
    """Outside upper-left figure legend; each row stays a row (no column-major interleave)."""
    rows = [r for r in rows if r]
    if not rows:
        return
    ncols = max(len(r) for r in rows)
    grid = [r + [blank()] * (ncols - len(r)) for r in rows]
    handles = [grid[i][j] for j in range(ncols) for i in range(len(grid))]
    fig.legend(
        handles=handles,
        loc="outside upper left",
        ncols=ncols,
        handler_map=handler_map,
        columnspacing=1.2,
        handletextpad=0.5,
    )


def colourbar(
    fig: Figure,
    mesh,
    axes: list,
    label: str,
    ticks: Sequence[float],
    stack_mm: float,
    shrink: float = 0.68,
) -> None:
    # Placed against the layout cell: right for heatmap axes, which fill their
    # cell. Aspect-locked map axes do not, so maps place their own bar
    # (``prepared_maps``) and share only ``style_colourbar``.
    cb = fig.colorbar(
        mesh,
        ax=axes,
        ticks=ticks,
        spacing="uniform",
        extendrect=True,
        pad=0.02,
        shrink=shrink,
        anchor=(0.0, 0.0),
        aspect=max(stack_mm / 3.0, 8),
    )
    style_colourbar(cb, label)


def style_colourbar(cb, label: str) -> None:
    """The one colourbar appearance: title above the bar, preset ticks and outline."""
    preset = active()
    colors, tick, edge = preset.colors, preset.fonts["tick"], preset.lines["edge"]
    cb.ax.set_title(label, color=colors["ink"], fontsize=tick, pad=10)
    cb.outline.set_edgecolor(colors["axis"])
    cb.outline.set_linewidth(edge)
    cb.ax.tick_params(
        labelsize=tick,
        labelcolor=colors["muted"],
        color=colors["axis"],
        width=edge,
        length=2.5,
    )


def add_caption(
    fig: Figure,
    text: str,
    *,
    above: bool = False,
    crop: bool = False,
    pad_inches: float = 0,
) -> None:
    """Add a caption below, or title above, without changing artists' physical size.

    Call after drawing the ordinary product and before its final captioned export.
    Layout and axes locators are frozen at the configured save DPI: the caption
    extends the canvas rather than asking constrained layout to resize the panels.
    Export at that same DPI; further figure resizing or artist edits need a new
    ordinary figure rather than rerunning layout on this finalized copy.
    ``above=True`` uses the label font and a band above the original canvas.
    ``crop=True`` freezes the tight artist bounds plus ``pad_inches`` before
    adding the band; export that finalized canvas without another tight crop.
    """
    preset = active()
    original_dpi = fig.dpi
    try:
        export_dpi = mpl.rcParams["savefig.dpi"]
        if export_dpi != "figure":
            fig.set_dpi(export_dpi)
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        old_width, old_height = fig.get_size_inches()
        bounds = (
            fig.get_tightbbox(renderer).padded(pad_inches)
            if crop
            else Bbox.from_bounds(0, 0, old_width, old_height)
        )
        width, height = bounds.width, bounds.height
        axes = [(ax, ax.get_position().frozen()) for ax in fig.axes]
        legends = [*fig.legends]
        legends.extend(ax.get_legend() for ax in fig.axes if ax.get_legend())
        anchors = [
            (
                legend,
                legend.get_bbox_to_anchor()
                .transformed(fig.dpi_scale_trans.inverted())
                .frozen(),
            )
            for legend in legends
        ]
        left = (
            min((box.x0 for _, box in axes), default=0.04) * old_width - bounds.x0
        ) / width
        right = (
            max((box.x1 for _, box in axes), default=0.96) * old_width - bounds.x0
        ) / width
        caption = fig.text(
            left,
            0,
            "",
            ha="left",
            va="bottom",
            fontsize=preset.fonts["label" if above else "note"],
            color=preset.colors["ink" if above else "muted"],
            linespacing=1.3,
        )
        available = (right - left) * width * fig.dpi
        lines, line = [], ""
        for word in text.split():
            trial = f"{line} {word}".strip()
            caption.set_text(trial)
            if caption.get_window_extent(renderer).width <= available:
                line = trial
                continue
            if line:
                lines.append(line)
                line = ""
            # Like textwrap.fill, break a token that cannot fit on its own line.
            for character in word:
                trial = line + character
                caption.set_text(trial)
                if line and caption.get_window_extent(renderer).width > available:
                    lines.append(line.rstrip())
                    line = character
                else:
                    line = trial
        lines.append(line)
        caption.set_text("\n".join(lines))
        extra = caption.get_window_extent(renderer).height / fig.dpi + 6 * MM
        fig.set_layout_engine("none")
        fig.set_size_inches(width, height + extra)
        mapping = (
            Affine2D()
            .scale(old_width / width, old_height / (height + extra))
            .translate(
                -bounds.x0 / width,
                (-bounds.y0 + (0 if above else extra)) / (height + extra),
            )
        )
        for ax, box in axes:
            ax.set_axes_locator(None)
            ax.set_position(mapping.transform_bbox(box))
        for legend, box in anchors:
            legend.set_bbox_to_anchor(
                Bbox.from_bounds(
                    box.x0 - bounds.x0,
                    box.y0 - bounds.y0 + (0 if above else extra),
                    box.width,
                    box.height,
                ),
                fig.dpi_scale_trans,
            )
        for artist in fig.get_children():
            if (
                artist is caption
                or artist in fig.axes
                or artist in fig.legends
                or artist is fig.patch
            ):
                continue
            transform = artist.get_transform()
            if transform.contains_branch(fig.transFigure):
                artist.set_transform(
                    transform - fig.transFigure + mapping + fig.transFigure
                )
        caption.set_position(
            (left, (height + 3 * MM if above else 3 * MM) / (height + extra))
        )
    finally:
        fig.set_dpi(original_dpi)


def inside_legend(
    fig: Figure,
    ax: plt.Axes,
    handles: list,
    bands: Sequence = (),
    lines: Sequence = (),
) -> None:
    """Stack legend entries on translucent white in the less occupied top corner."""
    if not handles:
        return
    legend = ax.legend(
        handles=handles,
        loc="upper right",
        ncols=1,
        frameon=True,
        facecolor="white",
        edgecolor="none",
        framealpha=0.88,
        borderpad=0.6,
        labelspacing=0.5,
        handletextpad=0.5,
    )
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    scores = []
    for location in ("upper right", "upper left"):
        legend.set_loc(location)
        box = legend.get_window_extent(renderer)
        xx, yy = np.meshgrid(
            np.linspace(box.x0, box.x1, 30), np.linspace(box.y0, box.y1, 20)
        )
        points = np.column_stack((xx.ravel(), yy.ravel()))
        overlap = 0
        for band in bands:
            for path in band.get_paths():
                overlap += path.contains_points(points, band.get_transform()).sum()
        for line in lines:
            path = line.get_transform().transform_path(line.get_path())
            if path.intersects_bbox(box, filled=False):
                overlap += points.shape[0]
        for collection in ax.collections:
            offsets = collection.get_offsets()
            if offsets.size:
                positions = collection.get_offset_transform().transform(offsets)
                overlap += (
                    sum(box.contains(x, y) for x, y in positions) * points.shape[0]
                )
        scores.append((overlap, location))
    legend.set_loc(min(scores, key=lambda item: item[0])[1])
