"""Measured physical panel grids shared by the weather plotting adapters.

Explicit axes geometry preserves physical panel size. Draw callbacks receive
prepared data through closures; they must not compute new scientific results.
This module does not change global rcParams or existing plotting consumers.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from math import ceil, isfinite

from matplotlib.axes import Axes
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure


@dataclass(frozen=True)
class PanelLayout:
    """Physical grid measurements and row-major panel assignments."""

    columns: int
    rows: int
    panel_width_mm: float
    panel_height_mm: float
    canvas_width_mm: float
    canvas_height_mm: float
    cells: tuple[tuple[int, int], ...]


def choose_grid(
    panel_count: int,
    candidate_widths_mm: Sequence[tuple[int, float]],
    max_width_mm: float,
) -> tuple[int, int]:
    """Choose the first fitting measured candidate; never shrink panels."""
    if (
        isinstance(panel_count, bool)
        or not isinstance(panel_count, int)
        or panel_count < 1
    ):
        raise ValueError("panel_count must be a positive integer.")
    if (
        isinstance(max_width_mm, bool)
        or not isfinite(max_width_mm)
        or max_width_mm <= 0
    ):
        raise ValueError("max_width_mm must be positive and finite.")
    for columns, width in candidate_widths_mm:
        if isinstance(columns, bool) or not isinstance(columns, int) or columns < 1:
            raise ValueError("Column counts must be positive integers.")
        if isinstance(width, bool) or not isfinite(width) or width <= 0:
            raise ValueError("Measured candidate widths must be positive and finite.")
        if (panel_count == 1 and columns != 1) or (
            panel_count > 1 and (columns < 2 or columns > panel_count)
        ):
            continue
        if width <= max_width_mm + 1e-6:
            return columns, ceil(panel_count / columns)
    raise ValueError(
        "Panels do not fit max_width_mm; increase the limit or explicitly reduce panel dimensions."
    )


def arrange_panels(
    draw_panels: Sequence[Callable[[Axes], None]],
    *,
    panel_width_mm: float = 70,
    panel_height_mm: float = 70,
    preferred_columns: Sequence[int] = (3, 2),
    max_width_mm: float = 180,
    dpi: float = 300,
    single_panel_heading: bool = False,
) -> tuple[Figure, tuple[Axes, ...], PanelLayout]:
    """Draw row-major panels and fit their measured labels without scaling.

    Axis legends, headings and tick/axis labels participate in measurement.
    Callers adding figure-level decorations must reserve their space separately;
    do not add such decorations or tight-crop after this geometry is finalized.
    A single panel omits its title unless single_panel_heading is true. Keep
    variable units in axis labels when they would otherwise disappear with it.

    Returns the figure, occupied axes and measured layout. No figure is saved.
    Raises ValueError for invalid geometry/no fit, RuntimeError for unstable
    text extents or fixed-aspect axes that change the requested panel geometry.
    """
    count = len(draw_panels)
    if count < 1 or not all(callable(draw) for draw in draw_panels):
        raise ValueError("Supply at least one panel drawing callback.")
    if any(
        isinstance(v, bool) or not isfinite(v) or v <= 0
        for v in (panel_width_mm, panel_height_mm, max_width_mm, dpi)
    ):
        raise ValueError("Panel dimensions, maximum width and dpi must be positive.")
    if (
        not preferred_columns
        or len(set(preferred_columns)) != len(preferred_columns)
        or any(
            isinstance(c, bool) or not isinstance(c, int) or c < 2
            for c in preferred_columns
        )
    ):
        raise ValueError(
            "preferred_columns must contain distinct integers of at least two."
        )
    if not isinstance(single_panel_heading, bool):
        raise ValueError("single_panel_heading must be boolean.")
    candidates = (
        (1,) if count == 1 else tuple(c for c in preferred_columns if c <= count)
    )
    measured = []
    for columns in candidates:
        if columns * panel_width_mm > max_width_mm + 1e-6:
            continue
        rows = ceil(count / columns)
        width = columns * (panel_width_mm + 20)
        height = rows * (panel_height_mm + 20)
        fig = Figure(figsize=(width / 25.4, height / 25.4), dpi=dpi)
        FigureCanvasAgg(fig)
        axes = tuple(
            fig.add_axes((0, 0, panel_width_mm / width, panel_height_mm / height))
            for _ in draw_panels
        )
        try:
            for draw, ax in zip(draw_panels, axes, strict=True):
                draw(ax)
                if count == 1 and not single_panel_heading:
                    for loc in ("left", "center", "right"):
                        ax.set_title("", loc=loc)
            for _ in range(8):
                fig.canvas.draw()
                renderer = fig.canvas.get_renderer()
                left, right = [2.0] * columns, [2.0] * columns
                bottom, top = [2.0] * rows, [2.0] * rows
                for i, ax in enumerate(axes):
                    row, col = divmod(i, columns)
                    box = ax.get_window_extent(renderer)
                    tight = ax.get_tightbbox(renderer)
                    actual = (box.width * 25.4 / dpi, box.height * 25.4 / dpi)
                    if (
                        max(
                            abs(actual[0] - panel_width_mm),
                            abs(actual[1] - panel_height_mm),
                        )
                        > 1e-3
                    ):
                        raise RuntimeError(
                            "An axes aspect constraint changed the requested panel size."
                        )
                    left[col] = max(left[col], (box.x0 - tight.x0) * 25.4 / dpi + 2)
                    right[col] = max(right[col], (tight.x1 - box.x1) * 25.4 / dpi + 2)
                    bottom[row] = max(bottom[row], (box.y0 - tight.y0) * 25.4 / dpi + 2)
                    top[row] = max(top[row], (tight.y1 - box.y1) * 25.4 / dpi + 2)
                new_width = columns * panel_width_mm + sum(left) + sum(right)
                new_height = rows * panel_height_mm + sum(bottom) + sum(top)
                stable = max(abs(new_width - width), abs(new_height - height)) < 1e-3
                width, height = new_width, new_height
                fig.set_size_inches(width / 25.4, height / 25.4, forward=False)
                for i, ax in enumerate(axes):
                    row, col = divmod(i, columns)
                    x = (
                        sum(left[:col])
                        + sum(right[:col])
                        + col * panel_width_mm
                        + left[col]
                    )
                    y = (
                        height
                        - sum(top[: row + 1])
                        - sum(bottom[:row])
                        - (row + 1) * panel_height_mm
                    )
                    ax.set_position(
                        (
                            x / width,
                            y / height,
                            panel_width_mm / width,
                            panel_height_mm / height,
                        )
                    )
                if stable:
                    break
            else:
                raise RuntimeError(
                    "Panel decoration measurements did not converge in eight passes."
                )
            measured.append((columns, width))
            if width > max_width_mm + 1e-6:
                fig.clear()
                continue
            selected = choose_grid(count, measured, max_width_mm)
            if selected != (columns, rows):
                raise RuntimeError(
                    "Measured grid selection disagrees with the planner."
                )
            fig.canvas.draw()
            layout = PanelLayout(
                columns,
                rows,
                panel_width_mm,
                panel_height_mm,
                width,
                height,
                tuple(divmod(i, columns) for i in range(count)),
            )
            return fig, axes, layout
        except (ValueError, RuntimeError):
            fig.clear()
            raise
    choose_grid(count, measured, max_width_mm)
    raise AssertionError("Unreachable grid selection.")
