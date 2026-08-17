# -*- coding: utf-8 -*-
"""Render an indicator table as a heatmap, so a reader sees the pattern.

A CSV of signed changes answers "what is the number for this indicator under
this scenario" one cell at a time. The same table as a heatmap answers "which
indicators move, which way, and where the exceptions are" at a glance — which is
the question a reader of a basin assessment actually arrives with.

Ported from the upstream ``fao`` branch (``src/plot_utils/plot_table_statistics.py``,
board item ``t2608131847b``), and deliberately NOT verbatim. Four changes, each
because the original encodes an assumption this toolbox does not share:

1. **No hidden transpose.** The original takes scenarios as the index and flips
   the frame internally, so a caller who already oriented the table gets it
   silently rotated. Here the frame arrives in the orientation it is drawn in:
   rows are indicators, columns are scenarios or horizons.
2. **No ``sns.set_theme``.** That call rewrites the global rc, so any figure
   drawn afterwards in the same process inherits seaborn's styling instead of
   this toolbox's page contract. Everything here draws inside
   ``_publication_rc()`` and leaves the rc as it found it.
3. **Symmetric about zero by default**, rather than a hard-coded ±100. A
   diverging map whose midpoint is not the data's meaningful zero mis-reads by
   construction, and ±100 is only right when the changes happen to be percentages
   of that magnitude. The limits are derived from the data unless stated.
4. **The absolute-and-relative variant is not a third near-copy of the drawing
   code.** Its actual subject is a text format, so it is
   :func:`absolute_with_relative_annotations` — a function over two frames — and
   the drawing takes ``annotations`` like any other caller.

**This module is what keeps ``seaborn`` in ``pixi.toml``.** As of 2026-08-17 the
only other usage — WF2's change-factor ``JointGrid`` — was replaced, leaving the
dependency declared with nothing importing it, which is exactly the state that
got ``flit`` removed under ADR 0008. If this module is ever deleted, take
``seaborn`` with it or give it another job.

Colour: ``RdBu`` is kept from the original. Red–blue is the diverging pair that
survives the common colour-vision deficiencies (unlike red–green), and its
direction matches the domain reading for the tables this draws — blue for a
wetter or larger value, red for a drier or smaller one.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from blueearth_cst.shared import plot_style
from blueearth_cst.shared.cartographic_map import _publication_rc

#: Marker appended to a row label whose colour scale was inverted, so the
#: inversion is visible on the figure rather than only in the call.
INVERTED_MARKER = " *"

#: Default diverging colormap. See the module docstring for why red-blue.
DEFAULT_CMAP = "RdBu"


def flatten_columns(frame, separator=":\n"):
    """Collapse a MultiIndex column axis into one readable label per column.

    ``(ssp245, far)`` becomes ``ssp245:\\nfar``. The newline is in the default
    separator because these labels sit on the x axis of a wide heatmap, where a
    two-line label is the difference between readable and rotated 45 degrees.

    A single-level column axis is returned unchanged, so a caller need not ask
    which kind it has.
    """
    if not isinstance(frame.columns, pd.MultiIndex):
        return frame
    out = frame.copy()
    out.columns = [
        separator.join(str(part) for part in parts) for parts in frame.columns
    ]
    return out


def relative_change(frame, reference_column):
    """Percent change of every column against ``reference_column``.

    The reference column is kept in the result at 0%, because dropping it would
    leave the figure without the baseline the percentages are measured from.

    A zero reference yields no percentage — the ratio is meaningless there while
    the absolute value is not — and the cell comes back as NaN rather than as
    ``inf`` or a silently substituted 0.0. The original replaced both with zero,
    which draws "no change" over a cell where nothing was computed.
    """
    if reference_column not in frame.columns:
        raise KeyError(
            f"reference column {reference_column!r} is not in the table; "
            f"have {list(frame.columns)}"
        )
    reference = frame[reference_column].astype(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = frame.astype(float).div(reference, axis=0).sub(1.0).mul(100.0)
    return out.mask(reference.eq(0.0), other=np.nan)


def absolute_with_relative_annotations(absolute, relative, decimals=(1, 3)):
    """``"4.1 (-6.4%)"`` per cell — the absolute value with its relative change.

    ``decimals`` is ``(large, small)``: a value above 1 is shown to the first,
    below it to the second. One decimal on a discharge of 4.1 m3/s is right and
    on a baseflow index of 0.003 is not, and a single setting cannot serve both
    in one table.

    A cell whose relative change is NaN — the zero-reference case above — shows
    the absolute value alone rather than inventing a percentage for it.
    """
    large, small = decimals
    out = pd.DataFrame(index=absolute.index, columns=absolute.columns, dtype=object)
    for row in absolute.index:
        for column in absolute.columns:
            value = float(absolute.loc[row, column])
            digits = large if abs(value) > 1 else small
            text = f"{value:.{digits}f}"
            change = relative.loc[row, column]
            if pd.notna(change):
                text += f" ({float(change):+.1f}%)"
            out.loc[row, column] = text
    return out


def statistics_heatmap(
    values,
    out_path,
    *,
    annotations=None,
    x_label="Scenario",
    y_label="Indicator",
    colorbar_label="Relative change (%)",
    cmap=DEFAULT_CMAP,
    vmin=None,
    vmax=None,
    invert_rows=None,
    bold_columns_containing=None,
    column_separator=":\n",
    dpi=None,
):
    """Draw ``values`` as an annotated heatmap and write it to ``out_path``.

    Parameters
    ----------
    values : pandas.DataFrame
        Rows are indicators (the y axis), columns are scenarios or horizons (the
        x axis). Drawn in the orientation given — nothing is transposed here.
    annotations : pandas.DataFrame, optional
        Cell text, same shape as ``values``. Defaults to the values formatted to
        one decimal. Use :func:`absolute_with_relative_annotations` for the
        "absolute with relative change in brackets" form.
    vmin, vmax : float, optional
        Colour limits. Default is SYMMETRIC about zero at the largest absolute
        value present, so the midpoint of a diverging map is the meaningful
        zero rather than the middle of the data's range.
    invert_rows : sequence of str, optional
        Rows where "up" is bad — a low-flow deficit, a shortage — so the colour
        should read in the opposite direction. Their colour is inverted and the
        row label gains ``*``; **the annotation keeps the true value**, since
        the point is to make the reading consistent, not to restate the number.
    bold_columns_containing : str, optional
        Substring; matching columns are annotated in bold. For marking the
        scenario a report is arguing about.

    Returns the number of cells drawn, so a caller can assert the table was not
    silently empty.
    """
    if values.empty:
        raise ValueError("nothing to draw: the statistics table has no rows")

    values = flatten_columns(values, column_separator)
    if annotations is None:
        annotations = values.map(lambda v: "" if pd.isna(v) else f"{float(v):.1f}")
    else:
        annotations = flatten_columns(annotations, column_separator)
        if annotations.shape != values.shape:
            raise ValueError(
                f"annotations {annotations.shape} do not match values "
                f"{values.shape}; every cell needs its own text or none do"
            )

    # Invert the COLOUR of the flagged rows, never the annotation. A reader
    # comparing the figure against the table it renders must find the same
    # number in both; the `*` is what says the colour was flipped.
    painted = values.astype(float).copy()
    inverted = []
    for row in invert_rows or []:
        if row not in painted.index:
            continue
        painted.loc[row] = -painted.loc[row]
        inverted.append(row)
    if inverted:
        renamed = {row: f"{row}{INVERTED_MARKER}" for row in inverted}
        painted = painted.rename(index=renamed)
        annotations = annotations.rename(index=renamed)

    if vmin is None or vmax is None:
        finite = painted.to_numpy(dtype=float)
        finite = finite[np.isfinite(finite)]
        extent = float(np.max(np.abs(finite))) if finite.size else 1.0
        extent = extent or 1.0
        vmin = -extent if vmin is None else vmin
        vmax = extent if vmax is None else vmax

    dpi = plot_style.RASTER_DPI if dpi is None else dpi

    with plt.rc_context(_publication_rc()):
        # Height follows the row count: a fixed figure height either crushes a
        # 20-indicator table or strands a 3-row one in white space.
        width = plot_style.FIGURE_WIDTH_MM / plot_style.MM_PER_INCH
        height = min(max(1.4 + 0.28 * len(painted.index), 2.4), width * 1.4)
        fig, ax = plt.subplots(figsize=(width, height), layout="constrained")
        drawn = sns.heatmap(
            painted,
            ax=ax,
            annot=annotations.to_numpy(),
            fmt="",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            linewidth=0.5,
            annot_kws={"fontsize": plot_style.FONT_SIZE_TICK},
            cbar_kws={"label": colorbar_label},
        )
        drawn.set_xlabel(x_label, fontweight="bold")
        drawn.set_ylabel(y_label, fontweight="bold")
        ax.tick_params(axis="both", labelsize=plot_style.FONT_SIZE_TICK)
        plt.setp(ax.get_xticklabels(), rotation=0, ha="center")
        plt.setp(ax.get_yticklabels(), rotation=0)

        if bold_columns_containing is not None:
            _embolden(ax, painted, bold_columns_containing)

        fig.savefig(out_path, dpi=dpi)
        plt.close(fig)
    return int(painted.size)


def _embolden(ax, frame, keyword):
    """Bold the annotations of every column whose label contains ``keyword``.

    seaborn writes the cell texts row-major, so a text's column is its index
    modulo the column count — which is why this reads the position rather than
    asking the artist what cell it belongs to.
    """
    columns = list(frame.columns)
    if not columns:
        return
    for index, text in enumerate(ax.texts):
        if keyword in str(columns[index % len(columns)]):
            text.set_fontweight("bold")
