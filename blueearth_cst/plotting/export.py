"""Explicit-figure export without workflow logging or pyplot state."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from matplotlib.figure import Figure


def export_figure(
    fig: Figure,
    path: str | os.PathLike[str],
    *,
    annotation: str = "none",
    title: str = "",
    caption: str = "",
    **kwargs,
) -> None:
    """Create destination parents and save the supplied figure.

    Save options and exceptions pass through unchanged. The caller owns the
    figure's size, style, backend and lifetime.
    """
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    if annotation not in ("none", "caption", "title"):
        raise ValueError("annotation must be none, caption, or title")
    if annotation != "none":
        import matplotlib as mpl

        from blueearth_cst.plotting.fundamentals import add_caption

        text = caption if annotation == "caption" else title
        if not text:
            raise ValueError(f"Missing {annotation} for {path}")
        with mpl.rc_context(
            {"savefig.dpi": kwargs.get("dpi", mpl.rcParams["savefig.dpi"])}
        ):
            crop = kwargs.get("bbox_inches") == "tight"
            pad_inches = kwargs.get("pad_inches", mpl.rcParams["savefig.pad_inches"])
            add_caption(
                fig, text, above=annotation == "title", crop=crop, pad_inches=pad_inches
            )
            if crop:
                kwargs.pop("bbox_inches")
                kwargs.pop("pad_inches", None)
    fig.savefig(path, **kwargs)
