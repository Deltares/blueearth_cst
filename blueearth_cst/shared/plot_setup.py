"""Shared plotting setup paths, appearance preferences and renderer identity."""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Any

import yaml

PLOTTING_DIR = Path(__file__).resolve().parents[2] / "config/plotting"


def load_style(path: Path | None = None) -> dict[str, Any]:
    """Read appearance preferences; keep Matplotlib validation in the renderer."""
    path = PLOTTING_DIR / "style.yml" if path is None else Path(path)
    try:
        style = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ValueError(f"Cannot read plot style {path}: {exc}") from exc
    expected = {"page", "typography", "charts", "maps", "evaluation"}
    if not isinstance(style, dict) or set(style) != expected:
        raise ValueError(f"{path}: style requires exactly {sorted(expected)}")
    for name, block in style.items():
        if not isinstance(block, dict) or not block:
            raise ValueError(f"{path}: {name} must be a nonempty mapping")
    numeric = {
        "page": ("figure_width_mm", "raster_dpi", "pdf_fonttype"),
        "typography": ("base", "tick", "legend", "title", "caveat"),
    }
    for section, names in numeric.items():
        for name in names:
            value = style[section].get(name)
            if (
                type(value) not in (int, float)
                or not math.isfinite(value)
                or value <= 0
            ):
                raise ValueError(
                    f"{path}: {section}.{name} must be a positive finite number"
                )
    charts = style["charts"]
    for group in (
        "widths",
        "heights",
        "padding",
        "fonts",
        "lines",
        "markers",
        "rcparams",
        "label_box",
    ):
        if not isinstance(charts.get(group), dict) or not charts[group]:
            raise ValueError(f"{path}: charts.{group} must be a nonempty mapping")
    markers = charts.get("fallback_markers")
    if (
        not isinstance(markers, list)
        or not markers
        or any(not isinstance(value, str) or not value for value in markers)
    ):
        raise ValueError(
            f"{path}: charts.fallback_markers must be a nonempty list of marker strings"
        )
    _check_maps(style["maps"], path)
    return style


#: The closed ``maps`` block: scalar pt sizes, pt lists, and the rest by name.
_MAP_NUMBERS = (
    "river_width",
    "river_halo_width",
    "location_label_wrap",
    "basin_outline_width",
    "divide_width",
    "location_size",
    "location_label_size",
    "scale_bar_segments",
)
_MAP_LISTS = ("river_order_widths", "divide_dash")


def _check_maps(maps: dict[str, Any], path: Path) -> None:
    expected = {*_MAP_NUMBERS, *_MAP_LISTS, "location_marker"}
    if set(maps) != expected:
        raise ValueError(f"{path}: maps requires exactly {sorted(expected)}")

    def positive(value: Any) -> bool:
        return type(value) in (int, float) and math.isfinite(value) and value > 0

    for name in _MAP_NUMBERS:
        if not positive(maps[name]):
            raise ValueError(f"{path}: maps.{name} must be a positive finite number")
    for name in _MAP_LISTS:
        values = maps[name]
        if not isinstance(values, list) or not values or not all(map(positive, values)):
            raise ValueError(f"{path}: maps.{name} must be a list of positive numbers")
    if not isinstance(maps["location_marker"], str) or not maps["location_marker"]:
        raise ValueError(f"{path}: maps.location_marker must be a marker string")


def setup_digest() -> str:
    """Track appearance files only in renderer params, outside scientific identity."""
    digest = hashlib.sha256()
    for name in ("colors.yml", "style.yml"):
        digest.update(name.encode("utf-8"))
        digest.update((PLOTTING_DIR / name).read_bytes())
    return digest.hexdigest()


STYLE_SETTINGS = load_style()
