"""Load toolbox colour defaults without importing Matplotlib or Cartopy.

Settings are read once per plotting process. Start a new process after editing
the YAML; workflow renderer params track its content independently of science.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml

from blueearth_cst.shared.plot_setup import PLOTTING_DIR

PALETTE_PATH = PLOTTING_DIR / "colors.yml"


def load_palette(path: Path = PALETTE_PATH) -> dict[str, Any]:
    """Read the closed palette format and resolve named comparison colours.

    Colours must be six-digit hex strings. Colormap names are resolved by
    Matplotlib when drawn, keeping workflow parsing free of plotting imports.
    """
    path = Path(path)
    try:
        palette = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ValueError(f"Cannot read plot palette {path}: {exc}") from exc

    def fail(message: str) -> None:
        raise ValueError(f"{path}: {message}")

    sections = {
        "default_cycle",
        "datasets",
        "evaluation",
        "scenarios",
        "climate_maps",
        "historical_maps",
        "precip_anomaly",
        "neutral",
        "elevation",
        "waterbodies",
        "map_features",
        "river_order",
    }
    if not isinstance(palette, dict) or set(palette) != sections:
        fail(f"palette requires exactly these sections: {sorted(sections)}")
    for section in sections - {"precip_anomaly", "elevation", "river_order"}:
        block = palette[section]
        if not isinstance(block, dict) or not block:
            fail(f"{section} must be a nonempty mapping")
        if any(not isinstance(key, str) or not key for key in block):
            fail(f"{section} keys must be nonempty strings")
    for section in ("default_cycle", "scenarios", "neutral", "map_features"):
        for name, color in palette[section].items():
            if (
                section in {"neutral", "map_features"}
                and isinstance(color, str)
                and re.fullmatch(r"(?:0(?:\.\d+)?|1(?:\.0+)?)", color)
            ):
                continue
            if not isinstance(color, str) or not re.fullmatch(
                r"#[0-9a-fA-F]{6}", color
            ):
                fail(f"{section}.{name} must be a quoted six-digit hex colour")
    cycle = palette["default_cycle"]
    if len({color.lower() for color in cycle.values()}) != len(cycle):
        fail("default_cycle colours must be distinct")
    for section in ("datasets", "evaluation"):
        for name, reference in palette[section].items():
            if not isinstance(reference, str) or reference not in cycle:
                fail(f"{section}.{name} must name a default_cycle colour")
        palette[section] = {name: cycle[ref] for name, ref in palette[section].items()}
    if set(palette["evaluation"]) != {"observed", "simulated"}:
        fail("evaluation requires observed and simulated")
    if not {"era5", "chirps"} <= set(palette["datasets"]):
        fail("datasets requires era5 and chirps")
    if set(palette["climate_maps"]) != {
        "precip",
        "temp",
        "temp_crossing_freezing",
        "pet",
    }:
        fail("climate_maps requires precip, temp, temp_crossing_freezing and pet")
    for name, cmap in {
        **palette["climate_maps"],
        "precip_anomaly": palette["precip_anomaly"],
    }.items():
        if not isinstance(cmap, str) or not cmap.strip():
            fail(f"{name} must be a nonempty Matplotlib colormap name")
    if set(palette["historical_maps"]) != {"temp", "precip_pet"}:
        fail("historical_maps requires temp and precip_pet")
    for name, ramp in palette["historical_maps"].items():
        if not isinstance(ramp, dict) or set(ramp) != {"colors", "under", "over"}:
            fail(f"historical_maps.{name} requires colors, under and over")
        if not isinstance(ramp["colors"], list) or len(ramp["colors"]) < 2:
            fail(f"historical_maps.{name}.colors requires at least two colours")
        for color in [*ramp["colors"], ramp["under"], ramp["over"]]:
            if not isinstance(color, str) or not re.fullmatch(
                r"#[0-9a-fA-F]{6}", color
            ):
                fail(f"historical_maps.{name} requires quoted six-digit hex colours")
    for name, values in {
        "elevation": palette["elevation"],
        "river_order": palette["river_order"],
        **palette["waterbodies"],
    }.items():
        if not isinstance(values, list) or len(values) < 2:
            fail(f"{name} requires at least two hex colours")
        if any(
            not isinstance(color, str) or not re.fullmatch(r"#[0-9a-fA-F]{6}", color)
            for color in values
        ):
            fail(f"{name} requires quoted six-digit hex colours")
    return palette


_PALETTE = load_palette()
DEFAULT_COLOR_CYCLE: tuple[str, ...] = tuple(_PALETTE["default_cycle"].values())
DATASET_COLORS: dict[str, str] = _PALETTE["datasets"]
EVALUATION_COLORS: dict[str, str] = _PALETTE["evaluation"]
SCENARIO_COLORS: dict[str, str] = _PALETTE["scenarios"]
CLIMATE_COLORMAPS: dict[str, str] = _PALETTE["climate_maps"]
PRECIP_ANOMALY_COLORMAP: str = _PALETTE["precip_anomaly"]
HISTORICAL_MAP_COLORS: dict[str, dict[str, Any]] = _PALETTE["historical_maps"]
NEUTRAL_COLORS: dict[str, str] = _PALETTE["neutral"]
MAP_FEATURE_COLORS: dict[str, str] = _PALETTE["map_features"]
ELEVATION_COLORS: tuple[str, ...] = tuple(_PALETTE["elevation"])
RIVER_ORDER_COLORS: tuple[str, ...] = tuple(_PALETTE["river_order"])
WATERBODY_COLORS: dict[str, tuple[str, ...]] = {
    name: tuple(values) for name, values in _PALETTE["waterbodies"].items()
}
