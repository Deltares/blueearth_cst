"""Parse and validate the Workflow 1 spatial-foundation configuration."""

from __future__ import annotations

import ast
import os
import warnings
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import PureWindowsPath
from typing import Any

from blueearth_cst.shared.snake_utils import DEFAULT_BASIN_INDEX, DEFAULT_HYDROGRAPHY

#: Default ceiling on automatically-derived subbasins, PER PARENT BASIN
#: (ADR 0003 §11). Eleven, not twelve or twenty: twelve is the practical limit
#: for a qualitative colour ramp a reader can tell apart (ColorBrewer ``Set3``
#: and ``Paired`` both stop there), so 11 keeps ONE basin's subbasin map legible
#: with a legend entry per unit. The argument holds per basin only — a
#: three-basin project reaches 33 units and exceeds any qualitative ramp
#: regardless, which is the figure's problem to solve, not the default's.
DEFAULT_MAX_SUBBASINS_PER_BASIN = 11
#: The hard cap, unchanged: `Bnnn-Snn` gives two digits of local subbasin
#: number. A limit, where the above is a default.
MAX_LOCAL_SUBBASIN_NUMBER = 99
DEFAULT_GAUGE_SNAP_TOLERANCE_M = 10_000.0
DEFAULT_RIVER_UPAREA_KM2 = 32.0


@dataclass(frozen=True)
class SpatialSources:
    """Catalog entry names for the engine-neutral spatial layers."""

    rivers: str = "rivers_lin2019_v1"
    lulc: str = "vito"
    lai: str = "modis_lai"
    soil: str = "soilgrids"


@dataclass(frozen=True)
class SpatialConfig:
    """Validated spatial-foundation settings resolved from the snake config."""

    region: dict[str, Any]
    resolution: float
    hydrography: str
    basin_index: str | None
    gauge_points_path: str | None
    max_subbasins_per_basin: int
    gauge_snap_tolerance_m: float
    river_uparea_km2: float
    sources: SpatialSources


def _is_unset(value: object) -> bool:
    """Return whether an optional path uses a supported unset spelling."""
    return value is None or str(value) == "None"


def _path_value(value: object, key: str) -> str:
    """Return one non-empty filesystem path value."""
    if not isinstance(value, (str, os.PathLike)):
        raise TypeError(f"{key} must be a path string, got {type(value).__name__}")
    text = os.fspath(value)
    if not text.strip():
        raise ValueError(f"{key} must not be empty")
    return text


def _normalized_path(value: str | os.PathLike[str]) -> str:
    """Normalize a path for conflict comparison without requiring it to exist."""
    text = os.fspath(value)
    windows_path = PureWindowsPath(text)
    if windows_path.drive or "\\" in text:
        return str(windows_path).casefold()
    return os.path.normcase(os.path.normpath(text))


def resolve_gauge_points_path(
    basin_cfg: Mapping[str, Any], model_cfg: Mapping[str, Any]
) -> str | None:
    """Resolve the canonical gauge file with one-release legacy compatibility.

    ``shared.basin.gauge_points`` is canonical. The former
    ``workflows.model_creation.output_locations`` key remains accepted for one
    release, but two different populated paths are an error rather than a
    precedence rule.
    """
    canonical = basin_cfg.get("gauge_points")
    legacy = model_cfg.get("output_locations")
    has_canonical = not _is_unset(canonical)
    has_legacy = not _is_unset(legacy)

    if has_canonical and has_legacy:
        canonical_path = _path_value(canonical, "shared.basin.gauge_points")
        legacy_path = _path_value(
            legacy, "workflows.model_creation.output_locations"
        )
        if _normalized_path(canonical_path) != _normalized_path(legacy_path):
            raise ValueError(
                "Conflicting gauge-point paths: shared.basin.gauge_points="
                f"{canonical!r} and workflows.model_creation.output_locations="
                f"{legacy!r}. Keep only shared.basin.gauge_points, or make the "
                "two values identical during migration."
            )
        return canonical_path
    if has_canonical:
        return _path_value(canonical, "shared.basin.gauge_points")
    if has_legacy:
        warnings.warn(
            "workflows.model_creation.output_locations is deprecated; move the "
            "path to shared.basin.gauge_points. The legacy key will be removed "
            "after one compatibility release.",
            FutureWarning,
            stacklevel=2,
        )
        return _path_value(legacy, "workflows.model_creation.output_locations")
    return None


def _parse_region(value: object) -> dict[str, Any]:
    """Parse a HydroMT basin-region dictionary and validate its primary kind."""
    if isinstance(value, str):
        try:
            value = ast.literal_eval(value)
        except (SyntaxError, ValueError) as exc:
            raise ValueError(
                "shared.basin.region must be a dictionary or a string containing "
                "one valid dictionary literal"
            ) from exc
    if not isinstance(value, Mapping):
        raise TypeError(
            "shared.basin.region must be a mapping or mapping-literal string, "
            f"got {type(value).__name__}"
        )
    region = dict(value)
    kinds = [key for key in ("basin", "subbasin") if key in region]
    if len(kinds) != 1:
        raise ValueError(
            "shared.basin.region must contain exactly one hydrologic region key: "
            "'basin' or 'subbasin'"
        )
    return region


def _positive_float(value: object, key: str) -> float:
    """Return a finite positive float for a named config key."""
    if isinstance(value, bool):
        raise TypeError(f"{key} must be a positive number, got bool")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{key} must be a positive number, got {value!r}") from exc
    if not number > 0 or number == float("inf"):
        raise ValueError(f"{key} must be finite and > 0, got {value!r}")
    return number


def _positive_int(value: object, key: str, maximum: int | None = None) -> int:
    """Return a bounded positive integer for a named config key."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{key} must be an integer, got {value!r}")
    if value < 1:
        raise ValueError(f"{key} must be >= 1, got {value}")
    if maximum is not None and value > maximum:
        raise ValueError(f"{key} must be <= {maximum}, got {value}")
    return value


def _source_name(sources_cfg: Mapping[str, Any], key: str, default: str) -> str:
    """Return one non-empty catalog source name."""
    value = sources_cfg.get(key, default)
    if not isinstance(value, str) or not value.strip():
        raise TypeError(
            f"shared.basin.spatial_sources.{key} must be a non-empty string"
        )
    return value


def parse_spatial_config(
    basin_cfg: Mapping[str, Any], model_cfg: Mapping[str, Any] | None = None
) -> SpatialConfig:
    """Parse the spatial-foundation contract from sectioned workflow config."""
    if not isinstance(basin_cfg, Mapping):
        raise TypeError("shared.basin must be a mapping")
    if model_cfg is None:
        model_cfg = {}
    if not isinstance(model_cfg, Mapping):
        raise TypeError("workflows.model_creation must be a mapping")

    automatic_cfg = basin_cfg.get("automatic_subbasins", {}) or {}
    if not isinstance(automatic_cfg, Mapping):
        raise TypeError("shared.basin.automatic_subbasins must be a mapping")
    # ADR 0003 §11: `max_count` was a GLOBAL budget shared across parents;
    # `max_per_basin` is a per-parent ceiling. Rejected BY NAME rather than
    # ignored, because `shared.basin` has no closed schema (unlike
    # `advanced_settings`, whose `_ADVANCED_SETTINGS_SCHEMA` rejects unknown
    # keys) — so a leftover `max_count` would be dropped in silence and the
    # project would run at the new default instead of the value its author
    # wrote. On a three-basin project that is a silently tripled partition.
    if "max_count" in automatic_cfg:
        raise ValueError(
            "shared.basin.automatic_subbasins.max_count was removed in ADR 0003 "
            "§11. Rename it to 'max_per_basin' — and note the MEANING changed "
            "with the name: max_count was one global budget shared across all "
            "parent basins, max_per_basin is a ceiling applied to each parent "
            "independently. A multi-basin project keeping the same number will "
            "produce more subbasins than before, which is the point of the "
            f"rename. The default is now {DEFAULT_MAX_SUBBASINS_PER_BASIN}."
        )
    sources_cfg = basin_cfg.get("spatial_sources", {}) or {}
    if not isinstance(sources_cfg, Mapping):
        raise TypeError("shared.basin.spatial_sources must be a mapping")

    hydrography = basin_cfg.get("hydrography", DEFAULT_HYDROGRAPHY)
    basin_index = basin_cfg.get("basin_index", DEFAULT_BASIN_INDEX)
    if not isinstance(hydrography, str) or not hydrography.strip():
        raise TypeError("shared.basin.hydrography must be a non-empty string")
    if basin_index is not None and (
        not isinstance(basin_index, str) or not basin_index.strip()
    ):
        raise TypeError("shared.basin.basin_index must be null or a non-empty string")

    return SpatialConfig(
        region=_parse_region(basin_cfg.get("region")),
        resolution=_positive_float(
            basin_cfg.get("resolution", 0.00833333),
            "shared.basin.resolution",
        ),
        hydrography=hydrography,
        basin_index=basin_index,
        gauge_points_path=resolve_gauge_points_path(basin_cfg, model_cfg),
        # `maximum` is now EXACTLY right rather than incidentally right: the
        # ceiling counts subbasins within ONE parent, and `Bnnn-Snn` gives that
        # parent two digits of local subbasin number. Under the old global
        # budget the same bound was a loose over-estimate.
        max_subbasins_per_basin=_positive_int(
            automatic_cfg.get("max_per_basin", DEFAULT_MAX_SUBBASINS_PER_BASIN),
            "shared.basin.automatic_subbasins.max_per_basin",
            maximum=MAX_LOCAL_SUBBASIN_NUMBER,
        ),
        gauge_snap_tolerance_m=_positive_float(
            basin_cfg.get(
                "gauge_snap_tolerance_m", DEFAULT_GAUGE_SNAP_TOLERANCE_M
            ),
            "shared.basin.gauge_snap_tolerance_m",
        ),
        river_uparea_km2=_positive_float(
            basin_cfg.get("river_uparea_km2", DEFAULT_RIVER_UPAREA_KM2),
            "shared.basin.river_uparea_km2",
        ),
        sources=SpatialSources(
            rivers=_source_name(sources_cfg, "rivers", "rivers_lin2019_v1"),
            lulc=_source_name(sources_cfg, "lulc", "vito"),
            lai=_source_name(sources_cfg, "lai", "modis_lai"),
            soil=_source_name(sources_cfg, "soil", "soilgrids"),
        ),
    )
