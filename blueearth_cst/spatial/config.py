# Current guidance: dev/reference/toolbox-spatial-foundation.md
"""Parse and validate the Workflow 1 spatial-foundation configuration."""

from __future__ import annotations

import ast
import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import PureWindowsPath
from typing import Any

from blueearth_cst.shared.snake_utils import (
    ADVANCED_SETTINGS,
    DEFAULT_BASIN_INDEX,
    DEFAULT_HYDROGRAPHY,
)

#: Default ceiling on automatically-derived subbasins, PER PARENT BASIN
#: The value lives in ``config/advanced_settings.yml`` under
#: ``defaults:``, with the colour-ramp argument for the number itself; a project
#: overrides it with ``basin.delineation.max_subbasins``.
DEFAULT_MAX_SUBBASINS_PER_BASIN = ADVANCED_SETTINGS["defaults"]["max_subbasins"]
#: The hard cap, unchanged: `Bnnn-Snn` gives two digits of local subbasin
#: number. A limit, where the above is a default — which is why it stays HERE,
#: beside the identifier format it comes from, rather than joining its
#: neighbour in the settings file: no config may relax it.
MAX_LOCAL_SUBBASIN_NUMBER = 99
#: How far a gauge may be moved onto the derived river network, in metres. The
#: VALUE lives in ``config/advanced_settings.yml`` under ``defaults:``; a project
#: overrides it with ``basin.delineation.snap_tolerance_m``.
DEFAULT_GAUGE_SNAP_TOLERANCE_M = ADVANCED_SETTINGS["defaults"]["snap_tolerance_m"]
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
    """Validated spatial-foundation settings resolved from the project config."""

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
    """Resolve the canonical gauge file, rejecting a legacy-only config.

    ``basin.output_locations`` is canonical. The former
    ``workflows.build_model.output_locations`` key is accepted only ALONGSIDE
    it, naming the same file, so a staged migration can carry both; two
    different populated paths are an error rather than a precedence rule.

    **A legacy-only config raises.** It cannot be honoured, which is why the
    former ``FutureWarning`` was not enough. Gauge points became an input to
    rule 1.02 ``delineate_subbasins_and_rivers``, and the spatial contract requires that
    rule's params to be a pure function of ``project`` + ``basin`` —
    it is declared by all three workflows and the other two carry no
    ``workflows.build_model`` section at all. So the legacy key reaches the
    evaluation rule that reads the registry back but NOT the rule that writes
    it: delineation silently falls back to the automatic partition, and the
    failure surfaces a whole model build later as observation station IDs that
    the registry does not contain. Measured on a real project 2026-08-08.
    """
    canonical = basin_cfg.get("output_locations")
    legacy = model_cfg.get("output_locations")
    has_canonical = not _is_unset(canonical)
    has_legacy = not _is_unset(legacy)

    if has_canonical and has_legacy:
        canonical_path = _path_value(canonical, "basin.output_locations")
        legacy_path = _path_value(legacy, "workflows.build_model.output_locations")
        if _normalized_path(canonical_path) != _normalized_path(legacy_path):
            raise ValueError(
                "Conflicting gauge-point paths: basin.output_locations="
                f"{canonical!r} and workflows.build_model.output_locations="
                f"{legacy!r}. Keep only basin.output_locations, or make the "
                "two values identical during migration."
            )
        return canonical_path
    if has_canonical:
        return _path_value(canonical, "basin.output_locations")
    if has_legacy:
        legacy_path = _path_value(legacy, "workflows.build_model.output_locations")
        raise ValueError(
            "workflows.build_model.output_locations is no longer honoured on "
            "its own: move the path to basin.output_locations.\n\n"
            "    basin:\n"
            f"      output_locations: {legacy_path}\n\n"
            "The gauge points control the basin/subbasin partition, not only "
            "the Wflow outputs, and only the canonical key reaches the rule "
            "that delineates it. Left under the legacy key they would be "
            "ignored there, the subbasins would come from the automatic "
            "fallback, and the run would fail much later comparing observation "
            "station IDs against a registry built without them.\n"
            "Migrating a project also renumbers wflow_id: see "
            "config/templates/README.md."
        )
    return None


def _parse_region(value: object) -> dict[str, Any]:
    """Parse a HydroMT basin-region dictionary and validate its primary kind."""
    if isinstance(value, str):
        try:
            value = ast.literal_eval(value)
        except (SyntaxError, ValueError) as exc:
            raise ValueError(
                "basin.region must be a dictionary or a string containing "
                "one valid dictionary literal"
            ) from exc
    if not isinstance(value, Mapping):
        raise TypeError(
            "basin.region must be a mapping or mapping-literal string, "
            f"got {type(value).__name__}"
        )
    region = dict(value)
    kinds = [key for key in ("basin", "subbasin") if key in region]
    if len(kinds) != 1:
        raise ValueError(
            "basin.region must contain exactly one hydrologic region key: "
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
        raise TypeError(f"basin.spatial_sources.{key} must be a non-empty string")
    return value


def parse_spatial_config(
    basin_cfg: Mapping[str, Any], model_cfg: Mapping[str, Any] | None = None
) -> SpatialConfig:
    """Parse the spatial-foundation contract from sectioned workflow config."""
    if not isinstance(basin_cfg, Mapping):
        raise TypeError("basin must be a mapping")
    if model_cfg is None:
        model_cfg = {}
    if not isinstance(model_cfg, Mapping):
        raise TypeError("workflows.build_model must be a mapping")

    # `C-13`/`C-42`: the section is `basin.delineation:` and it holds the
    # subbasin ceiling, the snap tolerance and the river threshold — three
    # keys that were flat on `basin:` and describe one thing.
    delineation_cfg = basin_cfg.get("delineation", {}) or {}
    if not isinstance(delineation_cfg, Mapping):
        raise TypeError("basin.delineation must be a mapping")
    # `max_count` was a GLOBAL budget shared across parents;
    # `max_subbasins` is a per-parent ceiling. Rejected BY NAME rather than
    # ignored, because `basin` has no closed schema (unlike
    # `advanced_settings`, whose `_ADVANCED_SETTINGS_SCHEMA` rejects unknown
    # keys) — so a leftover `max_count` would be dropped in silence and the
    # project would run at the new default instead of the value its author
    # wrote. On a three-basin project that is a silently tripled partition.
    if "max_count" in delineation_cfg:
        raise ValueError(
            "basin.delineation.max_count was removed. "
            "See dev/reference/toolbox-spatial-foundation.md. "
            "Rename it to 'max_subbasins' — and note the MEANING changed "
            "with the name: max_count was one global budget shared across all "
            "parent basins, max_subbasins is a ceiling applied to each parent "
            "independently. A multi-basin project keeping the same number will "
            "produce more subbasins than before, which is the point of the "
            f"rename. The default is now {DEFAULT_MAX_SUBBASINS_PER_BASIN}."
        )
    # `C-15`: one section for every spatial input. `hydrography` and
    # `basin_index` were flat beside `spatial_sources:` and are the same kind
    # of thing — a named dataset — so all three live under `basin.sources:`.
    sources_cfg = basin_cfg.get("sources", {}) or {}
    if not isinstance(sources_cfg, Mapping):
        raise TypeError("basin.sources must be a mapping")

    hydrography = sources_cfg.get("hydrography", DEFAULT_HYDROGRAPHY)
    basin_index = sources_cfg.get("basin_index", DEFAULT_BASIN_INDEX)
    if not isinstance(hydrography, str) or not hydrography.strip():
        raise TypeError("basin.sources.hydrography must be a non-empty string")
    if basin_index is not None and (
        not isinstance(basin_index, str) or not basin_index.strip()
    ):
        raise TypeError("basin.sources.basin_index must be null or a non-empty string")

    return SpatialConfig(
        region=_parse_region(basin_cfg.get("region")),
        resolution=_positive_float(
            basin_cfg.get("resolution", 0.00833333),
            "basin.resolution",
        ),
        hydrography=hydrography,
        basin_index=basin_index,
        gauge_points_path=resolve_gauge_points_path(basin_cfg, model_cfg),
        # `maximum` is now EXACTLY right rather than incidentally right: the
        # ceiling counts subbasins within ONE parent, and `Bnnn-Snn` gives that
        # parent two digits of local subbasin number. Under the old global
        # budget the same bound was a loose over-estimate.
        max_subbasins_per_basin=_positive_int(
            delineation_cfg.get("max_subbasins", DEFAULT_MAX_SUBBASINS_PER_BASIN),
            "basin.delineation.max_subbasins",
            maximum=MAX_LOCAL_SUBBASIN_NUMBER,
        ),
        gauge_snap_tolerance_m=_positive_float(
            delineation_cfg.get("snap_tolerance_m", DEFAULT_GAUGE_SNAP_TOLERANCE_M),
            "basin.delineation.snap_tolerance_m",
        ),
        river_uparea_km2=_positive_float(
            delineation_cfg.get("river_uparea_km2", DEFAULT_RIVER_UPAREA_KM2),
            "basin.delineation.river_uparea_km2",
        ),
        sources=SpatialSources(
            rivers=_source_name(sources_cfg, "rivers", "rivers_lin2019_v1"),
            lulc=_source_name(sources_cfg, "lulc", "vito"),
            lai=_source_name(sources_cfg, "lai", "modis_lai"),
            soil=_source_name(sources_cfg, "soil", "soilgrids"),
        ),
    )
