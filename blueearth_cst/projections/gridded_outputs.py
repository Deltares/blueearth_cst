"""Validate the removed WF2 gridded-output configuration keys."""

from __future__ import annotations

from collections.abc import Mapping

REMOVED_GRIDDED_KEYS = ("save_grids", "save_gridded")


class RemovedGriddedOutputsError(ValueError):
    """Raised when a config requests the removed gridded-output branch."""


def validate_removed_gridded_options(config: Mapping[str, object]) -> list[str]:
    """Reject true removed keys and return warnings for stale false keys."""
    warnings = []
    for key in REMOVED_GRIDDED_KEYS:
        if key not in config:
            continue
        if config[key]:
            raise RemovedGriddedOutputsError(
                f"analyze_projections: `{key}: true` -- the gridded outputs "
                "were removed (S8-08c). `raw/{series_key}.nc` is the basin slice on "
                "the source grid and is always written, so the gridded series was a "
                "near-duplicate of it. Remove the key from your config."
            )
        warnings.append(
            f"`{key}` is obsolete and ignored (S8-08c); the gridded outputs "
            "were removed. Delete the key."
        )
    return warnings
