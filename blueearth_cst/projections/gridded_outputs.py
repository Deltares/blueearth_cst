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
                "optional branch was removed (S8-08c). Native source slices are always "
                "written to raw/; gridded diagnostic changes and spatial maps are now "
                "default WF2 outputs. Remove the obsolete key from your config."
            )
        warnings.append(
            f"`{key}` is obsolete and ignored (S8-08c); the old optional branch "
            "was removed. Gridded diagnostics are now default outputs. Delete the key."
        )
    return warnings
