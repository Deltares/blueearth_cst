"""Resolve the project-wide presentation setting without changing scientific config."""

from collections.abc import Mapping
from typing import Any


def resolve_annotation(config: Mapping[str, Any]) -> str:
    """Validate plotting settings and return none, caption, or title."""
    plotting_config = config.get("plotting", {})
    if not isinstance(plotting_config, Mapping):
        raise ValueError("plotting must be a mapping with optional annotation")
    unknown = set(plotting_config) - {"annotation"}
    if unknown:
        raise ValueError(f"Unknown plotting settings: {sorted(unknown)}")
    annotation = plotting_config.get("annotation", "none")
    if not isinstance(annotation, str) or annotation not in {
        "none",
        "caption",
        "title",
    }:
        raise ValueError("plotting.annotation must be none, caption, or title")
    return annotation
