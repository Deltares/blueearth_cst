"""Normalize authored generation methods at the configuration boundary."""

from __future__ import annotations

from copy import deepcopy
from typing import Any


def normalize_generation(document: dict[str, Any], schema_version: int) -> dict:
    """Keep legacy effective stochastic settings and an explicit delta section."""
    if not {"method", "stochastic", "delta_change"} & document.keys():
        return dict(document)
    method = document.get("method")
    if method not in ("stochastic", "delta_change"):
        raise ValueError("generation method must be stochastic or delta_change")
    if schema_version != 3:
        raise ValueError("method sections require project schema_version: 3")
    if set(document) != {"method", method} or not isinstance(document[method], dict):
        raise ValueError("generation requires exactly method and its selected mapping")
    settings = deepcopy(document[method])
    if method == "stochastic":
        from blueearth_cst.shared.config_composition import GENERATION_KEYS

        unknown = set(settings) - GENERATION_KEYS
        if unknown:
            raise ValueError(f"unknown stochastic settings: {sorted(unknown)}")
        return settings
    from blueearth_cst.experiment.delta_scenarios import validate_settings

    validate_settings(settings)
    return {"method": method, method: settings}


def generation_method(config: dict) -> str:
    """Return the selected producer from a composed project."""
    return (
        config.get("workflows", {})
        .get("generate_scenarios", {})
        .get("method", "stochastic")
    )
