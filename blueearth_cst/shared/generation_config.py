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

    settings = internal_delta_settings(settings)
    validate_settings(settings)
    return {"method": method, method: settings}


#: The delta ensemble when it follows analyze_projections, which supports
#: only CMIP6; delta generation refuses a different enabled WF2 ensemble.
PROJECTIONS_ENSEMBLE = "cmip6"
_CURRENT_DELTA_SCHEMA = "delta-change/3"


def internal_delta_settings(settings: dict[str, Any]) -> dict[str, Any]:
    """Authored delta-change settings (t037) in the hashed delta-change/3 shape.

    `future_periods: [names]` -> `projections: {ensemble, future_windows}` (the
    ensemble follows analyze_projections); `scale_pet: true` -> `pet: {mode:
    projection_ratio}`; an omitted `schema_version` is the current one. Exactly
    the shape authored before t037, so retained identities are unchanged.
    Read-only delta-change/1 and /2 settings pass through untouched.
    """
    version = settings.get("schema_version", _CURRENT_DELTA_SCHEMA)
    if version != _CURRENT_DELTA_SCHEMA:
        return settings
    retired = {
        "projections": "future_periods: [<analyze_projections period names>]",
        "pet": "scale_pet: true",
    }
    found = sorted(key for key in retired if key in settings)
    if found:
        raise ValueError(
            "delta_change: "
            + "; ".join(
                f"`{key}` is replaced (t037) by `{retired[key]}`" for key in found
            )
        )
    result = {
        key: value
        for key, value in settings.items()
        if key not in {"future_periods", "scale_pet"}
    }
    result["schema_version"] = _CURRENT_DELTA_SCHEMA
    projections: dict[str, Any] = {"ensemble": PROJECTIONS_ENSEMBLE}
    if "future_periods" in settings:
        projections["future_windows"] = settings["future_periods"]
    result["projections"] = projections
    scale_pet = settings.get("scale_pet", False)
    if type(scale_pet) is not bool:
        raise ValueError("delta_change.scale_pet must be true or false")
    if scale_pet:
        result["pet"] = {"mode": "projection_ratio"}
    return result


def public_delta_settings(settings: dict[str, Any]) -> dict[str, Any]:
    """Inverse of `internal_delta_settings` for writers of execution YAML."""
    if (
        settings.get("schema_version") != _CURRENT_DELTA_SCHEMA
        or "projections" not in settings
    ):
        return settings
    result = {
        key: value
        for key, value in settings.items()
        if key not in {"projections", "pet"}
    }
    projections = settings["projections"]
    if "future_windows" in projections:
        result["future_periods"] = projections["future_windows"]
    if "pet" in settings:
        result["scale_pet"] = True
    return result


def generation_method(config: dict) -> str:
    """Return the selected producer from a composed project."""
    return (
        config.get("workflows", {})
        .get("generate_scenarios", {})
        .get("method", "stochastic")
    )
