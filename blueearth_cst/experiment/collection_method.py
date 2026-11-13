"""The one place that says which generation method produced a scenario collection.

Every WF4 consumer dispatches on :func:`collection_method` instead of comparing
schema strings, so a new collection format changes only this module.
"""

from __future__ import annotations

from typing import Any

STOCHASTIC = "stochastic"
DELTA_CHANGE = "delta_change"
METHODS = (STOCHASTIC, DELTA_CHANGE)

#: Formats retained before collections recorded ``scenario_type``.
_LEGACY = {
    "scenario-collection/3": STOCHASTIC,
    "delta-scenario-collection/1": DELTA_CHANGE,
}


def collection_method(marker: dict[str, Any]) -> str:
    """Return ``stochastic`` or ``delta_change`` for a collection marker.

    New markers state ``scenario_type`` (mirroring the intent); retained
    markers are recognised by their schema. Anything else is refused.
    """
    if not isinstance(marker, dict):
        raise ValueError("collection marker must be a mapping")
    if "scenario_type" in marker:
        method = marker["scenario_type"]
        if method not in METHODS:
            raise ValueError(f"unsupported collection scenario_type: {method!r}")
        return method
    schema = marker.get("schema_version")
    if schema in _LEGACY:
        return _LEGACY[schema]
    raise ValueError(f"unsupported collection schema: {schema!r}")


def is_delta(marker: dict[str, Any]) -> bool:
    """True for collections produced by the GCM delta method.

    Callers hold markers their reader already validated, so this keeps the
    former schema-string semantics: anything not recognised as delta is not
    delta. Use :func:`collection_method` where an unknown method must fail.
    """
    return (
        marker.get("scenario_type") == DELTA_CHANGE
        or _LEGACY.get(marker.get("schema_version")) == DELTA_CHANGE
    )
