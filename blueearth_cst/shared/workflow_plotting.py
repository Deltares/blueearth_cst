"""Workflow-local plotting selection and deprecated-key migration."""

import warnings
from collections.abc import Mapping

PLOTTING_DEFAULTS = {
    "analyze_climate": {"comparison": True, "source": False, "location": False},
    "build_model": {"spatial": True, "forcing": False, "location": False},
}
LEGACY_PLOTTING_KEYS = {
    "analyze_climate": {
        ("source_figures",): "source",
        ("location_figures",): "location",
        ("subbasin_figures",): "location",
    },
    "build_model": {
        ("forcing_plots",): "forcing",
        ("evaluation", "station_plots"): "location",
        ("evaluation", "subbasin_plots"): "location",
    },
}


def normalize_workflow_plotting(owner: str, body: Mapping) -> dict:
    """Validate local selections and migrate aliases without touching calculations.

    The project-wide annotation policy remains separate. Multiple declarations
    of one selection must agree; booleans are never inferred from strings or
    numbers. Workflows without a local plotting contract remain unchanged.
    """
    result = dict(body)
    if owner not in PLOTTING_DEFAULTS:
        return result
    block = body.get("plotting", {})
    if not isinstance(block, Mapping):
        raise ValueError(f"{owner}.plotting must be a mapping")
    if "annotation" in block:
        raise ValueError("plotting.annotation belongs in the project file")
    unknown = set(block) - set(PLOTTING_DEFAULTS[owner])
    if unknown:
        raise ValueError(f"Unknown {owner}.plotting options: {sorted(unknown)}")
    declarations = {key: [(f"plotting.{key}", value)] for key, value in block.items()}
    legacy = []
    for path, key in LEGACY_PLOTTING_KEYS[owner].items():
        parent = body
        for part in path[:-1]:
            parent = parent.get(part, {})
            if not isinstance(parent, Mapping):
                raise ValueError(f"{owner}.{part} must be a mapping")
        if path[-1] in parent:
            value = parent[path[-1]]
            name = ".".join(path)
            declarations.setdefault(key, []).append((name, value))
            legacy.append((path, key))
    for key, values in declarations.items():
        for name, value in values:
            if type(value) is not bool:
                raise ValueError(f"{owner}.{name} must be a boolean")
        if len({value for _, value in values}) > 1:
            names = ", ".join(name for name, _ in values)
            raise ValueError(
                f"Conflicting {owner}.plotting.{key} declarations: {names}"
            )
    if "plotting" not in body and not legacy:
        return result
    selected = dict(PLOTTING_DEFAULTS[owner])
    selected.update({key: values[0][1] for key, values in declarations.items()})
    for path, key in legacy:
        name = ".".join(path)
        warnings.warn(
            f"{owner}.{name} is deprecated; use {owner}.plotting.{key}",
            DeprecationWarning,
            stacklevel=2,
        )
        if len(path) == 1:
            result.pop(path[0])
        else:
            result[path[0]] = dict(result[path[0]])
            result[path[0]].pop(path[1], None)
    result["plotting"] = selected
    return result


def workflow_plotting(owner: str, body: Mapping) -> dict[str, bool]:
    """Resolve runtime defaults without adding absent sections to composed inputs."""
    normalized = normalize_workflow_plotting(owner, body)
    return normalized.get("plotting", dict(PLOTTING_DEFAULTS[owner]))
