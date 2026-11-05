"""Resolve schema-3 locators without changing HydroMT catalog internals."""

from __future__ import annotations

import ast
import copy
import os
from collections.abc import Mapping
from pathlib import Path, PureWindowsPath
from typing import Any

TOOLBOX_ROOT = Path(__file__).resolve().parents[2]


def resolve_path(value: str, source: str | Path, *, catalog: bool = False) -> str:
    """Resolve a declared file path beside its YAML, or a toolbox resource.

    Predefined catalog names and URLs retain their spelling. Foreign absolute
    paths fail rather than silently becoming relative paths.
    """
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{source}: expected a non-empty path, got {value!r}")
    if "://" in value:
        return value
    if value.startswith("toolbox:"):
        relative = value.removeprefix("toolbox:").replace("\\", "/")
        if (
            not relative
            or Path(relative).is_absolute()
            or PureWindowsPath(relative).drive
        ):
            raise ValueError(f"{source}: invalid toolbox locator {value!r}")
        path = (TOOLBOX_ROOT / relative).resolve()
        if not path.is_relative_to(TOOLBOX_ROOT.resolve()):
            raise ValueError(f"{source}: toolbox locator escapes toolbox: {value!r}")
        return path.as_posix()
    windows = PureWindowsPath(value)
    if windows.drive and (os.name != "nt" or not windows.is_absolute()):
        raise ValueError(f"{source}: path is not absolute on this platform: {value!r}")
    if (
        catalog
        and not any(char in value for char in "/\\")
        and not value.endswith((".yml", ".yaml"))
    ):
        return value
    path = Path(value.replace("\\", "/")).expanduser()
    if not path.is_absolute():
        path = Path(source).resolve().parent / path
    return path.resolve().as_posix()


def _resolve_fields(
    document: dict, fields: tuple[str, ...], source: str | Path
) -> None:
    """Resolve populated leaves of an explicitly enumerated file-field set."""
    for field in fields:
        parts = field.split(".")
        owner = document
        for part in parts[:-1]:
            owner = owner.get(part, {}) if isinstance(owner, Mapping) else {}
        if not isinstance(owner, dict) or parts[-1] not in owner:
            continue
        value = owner[parts[-1]]
        if value is None or value == "None":
            continue
        is_catalog = parts[-1] == "catalog"
        owner[parts[-1]] = (
            [resolve_path(item, source, catalog=is_catalog) for item in value]
            if isinstance(value, list) and is_catalog
            else resolve_path(value, source, catalog=is_catalog)
        )


def resolve_project_paths(document: Mapping[str, Any], source: str | Path) -> dict:
    """Return runtime project values; schema 2 keeps invocation-relative paths."""
    result = copy.deepcopy(dict(document))
    if document.get("schema_version") != 3:
        return result
    _resolve_fields(
        result,
        ("project.project_dir", "project.catalog", "basin.output_locations"),
        source,
    )
    region = result.get("basin", {}).get("region")
    if region is not None:
        literal = isinstance(region, str)
        parsed = ast.literal_eval(region) if literal else region
        if isinstance(parsed, dict):
            changed = False
            for key in ("basin", "subbasin", "geom"):
                if isinstance(parsed.get(key), str):
                    parsed[key] = resolve_path(parsed[key], source)
                    changed = True
            if changed:
                result["basin"]["region"] = str(parsed) if literal else parsed
    return result


def resolve_workflow_paths(
    document: Mapping[str, Any], source: str | Path, workflow: str, schema_version: int
) -> dict:
    """Resolve explicit workflow file fields beside their declaring YAML."""
    result = copy.deepcopy(dict(document))
    if schema_version != 3:
        return result
    fields = {
        "build_model": ("engine.build_config", "engine.waterbodies_config"),
        "analyze_projections": ("catalog",),
        "generate_scenarios": ("weathergen_config",),
        "simulate_system": ("scenario_collection.manifest_path",),
    }
    _resolve_fields(result, fields.get(workflow, ()), source)
    if workflow == "build_model":
        observations = result.get("observations") or {}
        for variable, value in observations.items():
            if value is not None and value != "None":
                observations[variable] = resolve_path(value, source)
    if workflow == "simulate_system":
        for definition in (result.get("indicator_set_definitions") or {}).values():
            if isinstance(definition, dict) and "config_path" in definition:
                definition["config_path"] = resolve_path(
                    definition["config_path"], source
                )
    return result
