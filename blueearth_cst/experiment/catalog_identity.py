"""Catalog source identity for admission, without loading data adapters."""

import os
from pathlib import Path
from typing import Any


def root_identity(locator: str | Path) -> str:
    """Anchor local roots to the run directory; preserve remote locators."""
    if not isinstance(locator, (str, Path)) or not str(locator):
        raise ValueError("catalog root must be a nonempty path or URI")
    if "://" in str(locator):
        return str(locator)
    return Path(os.path.abspath(locator)).as_posix()


def catalog_input(document: dict[str, Any], source: Path) -> dict[str, Any]:
    """Record YAML and its root context; consumers own HydroMT validation.

    Only the standard catalog root precedence is inspected: explicit root,
    first available candidate root, or the catalog's own directory. Source
    expansion, driver defaults and adapter validation belong to HydroMT.
    """
    if not isinstance(document, dict):
        raise ValueError(f"catalog must contain a YAML mapping: {source}")
    meta = document.get("meta", {})
    if not isinstance(meta, dict):
        raise ValueError(f"catalog meta must contain a mapping: {source}")
    root = meta.get("root", source.parent)
    if "root" not in meta and "roots" in meta:
        roots = meta["roots"]
        if not isinstance(roots, list) or any(not isinstance(r, str) for r in roots):
            raise ValueError(f"catalog roots must contain a list of paths: {source}")
        root = next(
            (candidate for candidate in roots if os.path.exists(candidate)), None
        )
        if root is None:
            raise ValueError(f"no catalog candidate root exists: {source}")
    return {"document": document, "root": root_identity(root)}


def legacy_catalog_input(value: dict[str, Any]) -> dict[str, Any]:
    """Project a retained v1 catalog for comparison without rewriting its bytes."""
    if set(value) != {"document", "resolved_sources"}:
        raise ValueError("legacy catalog lacks its document and resolved sources")
    resolved = value["resolved_sources"]
    if not isinstance(resolved, dict) or not isinstance(resolved.get("meta"), dict):
        raise ValueError("legacy catalog lacks its retained root context")
    root = resolved["meta"].get("root")
    return {"document": value["document"], "root": root_identity(root)}
