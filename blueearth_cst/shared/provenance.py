"""Deterministic digest helpers for CST configuration provenance."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterable, Mapping
from pathlib import Path, PurePath
from typing import Any

_CANONICAL_JSON_OPTIONS = {
    "allow_nan": False,
    "ensure_ascii": False,
    "separators": (",", ":"),
    "sort_keys": True,
}


def canonical_data(value: Any) -> dict[str, Any]:
    """Return a deterministic, explicitly typed JSON-compatible value.

    Supported values are mappings, lists, tuples, paths, and YAML scalar
    primitives (``None``, booleans, integers, floats, and strings). Unsupported
    objects raise instead of being silently converted with ``str`` or ``repr``.

    Args:
        value: Value to convert to canonical data.

    Returns:
        Type-tagged data composed only of JSON-compatible primitives.

    Raises:
        TypeError: If ``value`` or a nested value has an unsupported type.
        ValueError: If a mapping has duplicate canonical keys.
    """
    if value is None:
        return {"type": "null"}
    if isinstance(value, bool):
        return {"type": "bool", "value": value}
    if isinstance(value, int):
        return {"type": "int", "value": str(value)}
    if isinstance(value, float):
        return {"type": "float", "value": _canonical_float(value)}
    if isinstance(value, str):
        return {"type": "str", "value": value}
    if isinstance(value, PurePath):
        return {"type": "path", "value": value.as_posix()}
    if isinstance(value, Mapping):
        return _canonical_mapping(value)
    if isinstance(value, list):
        return {"type": "list", "items": [canonical_data(item) for item in value]}
    if isinstance(value, tuple):
        return {"type": "tuple", "items": [canonical_data(item) for item in value]}
    raise TypeError(
        f"unsupported provenance value of type {type(value).__name__}; "
        "expected a mapping, list, tuple, path, or YAML scalar primitive"
    )


def canonical_sha256(value: Any) -> str:
    """Return the SHA-256 of a value's canonical typed JSON representation."""
    payload = json.dumps(canonical_data(value), **_CANONICAL_JSON_OPTIONS).encode(
        "utf-8"
    )
    return hashlib.sha256(payload).hexdigest()


def file_sha256(path: str | Path) -> str:
    """Return the SHA-256 digest of a file's exact bytes."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def effective_config_document(
    config: Mapping[Any, Any], advanced_settings: Mapping[Any, Any]
) -> dict[str, Any]:
    """Build the scientific configuration document used for digesting.

    Execution-only options such as cores, dry-run, and verbosity do not belong
    in this document. ``config`` is the configuration mapping actually resolved
    for the workflow, while ``advanced_settings`` is the validated toolbox-wide
    settings mapping.
    """
    if not isinstance(config, Mapping):
        raise TypeError("config must be a mapping")
    if not isinstance(advanced_settings, Mapping):
        raise TypeError("advanced_settings must be a mapping")
    return {
        "schema_version": 1,
        "project_config": config,
        "advanced_settings": advanced_settings,
    }


def effective_config_digest(
    config: Mapping[Any, Any], advanced_settings: Mapping[Any, Any]
) -> str:
    """Return the canonical SHA-256 of the effective scientific config."""
    return canonical_sha256(effective_config_document(config, advanced_settings))


def snapshot_bundle_digest(
    config: Mapping[Any, Any],
    advanced_settings: Mapping[Any, Any],
    source_config_path: str | Path,
    referenced_inputs: Iterable[Mapping[str, Any]],
) -> str:
    """Digest effective config, source bytes, and explicit referenced inputs.

    Each reference descriptor requires string ``kind`` and ``identifier``
    fields. A descriptor with ``path`` identifies a local file: its logical
    identifier and byte SHA-256 are included, but its machine-specific physical
    path is not. A descriptor without ``path`` is a logical identifier such as
    a catalog source name. Descriptor input order does not affect the digest.

    Args:
        config: Effective project configuration mapping.
        advanced_settings: Resolved toolbox-wide settings mapping.
        source_config_path: Source YAML whose exact bytes are hashed.
        referenced_inputs: Explicit file or logical reference descriptors.

    Returns:
        Canonical SHA-256 for the complete snapshot bundle.
    """
    references = [_reference_document(item) for item in referenced_inputs]
    references.sort(key=_canonical_json)
    document = {
        "schema_version": 1,
        "effective_config": effective_config_document(config, advanced_settings),
        "source_config_sha256": file_sha256(source_config_path),
        "referenced_inputs": references,
    }
    return canonical_sha256(document)


def _canonical_float(value: float) -> str:
    """Return an exact, platform-stable representation of a float."""
    if math.isnan(value):
        return "nan"
    if math.isinf(value):
        return "+inf" if value > 0 else "-inf"
    return value.hex()


def _canonical_mapping(value: Mapping[Any, Any]) -> dict[str, Any]:
    """Canonicalize and sort a mapping by the canonical form of each key."""
    items: list[tuple[str, dict[str, Any], dict[str, Any]]] = []
    seen_keys: set[str] = set()
    for key, item in value.items():
        canonical_key = canonical_data(key)
        sort_key = _typed_json(canonical_key)
        if sort_key in seen_keys:
            raise ValueError("mapping contains duplicate canonical keys")
        seen_keys.add(sort_key)
        items.append((sort_key, canonical_key, canonical_data(item)))
    items.sort(key=lambda item: item[0])
    return {
        "type": "mapping",
        "items": [
            {"key": key, "value": item_value} for _, key, item_value in items
        ],
    }


def _canonical_json(value: Any) -> str:
    """Serialize supported data canonically for stable sorting."""
    return json.dumps(canonical_data(value), **_CANONICAL_JSON_OPTIONS)


def _typed_json(value: dict[str, Any]) -> str:
    """Serialize an already canonicalized value without adding more tags."""
    return json.dumps(value, **_CANONICAL_JSON_OPTIONS)


def _reference_document(reference: Mapping[str, Any]) -> dict[str, str]:
    """Validate and normalize one snapshot reference descriptor."""
    if not isinstance(reference, Mapping):
        raise TypeError("each referenced input must be a mapping")
    allowed = {"kind", "identifier", "path"}
    unknown = set(reference) - allowed
    kind = reference.get("kind")
    identifier = reference.get("identifier")
    if unknown or not isinstance(kind, str) or not isinstance(identifier, str):
        raise ValueError(
            "each referenced input requires string 'kind' and 'identifier' "
            "fields, plus only an optional 'path'"
        )
    document = {"kind": kind, "identifier": identifier}
    if "path" in reference:
        path = reference["path"]
        if not isinstance(path, (str, PurePath)):
            raise TypeError("referenced input 'path' must be a string or path")
        document["sha256"] = file_sha256(path)
    return document
