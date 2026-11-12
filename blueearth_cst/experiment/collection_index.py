"""Scenario collection names, folders and the rebuildable ``collections.csv`` index.

Each new collection lives in ``scenarios/<name>/`` with its marker at
``scenarios/<name>/_engine/collection.json``. Retained sets keep their
``scenarios/_engine/collections/<id>/`` markers and are listed under generated
names. Markers are the source of truth: the index is a view rebuilt from them,
and reuse is decided by collection identity, never by name.
"""

import csv
import io
import os
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from blueearth_cst.experiment.content_identity import read_canonical_json

INDEX_NAME = "collections.csv"
INDEX_HEADER = ("name", "aliases", "method", "id", "description", "created", "status")
_RESERVED = {"_engine", "collections.csv", "config", "series", "weathergenr", "delta"}
_WINDOWS_RESERVED = {"con", "prn", "aux", "nul"} | {
    f"{stem}{digit}" for stem in ("com", "lpt") for digit in range(1, 10)
}
_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}")
_HEX = re.compile(r"[0-9a-f]{8,}")


def validate_collection_name(name: object) -> str:
    """Return a safe folder name or refuse it with the rule it breaks."""
    if type(name) is not str or not _NAME.fullmatch(name) or name.endswith("."):
        raise ValueError(
            f"scenario collection name {name!r} must be 1-64 letters, digits, '.', '_' "
            "or '-', start with a letter or digit, and not end with '.'"
        )
    folded = name.casefold()
    if folded in _RESERVED or folded.split(".")[0] in _WINDOWS_RESERVED:
        raise ValueError(f"scenario collection name {name!r} is reserved")
    if _HEX.fullmatch(folded):
        raise ValueError(
            f"scenario collection name {name!r} looks like a collection identity"
        )
    return name


def default_stochastic_name(start: int, end: int) -> str:
    return f"stochastic-{start}-{end}"


def collection_marker_paths(project_root: Path) -> list[Path]:
    """Every marker candidate: named folders first, then retained engine records."""
    scenarios = Path(project_root).resolve() / "scenarios"
    if not scenarios.is_dir():
        return []
    paths = [
        folder / "_engine" / "collection.json"
        for folder in sorted(scenarios.iterdir())
        if folder.is_dir()
        and not folder.is_symlink()
        and folder.name != "_engine"
        and (folder / "_engine" / "collection.json").is_file()
    ]
    legacy = scenarios / "_engine" / "collections"
    if legacy.is_dir():
        paths += [
            folder / "collection.json"
            for folder in sorted(legacy.iterdir())
            if folder.is_dir() and (folder / "collection.json").is_file()
        ]
    return paths


def find_ready_marker(project_root: Path, collection_id: str) -> Path | None:
    """Locate a marker by full identity; the caller validates what it found."""
    found = [
        path
        for path in collection_marker_paths(project_root)
        if read_canonical_json(path).get("collection_id") == collection_id
    ]
    if len(found) > 1:
        raise ValueError(
            f"collection {collection_id[:12]} is published more than once: "
            + ", ".join(path.as_posix() for path in found)
        )
    return found[0] if found else None


def allocate_collection_name(project_root: Path, requested: str) -> str:
    """Return ``requested`` or the first free ``-2``, ``-3`` suffix.

    Any existing entry blocks a name, including incomplete folders and files
    left by earlier layouts; nothing is adopted or removed.
    """
    validate_collection_name(requested)
    scenarios = Path(project_root).resolve() / "scenarios"
    taken = (
        {entry.name.casefold() for entry in scenarios.iterdir()}
        if scenarios.is_dir()
        else set()
    )
    taken |= {
        alias.casefold()
        for row in _read_index(scenarios / INDEX_NAME)
        for alias in row["aliases"].split(";")
        if alias
    }
    candidate, suffix = requested, 1
    while candidate.casefold() in taken:
        suffix += 1
        candidate = validate_collection_name(f"{requested}-{suffix}")
    return candidate


def _read_index(path: Path) -> list[dict[str, str]]:
    try:
        with path.open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if tuple(reader.fieldnames or ()) != INDEX_HEADER:
                return []
            return list(reader)
    except (FileNotFoundError, csv.Error, UnicodeDecodeError):
        return []


def _description(intent: dict) -> str:
    spec = intent.get("scenario_spec") or {}
    runs = intent.get("run_count")
    window = spec.get("simulation_window")
    text = f"{runs} runs" if runs is not None else ""
    if isinstance(window, dict) and {"start", "end"} <= set(window):
        text += f", {window['start']}-{window['end']}"
    return text


def _row(marker_path: Path, name: str, aliases: list[str]) -> dict[str, str]:
    from blueearth_cst.experiment.collection_method import collection_method
    from blueearth_cst.experiment.scenario_collection_v2 import read_collection_v2

    try:
        marker = read_collection_v2(marker_path)
        status = "ready"
    except (ValueError, OSError, KeyError):
        marker, status = read_canonical_json(marker_path), "invalid"
    intent_path = marker_path.with_name("collection_intent.json")
    intent = read_canonical_json(intent_path) if intent_path.is_file() else {}
    created = datetime.fromtimestamp(marker_path.stat().st_mtime, timezone.utc)
    try:
        method = collection_method(marker)
    except ValueError:
        method = "unknown"
    return {
        "name": name,
        "aliases": ";".join(aliases),
        "method": method,
        "id": str(marker.get("collection_id", "")),
        "description": _description(intent),
        "created": created.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "status": status,
    }


def rebuild_collection_index(
    project_root: Path, *, alias: tuple[str, str] | None = None
) -> Path:
    """Rewrite ``scenarios/collections.csv`` from markers, atomically.

    ``alias`` is ``(collection_id, name)``: a configured name that differs from
    the reused folder's name. Aliases from the previous index are carried over;
    they are display data, so losing the index loses only them.
    Call under the project-execution lock.
    """
    scenarios = Path(project_root).resolve() / "scenarios"
    index = scenarios / INDEX_NAME
    previous = {row["id"]: row for row in _read_index(index) if row["id"]}
    rows = []
    for marker_path in collection_marker_paths(project_root):
        record = marker_path.parent
        legacy = record.parent.name == "collections"
        identity = str(read_canonical_json(marker_path).get("collection_id", ""))
        name = f"retained-{record.name}" if legacy else record.parent.name
        aliases = [
            a for a in previous.get(identity, {}).get("aliases", "").split(";") if a
        ]
        if (
            alias
            and alias[0] == identity
            and alias[1] != name
            and alias[1] not in aliases
        ):
            aliases.append(alias[1])
        rows.append(_row(marker_path, name, aliases))
    if scenarios.is_dir():
        for folder in sorted(scenarios.iterdir()):
            if (
                folder.is_dir()
                and folder.name != "_engine"
                and (folder / "_engine").is_dir()
                and not (folder / "_engine" / "collection.json").is_file()
            ):
                rows.append(
                    dict.fromkeys(INDEX_HEADER, "")
                    | {"name": folder.name, "status": "incomplete"}
                )
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=INDEX_HEADER, lineterminator="\n")
    writer.writeheader()
    writer.writerows(sorted(rows, key=lambda row: row["name"].casefold()))
    scenarios.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", dir=scenarios, delete=False, encoding="utf-8", newline=""
    ) as handle:
        handle.write(buffer.getvalue())
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(handle.name, index)
    return index
