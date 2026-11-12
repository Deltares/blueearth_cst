"""Select a ready v2 scenario collection: engine discovery or an explicit selector."""

import csv
from pathlib import Path


class GeneratedCollectionUnavailable(ValueError):
    """The exact project request has no completed plan or ready collection."""


class GeneratedCollectionStale(ValueError):
    """The project request, live inputs, or selected immutable state changed."""


def discover_collections_v2(project_dir: Path) -> list[dict]:
    """List only checked v2 engine markers; ignore bare scenario data roots."""
    from blueearth_cst.experiment.scenario_collection_v2 import read_collection_v2

    engine = Path(project_dir).resolve() / "scenarios" / "_engine" / "collections"
    if not engine.exists():
        return []
    collections = []
    for directory in sorted(engine.iterdir()):
        if directory.is_symlink() or not directory.is_dir():
            raise ValueError(f"aliased collection engine directory: {directory}")
        marker = directory / "collection.json"
        if marker.exists():
            collections.append(read_collection_v2(marker))
    return collections


def resolve_explicit_collection_v2(
    selector: dict, project_dir: Path | str | None = None
) -> tuple[dict, dict]:
    """Resolve an explicit selection: ``{name}``, ``{id}`` or ``{manifest_path}``.

    ``name`` matches a set folder or an index alias; ``id`` is the leading
    characters of a collection identity (at least 12). Both resolve to a
    verified marker and full identity, so later renames never change what a
    frozen request selected. ``manifest_path`` is deprecated.
    """
    from blueearth_cst.experiment.scenario_collection_v2 import read_collection_v2

    if type(selector) is not dict or len(selector) != 1:
        raise ValueError(
            "scenario_collection takes exactly one of name, id or manifest_path"
        )
    ((key, value),) = selector.items()
    if key == "manifest_path":
        marker_path = Path(value)
    elif key in {"name", "id"}:
        if project_dir is None:
            raise ValueError(
                f"scenario_collection.{key} requires the project directory"
            )
        marker_path = _marker_for(Path(project_dir), key, value)
    else:
        raise ValueError(
            f"scenario_collection: unknown selector {key!r}; use name, id or manifest_path"
        )
    marker = read_collection_v2(marker_path)
    return (
        {
            "resolution_mode": "explicit-manifest",
            "manifest_path": marker_path.resolve().as_posix(),
            "collection_id": marker["collection_id"],
            "collection_revision": marker["collection_revision"],
        },
        marker,
    )


def _available(project_dir: Path) -> str:
    from blueearth_cst.experiment.collection_index import (
        INDEX_NAME,
        rebuild_collection_index,
    )

    index = project_dir / "scenarios" / INDEX_NAME
    if not index.is_file() and (project_dir / "scenarios").is_dir():
        rebuild_collection_index(project_dir)
    rows = (
        list(csv.DictReader(index.open(encoding="utf-8", newline="")))
        if index.is_file()
        else []
    )
    if not rows:
        return "no scenario collections exist; run scenario generation"
    return "available: " + "; ".join(
        f"{row['name']} ({row['method']}, {row['id'][:12]}, {row['created'][:10]})"
        for row in rows
    )


def _marker_for(project_dir: Path, key: str, value: object) -> Path:
    """The marker a name or short identity selects; unknown values list the sets."""
    from blueearth_cst.experiment.collection_index import (
        INDEX_NAME,
        collection_marker_paths,
    )
    from blueearth_cst.experiment.content_identity import read_canonical_json

    project_dir = project_dir.resolve()
    if type(value) is not str or not value:
        raise ValueError(f"scenario_collection.{key} must be nonempty text")
    if key == "id":
        if len(value) < 12 or any(c not in "0123456789abcdef" for c in value):
            raise ValueError(
                "scenario_collection.id must be at least 12 lowercase hex characters"
            )
        found = [
            path
            for path in collection_marker_paths(project_dir)
            if str(read_canonical_json(path).get("collection_id", "")).startswith(value)
        ]
    else:
        folder = project_dir / "scenarios" / value / "_engine" / "collection.json"
        found = [folder] if folder.is_file() else []
        index = project_dir / "scenarios" / INDEX_NAME
        if not found and index.is_file():
            with index.open(encoding="utf-8", newline="") as handle:
                for row in csv.DictReader(handle):
                    if value in row.get("aliases", "").split(";"):
                        found = [
                            path
                            for path in collection_marker_paths(project_dir)
                            if read_canonical_json(path).get("collection_id")
                            == row["id"]
                        ]
    if len(found) != 1:
        problem = "is ambiguous" if found else "matches no collection"
        raise ValueError(
            f"scenario_collection.{key} {value!r} {problem}; {_available(project_dir)}"
        )
    return found[0]
