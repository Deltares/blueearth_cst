"""Select a ready v2 scenario collection: engine discovery or an explicit manifest."""

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


def resolve_explicit_collection_v2(selector: dict) -> tuple[dict, dict]:
    """Resolve an advanced v2 manifest selector without predecessor readers."""
    from blueearth_cst.experiment.scenario_collection_v2 import read_collection_v2

    if type(selector) is not dict or set(selector) != {"manifest_path"}:
        raise ValueError("scenario_collection accepts only manifest_path")
    marker_path = Path(selector["manifest_path"])
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
