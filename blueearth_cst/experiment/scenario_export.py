"""Stable copies of the complete last successfully selected scenario collection."""

import os
import re
import shutil
import uuid
from pathlib import Path

from blueearth_cst.experiment.content_identity import atomic_record, read_canonical_json
from blueearth_cst.experiment.scenario_collection_v2 import read_collection_v2
from blueearth_cst.shared.provenance import file_sha256


def _collection_files(source: Path) -> tuple[dict[str, str], list[str]]:
    """Inventory the complete data tree without traversing linked directories."""
    if source.is_symlink() or source.is_junction() or source.resolve() != source:
        raise ValueError(f"scenario export source is aliased: {source}")
    files = {}
    directories = []
    pending = [source]
    while pending:
        for path in sorted(pending.pop().iterdir()):
            if path.is_symlink() or path.is_junction() or path.resolve() != path:
                raise ValueError(f"scenario export source is aliased: {path}")
            if path.is_dir():
                directories.append(path.relative_to(source).as_posix())
                pending.append(path)
            elif path.is_file():
                files[path.relative_to(source).as_posix()] = file_sha256(path)
            else:
                raise ValueError(f"unsupported scenario export source: {path}")
    return files, sorted(directories)


def export_scenario_collection(project_root: Path, manifest_path: Path) -> None:
    """Refresh user-facing copies under the caller's project-execution lock.

    Keep the immutable collection as the WF4 handoff. Stage changed files only;
    move previous copies aside for rollback rather than duplicating large series.
    Readers must wait for workflow completion because replacement is per file.
    """
    root = Path(project_root).resolve()
    manifest_path = Path(manifest_path).absolute()
    marker = read_collection_v2(manifest_path)
    if manifest_path != (
        root
        / "scenarios/_engine/collections"
        / Path(marker["data_root"]).name
        / "collection.json"
    ):
        raise ValueError("scenario export marker is outside the selected project")
    destination = root / "scenarios"
    tracking = destination / "_engine/current.json"
    source = root / marker["data_root"]
    files, directories = _collection_files(source)
    previous = {}
    previous_dirs = []
    old_record = None
    if tracking.exists():
        old_record = read_canonical_json(tracking)
        if old_record.get("schema_version") != "scenario-export/1":
            raise ValueError(f"unsupported scenario export record: {tracking}")
        previous = old_record["files"]
        previous_dirs = old_record.get("directories", [])
        if not isinstance(previous, dict):
            raise ValueError(f"invalid scenario export inventory: {tracking}")
        if not isinstance(previous_dirs, list):
            raise ValueError(f"invalid scenario export directories: {tracking}")
    names = set(previous) | set(files)
    directory_names = set(directories) | set(previous_dirs)
    for name in names | directory_names:
        if (
            not isinstance(name, str)
            or not name
            or "\\" in name
            or ":" in name
            or Path(name).is_absolute()
            or any(part in {"", ".", ".."} for part in name.split("/"))
            or name.split("/")[0].casefold() == "_engine"
            or re.fullmatch(r"[a-f0-9]{12,64}", name.split("/")[0].casefold())
        ):
            raise ValueError(f"invalid scenario export filename: {name!r}")
    for path in (
        destination,
        tracking,
        *(destination / name for name in names | directory_names),
    ):
        if path.is_symlink() or path.resolve() != path:
            raise ValueError(f"scenario export path is aliased: {path}")
    for name in names:
        path = destination / name
        if path.exists() and (name not in previous or not path.is_file()):
            raise ValueError(f"scenario export would overwrite an unowned path: {path}")
    for name in directory_names:
        path = destination / name
        if path.exists() and not path.is_dir():
            raise ValueError(f"scenario export directory conflicts with a file: {path}")
    changed = sorted(
        name
        for name in names
        if name not in files
        or not (destination / name).exists()
        or file_sha256(destination / name) != files[name]
    )
    record = {
        "schema_version": "scenario-export/1",
        "collection_id": marker["collection_id"],
        "collection_revision": marker["collection_revision"],
        "manifest": manifest_path.relative_to(root).as_posix(),
        "files": files,
        "directories": directories,
    }
    missing_dirs = [name for name in directories if not (destination / name).exists()]
    if not changed and not missing_dirs and record == old_record:
        return
    staging = tracking.parent / f".export-{uuid.uuid4().hex}"
    staging.mkdir()
    applied = []
    created_dirs = []
    removed_dirs = []
    rollback_failed = False
    try:
        for name in changed:
            if name in files:
                staged = staging / "new" / name
                staged.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source / name, staged)
        try:
            for name in missing_dirs:
                (destination / name).mkdir()
                created_dirs.append(name)
            for name in changed:
                path = destination / name
                backup = staging / "previous" / name
                if path.exists():
                    backup.parent.mkdir(parents=True, exist_ok=True)
                    os.replace(path, backup)
                applied.append(name)
                if name in files:
                    path.parent.mkdir(parents=True, exist_ok=True)
                    os.replace(staging / "new" / name, path)
            for name in sorted(set(previous_dirs) - set(directories), reverse=True):
                path = destination / name
                if path.exists() and not any(path.iterdir()):
                    path.rmdir()
                    removed_dirs.append(name)
            atomic_record(tracking, record, replace=True)
        except OSError:
            try:
                for name in reversed(removed_dirs):
                    (destination / name).mkdir()
                for name in reversed(applied):
                    backup = staging / "previous" / name
                    path = destination / name
                    if backup.exists():
                        os.replace(backup, path)
                    else:
                        path.unlink(missing_ok=True)
                for name in reversed(created_dirs):
                    (destination / name).rmdir()
            except OSError as exc:
                rollback_failed = True
                raise OSError(
                    f"scenario export rollback failed; backups: {staging}"
                ) from exc
            raise
    finally:
        if not rollback_failed:
            shutil.rmtree(staging)
