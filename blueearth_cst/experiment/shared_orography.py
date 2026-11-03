"""Project-owned immutable orography copies for WF4 preparation bindings."""

import os
import re
import tempfile
from collections.abc import Callable, Mapping
from pathlib import Path, PureWindowsPath
from typing import Any

from blueearth_cst.shared.provenance import file_sha256, short_digest
from blueearth_cst.shared.workflow_config_snapshot import (
    file_reference,
    resolve_file_reference,
)


def _target(
    project_root: Path,
    source_name: str,
    digest: str,
    basename: str,
    *,
    full_digest: bool = False,
) -> Path:
    """Construct the accepted content-addressed project path.

    The folder is the repo's 12-character digest handle. ``full_digest`` builds
    the 64-character form written before 2026-09-24, which existing experiments
    still reference.
    """
    if not re.fullmatch(r"[a-z][a-z0-9_]*", source_name):
        raise ValueError("orography source name must be a registered simple token")
    if not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise ValueError("orography content digest must be full lowercase SHA-256")
    if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_.-]*\.nc", basename):
        raise ValueError("orography basename must be a single netCDF filename")
    if (
        basename in {".", ".."}
        or basename.rstrip(" .") != basename
        or PureWindowsPath(basename).is_reserved()
    ):
        raise ValueError("orography basename is not portable")
    return (
        Path(project_root).resolve()
        / "data"
        / "climate"
        / "ancillary"
        / source_name
        / (digest if full_digest else short_digest(digest))
        / basename
    )


def publish_shared_orography(
    source: Path,
    project_root: Path,
    *,
    source_name: str,
    basename: str,
    describe: Callable[[Path], Mapping[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Copy a checked source once, refusing changed bytes at an occupied path.

    Returns its project-relative FileRef and physical descriptor. The caller
    records both in the WF4 forcing-preparation document.
    """
    source = Path(source).resolve(strict=True)
    digest = file_sha256(source)
    descriptor = dict(describe(source))
    target = _target(project_root, source_name, digest, basename)
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.parent.resolve().is_relative_to(Path(project_root).resolve()):
        raise ValueError("shared orography directory escapes the project")
    if not target.exists():
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(
                dir=target.parent, suffix=".nc", delete=False
            ) as handle:
                temporary = Path(handle.name)
                with source.open("rb") as reader:
                    while chunk := reader.read(1024 * 1024):
                        handle.write(chunk)
                handle.flush()
                os.fsync(handle.fileno())
            if (
                file_sha256(temporary) != digest
                or dict(describe(temporary)) != descriptor
            ):
                raise ValueError("staged orography differs from its source")
            try:
                os.link(temporary, target)
            except FileExistsError:
                pass
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
    if target.is_symlink() or file_sha256(target) != digest:
        raise ValueError("shared orography target differs from content identity")
    if dict(describe(target)) != descriptor:
        raise ValueError("shared orography descriptor differs from source")
    return file_reference(target, "project_root", Path(project_root)), descriptor


def resolve_shared_orography(
    project_root: Path,
    reference: Mapping[str, Any],
    descriptor: Mapping[str, Any],
    *,
    source_name: str,
    basename: str,
    describe: Callable[[Path], Mapping[str, Any]],
) -> Path:
    """Validate an experiment elevation FileRef and its physical descriptor."""
    if reference["path_base"] != "project_root":
        raise ValueError("shared orography reference must be project-root anchored")
    expected = {
        _target(
            project_root,
            source_name,
            reference["sha256"],
            basename,
            full_digest=full,
        )
        for full in (False, True)
    }
    observed = resolve_file_reference(reference, {"project_root": Path(project_root)})
    if observed not in expected or dict(describe(observed)) != dict(descriptor):
        raise ValueError("shared orography path or descriptor differs")
    return observed


def relative_hydromt_uri(
    catalog_path: Path, reference: Mapping[str, Any], project_root: Path
) -> str:
    """Build a native HydroMT URI anchored at the retained catalog directory."""
    target = resolve_file_reference(reference, {"project_root": Path(project_root)})
    catalog_parent = Path(catalog_path).resolve().parent
    if not catalog_parent.is_relative_to(Path(project_root).resolve()):
        raise ValueError("HydroMT catalog is outside the project")
    return Path(os.path.relpath(target, catalog_parent)).as_posix()


def resolve_hydromt_uri(
    catalog_path: Path,
    uri: str,
    reference: Mapping[str, Any],
    project_root: Path,
) -> Path:
    """Check an upstream-native relative URI against the CST FileRef target."""
    if type(uri) is not str or not uri or ":" in uri or "\\" in uri:
        raise ValueError("HydroMT elevation URI must be relative POSIX syntax")
    target = resolve_file_reference(reference, {"project_root": Path(project_root)})
    resolved = (Path(catalog_path).resolve().parent / uri).resolve(strict=True)
    if resolved != target:
        raise ValueError("HydroMT elevation URI differs from checked FileRef")
    return resolved
