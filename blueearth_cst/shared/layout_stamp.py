"""Project layout version stamp, ``config/_engine/layout.json`` (rule R18).

A runner stamps a project folder it is about to create, and refuses a folder
whose layout differs from the one this toolbox writes. A folder without a
stamp predates the stamp and is layout 1; it is never backfilled, so an old
tree cannot be mislabelled once the version moves on. There is no migrator:
a different layout is rerun into a new folder.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from blueearth_cst.shared.layout_paths import LAYOUT_VERSION

STAMP = Path("config/_engine/layout.json")
#: Layout of a folder created before stamps existed.
UNSTAMPED_VERSION = 1


class LayoutMismatch(ValueError):
    """The project folder was written with a different output layout."""


def read(root: Path) -> int | None:
    """The stamped layout version, or ``None`` for an unstamped folder."""
    path = Path(root) / STAMP
    if not path.exists():
        return None
    document = json.loads(path.read_text(encoding="utf-8"))
    version = document.get("layout_version") if isinstance(document, dict) else None
    if type(version) is not int:
        raise LayoutMismatch(f"{path}: unreadable layout stamp")
    return version


def write(root: Path, version: int = LAYOUT_VERSION) -> Path:
    """Write the stamp atomically and return its path."""
    path = Path(root) / STAMP
    path.parent.mkdir(parents=True, exist_ok=True)
    staged = path.with_name(f".{path.name}.{os.getpid()}.staged")
    staged.write_text(
        json.dumps({"layout_version": version}, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(staged, path)
    return path


def admit(root: Path) -> int:
    """Stamp a new project folder, or refuse one with a different layout.

    Call before anything else writes to ``root``: a missing or empty folder is
    new and receives the stamp.
    """
    root = Path(root)
    version = read(root)
    if version is None:
        if not root.exists() or not any(root.iterdir()):
            write(root)
            return LAYOUT_VERSION
        version = UNSTAMPED_VERSION
    if version != LAYOUT_VERSION:
        raise LayoutMismatch(
            f"{root} uses output layout {version}; this toolbox writes layout "
            f"{LAYOUT_VERSION}. Set project.project_dir to a new folder and rerun, "
            "or use a toolbox release that writes layout "
            f"{version} for this folder."
        )
    return version
