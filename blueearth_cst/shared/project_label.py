"""The project's display name, shared by the launcher header and DAG render files.

One rule for both readers, so a console and a filename never disagree about
what a project is called:

1. an explicit top-level ``project_name`` key;
2. else the CASE folder, when the config sits in a case trial created by
   ``scripts/create_case.py`` (``<cases>/<case>/<trial>/project_config.yml``,
   recognised by the trial inventory marker in ``<case>/README.md``);
3. else the ``project_dir`` basename.

Step 2 exists because a scaffolded trial's ``project_dir`` is always
``outputs``, which names every case alike. The marker, not the folder depth,
decides it: a repository ``test_case/project_config_*.yml`` would otherwise
be named after the repository checkout.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

#: Written by `scripts/create_case.py` around a case README's trial inventory.
CASE_TRIALS_MARKER = "<!-- cst-trials -->"


def case_name(config_path: str | os.PathLike[str] | None) -> str | None:
    """The case folder name for a config inside a case trial, else ``None``."""
    if not config_path:
        return None
    case = Path(config_path).resolve().parent.parent
    try:
        readme = (case / "README.md").read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    return case.name if CASE_TRIALS_MARKER in readme else None


def project_name(
    cfg: Mapping[str, Any],
    project_dir: str | os.PathLike[str],
    config_path: str | os.PathLike[str] | None = None,
) -> str:
    """The display name: explicit key, then case folder, then folder basename."""
    name = cfg.get("project_name")
    if name:
        return str(name)
    return case_name(config_path) or Path(project_dir).name
