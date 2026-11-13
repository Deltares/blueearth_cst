"""Report and optionally delete old execution log folders (rule R13).

Each runner launch writes ``logs/<YYYY-MM-DD_HH-MM-SSZ>__<scope>[__dry-run]/``.
Executed runs are kept, failed and interrupted ones included, until a user
prunes them. This helper lists folders older than ``--older-than`` days and
deletes them only with ``--delete``.

Skipped, always:

* folders a retained WF2 product refers to: the execution folders of the
  invocations in each ``data/climate/projections/*/_engine/execution/production.json``;
* folders a still-running invocation uses;
* anything that is not an exact log-folder name, or is a link.

Usage (from the repo root, inside pixi)::

    python dev/scripts/prune_logs.py --project-dir <project_dir> --older-than 30
    python dev/scripts/prune_logs.py --project-dir <project_dir> --older-than 30 --delete

Not part of a run: this inspects and maintains a project tree
(see AGENTS.md, "Three homes for executables").
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from datetime import datetime, timedelta, timezone
from pathlib import Path

FOLDER = re.compile(
    r"(?P<stamp>\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})Z__[a-z0-9-]+"
    r"(?:__dry-run)?(?:__[0-9a-f]{12})?"
)


def _json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


def protected(project: Path) -> set[str]:
    """Log folder names a retained product or a running invocation needs."""
    invocations = project / "config/_engine/invocations"
    records = {path.stem: _json(path) for path in invocations.glob("*.json")}
    referenced = set()
    for production in project.glob(
        "data/climate/projections/*/_engine/execution/production.json"
    ):
        referenced |= set(_json(production).get("generations", {}))
    keep = set()
    for identity, record in records.items():
        if identity in referenced or record.get("status") == "running":
            directory = record.get("execution_directory")
            if directory:
                keep.add(Path(directory).name)
    return keep


def candidates(project: Path, older_than: timedelta, now: datetime) -> list[Path]:
    """Exact log folders older than the cutoff that nothing protects."""
    logs = project / "logs"
    if not logs.is_dir():
        return []
    keep = protected(project)
    result = []
    for folder in sorted(logs.iterdir()):
        match = FOLDER.fullmatch(folder.name)
        if (
            match is None
            or folder.name in keep
            or not folder.is_dir()
            or folder.is_symlink()
            or folder.is_junction()
        ):
            continue
        started = datetime.strptime(match["stamp"], "%Y-%m-%d_%H-%M-%S").replace(
            tzinfo=timezone.utc
        )
        if now - started > older_than:
            result.append(folder)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--project-dir", type=Path, required=True)
    parser.add_argument("--older-than", type=float, required=True, metavar="DAYS")
    parser.add_argument("--delete", action="store_true")
    args = parser.parse_args(argv)
    project = args.project_dir.resolve(strict=True)
    found = candidates(
        project, timedelta(days=args.older_than), datetime.now(timezone.utc)
    )
    for folder in found:
        print(folder.relative_to(project).as_posix())
    if not args.delete:
        print(
            f"{len(found)} log folder(s) older than {args.older_than:g} days; "
            "nothing deleted (pass --delete)"
        )
        return 0
    for folder in found:
        shutil.rmtree(folder)
    print(f"deleted {len(found)} log folder(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
