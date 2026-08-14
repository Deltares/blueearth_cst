"""Keep rendered outputs out of the tracked notebooks.

A Jupyter notebook that carries its outputs embeds every figure as base64 PNG,
so it does not delta-compress: EVERY edit mints a fresh multi-megabyte blob that
stays in history forever. Measured 2026-08-14, before this script existed:
``docs/notebooks/Model building.ipynb`` was 6.43 MB with 82 blob versions
already in history, and one rename sweep that merely rewrote three strings
inside the three notebooks turned a few hundred KB of text change into a 7.1 MB
push.

Two modes, one definition of "carries outputs" so the check and the fix cannot
disagree:

    python dev/scripts/notebook_outputs.py --check  [paths...]
    python dev/scripts/notebook_outputs.py --strip  [paths...]

With no paths, both modes walk ``docs/notebooks/*.ipynb``.

WHAT COUNTS AS AN OUTPUT, and why ``execution_count`` is included: a cleared
notebook whose cells still carry ``execution_count: 7`` produces a diff on the
next run for no reason a reader cares about, so the two are cleared together.
Notebook-level ``metadata`` is left alone -- ``kernelspec`` and
``language_info`` are what make the file openable, and neither is large.

This is enforcement-by-test, not enforcement-by-hook. ``core.hooksPath`` is a
per-clone setting that cloning does not install (AGENTS.md says so), so a hook
alone protects only the machines that opted in; ``tests/test_notebook_outputs.py``
runs on both CI legs and is what actually holds the line. The pre-commit hook is
the fast local echo of that test, not the gate.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
NOTEBOOK_DIR = REPO_ROOT / "docs" / "notebooks"


def default_paths() -> list[Path]:
    """Every tracked notebook, sorted so messages read the same way twice."""
    return sorted(NOTEBOOK_DIR.glob("*.ipynb"))


def cells_with_outputs(notebook: dict) -> list[int]:
    """Indices of cells carrying outputs or an execution count."""
    hits = []
    for i, cell in enumerate(notebook.get("cells", [])):
        if cell.get("outputs") or cell.get("execution_count") is not None:
            hits.append(i)
    return hits


def strip(notebook: dict) -> bool:
    """Clear outputs in place. Returns True if anything changed."""
    changed = False
    for cell in notebook.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        if cell.get("outputs"):
            cell["outputs"] = []
            changed = True
        if cell.get("execution_count") is not None:
            cell["execution_count"] = None
            changed = True
    return changed


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _dump(path: Path, notebook: dict) -> None:
    # Trailing newline and `ensure_ascii=False` match what Jupyter writes, so a
    # stripped notebook re-opened and saved does not diff on formatting alone.
    path.write_text(
        json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true", help="report, change nothing")
    mode.add_argument("--strip", action="store_true", help="clear outputs in place")
    ap.add_argument("paths", nargs="*", type=Path)
    args = ap.parse_args(argv)

    paths = [Path(p) for p in args.paths] or default_paths()
    offenders: list[tuple[Path, int]] = []
    stripped: list[Path] = []

    for path in paths:
        if path.suffix != ".ipynb" or not path.is_file():
            continue
        notebook = _load(path)
        if args.check:
            hits = cells_with_outputs(notebook)
            if hits:
                offenders.append((path, len(hits)))
        else:
            if strip(notebook):
                _dump(path, notebook)
                stripped.append(path)

    if args.check:
        if offenders:
            print("notebook outputs must not be committed:", file=sys.stderr)
            for path, n in offenders:
                rel = (
                    path.resolve().relative_to(REPO_ROOT)
                    if path.is_absolute()
                    else path
                )
                print(f"  {rel}: {n} cell(s) carry outputs", file=sys.stderr)
            print(
                "\nClear them with:\n"
                "    python dev/scripts/notebook_outputs.py --strip\n"
                "Rendered notebooks are published as Artifacts instead of being "
                "committed; see docs/notebooks/README.md.",
                file=sys.stderr,
            )
            return 1
        print(f"ok - {len(paths)} notebook(s) carry no outputs")
        return 0

    for path in stripped:
        print(f"stripped {path}")
    print(f"{len(stripped)} of {len(paths)} notebook(s) changed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
