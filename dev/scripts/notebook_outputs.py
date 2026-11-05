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

``--check --staged`` reads each notebook FROM THE GIT INDEX instead of the
working tree, and with no paths discovers the staged notebooks itself. A commit
records the index, so the index is the only content a pre-commit check may
judge: staging a notebook that carries outputs and then stripping the working
copy without re-staging leaves the outputs in the commit, and a working-tree
check calls that clean. Discovering the paths in-process also retires the
shell word-splitting that made this check silently pass in 2026-08-14 (see
``.githooks/pre-commit``).

Divergence between index and working tree is REPORTED, never rejected. The
hook checks only staged notebooks precisely so that an unstaged working copy
you are actively running does not block an unrelated commit; making the
working tree a second gate would take that back.

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
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
NOTEBOOK_DIR = REPO_ROOT / "docs" / "notebooks"


def default_paths() -> list[Path]:
    """Every tracked notebook, sorted so messages read the same way twice."""
    return sorted(NOTEBOOK_DIR.glob("*.ipynb"))


def _git(args: list[str], repo_root: Path) -> subprocess.CompletedProcess[bytes]:
    # Argument LIST, never a shell string. Two of the three tracked notebooks
    # have spaces in their names, and a shell string is how they got split into
    # nonexistent fragments the last time (see the hook's own comment).
    return subprocess.run(
        ["git", *args], cwd=repo_root, capture_output=True, check=False
    )


def staged_notebooks(repo_root: Path | None = None) -> list[str]:
    """Repo-relative paths of the notebooks staged for the next commit.

    NUL-delimited, and read here rather than piped in, so no caller can
    word-split a path containing spaces on the way over.
    """
    repo_root = repo_root or REPO_ROOT
    proc = _git(
        ["diff", "--cached", "--name-only", "--diff-filter=ACM", "-z", "--", "*.ipynb"],
        repo_root,
    )
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.decode("utf-8", "replace").strip())
    return [p for p in proc.stdout.decode("utf-8").split("\0") if p]


def _load_staged(rel: str, repo_root: Path | None = None) -> dict:
    """Parse a notebook from the index. Raises if the blob is not there."""
    proc = _git(["show", f":{rel}"], repo_root or REPO_ROOT)
    if proc.returncode != 0:
        raise FileNotFoundError(rel)
    return json.loads(proc.stdout.decode("utf-8"))


def diverges_from_index(rel: str, repo_root: Path | None = None) -> bool:
    """True when the working copy of a staged path differs from the index.

    Informational only. The commit records the index, so this never decides
    pass or fail -- it explains a result that would otherwise look wrong to
    whoever just edited the file.
    """
    proc = _git(["diff", "--quiet", "--", rel], repo_root or REPO_ROOT)
    return proc.returncode != 0


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


def _print_offenders(offenders: list[tuple[object, int]]) -> None:
    print("notebook outputs must not be committed:", file=sys.stderr)
    for label, n in offenders:
        print(f"  {label}: {n} cell(s) carry outputs", file=sys.stderr)
    print(
        "\nClear them with:\n"
        "    python dev/scripts/notebook_outputs.py --strip\n"
        "Rendered notebooks are published as Artifacts instead of being "
        "committed; see docs/notebooks/README.md.",
        file=sys.stderr,
    )


def _check_staged(rels: list[str]) -> int:
    """Check the staged content of notebooks. Returns a process exit code."""
    explicit = bool(rels)
    try:
        rels = rels or staged_notebooks()
    except RuntimeError as exc:
        print(f"cannot list staged notebooks: {exc}", file=sys.stderr)
        return 2

    if not rels:
        print("ok - no notebooks staged")
        return 0

    # Same rule as the working-tree path: a path the CALLER named and we cannot
    # read is an error, never a skip. Here "cannot read" means the index holds
    # no blob at that path -- which is what a word-split filename looks like
    # from this side, and what an unstaged path looks like too.
    offenders: list[tuple[object, int]] = []
    unreadable: list[str] = []
    diverged: list[str] = []
    for rel in rels:
        try:
            notebook = _load_staged(rel)
        except FileNotFoundError:
            unreadable.append(rel)
            continue
        if diverges_from_index(rel):
            diverged.append(rel)
        hits = cells_with_outputs(notebook)
        if hits:
            offenders.append((rel, len(hits)))

    if unreadable:
        print(
            "the git index holds no content for the path(s) given:\n"
            + "".join(f"  {rel}\n" for rel in unreadable)
            + (
                "If these look like fragments of a filename, the caller "
                "word-split a path containing spaces. Otherwise the path is "
                "not staged; stage it or drop it from the argument list."
                if explicit
                else "This should not happen: the paths came from the index."
            ),
            file=sys.stderr,
        )
        return 2

    if offenders:
        _print_offenders(offenders)
        for rel in diverged:
            print(
                f"note: the working copy of {rel} differs from what is staged. "
                "The commit records the STAGED content, so stripping the file "
                "is not enough -- `git add` it again.",
                file=sys.stderr,
            )
        return 1

    for rel in diverged:
        print(
            f"note: {rel} is staged clean but its working copy differs; only "
            "the staged content was checked."
        )
    print(f"ok - {len(rels)} staged notebook(s) carry no outputs")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true", help="report, change nothing")
    mode.add_argument("--strip", action="store_true", help="clear outputs in place")
    ap.add_argument(
        "--staged",
        action="store_true",
        help="read content from the git index, not the working tree (--check only)",
    )
    ap.add_argument("paths", nargs="*", type=Path)
    args = ap.parse_args(argv)

    if args.staged:
        if not args.check:
            ap.error("--staged is only meaningful with --check")
        return _check_staged([p.as_posix() for p in args.paths])

    explicit = bool(args.paths)
    paths = [Path(p) for p in args.paths] or default_paths()
    offenders: list[tuple[Path, int]] = []
    stripped: list[Path] = []

    # A path the CALLER named and we cannot read is an error, never a skip.
    #
    # This is the layer that should have caught the pre-commit hook's
    # word-splitting bug on 2026-08-14 and did not: the hook handed over
    # `docs/notebooks/Climate`, `Stress`, `Test.ipynb`, none of which exists,
    # and a silent skip turned "checked nothing" into "ok". Skipping is only
    # right for the DEFAULT glob, where a non-notebook simply is not our
    # business.
    if explicit:
        missing = [p for p in paths if not p.is_file()]
        if missing:
            print(
                "cannot read the path(s) given:\n"
                + "".join(f"  {p}\n" for p in missing)
                + "If these look like fragments of a filename, the caller "
                "word-split a path containing spaces.",
                file=sys.stderr,
            )
            return 2

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
            _print_offenders(
                [
                    (
                        path.resolve().relative_to(REPO_ROOT)
                        if path.is_absolute()
                        else path,
                        n,
                    )
                    for path, n in offenders
                ]
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
