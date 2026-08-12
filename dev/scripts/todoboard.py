"""Run the vendored `todoboard` CLI that backs `dev/tasks/` and `dev/TODO.md`.

`dev/TODO.md` is GENERATED — edit the note under `dev/tasks/` and re-render,
never the table. This wrapper exists because the CLI ships inside the
`todo-board` skill bundle rather than in this repo: the skill directory is
per-user, gitignored and symlinked (see AGENTS.md, "The agent-config
directories"), so its path cannot be committed. Without the wrapper the
invocation is an undocumented `PYTHONPATH=<machine path> python -m todoboard`,
which is how a board note landed on 2026-08-12 with the table left one row
stale — the CLI was simply not reachable and nothing said so.

Usage (from anywhere in the repo; `lane/devmeta` owns the board):

    python dev/scripts/todoboard.py render
    python dev/scripts/todoboard.py list
    python dev/scripts/todoboard.py add "Title" --area wf3

Every verb and flag is the CLI's own; this only resolves the package and
delegates. Set TODOBOARD_SKILL_DIR to override the search.
"""

from __future__ import annotations

import os
import pathlib
import runpy
import sys

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]

#: Ordered candidates for the skill bundle holding the `todoboard/` package.
#: `.claude` / `.agents` first: in a worktree those are symlinks to the primary's
#: copies, which are themselves symlinks into ~/workspace/brain/artifacts, so a
#: repo-relative hit is the same file the session's skills came from.
_CANDIDATES = (
    _REPO_ROOT / ".claude" / "skills" / "todo-board",
    _REPO_ROOT / ".agents" / "skills" / "todo-board",
    pathlib.Path.home() / "workspace" / "brain" / "artifacts" / "skills" / "todo-board",
    pathlib.Path.home() / ".claude" / "skills" / "todo-board",
)


def resolve_skill_dir() -> pathlib.Path:
    """First candidate that actually contains the importable package.

    An explicit ``TODOBOARD_SKILL_DIR`` is honoured or REFUSED — never fallen
    back from. Falling through to a different copy would answer a deliberate
    override with someone else's skill version and report success, which is the
    silent-degradation failure the search order exists to avoid, not cause.
    """
    override = os.environ.get("TODOBOARD_SKILL_DIR")
    if override:
        path = pathlib.Path(override).expanduser()
        if not (path / "todoboard" / "__main__.py").is_file():
            raise SystemExit(
                f"TODOBOARD_SKILL_DIR={path} does not contain todoboard/__main__.py.\n"
                "Point it at a todo-board skill directory, or unset it to search "
                "the default locations."
            )
        return path

    tried: list[str] = []
    for candidate in _CANDIDATES:
        path = pathlib.Path(candidate).expanduser()
        tried.append(str(path))
        if (path / "todoboard" / "__main__.py").is_file():
            return path
    raise SystemExit(
        "todoboard CLI not found. It ships inside the `todo-board` skill bundle,\n"
        "which is per-user state and not committed here. Tried:\n  "
        + "\n  ".join(tried)
        + "\n\nFix: re-link the agent-config dirs into this worktree\n"
        "  python ~/workspace/brain/artifacts/skills/git-workflow/scripts/"
        "worktree-session.py sync\n"
        "or point TODOBOARD_SKILL_DIR at the skill directory."
    )


def main() -> None:
    sys.path.insert(0, str(resolve_skill_dir()))
    # runpy rather than import+call: `python -m todoboard` is the CLI's own
    # documented entry point, so its argument parsing stays entirely upstream.
    sys.argv[0] = "todoboard"
    runpy.run_module("todoboard", run_name="__main__", alter_sys=True)


if __name__ == "__main__":
    main()
