"""Shared machinery for the repository sweeps.

A sweep here is a report-only check that reads the tree, classifies every hit,
and **fails closed**: a hit that matches no declared class is a defect, and a
sweep that cannot see its own subject matter refuses to report rather than
reporting nothing (D-14.4, D-14.5).

Extracted when R14 had several sweeps, so they could not drift on what an
allowance is, which directories are out of scope, or what a clean run means.
`sweep_unread_config_keys.py` (`C-37`) is the remaining caller; the R14
stale-spelling and identity-rename sweeps were retired (t2610041227).

The three pieces:

* `Allowance` — one class of legitimate hit, WITH the reason it is legitimate.
  A sweep that just suppressed hits would be an allowlist nobody can review.
* `iter_lines` — the walk, with the excluded directories in one place.
* `Floor` / `check_floors` — the fail-closed half. A sweep derives some GROUND
  TRUTH before it looks for defects (the retired-key vocabulary; the record
  forms it can recognise). If that derivation silently returns less than it
  used to, the sweep goes quiet while knowing LESS, and nothing in its output
  says so. R14 demonstrated exactly that: an extraction returned zero and the
  run reported MORE flags than the pass before it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

#: Never swept, in any sweep: generated output, the environment, scratch, and
#: git's own tree. Not a judgment about content — nothing here is source.
EXCLUDED_DIRS = ("_site", ".quarto", ".pixi", "__pycache__", ".tmp", ".git")


@dataclass(frozen=True)
class Allowance:
    """One class of legitimate hit, with the reason it is legitimate."""

    name: str
    reason: str
    paths: tuple[str, ...] = ()
    path_globs: tuple[str, ...] = ()
    line_patterns: tuple[str, ...] = field(default=())

    def covers(self, path: Path, line: str) -> bool:
        # A sweep run against a throwaway tree -- a test, or a specimen check --
        # hands in paths outside the repository. Those match no PATH allowance
        # by construction, which is correct: the reason a repository file is
        # excused does not travel to a fixture that merely shares its name.
        try:
            rel = path.relative_to(REPO_ROOT).as_posix()
        except ValueError:
            rel = path.as_posix()
        if rel in self.paths:
            return True
        if any(Path(rel).match(glob) for glob in self.path_globs):
            return True
        return any(re.search(pattern, line) for pattern in self.line_patterns)


def iter_lines(globs, root: Path = REPO_ROOT):
    """Yield ``(path, lineno, line)`` for every readable line under ``globs``.

    Order follows ``globs``, and a file matched twice is visited once, so a
    sweep's SEARCHED tuple can overlap without double-reporting.
    """
    seen: set[Path] = set()
    for glob in globs:
        for path in sorted(root.glob(glob)):
            if path in seen or not path.is_file():
                continue
            if any(part in EXCLUDED_DIRS for part in path.parts):
                continue
            seen.add(path)
            try:
                text = path.read_text(encoding="utf-8")
            except (UnicodeDecodeError, OSError):
                continue
            for lineno, line in enumerate(text.splitlines(), 1):
                yield path, lineno, line


def classify(hits, allowances):
    """Split ``(path, lineno, line)`` hits into ``(defects, allowed)``.

    The FIRST allowance that covers a hit is the reason recorded for it, so the
    order of a sweep's `ALLOWANCES` is part of its explanation. A hit no
    allowance covers is a defect — never a warning, and never dropped.
    """
    defects, allowed = [], []
    for path, lineno, line in hits:
        for allowance in allowances:
            if allowance.covers(path, line):
                allowed.append((path, lineno, line.strip(), allowance.name))
                break
        else:
            defects.append((path, lineno, line.strip(), "UNCLASSIFIED"))
    return defects, allowed


@dataclass(frozen=True)
class Floor:
    """A minimum this sweep must still be able to SEE before it reports.

    ``count`` is what the tree held when the floor was written, and ``minimum``
    is how far it may fall before the observation stops being credible. The
    two are separate on purpose: a floor of 1 passes when a matcher that used
    to find two hundred records finds one, which is the failure it exists to
    catch. Set ``minimum`` well under ``count`` to tolerate honest churn and
    well over 1 to catch a matcher that has gone blind.

    ``measured`` says when and against what, so a floor that starts failing can
    be told from a floor that was never true.
    """

    name: str
    minimum: int
    count: int
    measured: str
    why: str


def check_floors(floors, observed: dict) -> list[str]:
    """Return one message per floor the observation fell below.

    A caller that ignores the return value has re-created the fail-open shape
    this exists to prevent, so callers raise or exit on a non-empty list —
    they never log it.
    """
    failures = []
    for floor in floors:
        seen = observed.get(floor.name, 0)
        if seen < floor.minimum:
            failures.append(
                f"{floor.name}: saw {seen}, floor is {floor.minimum} "
                f"({floor.count} when measured {floor.measured}). {floor.why}"
            )
    return failures
