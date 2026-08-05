"""Experiment ID allocation: claim a name, or refuse to reuse someone else's.

R9 P4 commit 4. Experiment creation must not silently reuse or overwrite an
existing experiment, because `experiments/<id>/` is the directory every WF3
artifact hangs off — quietly landing on an occupied one mixes two experiments'
results under one name.

**Resume is not a collision, and that distinction is the whole design.** Running
an existing experiment again is the normal case: it is how incremental reruns
work, and treating it as a collision would fail every rerun. A collision is a
*new* experiment claiming an occupied name. The two are told apart by WHO is
asking, not by what is on disk — the creation path allocates, the workflow
resumes — so this module is called only at creation.

**Reservation is an atomic ``mkdir``.** ``os.mkdir`` is atomic on POSIX and
Windows: two concurrent creators racing for one name produce one winner and one
``FileExistsError``. An ``exists()``-then-``mkdir`` would leave a window in which
both callers believe they own the name, and this repository is routinely worked
by several sessions at once.

The threat model is deliberately bounded to one machine. A shared network
``project_dir`` with two users would need a lock protocol with staleness
handling; there is no multi-user story to design against, and inventing one
would be machinery without a requirement.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

#: Suffix applied to a GENERATED name whose directory is taken: `_v2`, `_v3`, …
#: A user-supplied name is never suffixed — silently renaming what a human chose
#: is the same surprise `validate_experiment_name` already refuses to make by
#: lowercasing.
_VERSION_SUFFIX = re.compile(r"_v(\d+)$")

#: Where experiments live under `project_dir`.
EXPERIMENTS_DIRNAME = "experiments"

#: Guards against an unbounded search when every candidate is taken. Reaching it
#: means something is wrong with the caller, not that a 1000th version is wanted.
MAX_VERSIONS = 999


class ExperimentCollisionError(RuntimeError):
    """A new experiment tried to claim a name that already exists."""


def experiments_root(project_dir) -> Path:
    return Path(project_dir) / EXPERIMENTS_DIRNAME


def experiment_exists(project_dir, name: str) -> bool:
    """Whether ``experiments/<name>/`` is already present."""
    return (experiments_root(project_dir) / name).is_dir()


def next_available_name(project_dir, base: str) -> str:
    """``base``, else ``base_v2``, ``base_v3``… — the first name not taken.

    Only for GENERATED names. Starts at ``_v2`` rather than ``_v1`` because the
    unsuffixed name IS version 1; a ``_v1`` directory beside an unsuffixed one
    would be two names for one thing.
    """
    if not experiment_exists(project_dir, base):
        return base
    for version in range(2, MAX_VERSIONS + 1):
        candidate = f"{base}_v{version}"
        if not experiment_exists(project_dir, candidate):
            return candidate
    raise ExperimentCollisionError(
        f"every candidate from {base!r} to {base}_v{MAX_VERSIONS} already "
        f"exists under {experiments_root(project_dir)}; this is a caller "
        f"problem, not a naming one"
    )


def reserve_experiment(project_dir, name: str) -> Path:
    """Claim ``experiments/<name>/`` atomically. Returns the created path.

    Raises
    ------
    ExperimentCollisionError
        If the name is already taken — including when a concurrent caller won
        the race, which is why the ``mkdir`` is not guarded by an ``exists()``
        check.
    """
    root = experiments_root(project_dir)
    root.mkdir(parents=True, exist_ok=True)
    target = root / name
    try:
        os.mkdir(target)
    except FileExistsError:
        raise ExperimentCollisionError(
            f"experiment {name!r} already exists at {target}. Re-running an "
            f"existing experiment is done by pointing the workflow at it, not "
            f"by creating it again; to start a NEW experiment, choose another "
            f"name."
        ) from None
    return target


def allocate_experiment_name(
    project_dir, base: str, user_supplied: bool
) -> str:
    """Choose and reserve a name for a NEW experiment.

    Parameters
    ----------
    project_dir : str | Path
        The run's output root.
    base : str
        The proposed name — already validated by
        ``snake_utils.validate_experiment_name``.
    user_supplied : bool
        Whether a human chose this name. A user-supplied collision is an ERROR;
        a generated one is versioned. The caller knows which it is; the
        filesystem cannot tell.

    Returns the reserved name.
    """
    if user_supplied:
        if experiment_exists(project_dir, base):
            raise ExperimentCollisionError(
                f"experiment {base!r} already exists at "
                f"{experiments_root(project_dir) / base}. A name you chose is "
                f"never silently versioned -- pick another, or re-run the "
                f"existing experiment if that is what you meant."
            )
        chosen = base
    else:
        chosen = next_available_name(project_dir, base)
    reserve_experiment(project_dir, chosen)
    return chosen
