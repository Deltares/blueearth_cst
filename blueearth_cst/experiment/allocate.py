"""Read-only naming compatibility; creation belongs to owned reservation admission.

The dated default and directory-only allocator are retired. Execution runners
use reservation.admit_experiment_owned while holding project-execution ownership.
Existing owned legacy directories are selected explicitly and remain in place.
"""

from pathlib import Path

from blueearth_cst.experiment.reservation import preview_experiment

EXPERIMENTS_DIRNAME = "experiments"


class ExperimentCollisionError(RuntimeError):
    """A retired creation entry point was called without owned admission."""


def experiments_root(project_dir: str | Path) -> Path:
    """Return the experiment directory without creating it."""
    return Path(project_dir) / EXPERIMENTS_DIRNAME


def experiment_exists(project_dir: str | Path, name: str) -> bool:
    """Report directory presence; presence alone is not experiment ownership."""
    return (experiments_root(project_dir) / name).is_dir()


def resolve_default_experiment_name(
    project_dir: str | Path, base=None, today=None
) -> str:
    """Preview the default numbered family; former date arguments are ignored."""
    selected = preview_experiment(project_dir)
    if selected.action in {"refuse", "missing"}:
        raise ValueError(selected.reason)
    return selected.name


def next_available_name(project_dir: str | Path, base: str) -> str:
    """Preview the next allocation; only the owned runner can claim it."""
    selected = preview_experiment(project_dir, base, new_experiment=True)
    if selected.action in {"refuse", "missing"}:
        raise ValueError(selected.reason)
    return selected.name


def reserve_experiment(project_dir: str | Path, name: str) -> None:
    """Refuse directory-only allocation, which cannot certify scientific intent."""
    raise ExperimentCollisionError(
        "directory-only allocation is retired; use an execution runner with --new-experiment"
    )


def allocate_experiment_name(
    project_dir: str | Path, base: str, user_supplied: bool
) -> str:
    """Refuse the retired allocator without changing configuration or directories."""
    raise ExperimentCollisionError(
        "allocation requires owned scientific admission; use an execution runner with --new-experiment"
    )
