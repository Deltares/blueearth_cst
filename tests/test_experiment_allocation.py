"""Experiment ID allocation (R9 P4 commit 4).

The falsifiers here assert things that must NOT happen, which is where this
feature can go wrong quietly:

* **resume is not a collision** — the one that can break the pipeline. Treating
  a re-run as a collision fails every incremental rerun;
* **a user-supplied name is never silently versioned** — the same surprise
  ``validate_experiment_name`` refuses to make by lowercasing;
* **`_v3`, not just `_v2`** — an implementation that only handles the second
  collision passes a `_v2`-only test;
* **reservation is atomic** — demonstrated by racing, not by reading the code.
"""

import os
import sys
import threading
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from blueearth_cst.experiment.allocate import (  # noqa: E402
    ExperimentCollisionError,
    allocate_experiment_name,
    experiment_exists,
    next_available_name,
    reserve_experiment,
)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
import suggest_experiment_name as runner  # noqa: E402


def _existing(project_dir, *names):
    for name in names:
        (Path(project_dir) / "experiments" / name).mkdir(parents=True)


# ---------------------------------------------------------------------------
# Collision vs resume
# ---------------------------------------------------------------------------


def test_a_user_supplied_collision_is_rejected_and_names_the_experiment(tmp_path):
    _existing(tmp_path, "gabon_dry")
    with pytest.raises(ExperimentCollisionError) as excinfo:
        allocate_experiment_name(tmp_path, "gabon_dry", user_supplied=True)
    msg = str(excinfo.value)
    assert "gabon_dry" in msg
    # "name already exists" is not actionable; the path is.
    assert str(tmp_path / "experiments" / "gabon_dry") in msg


def test_a_user_supplied_name_is_never_silently_versioned(tmp_path):
    """The rule that makes a chosen name trustworthy. If this ever versions, a
    user believes they are writing to `gabon_dry` while results land in
    `gabon_dry_v2`."""
    _existing(tmp_path, "gabon_dry")
    with pytest.raises(ExperimentCollisionError):
        allocate_experiment_name(tmp_path, "gabon_dry", user_supplied=True)
    assert not experiment_exists(tmp_path, "gabon_dry_v2")


def test_resume_allocates_nothing(tmp_path):
    """THE falsifier that can break the pipeline.

    Re-running an existing experiment is the normal case and how incremental
    reruns work. Allocation is only ever called at CREATION -- the workflow
    reads `experiment_name` from the config and never allocates -- so a resume
    must leave the directory set untouched.
    """
    _existing(tmp_path, "gabon_dry")
    before = sorted(p.name for p in (tmp_path / "experiments").iterdir())

    cfg = tmp_path / "cfg.yml"
    cfg.write_text(
        yaml.safe_dump(
            {
                "project": {"project_dir": str(tmp_path).replace("\\", "/")},
                "workflows": {"climate_experiment": {"experiment_name": "gabon_dry"}},
            }
        ),
        encoding="utf-8",
    )

    # The runner refuses to overwrite an existing name -- that IS the resume
    # path, and it must allocate nothing.
    rc = runner.main([str(cfg), "--date", "20260804"])
    assert rc == 1
    assert sorted(p.name for p in (tmp_path / "experiments").iterdir()) == before


# ---------------------------------------------------------------------------
# Versioning of generated names
# ---------------------------------------------------------------------------


def test_a_generated_collision_becomes_v2_then_v3(tmp_path):
    """The third collision is the discriminator: an implementation that only
    handles the second passes a `_v2`-only test."""
    base = "test_local_20260804"
    assert allocate_experiment_name(tmp_path, base, user_supplied=False) == base
    assert allocate_experiment_name(tmp_path, base, user_supplied=False) == f"{base}_v2"
    assert allocate_experiment_name(tmp_path, base, user_supplied=False) == f"{base}_v3"


def test_versioning_starts_at_v2_because_the_bare_name_is_version_1(tmp_path):
    _existing(tmp_path, "exp")
    assert next_available_name(tmp_path, "exp") == "exp_v2"
    assert not experiment_exists(tmp_path, "exp_v1")


def test_versioning_fills_a_gap_rather_than_counting_directories(tmp_path):
    """`_v2` removed by hand must be reused, not skipped -- the count of
    existing directories is not the version number."""
    _existing(tmp_path, "exp", "exp_v3")
    assert next_available_name(tmp_path, "exp") == "exp_v2"


# ---------------------------------------------------------------------------
# Reservation
# ---------------------------------------------------------------------------


def test_reservation_creates_the_directory(tmp_path):
    path = reserve_experiment(tmp_path, "exp")
    assert path.is_dir() and experiment_exists(tmp_path, "exp")


def test_reserving_a_taken_name_raises(tmp_path):
    reserve_experiment(tmp_path, "exp")
    with pytest.raises(ExperimentCollisionError):
        reserve_experiment(tmp_path, "exp")


def test_concurrent_reservation_yields_exactly_one_winner(tmp_path):
    """Atomicity DEMONSTRATED by racing, not asserted by reading the code.

    An `exists()`-then-`mkdir` would leave a window in which both callers
    believe they own the name -- and this repository is routinely worked by
    several sessions at once, so the race is real rather than theoretical.
    """
    winners, losers = [], []
    barrier = threading.Barrier(8)

    def attempt():
        barrier.wait()  # maximise overlap
        try:
            reserve_experiment(tmp_path, "contended")
            winners.append(1)
        except ExperimentCollisionError:
            losers.append(1)

    threads = [threading.Thread(target=attempt) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(winners) == 1, f"{len(winners)} callers each believed they owned it"
    assert len(losers) == 7


# ---------------------------------------------------------------------------
# The runner
# ---------------------------------------------------------------------------


def _cfg(tmp_path):
    cfg = tmp_path / "cfg.yml"
    cfg.write_text(
        yaml.safe_dump({"project": {"project_dir": str(tmp_path).replace("\\", "/")}}),
        encoding="utf-8",
    )
    return cfg


def test_the_runner_reserves_and_writes_the_name(tmp_path, capsys):
    cfg = _cfg(tmp_path)
    assert runner.main([str(cfg), "--date", "20260804"]) == 0
    doc = yaml.safe_load(cfg.read_text(encoding="utf-8"))
    name = doc["workflows"]["climate_experiment"]["experiment_name"]
    assert name.endswith("_20260804")
    assert experiment_exists(tmp_path, name), "the name was written but not reserved"


def test_the_runner_versions_a_generated_collision(tmp_path):
    cfg = _cfg(tmp_path)
    runner.main([str(cfg), "--date", "20260804"])
    first = yaml.safe_load(cfg.read_text(encoding="utf-8"))["workflows"][
        "climate_experiment"
    ]["experiment_name"]

    cfg2 = tmp_path / "cfg2.yml"
    cfg2.write_text(
        yaml.safe_dump({"project": {"project_dir": str(tmp_path).replace("\\", "/")}}),
        encoding="utf-8",
    )
    runner.main([str(cfg2), "--date", "20260804"])
    second = yaml.safe_load(cfg2.read_text(encoding="utf-8"))["workflows"][
        "climate_experiment"
    ]["experiment_name"]

    assert second == f"{first}_v2"


def test_the_runner_rejects_a_user_supplied_collision(tmp_path, capsys):
    _existing(tmp_path, "gabon_dry")
    cfg = _cfg(tmp_path)
    assert runner.main([str(cfg), "--name", "gabon_dry"]) == 1
    assert "gabon_dry" in capsys.readouterr().err
    # ...and the config is left untouched, so nothing points at a name that was
    # refused.
    assert "experiment_name" not in cfg.read_text(encoding="utf-8")


def test_dry_run_reserves_nothing(tmp_path, capsys):
    """It prints what WOULD be proposed. Reserving there would claim a name the
    user has not committed to -- and the help says the printed name may be taken
    by the time they use it."""
    cfg = _cfg(tmp_path)
    assert runner.main([str(cfg), "--date", "20260804", "--dry-run"]) == 0
    assert not (tmp_path / "experiments").exists()
    assert "experiment_name" not in cfg.read_text(encoding="utf-8")


def test_a_user_supplied_name_still_faces_the_grammar(tmp_path, capsys):
    """--name is not an escape hatch around validate_experiment_name."""
    cfg = _cfg(tmp_path)
    assert runner.main([str(cfg), "--name", "Gabon-Dry"]) == 2
    assert "grammar" in capsys.readouterr().err


# --- the unset-key default: reuse before create --------------------------------

from blueearth_cst.experiment.allocate import (  # noqa: E402
    resolve_default_experiment_name,
)
from blueearth_cst.shared.snake_utils import project_slug  # noqa: E402


def _proj(tmp_path, *existing):
    pd = tmp_path / "gabon_0108"
    (pd / "experiments").mkdir(parents=True)
    for name in existing:
        (pd / "experiments" / name).mkdir()
    return pd


def test_the_default_is_the_project_name_plus_today(tmp_path):
    """`gabon_0108` yields `gabon_0108_20260805` when nothing is set."""
    pd = _proj(tmp_path)
    base = project_slug(pd, reserve=len("_YYYYMMDD"))
    assert base == "gabon_0108"
    assert (
        resolve_default_experiment_name(pd, base, "20260805") == "gabon_0108_20260805"
    )


def test_a_later_run_REUSES_the_existing_experiment(tmp_path):
    """The whole point of reuse. Minting today's date unconditionally would send
    tomorrow's run at an empty directory: every job re-runs, today's outputs are
    orphaned, and --dry-run shows a full rebuild with no stated reason."""
    pd = _proj(tmp_path, "gabon_0108_20260805")
    assert (
        resolve_default_experiment_name(pd, "gabon_0108", "20260806")
        == "gabon_0108_20260805"
    )


def test_the_most_recently_allocated_wins(tmp_path):
    """Max by (date, version): _v2 beats the unsuffixed name of the same day,
    and a newer day beats both."""
    pd = _proj(tmp_path, "gabon_0108_20260805", "gabon_0108_20260805_v2")
    assert (
        resolve_default_experiment_name(pd, "gabon_0108", "20260806")
        == "gabon_0108_20260805_v2"
    )
    (pd / "experiments" / "gabon_0108_20260810").mkdir()
    assert (
        resolve_default_experiment_name(pd, "gabon_0108", "20260806")
        == "gabon_0108_20260810"
    )


def test_only_this_project_s_dated_names_are_reused(tmp_path):
    """A deliberately chosen name must never be picked up by accident, and
    neither must another project's experiments sharing the directory."""
    pd = _proj(tmp_path, "dry_scenario", "other_project_20260901", "gabon_0108_no_date")
    assert (
        resolve_default_experiment_name(pd, "gabon_0108", "20260805")
        == "gabon_0108_20260805"
    )


def test_resolving_creates_nothing(tmp_path):
    """Called at Snakefile parse time, which also runs under --dry-run and
    --unlock; a side effect on disk there would be wrong."""
    pd = _proj(tmp_path)
    name = resolve_default_experiment_name(pd, "gabon_0108", "20260805")
    assert not (pd / "experiments" / name).exists()
    assert list((pd / "experiments").iterdir()) == []


def test_a_missing_experiments_dir_is_not_an_error(tmp_path):
    """First run of a brand-new project: nothing exists yet."""
    pd = tmp_path / "gabon_0108"
    pd.mkdir()
    assert (
        resolve_default_experiment_name(pd, "gabon_0108", "20260805")
        == "gabon_0108_20260805"
    )


def test_files_beside_the_experiments_are_ignored(tmp_path):
    pd = _proj(tmp_path)
    (pd / "experiments" / "gabon_0108_20260805").write_text(
        "not a dir", encoding="utf-8"
    )
    assert (
        resolve_default_experiment_name(pd, "gabon_0108", "20260806")
        == "gabon_0108_20260806"
    )


def test_the_default_and_the_suggest_command_derive_one_stem(tmp_path):
    """Both naming paths go through project_slug, so they cannot disagree."""
    from blueearth_cst.shared.snake_utils import suggest_experiment_name

    for name in ("Gabon", "My Basin-2024", "gabon_0108"):
        pd = tmp_path / name
        pd.mkdir()
        base = project_slug(pd, reserve=len("_YYYYMMDD"))
        assert suggest_experiment_name(pd, "20260805") == f"{base}_20260805"
        assert (
            resolve_default_experiment_name(pd, base, "20260805") == f"{base}_20260805"
        )
