"""Exercise the repository temporary-directory policy through real fixtures."""

import os
import time
from pathlib import Path

import pytest

from dev.scripts.pytest_temp import STALE_RUN_SECONDS, prune_stale_runs


def test_temporary_directories_are_local_writable_and_isolated(
    tmp_path_factory, tmp_path, tmpdir, pytestconfig
):
    root = Path(__file__).resolve().parents[1]
    base = tmp_path_factory.getbasetemp()
    explicit = pytestconfig.getoption("basetemp")
    if explicit is None:
        assert base.is_relative_to(root / ".tmp" / "pytest")
    else:
        assert base == Path(explicit).resolve()
    first = tmp_path_factory.mktemp("same")
    second = tmp_path_factory.mktemp("same")
    assert first != second
    for directory in (
        first,
        second,
        tmp_path_factory.mktemp("plain", numbered=False),
        tmp_path,
        Path(str(tmpdir)),
    ):
        target = directory / "probe.txt"
        target.write_text("writable", encoding="utf-8")
        assert target.read_text(encoding="utf-8") == "writable"
    with pytest.raises(ValueError):
        tmp_path_factory.mktemp("../escape")


def test_prune_stale_runs_removes_only_old_run_roots(tmp_path):
    now = time.time()
    old = now - STALE_RUN_SECONDS - 60
    stale = tmp_path / "run-stale"
    (stale / "nested").mkdir(parents=True)
    (stale / "nested" / "file.txt").write_text("x", encoding="utf-8")
    fresh = tmp_path / "run-fresh"
    fresh.mkdir()
    other = tmp_path / "cache"
    other.mkdir()
    for directory in (stale, other):
        os.utime(directory, (old, old))

    prune_stale_runs(tmp_path, now=now)

    assert not stale.exists()
    assert fresh.exists()
    assert other.exists()
