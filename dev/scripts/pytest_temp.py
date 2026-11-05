"""Worktree-local pytest scratch with Windows inherited directory permissions.

Pytest's 0700 mkdir ACL excludes restricted Windows process tokens. Limit the
workaround to pytest's directory factory; ordinary application I/O is untouched.
The factory integration uses pytest internals and needs checking on upgrades.
"""

import os
import time
from pathlib import Path
from uuid import uuid4

import pytest
from _pytest.pathlib import make_numbered_dir, rm_rf
from _pytest.tmpdir import TempPathFactory

# Runs younger than this stay for debugging and protect concurrent invocations.
STALE_RUN_SECONDS = 2 * 24 * 3600


def prune_stale_runs(parent: Path, now: float | None = None) -> None:
    """Remove `run-*` roots under `parent` untouched for `STALE_RUN_SECONDS`."""
    cutoff = (time.time() if now is None else now) - STALE_RUN_SECONDS
    for run in parent.glob("run-*"):
        try:
            if run.is_symlink() or run.is_junction() or not run.is_dir():
                continue
            if run.stat().st_mtime < cutoff:
                rm_rf(run)
        except OSError:
            continue  # In use or inaccessible: leave it for a later run.


class WorktreeTempFactory(TempPathFactory):
    """Isolate independent runs, prune stale ones, inherit Windows directory ACLs."""

    def getbasetemp(self) -> Path:
        """Create a private run root, or honor explicit pytest basetemp semantics."""
        if self._basetemp is not None:
            return self._basetemp
        if self._given_basetemp is not None:
            if os.name != "nt":
                return super().getbasetemp()
            base = self._given_basetemp
            if base.exists():
                rm_rf(base)
            base.mkdir()
        else:
            parent = Path(__file__).resolve().parents[2] / ".tmp" / "pytest"
            parent.mkdir(parents=True, exist_ok=True)
            prune_stale_runs(parent)
            base = parent / f"run-{uuid4().hex}"
            base.mkdir(mode=0o777 if os.name == "nt" else 0o700)
        self._basetemp = base.resolve()
        return self._basetemp

    def mktemp(self, basename: str, numbered: bool = True) -> Path:
        """Preserve pytest naming and path validation with inherited Windows ACLs."""
        if os.name != "nt":
            return super().mktemp(basename, numbered=numbered)
        basename = self._ensure_relative_to_basetemp(basename)
        if numbered:
            return make_numbered_dir(self.getbasetemp(), prefix=basename, mode=0o777)
        directory = self.getbasetemp() / basename
        directory.mkdir()
        return directory


@pytest.hookimpl(specname="pytest_configure", tryfirst=True)
def pytest_configure_cache(config: pytest.Config) -> None:
    """Prepare the cache before pytest's cache provider initializes it."""
    if not config.pluginmanager.hasplugin("cacheprovider"):
        return
    # Avoid cacheprovider's tempfile.mkdtemp (also 0700 on Windows), without
    # changing tempfile behavior in the application or tests.
    cache = config.rootpath / config.getini("cache_dir")
    cache.mkdir(parents=True, exist_ok=True)


@pytest.hookimpl(trylast=True)
def pytest_configure(config: pytest.Config) -> None:
    """Install the directory factory before fixtures or xdist access it."""
    config._tmp_path_factory = WorktreeTempFactory.from_config(config, _ispytest=True)
