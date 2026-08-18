"""Tool caches live under `.tmp/`, and the repo root stays free of them.

The pytest and ruff caches were redirected out of the root on 2026-08-11
(`pyproject.toml` `cache_dir` / `cache-dir`) so that the root holds project
files rather than tool state. Nothing checked either half of that afterwards,
and both halves can fail quietly:

* **The setting can be dropped or misspelled.** Neither tool errors on a
  removed `cache_dir` -- each silently falls back to its own default at the
  root, which looks exactly like the state before the redirect.
* **A stale directory survives the redirect.** This is what actually happened:
  the root `.ruff_cache` and `.pytest_cache` were never deleted when the
  setting landed, so they sat there with content frozen at 2026-08-10 while
  every real run wrote to `.tmp/`. Seven days later they read as "the redirect
  is not working" -- a directory that is merely *stale* is indistinguishable,
  by eye, from one that is live.

The second case is why this checks the ROOT and not only the setting. A
config assertion alone would have been green throughout.

**The one invocation that still writes a root cache is `ruff check
--isolated`**, which discards config by definition -- and pyproject.toml and
the CI workflow both invite it as a diagnostic for the pinned rule set. Set
`RUFF_CACHE_DIR=.tmp/ruff_cache` when running it; that environment variable
takes precedence over `--isolated`'s config-less default (verified 2026-08-18),
so the honest fix is available at the call site rather than as an exception
here. `.gitignore` and ruff's self-written `.ruff_cache/.gitignore` keep such a
directory out of a commit either way -- this is about the root being READABLE,
which ignoring does not help with.
"""

import tomllib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

#: Where both tools are configured to write. One prefix, so the invariant is
#: "under the repo's one disposable directory" rather than two exact paths --
#: renaming `.tmp/ruff_cache` should not fail a hygiene test.
TMP_PREFIX = ".tmp/"


def _pyproject():
    with (REPO_ROOT / "pyproject.toml").open("rb") as handle:
        return tomllib.load(handle)


def test_pytest_cache_is_configured_under_tmp():
    cache_dir = _pyproject()["tool"]["pytest"]["ini_options"]["cache_dir"]
    assert cache_dir.replace("\\", "/").startswith(TMP_PREFIX), cache_dir


def test_ruff_cache_is_configured_under_tmp():
    cache_dir = _pyproject()["tool"]["ruff"]["cache-dir"]
    assert cache_dir.replace("\\", "/").startswith(TMP_PREFIX), cache_dir


def test_repo_root_carries_no_cache_directory():
    """No `.<something>cache<something>` directory directly in the root.

    Matched by SHAPE rather than by a list of the two names known today, so a
    tool nobody has added yet -- mypy, coverage's `.cache`, a future linter --
    is caught the first time it drops a directory here, instead of being
    noticed by eye weeks later. `.tmp/` itself is the destination and is not a
    match; nothing else in this repo's root is.
    """
    offenders = sorted(
        entry.name
        for entry in REPO_ROOT.iterdir()
        if entry.is_dir()
        and entry.name.startswith(".")
        and "cache" in entry.name.lower()
    )
    assert not offenders, (
        f"cache director{'y' if len(offenders) == 1 else 'ies'} in the repo root: "
        f"{', '.join(offenders)}. Tool caches belong under `.tmp/` "
        f"(pyproject.toml `cache_dir` / `cache-dir`). These are disposable -- "
        f"delete them. If one keeps coming back, the writer is running without "
        f"this repo's config: `ruff check --isolated` is the known case, and "
        f"`RUFF_CACHE_DIR=.tmp/ruff_cache` in front of it is the fix."
    )
