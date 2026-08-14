"""Report and optionally delete config artifacts the snapshot redesign retired.

Third sibling of ``prune_series_cache.py`` and ``prune_climate_store.py``, and
the narrowest of the three: it is a ONE-SHOT MIGRATION, not a recurring pruner.
Once a project has been brought to the new shape, this script has nothing left
to find.

**What became stale, and why nothing else reports it.** The config-snapshot
redesign (2026-08-13) removed the content-addressed bundle and replaced
bin-level copying with a per-file predicate:

- ``<project_dir>/config/runs/<workflow>/<digest>/`` bundles are no longer
  written by anything. Snakemake cannot clean a directory it no longer
  declares, so every bundle an existing project accumulated simply stays.
- ``config/templates/`` and ``config/catalogs/`` copies are now made only when
  the toolbox repository CANNOT give the file back. A copy that is
  byte-identical to a tracked toolbox file is exactly what the new predicate
  would no longer make.

**What this refuses to touch, and why each matters.**

- ``<exp_dir>/config/catalogs/`` — holds the GENERATED experiment catalog,
  written over generated forcing at run time. The design keeps it. A pattern
  match on ``config/catalogs/`` across the whole tree would delete it, which is
  why the experiment tree is excluded outright rather than filtered.
- ``config/basin_data/`` — copies of files that live outside the repository
  and outside ``project_dir``. The predicate copies them BY DESIGN: the toolbox
  cannot give them back, and losing them costs the project its record of what
  it was built and evaluated against. Named ``observations/`` until 2026-08-14;
  a pre-rename project tree keeps the old bin, which nothing here deletes.
- Anything in ``templates/`` or ``catalogs/`` that is NOT byte-identical to a
  tracked toolbox file. That is precisely the site-specific catalog R4 exists
  to protect, so it is reported and never deleted.

**Default is a dry run.** Nothing is deleted without ``--delete``, the same
contract the two sibling scripts state. Deletion here is irreversible in a way
theirs is not: a bundle is the only remaining record of a configuration a
project once ran under, and no rule will regenerate it.

Usage (from the repo root, inside pixi)::

    python dev/scripts/prune_config_snapshots.py --config test_case/snake_config_baseline.yml
    python dev/scripts/prune_config_snapshots.py --config <cfg> --delete

Not part of a run: this inspects and maintains a project tree
(see AGENTS.md, "Three homes for executables").
"""

from __future__ import annotations

import argparse
import hashlib
import re
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import yaml  # noqa: E402

#: A retired bundle directory: ``config/runs/<workflow>/<12-hex>/``.
#:
#: The digest length is the repo's ``SHORT_DIGEST_CHARS``. Anchored at both
#: ends and restricted to hex so a directory that merely LOOKS bundle-shaped --
#: a workflow subdirectory holding today's ``run_record.yml``, for instance --
#: cannot match.
BUNDLE_DIR = re.compile(r"^[0-9a-f]{12}$")

#: Bins whose copies are now made only when the repo cannot give the file back.
RECOVERABLE_BINS = ("config/templates", "config/catalogs")


def tracked_file_hashes() -> dict[str, set[str]]:
    """SHA-256 of every tracked file in the toolbox, keyed by basename.

    Keyed by basename rather than by path because a project copy has lost its
    original location: ``config/templates/wflow_build_model.yml`` in a project
    came from ``config/defaults/wflow_build_model.yml`` in the repo. The hash
    is what decides; the name only narrows the search.

    Returns an empty mapping when git cannot answer -- an exported tree, or a
    container. Every candidate then fails the "identical to a tracked file"
    test and is reported rather than deleted, which is the safe direction.
    """
    try:
        listing = subprocess.run(
            ["git", "ls-files", "-z"],
            cwd=REPO,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        return {}
    if listing.returncode != 0:
        return {}

    hashes: dict[str, set[str]] = {}
    for relative in listing.stdout.split("\0"):
        if not relative:
            continue
        path = REPO / relative
        try:
            if not path.is_file():
                continue
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
        except OSError:
            continue
        hashes.setdefault(path.name, set()).add(digest)
    return hashes


def _sha256(path: Path) -> str | None:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return None


def find_bundles(project_dir: Path) -> list[Path]:
    """Retired content-addressed bundle directories, project-wide.

    Includes the experiment trees, whose bundles lived at
    ``experiments/<id>/config/runs/run_stress_test/<digest>/`` -- those are
    retired too. The exclusion that matters is ``<exp_dir>/config/catalogs/``,
    which is a different directory entirely and is never a bundle.
    """
    found: list[Path] = []
    for runs_dir in project_dir.glob("**/config/runs"):
        for workflow_dir in sorted(p for p in runs_dir.iterdir() if p.is_dir()):
            if workflow_dir.name == "invocations":
                continue
            found.extend(
                sorted(
                    p
                    for p in workflow_dir.iterdir()
                    if p.is_dir() and BUNDLE_DIR.match(p.name)
                )
            )
    return found


def find_recoverable_copies(
    project_dir: Path, tracked: dict[str, set[str]]
) -> tuple[list[Path], list[Path]]:
    """Split the two recoverable bins into (deletable, reported-only).

    Deletable means byte-identical to a tracked toolbox file -- the repository
    can give it back, so the new predicate would not have copied it. Everything
    else is site-specific and is only reported.

    **Project scope only.** These are ``<project_dir>/config/...`` exactly, not
    a recursive walk: ``<exp_dir>/config/catalogs/`` holds the generated
    experiment catalog the design keeps, and a ``**`` glob would sweep it in.
    """
    deletable: list[Path] = []
    reported: list[Path] = []
    for bin_name in RECOVERABLE_BINS:
        directory = project_dir / bin_name
        if not directory.is_dir():
            continue
        for path in sorted(p for p in directory.rglob("*") if p.is_file()):
            digest = _sha256(path)
            if digest is not None and digest in tracked.get(path.name, ()):
                deletable.append(path)
            else:
                reported.append(path)
    return deletable, reported


def _dir_size(path: Path) -> int:
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="the --configfile the project was run with",
    )
    parser.add_argument(
        "--delete",
        action="store_true",
        help="actually delete the retired artifacts (default: report only)",
    )
    args = parser.parse_args(argv)

    with open(args.config, encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    project_dir = Path(config["project"]["project_dir"])
    print(f"project        : {project_dir}")

    if not project_dir.is_dir():
        print(f"nothing to do: {project_dir} does not exist")
        return 0

    tracked = tracked_file_hashes()
    if not tracked:
        print(
            "    NOTE     git could not list tracked files, so NOTHING in "
            "templates/ or catalogs/ can be proven recoverable. Those bins are "
            "reported only."
        )

    bundles = find_bundles(project_dir)
    deletable, reported = find_recoverable_copies(project_dir, tracked)

    total = 0
    print(f"bundles        : {len(bundles)}")
    for path in bundles:
        size = _dir_size(path)
        total += size
        print(
            f"    RETIRED  {path.relative_to(project_dir).as_posix()}/  "
            f"({size / 1024:.1f} KB)"
        )

    print(f"recoverable    : {len(deletable)}")
    for path in deletable:
        size = path.stat().st_size
        total += size
        print(
            f"    RETIRED  {path.relative_to(project_dir).as_posix()}  "
            f"({size / 1024:.1f} KB)"
        )

    if reported:
        print(f"kept (not in the repo, or modified): {len(reported)}")
        for path in reported:
            print(f"    KEEP     {path.relative_to(project_dir).as_posix()}")

    targets: list[Path] = [*bundles, *deletable]
    if not targets:
        print("\nnothing to migrate: this project is already in the new shape")
        return 0

    print(f"\nreclaimable    : {len(targets)} item(s), {total / 1024:.1f} KB")

    if not args.delete:
        print("\nDRY RUN — nothing deleted. Re-run with --delete to remove them.")
        print(
            "Deletion is irreversible: a bundle is the only remaining record "
            "of a configuration this project once ran under, and no rule "
            "regenerates it. Review the list above first."
        )
        return 0

    for path in targets:
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()
        print(f"deleted {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
