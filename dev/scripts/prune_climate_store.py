"""Report and optionally delete orphaned historical climate stores.

Sibling of ``prune_series_cache.py``, which covers the WF2 *series* class only.
That script's contract is keyed to the CMIP6 series filename grammar under
``climate_projections/<clim_project>/scalar/``; the historical store is a
different artifact class in a different tree with a different key, so it gets
its own script rather than a second mode in that one.

**Why orphans happen here.** ``climate_historical/<clim_source>_<start>_<end>/``
is a *cache key*, not multi-window support (R09 design Finding 3,
`dev/milestones/r09/migration_project-tree.md`): two experiments sharing a
source and a window reuse one extraction. The consequence is that changing
``shared.clim_historical`` or ``shared.historical_window`` mints a NEW key and
strands the predecessor — with nothing in the repository to report it. Snakemake
cannot clean a directory it no longer declares.

Orphaned stores cannot corrupt a product: every consumer declares the active
store path explicitly, resolved from the same config this script reads. This is
disk hygiene, and the stores are large (a gridded multi-decade extraction), so
it is worth doing — but it is an explicit owner action.

**Default is a dry run.** Nothing is deleted without ``--delete``, mirroring
``prune_series_cache.py``'s stated contract: a store may have cost a long
download to produce.

Usage (from the repo root, inside pixi)::

    python dev/scripts/prune_climate_store.py --config config/workflows/snake_config_model_test.yml
    python dev/scripts/prune_climate_store.py --config <cfg> --delete

Not part of a run: this inspects and maintains a project tree
(see AGENTS.md, "Three homes for executables").
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import yaml  # noqa: E402

from blueearth_cst.shared.snake_utils import slugify_window  # noqa: E402

#: The store root, project-root-relative. Kept as one constant because R09 P2
#: moves it to ``data/climate/historical/`` — one edit, not a search.
STORE_ROOT = "data/climate/historical"


def active_store_key(config: dict) -> str:
    """The store key the config resolves to, built exactly as the workflow builds it.

    Mirrors ``snake_utils.climate_store_spec``: ``<clim_source>_<window slug>``,
    keyed at day resolution. Derived rather than globbed, so a key the workflow
    would not produce counts as an orphan instead of as a second active store.
    """
    shared = config["shared"]
    window = shared["historical_window"]
    return f"{shared['clim_historical']}_" + slugify_window(
        window["starttime"], window["endtime"]
    )


def find_orphans(project_dir: Path, active_key: str) -> list[Path]:
    """Store directories under ``climate_historical/`` other than the active one."""
    root = project_dir / STORE_ROOT
    if not root.is_dir():
        return []
    return sorted(
        p for p in root.iterdir() if p.is_dir() and p.name != active_key
    )


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
        help="actually delete the orphaned stores (default: report only)",
    )
    args = parser.parse_args(argv)

    with open(args.config, encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    project_dir = Path(config["project"]["project_dir"])
    active = active_store_key(config)
    root = project_dir / STORE_ROOT

    print(f"project        : {project_dir}")
    print(f"store root     : {root}")
    print(f"active key     : {active}")

    if not root.is_dir():
        print(f"nothing to do: {root} does not exist")
        return 0

    if not (root / active).is_dir():
        print(f"    NOTE     the active store is absent; it will be extracted "
              f"on the next run")

    orphans = find_orphans(project_dir, active)
    print(f"orphans        : {len(orphans)}")
    total = 0
    for p in orphans:
        size = _dir_size(p)
        total += size
        print(f"    ORPHAN   {p.name}/  ({size / 1024 / 1024:.1f} MB)")

    if not orphans:
        return 0
    print(f"\nreclaimable    : {len(orphans)} store(s), {total / 1024 / 1024:.1f} MB")

    if not args.delete:
        print("\nDRY RUN — nothing deleted. Re-run with --delete to remove them.")
        print(
            "Note: an orphaned store cannot corrupt a product. Every consumer "
            "declares the ACTIVE store path, resolved from this same config, so "
            "this is disk hygiene only."
        )
        return 0

    for p in orphans:
        shutil.rmtree(p)
        print(f"deleted {p}")
    print(f"\ndeleted {len(orphans)} store(s), reclaimed {total / 1024 / 1024:.1f} MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
