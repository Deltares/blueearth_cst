"""Report and optionally delete orphaned WF2 series files.

The series store became persistent at migration step 2b, and correctness
deliberately does not depend on the directory's contents: stage B reads an
explicit expanded list built from resolution and asserts each file's digest, so a
stale or orphaned series **cannot** enter a product (design §5.3, risk **R4**).
Pruning is therefore disk hygiene, and the design names it "an explicit user
action; a pruning helper is a follow-up, not a correctness fix". This is that
follow-up.

Two things make orphans routine rather than exceptional:

* **Key-grammar changes.** Steps 2b, 3 and 4b each changed the series filename
  (``historical_stats_time_{model}`` → ``series/{key}`` → ``series/{key}_{member}``
  → ``scalar/{key}``),
  leaving the previous generation behind each time.
* **Config changes.** Dropping a model or a scenario from the config removes it
  from resolution, so its series stops being referenced but stays on disk.

**Default is a dry run.** Nothing is deleted without ``--delete``, because the
whole point of the persistent cache is that a series may have cost hours to
produce while being a few KB to keep.

Usage (from the repo root, inside pixi)::

    python dev/scripts/prune_series_cache.py --config test_case/snake_config_baseline.yml
    python dev/scripts/prune_series_cache.py --config <cfg> --delete

Not part of a run: this inspects and maintains a project tree
(see AGENTS.md, "Three homes for executables").
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import yaml  # noqa: E402

from blueearth_cst.projections import resolution as res  # noqa: E402


def expected_keys(config: dict) -> tuple[set[str], str, str]:
    """The series keys the config's resolved combinations imply.

    Deliberately built the same way the Snakefile builds them — from
    **resolution**, not from the config cross-product — so a key that resolution
    would skip counts as an orphan rather than as expected-but-missing.
    """
    project = config["project"]
    my = config["workflows"]["climate_projections"]
    clim_project = my["clim_project"]
    catalog_path = project["data_sources_climate"]

    with open(REPO / catalog_path, encoding="utf-8") as handle:
        catalog = yaml.safe_load(handle)

    combinations = res.resolve(
        catalog,
        clim_project=clim_project,
        models=my["models"],
        scenarios=my["scenarios"],
        members=my["members"],
    )
    needed = {(c.dataset, "historical", c.member) for c in combinations if c.resolved}
    needed |= {(c.dataset, c.scenario, c.member) for c in combinations if c.resolved}

    keys = {
        f"{clim_project}_{model.replace('/', '_')}_{experiment}_{member}"
        for model, experiment, member in needed
    }
    return keys, project["project_dir"], clim_project


def main() -> None:
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
        help="actually delete the orphans (default: report only)",
    )
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    keys, project_dir, clim_project = expected_keys(config)
    clim_dir = Path(project_dir) / "data" / "climate" / "projections" / clim_project
    if not clim_dir.is_dir():
        print(f"nothing to do: {clim_dir} does not exist")
        return

    # S8-03: the tier directory is `scalar/`. The filename grammar is unchanged.
    series_dir = clim_dir / "scalar"
    current = {p for p in series_dir.glob("*.nc")} if series_dir.is_dir() else set()
    expected = {series_dir / f"{k}.nc" for k in keys}

    orphans_current_grammar = sorted(current - expected)
    # Pre-4b generations, which lived directly under the clim dir rather than in
    # a tier directory. Matched by their old prefixes so this helper stays useful
    # for a project that has not been re-run since. `series/` joins them at S8-03:
    # a project last run before the rename has a full generation stranded there,
    # and Snakemake cannot clean a directory it no longer declares.
    legacy = sorted(
        p
        for pattern in ("historical_stats_time_*", "stats_time-*")
        for p in clim_dir.glob(f"{pattern}/*.nc")
    ) + sorted((clim_dir / "series").glob("*.nc"))

    missing = sorted(p for p in expected if not p.exists())

    print(f"project        : {project_dir}")
    print(f"scalar dir     : {series_dir}")
    print(f"expected keys  : {len(expected)}")
    print(f"present        : {len(current)}")
    print(f"missing        : {len(missing)}  (will be derived on the next run)")
    for p in missing:
        print(f"    MISSING  {p.name}")
    print(f"orphans        : {len(orphans_current_grammar)}  (current grammar)")
    for p in orphans_current_grammar:
        print(f"    ORPHAN   {p.name}")
    print(f"legacy files   : {len(legacy)}  (pre-step-4b key grammars)")
    for p in legacy:
        print(f"    LEGACY   {p.relative_to(clim_dir)}")

    removable = orphans_current_grammar + legacy
    total_bytes = sum(p.stat().st_size for p in removable)
    print(f"\nreclaimable    : {len(removable)} file(s), {total_bytes / 1024:.0f} KB")

    if not removable:
        return
    if not args.delete:
        print("\nDRY RUN — nothing deleted. Re-run with --delete to remove them.")
        print(
            "Note: orphans cannot corrupt a product. Stage B reads an explicit "
            "expanded list and asserts each digest, so this is disk hygiene only "
            "(design §5.3, R4)."
        )
        return

    for p in removable:
        os.remove(p)
        print(f"deleted {p}")
    # Prune now-empty legacy directories, never a non-empty one.
    for parent in {p.parent for p in legacy}:
        try:
            if parent.is_dir() and not any(parent.iterdir()):
                parent.rmdir()
                print(f"removed empty dir {parent}")
        except OSError:
            pass
    print(f"\ndeleted {len(removable)} file(s), reclaimed {total_bytes / 1024:.0f} KB")


if __name__ == "__main__":
    main()
