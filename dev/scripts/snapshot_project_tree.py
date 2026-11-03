"""Snapshot a project tree as a path list and check it holds nothing undeclared.

The check runs against `semantic_tree_diff.build_project_tree_rules`, the
post-migration INVENTORY. (The one-way R9 migration map this tool also drove
was retired 2026-08-11 -- `dev/records/reviews/2026-08-11_test-suite-bloat-assessment.md`
§6a -- so `--map` now has one choice.)

Wraps steps 0, 3 and 4 of `dev/records/milestones/r09/observed-tier-runbook.md` in one
command, and derives every map parameter from the config instead of asking for
it on the command line -- the experiment name, the historical store key and the
`clim_project` are all config-determined, and a mistyped `--dataset-key` would
silently turn a mapped store into an unmapped one.

Two uses, and the difference is only whether `--out` is given:

* **Adjudicate before pruning** (runbook step 0) -- no `--out`, nothing is
  written. Every `UNMAPPED` line is then triaged as a leftover orphan or a real
  map gap.
* **Record the observed tier** (runbook steps 3-4) -- with `--out`, the sorted
  path list is written with a provenance header that a reader can regenerate
  from.

`.snakemake/` is excluded: it is Snakemake's bookkeeping, not a project
artifact, and `semantic_tree_diff` excludes it from tree walks for the same
reason. Nothing else is filtered -- an observed snapshot is meant to carry the
undeclared engine artifacts that no `output:` declaration names.

**Reads only.** This script never writes into `project_dir` and never deletes;
pruning is `prune_series_cache.py` and `prune_climate_store.py`, both of which
require an explicit `--delete`.

Usage (from the repo root, inside pixi)::

    # step 0 -- what does the map not cover in the tree as it stands?
    python dev/scripts/snapshot_project_tree.py --config <cfg>

    # steps 3-4 -- record the snapshot and check it
    python dev/scripts/snapshot_project_tree.py --config <cfg> \
        --out <path>/observed_inventory.txt

Exit 0 when every path is classified, 1 when any path is UNMAPPED.

Not part of a run: this inspects a project tree
(see AGENTS.md, "Three homes for executables").
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import date
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(HERE))

import semantic_tree_diff as std  # noqa: E402

from blueearth_cst.experiment.reservation import preview_experiment  # noqa: E402
from blueearth_cst.shared.config_composition import (  # noqa: E402
    load_composed_config,
)
from blueearth_cst.shared.snake_utils import (  # noqa: E402
    historical_window_bounds,
    slugify_window,
)

#: Directory names whose contents are never part of a project snapshot.
EXCLUDED_DIRS = frozenset({".snakemake"})


def map_parameters(config: dict, project_dir: Path | None = None) -> dict:
    """Derive every path-map parameter from the config.

    Built the same way the workflows build them, so a snapshot cannot disagree
    with the tree it describes: the store key mirrors
    `snake_utils.climate_store_rule`, and the experiment name and clim_project
    come from the sections that own them.
    """
    project = config["project"]
    climate = config["climate"]
    workflows = config.get("workflows", {})
    experiment = workflows.get("simulate_system", {})
    projections = workflows.get("analyze_projections", {})

    # `C-70` retyped `climate.window` to INCLUSIVE YEARS, and the store key is
    # still ISO at day resolution. `historical_window_bounds` is the one place
    # that conversion lives — going through it is what keeps this tool's key
    # byte-identical to the one `climate_store_rule` builds for a real run.
    # Formatting the years here instead would be a second implementation of the
    # same rule, free to drift, and the symptom would be a snapshot that never
    # matches a tree.
    start, end = historical_window_bounds(climate["window"])
    root = project_dir or resolve_project_dir(project["project_dir"])
    selected = preview_experiment(root, experiment.get("experiment_name"))
    if selected.action in {"refuse", "missing"}:
        raise ValueError(f"cannot select inspected experiment: {selected.reason}")
    return {
        "project_dir": root.as_posix()
        if project_dir is not None
        else project["project_dir"],
        "experiment_name": selected.name,
        # `C-44`: the selected source. `C-25`: `clim_project` is `ensemble`.
        "dataset_key": f"{climate['selected']}_"
        + slugify_window(start.isoformat(), end.isoformat()),
        "clim_project": projections.get("ensemble", "cmip6"),
    }


def retained_experiment_names(project_dir: Path) -> tuple[str, ...]:
    """Recognize owned retained names; unknown occupants remain inventory orphans."""
    directory = project_dir / "experiments"
    if not directory.is_dir():
        return ()
    names = []
    for child in sorted(directory.iterdir()):
        if not child.is_dir():
            continue
        selected = preview_experiment(project_dir, child.name)
        if selected.name == child.name and selected.action in {"continue", "legacy"}:
            names.append(child.name)
    return tuple(names)


def list_tree(project_dir: Path) -> list[str]:
    """Every file under `project_dir` as a sorted project-relative POSIX path."""
    out: set[str] = set()
    for path in project_dir.rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(project_dir)
        if EXCLUDED_DIRS.intersection(rel.parts):
            continue
        out.add(rel.as_posix())
    return sorted(out)


def resolve_project_dir(raw: str, base: Path | None = None) -> Path:
    """Resolve `project.project_dir` the way Snakemake does -- against the CWD.

    Deliberately NOT against this script's own repository root. The runbook has
    the workflows run from the PRIMARY checkout while the comparator lives in a
    task worktree, so the two differ; resolving against the script's repo would
    silently look for the tree beside the tool instead of beside the run.
    """
    path = Path(raw)
    return path if path.is_absolute() else (base or Path.cwd()) / path


def _git_commit(cwd: Path) -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=cwd,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):  # pragma: no cover
        return "unknown"


def build_header(config_path: Path, params: dict, n_paths: int) -> str:
    """The provenance block a reader needs to regenerate the snapshot."""
    return "\n".join(
        [
            "# Project-tree snapshot --- a sorted list of project-relative paths.",
            "#",
            "# The OBSERVED tier of the two-tier inventory ruled in",
            "# dev/records/milestones/r09/migration_project-tree.md, *The inventory the map",
            "# is validated against*: the only tier that carries UNDECLARED engine",
            "# artifacts (hydromt, Wflow.jl, weathergenr), which appear in no",
            "# `output:` declaration and which --dry-run structurally cannot see.",
            "#",
            "# PROVENANCE",
            f"#   generated   {date.today().isoformat()} by dev/scripts/{Path(__file__).name}",
            # Two commits, because they can differ: the run checkout determines the
            # tree shape, the tool checkout determines the map it is checked against.
            f"#   run commit  {_git_commit(Path.cwd())}  ({Path.cwd().as_posix()})",
            f"#   tool commit {_git_commit(REPO)}  ({REPO.as_posix()})",
            f"#   config      {config_path.as_posix()}",
            f"#   project_dir {params['project_dir']}",
            f"#   experiment  {params['experiment_name']}",
            f"#   store key   {params['dataset_key']}",
            f"#   clim_project {params['clim_project']}",
            f"#   paths       {n_paths}",
            "#",
            "# POSIX separators, sorted, deduplicated. `.snakemake/` is excluded:",
            "# Snakemake bookkeeping, not a project artifact. Everything else is",
            "# kept, including artifacts no rule declares --- that is the point.",
            "#",
            "# REGENERATE",
            f"#   python dev/scripts/{Path(__file__).name} --config {config_path.as_posix()} \\",
            "#       --out <this file>",
            "",
        ]
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="the --configfile the project was run with; every map parameter "
        "is derived from it",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="write the snapshot here, with a provenance header. Omit to "
        "inspect without writing anything (runbook step 0)",
    )
    parser.add_argument(
        "--no-check",
        action="store_true",
        help="snapshot only; skip the path-map check (and its exit code)",
    )
    parser.add_argument(
        "--map",
        choices=("current",),
        default="current",
        help="which path map to check against. `current` is the "
        "POST-MIGRATION INVENTORY -- it asks 'does this tree hold "
        "anything nobody declared?'. The `r09` alternative, the one-way "
        "pre-R9 -> post-R9 migration map, was retired 2026-08-11 "
        "(dev/records/reviews/2026-08-11_test-suite-bloat-assessment.md); no "
        "un-migrated tree survives to point it at. See also "
        "pre-board record 45e3a564 [R10-11]",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="print only the UNMAPPED lines and the summary, not the full "
        "per-path table. The MOVED/IDENTITY rows are the gate's evidence, "
        "so omit this when producing that evidence",
    )
    parser.add_argument(
        "--project-dir",
        type=Path,
        default=None,
        help="override the config's project_dir. A relative project_dir "
        "resolves against the CURRENT DIRECTORY, as Snakemake resolves "
        "it -- pass this when running the tool from somewhere other than "
        "the checkout the workflows ran from",
    )
    args = parser.parse_args(argv)

    # COMPOSED, not raw (R13 D-12.0): the file passed here is the PROJECT
    # config, and since the split the workflow settings live in the files it
    # points at. A raw load would read a two-key stanza and fall back to its
    # defaults -- silently, which is the failure mode this tool has no way to
    # report.
    config = load_composed_config(args.config)
    project_dir = (
        args.project_dir
        if args.project_dir is not None
        else resolve_project_dir(config["project"]["project_dir"])
    )
    if not project_dir.is_dir():
        parser.error(
            f"project_dir does not exist: {project_dir}\n"
            f"  (config says {config['project']['project_dir']!r}; a relative path "
            f"resolves against the current directory, {Path.cwd()}). "
            "Run from the checkout the workflows ran from, or pass "
            "--project-dir."
        )
    try:
        params = map_parameters(config, project_dir)
    except ValueError as error:
        parser.error(str(error))

    paths = list_tree(project_dir)
    print(f"project_dir   : {project_dir}")
    print(f"experiment    : {params['experiment_name']}")
    print(f"store key     : {params['dataset_key']}")
    print(f"paths         : {len(paths)}")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            build_header(args.config, params, len(paths)) + "\n".join(paths) + "\n",
            encoding="utf-8",
        )
        print(f"wrote         : {args.out}")

    if args.no_check:
        return 0

    path_map = std.build_project_tree_rules(
        params["experiment_name"],
        params["dataset_key"],
        params["clim_project"],
        retained_experiment_names=retained_experiment_names(project_dir),
    )
    # No DELETED class: the one that existed (`indicators/RT_*.csv`, deleted
    # rather than migrated) belonged to the retired R9 map. The inventory is
    # identity-only, so every declared path is IDENTITY and everything else is
    # UNMAPPED -- which is the whole question this gate asks.
    rows = std.classify_path_map(paths, path_map, None)
    print(f"map           : {args.map}")

    print()
    report = std.format_path_map_report(rows)
    if args.quiet:
        report = "\n".join(
            line
            for line in report.splitlines()
            if line.startswith(("UNMAPPED", "MAP CLEAN"))
        )
    print(report)

    unmapped = [old for old, _, kind in rows if kind == "UNMAPPED"]
    if unmapped:
        print(
            f"\n{len(unmapped)} path(s) the map does not cover. Each is either a "
            "leftover ORPHAN (prune it -- see prune_series_cache.py, "
            "prune_climate_store.py, and the hand list in "
            "dev/records/milestones/r09/observed-tier-runbook.md) or a real INVENTORY "
            "GAP "
            "(stop and report it: amending the map is an owner decision)."
        )
    return 1 if unmapped else 0


if __name__ == "__main__":
    raise SystemExit(main())
