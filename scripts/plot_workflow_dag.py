"""Render a workflow's Snakemake DAG into the PROJECT folder, not the repo root.

The hand-rolled one-liner everyone reaches for::

    snakemake -s Snakefile_model_creation --configfile <cfg> --dag | dot -Tpng > dag_model.png

writes ``dag_model.png`` wherever the shell happens to be -- in practice the
repo root -- and names it after nothing in particular. The graph describes ONE
project's run, so it belongs under that config's own ``project_dir``, carrying
the project name and the workflow number:

    <project_dir>/logs/dag/<project_name>_wf<N>_dag.png              (wf1, wf2)
    <project_dir>/experiments/<id>/logs/dag/<name>_wf3_dag.png        (wf3)

**At the producing run's scope, under ``logs/``** (R9 design v10, principles P4
and P7). R7 put this under ``config/dag/`` on the reasoning that a run is
determined by the config plus the Snakefile, so a rendering of the second
belonged beside the snapshot of the first. R9 overrules that: ``config/`` is the
project's *editable + generated-provenance* root and P4 keeps generated run
records out of it, while P7 places every artifact at the scope of the producer
that wrote it. A DAG render is a generated record OF A RUN, so it goes with the
run's other records -- and WF3's belongs to its experiment, not to the project.

The old rationale's supporting claim still holds and is simply no longer
load-bearing: nothing digests the project's ``config/`` tree by listing it, so
the file never could have churned a fingerprint. It moves because of where it
belongs, not because it was dangerous.

Lives in ``scripts/`` rather than ``dev/scripts/`` because it writes a
user-facing artifact into a production ``project_dir`` from a user's project
config; every ``dev/scripts/`` tool reports on the repository instead
(AGENTS.md, "Three homes for executables").

Usage (inside ``pixi shell``, or via ``pixi run``, from the repo root)::

    python scripts/plot_workflow_dag.py -s Snakefile_model_creation --configfile <cfg>
    python scripts/plot_workflow_dag.py -s Snakefile_climate_experiment --configfile <cfg> \\
        --mode rulegraph --format svg

    # anything after `--` is forwarded to snakemake verbatim
    python scripts/plot_workflow_dag.py -s Snakefile_climate_projections --configfile <cfg> \\
        -- --config foo=bar

Not a Snakemake rule and deliberately so: a rule that renders the DAG would sit
inside the DAG it renders, and would show up in ``--summary`` and in the project
tree gates. It is also not an executing entry point -- ``--dag``/``--rulegraph``
build the graph and run nothing.

The PNG is an UNDECLARED artifact in the project tree (no rule produces it), so
running this against ``test_case/test_local`` adds a file that
``dev/scripts/semantic_tree_diff.py``'s whole-tree comparison will report.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import yaml

# Repo root = parent of scripts/. Snakemake must run from here: config values
# like `static_dir: config` and `data_sources: config/catalogs/*.yml` are
# repo-root relative, so inheriting the caller's cwd would break the DAG build
# for anyone invoking this from their project folder.
REPO_ROOT = Path(__file__).resolve().parents[1]

# Snakefile -> workflow number. The `wf<N>` labelling is the repo's existing
# convention for per-workflow artifacts; keep this in step with the merged-log
# names in blueearth_cst/shared/merge_logs.py (logs/wf1_model_creation.log ...).
WORKFLOW_NUMBER = {
    "Snakefile_model_creation": 1,
    "Snakefile_climate_projections": 2,
    "Snakefile_climate_experiment": 3,
}

# Where under project_dir the graph lands. Scope-dependent since R9 (P7): WF1
# and WF2 are project-scoped producers, WF3 is experiment-scoped.
PLOT_SUBDIR = Path("logs") / "dag"


def plot_subdir(number: int, config: dict) -> Path:
    """The DAG render's directory under ``project_dir``, at the run's scope.

    WF1 and WF2 write project-scoped records, so their graphs join the project's
    own ``logs/dag/``. WF3 is experiment-scoped -- its DAG describes one
    experiment's run -- so it lands in that experiment's ``logs/dag/``,
    alongside the merged ``wf3_climate_experiment.log`` (design v10, P7).

    Falls back to the project scope when a WF3 config carries no experiment
    name: the render is a convenience artifact and must not fail a user's
    command over a missing optional key.
    """
    if number != 3:
        return PLOT_SUBDIR
    experiment = (
        (config.get("workflows") or {}).get("climate_experiment") or {}
    ).get("experiment_name")
    if not experiment:
        return PLOT_SUBDIR
    return Path("experiments") / str(experiment) / "logs" / "dag"


class DagPlotError(Exception):
    """Raised for a bad Snakefile, config, or missing graphviz."""


def workflow_number(snakefile: Path) -> int:
    """The `wf<N>` number for a Snakefile, by filename."""
    try:
        return WORKFLOW_NUMBER[snakefile.name]
    except KeyError:
        known = ", ".join(sorted(WORKFLOW_NUMBER))
        raise DagPlotError(
            f"unknown Snakefile {snakefile.name!r}; expected one of: {known}"
        ) from None


def read_project(config_path: Path) -> tuple[Path, str, dict]:
    """``(project_dir, project_name, config)`` from a workflow config.

    The parsed config comes back too, so the caller can place a WF3 render
    at its experiment's scope without re-reading the file.

    Handles both config shapes in the repo: the R01 sectioned schema
    (``project.project_dir``) and the legacy single-workflow projections configs
    (top-level ``project_dir`` + ``project_name``). A relative ``project_dir``
    is resolved against the repo root -- the same directory Snakemake runs from,
    so the plot lands exactly where the workflow's own outputs do.

    The name falls back to the ``project_dir`` basename, which is what the R01
    schema offers; it is NOT routed through
    ``snake_utils.suggest_experiment_name`` (that adds a date stamp and enforces
    the experiment-name grammar -- neither wanted for a filename stem).
    """
    try:
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise DagPlotError(f"config file not found: {config_path}") from None
    if not isinstance(config, dict):
        raise DagPlotError(f"config file is not a YAML mapping: {config_path}")

    section = config.get("project")
    if isinstance(section, dict) and "project_dir" in section:
        raw_dir = section["project_dir"]
    elif "project_dir" in config:
        raw_dir = config["project_dir"]
    else:
        raise DagPlotError(
            f"no project_dir in {config_path} (looked for project.project_dir "
            f"and top-level project_dir)"
        )

    project_dir = Path(str(raw_dir))
    if not project_dir.is_absolute():
        project_dir = REPO_ROOT / project_dir
    name = config.get("project_name") or project_dir.name
    return project_dir, str(name), config


def build_graph(mode: str, snakefile: Path, config_path: Path, target: str,
                extra: list[str]) -> str:
    """Return the DOT text of one non-executing snakemake graph mode.

    ``python -m snakemake`` rather than the console script, so the interpreter
    running this helper is the one that resolves the workflow's imports (same
    reasoning as dev/scripts/rule_dag_levels.py).
    """
    command = [
        sys.executable, "-m", "snakemake", target,
        f"--{mode}", "dot",
        "-s", str(snakefile),
        "--configfile", str(config_path),
        *extra,
    ]
    result = subprocess.run(command, cwd=REPO_ROOT, capture_output=True, text=True)
    if result.returncode != 0:
        tail = "\n".join(result.stderr.strip().splitlines()[-25:])
        raise DagPlotError(
            f"snakemake --{mode} failed (exit {result.returncode}):\n{tail}"
        )
    # Snakemake writes its own progress to stderr, but be defensive: keep only
    # from the `digraph` header on, so a stray stdout line cannot reach `dot`.
    start = result.stdout.find("digraph")
    if start < 0:
        raise DagPlotError(
            f"snakemake --{mode} produced no DOT graph on stdout"
        )
    return result.stdout[start:]


def render(dot_text: str, output_path: Path, image_format: str) -> None:
    """Run graphviz `dot` over ``dot_text``, writing ``output_path``."""
    dot = shutil.which("dot")
    if dot is None:
        raise DagPlotError(
            "graphviz 'dot' not found on PATH -- run inside `pixi shell`, or "
            "prefix the command with `pixi run` (graphviz is a pixi dependency)"
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        [dot, f"-T{image_format}", "-o", str(output_path)],
        input=dot_text, capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise DagPlotError(
            f"graphviz dot failed (exit {result.returncode}):\n{result.stderr.strip()}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "-s", "--snakefile", type=Path, required=True,
        help="the Snakefile to graph (e.g. Snakefile_model_creation)",
    )
    parser.add_argument(
        "--configfile", type=Path, required=True,
        help="the --configfile the workflow would be run with",
    )
    parser.add_argument(
        "--mode", choices=("dag", "rulegraph"), default="dag",
        help="job-level DAG (default) or the rule-level graph",
    )
    parser.add_argument(
        "--format", dest="image_format", default="png",
        help="graphviz output format (default: png)",
    )
    parser.add_argument(
        "--target", default="all", help="target rule to graph (default: all)",
    )
    parser.add_argument(
        "extra", nargs="*", help="extra arguments forwarded to snakemake, after `--`",
    )
    args = parser.parse_args()

    try:
        number = workflow_number(args.snakefile)
        project_dir, project_name, config = read_project(args.configfile)
        output_path = (
            project_dir / plot_subdir(number, config)
            / f"{project_name}_wf{number}_{args.mode}.{args.image_format}"
        )
        dot_text = build_graph(
            args.mode, args.snakefile, args.configfile, args.target, args.extra
        )
        render(dot_text, output_path, args.image_format)
    except DagPlotError as error:
        sys.stderr.write(f"{error}\n")
        return 1

    print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
