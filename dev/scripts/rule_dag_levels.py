"""Print a Snakefile's rules in DAG (topological) order, with job counts.

Snakemake's own ``Job stats:`` table is sorted **alphabetically** and no flag
changes that (checked against snakemake 9.6.2), so it tells you what will run but
never what runs *before* what. This helper re-sorts the same information by
dependency depth.

Two non-executing snakemake calls do the work:

* ``--rulegraph dot`` — the rule-level dependency graph. Carries **every** rule
  the workflow defines, including ones with nothing left to do.
* ``--dag dot`` — the job-level graph. Read only to count jobs per rule and to
  separate the ones that will run from the ones already up to date (snakemake
  draws the latter ``style="rounded,dashed"``).

Both are read as **DOT**, never as ``--d3dag``: on 9.6.2 the D3 JSON silently
drops edges — 48 of them against the DOT graph's 73 for the same WF2 DAG — so a
level computed from it would be quietly wrong rather than merely uglier.
``--rulegraph mermaid-js`` is broken in the same version (emits self-edges and
mismatched pairs); ``dot`` is the only trustworthy renderer of the three.

The level is the **longest** path from a source, which makes it a lower bound on
the number of sequential stages: no ``--cores`` setting can start level *N*
before level *N-1* has finished. Rules sharing a level are mutually independent,
so that is where extra cores actually go.

What a level is **not** is a schedule. Snakemake starts a job the moment its own
inputs exist, not level by level, so independent chains overlap — one series can
be in ``reduce_gcm_series`` while another is still in ``fetch_gcm_slice``. Read the
levels as dependency depth, never as "these run together and nothing else does".

Usage (from the repo root, inside pixi)::

    python dev/scripts/rule_dag_levels.py -s analyze_projections.smk \\
        --configfile test_case/snake_config_baseline.yml

    # anything after `--` is forwarded to snakemake verbatim
    python dev/scripts/rule_dag_levels.py -s build_model.smk \\
        --configfile <cfg> -- --config project_dir=/tmp/probe

Not part of a run: this inspects a workflow, it never executes one
(see AGENTS.md, "Three homes for executables").
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import NamedTuple

REPO = Path(__file__).resolve().parents[2]

# `\t4[label = "fetch_gcm_slice\nseries_key: cmip6_...", color = "...", style="rounded,dashed"];`
# The label is taken non-greedily to the next quote. Snakemake does not escape
# quotes inside a label, so a wildcard VALUE containing `"` would truncate the
# rule name -- it cannot corrupt the graph, because only the first label line is
# used and the rule name never contains a quote.
_NODE = re.compile(r'^\s*(\d+)\[label\s*=\s*"(.*?)"([^\]]*)\]')
_EDGE = re.compile(r"^\s*(\d+)\s*->\s*(\d+)")


class Node(NamedTuple):
    """One DOT node: the rule it belongs to, and whether snakemake will skip it."""

    rule: str
    up_to_date: bool


def parse_dot(text: str) -> tuple[dict[int, Node], list[tuple[int, int]]]:
    """Read a snakemake ``--dag``/``--rulegraph`` DOT graph.

    Returns ``(nodes, edges)`` where an edge ``(u, v)`` means *v depends on u* —
    snakemake draws the arrow from dependency to dependent, so levels flow along
    the arrows rather than against them.
    """
    nodes: dict[int, Node] = {}
    edges: list[tuple[int, int]] = []
    for line in text.splitlines():
        node = _NODE.match(line)
        if node:
            node_id, label, attrs = node.groups()
            # A job label is `rule\nwildcard: value\n...`; the DOT file carries a
            # literal backslash-n, not a newline. Only the first line names the rule.
            rule = re.split(r"\\n|\n", label)[0].strip()
            nodes[int(node_id)] = Node(rule=rule, up_to_date="dashed" in attrs)
            continue
        edge = _EDGE.match(line)
        if edge:
            edges.append((int(edge.group(1)), int(edge.group(2))))
    return nodes, edges


def topological_levels(
    nodes: dict[int, Node], edges: list[tuple[int, int]]
) -> dict[int, int]:
    """Longest-path level per node: 0 for a source, ``max(preds) + 1`` otherwise.

    Longest path, not shortest — a rule reachable by both a short and a long
    chain cannot start until the long one finishes, so the short path would
    understate when it runs. WF2 has exactly this shape:
    ``plot_climate_proj_timeseries`` depends on both ``reduce_gcm_series``
    (level 2) and ``derive_change_factors`` (level 3), and belongs at 4.
    """
    successors: dict[int, list[int]] = defaultdict(list)
    indegree: dict[int, int] = {node_id: 0 for node_id in nodes}
    for upstream, downstream in edges:
        successors[upstream].append(downstream)
        indegree[downstream] = indegree.get(downstream, 0) + 1

    queue = [node_id for node_id, degree in indegree.items() if degree == 0]
    level = {node_id: 0 for node_id in queue}
    while queue:
        current = queue.pop()
        for downstream in successors[current]:
            level[downstream] = max(level.get(downstream, 0), level[current] + 1)
            indegree[downstream] -= 1
            if indegree[downstream] == 0:
                queue.append(downstream)

    if len(level) != len(nodes):
        stuck = sorted(nodes[n].rule for n in nodes if n not in level)
        raise ValueError(
            f"the graph is not acyclic — could not level {len(nodes) - len(level)} "
            f"node(s): {', '.join(stuck)}"
        )
    return level


def rule_levels(rulegraph_dot: str) -> dict[str, int]:
    """Level per RULE, from a ``--rulegraph`` graph (one node per rule)."""
    nodes, edges = parse_dot(rulegraph_dot)
    level = topological_levels(nodes, edges)
    return {nodes[node_id].rule: depth for node_id, depth in level.items()}


def job_counts(dag_dot: str) -> dict[str, tuple[int, int]]:
    """``{rule: (jobs_to_run, jobs_already_up_to_date)}`` from a ``--dag`` graph."""
    nodes, _ = parse_dot(dag_dot)
    counts: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for node in nodes.values():
        counts[node.rule][1 if node.up_to_date else 0] += 1
    return {rule: (pair[0], pair[1]) for rule, pair in counts.items()}


def run_snakemake(
    mode: str, snakefile: Path, configfile: Path, target: str, extra: list[str]
) -> str:
    """Invoke one non-executing snakemake graph mode and return its stdout.

    ``python -m snakemake`` rather than the console script, so the interpreter
    running this helper is the one that resolves the workflow's imports.
    """
    command = [
        sys.executable,
        "-m",
        "snakemake",
        target,
        mode,
        "dot",
        "-s",
        str(snakefile),
        "--configfile",
        str(configfile),
        *extra,
    ]
    result = subprocess.run(command, cwd=REPO, capture_output=True, text=True)
    if result.returncode != 0:
        sys.stderr.write(
            f"snakemake {mode} failed (exit {result.returncode}):\n"
            + "\n".join(result.stderr.strip().splitlines()[-25:])
            + "\n"
        )
        raise SystemExit(result.returncode)
    return result.stdout


def format_table(
    levels: dict[str, int], counts: dict[str, tuple[int, int]]
) -> list[str]:
    """The report body: one row per rule, ordered by level then name."""
    # A rule in the job graph but not the rule graph should be impossible; level
    # it last rather than dropping it, so the anomaly is visible.
    ordered = sorted(levels.items(), key=lambda item: (item[1], item[0]))
    unknown = sorted(set(counts) - set(levels))
    width = max([len(rule) for rule in list(levels) + unknown] + [4])

    lines = [f"{'level':>5}  {'rule':<{width}}  {'run':>5}  {'cached':>6}"]
    lines.append(f"{'-' * 5}  {'-' * width}  {'-' * 5}  {'-' * 6}")
    previous: int | None = None
    for rule, level in ordered:
        to_run, cached = counts.get(rule, (0, 0))
        if previous is not None and level != previous:
            lines.append("")
        lines.append(
            f"{level:>5}  {rule:<{width}}  {to_run or '-':>5}  {cached or '-':>6}"
        )
        previous = level
    for rule in unknown:
        to_run, cached = counts[rule]
        lines.append(
            f"{'?':>5}  {rule:<{width}}  {to_run or '-':>5}  {cached or '-':>6}"
        )
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "-s",
        "--snakefile",
        type=Path,
        required=True,
        help="the Snakefile to inspect (e.g. analyze_projections.smk)",
    )
    parser.add_argument(
        "--configfile",
        type=Path,
        required=True,
        help="the --configfile the workflow would be run with",
    )
    parser.add_argument(
        "--target",
        default="all",
        help="target rule to build the DAG for (default: all)",
    )
    parser.add_argument(
        "extra", nargs="*", help="extra arguments forwarded to snakemake, after `--`"
    )
    args = parser.parse_args()

    levels = rule_levels(
        run_snakemake(
            "--rulegraph", args.snakefile, args.configfile, args.target, args.extra
        )
    )
    counts = job_counts(
        run_snakemake("--dag", args.snakefile, args.configfile, args.target, args.extra)
    )

    print(f"workflow : {args.snakefile}")
    print(f"config   : {args.configfile}")
    print(f"target   : {args.target}")
    print()
    print("\n".join(format_table(levels, counts)))
    print()

    depth = max(levels.values()) + 1 if levels else 0
    total_run = sum(run for run, _ in counts.values())
    total_cached = sum(cached for _, cached in counts.values())
    widest = max(
        (
            sum(counts.get(r, (0, 0))[0] for r, lv in levels.items() if lv == level)
            for level in set(levels.values())
        ),
        default=0,
    )
    print(
        f"{depth} levels — the critical path is {depth} jobs long, so at least "
        f"{depth} stages run in sequence whatever --cores says."
    )
    print(f"{total_run} job(s) to run, {total_cached} already up to date.")
    # NOT a cap on --cores. Snakemake starts a job the moment its OWN inputs
    # exist, never level by level, so independent chains overlap across levels --
    # one series can be reducing while another is still fetching. Treat this as
    # the width of the busiest stage, not as peak concurrency.
    print(
        f"widest level holds {widest} runnable job(s) — a guide for --cores, not "
        f"a ceiling: chains at different levels overlap."
    )


if __name__ == "__main__":
    main()
