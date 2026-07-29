"""Scaffold a dummy `project_dir` tree from the Snakefiles, without running anything.

Layout review tool. `snakemake --summary` builds the DAG and prints every
**declared** output plus its log path; this script parses that, optionally
rewrites paths through a rename map, and materializes the result as empty files
(with small placeholders where the file's *shape* is the thing under review —
logs and benchmark reports). Nothing is computed and no workflow runs.

Two gaps are covered deliberately rather than pretended away:

- `--summary` sees only declared outputs. Undeclared artifacts (``signatures_*.png``,
  wflow's own ``run_default/`` files, weathergenr's R-written outputs) come from
  an explicit overlay file, ``scaffold_extras.yml``, so what is guessed stays
  reviewable. Dry-runs being blind to ``params:``-string paths and R ``shell:``
  bodies is a known property of this repo (``dev/r07/project-layout-design.md``).
- WF2/WF3 declare wf1 leaves as `ancient(...)` cross-workflow inputs that
  Snakemake will not satisfy on its own. They are staged into the scratch tree
  first, exactly as ``tests/test_cli.py``'s ``config_with_staged_region`` does.

Usage
-----
    pixi run python dev/scripts/scaffold_project_tree.py --print-tree
    pixi run python dev/scripts/scaffold_project_tree.py \
        --rename-map dev/tmp/proposed_layout.yml --print-tree

The rename map is a YAML file of prefix rewrites applied to project-relative
paths, longest prefix first::

    renames:
      - from: hydrology_model/forcing/plots/
        to:   hydrology_model/plots/
      - from: logs/1.11_plot_results.log
        to:   logs/_parts/1.14_plot_results.log

It doubles as the migration checklist once a layout is agreed.
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "config/workflows/snake_config_model_test.yml"
DEFAULT_EXTRAS = Path(__file__).resolve().parent / "scaffold_extras.yml"
DEFAULT_OUT = REPO_ROOT / "dev/tmp/scaffold"

SNAKEFILES = {
    1: "Snakefile_model_creation",
    2: "Snakefile_climate_projections",
    3: "Snakefile_climate_experiment",
}

# Minimal valid region so WF2/WF3 DAGs resolve; same role as the test fixture's.
_MINIMAL_REGION_GEOJSON = """{
  "type": "FeatureCollection",
  "features": [{
    "type": "Feature", "properties": {"value": 1},
    "geometry": {"type": "Polygon", "coordinates": [[
      [11.3, -1.05], [13.6, -1.05], [13.6, 0.9], [11.3, 0.9], [11.3, -1.05]]]}
  }]
}
"""

_LOG_PLACEHOLDER = """\
# BlueEarth-CST | project: {project} | <date>
# project dir: {project_dir}
# log: {name} | started <hh:mm:ss>

<scaffold placeholder — no run output>
"""

_BENCHMARK_PLACEHOLDER = """\
# scaffold placeholder — no run output

| rule | s | h:m:s |
|:-----|--:|:------|
| <rule> | 0.00 | 0:00:00 |
| **TOTAL** | 0.00 | 0:00:00 |
"""


def _scratch_config(config_path: Path, project_dir: Path, dest: Path) -> Path:
    """Write a copy of ``config_path`` whose project_dir points at the scratch tree."""
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    cfg["project"]["project_dir"] = project_dir.as_posix()
    dest.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return dest


def _stage_cross_workflow_inputs(project_dir: Path, config_path: Path) -> None:
    """Stage the wf1 leaves WF2/WF3 declare as ancient() inputs."""
    region = project_dir / "hydrology_model/staticgeoms/region.geojson"
    region.parent.mkdir(parents=True, exist_ok=True)
    region.write_text(_MINIMAL_REGION_GEOJSON, encoding="utf-8")

    snapshot = project_dir / "config/runs/snake_config_model_creation.yml"
    snapshot.parent.mkdir(parents=True, exist_ok=True)
    snapshot.write_text(config_path.read_text(encoding="utf-8"), encoding="utf-8")


def _summary(snakefile: str, config_path: Path) -> list[str]:
    """Return declared output + log paths for one workflow, via `snakemake --summary`."""
    cmd = [
        "snakemake", "all", "-c", "1",
        "-s", str(REPO_ROOT / snakefile),
        "--configfile", str(config_path),
        "--summary",
    ]
    proc = subprocess.run(
        cmd, cwd=REPO_ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace"
    )
    if proc.returncode != 0:
        sys.stderr.write(f"\n!! {snakefile} --summary failed:\n{proc.stderr[-2000:]}\n")
        return []

    paths: list[str] = []
    for line in proc.stdout.splitlines():
        cols = [c.strip() for c in line.rstrip().split("\t")]
        if len(cols) < 4 or cols[0] == "output_file":
            continue
        paths.append(cols[0])
    return paths


def _log_paths(snakefile: str) -> list[str]:
    """Synthesize `logs/<W.NN>_<rule>.log` from the Snakefile's rule_banner() calls.

    `--summary`'s log column reports only logs that already EXIST on disk (it
    prints `-` otherwise), so it is useless against a fresh project_dir. The
    numbered log filename is a repo convention (`rule_banner`'s docstring: the
    W.NN matches the rule's log/benchmark filenames), so reading the banners
    reproduces it — and keeps the scaffold honest when the numbering changes.
    """
    text = (REPO_ROOT / snakefile).read_text(encoding="utf-8")
    pattern = re.compile(r"""rule_banner\(\s*["']([^"']+)["']\s*,\s*f?["']([^"']+)["']""")
    logs = []
    for number, name in pattern.findall(text):
        # wf3 builds batch rule names by f-string: run_wflow_batch_{_b}. Show one.
        name = re.sub(r"\{[^}]*\}", "1", name)
        logs.append(f"logs/{number}_{name}.log")
    return logs


def _load_delta(path: Path | None) -> tuple[list[tuple[str, str]], list[str], list[str]]:
    """Load a layout delta: `renames` (prefix or exact), `drops`, `adds`.

    Three verbs are enough to express every layout revision discussed so far:
    a move (`renames`), an artifact that stops existing (`drops` — e.g. per-rule
    logs that a merge step consumes), and one that starts (`adds`).
    """
    if path is None:
        return [], [], []
    spec = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    pairs = [(str(r["from"]), str(r["to"])) for r in spec.get("renames", [])]
    # Longest prefix first, so a specific rule beats a general one.
    pairs.sort(key=lambda p: len(p[0]), reverse=True)
    return pairs, list(spec.get("drops", []) or []), list(spec.get("adds", []) or [])


def _apply_renames(rel: str, renames: list[tuple[str, str]]) -> str:
    for src, dst in renames:
        if rel == src:
            return dst
        if src.endswith("/") and rel.startswith(src):
            return dst + rel[len(src):]
    return rel


_RULE_SEP = "# " + "═" * 72


def _merged_log(project_dir: Path, part_logs: list[str]) -> str:
    """Render the merged per-workflow log: contents index + one block per rule.

    The separator line *is* the index — `grep "^# 1\\."` reproduces the contents —
    and carries the status, so failures are one grep away.
    """
    rules = []
    for rel in sorted(part_logs):
        stem = Path(rel).stem
        number, _, name = stem.partition("_")
        rules.append((number, name))

    lines = [
        f"# BlueEarth-CST | project: {project_dir.name} | <date>",
        f"# project dir: {project_dir.as_posix()}",
        f"# log: wf1_run.log | {len(rules)} rules, {len(rules)} ran / 0 cached"
        " | <hh:mm:ss> → <hh:mm:ss> | total <h:mm:ss>",
        "#",
        "# contents",
    ]
    line_no = len(lines) + len(rules) + 2
    for number, name in rules:
        lines.append(f"#   {number}  {name:<32} <s>   line {line_no:>5}")
        line_no += 8  # each block: blank + 3 separator lines + ~4 lines of output
    lines.append("")

    for number, name in rules:
        lines += [
            _RULE_SEP,
            f"# {number}  {name:<32} <hh:mm:ss> → <hh:mm:ss>   <s>  ok",
            _RULE_SEP,
            "<scaffold placeholder — this rule's captured output>",
            "",
        ]
    return "\n".join(lines) + "\n"


def _placeholder(rel: str, project_dir: Path, part_logs: list[str]) -> str:
    """Content for files whose shape is under review; everything else stays empty."""
    name = Path(rel).name
    if re.fullmatch(r"wf\d+_run(\.partial)?\.log", name):
        return _merged_log(project_dir, part_logs)
    if rel.startswith("logs/") and name.endswith(".log"):
        return _LOG_PLACEHOLDER.format(
            project=project_dir.name, project_dir=project_dir.as_posix(), name=name
        )
    if rel.startswith("benchmarks/") and name.endswith((".md", ".tsv")):
        return _BENCHMARK_PLACEHOLDER
    return ""


def _print_tree(root: Path) -> None:
    """Render the scaffolded tree, directories first, one indent level per depth."""
    def walk(d: Path, prefix: str) -> None:
        entries = sorted(d.iterdir(), key=lambda p: (p.is_file(), p.name.lower()))
        for i, entry in enumerate(entries):
            last = i == len(entries) - 1
            elbow = "└── " if last else "├── "
            print(f"{prefix}{elbow}{entry.name}{'/' if entry.is_dir() else ''}")
            if entry.is_dir():
                walk(entry, prefix + ("    " if last else "│   "))

    print(f"{root.name}/")
    walk(root, "")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    ap.add_argument("--workflows", default="1,2,3", help="comma-separated, e.g. 1 or 1,3")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--rename-map", type=Path, default=None)
    ap.add_argument("--extras", type=Path, default=DEFAULT_EXTRAS)
    ap.add_argument("--print-tree", action="store_true")
    ap.add_argument("--keep", action="store_true", help="do not wipe --out first")
    args = ap.parse_args(argv)

    # The tree uses box-drawing characters; Windows consoles default to cp1252.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

    workflows = [int(w) for w in args.workflows.split(",") if w.strip()]
    project_dir = args.out.resolve()
    if project_dir.exists() and not args.keep:
        shutil.rmtree(project_dir)
    project_dir.mkdir(parents=True, exist_ok=True)

    scratch_cfg = _scratch_config(
        args.config, project_dir, project_dir.parent / "_scaffold_config.yml"
    )
    _stage_cross_workflow_inputs(project_dir, scratch_cfg)

    raw: list[str] = []
    for w in workflows:
        found = _summary(SNAKEFILES[w], scratch_cfg)
        logs = _log_paths(SNAKEFILES[w])
        print(f"wf{w}: {len(found)} declared outputs, {len(logs)} logs", file=sys.stderr)
        raw.extend(found)
        raw.extend(logs)

    if args.extras.exists():
        overlay = yaml.safe_load(args.extras.read_text(encoding="utf-8")) or {}
        for w in workflows:
            raw.extend(overlay.get(f"wf{w}", []) or [])

    renames, drops, adds = _load_delta(args.rename_map)
    rels: set[str] = set()
    renamed: set[str] = set()
    for p in raw:
        rel = Path(p).as_posix()
        prefix = project_dir.as_posix() + "/"
        rel = rel[len(prefix):] if rel.startswith(prefix) else rel
        # Rename first, then drop: a drop names the POST-move location, so a
        # delta can move files into a transient area and then retire the area.
        rel = _apply_renames(rel, renames)
        renamed.add(rel)
        if any(rel == d or (d.endswith("/") and rel.startswith(d)) for d in drops):
            continue
        rels.add(rel)
    rels.update(adds)

    # Captured BEFORE drops: a merged log is rendered from the per-rule parts a
    # successful run consumed, so it must survive the drop that retires them.
    part_logs = [r for r in renamed if "/_parts/" in r and r.endswith(".log")]

    for rel in sorted(rels):
        target = project_dir / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            target.write_text(_placeholder(rel, project_dir, part_logs), encoding="utf-8")

    print(f"scaffolded {len(rels)} paths under {project_dir}", file=sys.stderr)
    if args.print_tree:
        _print_tree(project_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
