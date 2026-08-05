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
  bodies is a known property of this repo (``dev/milestones/r07/project-layout-design.md``).
- WF2/WF3 declare wf1 leaves as `ancient(...)` cross-workflow inputs that
  Snakemake will not satisfy on its own. They are staged into the scratch tree
  first, exactly as ``tests/test_cli.py``'s ``config_with_staged_region`` does.

Usage
-----
    pixi run python dev/scripts/scaffold_project_tree.py --print-tree
    pixi run python dev/scripts/scaffold_project_tree.py \
        --rename-map .tmp/dev-scratch/proposed_layout.yml --print-tree

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
DEFAULT_OUT = REPO_ROOT / ".tmp/scaffold"

SNAKEFILES = {
    1: "Snakefile_model_creation",
    2: "Snakefile_climate_projections",
    3: "Snakefile_climate_experiment",
}

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


# Enough of a wflow TOML for rule 3.01c to read `input.path_static`; same role
# as the test fixture's.
_MINIMAL_WFLOW_TOML = '[input]\npath_static = "staticmaps.nc"\n'

#: The wf1 leaves WF2/WF3 declare as `ancient()` inputs and Snakemake will not
#: satisfy on its own. Keep in step with `tests/test_cli.py`'s
#: `config_with_staged_region`, which stages the same contract for the dry-run
#: tests; the two lists are maintained by hand and have drifted apart before
#: (the R9 P5 sweep found this one two milestones behind).
_STAGED_LEAVES = (
    # WF3 rule 3.00b `check_project_consistency` (P3-1 commit 1).
    "config/runs/snake_config_model_creation.yml",
    # WF3 rule 3.01c `write_model_reference` (R9 P4) -- the FIRST WF3 rule to
    # declare model files as inputs. Before P4 the model was reached only
    # through `params`, so the DAG could not see the dependency at all.
    "models/hydrology/wflow/wflow_sbm.toml",
    "models/hydrology/wflow/.outputs_configured",
)


def _stage_cross_workflow_inputs(project_dir: Path, config_path: Path) -> None:
    """Stage the wf1 leaves WF2/WF3 declare as ancient() inputs.

    NO REGION is staged, and that is the current contract rather than an
    omission. Until ADR 0003 both downstream workflows consumed wf1's
    `staticgeoms/region.geojson`; the extent is now model-free, so WF2 and WF3
    each DELINEATE their own `data/spatial/geoms/region.geojson` (rule
    `delineate_region`) and declare it as an OUTPUT. Staging one would be
    staging a file the workflow is about to produce.
    """
    snapshot_rel = "config/runs/snake_config_model_creation.yml"
    for rel in _STAGED_LEAVES:
        target = project_dir / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        if rel == snapshot_rel:
            # Serialized from the config the summary itself consumes, so the
            # drift guard's comparands match by construction.
            target.write_text(config_path.read_text(encoding="utf-8"), encoding="utf-8")
        elif rel.endswith("wflow_sbm.toml"):
            target.write_text(_MINIMAL_WFLOW_TOML, encoding="utf-8")
        else:
            target.write_text("", encoding="utf-8")


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


def _log_paths(snakefile: str, experiment: str) -> list[str]:
    """Read each rule's `log:` path straight out of the Snakefile.

    `--summary`'s log column reports only logs that already EXIST on disk (it
    prints `-` otherwise), so it is useless against a fresh project_dir. Reading
    the declarations instead keeps the scaffold honest about BOTH the numbering
    and the root: WF3 writes under ``{exp_dir}``, not the project root, so
    synthesizing ``logs/<W.NN>_<rule>.log`` would have misplaced all 13 of them.
    """
    text = (REPO_ROOT / snakefile).read_text(encoding="utf-8")
    roots = {"project_dir": "", "exp_dir": f"experiments/{experiment}/"}

    # ONE LEVEL OF INDIRECTION, resolved from the same file. Rules no longer
    # interpolate `project_dir` directly -- they interpolate `LOG_PARTS_DIR`,
    # itself assigned f"{project_dir}/logs/_parts" (WF1/WF2) or
    # f"{exp_dir}/logs/_parts" (WF3). Matching only the two ROOT names therefore
    # matched nothing and every workflow silently scaffolded ZERO logs: the same
    # failure mode as the stale staging above -- a helper left behind by a
    # refactor, reporting success while producing a wrong tree.
    for name, base, tail in re.findall(
        r"""^(\w+)\s*=\s*f["']\{(\w+)\}/([^"']+)["']""", text, re.M
    ):
        # `not in roots` matters: `exp_dir` is itself assigned from project_dir
        # as f"{project_dir}/experiments/{experiment_name}", and resolving that
        # would replace the seeded value with one still carrying a wildcard --
        # which the substitution below then renders as `experiments/1/`.
        if base in roots and name not in roots:
            roots[name] = roots[base] + tail.rstrip("/") + "/"
    # Bare `.log` string constants (WORKFLOW_LOG_NAME = "wf1_model_creation.log")
    # are interpolated into the merged-log path, which would otherwise reduce to
    # `logs/1` under the wildcard substitution below and be dropped.
    consts = dict(
        re.findall(r"""^(\w+)\s*=\s*["']([^"'{}]+\.log)["']""", text, re.M)
    )

    logs = []
    # Three declaration forms are in use across the Snakefiles and all three
    # must be read, or a workflow silently loses logs from the scaffold:
    #   f"{LOG_PARTS_DIR}/2.05_….log"                          (plain f-string)
    #   f"{exp_dir}/logs/3.05_…/" + "rlz_{rlz_num}_….log"      (f-string + concat)
    #   project_dir + "/logs/_parts/2.02_…/{model}.log"        (bare concat)
    #
    # The tail is NOT anchored to `logs/` -- with LOG_PARTS_DIR resolved above,
    # the `logs/` segment lives in the ROOT, not the tail. The `.log` suffix
    # test below is what keeps non-log f-strings out.
    patterns = (
        r"""f["']\{(\w+)\}/([^"']*?)["'](?:\s*\+\s*["']([^"']+)["'])?""",
        r"""\b(\w+)\s*\+\s*["']/([^"']+)["']()""",
    )
    for pattern in patterns:
        for var, tail, extra in re.findall(pattern, text):
            if var not in roots:
                continue
            rel = roots[var] + tail + extra
            for const, value in consts.items():
                rel = rel.replace("{" + const + "}", value)
            if not rel.endswith(".log"):
                continue
            # Wildcards ({rlz_num}, {_b}) stand in for a fan-out; show one instance.
            logs.append(re.sub(r"\{[^}]*\}", "1", rel))
    return sorted(set(logs))


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


_RULE_SEP = "# " + "=" * 72


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
        " | <hh:mm:ss> -> <hh:mm:ss> | total <h:mm:ss>",
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
            f"# {number}  {name:<32} <hh:mm:ss> -> <hh:mm:ss>   <s>  ok",
            _RULE_SEP,
            "<scaffold placeholder - this rule's captured output>",
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

    experiment = (
        yaml.safe_load(args.config.read_text(encoding="utf-8"))
        .get("workflows", {})
        .get("climate_experiment", {})
        .get("experiment_name", "experiment")
    )

    scratch_cfg = _scratch_config(
        args.config, project_dir, project_dir.parent / "_scaffold_config.yml"
    )
    _stage_cross_workflow_inputs(project_dir, scratch_cfg)

    raw: list[str] = []
    for w in workflows:
        found = _summary(SNAKEFILES[w], scratch_cfg)
        logs = _log_paths(SNAKEFILES[w], experiment)
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
