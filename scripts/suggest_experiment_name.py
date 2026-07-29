"""Write a suggested ``experiment_name`` into a config, only if absent.

R07 B8. Run once, deliberately, before the first climate-experiment run::

    python scripts/suggest_experiment_name.py config/workflows/snake_config_model_test.yml

Reads ``project.project_dir``, slugifies its basename, appends today's date,
validates the result through the same grammar the workflow enforces, and writes
it to ``workflows.climate_experiment.experiment_name``.

**An existing value is never overwritten** — the command exits nonzero naming
the value already present. The experiment name is the directory every wf3
artifact hangs off, so silently changing it would strand a completed
experiment's outputs under a name nothing points at any more.

The name is NEVER generated at run time. A runtime timestamp would make every
invocation target a fresh ``experiments/<id>/``: nothing would ever be up to
date, incremental reruns would be impossible, ``--dry-run`` would mislead, and
the baseline gate would have no fixed path to check.
"""
from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from blueearth_cst.shared.snake_utils import suggest_experiment_name  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("config", help="path to the orchestration config YAML")
    ap.add_argument(
        "--date", default=None, metavar="YYYYMMDD",
        help="date stamp to append (default: today). Explicit values keep the "
             "command reproducible in tests and scripted setups",
    )
    ap.add_argument(
        "--dry-run", action="store_true",
        help="print the suggestion and leave the config untouched",
    )
    args = ap.parse_args(argv)

    cfg_path = Path(args.config)
    if not cfg_path.is_file():
        print(f"error: no such config: {cfg_path}", file=sys.stderr)
        return 2
    doc = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}

    try:
        project_dir = doc["project"]["project_dir"]
    except (KeyError, TypeError):
        print("error: config has no project.project_dir", file=sys.stderr)
        return 2

    stamp = args.date or date.today().strftime("%Y%m%d")
    try:
        name = suggest_experiment_name(project_dir, stamp)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    # --dry-run reports the suggestion even when a value is already set: the
    # point of inspecting first is to SEE what would be proposed. It still
    # writes nothing.
    if args.dry_run:
        print(name)
        return 0

    existing = (
        (doc.get("workflows") or {}).get("climate_experiment") or {}
    ).get("experiment_name")
    if existing is not None:
        print(
            f"error: experiment_name is already set to {existing!r}; refusing "
            f"to overwrite (would have suggested {name!r}). Remove the key "
            f"first if you really want a new experiment directory.",
            file=sys.stderr,
        )
        return 1

    doc.setdefault("workflows", {}).setdefault("climate_experiment", {})[
        "experiment_name"
    ] = name
    cfg_path.write_text(yaml.safe_dump(doc, sort_keys=False), encoding="utf-8")
    print(f"wrote workflows.climate_experiment.experiment_name: {name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
