"""Run one of the five workflows through its owned runner."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from blueearth_cst.shared.workflow_archive_launch import (  # noqa: E402
    WORKFLOWS,
    run_workflow,
)


def main(argv: list[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if "--" in arguments:
        boundary = arguments.index("--")
        extra = arguments[boundary + 1 :]
        arguments = arguments[:boundary]
    else:
        extra = []
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "workflow", choices=(*WORKFLOWS, "generate_scenarios", "simulate_system")
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--project-dir", required=True, type=Path)
    parser.add_argument("--cores", type=int, default=3)
    parser.add_argument("--target", action="append", default=[])
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--keep-going", action="store_true")
    args = parser.parse_args(arguments)
    if args.workflow in {"generate_scenarios", "simulate_system"}:
        if (
            args.workflow == "generate_scenarios"
            and args.target
            and args.target != ["all"]
        ):
            parser.error("generate_scenarios supports only --target all")
        owned_args = [
            "--config",
            str(args.config),
            "--project-dir",
            str(args.project_dir),
            "--cores",
            str(args.cores),
        ]
        if args.workflow == "simulate_system":
            from scripts.simulate_system import main as owned_main

            owned_args += ["--target", *(args.target or ["all"])]
        else:
            from scripts.generate_scenarios import main as owned_main

        forwarded = []
        if args.dry_run:
            if args.workflow == "simulate_system":
                owned_args.append("--dry-run")
            else:
                forwarded.append("--dry-run")
        if args.keep_going:
            forwarded.append("--keep-going")
        if forwarded or extra:
            owned_args += ["--", *forwarded, *extra]
        return owned_main(owned_args)
    return run_workflow(
        args.workflow,
        args.config,
        args.project_dir,
        cores=args.cores,
        targets=args.target or ["all"],
        dry_run=args.dry_run,
        keep_going=args.keep_going,
        extra=extra,
    )


if __name__ == "__main__":
    raise SystemExit(main())
