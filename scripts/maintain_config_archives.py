"""Inspect configuration archives, or migrate and clean completed launch state."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from blueearth_cst.shared.configuration_archive import (  # noqa: E402
    maintain_archives,
    maintenance_report,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-dir", type=Path, required=True)
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Retain snapshots and remove completed launch/recovery files.",
    )
    args = parser.parse_args(argv)
    report = (
        maintain_archives(args.project_dir)
        if args.apply
        else maintenance_report(args.project_dir)
    )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
