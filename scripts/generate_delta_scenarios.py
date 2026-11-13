"""Retired standalone delta producer; use the workflow runner (delta-change/3)."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main() -> None:
    """Retired: delta generation runs through the workflow runner."""
    argparse.ArgumentParser(description=__doc__).parse_args()
    raise SystemExit(
        "scripts/generate_delta_scenarios.py is retired: delta-change/1 and /2 are "
        "read-only. Configure generate_scenarios with method: delta_change and "
        "delta-change/3, then run scripts/run_workflows.py. See "
        "docs/technical-manual/delta-change-scenarios.md"
    )


if __name__ == "__main__":
    main()
