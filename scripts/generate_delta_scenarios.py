"""Publish monthly delta-change forcing for the existing WF4 runner."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from blueearth_cst.experiment.delta_scenarios import generate_collection


def main() -> None:
    """Generate an explicit delta collection and print its WF4 manifest path."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--project-dir", type=Path, required=True)
    args = parser.parse_args()
    print(generate_collection(args.config, args.project_dir))


if __name__ == "__main__":
    main()
