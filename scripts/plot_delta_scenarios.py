"""Plot retained delta factors with individual members and spatial diagnostics."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from blueearth_cst.experiment.delta_plots import plot_delta_collection


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    for path in plot_delta_collection(args.manifest, args.output_dir):
        print(path)


if __name__ == "__main__":
    main()
