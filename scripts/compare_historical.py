"""Compare explicitly selected retained historical hydrology runs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main(argv: list[str] | None = None) -> int:
    """Prepare tables and render them; errors leave existing comparisons intact."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="examples:\n  pixi run python scripts/compare_historical.py --config comparison.yml",
    )
    parser.add_argument(
        "--config",
        required=True,
        type=Path,
        help="comparison YAML; paths resolve beside it",
    )
    parser.add_argument(
        "--annotation",
        choices=("none", "caption", "title"),
        default="none",
        help="optional text embedded in the exported PNGs",
    )
    args = parser.parse_args(argv)
    from blueearth_cst.model.historical_comparison import compare
    from blueearth_cst.model.plot_historical_comparison import render

    try:
        output = compare(args.config)
        render(output, annotation=args.annotation)
    except (ValueError, OSError, KeyError) as error:
        parser.exit(
            1,
            f"Comparison failed: {error}\nCheck the comparison configuration and retained inputs.\n",
        )
    print(f"Historical comparison written to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
