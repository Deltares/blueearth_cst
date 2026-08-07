"""CLI tee wrapper for Snakemake ``shell:`` rules.

Runs a command, streams its combined stdout+stderr to the console AND a log
file, and exits with the command's own return code -- unlike ``<cmd> | tee
{log}``, which masks a non-zero exit on Windows/cmd.exe (no ``pipefail``).

Usage (from a ``shell:`` rule)::

    python -u {run_logged} {log} -- <command> [args...]

The word after ``--`` onward is the command, passed through verbatim (no shell
re-tokenization). Rationale and the masking bug it fixes: see
``blueearth_cst/shared/snake_utils.run_and_tee`` (t260721a; dev/tasks/).
"""
import os
import sys

# Make ``blueearth_cst.shared.snake_utils`` importable when run as a standalone
# script from the repo root (the Snakefile's sys.path insert does not apply to
# shell children). This file sits two levels under the repo root
# (blueearth_cst/shared/run_logged.py), so three ``dirname`` levels reach the
# repo root — the parent of the ``blueearth_cst`` package.
sys.path.insert(
    0,
    os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ),
)
from blueearth_cst.shared.snake_utils import run_and_tee  # noqa: E402

_USAGE = "run_logged.py: usage: python run_logged.py LOG -- CMD [ARGS...]\n"


def main(argv):
    if "--" not in argv:
        sys.stderr.write(_USAGE)
        return 2
    sep = argv.index("--")
    before, command = argv[:sep], argv[sep + 1:]
    if len(before) != 1 or not command:
        sys.stderr.write(_USAGE)
        return 2
    return run_and_tee(command, before[0])


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
