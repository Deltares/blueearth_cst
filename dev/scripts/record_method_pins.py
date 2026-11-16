"""Check or re-record the method-revision pins in blueearth_cst/experiment.

    python dev/scripts/record_method_pins.py --check
    python dev/scripts/record_method_pins.py --producer wf4-simulation --bump "reason"
    python dev/scripts/record_method_pins.py --producer wf4-simulation \
        --no-numerical-change "reason"

``--bump`` increments the revision, so every identity of that producer changes;
use it when produced values can change. ``--no-numerical-change`` keeps the
revision and only re-pins the code. See dev/reference/toolbox-run-identity.md.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from blueearth_cst.experiment.content_identity import content_sha256  # noqa: E402
from blueearth_cst.experiment.method_revisions import (  # noqa: E402
    PRODUCERS,
    closure_drift,
    drift_message,
    producer_closure,
    read_pins,
    record_pin,
)


def check() -> int:
    pins = read_pins()["producers"]
    failed = 0
    for producer in PRODUCERS:
        files = producer_closure(producer)
        if content_sha256(files) == pins[producer]["closure_sha256"]:
            print(f"{producer}: revision {pins[producer]['revision']}, pin current")
            continue
        failed += 1
        print(drift_message(producer, closure_drift(pins[producer], files)))
    return 1 if failed else 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true", help="report drift only")
    parser.add_argument("--producer", choices=PRODUCERS)
    kind = parser.add_mutually_exclusive_group()
    kind.add_argument("--bump", metavar="REASON")
    kind.add_argument("--no-numerical-change", metavar="REASON")
    args = parser.parse_args(argv)
    if args.check:
        return check()
    if args.producer is None or (
        args.bump is None and args.no_numerical_change is None
    ):
        parser.error("give --check, or --producer with --bump or --no-numerical-change")
    record = record_pin(
        args.producer,
        "bump" if args.bump is not None else "no-numerical-change",
        args.bump if args.bump is not None else args.no_numerical_change,
    )
    print(f"{args.producer}: revision {record['revision']} recorded")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
