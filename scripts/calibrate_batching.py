"""Measure WF4 batching on YOUR basin and machine, and print the settings to use.

WF4 runs its members in batches: one Julia session per batch, with some
number of batches running at once. The fastest threads-per-batch and
batches-at-once depend on the basin's size and on the machine, and
`config/advanced_settings.yml` only has a MEASURED default for small basins
(t2609242342). This script measures your case.

It times copies of the WF1 historical model -- the one Wflow model that stays
on disk after a run -- through the same batch driver WF4 uses:

1. one batch of ``--members`` at 1, 2 and 4 threads, one after another;
2. ``--members x width`` members split into ``width`` single-thread-or-best
   batches running at once, for each ``--widths`` value.

Each copy writes into ``<project>/_engine/calibration/`` (its own
``dir_output``), so the WF1 model and its outputs are never touched. Nothing is
written to your config: the script prints the ``compute:`` keys to set in the
simulate_system settings file, and the full timings.

Usage::

    python scripts/calibrate_batching.py --project-dir <project> [--years 5]

The choice is made on a PREDICTION, not on the probe's wall times: from each
configuration it takes the cold (first member) and warm (later members) times
and predicts a WF4 run of ``--runs`` members, ``cold + (per-batch - 1) x warm``
per batch. A short probe is dominated by cold starts, and ranking settings on
its raw wall time mostly ranks noise; a warm-up batch is run and discarded
first for the same reason. ``--years`` shortens the simulated period to keep
the probe quick, at the cost of warm times the prediction must scale up; the
script warns when warm is too small next to cold to decide on.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY))
DRIVER = REPOSITORY / "blueearth_cst/experiment/run_wflow_batch.jl"
_ROW = re.compile(r"\[(\d+)/(\d+)\] (\S+)\s+(\S+)")


def _seconds(text: str) -> float:
    parts = [float(part) for part in text.split(":")]
    return sum(value * 60**index for index, value in enumerate(reversed(parts)))


def _copy_model_toml(model_toml: Path, out_dir: Path, years: int | None) -> Path:
    """A copy beside the original whose outputs all land in ``out_dir``."""
    text = model_toml.read_text(encoding="utf-8")
    text, count = re.subn(
        r"(?m)^dir_output\s*=.*$", f'dir_output = "{out_dir.as_posix()}"', text
    )
    if count != 1:
        raise ValueError(f"{model_toml}: expected one top-level dir_output")
    if years:
        start = re.search(r'(?m)^starttime\s*=\s*"([^"]+)"', text).group(1)
        begin = datetime.fromisoformat(start)
        end = begin.replace(year=begin.year + years)
        text = re.sub(
            r"(?m)^endtime\s*=.*$", f'endtime = "{end.isoformat()}"', text, count=1
        )
    copy = model_toml.with_name(
        f"calibration_{out_dir.parent.name}_{out_dir.name}.toml"
    )
    copy.write_text(text, encoding="utf-8")
    return copy


def _launch(tag, count, threads, model_toml, root, years, julia):
    records = [tag]
    for index in range(1, count + 1):
        run = f"run_{index:02d}"
        out = root / tag / run
        out.mkdir(parents=True, exist_ok=True)
        toml = _copy_model_toml(model_toml, out, years)
        records += [run, str(toml), str(out / "output.csv")]
    log = (root / f"{tag}.log").open("w", encoding="utf-8")
    process = subprocess.Popen(
        [*julia, f"--threads={threads}", str(DRIVER), *records],
        cwd=REPOSITORY,
        stdout=log,
        stderr=subprocess.STDOUT,
    )
    return process, log


def _members(root: Path, tag: str) -> list[float]:
    text = (root / f"{tag}.log").read_text(encoding="utf-8", errors="replace")
    return [
        _seconds(match.group(4))
        for line in text.splitlines()
        if "FAILED" not in line and (match := _ROW.search(line))
    ]


def _cold_warm(members):
    """Mean cold (first) and warm (later) member time over a configuration's batches."""
    colds = [batch[0] for batch in members if batch]
    warms = [time for batch in members for time in batch[1:]]
    cold = sum(colds) / len(colds)
    warm = sum(warms) / len(warms) if warms else 0.0
    return cold, warm


def _predict(cold, warm, runs, width):
    """Wall time of ``runs`` members in ``width`` even batches running at once."""
    per_batch = -(-runs // width)
    return cold + (per_batch - 1) * warm


def _run(configs, model_toml, root, years, julia):
    """Run each configuration's batches at once; return wall and member times."""
    results = {}
    for name, batches in configs:
        start = time.monotonic()
        launched = [
            _launch(f"{name}_{index}", count, threads, model_toml, root, years, julia)
            for index, (count, threads) in enumerate(batches)
        ]
        codes = [process.wait() for process, _ in launched]
        for _, log in launched:
            log.close()
        if any(codes):
            raise RuntimeError(f"{name}: a batch failed; see {root}/{name}_*.log")
        members = [_members(root, f"{name}_{i}") for i in range(len(batches))]
        results[name] = {
            "wall_s": round(time.monotonic() - start, 1),
            "members_s": members,
        }
        print(f"  {name:<14} {results[name]['wall_s']:>7.1f} s", flush=True)
    return results


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--project-dir", required=True, type=Path)
    parser.add_argument("--members", type=int, default=4)
    parser.add_argument(
        "--runs", type=int, default=14, help="WF4 member count to predict for"
    )
    parser.add_argument("--widths", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--years", type=int, default=None)
    parser.add_argument(
        "--keep", action="store_true", help="keep the calibration outputs"
    )
    args = parser.parse_args(argv)

    from blueearth_cst.experiment.batch_sizing import count_active_cells
    from blueearth_cst.shared.snake_utils import JULIA_VERSION

    model_dir = args.project_dir.resolve() / "models/hydrology/wflow"
    model_toml = model_dir / "wflow_sbm.toml"
    if not model_toml.is_file():
        parser.error(f"no WF1 model at {model_dir}; run build_model first")
    root = args.project_dir.resolve() / "_engine/calibration"
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)
    julia = ["julia", f"+{JULIA_VERSION}", "--project=."]
    cells = count_active_cells(model_dir / "staticmaps.nc")
    print(f"calibrating on {cells} active cells, {args.members} members per batch")

    try:
        print("warm-up batch (discarded)")
        _run([("warmup", [(1, 1)])], model_toml, root, args.years, julia)
        print("threads per batch (one batch):")
        threads = _run(
            [(f"threads{t}", [(args.members, t)]) for t in (1, 2, 4)],
            model_toml,
            root,
            args.years,
            julia,
        )
        predicted = {}
        for t in (1, 2, 4):
            cold, warm = _cold_warm(threads[f"threads{t}"]["members_s"])
            predicted[t] = _predict(cold, warm, args.runs, 1)
            threads[f"threads{t}"].update(cold_s=round(cold, 1), warm_s=round(warm, 1))
        best_threads = min(predicted, key=predicted.get)
        print(f"batches at once ({best_threads} thread(s) each, same total members):")
        total = args.members * max(args.widths)
        widths = _run(
            [(f"width{w}", [(total // w, best_threads)] * w) for w in args.widths],
            model_toml,
            root,
            args.years,
            julia,
        )
        for w in args.widths:
            cold, warm = _cold_warm(widths[f"width{w}"]["members_s"])
            widths[f"width{w}"].update(
                cold_s=round(cold, 1),
                warm_s=round(warm, 1),
                predicted_s=round(_predict(cold, warm, args.runs, w), 1),
            )
        best_width = min(args.widths, key=lambda w: widths[f"width{w}"]["predicted_s"])
    finally:
        for copy in model_dir.glob("calibration_*.toml"):
            copy.unlink()
        if not args.keep and root.exists():
            shutil.rmtree(root, ignore_errors=True)

    print()
    print(f"predicted WF4 time for {args.runs} members:")
    for t in (1, 2, 4):
        print(f"  {t} thread(s), one batch     {predicted[t]:>7.1f} s")
    for w in args.widths:
        label = f"{w} at once, {best_threads} thread(s)"
        print(f"  {label:<26} {widths[f'width{w}']['predicted_s']:>7.1f} s")
    cold, warm = _cold_warm(threads[f"threads{best_threads}"]["members_s"])
    if warm and cold / warm > 10:
        print(
            f"  warning: a warm member ({warm:.0f} s) is under a tenth of a cold "
            f"start ({cold:.0f} s); the choice rests on little signal -- rerun "
            "with more --years or --members"
        )
    print()
    print("set in the simulate_system settings file:")
    print("  compute:")
    print(f"    julia_threads: {best_threads}")
    print(f"    max_parallel_batches: {best_width}")
    print()
    print(json.dumps({"active_cells": cells, "threads": threads, "widths": widths}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
