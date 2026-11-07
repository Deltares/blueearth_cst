"""Keep a failed Wflow batch's finished members, so a retry re-runs only the failures.

Rule 4.05 runs several members in one Julia session. Snakemake treats the batch
as one job, so when one member fails it deletes every declared output of that
job -- the finished siblings included -- and a retry reruns all of them.

The driver (``run_wflow_batch.jl``) therefore MOVES each member's CSV and optional
grid out of the declared path into staging when the member succeeds, then turns
its ``.expect`` marker into ``.ok`` after both files and their hashes exist.
Moving them at once, rather than in a Python post-pass, keeps them safe on death:
Snakemake's cleanup only knows the declared paths.

Staging is keyed by RUN ID, not batch index, because batch membership is not
stable across attempts: ``resolve_batch_size`` depends on ``--cores``, the
Julia thread count and disk headroom. A staged member is reused only when its
marker records the same TOML digest and frozen-simulation digest the retry
would run with. Grid pairs also require matching hashes and native-file validation;
anything else is discarded and re-run.

Not covered: the failing member's own partial output is still cleaned, so that
member always re-runs -- which is the point.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tomllib
from dataclasses import dataclass
from pathlib import Path

from blueearth_cst.shared.grid_output import configured_grid_path, validate_grid
from blueearth_cst.shared.provenance import file_sha256

#: The environment variable that tells the Julia driver where to stage.
STAGING_ENV = "CST_BATCH_STAGING"


@dataclass(frozen=True)
class BatchMember:
    run_id: str
    toml_path: Path
    native_output_path: Path


def members_from_records(records):
    """Rule 4.05's ``params.records`` -- ``[batch, (run, toml, csv)...]`` -- as members."""
    fields = list(records)[1:]
    return [
        BatchMember(fields[i], Path(fields[i + 1]), Path(fields[i + 2]))
        for i in range(0, len(fields), 3)
    ]


def _paths(staging, run_id):
    root = Path(staging)
    return (
        root / f"{run_id}.csv",
        root / f"{run_id}.expect",
        root / f"{run_id}.ok",
    )


def _fingerprint(member, simulation_sha256):
    return {
        "run_id": member.run_id,
        "toml_sha256": file_sha256(member.toml_path),
        "simulation_sha256": simulation_sha256,
    }


def prepare_staging(members, staging, simulation_sha256):
    """Return the members that must run; write an ``.expect`` marker for each.

    A member is reusable when its staged CSV and ``.ok`` marker both exist and
    the marker matches this attempt's fingerprint. Any other staged state for
    the member -- a stale fingerprint, a CSV without a marker (the process died
    between the move and the rename) -- is deleted, and the member runs.
    """
    Path(staging).mkdir(parents=True, exist_ok=True)
    pending = []
    for member in members:
        csv, expect, ok = _paths(staging, member.run_id)
        wanted = _fingerprint(member, simulation_sha256)
        grid_output = configured_grid_path(member.toml_path)
        grid = csv.with_suffix(".nc")
        checksums = csv.with_suffix(".sha256")
        if csv.is_file() and ok.is_file():
            try:
                if json.loads(ok.read_text(encoding="utf-8")) == wanted:
                    if grid_output is None:
                        continue
                    recorded = tomllib.loads(checksums.read_text(encoding="utf-8"))
                    if recorded == {"csv": file_sha256(csv), "grid": file_sha256(grid)}:
                        validate_grid(grid, member.toml_path)
                        continue
            except (OSError, ValueError, KeyError):
                pass
        for path in (csv, expect, ok, grid, checksums):
            path.unlink(missing_ok=True)
        expect.write_text(json.dumps(wanted, sort_keys=True), encoding="utf-8")
        pending.append(member)
    return pending


def restore_staged(members, staging):
    """Restore verified outputs, retaining grid pairs for publication recovery.

    Called only after the driver exited 0 (or when nothing needed to run), so
    every member must have staged outputs. Scalar-only CSVs move back; enabled
    grid pairs are copied and kept until experiment publication succeeds.
    """
    missing = []
    for member in members:
        csv, expect, ok = _paths(staging, member.run_id)
        if not (csv.is_file() and ok.is_file()):
            missing.append(member.run_id)
            continue
        member.native_output_path.parent.mkdir(parents=True, exist_ok=True)
        grid_output = configured_grid_path(member.toml_path)
        if grid_output is not None:
            grid = csv.with_suffix(".nc")
            validate_grid(grid, member.toml_path)
            # Keep the verified staging pair until experiment publication. A
            # crash between copies or failed-job cleanup can then safely retry.
            shutil.copyfile(grid, grid_output)
            shutil.copyfile(csv, member.native_output_path)
        else:
            os.replace(csv, member.native_output_path)
            ok.unlink()
            expect.unlink(missing_ok=True)
    if missing:
        raise RuntimeError(
            f"batch finished but {len(missing)} member(s) have no staged output: "
            + ", ".join(missing)
        )
    try:
        Path(staging).rmdir()
    except OSError:
        pass  # other batches still hold staged members


def run_staged_batch(records, simulation_path, staging, launch):
    """Rule 4.05's body: run what is not yet staged, then restore the batch.

    ``records`` is the rule's ``params.records``; ``launch`` is the command
    prefix the records are appended to (the logged Julia driver in the rule, a
    stub in the contract test). Returns the members that ran.
    """
    from blueearth_cst.shared.snake_utils import log_row, plural

    members = members_from_records(records)
    pending = prepare_staging(members, staging, file_sha256(simulation_path))
    if len(pending) < len(members):
        log_row(
            f"Reusing {plural(len(members) - len(pending), 'verified staged run output')} "
            "from an earlier attempt",
            module="batching",
        )
    if pending:
        tail = [
            value
            for member in pending
            for value in (
                member.run_id,
                str(member.toml_path),
                str(member.native_output_path),
            )
        ]
        subprocess.run(
            [*launch, str(records[0]), *tail],
            check=True,
            env={**os.environ, STAGING_ENV: str(staging)},
        )
    restore_staged(members, staging)
    return pending


def cleanup_published_staging(staging: str | Path, run_ids: list[str]) -> None:
    """Drop recovery copies only after the complete experiment is published."""
    root = Path(staging)
    for run in run_ids:
        for suffix in ("csv", "nc", "sha256", "ok", "expect"):
            (root / f"{run}.{suffix}").unlink(missing_ok=True)
    if root.exists() and not any(root.iterdir()):
        root.rmdir()
