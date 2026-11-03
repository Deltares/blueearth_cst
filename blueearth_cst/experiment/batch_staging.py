"""Keep a failed Wflow batch's finished members, so a retry re-runs only the failures.

Rule 4.05 runs several members in one Julia session. Snakemake treats the batch
as one job, so when one member fails it deletes every declared output of that
job -- the finished siblings included -- and a retry reruns all of them.

The driver (``run_wflow_batch.jl``) therefore MOVES each member's CSV out of
the declared path into this staging directory the moment the member succeeds,
then turns its ``.expect`` marker into ``.ok``. Moving it at once, rather than
in a Python post-pass, is what keeps it safe when the process dies outright:
Snakemake's cleanup only knows the declared paths.

Staging is keyed by RUN ID, not batch index, because batch membership is not
stable across attempts: ``resolve_batch_size`` depends on ``--cores``, the
Julia thread count and disk headroom. A staged member is reused only when its
marker records the same TOML digest and frozen-simulation digest the retry
would run with; anything else is discarded and re-run.

Not covered: the failing member's own partial output is still cleaned, so that
member always re-runs -- which is the point.
"""

from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path

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
        if csv.is_file() and ok.is_file():
            try:
                if json.loads(ok.read_text(encoding="utf-8")) == wanted:
                    continue
            except ValueError:
                pass
        for path in (csv, expect, ok):
            path.unlink(missing_ok=True)
        expect.write_text(json.dumps(wanted, sort_keys=True), encoding="utf-8")
        pending.append(member)
    return pending


def restore_staged(members, staging):
    """Move every member's staged CSV to its declared path; drop the markers.

    Called only after the driver exited 0 (or when nothing needed to run), so
    every member must have a staged CSV; a missing one is an error rather than a
    silently short batch.
    """
    missing = []
    for member in members:
        csv, expect, ok = _paths(staging, member.run_id)
        if not (csv.is_file() and ok.is_file()):
            missing.append(member.run_id)
            continue
        member.native_output_path.parent.mkdir(parents=True, exist_ok=True)
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
