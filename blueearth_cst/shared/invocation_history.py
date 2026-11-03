"""Atomic, common history for supported workflow launcher attempts."""

from __future__ import annotations

import json
import os
import re
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PARENT_ENV = "CST_PARENT_INVOCATION_ID"
INVOCATION_ENV = "CST_INVOCATION_ID"
_STARTED: dict[Path, tuple[int, float]] = {}
_SENSITIVE = re.compile(
    r"api[-_]?key|auth|credential|pass(?:word|wd)?|private[-_]?key|secret|token",
    re.IGNORECASE,
)


def sanitize_command(command: list[str]) -> list[str]:
    """Remove credential-like argument values from durable history."""
    result = []
    redact_next = False
    for token in command:
        if redact_next:
            result.append("<redacted>")
            redact_next = False
            continue
        key, separator, _ = token.partition("=")
        if _SENSITIVE.search(key.lstrip("-")):
            if separator:
                result.append(f"{key}=<redacted>")
            elif token.startswith("-"):
                result.append(token)
                redact_next = True
            else:
                result.append("<redacted>")
            continue
        result.append(token)
    return result


def _now() -> str:
    return (
        datetime.now(timezone.utc)
        .isoformat(timespec="milliseconds")
        .replace("+00:00", "Z")
    )


def _write(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding="utf-8", newline="\n") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        for attempt in range(10):
            try:
                os.replace(temporary, path)
                break
            except PermissionError as error:
                # Windows readers and indexers can momentarily hold the old
                # file without delete sharing during a record refresh.
                if os.name != "nt" or error.winerror != 5 or attempt == 9:
                    raise
                time.sleep(0.02)
    finally:
        temporary.unlink(missing_ok=True)


def start(
    project_root: Path,
    *,
    workflow: str | None,
    entry_point: str,
    command: list[str],
    targets: list[str],
    mode: str,
    contract_mode: str,
    invocation_id: str | None = None,
    parent_invocation_id: str | None = None,
    requested_workflows: list[str] | None = None,
) -> tuple[Path, dict[str, Any]]:
    """Publish a running record before configuration validation or launch."""
    identity = invocation_id or uuid.uuid4().hex
    path = project_root.resolve() / "config/_engine/invocations" / f"{identity}.json"
    if path.exists():
        raise ValueError(f"invocation already exists: {identity}")
    _STARTED[path] = (os.getpid(), time.monotonic())
    record: dict[str, Any] = {
        "schema_version": "invocation/1",
        "invocation_id": identity,
        "parent_invocation_id": parent_invocation_id,
        "workflow": workflow,
        "entry_point": entry_point,
        "command": sanitize_command(command),
        "targets": targets,
        "working_directory": str(Path.cwd()),
        "mode": mode,
        "contract_mode": contract_mode,
        "work_performed": "no" if mode == "dry_run" else "unknown",
        "status": "running",
        "exit_code": None,
        "started_at_utc": _now(),
        "ended_at_utc": None,
        "elapsed_seconds": None,
        "configuration": {
            "source_config_sha256": None,
            "effective_config_sha256": None,
            "configuration_inputs_sha256": None,
            "run_record": None,
            "archive_state": "unavailable",
        },
        "plan": None,
        "artifacts": [],
        "children": [],
        "requested_workflows": requested_workflows or [],
        "launch_failures": [],
        "error": None,
    }
    from blueearth_cst.shared.execution_outputs import bind_directory

    bind_directory(path, record)
    _write(path, record)
    if workflow == "simulate_system" and mode == "execute":
        from blueearth_cst.shared import resource_tracking

        record["resources"] = resource_tracking.start(path)
        _write(path, record)
    _publish_summary(path, record)
    return path, record


def update(path: Path, record: dict[str, Any]) -> None:
    """Atomically refresh a running record, including child linkage."""
    _write(path, record)
    _publish_summary(path, record)


def finish(
    path: Path,
    record: dict[str, Any],
    *,
    exit_code: int | None,
    error: BaseException | None = None,
    elapsed_seconds: float | None = None,
) -> None:
    """Finish only when a launcher observes an outcome, not on hard kill."""
    if record["status"] != "running":
        return
    if record.get("resources") is not None:
        from blueearth_cst.shared import resource_tracking

        summary = resource_tracking.finish(path)
        if summary is not None:
            record["resources"] = summary
    record["status"] = "succeeded" if error is None and exit_code == 0 else "failed"
    record["exit_code"] = exit_code
    record["ended_at_utc"] = _now()
    timer = _STARTED.pop(path, None)
    if elapsed_seconds is None and timer is not None and timer[0] == os.getpid():
        elapsed_seconds = max(0.0, time.monotonic() - timer[1])
    record["elapsed_seconds"] = elapsed_seconds
    if error is not None:
        record["error"] = {
            "phase": "launch",
            "type": type(error).__name__,
            "message": str(error),
        }
    _write(path, record)
    try:
        from blueearth_cst.shared.execution_outputs import finalize

        finalize(path, record)
        _write(path, record)
    except (OSError, ValueError, KeyError) as exc:
        print(f"(execution diagnostics unavailable: {exc})", file=sys.stderr)
    report = _publish_summary(path, record)
    if report is not None and record.get("workflow") is not None:
        try:
            print(f"Run summary: {report}", flush=True)
        except Exception:  # noqa: BLE001 -- preserve the observed execution outcome
            return


def _publish_summary(path: Path, record: dict[str, Any]) -> Path | None:
    """Publish one parent/standalone view without changing execution outcomes."""
    if record.get("parent_invocation_id") is not None:
        return
    try:
        from blueearth_cst.shared.invocation_summary import write_summary

        return write_summary(path)
    except Exception as exc:  # noqa: BLE001 -- a derived report cannot fail a run
        try:
            print(f"(invocation summary unavailable: {exc})", file=sys.stderr)
        except Exception:  # noqa: BLE001 -- diagnostic streams can fail too
            # Even a closed diagnostic stream must preserve the run outcome.
            return
