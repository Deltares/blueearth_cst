"""Execution-owned diagnostic paths and plain console transcripts.

Invocation JSON remains authoritative. These paths contain disposable diagnostic
views; no scientific output or configuration archive is moved here.
"""

from __future__ import annotations

import contextlib
import json
import os
import re
import sys
import threading
import traceback
from contextvars import ContextVar
from dataclasses import dataclass
from functools import wraps
from pathlib import Path
from typing import Any

DIRECTORY_ENV = "CST_EXECUTION_DIRECTORY"
RECORD_ENV = "CST_EXECUTION_RECORD"
_ANSI = re.compile(r"\x1b(?:\[[0-?]*[ -/]*[@-~]|\][^\x07]*(?:\x07|\x1b\\))")
_ACTIVE: set[Path] = set()
_CAPTURE_STACK: ContextVar[Any] = ContextVar("execution_capture_stack", default=None)
_CURRENT_DIRECTORY: ContextVar[Path | None] = ContextVar(
    "execution_directory", default=None
)
WORKFLOWS = {
    "analyze_climate": "wf0",
    "build_model": "wf1",
    "analyze_projections": "wf2",
    "generate_scenarios": "wf3",
    "simulate_system": "wf4",
}


def _directory_name(record: dict[str, Any]) -> str:
    """Describe the requested scope, independently of the eventual outcome."""
    stamp = (
        record["started_at_utc"]
        .split(".")[0]
        .removesuffix("Z")
        .replace("T", "_")
        .replace(":", "-")
        + "Z"
    )
    selected = (
        [record["workflow"]]
        if record.get("workflow")
        else record.get("requested_workflows", [])
    )
    if set(selected) == set(WORKFLOWS):
        scope = "all-workflows"
    elif len(selected) == 1 and selected[0] in WORKFLOWS:
        scope = selected[0].replace("_", "-")
    else:
        scope = "-".join(WORKFLOWS[name] for name in WORKFLOWS if name in selected)
        scope = scope or "workflows"
    mode = "__dry-run" if record["mode"] == "dry_run" else ""
    return f"{stamp}__{scope}{mode}"


def bind_directory(path: Path, record: dict[str, Any]) -> None:
    """Allocate a root attempt directory or inherit its explicit parent's."""
    root = path.parents[3]
    parent = record.get("parent_invocation_id")
    if parent:
        parent_record = json.loads(
            path.with_name(f"{parent}.json").read_text(encoding="utf-8")
        )
        relative = parent_record.get("execution_directory")
        if relative is None:
            relative = f"logs/{parent}"
            (root / relative).mkdir(parents=True, exist_ok=True)
    else:
        relative = f"logs/{_directory_name(record)}"
        try:
            (root / relative).mkdir(parents=True, exist_ok=False)
        except FileExistsError:
            relative += f"__{record['invocation_id'][:12]}"
            (root / relative).mkdir(parents=True, exist_ok=False)
    directory = (root / relative).resolve()
    if not directory.is_relative_to((root / "logs").resolve()):
        raise ValueError("execution directory must be beneath project logs")
    record["execution_directory"] = relative
    stack = _CAPTURE_STACK.get()
    if stack is not None:
        stack.enter_context(console_capture((path, record)))


def capture_launch(function):
    """Close execution capture even when validation or launch raises."""

    @wraps(function)
    def wrapped(*args, **kwargs):
        with contextlib.ExitStack() as stack:
            token = _CAPTURE_STACK.set(stack)
            try:
                return function(*args, **kwargs)
            finally:
                _CAPTURE_STACK.reset(token)

    return wrapped


def current_directory() -> Path | None:
    """Current managed output directory, including in-process export steps."""
    value = os.environ.get(DIRECTORY_ENV)
    return Path(value) if value else _CURRENT_DIRECTORY.get()


def environment(path: Path, record: dict[str, Any]) -> dict[str, str]:
    """Environment additions for one workflow's children, never global state."""
    from blueearth_cst.shared.console_terminal import terminal_environment

    return {
        **terminal_environment(),
        DIRECTORY_ENV: str(path.parents[3] / record["execution_directory"]),
        RECORD_ENV: str(path),
        "PYTHONUNBUFFERED": "1",
        "PYTHONIOENCODING": "utf-8",
    }


@dataclass(frozen=True)
class WorkflowPaths:
    log: str
    benchmark: str
    log_parts: str
    benchmark_parts: str
    managed: bool = False

    @property
    def gather_targets(self) -> dict[str, str]:
        """Require gathered files only when no launcher will finalize diagnostics.

        Managed diagnostics belong to a fresh directory on every invocation.
        Requiring them in the DAG would schedule gather jobs on an unchanged run.
        """
        if self.managed:
            return {}
        return {"workflow_log": self.log, "benchmarks": self.benchmark}


def workflow_paths(
    project: str | Path,
    workflow: str,
    log_name: str | None,
    benchmark_name: str | None,
    log_parts: str,
    benchmark_parts: str,
) -> WorkflowPaths:
    """Use managed paths when launched; preserve unowned dry-run declarations."""
    directory = os.environ.get(DIRECTORY_ENV)
    if not directory:
        return WorkflowPaths(
            f"{project}/logs/{log_name}",
            f"{project}/benchmarks/{benchmark_name}",
            log_parts,
            benchmark_parts,
        )
    base = Path(directory).resolve()
    if not base.is_relative_to((Path(project).resolve() / "logs")):
        raise ValueError("execution directory belongs to another project")
    identity = Path(os.environ[RECORD_ENV]).stem
    parts = base / "_parts" / f"{workflow}-{identity}"
    return WorkflowPaths(
        (base / f"{workflow}.log").as_posix(),
        (base / f"{workflow}-benchmarks.md").as_posix(),
        (parts / "logs").as_posix(),
        (parts / "benchmarks").as_posix(),
        managed=True,
    )


def declare_warning_tally(project: str, log_name: str) -> None:
    """Keep managed warning counters in this execution's working directory."""
    from blueearth_cst.shared import snake_utils

    directory = os.environ.get(DIRECTORY_ENV)
    if not directory:
        snake_utils.declare_warning_tally(project, log_name)
        return
    path = Path(directory) / "_parts" / f".{Path(os.environ[RECORD_ENV]).stem}.warnings"
    path.parent.mkdir(parents=True, exist_ok=True)
    os.environ[snake_utils._WARNING_TALLY_ENV] = str(path)


class _Transcript:
    """Collapse redraws and strip ANSI only in the saved UTF-8 transcript."""

    def __init__(self, path: Path) -> None:
        self.handle = path.open("a", encoding="utf-8", newline="\n")
        self.lock = threading.Lock()
        self.pending = ""
        self.failed = False

    def write(self, text: str) -> None:
        if self.failed or self.handle.closed:
            return
        with self.lock:
            self.pending += text
            self.pending = self.pending.replace("\r\n", "\n")
            while "\n" in self.pending:
                line, self.pending = self.pending.split("\n", 1)
                self.handle.write(_ANSI.sub("", line).rsplit("\r", 1)[-1] + "\n")
            self.pending = "\r".join(self.pending.split("\r")[-2:])
            # A killed process retains ordinary rows; redraws remain pending.
            self.handle.flush()

    def close(self) -> None:
        with self.lock:
            if self.pending:
                self.handle.write(
                    _ANSI.sub("", self.pending).rstrip("\r").rsplit("\r", 1)[-1] + "\n"
                )
            self.handle.close()


class _ConsoleStream:
    def __init__(self, stream, transcript: _Transcript) -> None:
        self.stream = stream
        self.transcript = transcript

    def __getattr__(self, name):
        return getattr(self.stream, name)

    def write(self, text: str) -> int:
        result = self.stream.write(text)
        try:
            self.transcript.write(text)
        except OSError as error:
            self.transcript.failed = True
            self.stream.write(f"\n(console transcript write failed: {error})\n")
        return result

    def flush(self) -> None:
        self.stream.flush()


@contextlib.contextmanager
def console_capture(pair):
    """Mirror a root launcher; nested same-process workflow calls share it."""
    if pair is None or pair[1].get("parent_invocation_id"):
        yield
        return
    path, record = pair
    directory = path.parents[3] / record["execution_directory"]
    if directory in _ACTIVE:
        yield
        return
    try:
        transcript = _Transcript(directory / "console.log")
    except OSError as error:
        print(f"Console transcript unavailable: {error}", file=sys.stderr)
        yield
        return
    _ACTIVE.add(directory)
    token = _CURRENT_DIRECTORY.set(directory)
    stdout, stderr = sys.stdout, sys.stderr
    sys.stdout = _ConsoleStream(stdout, transcript)
    sys.stderr = _ConsoleStream(stderr, transcript)
    from blueearth_cst.shared.output_capture import _console_log_handlers

    handlers = _console_log_handlers(stdout, stderr)
    for handler, stream in handlers.items():
        handler.stream = sys.stdout if stream is stdout else sys.stderr
    try:
        yield
    except BaseException as error:
        # The caller may finish its record before an outer CLI prints the error.
        with contextlib.suppress(OSError):
            transcript.write("".join(traceback.format_exception(error)))
        raise
    finally:
        for handler, stream in handlers.items():
            handler.stream = stream
        sys.stdout, sys.stderr = stdout, stderr
        _ACTIVE.remove(directory)
        _CURRENT_DIRECTORY.reset(token)
        try:
            transcript.close()
        except OSError as error:
            print(f"Console transcript close failed: {error}", file=stderr)


def finalize(path: Path, record: dict[str, Any]) -> None:
    """Publish only this workflow's diagnostics, including partial failures."""
    workflow = WORKFLOWS.get(record.get("workflow"))
    if not workflow or record.get("mode") != "execute":
        return
    from blueearth_cst.shared.merge_benchmarks import merge_benchmarks
    from blueearth_cst.shared.merge_logs import merge_logs

    directory = path.parents[3] / record["execution_directory"]
    parts = directory / "_parts" / f"{workflow}-{record['invocation_id']}"
    remove_parts = record["status"] == "succeeded"
    logs = parts / "logs"
    output = directory / f"{workflow}.log"
    if logs.is_dir():
        labels = [item.stem if item.is_file() else item.name for item in logs.iterdir()]
        merge_logs(sorted(labels), str(output), str(logs), remove_parts=remove_parts)
    report = directory / f"{workflow}-benchmarks.md"
    benchmark_parts = parts / "benchmarks"
    if report.with_suffix(".tsv").is_file():
        merge_benchmarks(
            str(benchmark_parts),
            workflow[-1],
            str(report),
            invocation_path=path,
            refresh=True,
        )
    elif benchmark_parts.is_dir():
        merge_benchmarks(
            str(benchmark_parts),
            workflow[-1],
            str(report),
            invocation_path=path,
            remove_parts=remove_parts,
        )
    artifacts = record.setdefault("artifacts", [])
    for candidate in (
        output,
        report,
        report.with_suffix(".tsv"),
        directory / "wf4-indicators.log",
    ):
        if candidate.is_file() and (
            candidate.name != "wf4-indicators.log" or workflow == "wf4"
        ):
            entry = {
                "root": "project_root",
                "path": candidate.relative_to(path.parents[3]).as_posix(),
            }
            if entry not in artifacts:
                artifacts.append(entry)
