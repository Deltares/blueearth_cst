"""Readable views of explicit invocation records, never inferred job history."""

from __future__ import annotations

import json
import math
import os
import re
import uuid
from pathlib import Path
from typing import Any
from urllib.parse import quote


def summary_path(record_path: Path) -> Path:
    """Return the stable report location for this invocation, not this project."""
    if record_path.is_file():
        record = json.loads(record_path.read_text(encoding="utf-8"))
        if record.get("execution_directory"):
            return record_path.parents[3] / record["execution_directory"] / "summary.md"
    return record_path.parents[3] / "benchmarks" / f"run-{record_path.stem}.md"


def _cell(value: Any) -> str:
    """Keep arbitrary saved labels within one Markdown table cell."""
    return str(value).replace("|", "\\|").replace("\n", " ").replace("\r", " ")


def _link(label: str, target: Path, report: Path) -> str:
    """Use relative, URL-escaped file links."""
    relative = Path(os.path.relpath(target, report.parent)).as_posix()
    return f"[{_cell(label)}]({quote(relative, safe='/')})"


def _duration(record: dict[str, Any]) -> str:
    """Render recorded durations only; timestamps are not a duration fallback."""
    value = record.get("elapsed_seconds")
    if (
        record.get("ended_at_utc") is None
        or record.get("status") == "running"
        or isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value < 0
    ):
        return "unavailable"
    total = int(value)
    return f"{total // 3600}:{total % 3600 // 60:02d}:{total % 60:02d} ({value:.3f} s)"


def _outcome(record: dict[str, Any]) -> str:
    """Describe observed completion without converting unfinished records to success."""
    if record.get("status") == "running":
        return "unfinished (last observed running)"
    if record.get("ended_at_utc") is None:
        return "unconfirmed (no completion recorded)"
    outcome = str(record.get("status", "unknown"))
    if record.get("exit_code") is not None:
        outcome += f" (exit {record['exit_code']})"
    return outcome


def render_summary(record_path: Path) -> str:
    """Read one invocation and its explicitly linked children, including old records.

    An unfinished attempt stays unfinished. Missing elapsed seconds stay
    unavailable. Child IDs resolve beside the parent, including old records
    whose project-relative reference used the project's parent as its base.
    """
    record_path = record_path.resolve()
    record = json.loads(record_path.read_text(encoding="utf-8"))
    report = summary_path(record_path)
    root = record_path.parents[3]
    lines = [
        f"# Run {record['invocation_id']}",
        "",
        f"Source: {_link('invocation JSON', record_path, report)}. Structured records are authoritative.",
        "",
        f"Outcome: {_outcome(record)}. Mode: {_cell(record.get('mode', 'unknown'))}.",
        f"Elapsed: {_duration(record)}.",
        f"Started (UTC): {_cell(record.get('started_at_utc', 'unavailable'))}.",
        f"Ended (UTC): {_cell(record.get('ended_at_utc') or 'not observed')}.",
        f"Work performed: {_cell(record.get('work_performed', 'unknown'))}.",
        "",
        "Exit zero records launcher success; it does not establish completed computation or scientific completeness.",
        "Dry-run outcomes describe planning checks. Unavailable durations are never reconstructed from UTC timestamps.",
        "",
        "| Workflow | Outcome | Elapsed | Mode | Warnings | Work performed | Source |",
        "|---|---|---|---|---|---|---|",
    ]
    children = record.get("children", [])
    selection = record.get("workflow_selection", {})
    requested = record.get("requested_workflows", [])
    names = list(record.get("workflow_order", [])) or list(selection) or list(requested)
    for item in children:
        if item["workflow"] not in names:
            names.append(item["workflow"])
    failures = {item["workflow"]: item for item in record.get("launch_failures", [])}
    for name in failures:
        if name not in names:
            names.append(name)
    standalone = record.get("workflow")
    if standalone:
        names = [standalone]
    diagnostics = []
    for name in names:
        source = None
        observed = record if standalone else None
        if standalone:
            source = record_path
        else:
            for item in children:
                if item["workflow"] != name:
                    continue
                identity = item.get("invocation_id", "")
                if not re.fullmatch(r"[A-Za-z0-9_-]+", identity):
                    raise ValueError("invalid linked child invocation ID")
                source = record_path.with_name(f"{identity}.json")
                try:
                    observed = json.loads(source.read_text(encoding="utf-8"))
                except FileNotFoundError:
                    source = None
                    break
                if (
                    observed.get("invocation_id") != identity
                    or observed.get("parent_invocation_id") != record["invocation_id"]
                    or observed.get("workflow") != name
                ):
                    raise ValueError("linked child does not belong to this invocation")
                break
        if observed is not None:
            warning = record.get("warning_counts", {}).get(
                name, observed.get("warning_count")
            )
            values = [
                name,
                _outcome(observed),
                _duration(observed),
                observed.get("mode", "unknown"),
                "unavailable" if warning is None else warning,
                observed.get("work_performed", "unknown"),
            ]
            if observed.get("resources"):
                diagnostics.append((name, observed["resources"]))
        else:
            elapsed = "unavailable"
            if name in failures:
                outcome = "launch failed (no child record)"
                elapsed = _duration(
                    {
                        **failures[name],
                        "status": "failed",
                        "ended_at_utc": record.get("ended_at_utc"),
                    }
                )
            elif selection.get(name) is False:
                outcome = "disabled"
            elif any(item["workflow"] == name for item in children):
                outcome = "child record unavailable"
            else:
                outcome = "not invoked"
            values = [
                name,
                outcome,
                elapsed,
                record.get("mode", "unknown"),
                "unavailable",
                "unknown",
            ]
        lines.append(
            "| "
            + " | ".join(_cell(value) for value in values)
            + " | "
            + (_link("record", source, report) if source else "unavailable")
            + " |"
        )
    if not names:
        lines.extend(
            ["", "Workflow selection unavailable; no child invocations recorded."]
        )
    if record.get("status") == "failed":
        lines.extend(
            [
                "",
                "Stop boundary: workflows marked not invoked have no observed child launch in this attempt.",
            ]
        )
    for item in record.get("launch_failures", []):
        lines.extend(
            [
                "",
                f"Launch failure: {_cell(item['workflow'])} — {_cell(item.get('error', {}).get('type', 'unknown'))}.",
            ]
        )
    if record.get("error"):
        lines.extend(
            ["", f"Launcher error: {_cell(record['error'].get('type', 'unknown'))}."]
        )
    if record.get("execution_directory"):
        directory = root / record["execution_directory"]
        lines.extend(
            ["", _link("Execution logs and benchmarks", directory, report) + "."]
        )
        if (directory / "console.log").is_file():
            lines.append(
                _link("Console transcript", directory / "console.log", report) + "."
            )
        for item in children:
            source = record_path.with_name(f"{item['invocation_id']}.json")
            if source.is_file():
                child = json.loads(source.read_text(encoding="utf-8"))
                for artifact in child.get("artifacts", []):
                    target = root / artifact["path"]
                    lines.append(_link(target.name, target, report) + ".")
        for artifact in record.get("artifacts", []):
            target = root / artifact["path"]
            lines.append(_link(target.name, target, report) + ".")
    elif (root / "logs").is_dir():
        lines.extend(
            [
                "",
                _link(
                    "Project logs (shared; may include earlier attempts)",
                    root / "logs",
                    report,
                )
                + ".",
            ]
        )
    for name, resources in diagnostics:
        directory = resources.get("directory")
        if directory:
            lines.extend(
                [
                    "",
                    f"## {_cell(name)} resource diagnostics",
                    "",
                    _link(
                        "Resource trace and Julia/batch diagnostics",
                        Path(directory),
                        report,
                    )
                    + ".",
                    f"Tracking status: {_cell(resources.get('status', 'unknown'))}.",
                    f"Sampled mean CPU cores: {_cell(resources.get('sampled_mean_cpu_cores', 'unavailable'))}.",
                    f"Sampled peak process-tree RSS (bytes): {_cell(resources.get('sampled_peak_tree_rss_bytes', 'unavailable'))}.",
                    "Sampled CPU totals are lower bounds; summed RSS may double-count shared pages.",
                ]
            )
    lines.extend(
        [
            "",
            "Per-rule benchmark sums are job time, not end-to-end elapsed time.",
            "Maximum individual-job memory is not peak concurrent workflow memory.",
            "",
        ]
    )
    return "\n".join(lines)


def write_summary(record_path: Path) -> Path:
    """Atomically refresh only this invocation's derived report."""
    record_path = record_path.resolve()
    report = summary_path(record_path)
    content = render_summary(record_path)
    report.parent.mkdir(parents=True, exist_ok=True)
    temporary = report.with_name(f".{report.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(content, encoding="utf-8", newline="\n")
        os.replace(temporary, report)
    finally:
        temporary.unlink(missing_ok=True)
    return report
