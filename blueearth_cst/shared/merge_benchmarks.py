"""Gather execution-owned benchmark parts into a compact report and raw TSV.

The workflow prefix filters both reading and deletion. Nested parts from other
workflows or collections are never consumed. Retained TSV rows allow the launcher
to refresh elapsed time after the canonical invocation has finished.
"""

import glob
import json
import math
import os
import re
from pathlib import Path
from typing import Any

import pandas as pd

from blueearth_cst.shared.invocation_summary import _duration
from blueearth_cst.shared.snake_utils import _log_header_lines


def _hms(seconds: float) -> str:
    """Format a measured duration without hiding missing values."""
    if pd.isna(seconds) or not math.isfinite(seconds):
        return "unavailable"
    seconds = int(round(seconds))
    return f"{seconds // 3600}:{(seconds % 3600) // 60:02d}:{seconds % 60:02d}"


def _measurements(frame: pd.DataFrame, column: str) -> pd.Series:
    """Select finite, nonnegative observations for explicit partial summaries."""
    if column not in frame:
        return pd.Series(index=frame.index, dtype=float)
    values = pd.to_numeric(frame[column], errors="coerce")
    return values.where(values.ge(0) & values.map(math.isfinite))


def _rule_name(label: str) -> str:
    """Keep identifiers in details and readable rule labels in the overview."""
    rule = label.split("/")[0]
    return re.sub(r"^\d+\.\d+[a-z]?_", "", rule).replace("_", " ").capitalize()


def _rule_summary(merged: pd.DataFrame) -> pd.DataFrame:
    """Summarize fan-out jobs while accounting separately for benchmark repeats."""
    rows = []
    groups = merged.groupby(merged["rule"].str.split("/").str[0], sort=True)
    for rule, group in groups:
        times = _measurements(group, "s")
        rss = _measurements(group, "max_rss")
        # Repeats are observations of the same job, not additional fan-out jobs.
        # Sum per-job medians so repetition does not inflate estimated job cost.
        job_times = times.groupby(group["rule"]).median()
        missing = int(times.isna().sum())
        rows.append(
            {
                "Rule": _rule_name(rule),
                "Jobs": group["rule"].nunique(),
                "Samples": len(group),
                "Summed job time": _hms(job_times.sum(min_count=len(job_times))),
                "Median/job": _hms(job_times.median()),
                "Slowest job": _hms(job_times.max()),
                "Peak job RAM (MB)": (
                    f"{rss.max():.1f}" if rss.notna().any() else "unavailable"
                ),
                "Missing timing samples": missing,
            }
        )
    return pd.DataFrame(rows)


def _julia_phases(record: dict[str, Any]) -> str:
    """Summarize available process-local Julia timings without imputing gaps."""
    directory = (record.get("resources") or {}).get("directory")
    if not directory:
        return ""
    rows = []
    failed = 0
    for path in sorted(Path(directory).glob("julia-*.tsv")):
        frame = pd.read_csv(path, sep="\t")
        required = {"phase", "status", "wall_s", "compile_s"}
        if not required.issubset(frame.columns):
            continue
        failed += int(frame["status"].ne("ok").sum())
        members = frame.loc[frame["phase"].eq("member")]
        # Each file is one Julia process; the first observed member can fail.
        first = members.iloc[:1]
        later = members.iloc[1:]
        load = frame.loc[frame["phase"].eq("wflow_load") & frame["status"].eq("ok")]
        for label, values in (
            ("Wflow load", _measurements(load, "wall_s")),
            (
                "First observed member compilation",
                _measurements(first.loc[first["status"].eq("ok")], "compile_s"),
            ),
            (
                "Later observed members",
                _measurements(later.loc[later["status"].eq("ok")], "wall_s"),
            ),
        ):
            rows.extend((label, value) for value in values.dropna())
    if not rows and not failed:
        return ""
    lines = ["", "## Julia phases", ""]
    if rows:
        observed = pd.DataFrame(rows, columns=["Phase", "seconds"])
        summary = observed.groupby("Phase", sort=False)["seconds"].agg(
            ["count", "median"]
        )
        summary.columns = ["Observations", "Median duration"]
        summary["Median duration"] = summary["Median duration"].map(_hms)
        lines.extend([summary.reset_index().to_markdown(index=False), ""])
    lines.append(
        "Available successful observations only; missing phases are unmeasured. "
        f"Failed observations: {failed}. First/later members are identified per Julia process."
    )
    return "\n".join(lines) + "\n"


def merge_benchmarks(
    parts_dir: str | os.PathLike[str],
    workflow_num: str | int,
    out_path: str | os.PathLike[str],
    *,
    invocation_path: str | os.PathLike[str] | None = None,
    remove_parts: bool = True,
    refresh: bool = False,
) -> None:
    """Write a rule overview, detailed measurements, and adjacent raw TSV.

    Args:
        parts_dir: Root containing this execution's Snakemake benchmark parts.
        workflow_num: Workflow prefix used to isolate parts before deletion.
        out_path: Markdown destination; raw TSV uses the same stem.
        invocation_path: Canonical workflow invocation JSON, if available.
        remove_parts: Delete only successfully merged parts; disable on failure.
        refresh: Read retained TSV instead of parts to update finalized metadata.

    Repeated samples stay in the TSV and detail table. Overview durations use
    each job's median across repeats; incomplete sums remain unavailable.
    """
    out = Path(out_path)
    raw = out.with_suffix(".tsv")
    out.parent.mkdir(parents=True, exist_ok=True)
    prefix = f"{workflow_num}."
    tsvs = []
    if refresh:
        merged = pd.read_csv(raw, sep="\t")
    else:
        tsvs = sorted(
            p
            for p in glob.glob(os.path.join(parts_dir, "**", "*.tsv"), recursive=True)
            if os.path.relpath(p, parts_dir)
            .replace("\\", "/")
            .split("/")[0]
            .startswith(prefix)
        )
        frames = []
        for path in tsvs:
            rule = os.path.splitext(os.path.relpath(path, parts_dir))[0].replace(
                "\\", "/"
            )
            frame = pd.read_csv(path, sep="\t")
            frame.insert(0, "sample", range(1, len(frame) + 1))
            frame.insert(0, "rule", rule)
            frames.append(frame)
        merged = (
            pd.concat(frames, ignore_index=True)
            if frames
            else pd.DataFrame(columns=["rule", "sample"])
        )
        merged.to_csv(raw, sep="\t", index=False)

    record = (
        json.loads(Path(invocation_path).read_text(encoding="utf-8"))
        if invocation_path is not None
        else {}
    )
    duration = _duration(record)
    if record.get("status") == "running":
        duration = "running (not finalized)"
    lines = [
        _log_header_lines(
            out_path, kind="benchmark", time_label="generated", markdown=True
        ),
        f"# WF{workflow_num} benchmarks",
        "",
        f"Workflow elapsed: {duration}.",
        f"Outcome: {record.get('status', 'unknown')}.",
        f"Work performed: {record.get('work_performed', 'unknown')}.",
        "",
        "Coverage: collected benchmark samples only; reused, failed and unmeasured "
        "jobs cannot be counted from these files. Completeness: unknown.",
        "",
    ]
    if merged.empty:
        lines.append(
            "No benchmark parts found; missing measurements do not mean zero cost."
        )
    else:
        lines.extend(
            [
                "## By rule",
                "",
                _rule_summary(merged).to_markdown(index=False),
                "",
                "Summed job time includes overlapping jobs and is not workflow elapsed time. "
                "Repeated samples use each job's median; missing job durations make sums unavailable. "
                "Peak job RAM is the largest individual-job RSS measurement, not concurrent memory.",
                "",
                "<details>",
                "<summary>Individual jobs and resource measurements</summary>",
                "",
                merged.fillna("unavailable").to_markdown(index=False),
                "",
                "</details>",
            ]
        )
    lines.extend(
        [
            "",
            f"Raw measurements: [{raw.name}]({raw.name}).",
            "`s` is job wall time; `cpu_time` is CPU seconds; memory and I/O are MB. "
            "On Windows, `max_pss` mirrors `max_uss` and I/O may under-report.",
            _julia_phases(record),
        ]
    )
    out.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    if remove_parts and not refresh:
        _remove_parts(tsvs, parts_dir)


def _remove_parts(tsvs, parts_dir):
    """Delete only merged parts and prune empty directories, preserving siblings."""
    for path in tsvs:
        try:
            os.remove(path)
        except OSError:
            pass
    for root, _dirs, _files in os.walk(parts_dir, topdown=False):
        try:
            if not os.listdir(root):
                os.rmdir(root)
        except OSError:
            pass


if __name__ == "__main__":
    if "snakemake" in globals():
        sm = globals()["snakemake"]
        merge_benchmarks(
            sm.params.parts_dir,
            str(sm.params.workflow_num),
            sm.output[0],
            invocation_path=os.environ.get("CST_EXECUTION_RECORD"),
        )
