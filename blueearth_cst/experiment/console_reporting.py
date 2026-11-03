"""Console summaries from already-validated retained artifacts; no model calculations."""

import csv
import logging
import math
import re
import sys
from collections import Counter
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime
from pathlib import Path
from threading import RLock, local

from blueearth_cst.shared.run_log_core import _write_console, log_row, plural


def record_indicator_detail(root, message):
    """Retain technical details even in Snakemake workers without file handlers."""
    # Snakemake's asynchronous file handler can inherit a legacy Windows
    # encoding. Keep that copy ASCII-safe; the dedicated log retains UTF-8.
    native_message = message.encode("ascii", "backslashreplace").decode("ascii")
    logger = logging.getLogger("snakemake.logging")
    # Publication also runs in the launcher, where Snakemake has no handlers.
    # An unconfigured logger would leak these details to root console handlers.
    if logger.handlers:
        logger.info("Indicator details: " + native_message)
    if root is not None:
        from blueearth_cst.shared.execution_outputs import current_directory

        directory = current_directory()
        path = (
            directory / "wf4-indicators.log"
            if directory
            else Path(root) / "logs/wf4_indicators.log"
        )
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as handle:
                handle.write(f"{datetime.now():%H:%M:%S} - {message}\n")
        except OSError:
            # A console-only failure must not stop calculation or lose details.
            _write_console(sys.stdout, "Indicator details: " + message + "\n")


def matching_sets(sets, document):
    """Return every user-facing set sharing this request or manifest.

    ``document`` is a request, plan request or manifest; its calculable and
    unavailable records together are the set's selection.
    """
    from blueearth_cst.experiment.indicator_plan import selected_calculations
    from blueearth_cst.experiment.indicator_sets import calculation_records

    calculations = selected_calculations(document)
    return [
        item for item in sets if calculation_records(item.indicators) == calculations
    ]


def unavailable_summary(document) -> list[str]:
    """One short phrase per unavailable indicator: its locations or response."""
    phrases = [
        f"{item['calculation']['indicator']} ({', '.join(item['reasons'])})"
        for item in document.get("unavailable_calculations", [])
    ]
    by_indicator = {}
    for row in document.get("unavailable_results", []):
        by_indicator.setdefault((row["indicator"], row["reason"]), []).append(
            row["location"]
        )
    phrases += [
        f"{indicator} at {plural(len(locations), 'location')} ({reason})"
        for (indicator, reason), locations in sorted(by_indicator.items())
    ]
    return phrases


_PLAN_ROOT: ContextVar[str | Path | None] = ContextVar(
    "indicator_plan_log_root", default=None
)


class _PlanStream:
    """Route the producer's technical planning row to the full Snakemake log."""

    def __init__(self, stream, root=None):
        self.stream = stream
        self.root = root
        self.pending = local()
        self.users = 0

    def __getattr__(self, name):
        return getattr(self.stream, name)

    def write(self, text):
        if self.root is None and _PLAN_ROOT.get() is None:
            return self.stream.write(text)
        pending = getattr(self.pending, "text", "") + text
        while "\n" in pending:
            line, pending = pending.split("\n", 1)
            plain = re.sub(r"\x1b\[[0-9;]*m", "", line).rstrip("\r")
            if re.fullmatch(
                r"\d{2}:\d{2}:\d{2} - indicators - (?:"
                r"Indicator set \w+ planned: \d+ indicator declarations?; "
                r"\d+ expected result keys?|\w+: \d+ declarations? over \d+ result keys?)",
                plain,
            ):
                record_indicator_detail(self.root or _PLAN_ROOT.get(), plain)
            else:
                self.stream.write(line + "\n")
        self.pending.text = pending
        return len(text)

    def flush(self):
        pending = getattr(self.pending, "text", "")
        if pending:
            self.stream.write(pending)
            self.pending.text = ""
        self.stream.flush()


_PLAN_STREAM_LOCK = RLock()
_ACTIVE_PLAN_STREAM: _PlanStream | None = None


@contextmanager
def _planning_output(root: str | Path) -> Iterator[None]:
    """Keep overlapping plans on one proxy until the last plan finishes.

    Per-call redirect_stdout is process-global: an earlier plan can restore
    stdout while a later plan still runs. Roots and partial rows stay local
    to each planning thread; warnings and unrelated output pass through.
    """
    global _ACTIVE_PLAN_STREAM
    with _PLAN_STREAM_LOCK:
        if _ACTIVE_PLAN_STREAM is None:
            _ACTIVE_PLAN_STREAM = _PlanStream(sys.stdout)
            sys.stdout = _ACTIVE_PLAN_STREAM
        stream = _ACTIVE_PLAN_STREAM
        stream.users += 1
    token = _PLAN_ROOT.set(root)
    try:
        yield
    finally:
        try:
            stream.flush()
        finally:
            _PLAN_ROOT.reset(token)
            with _PLAN_STREAM_LOCK:
                stream.users -= 1
                if stream.users == 0:
                    if sys.stdout is stream:
                        sys.stdout = stream.stream
                    _ACTIVE_PLAN_STREAM = None


def prepare_indicator_plan(root, request, sets, evaluation="strict"):
    """Keep numerical planning unchanged and present its outcome by set name.

    Strict evaluation refuses a request with unavailable entries before planning.
    """
    from blueearth_cst.experiment.indicator_plan import write_indicator_plan
    from blueearth_cst.shared.snake_utils import listed

    unavailable = unavailable_summary(request)
    if unavailable and evaluation == "strict":
        names = ", ".join(item.set_id for item in matching_sets(sets, request))
        raise ValueError(
            f"{names or 'selected set'}: selected indicators lack inputs: "
            f"{listed(unavailable)}. Exclude them with indicator_selection, supply "
            "the missing parameter, or set indicator_evaluation: partial"
        )
    with _planning_output(root):
        plan = write_indicator_plan(root, request)
    for item in matching_sets(sets, request):
        log_row(
            f"{item.set_id}: {plural(len(item.indicators), 'indicator')}; "
            f"{plural(len(plan['expected_result_keys']), 'result')} across simulations and locations",
            module="indicators",
        )
    return plan


def undefined_result_summary(root: Path, manifest: dict) -> tuple[int, list[str]]:
    """Separate expected absent-event timing from other undefined estimates.

    ERFA zero-flow timing has no numerical value when no event occurs. Group
    means propagate an undefined member, so the cause can be one contributing
    simulation. Other undefined values retain their indicator names for review.
    """
    zero_timing = {
        item["indicator"]
        for item in manifest.get("calculations", [])
        if item["method"] == "erfa_annual_statistic/1"
        and item["parameters"].get("metric") == "time_main_zero_seq"
    }
    expected = 0
    other = Counter()
    for table in manifest["tables"]:
        with (root / table["file"]["path"]).open(
            encoding="utf-8", newline=""
        ) as handle:
            for row in csv.DictReader(handle):
                status = row.get("status")
                undefined = status == "undefined" or (
                    not status
                    and (not row["value"] or not math.isfinite(float(row["value"])))
                )
                if not undefined:
                    continue
                if row["indicator"] in zero_timing:
                    expected += 1
                else:
                    other[row["indicator"]] += 1
    return expected, [
        f"{name} ({plural(count, 'result')})" for name, count in sorted(other.items())
    ]


def result_counts(root: Path, manifest: dict) -> tuple[int, int, int]:
    """Count available, undefined and unavailable results in validated tables.

    Tables before indicator-set/6 have no status; their NaN is undefined.
    """
    counts = {"available": 0, "undefined": 0, "unavailable": 0}
    for table in manifest["tables"]:
        with (root / table["file"]["path"]).open(
            encoding="utf-8", newline=""
        ) as handle:
            for row in csv.DictReader(handle):
                value = row["value"]
                status = row.get("status") or (
                    "available"
                    if value and math.isfinite(float(value))
                    else "undefined"
                )
                counts[status] += 1
    return counts["available"], counts["undefined"], counts["unavailable"]


def report_indicator_set(root: Path, manifest: dict, sets=(), *, reused=False) -> None:
    """Retain calculation details; rule markers already report completion."""
    record_indicator_detail(
        root,
        f"Indicator set {manifest['indicator_set_id']}; "
        f"manifest → {root / manifest['engine_root'] / 'indicators.json'}",
    )
    from blueearth_cst.shared.snake_utils import listed

    unavailable = unavailable_summary(manifest)
    for item in matching_sets(sets, manifest):
        action = "reusing calculated results" if reused else "calculations complete"
        record_indicator_detail(root, f"{item.set_id}: {action}")
        if unavailable:
            # The warning row belongs to publication; keep the detail here.
            record_indicator_detail(
                root, f"{item.set_id}: unavailable indicators: {listed(unavailable)}"
            )
