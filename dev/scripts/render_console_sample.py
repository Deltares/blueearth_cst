"""Render a console transcript per workflow through the REAL handler.

Drives `_ConsoleHandler` with the same synthetic records `tests/test_snake_utils.py`
uses, so what comes out is what a run prints. Every rule row below is the
module's own string literal with only its interpolated values staged; the clock
and the durations are staged too, so a transcript is byte-reproducible -- two
renders of an unchanged spec are identical, which is what makes a diff between
renders readable as the console change and nothing else.

Both modules that read a clock are staged. `snake_utils` owns `log_row`, which
stamps a rule's own output; `console_style` owns the handler, which stamps the
RUN/DONE rows and measures each job's elapsed. Staging only the first left a
transcript carrying two clocks -- body rows at the staged time, job rows at
whatever the wall clock said -- so job timings could not be read off the page
and every re-render diffed against the last.

One spec per workflow in `console_sample_specs.py`; a new workflow costs a
spec rather than a script. Writes `<name>.clean.txt` per workflow, plus
`<name>.failed.txt` for a workflow whose spec carries a `Failure`, plus
`transcripts.json`, into this script's own directory unless `--out` says
otherwise.

A run has two shapes a reviewer has to see, not one. The success shape ends
with rule `all` listing its targets; the failure shape never reaches `all`, and
its tail is the only place the log-parts directory is named. A spec without a
`Failure` renders the clean shape alone, and `main` says which those are.

Run it to see what a console change actually looks like without executing a
workflow -- which is how the 2026-09-17 styling pass was reviewed:

    pixi run python dev/scripts/render_console_sample.py --out .tmp/console

`dev/scripts/` because it inspects the repo and is never part of a run.
"""

from __future__ import annotations

import datetime as _dt
import io
import json
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path

from blueearth_cst.shared import console_style as cs
from blueearth_cst.shared import snake_utils as su

HERE = Path(__file__).parent

#: The real clocks, captured before any render swaps a fake one in.
_REAL_SU = (su.time, su.datetime)
_REAL_CS = (cs.time, cs.datetime)


# --------------------------------------------------------------------------
# harness
# --------------------------------------------------------------------------


class Clock:
    """A fake wall/monotonic clock, so durations and stamps are reproducible."""

    def __init__(self, start_at=_dt.datetime(2026, 9, 17, 9, 12, 3)):
        self.t = 0.0
        self.start_at = start_at

    def advance(self, seconds):
        self.t += seconds

    def monotonic(self):  # `time` shim
        return self.t

    def now(self):  # `datetime` shim
        return self.start_at + _dt.timedelta(seconds=self.t)


def _handler():
    base = logging.StreamHandler(io.StringIO())
    base.name = "DefaultStreamHandler"
    base.setFormatter(logging.Formatter("%(message)s"))
    return cs._ConsoleHandler(base)


def _record(msg="", level=logging.INFO, **extra):
    rec = logging.LogRecord("snakemake", level, __file__, 0, msg, (), None)
    rec.__dict__.update(extra)
    return rec


def _job_info(jobid, name, msg, wildcards=None):
    return _record(
        event="job_info",
        jobid=jobid,
        rule_name=name,
        rule_msg=msg,
        wildcards=dict(wildcards or {}),
    )


@dataclass
class Rule:
    """One `rule_banner(...)` call, as the Snakefile makes it."""

    number: str
    name: str
    context: str | None = None
    summary: str | None = None


@dataclass
class Job:
    """One scheduled job: which rule, its wildcards, how long, what it printed."""

    rule: str
    seconds: float = 2
    wildcards: dict = field(default_factory=dict)
    body: tuple = ()  # ((module, message, gap_before), ...)


@dataclass
class Failure:
    """How a run dies: which job fails, what it printed, what Snakemake said.

    `error` is Snakemake's own block, which arrives as an ERROR record with no
    event and therefore keeps its own colouring -- the one thing on this
    console that stays louder than a start line.
    """

    rule: str
    seconds: float = 2
    body: tuple = ()  # rows the rule printed before it died
    error: str = ""  # Snakemake's error block, verbatim
    log_parts_dir: str = "logs/_parts"


@dataclass
class Workflow:
    name: str  # `wf0 analyze_climate`
    project: str
    config: str
    rules: list  # [Rule, ...], `all` included
    stats: str  # Snakemake's `Job stats:` text
    jobs: list  # [Job, ...] in execution order
    targets: list
    elapsed: int
    details: dict = field(default_factory=dict)
    tokens: dict = field(default_factory=dict)
    failure: object = None  # Failure | None -- absent means clean-only


def render(wf: Workflow, failed: bool = False) -> str:
    cs._RULE_NUMBERS.clear()
    cs._RULE_SUMMARIES.clear()
    # Tokens travel through the environment (`_PATH_TOKENS_ENV`), so clear
    # before declaring or one workflow's tokens leak into the next.
    import os

    os.environ.pop(su._PATH_TOKENS_ENV, None)
    if wf.tokens:
        su.declare_path_tokens(**wf.tokens)
    banners = {
        r.name: cs.rule_banner(r.number, r.name, r.context, r.summary) for r in wf.rules
    }
    cs._RUN_HEADER = (wf.name, wf.project, wf.config, dict(wf.details))

    clock = Clock()
    # BOTH modules, or the transcript carries two clocks: `snake_utils` stamps
    # a rule's own `log_row` output, `console_style` stamps the RUN/DONE rows
    # and measures each job's elapsed from its own `time.monotonic`.
    su.time = su.datetime = clock
    cs.time = cs.datetime = clock

    h = _handler()
    out = h.stream

    def emit(*recs):
        for r in recs:
            h.emit(r)

    def rows(items):
        """Rule stdout, as the tee passes it through: `log_row` lines."""
        saved = sys.stdout
        sys.stdout = out
        try:
            for module, message, gap, *rest in items:
                clock.advance(gap)
                su.log_row(message, module=module, level=(rest[0] if rest else "INFO"))
        finally:
            sys.stdout = saved

    emit(_record(wf.stats, event="run_info"))

    fail = wf.failure if failed else None
    if failed and fail is None:
        raise ValueError(f"{wf.name}: no Failure spec to render")

    total = len(wf.jobs)
    done = 0
    for jobid, job in enumerate(wf.jobs, start=1):
        # A target rule prints an unnumbered target_banner, not a rule_banner:
        # the banner plus one line per declared target (only `all` lists any).
        msg = (
            cs.target_banner(
                job.rule, wf.targets if job.rule == "all" else [], wf.project
            )
            if job.rule in cs._PLAN_EXCLUDED_RULES
            else banners[job.rule]
        )
        if job.wildcards:
            msg = msg.format(**job.wildcards)
        emit(_job_info(jobid, job.rule, msg, job.wildcards))
        if fail is not None and job.rule == fail.rule:
            # The run dies HERE: the rule prints what it got to, Snakemake
            # prints its block, and no finish line or progress record follows.
            # Every later job is never scheduled, so the loop ends rather than
            # continuing quietly -- a transcript that kept going would show a
            # run that cannot happen.
            rows(fail.body or job.body)
            clock.advance(fail.seconds)
            emit(_record(fail.error, level=logging.ERROR))
            break
        if job.body:
            rows(job.body)
        clock.advance(job.seconds)
        done += 1
        emit(
            _record(event="job_finished", job_id=jobid),
            _record(event="progress", done=done, total=total),
        )
    h.close()

    text = out.getvalue()
    text += (
        cs.run_summary(
            wf.name,
            wf.project,
            "log",
            "benchmarks",
            elapsed_seconds=int(clock.t) if fail is not None else wf.elapsed,
            failed=fail is not None,
            log_parts_dir=fail.log_parts_dir if fail is not None else None,
            # Counted off the rows this render actually emitted, not declared
            # in the spec: a fabricated tally that disagreed with the rows
            # above it would make the dummy lie about the one thing the field
            # exists to state. A real run counts through the sidecar tally.
            warnings=_staged_warnings(wf, fail),
        )
        + "\n"
    )
    su.time, su.datetime = _REAL_SU
    cs.time, cs.datetime = _REAL_CS

    # Strip the SGR a verdict adds when stderr looks like a console.
    import re

    return re.sub(r"\x1b\[[0-9;]*m", "", text).replace("\r", "")


def _staged_warnings(wf, fail):
    """WARNING rows in the transcript this render just produced."""
    bodies = []
    for job in wf.jobs:
        if fail is not None and job.rule == fail.rule:
            bodies.append(fail.body or job.body)
            break
        bodies.append(job.body)
    # A stall notice counts whatever its level says, mirroring the tally
    # itself: `note_warning("heartbeat")` sits at the stall site rather than
    # behind a level test, because the watchdog emits at INFO and paints the
    # row yellow. Counting it by level here would report a clean run.
    return sum(
        1
        for body in bodies
        for row in body
        if (len(row) > 3 and str(row[3]).upper() in ("WARNING", "ERROR"))
        or row[0] == "heartbeat"
    )


def write(name: str, wf: Workflow) -> dict:
    """Render every shape the spec covers. Keyed by shape, `clean` always."""
    shapes = {"clean": render(wf)}
    if wf.failure is not None:
        shapes["failed"] = render(wf, failed=True)
    for shape, text in shapes.items():
        (HERE / f"{name}.{shape}.txt").write_text(text, encoding="utf-8", newline="\n")
    return shapes


def main(out_dir=None):
    from console_sample_specs import WORKFLOWS

    global HERE
    if out_dir:
        HERE = Path(out_dir)
        HERE.mkdir(parents=True, exist_ok=True)

    payload = {}
    clean_only = []
    for name, wf in WORKFLOWS.items():
        shapes = write(name, wf)
        payload[name] = {
            "label": wf.name,
            "shapes": {s: t.split("\n") for s, t in shapes.items()},
        }
        counts = "  ".join(
            f"{s} {len(t.split(chr(10)))} lines" for s, t in shapes.items()
        )
        sys.stdout.write(f"{name}: {counts}\n")
        if "failed" not in shapes:
            clean_only.append(name)
    (HERE / "transcripts.json").write_text(
        json.dumps(payload, indent=1), encoding="utf-8", newline="\n"
    )
    # A renderer that bounds its own coverage says so: a reviewer who sees only
    # clean transcripts should know which are clean BY SPEC and which are
    # clean because nobody wrote the failure down.
    if clean_only:
        sys.stdout.write(
            f"no failure spec, clean shape only: {', '.join(clean_only)}\n"
        )


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent))
    _args = sys.argv[1:]
    main(_args[_args.index("--out") + 1] if "--out" in _args else None)
