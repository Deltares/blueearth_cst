"""Run enabled CST workflows in the accepted five-stage order.

The closed workflow set is analyze_climate, build_model, analyze_projections,
generate_scenarios, simulate_system. Every enabled flag must be a boolean.
Disabled workflows are reported and skipped; the first failure stops the run.
Preflights occur immediately before their consumer, after preceding producers.
Startup validates static WF4 settings without preparing historical Qcrit. A
combined dry-run with WF1 enabled reports missing historical Q90 inputs as
pending; it cannot inspect the data-dependent WF4 DAG until those inputs exist.
Simulation uses the dedicated runner's operation/target validation and starts
one Snakemake invocation. Its execution-option allowlist also applies here.
Forwarded --unlock unlocks each enabled workflow without generating scenarios,
capturing execution sources, or publishing indicators.

--rebuild permanently erases the project's generated namespaces before normal
admission and execution. It requires default-no interactive confirmation or
--yes. With -- --dry-run it only previews the reset, without writing records or
invoking workflows. Sources, unrelated paths, logs and attempt history survive;
all experiments and allocations are erased, including disabled workflows' outputs.

Each invocation records resolved configuration provenance, checked commands,
selected simulation operation/targets, stage timing and final status in a
unique config/_engine/invocations JSON record. Console handoffs are flushed before
the child starts; sensitive argument values are redacted in records and output.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import re
import shlex
import shutil
import subprocess
import sys
import textwrap
import time
import uuid
from collections.abc import Mapping
from contextlib import contextmanager, suppress
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

# Make the plain source tree importable when this file is executed directly as
# ``python scripts/run_workflows.py`` rather than imported by pytest.
_REPO_ROOT_PATH = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT_PATH) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT_PATH))

from blueearth_cst.shared import (  # noqa: E402
    console_style,
    execution_outputs,
    invocation_history,
    product_record,
)
from blueearth_cst.shared.config_composition import (  # noqa: E402
    capture_configuration_sources,
)
from blueearth_cst.shared.cross_workflow_leaves import (  # noqa: E402
    LEAF_PRODUCER,
    LEAVES,
)
from blueearth_cst.shared.provenance import (  # noqa: E402
    effective_config_digest,
    environment_file_hashes,
    file_sha256,
    toolbox_identity,
    toolbox_release_tag,
)
from blueearth_cst.shared.run_log_core import (  # noqa: E402
    RUN_WARNING_TALLY_ENV,
    _Heartbeat,
    _line_reset,
)
from blueearth_cst.shared.snake_utils import (  # noqa: E402
    _ANSI_FAIL,
    ADVANCED_SETTINGS,
    _ansi,
    _console_colour,
    format_elapsed,
    region_geojson_path,
)
from blueearth_cst.shared.windows_job import run_project_child  # noqa: E402
from blueearth_cst.shared.workflow_config_snapshot import (  # noqa: E402
    archive_lock,
    file_reference,
    read_archive,
)

# Fixed order: climate -> model -> projections -> generation -> simulation.
# WF0 is optional and supplies shared region and climate inputs. Generation and
# model construction are independent prerequisites of simulation; projections
# remain a plausibility overlay and do not drive generation. Consumer preflights
# run after the preceding enabled producers.
# Each workflow maps to its Snakefile and preserved execution flags below;
# only analyze_projections uses --keep-going.
WORKFLOW_ORDER = (
    "analyze_climate",
    "build_model",
    "analyze_projections",
    "generate_scenarios",
    "simulate_system",
)

SNAKEFILE = {
    "analyze_climate": "analyze_climate.smk",
    "build_model": "build_model.smk",
    "analyze_projections": "analyze_projections.smk",
    "generate_scenarios": "generate_scenarios.smk",
    "simulate_system": "simulate_system.smk",
}

PER_WORKFLOW_FLAGS = {
    "analyze_climate": [],
    "build_model": [],
    "analyze_projections": ["--keep-going"],
    "generate_scenarios": [],
    "simulate_system": [],
}

# Workflow -> its `wf<N>` id, for the console only. This is the THIRD copy of
# that mapping in the repo and the other two must be kept in step with it:
# `scripts/plot_workflow_dag.py::WORKFLOW_NUMBER` keys it by Snakefile name for
# the render filenames, and each Snakefile hardcodes the assembled label in its
# own `run_header`/`run_summary` calls ("wf0 analyze_climate", ...). The merged
# log names in blueearth_cst/shared/merge_logs.py carry the same ids.
#
# 0 for analyze_climate because it runs BEFORE model creation; `W` is a workflow
# id, not a position (dev/reference/dev-naming-conventions.md §9).
WORKFLOW_ID = {
    "analyze_climate": "wf0",
    "build_model": "wf1",
    "analyze_projections": "wf2",
    "generate_scenarios": "wf3",
    "simulate_system": "wf4",
}

# Repo root = parent of scripts/. Snakefiles and config paths are repo-root
# relative and the wrapper is invoked from repo root, mirroring the runners.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

_SENSITIVE_KEY_RE = re.compile(
    r"api[-_]?key|auth|credential|pass(?:word|wd)?|private[-_]?key|secret|token",
    re.IGNORECASE,
)


class ConfigError(Exception):
    """Raised for a config that violates the wrapper's contract (a)-(c)."""


class PrerequisiteError(Exception):
    """Raised when an enabled workflow's cross-workflow inputs are ABSENT.

    Distinct from `ConfigError`: the config may be perfectly well-formed and
    still name a combination this project cannot satisfy. Both exit 2 -- the
    wrapper has one "it never started" code and inventing a second would be a
    contract change nobody asked for (contract (i)).
    """


def read_enabled_flags(config_path: str) -> dict[str, bool]:
    """Parse and validate the per-workflow enabled flags (contract (a)-(c)).

    Raises ConfigError on: no `workflows:` section, a missing `<name>` or
    `<name>.enabled` key, or an `enabled:` value that does not parse to a bool.
    """
    cfg = _read_config(config_path)
    return _enabled_flags(cfg, config_path)


def _read_config(config_path: str) -> Mapping[str, Any]:
    """Load a wrapper source config as a YAML mapping."""
    path = Path(config_path).expanduser()
    if not path.is_file():
        # Name the ABSOLUTE path that was tried. `pixi run run-workflows`
        # executes at the manifest root whatever directory it was invoked from,
        # so a relative --config resolves against the repo root rather than the
        # caller's cwd -- and pixi exposes no INIT_CWD to recover that cwd, so
        # the only cure is showing where the lookup actually went.
        raise ConfigError(
            f"{config_path}: config not found at {path.resolve()} "
            f"(a relative --config resolves against the current directory, which "
            f"is the repo root under `pixi run`; pass an absolute path instead)"
        )
    with open(path, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, Mapping):
        raise ConfigError(f"{config_path}: config is not a mapping")
    from blueearth_cst.shared.config_paths import resolve_project_paths

    return resolve_project_paths(config, path)


def _enabled_flags(cfg: Mapping[str, Any], config_path: str) -> dict[str, bool]:
    """Validate and return workflow enablement from an already loaded config."""

    if "workflows" not in cfg:
        raise ConfigError(
            f"{config_path}: no 'workflows:' section -- this is not a full "
            f"orchestration config. The single-workflow projections configs are "
            f"direct 'snakemake -s' inputs, not wrapper inputs."
        )
    workflows = cfg["workflows"]
    if not isinstance(workflows, dict):
        raise ConfigError(f"{config_path}: 'workflows:' is not a mapping")
    if "run_stress_test" in workflows:
        raise ConfigError(
            "run_stress_test is retired; migrate to generate_scenarios and simulate_system with scripts/migrate_project_config.py from toolbox v0.3.0"
        )
    unknown = set(workflows) - set(WORKFLOW_ORDER)
    if unknown:
        raise ConfigError(f"{config_path}: unknown workflows {sorted(unknown)}")

    flags: dict[str, bool] = {}
    for name in WORKFLOW_ORDER:
        if name not in workflows or not isinstance(workflows[name], dict):
            raise ConfigError(
                f"{config_path}: missing 'workflows.{name}' section "
                f"(required for a full orchestration config)"
            )
        section = workflows[name]
        if "enabled" not in section:
            raise ConfigError(f"{config_path}: missing 'workflows.{name}.enabled' key")
        value = section["enabled"]
        if not isinstance(value, bool):
            raise ConfigError(
                f"{config_path}: 'workflows.{name}.enabled' must parse to a "
                f"boolean (got {value!r} of type {type(value).__name__}); use an "
                f"unquoted true/false (yes/no/on/off also accepted), not a "
                f"quoted string or integer"
            )
        flags[name] = value
    return flags


def _project_dir(cfg: Mapping[str, Any], config_path: str) -> Path:
    """Resolve the configured project output root from a wrapper config."""
    project = cfg.get("project")
    if not isinstance(project, Mapping):
        raise ConfigError(f"{config_path}: missing or invalid 'project:' section")
    project_dir = project.get("project_dir")
    if (
        not isinstance(project_dir, (str, os.PathLike))
        or not os.fspath(project_dir).strip()
    ):
        raise ConfigError(
            f"{config_path}: 'project.project_dir' must be a non-empty path"
        )
    path = Path(project_dir).expanduser()
    if not path.is_absolute():
        path = _REPO_ROOT_PATH / path
    return path.resolve()


def missing_wf1_leaves(flags: Mapping[str, bool], project_dir: Path) -> list[str]:
    """Check WF4 model leaves after any enabled WF1 producer has completed."""
    if not flags["simulate_system"]:
        return []
    return [leaf for leaf in LEAVES if not (project_dir / leaf).exists()]


def _check_wf1_leaves(
    flags: Mapping[str, bool], project_dir: Path, config_path: str
) -> None:
    """Raise `PrerequisiteError` if wf3 is enabled with no wf1 behind it."""
    missing = missing_wf1_leaves(flags, project_dir)
    if not missing:
        return
    listed = "\n".join(f"    {leaf}" for leaf in missing)
    raise PrerequisiteError(
        f"{config_path}: simulate_system requires {LEAF_PRODUCER} artifacts; {len(missing)} of "
        f"{len(LEAVES)} required model files are absent from "
        f"{project_dir}:\n{listed}\n"
        f"Only a {LEAF_PRODUCER} run produces them. Set "
        f"'workflows.{LEAF_PRODUCER}.enabled: true', or disable "
        f"'simulate_system'. Simulation has not been invoked."
    )


def build_command(
    name: str, config_path: str, cores: int, extra: list[str]
) -> list[str]:
    """Assemble the snakemake argv for one workflow (contract (e)/(f))."""
    if name == "simulate_system":
        return [
            sys.executable,
            str(_REPO_ROOT_PATH / "scripts/simulate_system.py"),
            "--config",
            config_path,
            "--cores",
            str(cores),
            "--",
            *extra,
        ]
    return [
        "snakemake",
        "all",
        "-c",
        str(cores),
        "-s",
        SNAKEFILE[name],
        "--configfile",
        config_path,
        *PER_WORKFLOW_FLAGS[name],
        *extra,
    ]


# --------------------------------------------------------------------------
# Console: the wrapper's own narration (contract (h))
# --------------------------------------------------------------------------
#
# The wrapper had no voice of its own until 2026-08-16: it echoed a snakemake
# command line per workflow and nothing else, so a console showing four
# back-to-back Snakemake runs never said which PROJECT they belonged to, which
# of the four were going to run, or how long any of it took. Each Snakefile
# already opens with `run_header` and closes with `run_summary`; the runner
# ABOVE them said less about the run than any one of them.
#
# Row grammar below is deliberately theirs -- labelled groups whose `key  value`
# rows indent one step further and share one value column across the block -- so
# the facts inside a wrapper block read the same way as the facts inside a
# workflow's. It is restated here rather than imported because
# `run_header`/`run_summary` are shaped for a Snakefile's facts (declared path
# tokens, a merged-log and benchmark name), none of which a runner has.
#
# **What the wrapper does NOT share with them is how it is FRAMED**, and that is
# the point. Nested one inside the other, the two blocks were indistinguishable:
# the wrapper's `run` group and `wf0 analyze_climate`'s own `run` group are the
# same label over the same-shaped rows, so a console scrolled back to could not
# be parsed into "the runner said this, then the workflow said that". Every
# wrapper utterance is therefore bounded by a full-width `=` rule -- the opening
# banner, each per-workflow hand-off, the closing banner -- and nothing else in
# this console draws one. A rule means the RUNNER is speaking.
#
# For the same reason the per-workflow rows are NOT `log_row`'s
# `HH:MM:SS - <module> - ...`. That grammar belongs to reporting from INSIDE a
# workflow: every rule log line, every hydromt record and every heartbeat marker
# wears it, so a runner wearing it too reads as one more rule in the run it is
# actually supervising. The wrapper keeps the clock (a start time, an elapsed)
# and drops the costume. It consequently ignores `CST_LOG_LEVEL` entirely, which
# is correct rather than a loss -- that floor exists to quieten rule logs, and
# the runner's dozen lines are the frame around them.
#
# Rules cap at 80 columns; readable rows fit the terminal. Redirected output
# uses a stable 100-column fallback. Unwrapped arguments live in the command receipt.
#
# Rail glyphs, disabled markers and km² units use UTF-8, which
# `_enable_utf8_stdout` enables. A Windows locale is cp1252 and a
# redirected stream raises UnicodeEncodeError on anything outside it -- the
# constraint `console_style.rule_banner` still records for the rules, which do
# not reconfigure anything. This runner declares UTF-8 on its own stdout before
# it prints, so `●`, `○` and `│` reach both a console and a `*> log.txt`.
#
# That declaration does NOT extend to the Snakemake children: they inherit the
# handle, not the Python text layer, and keep writing their own bytes to the
# same file. A redirected run is therefore mixed-encoding -- UTF-8 for the
# runner's dozen lines, the locale's codec for everything the workflows say --
# which is legible only because the children's output is itself ASCII. Adding a
# non-ASCII glyph to a RULE's output would break that and belongs behind the
# same reconfiguration, not behind this comment.

_RULE_WIDTH = 80
_RULE = "=" * _RULE_WIDTH

#: The sequence rail: a node per workflow threaded on a vertical line. Filled
#: for a workflow that has completed, hollow for one that is not done (yet, or
# at all), with a dash for disabled steps. The rail replaced a framed box per workflow, which spent
#: three lines and forty columns to say what a node says in one.
_NODE_ON = "●"  # ● -- completed
_NODE_OFF = "○"  # ○ -- pending or failed
_NODE_DISABLED = "–"
_RAIL = "│"  # │ -- the thread between them


def _enable_utf8_stdout() -> None:
    """Declare UTF-8 on this process's stdout, so the rail glyphs can be printed.

    Without this a redirected run dies: the locale codec on Windows is cp1252,
    `print()` on a non-tty encodes through it, and `●` raises
    UnicodeEncodeError before the first workflow starts. One call covers both
    destinations -- a console and a `*> log.txt` -- because it replaces the
    text layer rather than probing what is behind it.

    Fail-open like every other console concern in this file. `reconfigure` is
    absent on a stdout something else already replaced (pytest's capture object
    is the common one), and a runner that refused to start because it could not
    style its own banner would be trading a whole run for a glyph. The banner
    builders stay ASCII apart from the three rail characters, so the fallback
    path loses exactly those and prints everything else.
    """
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    except (AttributeError, OSError, ValueError):
        pass


def _label(name: str) -> str:
    """A workflow's console identity: ``wf0 analyze_climate``."""
    return f"{WORKFLOW_ID[name]} {name}"


def _handoff_label(name: str) -> str:
    """An emphasized workflow identity for a run hand-off heading."""
    return f"{WORKFLOW_ID[name].upper()}  {name.replace('_', ' ').upper()}"


def _clock() -> str:
    """Local wall-clock ``YYYY-MM-DD HH:MM:SS`` for a hand-off line.

    A wall clock, unlike the monotonic durations everywhere else here: this
    answers "when did this start", which someone reads against the timestamps in
    the workflow's own log rows below it.
    """
    return f"{datetime.now():%Y-%m-%d %H:%M:%S}"


def _banner(head: str) -> list[str]:
    """A rule, one head line, a rule -- the wrapper announcing a block."""
    rule = _RULE[: _console_width()]
    return [rule, *_wrap_console(head, initial="  ", continuation="  "), rule]


def _console_width() -> int:
    """Return readable columns, with a stable width when output is redirected."""
    if not sys.stdout.isatty():
        return 100
    return max(40, shutil.get_terminal_size(fallback=(100, 24)).columns)


def _wrap_console(
    text: str, *, initial: str = "", continuation: str = "", width: int | None = None
) -> list[str]:
    """Wrap display text without changing the arguments used for execution."""
    return textwrap.wrap(
        text,
        width=width or _console_width(),
        initial_indent=initial,
        subsequent_indent=continuation,
        break_on_hyphens=False,
        replace_whitespace=False,
    ) or [initial.rstrip()]


def _command_groups(argv: list[str]) -> list[list[str]]:
    """Keep each long option and its values together after redaction."""
    groups: list[list[str]] = [[]]
    for argument in sanitize_argv(argv):
        if argument.startswith("--") and groups[-1]:
            groups.append([])
        groups[-1].append(argument)
    return groups


def _format_command(argv: list[str], *, width: int | None = None) -> str:
    """Display option groups and long paths on aligned continuation lines."""
    lines = []
    for index, group in enumerate(_command_groups(argv)):
        limit = width or _console_width()
        if index and len(" ".join(group)) + 14 > limit:
            lines.append(" " * 14 + group[0])
            lines.extend(
                _wrap_console(
                    " ".join(group[1:]),
                    initial=" " * 18,
                    continuation=" " * 18,
                    width=limit,
                )
            )
            continue
        lines.extend(
            _wrap_console(
                " ".join(group),
                initial=" " * 14 if index else "",
                continuation=" " * 18 if index else " " * 14,
                width=width,
            )
        )
    return "\n".join(lines)


def _copyable_command(argv: list[str], *, powershell: bool) -> str:
    """Quote complete arguments for copying, with valid shell continuations."""

    def quote(argument: str) -> str:
        if not powershell:
            return shlex.quote(argument)
        if argument != "--" and re.fullmatch(r"[\w./:=+\-]+", argument, flags=re.ASCII):
            return argument
        return "'" + argument.replace("'", "''") + "'"

    groups = [
        " ".join(quote(value) for value in group) for group in _command_groups(argv)
    ]
    if powershell:
        groups[0] = "& " + groups[0]
    return (" `\n    " if powershell else " \\\n    ").join(groups)


def _write_command_receipt(
    history_path: Path,
    *,
    config_path: str,
    project_dir: Path,
    flags: Mapping[str, bool],
    cores: int,
    extra: list[str],
    simulation_targets: tuple[str, ...],
) -> Path:
    """Retain copyable owned-runner commands beside this invocation record."""
    powershell = os.name == "nt"
    directory = str(REPO_ROOT)
    location = (
        "Set-Location -LiteralPath '" + directory.replace("'", "''") + "'"
        if powershell
        else "cd " + shlex.quote(directory)
    )
    lines = [
        "# Copy one workflow block; arguments are redacted where needed.",
        location,
    ]
    for name in WORKFLOW_ORDER:
        if not flags[name]:
            continue
        argv = [
            "pixi",
            "run",
            "python",
            "scripts/run_workflow.py",
            name,
            "--config",
            str(Path(config_path).resolve()),
            "--project-dir",
            str(project_dir),
            "--cores",
            str(cores),
        ]
        if name == "simulate_system":
            for target in simulation_targets:
                argv.extend(["--target", target])
        forwarded = [*PER_WORKFLOW_FLAGS[name], *extra]
        if forwarded:
            argv.extend(["--", *forwarded])
        lines.extend(
            ["", f"# {_label(name)}", _copyable_command(argv, powershell=powershell)]
        )
    path = history_path.with_suffix(".commands.txt")
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write("\n".join(lines) + "\n")
    return path


def _handoff(tag: str, detail: str, *, note: str | None = None) -> str:
    """One hand-off band: a rule, the workflow and what just happened to it.

    Ruled rather than boxed, and ruled on the TOP side only. What has to survive
    is that the line is the runner's; a closing rule under a one-line utterance
    doubles the cost to say it twice, and this fires twice per workflow.

    The title and command's first line are flush left under the rule. Long
    options continue on indented lines. A blank line separates the command
    from the title so the title does not read as part of the command.
    """
    lines = [_RULE[: _console_width()], *_wrap_console(f"{tag}  --  {detail}")]
    if note:
        # A blank line parts the title from the long command under it.
        lines.extend(["", note])
    return "\n".join(lines)


def _unusual_exit_diagnosis(exit_code: int | None) -> str | None:
    """A note for an exit code neither this wrapper nor Snakemake's own CLI raises.

    Snakemake's ``cli.py`` entry point only ever calls ``sys.exit(0)`` (success)
    or ``sys.exit(1)`` (a DAG/job failure it reported itself, with its own
    traceback already printed above by the child). Any OTHER code was raised
    by something outside Snakemake's own reporting -- process/job-object
    teardown, an OS-level kill, an interpreter crash after Snakemake had
    already finished -- so the child's own output above may be silent or
    incomplete even though the exit code says something went wrong. Pointing
    at the newest ``.snakemake/log`` file lets a reader tell the two apart: if
    IT ends cleanly, the failure happened after Snakemake was done.
    """
    if exit_code is None or exit_code in (0, 1):
        return None
    note = f"exit code {exit_code} is not one Snakemake's own CLI ever raises (only 0 or 1)"
    log_dir = _REPO_ROOT_PATH / ".snakemake" / "log"
    newest = None
    try:
        newest = max(
            log_dir.glob("*.snakemake.log"),
            key=lambda path: path.stat().st_mtime,
            default=None,
        )
    except OSError:
        newest = None
    if newest is not None:
        newest_display = os.fspath(newest).replace(os.sep, "/")
        note += (
            f"; check {newest_display} for whether Snakemake's own account "
            "ended cleanly despite it -- a clean ending there points to a "
            "process-teardown race rather than a rule failure"
        )
    return note


def _console_block(
    head: str,
    groups: list[tuple[str, list[Any]]],
    *,
    close: bool = False,
    include_banner: bool = True,
) -> str:
    """A banner, then labelled groups of rows.

    A group's rows are either ``(key, value)`` pairs -- aligned on ONE value
    column across the whole block, so the column does not restart at each label
    -- or bare strings, emitted verbatim at the row indent (the sequence
    diagram). A group with no rows is a NOTE: its label carries a sentence
    rather than heading a set of rows, which is how `run_summary` prints the
    one line it has that names no artifact.

    ``close`` appends the trailing rule. Only the CLOSING block sets it: the
    opening block is followed either by a hand-off band or by the closing block,
    both of which open with a rule of their own, so closing it too would print
    two rules separated by one blank line. The final rule is what says the
    runner has finished talking, and it has to mean that.
    """
    pairs = [row for _, rows in groups for row in rows if isinstance(row, tuple)]
    width = max((len(key) for key, _ in pairs), default=0)
    lines = _banner(head) if include_banner else []
    for label, rows in groups:
        lines.extend(["", *_wrap_console(label, initial="  ", continuation="  ")])
        if rows and not any(isinstance(row, tuple) for row in rows):
            # A diagram, not key/value rows: a blank line parts it from its label.
            lines.append("")
        for row in rows:
            if isinstance(row, tuple):
                key, value = row
                prefix = f"    {key.ljust(width)}  "
                lines.extend(
                    _wrap_console(
                        str(value), initial=prefix, continuation=" " * len(prefix)
                    )
                )
            else:
                lines.extend(
                    _wrap_console(str(row), initial="    ", continuation=" " * 14)
                )
    if close:
        lines.extend(["", _RULE[: _console_width()]])
    return "\n".join(lines)


def _sequence_lines(
    flags: Mapping[str, bool], done: frozenset[str] = frozenset()
) -> list[str]:
    """The enabled/disabled pipeline as a rail of nodes, in WORKFLOW_ORDER.

    Every workflow appears, enabled or not, because the question this answers is
    "what is about to happen" and a disabled workflow silently absent from the
    list is indistinguishable from one this wrapper does not know about. Enabled
    entries carry their `[position/total]` so the per-workflow timeline rows
    below can be matched to the plan without counting.

    A compact rail replaces framed boxes: `●` means completed, `○` means
    pending and `–` means disabled. The `│` between them states their order.
    The opening block passes no completed workflows because nothing has run.

    Non-ASCII, which nothing else in this file's output is: see the section
    comment above and `_enable_utf8_stdout`, which is what makes it safe.
    """
    total = sum(1 for name in WORKFLOW_ORDER if flags[name])
    # Width from the widest marker this run can print, so a disabled row's `-`
    # stays centred under an enabled row's position past nine workflows.
    mark_width = max(5, len(f"[{total}/{total}]"))
    lines: list[str] = []
    position = 0
    for name in WORKFLOW_ORDER:
        if lines:
            # The rail sits under the node's own column, not the text's: what it
            # threads is the sequence of nodes.
            lines.append(_RAIL)
        if flags[name]:
            position += 1
            mark = f"[{position}/{total}]".ljust(mark_width)
            node = _NODE_ON if name in done else _NODE_OFF
            lines.append(f"{node}  {mark}  {_label(name)}")
        else:
            mark = "-".center(mark_width)
            lines.append(f"{_NODE_DISABLED}  {mark}  {_label(name)}  disabled")
    return lines


#: Kilometres per degree of latitude (WGS84 meridian arc / 180) and per degree
#: of longitude at the equator. Both are averages: this is the scale bar under a
#: bounding box, printed as `approx`, not a projection.
_KM_PER_DEG_LAT = 111.19
_KM_PER_DEG_LON = 111.32


def _geojson_bbox(path: Path) -> tuple[float, float, float, float] | None:
    """``(lon_min, lat_min, lon_max, lat_max)`` of a GeoJSON, or None.

    Read with the stdlib `json` rather than with geopandas: this runs in the
    wrapper, before any workflow, and a several-second import for four numbers
    would be paid by every invocation including the ones that print them as
    "not delineated yet". A GeoJSON's coordinates are lon/lat by specification
    (RFC 7946 §4), so no CRS handling is needed to read them as degrees -- and
    `_bbox_extent_km` refuses anything outside degree range anyway.
    """
    document = json.loads(path.read_text(encoding="utf-8"))
    declared = document.get("bbox")
    if isinstance(declared, list) and len(declared) >= 4:
        # The file's own answer wins: it is what every other reader of this
        # artifact sees, and a 3D bbox states its planar box in the same first
        # two and last two slots.
        values = [float(v) for v in declared]
        pairs = 3 if len(values) >= 6 else 2
        return values[0], values[1], values[pairs], values[pairs + 1]

    lons: list[float] = []
    lats: list[float] = []

    def collect(node: Any) -> None:
        if not isinstance(node, list):
            return
        if len(node) >= 2 and all(isinstance(v, (int, float)) for v in node[:2]):
            lons.append(float(node[0]))
            lats.append(float(node[1]))
            return
        for item in node:
            collect(item)

    for feature in document.get("features", [document]):
        geometry = feature.get("geometry") or feature
        collect(geometry.get("coordinates", []))
    if not lons:
        return None
    return min(lons), min(lats), max(lons), max(lats)


def _bbox_extent_km(
    bbox: tuple[float, float, float, float],
) -> tuple[float, float] | None:
    """Approximate width and height of ``bbox`` in km, or None if it is not degrees.

    The range guard is the point of the None: a geometry that reached this in a
    projected CRS would otherwise be multiplied by 111 and printed as a
    confident, enormous, wrong number -- in the one row a reader has no way to
    check. Without the extent the bbox itself is still shown, and it is the
    part that carries the units on its face.
    """
    lon_min, lat_min, lon_max, lat_max = bbox
    if not (-180.0 <= lon_min <= lon_max <= 180.0):
        return None
    if not (-90.0 <= lat_min <= lat_max <= 90.0):
        return None
    mean_lat = math.radians((lat_min + lat_max) / 2.0)
    width = (lon_max - lon_min) * _KM_PER_DEG_LON * math.cos(mean_lat)
    height = (lat_max - lat_min) * _KM_PER_DEG_LAT
    return width, height


def _km(value: float) -> str:
    """A distance a reader can compare at a glance, not one they must parse."""
    return f"{value:.1f}" if value < 10 else f"{value:.0f}"


def _section(cfg: Mapping[str, Any], *keys: str) -> Mapping[str, Any]:
    """A nested mapping from the config, or an empty one at the first gap.

    The wrapper's contract validates `workflows:` and nothing else, so every
    key this block reads is genuinely optional -- a settings row that is absent
    from the config is simply not printed, and an alternative spelling of the
    tree (a project mid-migration, a hand-written config) must not be able to
    stop the run before it starts.
    """
    node: Any = cfg
    for key in keys:
        if not isinstance(node, Mapping):
            return {}
        node = node.get(key)
    return node if isinstance(node, Mapping) else {}


def _format_region(region: Any) -> str:
    """Describe a simple subbasin outlet without losing unsupported settings."""
    original = str(region)
    if isinstance(region, str):
        try:
            region = yaml.safe_load(region)
        except yaml.YAMLError:
            return original
    if not isinstance(region, Mapping) or set(region) - {"subbasin", "uparea"}:
        return original
    outlet = region.get("subbasin")
    if not isinstance(outlet, (list, tuple)) or len(outlet) != 2:
        return original
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        for value in outlet
    ):
        return original
    text = f"subbasin outlet ({outlet[0]:.4f}, {outlet[1]:.4f})"
    if "uparea" in region:
        text += f"; uparea >= {region['uparea']} km²"
    return text


def _settings_rows(cfg: Mapping[str, Any], project_dir: Path) -> list[tuple[str, str]]:
    """The basin the run is ABOUT: its region specification and its resolution.

    Assembled into the one `settings` group by `_opening_block`, between the
    project's name and the two paths. Every row is optional -- the wrapper's
    contract validates `workflows:` and nothing else, so a key absent from the
    config is simply not printed and an alternative spelling of the tree cannot
    stop the run before it starts.

    The region's own row states the SPECIFICATION, and a continuation line
    beneath it states the box that specification actually delineated. Two lines
    under one key rather than two keys: `bbox` and `extent` name no setting a
    reader could go and change, and the box is only meaningful as an answer to
    the specification above it.

    That box is DERIVED from `data/spatial/geoms/region.geojson` and never from
    the specification: `{'subbasin': [9.666, 0.4476], 'uparea': 100}` is an
    outlet and an upstream-area threshold, so a box computed from those would be
    an invention in exactly the line a reader is least able to check. Before the
    first delineation there is no box, and the line says so.
    """
    # `basin` at TOP level, which is where R14 put it when it dissolved the
    # `shared:` heading these keys used to sit under (`docs/migration-config-
    # shape.md`). The old spelling did not fail -- `_section` returns `{}` at
    # the first gap, by design -- so every row this function exists to print
    # silently stopped printing instead.
    basin = _section(cfg, "basin")
    rows: list[tuple[str, str]] = []

    region = basin.get("region")
    if region is not None:
        rows.append(("region", _format_region(region)))

    region_path = Path(region_geojson_path(os.fspath(project_dir)))
    bbox = None
    if region_path.is_file():
        try:
            bbox = _geojson_bbox(region_path)
        except (OSError, ValueError, TypeError, KeyError, AttributeError):
            # A region file that cannot be read is a diagnostic for the workflow
            # that reads it for real, not a reason to fail the opening block.
            bbox = None
    if bbox is None:
        if region is not None:
            # Only where a region was ASKED for. A config that declares no basin
            # has nothing absent, and a lone `not delineated yet` would be a
            # line about a question nobody put.
            rows.append(("", "not delineated yet -- the first workflow writes it"))
    else:
        lon_min, lat_min, lon_max, lat_max = bbox
        box = f"lon {lon_min:.4f} .. {lon_max:.4f}, lat {lat_min:.4f} .. {lat_max:.4f}"
        extent = _bbox_extent_km(bbox)
        if extent is not None:
            box += f"  (approx {_km(extent[0])} x {_km(extent[1])} km)"
        # An empty key, so the line lands in the shared VALUE column rather than
        # at the row indent -- it is the row above continuing, not a row of its
        # own, and the alignment is what says so.
        rows.append(("", box))

    resolution = basin.get("resolution")
    if resolution is not None:
        try:
            cell_km = float(resolution) * _KM_PER_DEG_LAT
            rows.append(("resolution", f"{resolution} deg  (approx {_km(cell_km)} km)"))
        except (TypeError, ValueError):
            rows.append(("resolution", str(resolution)))
    return rows


def _toolbox_label(identity: Mapping[str, Any]) -> str:
    """Show the ancestral release and the commit actually used by this run."""
    commit = identity.get("commit")
    if not commit:
        return "version unavailable"
    details = str(commit)[:8]
    if identity.get("dirty") is True:
        details += "; local changes"
    elif identity.get("dirty") is None:
        details += "; working tree unknown"
    return f"{identity.get('release_tag') or 'unversioned'} ({details})"


def _opening_block(
    *,
    cfg: Mapping[str, Any],
    project_name: str,
    project_dir: Path,
    config_path: str,
    dry_run: bool,
    flags: Mapping[str, bool],
    toolbox: Mapping[str, Any],
    experiment: tuple[str, str | None] | None = None,
    include_banner: bool = True,
) -> str:
    """The start-of-invocation block: which run this is, and what will run.

    ONE group, not the `run` / `settings` pair this carried until 2026-08-18.
    Splitting where the values come FROM -- the command line versus the config
    -- is the writer's distinction, not the reader's: both halves answer "which
    run is this", they were adjacent, and each label cost a line to say so.

    `config` prints as the caller SPELLED it (forward-slashed), not resolved:
    a relative `--config` is how every documented invocation passes it, and the
    absolute form is both longer and machine-specific. `folder` is resolved,
    because that one is an answer -- where the outputs land -- rather than an
    echo of the command line.

    `repository` is the resolved checkout this wrapper is actually running
    from (`_REPO_ROOT_PATH`, the parent of `scripts/`), printed for the same
    reason `folder` is: a worktree repo, this one included, is invoked
    identically from the primary checkout or any linked worktree, so the
    console is the only place that distinguishes which tree's code is
    executing without inspecting `sys.argv[0]` or `pwd`.

    `cores` is gone with the split. It is not a property of the project, and
    every hand-off band below prints the `-c N` it was forwarded to, verbatim,
    in the command it is about to run.
    """
    rows: list[Any] = [("project", project_name)]
    if experiment is not None:
        name, action = experiment
        rows.append(("experiment", f"{name} ({action})" if action else name))
    rows.extend(_settings_rows(cfg, project_dir))
    rows.append(("toolbox", _toolbox_label(toolbox)))
    rows.append(("repository", os.fspath(_REPO_ROOT_PATH).replace(os.sep, "/")))
    rows.append(("folder", os.fspath(project_dir).replace(os.sep, "/")))
    rows.append(("config", os.fspath(config_path).replace(os.sep, "/")))
    if dry_run:
        # Kept out of the five above, and kept: a dry run's console is otherwise
        # nearly identical to a real one's, and this is a notice about what will
        # HAPPEN rather than a setting a reader could go and change.
        rows.append(("mode", "dry run -- nothing is executed"))
    enabled = sum(1 for name in WORKFLOW_ORDER if flags[name])
    total = len(WORKFLOW_ORDER)
    groups: list[tuple[str, list[Any]]] = [("settings", rows)]
    if enabled:
        groups.append(
            (
                "sequence",
                list(_sequence_lines(flags)),
            )
        )
    else:
        # One note, not an empty `sequence` group plus a note: there is no
        # order to diagram, and two adjacent labels saying the same nothing
        # read as a formatting accident.
        groups.append(
            (f"nothing to invoke -- 0 of {total} workflows are enabled here", [])
        )
    return _console_block("run_workflows", groups, include_banner=include_banner)


def _closing_block(
    *,
    project_dir: Path,
    manifest_path: Path,
    manifest: Mapping[str, Any],
    ran: list[tuple[str, str]],
    elapsed_seconds: float,
    failed: bool,
    colour: bool = False,
) -> str:
    """The end-of-invocation block: what ran, how long, and where it landed.

    With ``colour``, the FAILED verdict and each FAILED outcome row are painted
    red, the one failure colour on this console.

    `ran` holds only workflows this invocation actually INVOKED, each with its
    already-formatted outcome; everything the stop boundary left behind is named
    in one `not run` note instead. A group headed "ran" listing workflows that
    did not is worse than not printing them.

    `logs` names this execution's directory; workflow details are linked from
    its summary rather than reconstructing individual filenames here. The
    invocation record and command receipt are not listed: they are machine
    files under `config/_engine/`, and the summary is the reader's entry point.
    """
    root = os.fspath(project_dir).replace(os.sep, "/")
    verdict = "FAILED" if failed else "done"
    counted = {
        name: workflow.get("warning_count")
        for name, workflow in manifest["workflows"].items()
        if workflow.get("warning_count")
    }
    total_warnings = sum(counted.values())
    unavailable = [
        name
        for name, workflow in manifest["workflows"].items()
        if "warning_count" in workflow and workflow["warning_count"] is None
    ]
    warnings = (
        f" with {total_warnings} warning{'s' if total_warnings != 1 else ''}"
        if total_warnings
        else ""
    )
    head = f"run_workflows {verdict}{warnings} in {format_elapsed(elapsed_seconds)}"
    if unavailable:
        head += " (warning count incomplete)"
    # `error_type` is set only on the path where a child never LAUNCHED (an
    # OSError out of subprocess.run). Two of the rows below are claims about a
    # child having produced something, and both are false there.
    launched = ran and not manifest.get("error_type")

    groups: list[tuple[str, list[Any]]] = []
    if ran:
        groups.append(("ran", [(_label(name), outcome) for name, outcome in ran]))
    elif manifest["no_op"]:
        groups.append(("nothing ran -- every workflow was disabled", []))
    pending = [
        (_label(name), workflow["pending_reason"])
        for name, workflow in manifest["workflows"].items()
        if workflow.get("pending_reason")
    ]
    if pending:
        groups.append(("pending -- DAG not inspected", pending))

    if counted or unavailable:
        groups.append(
            (
                "warnings",
                [(_label(name), str(count)) for name, count in counted.items()]
                + [(_label(name), "count unavailable") for name in unavailable],
            )
        )

    # `project` defines the `<project>` token the rows below it use, the same
    # spelling the rule logs use (`run_log_core`).
    wrote: list[Any] = [("project", root)]
    if launched:
        directory = manifest.get("execution_directory", "logs")
        wrote.append(("logs", f"<project>/{directory}/"))
    from blueearth_cst.shared.invocation_summary import summary_path

    run_summary_path = summary_path(manifest_path)
    if run_summary_path.is_file():
        wrote.append(("run summary", _under_project(run_summary_path, project_dir)))
    groups.append(("wrote", wrote))

    not_run = [
        name
        for name in WORKFLOW_ORDER
        if manifest["workflows"][name]["status"] == "not_run"
    ]
    if not_run:
        # A sentence, not a row: `key  value` under no label would read as an
        # artifact path like the group above it.
        groups.append(("not run: " + ", ".join(_label(name) for name in not_run), []))
    if failed and launched:
        # "output", not "report": a workflow that failed at PARSE time never
        # reached its own `onerror` hook, so what is above it is a traceback
        # rather than a `run_summary` block. Both are the child's own account
        # of the failure, which is what this points at.
        groups.append(("the failing workflow's own output is printed above", []))
        failed_name = next(
            (
                name
                for name in WORKFLOW_ORDER
                if manifest["workflows"][name]["status"] == "failed"
            ),
            None,
        )
        if failed_name is not None:
            diagnosis = _unusual_exit_diagnosis(
                manifest["workflows"][failed_name]["exit_code"]
            )
            if diagnosis is not None:
                groups.append((diagnosis, []))
    block = _console_block(head, groups, close=True)
    if not colour:
        return block
    return "\n".join(
        _ansi(line, _ANSI_FAIL) if "FAILED" in line else line
        for line in block.split("\n")
    )


def _under_project(path: Path, project_dir: Path) -> str:
    """``path`` as ``<project>/...`` when it sits under ``project_dir``, else whole."""
    try:
        relative = Path(path).resolve().relative_to(Path(project_dir).resolve())
    except ValueError:
        return os.fspath(path).replace(os.sep, "/")
    return "<project>/" + relative.as_posix()


def _project_name(cfg: Mapping[str, Any], project_dir: Path) -> str:
    """The project's display name: the explicit key, else the folder basename.

    Same derivation as `scripts/plot_workflow_dag.py::read_project`, which names
    its renders with it -- one console and one filename disagreeing about what
    the project is called is exactly what a shared rule prevents.

    Top level ONLY, matching that function exactly. A `project.project_name`
    fallback looks like a kindness and is the defect this docstring claims to
    prevent: no schema defines the key there, and a config carrying it would
    make this console say one name while the DAG render filename said the
    folder basename.
    """
    name = cfg.get("project_name")
    return str(name) if name else project_dir.name


def _validate_rebuild(
    config_path: str,
    flags: Mapping[str, bool],
    cores: int,
    extra: list[str],
    simulation_targets: tuple[str, ...],
) -> None:
    """Reject unusable or redirected rebuilds before confirmation or deletion."""
    from blueearth_cst.shared.config_composition import load_composed_config

    if cores < 1 or not any(flags.values()):
        raise ValueError(
            "--rebuild requires positive --cores and at least one enabled workflow"
        )
    # Rebuild always targets this project. Config/directory overrides, arbitrary
    # targets and Snakemake bookkeeping modes must not redirect the post-reset run.
    switches = {
        "--dry-run",
        "--dryrun",
        "-n",
        "--forceall",
        "-F",
        "--printshellcmds",
        "-p",
        "--keep-going",
        "--rerun-incomplete",
        "--notemp",
    }
    values = {
        "--resources",
        "--set-resources",
        "--set-threads",
        "--forcerun",
        "--rerun-triggers",
    }
    index = 0
    while index < len(extra):
        option = extra[index]
        index += 1
        if option in switches:
            continue
        if option not in values:
            raise ValueError(
                f"--rebuild does not support forwarded option or target {option!r}"
            )
        start = index
        while index < len(extra) and not extra[index].startswith("-"):
            index += 1
        if index == start:
            raise ValueError(f"{option} requires a value")
    for name, enabled in flags.items():
        if enabled:
            load_composed_config(config_path, entry=name)
    if flags["simulate_system"]:
        if not flags["build_model"] or not flags["generate_scenarios"]:
            raise ValueError(
                "--rebuild with WF4 requires enabled build_model and generate_scenarios; retained inputs will be erased"
            )
        from blueearth_cst.experiment.simulation_runner import (
            simulation_command,
            simulation_settings,
        )

        _, settings = simulation_settings(config_path)
        if settings["operation"] == "indicators-only":
            raise ValueError(
                "--rebuild cannot use indicators-only; enable simulations as retained responses will be erased"
            )
        simulation_command(
            config_path, list(simulation_targets), cores, extra, prepare_baseline=False
        )


def _run_rebuild(
    config_path: str,
    cores: int,
    extra: list[str],
    *,
    simulation_targets: tuple[str, ...],
    project_dir: Path | None,
    yes: bool,
    note: str | None,
) -> int:
    """Confirm, clear, and run under the existing project execution lock."""
    from blueearth_cst.shared.project_rebuild import erase_outputs, plan_rebuild

    cfg = _read_config(config_path)
    flags = _enabled_flags(cfg, config_path)
    root = _project_dir(cfg, config_path)
    if project_dir is not None and root != project_dir.resolve():
        raise ConfigError("--project-dir differs from composed project root")
    if note is not None and not flags["simulate_system"]:
        raise ValueError("--note requires enabled simulate_system")
    _validate_rebuild(config_path, flags, cores, extra, simulation_targets)
    plan = plan_rebuild(root, Path(config_path))
    preview = any(flag in {"--dry-run", "--dryrun", "-n"} for flag in extra)
    print(
        "Rebuild preview"
        if preview
        else "Rebuild will permanently erase existing generated outputs in:"
    )
    print(f"  {root.as_posix()}")
    for target in plan.targets:
        print(f"  erase  {target.relative_to(root).as_posix()}")
    if not plan.targets:
        print("  No existing generated outputs to erase.")
    print("All existing experiments and their numbering will be reset.")
    print(
        "Source configuration, source datasets, unrelated files, and logs/history will be preserved."
    )
    print("Only enabled workflows will run after the reset.", flush=True)
    if preview:
        print("Preview only: nothing erased; no workflows started.")
        return 0
    if not yes:
        if not sys.stdin.isatty():
            raise ValueError(
                "--rebuild needs interactive confirmation; use --yes for a non-interactive run"
            )
        try:
            answer = input("Continue? [y/N]: ")
        except EOFError:
            answer = ""
        if answer.strip().lower() not in {"y", "yes"}:
            print("Rebuild cancelled; nothing erased.")
            return 0
    history_pair = None
    with archive_lock(root, "project-execution"):
        if plan_rebuild(root, Path(config_path)) != plan:
            raise ValueError(
                "project outputs or source configuration changed; preview and confirm rebuild again"
            )
        history_pair = invocation_history.start(
            root,
            workflow=None,
            entry_point="scripts/run_workflows.py",
            command=[
                "run_workflows",
                "--config",
                config_path,
                "--cores",
                str(cores),
                "--rebuild",
                *(["--yes"] if yes else []),
                *extra,
            ],
            targets=["all"],
            mode="execute",
            contract_mode="new_schema",
            requested_workflows=[name for name, enabled in flags.items() if enabled],
        )
        path, record = history_pair
        record["rebuild"] = {
            "status": "erasing",
            "targets": [target.relative_to(root).as_posix() for target in plan.targets],
        }
        invocation_history.update(path, record)
        try:
            erase_outputs(plan, Path(config_path))
            record["rebuild"]["status"] = "completed"
            invocation_history.update(path, record)
            return _run_owned(
                config_path,
                cores,
                extra,
                simulation_targets=simulation_targets,
                cfg=cfg,
                flags=flags,
                project_dir=root,
                history_pair=history_pair,
                note=note,
            )
        except BaseException as error:
            if record["rebuild"]["status"] != "completed":
                record["rebuild"]["status"] = "failed"
            if record["status"] == "running":
                invocation_history.update(path, record)
                invocation_history.finish(path, record, exit_code=None, error=error)
            raise


@execution_outputs.capture_launch
def run(
    config_path: str,
    cores: int,
    extra: list[str],
    *,
    simulation_targets=("all",),
    project_dir: Path | None = None,
    new_experiment: bool = False,
    note: str | None = None,
    recover_reservation: bool = False,
    rebuild: bool = False,
    yes: bool = False,
) -> int:
    """Hold project ownership and record explicit-root validation failures."""
    if yes and not rebuild:
        raise ValueError("--yes requires --rebuild")
    if rebuild:
        if new_experiment or recover_reservation:
            raise ValueError(
                "--rebuild cannot combine with --new-experiment or --recover-reservation"
            )
        try:
            return _run_rebuild(
                config_path,
                cores,
                extra,
                simulation_targets=tuple(simulation_targets),
                project_dir=project_dir,
                yes=yes,
                note=note,
            )
        except OSError as error:
            raise ValueError(f"rebuild stopped: {error}") from error
    history_pair = None
    try:
        flags = None
        try:
            cfg = _read_config(config_path)
            flags = _enabled_flags(cfg, config_path)
        finally:
            # Name the attempt from its selection when available, while still
            # retaining explicit-root failures that cannot read configuration.
            if project_dir is not None:
                supplied_root = Path(project_dir).resolve()
                history_pair = invocation_history.start(
                    supplied_root,
                    workflow=None,
                    entry_point="scripts/run_workflows.py",
                    command=sanitize_argv(
                        [
                            "run_workflows",
                            "--config",
                            config_path,
                            "--cores",
                            str(cores),
                            *extra,
                        ]
                    ),
                    targets=["all"],
                    mode="dry_run"
                    if any(flag in extra for flag in ("--dry-run", "-n", "--unlock"))
                    else "execute",
                    contract_mode="new_schema",
                    requested_workflows=(
                        [name for name in WORKFLOW_ORDER if flags[name]]
                        if flags is not None
                        else None
                    ),
                )
        resolved_root = _project_dir(cfg, config_path)
        if project_dir is not None and resolved_root != supplied_root:
            raise ConfigError("--project-dir differs from composed project root")
        with archive_lock(resolved_root, "project-execution"):
            return _run_owned(
                config_path,
                cores,
                extra,
                simulation_targets=simulation_targets,
                cfg=cfg,
                flags=flags,
                project_dir=resolved_root,
                history_pair=history_pair,
                new_experiment=new_experiment,
                note=note,
                recover_reservation=recover_reservation,
            )
    except BaseException as error:
        if history_pair is not None and history_pair[1]["status"] == "running":
            invocation_history.finish(
                history_pair[0], history_pair[1], exit_code=None, error=error
            )
        raise


class _PreparationHeartbeat(_Heartbeat):
    """Keep startup progress aligned with the runner's introductory text."""

    def start(self) -> _PreparationHeartbeat:
        if self._interactive:
            self._emit("0:00:00 elapsed", redraw=True)
        else:
            print(f"  {self._label}...", file=self._stream, end="", flush=True)
        return super().start()

    def _tick(self, now: float) -> None:
        if self._interactive:
            super()._tick(now)

    def _emit(
        self, text: str, code: str | None = None, *, redraw: bool = False
    ) -> None:
        if redraw and not self._interactive:
            return
        prefix = (
            f"{_line_reset(self._stream)}  {self._label}..."
            if self._interactive
            else ""
        )
        with suppress(OSError, ValueError):
            print(
                f"{prefix} ({text})",
                file=self._stream,
                end="" if redraw else "\n",
                flush=True,
            )

    def stop(self, failed: bool = False) -> None:
        super().stop(failed=failed)
        if not failed:
            self._emit(f"{format_elapsed(time.monotonic() - self._start)} elapsed")


@contextmanager
def _preparation_progress(enabled: bool):
    """Announce the runner before experiment imports and report long waits."""
    heartbeat = None
    try:
        print("\n" + "\n".join(_banner("run_workflows")), flush=True)
        if enabled:
            print(file=sys.stderr, flush=True)
            heartbeat = _PreparationHeartbeat(
                "Preparing configuration and experiment checks", sys.stderr
            ).start()
    except Exception as exc:  # noqa: BLE001 -- reporting must never prevent admission
        with suppress(OSError, ValueError):
            print(
                f"(preparation output unavailable: {exc})", file=sys.stderr, flush=True
            )
    try:
        yield
    finally:
        if heartbeat is not None:
            try:
                heartbeat.stop()
            except Exception as exc:  # noqa: BLE001 -- preserve the admission outcome
                with suppress(OSError, ValueError):
                    print(
                        f"(preparation status unavailable: {exc})",
                        file=sys.stderr,
                        flush=True,
                    )


@execution_outputs.capture_launch
def _run_owned(
    config_path: str,
    cores: int,
    extra: list[str],
    *,
    simulation_targets: tuple[str, ...],
    cfg: Mapping[str, Any],
    flags: Mapping[str, bool],
    project_dir: Path,
    history_pair: tuple[Path, dict[str, Any]] | None = None,
    new_experiment: bool = False,
    note: str | None = None,
    recover_reservation: bool = False,
) -> int:
    """Invoke each enabled workflow in fixed order; stop on first nonzero exit
    and return that code (contract (d)). Returns 0 if all enabled workflows
    succeed (or all are disabled)."""
    # Contract (i), BEFORE the manifest: a run that cannot start should not
    # mint an invocation record, which exists to describe runs that did.
    manifest_path, manifest = _initialize_manifest(
        cfg=cfg,
        config_path=config_path,
        project_dir=project_dir,
        flags=flags,
        cores=cores,
        extra=extra,
    )
    preview = manifest["dry_run"] or "--unlock" in extra
    history_path, history = history_pair or invocation_history.start(
        project_dir,
        workflow=None,
        entry_point="scripts/run_workflows.py",
        command=sanitize_argv(
            ["run_workflows", "--config", config_path, "--cores", str(cores), *extra]
        ),
        targets=["all"],
        mode="dry_run" if preview else "execute",
        contract_mode="new_schema",
        requested_workflows=[name for name in WORKFLOW_ORDER if flags[name]],
    )
    manifest_path = history_path
    manifest["execution_directory"] = history["execution_directory"]
    history["workflow_selection"] = dict(flags)
    history["workflow_order"] = list(WORKFLOW_ORDER)
    history["requested_workflows"] = [name for name in WORKFLOW_ORDER if flags[name]]
    invocation_history.update(history_path, history)
    reservation_context = None
    try:
        with _preparation_progress(flags["simulate_system"]):
            if not flags["simulate_system"]:
                if new_experiment or note is not None or recover_reservation:
                    raise ValueError("experiment flags require enabled simulate_system")
            else:
                from blueearth_cst.experiment.baseline_qcrit import (
                    HistoricalQcritUnavailable,
                )
                from blueearth_cst.experiment.reservation_context import (
                    admit_context_owned,
                )
                from blueearth_cst.experiment.simulation_runner import (
                    simulation_command,
                )

                simulation_command(
                    config_path,
                    list(simulation_targets),
                    cores,
                    extra,
                    prepare_baseline=False,
                )
                reservation_context = admit_context_owned(
                    Path(config_path),
                    REPO_ROOT,
                    project_dir,
                    history["invocation_id"],
                    execute_generation=flags["generate_scenarios"],
                    execute_model=flags["build_model"],
                    new_experiment=new_experiment,
                    note=note,
                    recover=recover_reservation,
                    preview=preview,
                    announce=False,
                )
    except BaseException as exc:
        invocation_history.finish(history_path, history, exit_code=None, error=exc)
        raise
    history["requested_workflows"] = [name for name in WORKFLOW_ORDER if flags[name]]
    history["configuration"].update(
        source_config_sha256=manifest["source_config"]["sha256"],
        effective_config_sha256=manifest["effective_config"]["sha256"],
    )
    if manifest["no_op"]:
        history["work_performed"] = "no"
    invocation_history.update(history_path, history)

    if not manifest["no_op"]:
        try:
            receipt_path = _write_command_receipt(
                history_path,
                config_path=config_path,
                project_dir=project_dir,
                flags=flags,
                cores=cores,
                extra=extra,
                simulation_targets=simulation_targets,
            )
            manifest["command_receipt"] = str(receipt_path)
            history["command_receipt"] = file_reference(
                receipt_path, "project_root", project_dir
            )
            invocation_history.update(history_path, history)
        except OSError as exc:
            print(
                f"(copyable commands unavailable: {exc})", file=sys.stderr, flush=True
            )

    # Every hand-off band below names the workflow it is about to start, so the
    # workflow's own opening block does not print its name a second time three
    # lines later. Handed to each CHILD rather than set on this process's own
    # environment: `os.environ[...] = "1"` covered both spawn paths in one line
    # and leaked, because nothing ever unsets it -- inside a test session that
    # is one `run()` call silencing the title for every later test in the
    # process, which is how the full suite found it. The variable describes the
    # child's console, so it belongs on the child's environment.
    child_env = {**os.environ, console_style.ANNOUNCED_ENV: "1"}

    # monotonic, matching each Snakefile's own `_RUN_STARTED`: a wall clock can
    # step backwards mid-run and these are durations, never timestamps.
    started = time.monotonic()
    try:
        # The leading blank line parts the banner from the shell prompt above it.
        print(
            "\n"
            + _opening_block(
                cfg=cfg,
                project_name=_project_name(cfg, project_dir),
                project_dir=project_dir,
                config_path=config_path,
                dry_run=manifest["dry_run"],
                flags=flags,
                toolbox=manifest["git"],
                experiment=(reservation_context.name, reservation_context.action)
                if reservation_context is not None
                else None,
                include_banner=False,
            ),
            flush=True,
        )
    except Exception as exc:  # noqa: BLE001 -- never break a run over a banner
        # Nested, for the reason given at the summary site below: sys.stderr
        # may be exactly what failed, and a raise here would end the run before
        # a single workflow was invoked.
        try:
            print(f"(run header unavailable: {exc})", file=sys.stderr, flush=True)
        except Exception:  # noqa: BLE001
            pass

    total = sum(1 for name in WORKFLOW_ORDER if flags[name])
    ran: list[tuple[str, str]] = []
    position = 0
    exit_code = 0
    try:
        for name in WORKFLOW_ORDER:
            workflow = manifest["workflows"][name]
            if not flags[name]:
                # No row: the opening block's sequence diagram already named
                # every disabled workflow, once, before anything started.
                continue
            if reservation_context is not None:
                reservation_context.recheck()
            position += 1
            tag = f"[{position}/{total}]  {_handoff_label(name)}"
            cmd = build_command(name, config_path, cores, extra)
            if name == "simulate_system":
                # `build_command` gives simulate_system's own Python entry
                # point, which the banner would otherwise show verbatim
                # instead of the snakemake invocation it launches. Re-derive
                # the real command at the WF4 boundary. Preparation errors must
                # surface here, including corrupt existing historical inputs.
                from blueearth_cst.experiment.simulation_runner import (
                    simulation_command,
                )

                try:
                    cmd, _ = simulation_command(
                        config_path, list(simulation_targets), cores, extra
                    )
                except HistoricalQcritUnavailable as exc:
                    if not (manifest["dry_run"] and flags["build_model"]):
                        raise
                    reason = (
                        f"WF4 historical Q90 preparation pending WF1: {exc} "
                        "WF4's data-dependent indicator DAG cannot be inspected until "
                        "the historical inputs exist; no threshold or calculation identity was created."
                    )
                    workflow["status"] = "pending"
                    workflow["pending_reason"] = reason
                    history.setdefault("pending_workflows", {})[name] = reason
                    invocation_history.update(history_path, history)
                    print(
                        "\n"
                        + _handoff(tag, "pending WF1 historical inputs", note=reason),
                        flush=True,
                    )
                    continue
            workflow["command"] = sanitize_argv(cmd)
            workflow["status"] = "running"
            print(
                "\n"
                + _handoff(
                    tag,
                    f"starting at {_clock()}",
                    note=_format_command(cmd),
                )
                + "\n",
            )
            # Before the child, not after: see contract (h) on buffering.
            sys.stdout.flush()
            workflow_started = time.monotonic()
            child_id = uuid.uuid4().hex
            child_path = history_path.with_name(f"{child_id}.json")
            child_record = None
            capture_context = None
            observation = None
            wf3_source_hash = None
            warning_path = history_path.parent / f".{child_id}.warnings"
            try:
                with warning_path.open("x", encoding="utf-8"):
                    pass
                tracking_warnings = True
            except OSError:
                tracking_warnings = False
            previous_tally = os.environ.get(RUN_WARNING_TALLY_ENV)
            os.environ[RUN_WARNING_TALLY_ENV] = str(warning_path)
            try:
                if name == "simulate_system":
                    from blueearth_cst.experiment.simulation_record import (
                        capture_simulation_sources_v2,
                    )
                    from blueearth_cst.experiment.simulation_runner import (
                        current_targets,
                        simulation_command,
                        simulation_settings,
                    )

                    _, settings = simulation_settings(config_path)
                    workflow["operation"] = settings["operation"]
                    if (
                        "--unlock" not in extra
                        and settings["operation"] == "simulate-and-indicators"
                    ):
                        _check_wf1_leaves(flags, project_dir, config_path)
                    cmd, environment = simulation_command(
                        config_path,
                        list(simulation_targets),
                        cores,
                        extra,
                        snapshot_path=history_path.with_name(
                            f"{child_id}.indicator-selection"
                        ),
                    )
                    if reservation_context is not None:
                        if not preview:
                            from blueearth_cst.experiment.simulation_runner import (
                                validate_retained_indicator_sets,
                            )

                            validate_retained_indicator_sets(
                                project_dir / "experiments" / reservation_context.name,
                                cmd,
                            )
                            reservation_context.bind_selection()
                        environment.update(reservation_context.environment())
                    workflow["command"] = sanitize_argv(cmd)
                    child_path, child_record = invocation_history.start(
                        project_dir,
                        workflow=name,
                        entry_point="scripts/run_workflows.py",
                        command=sanitize_argv(cmd),
                        targets=list(simulation_targets),
                        mode="dry_run" if preview else "execute",
                        contract_mode="legacy_wf4_interim",
                        invocation_id=child_id,
                        parent_invocation_id=history["invocation_id"],
                    )
                    _link_child(history_path, history, child_path, name, child_id)
                    if (
                        not preview
                        and settings["operation"] == "simulate-and-indicators"
                    ):
                        capture = capture_simulation_sources_v2(
                            config_path, project_dir, child_id
                        )
                        child_record["configuration"]["archive_state"] = "pending"
                        child_record["configuration"]["source_capture"] = str(capture)
                    else:
                        child_record["configuration"]["archive_state"] = (
                            "not_applicable"
                        )
                    invocation_history.update(child_path, child_record)
                    from blueearth_cst.shared import resource_tracking

                    if child_record.get("resources", {}).get("directory"):
                        environment[resource_tracking.DIRECTORY_ENV] = child_record[
                            "resources"
                        ]["directory"]
                    resource_tracking.phase(child_path, "snakemake")
                    # The simulation runner builds its own environment; the
                    # announcement rides on top of it rather than replacing it.
                    result = subprocess.CompletedProcess(
                        cmd,
                        run_project_child(
                            cmd,
                            cwd=REPO_ROOT,
                            writing=not preview,
                            env={
                                **environment,
                                **execution_outputs.environment(
                                    child_path, child_record
                                ),
                                "CST_SIMULATION_INVOCATION_ID": child_id,
                                "CST_SIMULATION_FINALIZE": "1",
                                console_style.ANNOUNCED_ENV: "1",
                                invocation_history.PARENT_ENV: history["invocation_id"],
                                invocation_history.INVOCATION_ENV: child_id,
                                RUN_WARNING_TALLY_ENV: str(warning_path),
                            },
                        ),
                    )
                    resource_tracking.phase(child_path, "finalization")
                    snapshot = environment.get("CST_INDICATOR_SELECTION")
                    if (
                        not preview
                        and snapshot
                        and reservation_context is not None
                        and current_targets(list(simulation_targets))
                        != ["simulations_only"]
                    ):
                        from blueearth_cst.experiment.indicator_export import (
                            publish_selected_sets,
                            report_publication,
                        )
                        from blueearth_cst.experiment.indicator_sets import (
                            read_selection_snapshot,
                        )

                        selection = read_selection_snapshot(snapshot)
                        if selection.indicators:
                            wf4_project, _ = simulation_settings(config_path)
                            experiment_root = (
                                project_dir / "experiments" / reservation_context.name
                            )
                            resource_tracking.phase(child_path, "indicator_exports")
                            outcomes = publish_selected_sets(
                                experiment_root, wf4_project, settings, selection
                            )
                            child_record["indicator_set_outcomes"] = outcomes
                            if (
                                not report_publication(experiment_root, outcomes)
                                and result.returncode == 0
                            ):
                                result = subprocess.CompletedProcess(cmd, 1)
                else:
                    env_for_child = {
                        **child_env,
                        RUN_WARNING_TALLY_ENV: str(warning_path),
                    }
                    if name == "generate_scenarios":
                        cmd = [
                            sys.executable,
                            "scripts/generate_scenarios.py",
                            "--config",
                            str(config_path),
                            "--project-dir",
                            str(project_dir),
                            "--cores",
                            str(cores),
                            "--",
                            *extra,
                        ]
                        workflow["command"] = sanitize_argv(cmd)
                    if (
                        name
                        in {"analyze_climate", "build_model", "analyze_projections"}
                        and not preview
                    ):
                        from blueearth_cst.shared.workflow_archive_launch import (
                            CONTEXT_ENV,
                            prepare_workflow,
                            split_config_overrides,
                        )

                        overrides, forwarded = split_config_overrides(extra)
                        cmd = build_command(name, config_path, cores, forwarded)
                        execution_config, capture_context = prepare_workflow(
                            name,
                            Path(config_path),
                            project_dir,
                            command=cmd,
                            targets=["all"],
                            overrides=overrides,
                            entry_point="scripts/run_workflows.py",
                            invocation_id=child_id,
                        )
                        cmd[cmd.index("--configfile") + 1] = str(execution_config)
                        env_for_child = {
                            **env_for_child,
                            CONTEXT_ENV: str(capture_context),
                        }
                        workflow["command"] = sanitize_argv(cmd)
                    child_path, child_record = invocation_history.start(
                        project_dir,
                        workflow=name,
                        entry_point="scripts/run_workflows.py",
                        command=sanitize_argv(cmd),
                        targets=["all"],
                        mode="dry_run" if preview else "execute",
                        contract_mode="new_schema",
                        invocation_id=child_id,
                        parent_invocation_id=history["invocation_id"],
                    )
                    if wf3_source_hash is not None:
                        child_record["configuration"]["source_config_sha256"] = (
                            wf3_source_hash
                        )
                        invocation_history.update(child_path, child_record)
                    if (
                        name
                        in {"analyze_climate", "build_model", "analyze_projections"}
                        and not preview
                    ):
                        _bind_workflow_archive(
                            child_path, child_record, project_dir, name
                        )
                        observation = product_record.observe(name, project_dir)
                    _link_child(history_path, history, child_path, name, child_id)
                    if name == "generate_scenarios":
                        from scripts.generate_scenarios import run_owned

                        result = subprocess.CompletedProcess(
                            cmd,
                            run_owned(
                                Path(config_path),
                                project_dir,
                                cores,
                                extra,
                                invocation_id=child_id,
                                history_pair=(child_path, child_record),
                                reservation_context=reservation_context,
                            ),
                        )
                    else:
                        env_for_child.update(
                            execution_outputs.environment(child_path, child_record)
                        )
                        result = subprocess.CompletedProcess(
                            cmd,
                            run_project_child(
                                cmd,
                                cwd=REPO_ROOT,
                                env=env_for_child,
                                writing=not preview,
                            ),
                        )
            except BaseException as exc:
                if child_record is not None:
                    invocation_history.finish(
                        child_path,
                        child_record,
                        exit_code=None,
                        error=exc,
                        elapsed_seconds=time.monotonic() - workflow_started,
                    )
                    product_record.publish(observation, child_path, child_record)
                    _link_child(history_path, history, child_path, name, child_id)
                else:
                    history["launch_failures"].append(
                        {
                            "workflow": name,
                            "attempted_child_id": child_id,
                            "phase": "launch",
                            "error": {"type": type(exc).__name__, "message": str(exc)},
                            "exit_code": None,
                            "elapsed_seconds": time.monotonic() - workflow_started,
                        }
                    )
                    invocation_history.update(history_path, history)
                elapsed = format_elapsed(time.monotonic() - workflow_started)
                ran.append((name, f"FAILED ({type(exc).__name__}) after {elapsed}"))
                raise
            finally:
                if capture_context is not None:
                    from blueearth_cst.shared.configuration_archive import (
                        remove_launch_context,
                    )

                    remove_launch_context(project_dir, capture_context)
                if previous_tally is None:
                    os.environ.pop(RUN_WARNING_TALLY_ENV, None)
                else:
                    os.environ[RUN_WARNING_TALLY_ENV] = previous_tally
                try:
                    with warning_path.open(encoding="utf-8") as handle:
                        workflow["warning_count"] = sum(
                            1 for line in handle if line.strip()
                        )
                except FileNotFoundError:
                    workflow["warning_count"] = 0 if tracking_warnings else None
                except OSError:
                    workflow["warning_count"] = None
                history.setdefault("warning_counts", {})[name] = workflow[
                    "warning_count"
                ]
                try:
                    warning_path.unlink(missing_ok=True)
                except OSError:
                    # Unique invocation paths cannot pollute a later run's count.
                    pass
            elapsed = format_elapsed(time.monotonic() - workflow_started)
            if child_record is not None:
                child_record["warning_count"] = workflow["warning_count"]
                invocation_history.finish(
                    child_path,
                    child_record,
                    exit_code=result.returncode,
                    elapsed_seconds=time.monotonic() - workflow_started,
                )
                product_record.publish(observation, child_path, child_record)
            if child_path.exists():
                _link_child(history_path, history, child_path, name, child_id)
            else:
                history["launch_failures"].append(
                    {
                        "workflow": name,
                        "attempted_child_id": child_id,
                        "phase": "child_start",
                        "error": {
                            "type": "MissingChildRecord",
                            "message": "child returned without invocation record",
                        },
                        "exit_code": result.returncode,
                    }
                )
                invocation_history.update(history_path, history)
            if name == "simulate_system" and not preview:
                # Publication is part of WF4; only its owner can issue a verdict.
                sys.stdout.flush()
                parts = (
                    project_dir
                    / child_record.get("execution_directory", "logs")
                    / "_parts"
                    / f"wf4-{child_id}"
                    / "logs"
                )
                sys.stderr.write(
                    "\n"
                    + console_style.run_summary(
                        "wf4 simulate_system",
                        project_dir,
                        None,
                        None,
                        elapsed_seconds=time.monotonic() - workflow_started,
                        failed=result.returncode != 0,
                        log_parts_dir=str(parts),
                        warnings=workflow["warning_count"],
                    )
                    + "\n"
                )
                if result.returncode == 0:
                    sys.stderr.write(f"{datetime.now():%H:%M:%S} - DONE all\n")
                sys.stderr.flush()
            workflow["exit_code"] = result.returncode
            if result.returncode != 0:
                workflow["status"] = "failed"
                exit_code = result.returncode
                ran.append((name, f"FAILED (exit {result.returncode}) after {elapsed}"))
                print(
                    "\n"
                    + _handoff(
                        tag,
                        f"FAILED (exit {result.returncode}) after {elapsed}",
                        note="stopping; later workflows not invoked",
                    ),
                    flush=True,
                )
                break
            workflow["status"] = "succeeded"
            ran.append((name, elapsed))
    except BaseException as exc:
        manifest["status"] = "failed"
        manifest["error_type"] = type(exc).__name__
        for workflow in manifest["workflows"].values():
            if workflow["status"] == "running":
                workflow["status"] = "failed"
        _mark_pending_not_run(manifest)
        invocation_history.finish(history_path, history, exit_code=None, error=exc)
        # Before the re-raise, so a launch failure closes with a report rather
        # than with a traceback and nothing else -- the same reason the manifest
        # is finalized on this path.
        _report(
            project_dir=project_dir,
            manifest_path=manifest_path,
            manifest=manifest,
            ran=ran,
            elapsed_seconds=history.get("elapsed_seconds")
            or time.monotonic() - started,
            failed=True,
        )
        raise

    if exit_code != 0:
        manifest["status"] = "failed"
        _mark_pending_not_run(manifest)
    else:
        manifest["status"] = "succeeded"
    invocation_history.finish(history_path, history, exit_code=exit_code)
    _report(
        project_dir=project_dir,
        manifest_path=manifest_path,
        manifest=manifest,
        ran=ran,
        elapsed_seconds=history.get("elapsed_seconds") or time.monotonic() - started,
        failed=exit_code != 0,
    )
    return exit_code


def _report(**kwargs: Any) -> None:
    """Print the closing block, never letting it break the invocation.

    Fail-open like every other banner in this toolbox (`install_console_style`,
    each Snakefile's `_header`): a cosmetic layer must not be able to turn a
    successful run into a failed one, and this one runs on the path that is
    already handling an exception.
    """
    try:
        block = _closing_block(**kwargs, colour=_console_colour(sys.stdout))
        print("\n" + block, flush=True)
    except Exception as exc:  # noqa: BLE001 -- never break a run over a banner
        # Nested, because sys.stderr may be exactly what failed above. An
        # OSError escaping here would replace the wrapper's own exit code with
        # a traceback about the banner, masking the run it was summarizing --
        # the same defect 32e506c fixed in the four Snakefiles.
        try:
            print(f"(run summary unavailable: {exc})", file=sys.stderr, flush=True)
        except Exception:  # noqa: BLE001
            pass


def _link_child(
    parent_path: Path,
    parent: dict[str, Any],
    child_path: Path,
    workflow: str,
    child_id: str,
) -> None:
    """Link only an observed child record, refreshing its changing checksum."""
    if not child_path.is_file():
        return
    item = {
        "invocation_id": child_id,
        "workflow": workflow,
        "record": file_reference(child_path, "project_root", parent_path.parents[3]),
    }
    children = parent["children"]
    for index, child in enumerate(children):
        if child["invocation_id"] == child_id:
            children[index] = item
            break
    else:
        children.append(item)
    invocation_history.update(parent_path, parent)


def _bind_workflow_archive(
    child_path: Path, child: dict[str, Any], project_dir: Path, workflow: str
) -> None:
    """Bind a WF0--WF2 child to its immutable configuration generation."""
    from blueearth_cst.shared.configuration_archive import retain_archive

    archive = read_archive(project_dir, workflow, f"config/{workflow}")
    projection = archive["configuration_projection"]
    child["configuration"].update(
        source_config_sha256=archive["source_files"][0]["sha256"],
        effective_config_sha256=projection["effective_config_sha256"],
        configuration_inputs_sha256=projection["configuration_inputs_sha256"],
        run_record=file_reference(
            retain_archive(project_dir, project_dir / "config" / workflow),
            "project_root",
            project_dir,
        ),
        archive_state="retained",
    )
    invocation_history.update(child_path, child)


def _capture_legacy_wf3_attempt(
    config_path: Path, project_dir: Path, invocation_id: str
) -> str:
    """Retain exact WF3 config source bytes before legacy Snakemake parsing."""
    sources = capture_configuration_sources(
        config_path,
        "generate_scenarios",
        ("project", "basin", "climate", "workflows.generate_scenarios"),
    )
    directory = (
        project_dir / "config/_engine/invocations" / invocation_id / "config/sources"
    )
    for source in sources:
        destination = directory / source.id / source.original_path.name
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("xb") as handle:
            handle.write(source.data)
            handle.flush()
            os.fsync(handle.fileno())
    return hashlib.sha256(sources[0].data).hexdigest()


def sanitize_argv(argv: list[str]) -> list[str]:
    """Redact values attached to credential-like flags or assignments."""
    sanitized: list[str] = []
    redact_next = False
    for value in argv:
        if redact_next:
            sanitized.append("<redacted>")
            redact_next = False
            continue
        if value.startswith("-"):
            flag, separator, assignment = value.partition("=")
            if _is_sensitive_key(flag.lstrip("-")):
                if separator:
                    sanitized.append(f"{flag}=<redacted>")
                else:
                    sanitized.append(value)
                    redact_next = True
                continue
        key, separator, _ = value.partition("=")
        if separator and _is_sensitive_key(key):
            sanitized.append(f"{key}=<redacted>")
        else:
            sanitized.append(value)
    return sanitized


def _is_sensitive_key(key: str) -> bool:
    """Return whether a flag or assignment key may carry a secret."""
    return bool(_SENSITIVE_KEY_RE.search(key))


def _config_overrides(extra: list[str]) -> list[str]:
    """Extract sanitized Snakemake ``--config`` assignments for disclosure."""
    overrides: list[str] = []
    is_config = False
    for value in extra:
        if value == "--config":
            is_config = True
            continue
        if value.startswith("--config="):
            overrides.extend(sanitize_argv([value.removeprefix("--config=")]))
            is_config = False
            continue
        if value.startswith("-"):
            is_config = False
            continue
        if is_config:
            overrides.extend(sanitize_argv([value]))
    return overrides


def _initialize_manifest(
    *,
    cfg: Mapping[str, Any],
    config_path: str,
    project_dir: Path,
    flags: Mapping[str, bool],
    cores: int,
    extra: list[str],
) -> tuple[Path, dict[str, Any]]:
    """Create an in-memory initial invocation record and its unique path."""
    started_at = _utc_now()
    # Under `config/`, NOT a `provenance/` root of its own (R9 follow-up,
    # ruled 2026-08-05). The migration map's Finding 1 disqualified `logs/` for
    # the config snapshot because logs are what a user deletes to reclaim space
    # and their parts are merged-then-deleted by design, while the snapshot is
    # immutable and retained. This manifest is immutable and retained for the
    # same reasons, so the same reasoning places it here. `invocations/` is a
    # SIBLING of the `<workflow>/<digest>/` bundles rather than a fourth
    # workflow entry: an invocation spans workflows.
    runs_dir = project_dir / "config" / "_engine" / "invocations"
    runs_dir.mkdir(parents=True, exist_ok=True)
    filename_stamp = started_at.replace("-", "").replace(":", "")
    filename = f"{filename_stamp}-{uuid.uuid4().hex[:12]}.json"
    source_path = Path(config_path).expanduser().resolve()
    overrides = _config_overrides(extra)
    workflows = {
        name: {
            "enabled": flags[name],
            "status": "pending" if flags[name] else "disabled",
            "command": (
                sanitize_argv(build_command(name, config_path, cores, extra))
                if flags[name]
                else None
            ),
            "exit_code": None,
        }
        for name in WORKFLOW_ORDER
    }
    manifest = {
        "schema_version": 1,
        "started_at_utc": started_at,
        "ended_at_utc": None,
        "status": "running",
        "exit_code": None,
        "source_config": {
            "path": str(source_path),
            "sha256": file_sha256(source_path),
        },
        "effective_config": {
            # projection=None: the wrapper spans all three workflows, so no
            # single workflow's consumed-key projection is the right scope
            # here. A Snakefile passes its own; this manifest keeps the whole
            # config, which is what its "scope" field has always declared.
            "sha256": effective_config_digest(cfg, ADVANCED_SETTINGS, None),
            "scope": "source_config_plus_resolved_advanced_settings",
            "includes_cli_config_overrides": False,
        },
        "snakemake_config_overrides": overrides,
        "argv": sanitize_argv(
            ["--config", config_path, "--cores", str(cores), "--", *extra]
        ),
        "extra_args": sanitize_argv(extra),
        "cores": cores,
        "dry_run": "--dry-run" in extra or "-n" in extra,
        "no_op": not any(flags.values()),
        "workflows": workflows,
        "git": toolbox_identity(),
        "environment_files": environment_file_hashes(),
        "runtime": _runtime_versions(),
    }
    manifest["git"]["release_tag"] = toolbox_release_tag(manifest["git"]["commit"])
    return runs_dir / filename, manifest


def _mark_pending_not_run(manifest: dict[str, Any]) -> None:
    """Mark enabled workflows skipped after an earlier failure."""
    for workflow in manifest["workflows"].values():
        if workflow["status"] == "pending":
            workflow["status"] = "not_run"


def _finalize_manifest(
    path: Path, manifest: dict[str, Any], exit_code: int | None
) -> None:
    """Atomically replace an initial manifest with its terminal record."""
    manifest["ended_at_utc"] = _utc_now()
    manifest["exit_code"] = exit_code
    _write_json_atomic(path, manifest)


def _write_json_atomic(path: Path, document: Mapping[str, Any]) -> None:
    """Write deterministic JSON via same-directory atomic replacement."""
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding="utf-8", newline="\n") as handle:
            json.dump(document, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _utc_now() -> str:
    """Return a millisecond UTC timestamp in ISO 8601 ``Z`` form."""
    return (
        datetime.now(timezone.utc)
        .isoformat(timespec="milliseconds")
        .replace("+00:00", "Z")
    )


# Toolbox and environment identity moved to blueearth_cst.shared.provenance
# (imported above). They were defined here first, but a Snakefile now needs the
# same answers, and a second definition is how the two drift -- the wrapper
# saying one thing about the checkout while the run record says another.
#
# toolbox_identity() is not a rename of _git_metadata(): it adds commit_source
# and a baked-commit fallback, so a container run -- which has no .git -- can
# report its revision instead of a bare null.


def _runtime_versions() -> dict[str, str | None]:
    """Return runtime versions available without launching another process."""
    try:
        snakemake_version = importlib.metadata.version("snakemake")
    except importlib.metadata.PackageNotFoundError:
        snakemake_version = None
    return {"python": platform.python_version(), "snakemake": snakemake_version}


def main(argv: list[str] | None = None) -> int:
    # Before argparse, so `--help` and an argument error are printed through
    # the same text layer the run will use.
    _enable_utf8_stdout()
    ap = argparse.ArgumentParser(
        description="Run the enabled CST workflows in fixed order.",
    )
    ap.add_argument(
        "--config",
        required=True,
        help="path to a full-orchestration project config (see test_case/ for examples)",
    )
    ap.add_argument(
        "--project-dir", type=Path, help="project root for complete startup history"
    )
    ap.add_argument(
        "--cores",
        type=int,
        default=3,
        help="cores forwarded to every snakemake invocation (default: 3)",
    )
    ap.add_argument(
        "--simulation-target",
        nargs="+",
        default=["all"],
        help="explicit WF4 targets for a legacy operation config (indicators-only requires "
        "indicators or one selected indicator-set file); operations/indicator_sets infer the target",
    )
    ap.add_argument(
        "extra",
        nargs=argparse.REMAINDER,
        help="args after `--` are appended verbatim to every invocation",
    )
    ap.add_argument("--new-experiment", action="store_true")
    ap.add_argument("--note")
    ap.add_argument("--recover-reservation", action="store_true")
    ap.add_argument(
        "--rebuild",
        action="store_true",
        help="permanently erase generated project outputs, then run enabled workflows; prompts unless --yes",
    )
    ap.add_argument(
        "--yes",
        action="store_true",
        help="confirm --rebuild without an interactive prompt",
    )
    args = ap.parse_args(argv)

    # argparse.REMAINDER captures the leading `--` sentinel; strip it.
    extra = args.extra
    if extra and extra[0] == "--":
        extra = extra[1:]

    try:
        return run(
            args.config,
            args.cores,
            extra,
            simulation_targets=args.simulation_target,
            project_dir=args.project_dir,
            new_experiment=args.new_experiment,
            note=args.note,
            recover_reservation=args.recover_reservation,
            rebuild=args.rebuild,
            yes=args.yes,
        )
    except (ConfigError, PrerequisiteError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
