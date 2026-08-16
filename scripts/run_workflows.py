"""Enabled-aware wrapper over the four CST Snakefiles (design §7).

Reads a full-orchestration `--configfile` YAML, checks each
`workflows.<name>.enabled` flag, and invokes `snakemake -s <name>.smk
--configfile <cfg> ...` for exactly the enabled workflows, in the fixed order
analyze_climate -> build_model -> analyze_projections -> run_stress_test.

This is the evolution of the run_snake_test.cmd / run_snake_docker.sh runners --
a *runner over* the four Snakefiles, NOT a fifth Snakemake entry point. The
Snakefiles do not read `enabled:`; the flag governs this wrapper only.

Contract (pinned, design §7 (a)-(g), plus (h) for the console):

 (a) Full-orchestration configs only: a `workflows:` section with all FOUR
     subsections each carrying an `enabled:` key. The single-workflow
     projections configs (no `workflows:` section) are direct `snakemake -s`
     inputs, not wrapper inputs.

     The set widened from three to four on 2026-08-14 with `analyze_climate`,
     and it stayed a CLOSED, all-required set deliberately. Treating an absent
     subsection as "disabled" would have let existing three-workflow configs
     keep working untouched, and was rejected: the hazard clause (b) exists to
     prevent is SILENCE, not polarity. Under that rule a section misspelled
     `analyse_climate`, or dropped during an edit, skips a workflow while the
     wrapper exits 0 -- which is the same class of failure as the silent
     default-to-true, arriving from the other direction.
 (b) A missing `workflows:` section or a missing `<name>.enabled` subkey is a
     HARD ERROR (nonzero exit, message naming the absent key) -- never a silent
     default to true, and never a silent default to false.
 (c) Each `enabled:` value must PARSE to a real boolean (isinstance(v, bool) on
     the post-yaml.safe_load value). YAML 1.1 resolves unquoted
     true/false/yes/no/on/off to booleans, so all those spellings are accepted;
     quoted strings ("true"), integers (1/0), or any non-bool are REJECTED.
 (d) Enabled workflows run in fixed order; on the first nonzero snakemake exit
     the wrapper STOPS and returns that exit code (does not continue).
 (e) --cores N (default 3) is forwarded to every invocation; args after a `--`
     sentinel are appended verbatim to every invocation. --configfile is
     supplied by the wrapper.
 (f) Per-workflow flags are preserved from a hardcoded map matching the runners:
     --keep-going on analyze_projections only.
 (g) Every valid wrapper invocation creates and atomically finalizes a unique
     `<project_dir>/config/runs/invocations/*.json` lifecycle manifest -- a
     SIBLING of the per-workflow `config/runs/<workflow>/<digest>/` bundles,
     because an invocation spans workflows. Its runner-side config digest
     covers the source YAML plus resolved advanced settings;
     passthrough `--config` overrides are recorded but intentionally excluded,
     because the Snakefile snapshot owns Snakemake's authoritative merged config.
 (h) The wrapper narrates its own run on stdout, and every utterance is
     bounded by a full-width `=` rule (see the Console section for the grammar
     and why it is the one it is). Nothing else in this console draws one, so a
     rule means the RUNNER is speaking rather than the workflow it launched:

     * an OPENING block -- rule, `run_workflows`, rule, then the `run` group
       (project, folder, config, cores, and `mode` on a dry run) and an ASCII
       `sequence` diagram numbering the enabled workflows in invocation order
       and marking the disabled ones in place;
     * one HAND-OFF band per invoked workflow at each of its two edges --
       `<rule>` then `[1/4]  wf0 analyze_climate  --  starting HH:MM:SS`, with
       the sanitized command below it, and later the same tag with
       `done in <h:mm:ss>` or `FAILED (exit N) after <h:mm:ss>`;
     * a CLOSING block, framed like the opening one and terminated by a final
       rule: the verdict and total elapsed, each invoked workflow's duration,
       and the paths written.

     Deliberately NOT `log_row`'s `HH:MM:SS - <module> - ...`: that grammar is
     worn by every line reported from inside a workflow, so a runner wearing it
     reads as one more rule in the run it supervises. The wrapper therefore also
     ignores `CST_LOG_LEVEL`, which quietens rule logs -- the frame around them
     is not part of what that floor governs.

     Disabled workflows get NO hand-off band: the opening block's sequence
     diagram states every one of them, once, before anything runs. Every line
     that can carry `extra` goes through `sanitize_argv` first.

     stdout is FLUSHED before each `subprocess.run`. Python block-buffers a
     redirected stdout while the child inherits the fd directly, so without the
     flush a `run_workflows.py ... *> log.txt` interleaves each banner AFTER the
     output of the workflow it announces. No test can catch this -- the suite
     fakes `subprocess.run` and the child writes nothing.

Disabling a workflow neither deletes its prior outputs nor guarantees downstream
freshness: the wrapper invokes each Snakefile independently with no
prerequisite-freshness check -- identical to invoking a single Snakefile
directly today. A user who disables a prerequisite owns the staleness of what
downstream consumes.

Usage::

    python scripts/run_workflows.py --config test_case/snake_config_baseline.yml
    python scripts/run_workflows.py --config <cfg> --cores 4 -- --dry-run
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import re
import subprocess
import sys
import time
import uuid
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

# Make the plain source tree importable when this file is executed directly as
# ``python scripts/run_workflows.py`` rather than imported by pytest.
_REPO_ROOT_PATH = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT_PATH) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT_PATH))

from blueearth_cst.shared.provenance import (  # noqa: E402
    effective_config_digest,
    environment_file_hashes,
    file_sha256,
    toolbox_identity,
)
from blueearth_cst.shared.snake_utils import (  # noqa: E402
    ADVANCED_SETTINGS,
    format_elapsed,
)

# Fixed run order (climate -> model -> projections -> experiment). Each maps to
# its Snakefile and the per-workflow flags preserved verbatim from the runners
# (design §7(f)): --keep-going on analyze_projections only.
#
# `analyze_climate` leads because it is model-free and the other three are not:
# it produces the shared region, vector foundation and climate store that
# build_model reads, so running it first means those exist before the model
# build asks for them. The order is fixed, not derived -- the wrapper does no
# freshness checking (see the module docstring).
WORKFLOW_ORDER = (
    "analyze_climate",
    "build_model",
    "analyze_projections",
    "run_stress_test",
)

SNAKEFILE = {
    "analyze_climate": "analyze_climate.smk",
    "build_model": "build_model.smk",
    "analyze_projections": "analyze_projections.smk",
    "run_stress_test": "run_stress_test.smk",
}

PER_WORKFLOW_FLAGS = {
    "analyze_climate": [],
    "build_model": [],
    "analyze_projections": ["--keep-going"],
    "run_stress_test": [],
}

# Workflow -> its `wf<N>` id, for the console only. This is the THIRD copy of
# that mapping in the repo and the other two must be kept in step with it:
# `scripts/plot_workflow_dag.py::WORKFLOW_NUMBER` keys it by Snakefile name for
# the render filenames, and each Snakefile hardcodes the assembled label in its
# own `run_header`/`run_summary` calls ("wf0 analyze_climate", ...). The merged
# log names in blueearth_cst/shared/merge_logs.py carry the same ids.
#
# 0 for analyze_climate because it runs BEFORE model creation; `W` is a workflow
# id, not a position (dev/reference/naming.md §9).
WORKFLOW_ID = {
    "analyze_climate": "wf0",
    "build_model": "wf1",
    "analyze_projections": "wf2",
    "run_stress_test": "wf3",
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
    return config


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


def build_command(
    name: str, config_path: str, cores: int, extra: list[str]
) -> list[str]:
    """Assemble the snakemake argv for one workflow (contract (e)/(f))."""
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
# The rule is a FIXED 80 columns, not the terminal's width. A redirected run and
# a console run must produce the same bytes: this output is routinely piped to a
# file (`*> log.txt`), and a frame whose width depended on whether stdout was a
# tty would make the same run look different depending on how it was launched.
#
# ASCII only, everywhere in this section. A Windows console defaults to cp1252
# and raises UnicodeEncodeError on box-drawing characters and arrows -- the same
# constraint `snake_utils.rule_banner` records. `=`, `|`, `-` and `[1/3]`, never
# `═`, `│`, `→` or `✓`.

_RULE_WIDTH = 80
_RULE = "=" * _RULE_WIDTH


def _label(name: str) -> str:
    """A workflow's console identity: ``wf0 analyze_climate``."""
    return f"{WORKFLOW_ID[name]} {name}"


def _clock() -> str:
    """Wall-clock ``HH:MM:SS`` for a hand-off line.

    A wall clock, unlike the monotonic durations everywhere else here: this
    answers "when did this start", which someone reads against the timestamps in
    the workflow's own log rows below it.
    """
    return f"{datetime.now():%H:%M:%S}"


def _banner(head: str) -> list[str]:
    """A rule, one head line, a rule -- the wrapper announcing a block."""
    return [_RULE, f"  {head}", _RULE]


def _handoff(tag: str, detail: str, *, note: str | None = None) -> str:
    """One hand-off band: a rule, the workflow and what just happened to it.

    Ruled rather than boxed, and ruled on the TOP side only. What has to survive
    is that the line is the runner's; a closing rule under a one-line utterance
    doubles the cost to say it twice, and this fires twice per workflow.
    """
    lines = [_RULE, f"  {tag}  --  {detail}"]
    if note:
        lines.append(f"    {note}")
    return "\n".join(lines)


def _console_block(
    head: str, groups: list[tuple[str, list[Any]]], *, close: bool = False
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
    lines = _banner(head)
    for label, rows in groups:
        lines.extend(["", f"  {label}"])
        for row in rows:
            if isinstance(row, tuple):
                key, value = row
                lines.append(f"    {key.ljust(width)}  {value}")
            else:
                lines.append(f"    {row}".rstrip())
    if close:
        lines.extend(["", _RULE])
    return "\n".join(lines)


def _sequence_lines(flags: Mapping[str, bool]) -> list[str]:
    """The enabled/disabled pipeline as an ASCII chain, in WORKFLOW_ORDER.

    Every workflow appears, enabled or not, because the question this answers is
    "what is about to happen" and a disabled workflow silently absent from the
    list is indistinguishable from one this wrapper does not know about. Enabled
    entries carry their `[position/total]` so the per-workflow timeline rows
    below can be matched to the plan without counting.
    """
    total = sum(1 for name in WORKFLOW_ORDER if flags[name])
    # Width from the widest marker this run can print, so the connector's `|`
    # and a disabled row's `-` stay centred under it past nine workflows.
    mark_width = max(5, len(f"[{total}/{total}]"))
    label_width = max(len(_label(name)) for name in WORKFLOW_ORDER)
    lines: list[str] = []
    position = 0
    for index, name in enumerate(WORKFLOW_ORDER):
        if index:
            lines.append("|".center(mark_width))
        if flags[name]:
            position += 1
            lines.append(f"{f'[{position}/{total}]'.ljust(mark_width)}  {_label(name)}")
        else:
            lines.append(
                f"{'-'.center(mark_width)}  {_label(name).ljust(label_width)}"
                f"  (disabled, not invoked)"
            )
    return lines


def _opening_block(
    *,
    project_name: str,
    project_dir: Path,
    config_path: str,
    cores: int,
    dry_run: bool,
    flags: Mapping[str, bool],
) -> str:
    """The start-of-invocation block: which run this is, and what will run.

    `config` prints as the caller SPELLED it (forward-slashed), not resolved:
    a relative `--config` is how every documented invocation passes it, and the
    absolute form is both longer and machine-specific. `folder` is resolved,
    because that one is an answer -- where the outputs land -- rather than an
    echo of the command line.
    """
    run_rows: list[Any] = [
        ("project", project_name),
        ("folder", os.fspath(project_dir).replace(os.sep, "/")),
        ("config", os.fspath(config_path).replace(os.sep, "/")),
        ("cores", str(cores)),
    ]
    if dry_run:
        # Stated because a dry run's console is otherwise nearly identical to a
        # real one's, and the two are worth minutes of confusion apart.
        run_rows.append(("mode", "dry run -- nothing is executed"))
    enabled = sum(1 for name in WORKFLOW_ORDER if flags[name])
    total = len(WORKFLOW_ORDER)
    groups: list[tuple[str, list[Any]]] = [("run", run_rows)]
    if enabled:
        groups.append(
            (
                f"sequence -- {enabled} of {total} workflows enabled, "
                f"invoked in this order",
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
    return _console_block("run_workflows", groups)


def _closing_block(
    *,
    project_dir: Path,
    manifest_path: Path,
    manifest: Mapping[str, Any],
    ran: list[tuple[str, str]],
    elapsed_seconds: float,
    failed: bool,
) -> str:
    """The end-of-invocation block: what ran, how long, and where it landed.

    `ran` holds only workflows this invocation actually INVOKED, each with its
    already-formatted outcome; everything the stop boundary left behind is named
    in one `not run` note instead. A group headed "ran" listing workflows that
    did not is worse than not printing them.

    `logs` names the DIRECTORY rather than the per-workflow log files. Those
    names are Snakefile constants (`wf3_run_stress_test_<experiment>.log` is
    built from a resolved experiment id), so reconstructing them here would put
    a second definition of each in the one place that cannot notice when it
    drifts -- and each workflow's own `run_summary` has already named its exact
    log a few lines above.
    """
    root = os.fspath(project_dir).replace(os.sep, "/")
    verdict = "FAILED" if failed else "done"
    head = f"run_workflows {verdict} in {format_elapsed(elapsed_seconds)}"
    # `error_type` is set only on the path where a child never LAUNCHED (an
    # OSError out of subprocess.run). Two of the rows below are claims about a
    # child having produced something, and both are false there.
    launched = ran and not manifest.get("error_type")

    groups: list[tuple[str, list[Any]]] = []
    if ran:
        groups.append(("ran", [(_label(name), outcome) for name, outcome in ran]))
    elif manifest["no_op"]:
        groups.append(("nothing ran -- every workflow was disabled", []))

    wrote: list[Any] = [("project", root)]
    if launched:
        wrote.append(("logs", f"{root}/logs/"))
    wrote.append(
        ("invocation", os.fspath(manifest_path).replace(os.sep, "/")),
    )
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
    return _console_block(head, groups, close=True)


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


def run(config_path: str, cores: int, extra: list[str]) -> int:
    """Invoke each enabled workflow in fixed order; stop on first nonzero exit
    and return that code (contract (d)). Returns 0 if all enabled workflows
    succeed (or all are disabled)."""
    cfg = _read_config(config_path)
    flags = _enabled_flags(cfg, config_path)
    project_dir = _project_dir(cfg, config_path)
    manifest_path, manifest = _initialize_manifest(
        cfg=cfg,
        config_path=config_path,
        project_dir=project_dir,
        flags=flags,
        cores=cores,
        extra=extra,
    )
    _write_json_atomic(manifest_path, manifest)

    # monotonic, matching each Snakefile's own `_RUN_STARTED`: a wall clock can
    # step backwards mid-run and these are durations, never timestamps.
    started = time.monotonic()
    try:
        print(
            _opening_block(
                project_name=_project_name(cfg, project_dir),
                project_dir=project_dir,
                config_path=config_path,
                cores=cores,
                dry_run=manifest["dry_run"],
                flags=flags,
            ),
            flush=True,
        )
    except Exception as exc:  # noqa: BLE001 -- never break a run over a banner
        print(f"(run header unavailable: {exc})", file=sys.stderr, flush=True)

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
            position += 1
            tag = f"[{position}/{total}]  {_label(name)}"
            cmd = build_command(name, config_path, cores, extra)
            workflow["command"] = sanitize_argv(cmd)
            workflow["status"] = "running"
            print(
                "\n"
                + _handoff(
                    tag,
                    f"starting {_clock()}",
                    note=" ".join(sanitize_argv(cmd)),
                )
                + "\n",
            )
            # Before the child, not after: see contract (h) on buffering.
            sys.stdout.flush()
            workflow_started = time.monotonic()
            try:
                result = subprocess.run(cmd, cwd=REPO_ROOT)
            except BaseException as exc:
                elapsed = format_elapsed(time.monotonic() - workflow_started)
                ran.append((name, f"FAILED ({type(exc).__name__}) after {elapsed}"))
                raise
            elapsed = format_elapsed(time.monotonic() - workflow_started)
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
            print("\n" + _handoff(tag, f"done in {elapsed}"), flush=True)
    except BaseException as exc:
        manifest["status"] = "failed"
        manifest["error_type"] = type(exc).__name__
        for workflow in manifest["workflows"].values():
            if workflow["status"] == "running":
                workflow["status"] = "failed"
        _mark_pending_not_run(manifest)
        _finalize_manifest(manifest_path, manifest, exit_code=None)
        # Before the re-raise, so a launch failure closes with a report rather
        # than with a traceback and nothing else -- the same reason the manifest
        # is finalized on this path.
        _report(
            project_dir=project_dir,
            manifest_path=manifest_path,
            manifest=manifest,
            ran=ran,
            elapsed_seconds=time.monotonic() - started,
            failed=True,
        )
        raise

    if exit_code != 0:
        manifest["status"] = "failed"
        _mark_pending_not_run(manifest)
    else:
        manifest["status"] = "succeeded"
    _finalize_manifest(manifest_path, manifest, exit_code=exit_code)
    _report(
        project_dir=project_dir,
        manifest_path=manifest_path,
        manifest=manifest,
        ran=ran,
        elapsed_seconds=time.monotonic() - started,
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
        print("\n" + _closing_block(**kwargs), flush=True)
    except Exception as exc:  # noqa: BLE001 -- never break a run over a banner
        print(f"(run summary unavailable: {exc})", file=sys.stderr, flush=True)


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
    # Under `config/runs/`, NOT a `provenance/` root of its own (R9 follow-up,
    # ruled 2026-08-05). The migration map's Finding 1 disqualified `logs/` for
    # the config snapshot because logs are what a user deletes to reclaim space
    # and their parts are merged-then-deleted by design, while the snapshot is
    # immutable and retained. This manifest is immutable and retained for the
    # same reasons, so the same reasoning places it here. `invocations/` is a
    # SIBLING of the `<workflow>/<digest>/` bundles rather than a fourth
    # workflow entry: an invocation spans workflows.
    runs_dir = project_dir / "config" / "runs" / "invocations"
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
    ap = argparse.ArgumentParser(
        description="Run the enabled CST workflows in fixed order.",
    )
    ap.add_argument(
        "--config",
        required=True,
        help="path to a full-orchestration snake config (see test_case/ for examples)",
    )
    ap.add_argument(
        "--cores",
        type=int,
        default=3,
        help="cores forwarded to every snakemake invocation (default: 3)",
    )
    ap.add_argument(
        "extra",
        nargs=argparse.REMAINDER,
        help="args after `--` are appended verbatim to every invocation",
    )
    args = ap.parse_args(argv)

    # argparse.REMAINDER captures the leading `--` sentinel; strip it.
    extra = args.extra
    if extra and extra[0] == "--":
        extra = extra[1:]

    try:
        return run(args.config, args.cores, extra)
    except ConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
