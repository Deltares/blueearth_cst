"""Canonical workflow orientation, literal commands and logger lifecycle."""

import io
import json
import logging
import os
import shlex
import shutil
import subprocess
import sys
from types import SimpleNamespace

import pytest

from blueearth_cst.shared import console_style as cs


@pytest.fixture
def presentation(monkeypatch):
    for name, value in {
        "_ENGINE_ARGV": [
            "snakemake",
            "all",
            "--cores",
            "3",
            "--configfile",
            "captured config.yml",
        ],
        "_RULE_NUMBERS": {"fetch": "2.03", "analysis": "2.05"},
        "_PRE_DAG_STEPS": {},
        "_AFTER_CHECKPOINT_RULES": {},
        "_PLANNED_FROM": {},
        "_INPUT_CHECKS": [],
        "_COUNTED_INPUT_CHECKS": set(),
        "_DEFERRED_WARNINGS": [],
        "_HEADER_PRINTED": False,
        "_RUN_HEADER": (
            "wf2 analyze_projections",
            "/project",
            "/source config.yml",
            {"cores": 3, "period": "1995–2014"},
        ),
    }.items():
        monkeypatch.setattr(cs, name, value)
    sink = io.StringIO()
    base = logging.StreamHandler(sink)
    base.setFormatter(logging.Formatter("%(message)s"))
    return cs._ConsoleHandler(base), sink


def emit(handler, message, event=None, **fields):
    record = logging.LogRecord("snakemake", logging.INFO, "", 0, message, (), None)
    record.event = event
    record.__dict__.update(fields)
    handler.emit(record)


def test_commands_quote_literal_arguments_for_both_shells():
    arguments = ["snakemake", "a b", "a'b", "$literal", "`literal", "c:\\case\\x.yml"]
    assert shlex.split(cs.shell_command(arguments, powershell=False)) == arguments
    assert (
        cs.shell_command(arguments, powershell=True)
        == "& 'snakemake' 'a b' 'a''b' '$literal' '`literal' 'c:\\case\\x.yml'"
    )


def test_embedded_api_does_not_invent_a_command_from_private_process_args(monkeypatch):
    monkeypatch.setattr(cs, "_ENGINE_ARGV", None)
    monkeypatch.setattr(sys, "argv", ["pytest", "private-argument", "--dry-run"])
    output = cs.run_header("wf0", "/project")
    assert "Unavailable (embedded API invocation)" in output
    assert "private-argument" not in output and "pytest" not in output
    assert "DRY RUN" not in output


@pytest.mark.parametrize(
    "entry",
    [
        "C:/env/Lib/site-packages/snakemake/__main__.py",
        "/env/lib/snakemake/__main__.py",
        "C:\\env\\Scripts\\snakemake.exe",
    ],
)
def test_native_entrypoints_retain_actual_cli_arguments(monkeypatch, entry):
    monkeypatch.setattr(cs, "_ENGINE_ARGV", None)
    monkeypatch.setattr(sys, "argv", [entry, "all", "-np"])
    assert cs.engine_command() == ["snakemake", "all", "-np"]
    assert cs._dry_run()


@pytest.mark.process_isolation
@pytest.mark.parametrize("multiline", [False, True])
def test_powershell_command_round_trips_literal_arguments(multiline):
    pwsh = shutil.which("pwsh")
    if pwsh is None:
        pytest.skip("PowerShell unavailable")
    arguments = ["space here", "apostrophe's", "$literal", "`literal", 'quote"here']
    command = cs.shell_command(
        [
            sys.executable,
            "-c",
            "import json, sys; print(json.dumps(sys.argv[1:]))",
            *arguments,
        ],
        powershell=True,
        multiline=multiline,
    )
    result = subprocess.run(
        [pwsh, "-NoProfile", "-Command", command],
        capture_output=True,
        text=True,
        check=True,
        env={**os.environ, "PYTHONIOENCODING": "utf-8"},
        timeout=30,
    )
    assert json.loads(result.stdout) == arguments


@pytest.mark.parametrize("powershell", [False, True])
def test_multiline_command_keeps_option_values_together(powershell):
    arguments = ["snakemake", "all", "-c", "3", "--configfile", "case config.yml"]
    output = cs.shell_command(arguments, powershell=powershell, multiline=True)
    if powershell:
        assert output.splitlines() == [
            "& 'snakemake' 'all' `",
            "  '-c' '3' `",
            "  '--configfile' 'case config.yml'",
        ]
    else:
        assert output.splitlines() == [
            "snakemake all \\",
            "  -c 3 \\",
            "  --configfile 'case config.yml'",
        ]
        assert shlex.split(output.replace("\\\n", "")) == arguments


def test_child_command_display_preserves_credential_redaction(presentation):
    cs.set_engine_command(
        [
            "snakemake",
            "--config",
            "api_token=secret-value",
            "--password",
            "pw-secret",
            "ordinary=$literal",
        ]
    )
    output = "\n".join(line for line, _ in cs.opening_block("wf2", "/project"))
    assert "secret-value" not in output and "pw-secret" not in output
    assert "api_token=<redacted>" in output and "ordinary=$literal" in output
    assert output.count("snakemake") == 1
    assert "Single-line command" not in output


def test_opening_groups_checks_before_explicit_plan(presentation, monkeypatch):
    handler, sink = presentation
    warnings = []
    monkeypatch.setattr(cs, "note_warning", warnings.append)
    cs.defer_input_check(
        "fetch", "Availability partially verified", ["Checked during fetching: kin."]
    )
    emit(handler, "Job stats:\nfetch 2\nanalysis 1\nall 1\ntotal 4", "run_info")
    output = sink.getvalue()
    sections = [
        "wf2 analyze_projections",
        "Snakemake command",
        "Input checks",
        "Work plan",
        "Run settings",
        "Paths",
        "Execution log",
    ]
    assert [output.index(section) for section in sections] == sorted(
        output.index(section) for section in sections
    )
    assert (
        "status" in output
        and "step" in output
        and "rule" in output
        and "jobs" in output
    )
    assert "3 jobs" in output and "2 rules scheduled" in output
    assert "WARN  Availability partially verified" in output
    assert warnings == ["resolution"]
    cs.opening_block("wf2 analyze_projections", "/project", counts={"fetch": 2})
    assert warnings == ["resolution"]


@pytest.mark.parametrize("flag", ["--dry-run", "-np"])
def test_dry_run_has_plan_without_execution_rows(presentation, monkeypatch, flag):
    handler, sink = presentation
    monkeypatch.setattr(cs, "_ENGINE_ARGV", ["snakemake", "all", flag])
    emit(handler, "Job stats:\nanalysis 1\nall 1\ntotal 2", "run_info")
    emit(
        handler,
        "would execute",
        "job_info",
        rule_name="analysis",
        rule_msg="Rule 2.05: analysis",
        jobid=0,
    )
    output = sink.getvalue()
    assert output.index("DRY RUN") < output.index("Work plan")
    assert "Planning complete" in output and "No jobs executed" in output
    assert "Execution log" not in output and "would execute" not in output
    assert " - RUN " not in output


def test_noop_is_compact_and_suppresses_acquisition_check(presentation):
    handler, sink = presentation
    cs.defer_input_check("fetch", "must not show", [])
    emit(
        handler, "Nothing to be done (all requested files are present and up to date)."
    )
    output = sink.getvalue()
    assert "wf2 analyze_projections" in output and "Snakemake command" in output
    assert "Up to date — no jobs scheduled" in output
    assert all(
        text not in output
        for text in (
            "Input checks",
            "must not show",
            "Work plan",
            "Run settings",
            "Paths",
            "Execution log",
        )
    )


def test_startup_none_and_native_planning_footer_are_not_terminal_rows(
    presentation, monkeypatch
):
    handler, sink = presentation
    monkeypatch.setattr(cs, "_ENGINE_ARGV", ["snakemake", "--dry-run"])
    emit(handler, None, "resources_info")
    emit(handler, "Reasons:\n input files updated by another job")
    emit(handler, "This was a dry-run (flag -n).")
    assert sink.getvalue() == ""
    monkeypatch.setattr(cs, "_ENGINE_ARGV", ["snakemake", "--dry-run", "--verbose"])
    emit(handler, "Reasons:\n input files updated by another job")
    assert "Reasons:" in sink.getvalue()


def test_pending_checkpoint_and_completed_planning_are_not_reuse(
    presentation, monkeypatch
):
    monkeypatch.setattr(cs, "_PRE_DAG_STEPS", {"prepare": "2.01"})
    monkeypatch.setattr(cs, "_AFTER_CHECKPOINT_RULES", {"analysis": 4})
    head, rows = cs._plan_lines({"fetch": 1, "all": 1})
    text = "\n".join(row for row, _ in rows)
    assert "done" in text and "pending" in text and "4 planned" in text
    assert "0 rules up to date" in head and "1 job" in head


def test_deferred_factory_is_idempotent_and_retains_signature_and_file_handler(
    monkeypatch,
):
    import snakemake.logging

    calls = []

    def factory(*args, **kwargs):
        calls.append((args, kwargs))
        base = logging.StreamHandler(io.StringIO())
        base.name = "DefaultStreamHandler"
        return base

    manager = SimpleNamespace(queue_listener=None, _default_streamhandler=factory)
    monkeypatch.setattr(snakemake.logging, "logger_manager", manager)
    monkeypatch.setattr(cs, "_CONSOLE_STYLE_ACTIVE", False)
    monkeypatch.setattr(cs, "_CONSOLE_STYLE_PENDING", False)
    assert cs.install_console_style() is False
    registered = manager._default_streamhandler
    assert cs._CONSOLE_STYLE_PENDING and not cs._CONSOLE_STYLE_ACTIVE
    assert cs.install_console_style() is False
    assert manager._default_streamhandler is registered
    first = registered("literal", setting=True)
    assert calls == [(("literal",), {"setting": True})]
    assert isinstance(first, cs._ConsoleHandler)
    assert cs._CONSOLE_STYLE_ACTIVE and not cs._CONSOLE_STYLE_PENDING
    file_handler = logging.StreamHandler(io.StringIO())
    file_handler.name = "DefaultLogFileHandler"
    manager.queue_listener = SimpleNamespace(handlers=(first, file_handler))
    monkeypatch.setattr(cs, "_HEADER_PRINTED", True)
    assert cs.install_console_style()
    assert cs._HEADER_PRINTED
    assert manager.queue_listener.handlers[1] is file_handler
    manager.queue_listener = None
    assert cs.install_console_style() is False
    assert not cs._HEADER_PRINTED
    assert isinstance(registered(), cs._ConsoleHandler)
