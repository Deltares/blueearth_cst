"""Tests for the portable tee wrapper (t260721a).

The contract that matters: the wrapper returns the *child's* exit code (a bare
``| tee`` returns tee's, masking failures on cmd.exe) and mirrors the child's
output into the log file. Child commands are ``python -c`` snippets so the tests
are OS-independent and need no hydromt/julia.
"""

import sys

from blueearth_cst.shared.run_logged import main
from blueearth_cst.shared.snake_utils import run_and_tee


def test_run_and_tee_returns_child_exit_code(tmp_path):
    log = tmp_path / "fail.log"
    rc = run_and_tee(
        [sys.executable, "-c", "import sys; print('boom'); sys.exit(3)"], log
    )
    assert rc == 3  # the child's code, NOT tee's 0
    assert "boom" in log.read_text(encoding="utf-8")


def test_run_and_tee_success_writes_log(tmp_path):
    log = tmp_path / "ok.log"
    rc = run_and_tee([sys.executable, "-c", "print('hello world')"], log)
    assert rc == 0
    assert "hello world" in log.read_text(encoding="utf-8")


def test_run_and_tee_merges_stderr(tmp_path):
    log = tmp_path / "err.log"
    rc = run_and_tee(
        [sys.executable, "-c", "import sys; sys.stderr.write('to-stderr\\n')"], log
    )
    assert rc == 0
    assert "to-stderr" in log.read_text(encoding="utf-8")


def test_run_and_tee_creates_parent_dirs(tmp_path):
    log = tmp_path / "nested" / "deep" / "run.log"
    rc = run_and_tee([sys.executable, "-c", "print('ok')"], log)
    assert rc == 0
    assert log.exists()


def test_run_and_tee_collapses_shutdown_excepthook_cascade(tmp_path):
    # A pure trailing cascade of empty-bodied excepthook markers (the benign
    # hydromt-build shutdown noise) is collapsed into one summary line.
    log = tmp_path / "cascade.log"
    # Write everything to stderr (unbuffered) so ordering is deterministic and
    # the cascade is genuinely trailing.
    snippet = (
        "import sys\n"
        "sys.stderr.write('real work line\\n')\n"
        "[sys.stderr.write('Error in sys.excepthook:\\n\\n"
        "Original exception was:\\n\\n') for _ in range(5)]"
    )
    rc = run_and_tee([sys.executable, "-c", snippet], log)
    text = log.read_text(encoding="utf-8")
    assert rc == 0
    assert "real work line" in text  # real content preserved
    # The 20-line cascade is gone; the phrase survives only inside the one
    # summary line (which quotes the marker text).
    assert "[run_logged] collapsed 20 benign" in text
    assert text.count("Error in sys.excepthook:") == 1
    assert "child rc=0" in text


def test_run_and_tee_preserves_real_traceback_between_markers(tmp_path):
    # A genuine excepthook failure interleaves the markers with a real
    # traceback; non-empty bodies must NOT be collapsed.
    log = tmp_path / "real.log"
    snippet = (
        "import sys\n"
        "sys.stderr.write('Error in sys.excepthook:\\n')\n"
        "sys.stderr.write('Traceback (most recent call last):\\n')\n"
        "sys.stderr.write('ValueError: boom\\n')\n"
        "sys.stderr.write('Original exception was:\\n')\n"
        "sys.stderr.write('RuntimeError: real\\n')"
    )
    rc = run_and_tee([sys.executable, "-c", snippet], log)
    text = log.read_text(encoding="utf-8")
    assert rc == 0
    assert "ValueError: boom" in text
    assert "RuntimeError: real" in text
    assert "Error in sys.excepthook:" in text  # kept verbatim, not collapsed
    assert "[run_logged] collapsed" not in text


def test_run_and_tee_decodes_utf8_child_output(tmp_path):
    # The child writes raw UTF-8 bytes (as Julia/Wflow progress bars do): a box
    # char and full blocks. They must land in the log intact, NOT mangled via
    # the Windows locale code page (which would turn `█` into `â–ˆ`).
    log = tmp_path / "utf8.log"
    # Write bytes straight to the buffer so the child's own stdout encoding
    # (cp1252 on Windows) can't corrupt them first — this mimics Julia.
    snippet = (
        "import sys; "
        "sys.stdout.buffer.write("
        "'\\u250c Progress 100%|\\u2588\\u2588\\u2588|\\n'.encode('utf-8'))"
    )
    rc = run_and_tee([sys.executable, "-c", snippet], log)
    text = log.read_text(encoding="utf-8")
    assert rc == 0
    assert "┌" in text  # ┌ preserved
    assert "███" in text  # ███ preserved
    assert "â" not in text  # no 'â' mojibake


def test_run_and_tee_compacts_hydromt_log_format(tmp_path):
    # A child emitting a hydromt-format record (as the hydromt build/update CLI
    # does) has its redundant dotted logger name dropped in the captured log.
    log = tmp_path / "compact.log"
    snippet = (
        "print('2026-07-21 18:03:38,474 - hydromt.model.model - model - "
        "INFO - Initializing wflow_sbm model.')"
    )
    rc = run_and_tee([sys.executable, "-c", snippet], log)
    text = log.read_text(encoding="utf-8")
    assert rc == 0
    assert "18:03:38 - model - Initializing wflow_sbm model." in text
    assert "hydromt.model.model" not in text


def test_cli_requires_separator():
    assert main(["only-a-log.log"]) == 2


def test_cli_rejects_missing_command(tmp_path):
    assert main([str(tmp_path / "l.log"), "--"]) == 2


def test_cli_runs_command_and_returns_code(tmp_path):
    log = tmp_path / "cli.log"
    rc = main([str(log), "--", sys.executable, "-c", "import sys; sys.exit(5)"])
    assert rc == 5


def test_run_and_tee_collapses_a_cascade_that_is_not_trailing(tmp_path):
    """The `-c 3` case: another job's line lands after the cascade.

    Until 2026-08-10 the buffered block was flushed VERBATIM whenever real
    content followed, so the collapse fired only when the noise happened to end
    the stream. Concurrent jobs make that the uncommon case, which is why the
    filter looked present and did nothing in a real run.
    """
    log = tmp_path / "midstream.log"
    snippet = (
        "import sys\n"
        "sys.stderr.write('first job line\\n')\n"
        "[sys.stderr.write('Error in sys.excepthook:\\n\\n"
        "Original exception was:\\n\\n') for _ in range(4)]\n"
        "sys.stderr.write('another job line\\n')\n"
    )
    rc = run_and_tee([sys.executable, "-c", snippet], log)
    text = log.read_text(encoding="utf-8")
    assert rc == 0
    assert "first job line" in text and "another job line" in text
    assert "[run_logged] collapsed 16 benign" in text
    assert "mid-run" in text
    assert text.count("Error in sys.excepthook:") == 1


def test_run_and_tee_still_keeps_a_real_traceback_mid_stream(tmp_path):
    """The mid-stream collapse must stay as conservative as the trailing one."""
    log = tmp_path / "midreal.log"
    snippet = (
        "import sys\n"
        "sys.stderr.write('Error in sys.excepthook:\\n')\n"
        "sys.stderr.write('ValueError: boom\\n')\n"
        "sys.stderr.write('Original exception was:\\n')\n"
        "sys.stderr.write('RuntimeError: real\\n')\n"
        "sys.stderr.write('trailing normal line\\n')\n"
    )
    rc = run_and_tee([sys.executable, "-c", snippet], log)
    text = log.read_text(encoding="utf-8")
    assert rc == 0
    assert "ValueError: boom" in text and "RuntimeError: real" in text
    assert "[run_logged] collapsed" not in text


def _body(log):
    """The log's rows, with the four-line header and blank lines dropped."""
    text = log.read_text(encoding="utf-8")
    return [r for r in text.splitlines() if r and not r.startswith("#")]


def test_a_carriage_return_progress_bar_collapses_to_its_final_frame(tmp_path):
    """Wflow redraws one bar ~40 times per model run. Popen's universal-newline
    default turned every `\\r` into a `\\n`, so a WF3 experiment spent ~2000 of
    its 4000 log rows on frames of bars meant to occupy twenty."""
    log = tmp_path / "bar.log"
    snippet = (
        "import sys\n"
        "for pct in (0, 50, 100):\n"
        "    sys.stdout.write('\\rProgress: %3d%%' % pct)\n"
        "sys.stdout.write('\\n')\n"
        "print('done')\n"
    )
    rc = run_and_tee([sys.executable, "-c", snippet], log)
    assert rc == 0
    assert _body(log) == ["Progress: 100%", "done"]


def test_a_bar_stays_off_a_non_tty_console_but_lands_in_the_log(tmp_path, capsys):
    """`_Tee` drops redraw frames; the shell path has to agree.

    Under snakemake `sys.stdout` is a tee whose `isatty()` is False, so no frame
    was ever streamed and the final one would print as a row of its own.
    """
    log = tmp_path / "bar.log"
    snippet = (
        "import sys\n"
        "for pct in (0, 50, 100):\n"
        "    sys.stdout.write('\\rProgress: %3d%%' % pct)\n"
        "sys.stdout.write('\\n')\n"
        "print('done')\n"
    )
    assert run_and_tee([sys.executable, "-c", snippet], log) == 0
    console = capsys.readouterr().out
    assert "Progress:" not in console
    assert "done" in console
    assert _body(log) == ["Progress: 100%", "done"]


def test_crlf_is_a_line_ending_and_not_a_redraw(tmp_path):
    """`_cr_overwrite` would split `text\\r\\n` on the `\\r` and keep only the
    newline, blanking a row that a Windows child wrote perfectly normally."""
    log = tmp_path / "crlf.log"
    snippet = "import sys; sys.stdout.buffer.write(b'alpha\\r\\nbeta\\r\\n')"
    rc = run_and_tee([sys.executable, "-c", snippet], log)
    assert rc == 0
    assert _body(log) == ["alpha", "beta"]


def test_a_julia_log_record_folds_onto_one_row(tmp_path):
    """Julia hard-wraps one message across `+`/`|`/`+` lines; Wflow emits dozens
    per run, and each was several console rows."""
    log = tmp_path / "julia.log"
    snippet = (
        "import sys\n"
        "sys.stdout.reconfigure(encoding='utf-8')\n"
        "sys.stdout.write('\\u250c Info: Set precipitation using netCDF\\n')\n"
        "sys.stdout.write('\\u2514 variable precip as forcing parameter.\\n')\n"
    )
    rc = run_and_tee([sys.executable, "-c", snippet], log)
    assert rc == 0
    assert _body(log) == [
        "Info: Set precipitation using netCDF variable precip as forcing parameter."
    ]


def test_a_julia_keyword_record_folds_to_a_parenthesised_list(tmp_path):
    """A three-space indent is a kwarg table, not wrapped prose: joining those
    with spaces would read as a sentence and lose that they are a list."""
    log = tmp_path / "kwargs.log"
    snippet = (
        "import sys\n"
        "sys.stdout.reconfigure(encoding='utf-8')\n"
        "sys.stdout.write('\\u250c Info: General model settings\\n')\n"
        "sys.stdout.write('\\u2502   snow = true\\n')\n"
        "sys.stdout.write('\\u2514   glacier = false\\n')\n"
    )
    rc = run_and_tee([sys.executable, "-c", snippet], log)
    assert rc == 0
    assert _body(log) == ["Info: General model settings (snow = true, glacier = false)"]


def test_an_unterminated_julia_record_is_released_verbatim(tmp_path):
    """A cosmetic filter must never eat a Wflow diagnostic. The head line is
    also the case that regressed once: flushing it back through the folder
    re-buffered and lost it."""
    log = tmp_path / "partial.log"
    snippet = (
        "import sys\n"
        "sys.stdout.reconfigure(encoding='utf-8')\n"
        "sys.stdout.write('\\u250c Info: truncated mid-record\\n')\n"
    )
    rc = run_and_tee([sys.executable, "-c", snippet], log)
    assert rc == 0
    assert _body(log) == ["\u250c Info: truncated mid-record"]


def test_an_interrupted_julia_record_releases_what_it_held(tmp_path):
    """Another thread's line landing inside a record must not swallow it."""
    log = tmp_path / "interrupted.log"
    snippet = (
        "import sys\n"
        "sys.stdout.reconfigure(encoding='utf-8')\n"
        "sys.stdout.write('\\u250c Info: opened\\n')\n"
        "sys.stdout.write('unrelated line\\n')\n"
        "sys.stdout.write('[ Info: after\\n')\n"
    )
    rc = run_and_tee([sys.executable, "-c", snippet], log)
    assert rc == 0
    assert _body(log) == [
        "\u250c Info: opened",
        "unrelated line",
        "[ Info: after",
    ]
