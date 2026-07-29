"""Tests for the portable tee wrapper (t260721a).

The contract that matters: the wrapper returns the *child's* exit code (a bare
``| tee`` returns tee's, masking failures on cmd.exe) and mirrors the child's
output into the log file. Child commands are ``python -c`` snippets so the tests
are OS-independent and need no hydromt/julia.
"""
import sys

from blueearth_cst.shared.snake_utils import run_and_tee
from blueearth_cst.shared.run_logged import main


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
    assert "18:03:38 - model - INFO - Initializing wflow_sbm model." in text
    assert "hydromt.model.model" not in text


def test_cli_requires_separator():
    assert main(["only-a-log.log"]) == 2


def test_cli_rejects_missing_command(tmp_path):
    assert main([str(tmp_path / "l.log"), "--"]) == 2


def test_cli_runs_command_and_returns_code(tmp_path):
    log = tmp_path / "cli.log"
    rc = main([str(log), "--", sys.executable, "-c", "import sys; sys.exit(5)"])
    assert rc == 5
