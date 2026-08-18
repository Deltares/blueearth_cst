"""Tests for the portable tee wrapper (t260721a).

The contract that matters: the wrapper returns the *child's* exit code (a bare
``| tee`` returns tee's, masking failures on cmd.exe) and mirrors the child's
output into the log file. Child commands are ``python -c`` snippets so the tests
are OS-independent and need no hydromt/julia.
"""

import io
import re
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


# --- wflow.jl progress frames through the tee ---------------------------------
#
# `shared/wflow_progress.jl` emits `[cst-progress] <label> <fraction>` and the
# tee re-renders it as the house bar. These tests use a python child rather than
# julia so they run in CI, where no julia exists -- the wire format is the
# contract, not the language that produced it.


def test_run_and_tee_renders_wflow_frames_as_a_bar(tmp_path):
    log = tmp_path / "wflow.log"
    snippet = (
        "print('00:01 - wflow - starting')\n"
        "for f in (0.0, 0.5, 1.0):\n"
        "    print(f'[cst-progress] wflow {f}')\n"
        "print('00:02 - wflow - simulation complete  1.0 s')\n"
    )
    rc = run_and_tee([sys.executable, "-c", snippet], log)
    text = log.read_text(encoding="utf-8")

    assert rc == 0
    # The raw frames never reach the log -- they are rendered, not passed through.
    assert "[cst-progress]" not in text
    # The bar's completed frame survives as the one row for the whole run.
    assert "100.0%" in text
    assert "wflow" in text
    # `_cr_overwrite` collapses the redraws, so the intermediate frame is gone.
    assert "50.0%" not in text
    # Ordinary rows either side are untouched.
    assert "starting" in text
    assert "simulation complete" in text


def test_run_and_tee_leaves_a_bar_cut_short_on_its_own_line(tmp_path):
    """A child that dies mid-bar must not strand the next row inside a frame."""
    log = tmp_path / "crash.log"
    snippet = "import sys\nprint('[cst-progress] wflow 0.25')\nsys.exit(7)\n"
    rc = run_and_tee([sys.executable, "-c", snippet], log)
    text = log.read_text(encoding="utf-8")

    assert rc == 7
    assert "[cst-progress]" not in text
    assert "25.0%" in text
    assert text.endswith("\n")


def test_run_and_tee_passes_through_a_malformed_frame(tmp_path):
    """A frame the relay cannot parse costs the bar, never the row."""
    log = tmp_path / "malformed.log"
    rc = run_and_tee(
        [sys.executable, "-c", "print('[cst-progress] wflow not-a-number')"], log
    )
    assert rc == 0
    assert "[cst-progress] wflow not-a-number" in log.read_text(encoding="utf-8")


def test_run_and_tee_drops_the_duplicate_final_frame_entirely(tmp_path):
    """The swallowed repeat must vanish, not surface as a raw sentinel.

    `feed` returning `""` (recognised, not drawn) and `None` (not a frame) are
    different instructions to the tee; conflating them put a literal
    `[cst-progress] wflow 1.0` in the log (observed 2026-08-18).
    """
    log = tmp_path / "dup.log"
    snippet = (
        "print('[cst-progress] wflow 0.5')\n"
        "print('[cst-progress] wflow 1.0')\n"
        "print('[cst-progress] wflow 1.0')\n"
    )
    rc = run_and_tee([sys.executable, "-c", snippet], log)
    text = log.read_text(encoding="utf-8")

    assert rc == 0
    assert "[cst-progress]" not in text
    assert text.count("100.0%") == 1


# --- the console line those frames share --------------------------------------
#
# Everything above reads the LOG, where `_cr_overwrite` collapses a redrawn line
# to its final segment. A terminal does not collapse: `\r` moves the cursor back
# and the next write overwrites from the left, leaving whatever it is too short
# to cover standing. That difference is the whole of the 2026-08-18 defect, so
# these tests replay the console stream the way a terminal would.


class _TTY(io.StringIO):
    """A stdout `run_and_tee` treats as a terminal, so it streams frames."""

    def isatty(self):
        return True


def _screen(text):
    """The lines a terminal would SHOW for ``text``.

    `\\r` returns the cursor to column 0 without erasing; a later, shorter write
    therefore leaves the tail of the earlier one visible. Emulating that is the
    only way a test can see the defect at all -- reading the same stream with
    `_cr_overwrite` reports the intended line and hides the residue.
    """
    lines, current, cursor = [], "", 0
    for char in _ANSI_RE.sub("", text):
        if char == "\n":
            lines.append(current)
            current, cursor = "", 0
        elif char == "\r":
            cursor = 0
        else:
            current = current[:cursor] + char + current[cursor + 1 :]
            cursor += 1
    if current:
        lines.append(current)
    return lines


_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def _console(monkeypatch):
    """Point the tee's console at a fake terminal and return both streams."""
    out, err = _TTY(), io.StringIO()
    monkeypatch.setattr(sys, "stdout", out)
    monkeypatch.setattr(sys, "stderr", err)
    return out, err


def test_the_row_after_a_bar_covers_it_instead_of_hanging_off_its_end(
    tmp_path, monkeypatch
):
    """The reported symptom: a row that looks appended to the bar's frame.

    The running frames are the LONG ones (`0:45 | eta 0:12`); both the bar's own
    summary and any ordinary row that follows are shorter, so before the padding
    they overwrote a prefix and left the rest of the frame standing -- which
    reads as a row with junk stuck to its end rather than as a missing newline.
    """
    out, _ = _console(monkeypatch)
    snippet = (
        "print('[cst-progress] rlz_1_st_2 0.5')\n"
        "print('00:02 - wflow - [1/2] rlz_1_st_2  1.0 s')\n"
    )
    rc = run_and_tee([sys.executable, "-c", snippet], tmp_path / "bar.log")
    assert rc == 0

    lines = _screen(out.getvalue())
    assert lines, out.getvalue()
    # Exactly the row, with nothing of the frame it overwrote left behind. The
    # equality is the assertion: `in` or an `endswith` on the row would pass on
    # the defect, since the residue lands PAST the row's own end.
    assert lines[-1].rstrip() == "00:02 - wflow - [1/2] rlz_1_st_2  1.0 s"


def test_a_bar_summary_covers_the_running_frame_it_replaces(tmp_path, monkeypatch):
    """The final frame is SHORTER than the ones it overwrites, by construction.

    A running frame ends `0:45 | eta 0:12` and the summary ends `1:02 elapsed`,
    so the last characters of the ETA survived past the end of the summary --
    on every completed bar, not occasionally.
    """
    out, _ = _console(monkeypatch)
    snippet = (
        "print('[cst-progress] rlz_1_st_2 0.5')\n"
        "print('[cst-progress] rlz_1_st_2 1.0')\n"
    )
    rc = run_and_tee([sys.executable, "-c", snippet], tmp_path / "summary.log")
    assert rc == 0

    summary = next(line for line in _screen(out.getvalue()) if "100.0%" in line)
    assert summary.rstrip().endswith("elapsed"), summary


def test_a_stall_under_an_open_bar_redraws_it_instead_of_beeping(tmp_path, monkeypatch):
    """Item 1+2: WF3's yellow `still running` while a bar was on the console.

    Wflow leaves two windows uninstrumented -- package load plus JIT, and the
    ~45 s `Wflow.Model(config)` construction -- and WF1 pays them once where a
    WF3 batch pays them per member. The bar is opened across both by the Julia
    driver, so the watchdog now has something to redraw and the notice (which
    would land ON the bar's line) is not printed.
    """
    monkeypatch.setenv("CST_HEARTBEAT_SECS", "0.1")
    out, err = _console(monkeypatch)
    snippet = (
        "import time\n"
        "print('[cst-progress] rlz_1_st_2 0.0', flush=True)\n"
        "time.sleep(0.45)\n"
        "print('[cst-progress] rlz_1_st_2 1.0', flush=True)\n"
    )
    rc = run_and_tee([sys.executable, "-c", snippet], tmp_path / "stall.log")
    assert rc == 0

    assert "still running" not in err.getvalue()
    assert "done in" not in err.getvalue()  # no bracket opened, none to close
    # The bar was redrawn in place: more 0.0% frames than the child ever sent,
    # and still one console line for all of them.
    assert out.getvalue().count("0.0%") > 1


def test_a_stall_with_no_bar_open_still_says_so(tmp_path, monkeypatch):
    """Rules 3.06, 3.12 and 3.14 draw no bar, so the notice is all they have.

    Suppressing it everywhere would trade one wrong console for another: on a
    genuinely silent rule the yellow line is the only evidence the job is alive.
    """
    monkeypatch.setenv("CST_HEARTBEAT_SECS", "0.1")
    _, err = _console(monkeypatch)
    snippet = (
        "import time\nprint('00:01 - weathergen - generating')\ntime.sleep(0.45)\n"
    )
    rc = run_and_tee([sys.executable, "-c", snippet], tmp_path / "quiet.log")

    assert rc == 0
    assert "still running" in err.getvalue()
