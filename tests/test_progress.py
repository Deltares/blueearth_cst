"""Contract tests for the in-place progress bar (`blueearth_cst.shared.progress`).

The bar is console furniture, so what is pinned here is what a rule LOG and a
non-UTF-8 console would otherwise silently get wrong: no escape codes, one line
per compute, an ASCII fallback, and a final frame that reads as a summary.
"""

import io
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from blueearth_cst.shared.progress import (  # noqa: E402
    _GLYPHS_ASCII,
    _GLYPHS_UNICODE,
    DaskFrameRelay,
    DaskProgress,
    WflowFrameRelay,
    _fraction,
    _stream_glyphs,
    format_duration,
    hydromt_progress,
    render_bar,
)


class _FakeStream:
    """A text sink carrying a declared encoding, like a real console does.

    Not an ``io.StringIO`` subclass: ``encoding`` is read-only on the io base
    classes, and the encoding is the whole point of the glyph-fallback tests.
    """

    def __init__(self, encoding="utf-8"):
        self._buffer = io.StringIO()
        self.encoding = encoding
        self.closed = False

    def write(self, text):
        if self.closed:
            raise ValueError("I/O operation on closed file")
        return self._buffer.write(text)

    def flush(self):
        if self.closed:
            raise ValueError("I/O operation on closed file")

    def close(self):
        self.closed = True

    def isatty(self):
        return False

    def getvalue(self):
        return self._buffer.getvalue()


def _state(finished=0, ready=0, waiting=0, running=0):
    return {
        "finished": ["x"] * finished,
        "ready": ["x"] * ready,
        "waiting": ["x"] * waiting,
        "running": ["x"] * running,
    }


# --- duration formatting ------------------------------------------------------


@pytest.mark.parametrize(
    ("seconds", "expected"),
    [
        (0, "0:00"),
        (7, "0:07"),
        (67, "1:07"),
        (599, "9:59"),
        (3600, "1:00:00"),
        (3725, "1:02:05"),
        (-5, "0:00"),  # a clock never runs backwards on screen
    ],
)
def test_format_duration_widens_only_at_an_hour(seconds, expected):
    assert format_duration(seconds) == expected


# --- rendering ----------------------------------------------------------------


def test_render_bar_is_plain_text():
    """No ANSI: the tee writes this same string to the console AND the log."""
    line = render_bar(0.5, 10.0, label="era5 store")
    assert "\033" not in line
    assert "\r" not in line and "\n" not in line


def test_render_bar_reports_label_percentage_and_eta():
    line = render_bar(0.25, 30.0, label="era5 store", glyphs=_GLYPHS_ASCII)
    assert line.startswith("era5 store  ")
    assert " 25.0%" in line
    # A quarter done after 30s implies 90s remaining.
    assert "eta 1:30" in line


def test_render_bar_omits_eta_before_any_progress():
    """Zero progress gives no basis for an estimate, so none is printed."""
    line = render_bar(0.0, 4.0, label="era5 store", glyphs=_GLYPHS_ASCII)
    assert "eta" not in line
    assert "0:04 elapsed" in line


def test_render_bar_final_frame_reads_as_a_summary():
    line = render_bar(1.0, 44.0, label="era5 store", glyphs=_GLYPHS_ASCII)
    assert "100.0%" in line
    assert "0:44 elapsed" in line
    assert "eta" not in line


def test_render_bar_is_solid_when_complete():
    """No half-cell cap at 1.0 -- a finished bar must be visibly full."""
    line = render_bar(1.0, 1.0, width=10, glyphs=_GLYPHS_UNICODE)
    assert _GLYPHS_UNICODE["fill"] * 10 in line
    assert _GLYPHS_UNICODE["cap"] not in line
    assert _GLYPHS_UNICODE["rest"] not in line


def test_render_bar_shows_a_cap_below_one_cell():
    """The reason the cap exists: early progress must not read as zero."""
    line = render_bar(0.05, 1.0, width=10, glyphs=_GLYPHS_UNICODE)
    assert _GLYPHS_UNICODE["cap"] in line
    assert _GLYPHS_UNICODE["fill"] not in line


def test_render_bar_keeps_constant_bar_width():
    widths = {
        len(render_bar(f / 20, 1.0, width=20, glyphs=_GLYPHS_ASCII).split("  ")[0])
        for f in range(21)
    }
    assert widths == {20}


def test_render_bar_clamps_out_of_range_fractions():
    """A dask graph can grow mid-flight; the bar must not overflow its width."""
    over = render_bar(1.4, 5.0, width=12, glyphs=_GLYPHS_ASCII)
    under = render_bar(-0.3, 5.0, width=12, glyphs=_GLYPHS_ASCII)
    assert "100.0%" in over
    assert _GLYPHS_ASCII["fill"] * 12 in over
    assert "  0.0%" in under


# --- encoding fallback --------------------------------------------------------


def test_stream_glyphs_falls_back_to_ascii_on_cp1252():
    """A legacy Windows console gets ASCII rather than mojibake."""
    assert _stream_glyphs(_FakeStream(encoding="cp1252")) is _GLYPHS_ASCII


def test_stream_glyphs_uses_unicode_on_utf8():
    assert _stream_glyphs(_FakeStream(encoding="utf-8")) is _GLYPHS_UNICODE


def test_stream_glyphs_survives_an_unknown_encoding():
    assert _stream_glyphs(_FakeStream(encoding="not-a-codec")) is _GLYPHS_ASCII


def test_ascii_glyphs_are_encodable_in_cp1252():
    """The fallback must actually be safe on the console it exists for."""
    "".join(_GLYPHS_ASCII.values()).encode("cp1252")


# --- the dask callback --------------------------------------------------------


def test_progress_writes_one_line_and_terminates_it():
    out = _FakeStream()
    bar = DaskProgress("era5 store", out=out, width=12, min_interval=0.0)
    bar._start_state({}, _state(ready=4))
    bar._posttask(None, None, {}, _state(finished=2, ready=2), None)
    bar._finish({}, _state(finished=4), False)

    text = out.getvalue()
    assert text.count("\n") == 1
    assert text.endswith("\n")
    # Every frame but the first is a redraw in place.
    assert text.count("\r") >= 2
    assert "\033" not in text


def test_progress_final_frame_is_the_summary_line():
    """What survives in the rule log: the tee keeps the last \\r segment."""
    out = _FakeStream()
    bar = DaskProgress("era5 store", out=out, width=12, min_interval=0.0)
    bar._start_state({}, _state(ready=2))
    bar._finish({}, _state(finished=2), False)

    last = [seg for seg in out.getvalue().rstrip("\n").split("\r") if seg][-1]
    assert "100.0%" in last
    assert "era5 store" in last


def test_progress_does_not_claim_success_when_the_compute_failed():
    out = _FakeStream()
    bar = DaskProgress("era5 store", out=out, width=12, min_interval=0.0)
    bar._start_state({}, _state(ready=4))
    bar._posttask(None, None, {}, _state(finished=1, ready=3), None)
    bar._finish({}, _state(finished=1, ready=3), True)

    text = out.getvalue()
    assert "100.0%" not in text
    assert text.endswith("\n")


def test_progress_throttles_redraws():
    out = _FakeStream()
    bar = DaskProgress("era5 store", out=out, width=12, min_interval=1000.0)
    bar._start_state({}, _state(ready=100))
    for done in range(1, 20):
        bar._posttask(None, None, {}, _state(finished=done, ready=100 - done), None)
    # Only the forced opening frame got through the throttle.
    assert out.getvalue().count("\r") == 1


def test_progress_pads_over_a_shortening_line():
    """A frame narrower than its predecessor must not leave a stale tail."""
    out = _FakeStream()
    bar = DaskProgress("s", out=out, width=10, min_interval=0.0)
    bar._start_state({}, _state(ready=2))
    long_line = "x" * 200
    bar._last_len = len(long_line)
    bar._draw(0.5, force=True)

    frame = out.getvalue().split("\r")[-1]
    assert len(frame) == len(long_line)
    assert frame.rstrip() != frame  # padded, not truncated


def test_progress_resolves_the_stream_at_compute_time(monkeypatch):
    """Snakemake swaps sys.stdout for the tee AFTER this module is imported."""
    bar = DaskProgress("era5 store", width=12, min_interval=0.0)
    tee = _FakeStream()
    monkeypatch.setattr(sys, "stdout", tee)
    bar._start_state({}, _state(ready=2))
    bar._finish({}, _state(finished=2), False)

    assert "era5 store" in tee.getvalue()


def test_progress_survives_a_closed_stream():
    """Teardown must never turn a finished compute into a failed rule."""
    out = _FakeStream()
    bar = DaskProgress("era5 store", out=out, width=12, min_interval=0.0)
    bar._start_state({}, _state(ready=2))
    out.close()
    bar._finish({}, _state(finished=2), False)  # must not raise


def test_progress_finish_is_silent_when_nothing_was_drawn():
    out = _FakeStream()
    bar = DaskProgress("era5 store", out=out)
    bar._finish({}, _state(), False)
    assert out.getvalue() == ""


# --- dask state accounting ----------------------------------------------------


@pytest.mark.parametrize(
    ("state", "expected"),
    [
        (_state(), 0.0),
        (_state(ready=4), 0.0),
        (_state(finished=1, ready=1, waiting=1, running=1), 0.25),
        (_state(finished=3), 1.0),
    ],
)
def test_fraction_counts_finished_over_everything_known(state, expected):
    assert _fraction(state) == expected


def test_fraction_tracks_a_graph_that_grows_mid_flight():
    """The total is not fixed up front, so the fraction can move backwards."""
    before = _fraction(_state(finished=5, ready=5))
    after = _fraction(_state(finished=5, ready=15))
    assert before == 0.5
    assert after < before


# --- hydromt's own writes -----------------------------------------------------


class _FakeWriters:
    """Stands in for ``hydromt.writers``, whose module global is what is rebound."""

    ProgressBar = "the original dask bar"


def test_hydromt_progress_rebinds_the_name_hydromt_resolves():
    module = _FakeWriters()
    with hydromt_progress("forcing", module=module):
        bar = module.ProgressBar()
    assert isinstance(bar, DaskProgress)
    assert bar._label == "forcing"


def test_hydromt_progress_tolerates_the_arguments_dask_accepts():
    """hydromt calls ``ProgressBar()`` bare today; a width is dask's vocabulary."""
    module = _FakeWriters()
    with hydromt_progress("forcing", module=module):
        assert isinstance(module.ProgressBar(minimum=0, width=40), DaskProgress)


def test_hydromt_progress_restores_the_original_binding():
    module = _FakeWriters()
    with hydromt_progress("forcing", module=module):
        pass
    assert module.ProgressBar == "the original dask bar"


def test_hydromt_progress_restores_the_binding_when_the_write_raises():
    module = _FakeWriters()
    with pytest.raises(RuntimeError):
        with hydromt_progress("forcing", module=module):
            raise RuntimeError("write failed")
    assert module.ProgressBar == "the original dask bar"


def test_hydromt_progress_is_a_no_op_when_the_name_is_gone():
    """A progress bar may never be the reason a model write fails."""

    class _Moved:
        pass

    module = _Moved()
    with hydromt_progress("forcing", module=module):
        pass
    assert not hasattr(module, "ProgressBar")


# --- a dask bar arriving from a child process ---------------------------------


def _frame(percent, filled=None):
    """One frame exactly as ``dask.diagnostics.ProgressBar`` renders it."""
    filled = percent * 40 // 100 if filled is None else filled
    bar = "#" * filled
    return f"[{bar:<40}] | {percent}% Completed | 106.82 ms\r"


def test_relay_consumes_a_dask_frame_and_redraws_it_in_the_house_style():
    out = _FakeStream()
    relay = DaskFrameRelay("forcing", out=out)

    assert relay.feed(_frame(0)) is True
    assert relay.feed(_frame(39)) is True
    relay.close()

    text = out.getvalue()
    assert "forcing" in text  # the label dask's bar never had
    assert "Completed" not in text  # dask's own wording is gone
    assert "\033" not in text
    assert text.count("\n") == 1
    assert text.endswith("\n")


def test_relay_refuses_anything_that_is_not_a_frame():
    """A row it does not consume falls through to the caller unchanged."""
    relay = DaskFrameRelay("forcing", out=_FakeStream())
    assert relay.feed("00:01:09 - forcing - Write forcing file\n") is False
    assert relay.feed("reached 100% Completed of the quota\n") is False
    assert relay.feed("\n") is False
    assert relay.active is False


def test_relay_final_frame_reads_as_a_summary():
    out = _FakeStream()
    relay = DaskFrameRelay("forcing", out=out)
    relay.feed(_frame(0))
    relay.feed(_frame(100, filled=40))
    relay.close()

    last = [seg for seg in out.getvalue().rstrip("\n").split("\r") if seg][-1]
    assert "100.0%" in last
    assert "forcing" in last


def test_relay_starts_a_new_bar_when_the_percentage_goes_backwards():
    """hydromt writes one netCDF per forcing file, each with its own bar."""
    out = _FakeStream()
    relay = DaskFrameRelay("forcing", out=out)
    relay.feed(_frame(0))
    relay.feed(_frame(100, filled=40))
    relay.feed(_frame(0))  # the second write opens
    relay.close()

    assert out.getvalue().count("\n") == 2


def test_relay_close_is_idempotent_and_silent_when_nothing_was_fed():
    out = _FakeStream()
    relay = DaskFrameRelay("forcing", out=out)
    relay.close()
    relay.close()
    assert out.getvalue() == ""


def test_relay_clamps_a_percentage_above_one_hundred():
    out = _FakeStream()
    relay = DaskFrameRelay("forcing", out=out)
    assert relay.feed("[####] | 137% Completed | 1.00 s\r") is True
    relay.close()
    assert "137" not in out.getvalue()


# --- a wflow.jl bar arriving from a child process -----------------------------
#
# The counterpart of the dask relay above. The contract is OURS on both sides
# (`shared/wflow_progress.jl` emits, this consumes), so these tests pin the wire
# format as much as the rendering.


def _cst(label, fraction):
    """One frame exactly as ``shared/wflow_progress.jl`` writes it."""
    return f"[cst-progress] {label} {fraction}\n"


def test_wflow_relay_renders_a_frame_in_the_house_style():
    relay = WflowFrameRelay(width=20)
    out = relay.feed(_cst("wflow", 0.5), stream=_FakeStream())

    assert out is not None
    assert "wflow" in out
    assert "50.0%" in out
    # A redraw, not a new row: the tee collapses `\r` frames to one log line.
    assert out.endswith("\r")
    assert "\n" not in out


def test_wflow_relay_refuses_anything_that_is_not_a_frame():
    relay = WflowFrameRelay()
    for line in (
        "00:01:09 - forcing - Write forcing file\n",
        "[cst-progress] wflow\n",  # no fraction
        "[cst-progress] 0.5\n",  # no label
        "prefixed [cst-progress] wflow 0.5\n",  # must anchor at the start
        "",
    ):
        assert relay.feed(line, stream=_FakeStream()) is None
    # Refusing a line must not open a bar, or `close` would emit a stray row.
    assert relay.active is False
    assert relay.close() is None


def test_wflow_relay_final_frame_terminates_the_row():
    relay = WflowFrameRelay(width=20)
    relay.feed(_cst("wflow", 0.4), stream=_FakeStream())
    out = relay.feed(_cst("wflow", 1.0), stream=_FakeStream())

    assert out.endswith("\n")
    assert "100.0%" in out
    # The bar closed itself, so `close` has nothing left to terminate.
    assert relay.active is False
    assert relay.close() is None


def test_wflow_relay_restarts_the_timer_when_the_member_changes():
    """Rule 3.15 runs several members in one process, each with its own bar."""
    relay = WflowFrameRelay(width=20)
    relay.feed(_cst("rlz_1_st_0", 0.9), stream=_FakeStream())
    out = relay.feed(_cst("rlz_1_st_1", 0.1), stream=_FakeStream())

    assert "rlz_1_st_1" in out
    assert "rlz_1_st_0" not in out
    assert "10.0%" in out
    # No embedded newline: `_cr_overwrite` keeps only the last non-empty `\r`
    # segment, so a prefixed row would survive as a blank line and nothing else.
    assert "\n" not in out


def test_wflow_relay_clamps_a_fraction_above_one():
    relay = WflowFrameRelay(width=20)
    out = relay.feed(_cst("wflow", 1.7), stream=_FakeStream())
    assert "170" not in out
    assert "100.0%" in out


def test_wflow_relay_close_terminates_a_bar_cut_short_exactly_once():
    """A crashed child must not leave the next row starting mid-frame."""
    relay = WflowFrameRelay(width=20)
    relay.feed(_cst("wflow", 0.3), stream=_FakeStream())

    assert relay.close() == "\n"
    assert relay.close() is None


def test_wflow_relay_falls_back_to_ascii_on_a_cp1252_console():
    relay = WflowFrameRelay(width=20)
    out = relay.feed(_cst("wflow", 0.5), stream=_FakeStream(encoding="cp1252"))
    out.encode("cp1252")  # raises if a box-drawing glyph leaked through


def test_wflow_relay_prints_one_row_for_a_run_that_reports_done_twice():
    """ProgressLogging emits `i/n == 1.0` and then a `"done"` sentinel.

    Both reach the relay as 1.0. Rendering both put the finished bar on screen
    twice (observed on rule 1.14, 2026-08-18).
    """
    relay = WflowFrameRelay(width=20)
    relay.feed(_cst("wflow", 0.5), stream=_FakeStream())
    first = relay.feed(_cst("wflow", 1.0), stream=_FakeStream())
    second = relay.feed(_cst("wflow", 1.0), stream=_FakeStream())

    assert first is not None and first.endswith(
        "\
"
    )
    # "" -- recognised, deliberately not drawn -- not None, which would let
    # the raw sentinel through to the log.
    assert second == ""


def test_wflow_relay_starts_a_fresh_bar_after_a_completed_one(tmp_path=None):
    """A second run under the SAME label is a new bar, not a suppressed repeat."""
    relay = WflowFrameRelay(width=20)
    relay.feed(_cst("wflow", 1.0), stream=_FakeStream())
    assert relay.feed(_cst("wflow", 1.0), stream=_FakeStream()) == ""

    reopened = relay.feed(_cst("wflow", 0.2), stream=_FakeStream())
    assert reopened is not None and reopened.endswith("\r")
    # ...and its own completion is rendered again, not swallowed.
    assert relay.feed(_cst("wflow", 1.0), stream=_FakeStream()) is not None
