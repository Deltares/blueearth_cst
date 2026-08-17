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
    DaskProgress,
    _fraction,
    _stream_glyphs,
    format_duration,
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
