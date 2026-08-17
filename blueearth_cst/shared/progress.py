"""In-place progress reporting for the long dask computes a rule waits on.

Replaces ``dask.diagnostics.ProgressBar`` where a rule blocks on one big graph
(rule 0.04's climate-store write is the case this was built for). Same
mechanism -- a dask ``Callback`` redrawing one line in place -- with three
things dask's bar does not give:

* **a label**, so a console running several sources says WHICH one is writing;
* an **ETA**, which is the number someone watching a multi-minute write wants;
* a **final frame that reads as a summary**, because the tee keeps exactly that
  one line in the rule log (``_cr_overwrite`` collapses every redraw to the
  last non-empty segment).

Three constraints shape the implementation, and each one rules out an
off-the-shelf bar:

* **No ANSI, ever.** The tee writes ONE string to both the console and the log
  file, so an escape code emitted here lands in ``logs/`` -- the defect fixed
  on 2026-08-14 when the start banner stopped colouring its own fields. Colour
  is the caller's business and there is no caller that can add it safely here.
* **Not a TTY at run time.** Under Snakemake a ``script:`` rule's stdout is the
  tee, whose ``isatty()`` is False by design. ``rich`` disables live rendering
  on a non-TTY, which would silently reduce the bar to nothing on every real
  run; both it and ``tqdm`` are also only transitively present in the env, so
  using either means a new declared dependency for a progress bar.
* **cp1252 consoles exist.** A Windows console that cannot encode ``\u2501``
  gets the ASCII rendering instead of mojibake -- the same reason
  :func:`snake_utils.rule_message` is ASCII-only. The probe reads the real
  stream's encoding, since the tee exposes none of its own.

Redraws are throttled and driven by task completion rather than by a timer
thread: a thread would interleave its writes with the rule's own logging on a
sink that makes no thread-safety promise, and a stalled compute is already
covered by the silence-triggered heartbeat in ``snake_utils``.
"""

from __future__ import annotations

import shutil
import sys
import time

from dask.callbacks import Callback

# Filled body, half-cell cap, remainder. The cap is what keeps a bar readable
# at small widths: without it a fraction under one cell renders as an empty bar
# for the first several percent.
_GLYPHS_UNICODE = {"fill": "\u2501", "cap": "\u2578", "rest": "\u2500", "sep": "\u00b7"}
_GLYPHS_ASCII = {"fill": "=", "cap": "-", "rest": "-", "sep": "|"}

_BAR_MIN = 10
_BAR_MAX = 32
_BAR_DEFAULT = 28

# Everything on the line that is not the bar or the label: percentage, the two
# clocks, the separator and the padding between fields.
_LINE_OVERHEAD = 30

_MIN_REDRAW_SECONDS = 0.1


def _stream_glyphs(stream) -> dict[str, str]:
    """Pick the glyph set ``stream`` can actually encode.

    The tee exposes no ``encoding``, so fall back to the real stdout's -- which
    is what the tee ultimately writes through. An unknown encoding degrades to
    ASCII rather than raising: this is a progress bar, and no rendering choice
    it makes may fail a run.
    """
    for candidate in (stream, sys.__stdout__):
        encoding = getattr(candidate, "encoding", None)
        if not encoding:
            continue
        try:
            "".join(_GLYPHS_UNICODE.values()).encode(encoding)
        except (LookupError, UnicodeEncodeError):
            return _GLYPHS_ASCII
        return _GLYPHS_UNICODE
    return _GLYPHS_ASCII


def format_duration(seconds: float) -> str:
    """``M:SS``, widening to ``H:MM:SS`` only once there is an hour to show."""
    seconds = max(int(seconds), 0)
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours:d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:d}:{secs:02d}"


def render_bar(
    fraction: float,
    elapsed: float,
    label: str = "",
    width: int = _BAR_DEFAULT,
    glyphs: dict[str, str] | None = None,
) -> str:
    """Render one frame as plain text, without the carriage return.

    Pure, so the rendering is testable without a dask graph. ``fraction`` is
    clamped to ``[0, 1]``: a dask state can briefly report more finished tasks
    than the total it was asked about when a graph is rewritten mid-flight, and
    a bar longer than its own width is worse than an early 100%.
    """
    glyphs = glyphs or _GLYPHS_UNICODE
    fraction = min(max(fraction, 0.0), 1.0)
    width = max(int(width), _BAR_MIN)

    filled = fraction * width
    whole = int(filled)
    # A half-filled trailing cell, but never past the end: at fraction 1.0 the
    # bar must be solid, not solid-minus-one-plus-cap.
    cap = glyphs["cap"] if (filled - whole) >= 0.5 and whole < width else ""
    rest = width - whole - len(cap)
    bar = glyphs["fill"] * whole + cap + glyphs["rest"] * max(rest, 0)

    if fraction >= 1.0:
        tail = f"{format_duration(elapsed)} elapsed"
    elif fraction > 0.0:
        eta = elapsed * (1.0 - fraction) / fraction
        tail = f"{format_duration(elapsed)} {glyphs['sep']} eta {format_duration(eta)}"
    else:
        # No fraction yet means no basis for an ETA. Printing one anyway would
        # be inventing a number, and the first frame is exactly where a reader
        # is most likely to believe it.
        tail = f"{format_duration(elapsed)} elapsed"

    prefix = f"{label}  " if label else ""
    return f"{prefix}{bar}  {fraction * 100:5.1f}%  {tail}"


def _bar_width(label: str) -> int:
    """Fit the bar to the terminal, within bounds that stay readable."""
    columns = shutil.get_terminal_size(fallback=(80, 24)).columns
    available = columns - len(label) - _LINE_OVERHEAD
    return max(_BAR_MIN, min(_BAR_MAX, available))


class DaskProgress(Callback):
    """A labelled, ETA-carrying in-place bar for one dask compute.

    Drop-in for ``dask.diagnostics.ProgressBar`` at the call site::

        with DaskProgress(f"{clim_source} store"):
            delayed_obj.compute()

    The stream is resolved when the compute STARTS, not at construction: under
    Snakemake ``sys.stdout`` is replaced by the tee, and a bar that captured
    the stream earlier would write past the log.
    """

    def __init__(
        self,
        label: str = "",
        *,
        out=None,
        width: int | None = None,
        min_interval: float = _MIN_REDRAW_SECONDS,
    ):
        super().__init__()
        self._label = label
        self._out = out
        self._width = width
        self._min_interval = min_interval
        self._start_time = 0.0
        self._last_draw = 0.0
        self._stream = None
        self._glyphs = _GLYPHS_UNICODE
        self._drawn = False
        self._last_len = 0

    def _start_state(self, dsk, state):
        self._stream = self._out if self._out is not None else sys.stdout
        self._glyphs = _stream_glyphs(self._stream)
        self._start_time = time.monotonic()
        self._last_draw = 0.0
        self._drawn = False
        self._last_len = 0
        self._draw(0.0, force=True)

    def _posttask(self, key, result, dsk, state, worker_id):
        self._draw(_fraction(state))

    def _finish(self, dsk, state, errored):
        if not self._drawn:
            return
        if errored:
            # Leave the last drawn frame standing and just close the line, so a
            # traceback starts at column 0. Claiming 100% for a compute that
            # raised would put a false success in the rule log.
            self._write("\n")
        else:
            self._draw(1.0, force=True)
            self._write("\n")
        self._flush()

    def _draw(self, fraction: float, force: bool = False) -> None:
        now = time.monotonic()
        if not force and (now - self._last_draw) < self._min_interval:
            return
        self._last_draw = now
        width = self._width or _bar_width(self._label)
        line = render_bar(
            fraction,
            now - self._start_time,
            label=self._label,
            width=width,
            glyphs=self._glyphs,
        )
        # Pad to the previous frame's length so a shortening line (a shrinking
        # ETA, an hour clock narrowing to minutes) leaves no tail behind.
        self._write("\r" + line.ljust(self._last_len))
        self._last_len = len(line)
        self._drawn = True
        self._flush()

    def _write(self, text: str) -> None:
        stream = self._stream if self._stream is not None else sys.stdout
        # A tee DROPS carriage-return frames from its console copy, because a
        # library bar cannot animate under a multi-job snakemake console. This
        # bar is the sanctioned exception and says so by duck-typing; on a real
        # terminal, or any other stream, `write` is what there is.
        writer = getattr(stream, "write_redraw", stream.write)
        try:
            writer(text)
        except (ValueError, OSError):
            # A closed or detached stream at interpreter shutdown. A progress
            # bar must never be the reason a finished compute reports failure.
            pass

    def _flush(self) -> None:
        stream = self._stream if self._stream is not None else sys.stdout
        try:
            stream.flush()
        except (ValueError, OSError):
            pass


def _fraction(state) -> float:
    """Finished tasks over all tasks dask currently knows about.

    Mirrors ``dask.diagnostics.ProgressBar``'s own accounting: the total is not
    fixed up front, because a graph can gain tasks while it runs.
    """
    done = len(state["finished"])
    total = done + sum(len(state[key]) for key in ("ready", "waiting", "running"))
    if total <= 0:
        return 0.0
    return done / total
