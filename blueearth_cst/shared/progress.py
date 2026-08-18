"""In-place progress reporting for the long dask computes a rule waits on.

Replaces ``dask.diagnostics.ProgressBar`` wherever a rule blocks on one big
graph, so every workflow's long write animates the same labelled line:

* wf0's climate-store write and wf2's series write call :class:`DaskProgress`
  directly, as a dask ``Callback``;
* wf1's and wf3's writes go through hydromt, which hard-codes
  ``progressbar=True`` and offers no way in for a caller's bar --
  :func:`hydromt_progress` rebinds the name hydromt resolves;
* rule 1.08 drives hydromt through its CLI, so the bar is drawn in a child
  process no rebind can reach -- :class:`DaskFrameRelay` re-renders the frames
  arriving on the pipe.

Same mechanism as dask's bar -- one line redrawn in place -- with three things
dask's does not give:

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

import contextlib
import re
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
        self.start()

    def _posttask(self, key, result, dsk, state, worker_id):
        self.update(_fraction(state))

    def _finish(self, dsk, state, errored):
        self.finish(errored=bool(errored))

    # -- the same bar, driven by something other than a dask graph ----------
    #
    # Rule 1.08's bar arrives as text from a child process, so there is no
    # graph to attach a Callback to. These three are that entry point; the
    # Callback hooks above are thin wrappers over them, so both drivers render
    # through exactly one implementation.

    def start(self) -> None:
        """Resolve the output stream and draw the opening frame."""
        self._stream = self._out if self._out is not None else sys.stdout
        self._glyphs = _stream_glyphs(self._stream)
        self._start_time = time.monotonic()
        self._last_draw = 0.0
        self._drawn = False
        self._last_len = 0
        self._draw(0.0, force=True)

    def update(self, fraction: float, *, force: bool = False) -> None:
        """Redraw at ``fraction``, subject to the redraw throttle."""
        self._draw(fraction, force=force)

    def finish(self, *, errored: bool = False) -> None:
        """Close the line, completing the bar unless the compute errored."""
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


# --------------------------------------------------------------------------
# hydromt's netCDF writes
# --------------------------------------------------------------------------
#
# ``hydromt.writers._nc_progress`` wraps its ``obj.compute()`` in
# ``ProgressBar() if progressbar else nullcontext()``, and
# ``hydromt_wflow.components.forcing`` passes ``progressbar=True`` unconditionally
# -- there is no flag on the way in that a caller can set. So a wf1 or wf3
# forcing write emitted dask's bar while wf0 and wf2 emitted ours, which is the
# inconsistency this section removes.
#
# The name is rebound in hydromt's OWN module namespace rather than in
# ``dask.diagnostics``: ``writers.py`` does ``from dask.diagnostics import
# ProgressBar`` at import, so the module global is what the call site resolves.
# Patching ``dask.diagnostics.ProgressBar`` instead would change nothing here
# and would silently redirect every other dask consumer in the process.
#
# This is a rebind at run time in our own code, not an edit to the vendored
# package -- the distinction AGENTS.md draws. It degrades to a no-op rather
# than raising when hydromt is absent or has moved the name, because a progress
# bar must never be the reason a model write fails.


@contextlib.contextmanager
def hydromt_progress(label: str = "", *, module=None):
    """Render hydromt's netCDF writes through :class:`DaskProgress`.

    Wrap any hydromt call that may write a lazy dataset::

        with hydromt_progress("forcing"):
            model.write()

    ``module`` is for tests, which pass a stand-in rather than importing
    hydromt. Restores the original binding on the way out, including when the
    wrapped call raises.
    """
    target = module
    if target is None:
        try:
            import hydromt.writers as target  # noqa: PLC0415  (optional, lazy)
        except Exception:  # pragma: no cover - hydromt absent or restructured
            target = None

    original = getattr(target, "ProgressBar", None) if target is not None else None
    if original is None:
        yield
        return

    def _factory(*_args, **_kwargs):
        # hydromt calls ``ProgressBar()`` with no arguments; anything it might
        # start passing (a width, a minimum) is dask's vocabulary, not ours, so
        # it is accepted and ignored rather than crashing the write.
        return DaskProgress(label)

    target.ProgressBar = _factory
    try:
        yield
    finally:
        target.ProgressBar = original


# --------------------------------------------------------------------------
# a dask bar arriving from a CHILD process
# --------------------------------------------------------------------------

#: One frame of ``dask.diagnostics.ProgressBar``, e.g.
#: ``[####      ] | 39% Completed | 106.82 ms``. Anchored on the bracketed bar
#: so an ordinary log row that happens to contain a percentage cannot match.
_DASK_FRAME_RE = re.compile(r"^\[[#\s]*\]\s*\|\s*(\d{1,3})%\s+Completed\b")


class DaskFrameRelay:
    """Re-render a child process's dask bar frames as one of ours.

    Rule 1.08 drives hydromt through its CLI, so the bar is drawn inside a
    process whose module namespace :func:`hydromt_progress` cannot reach. The
    frames still arrive on the pipe, though, and they carry the one number that
    matters -- so the parent parses the percentage out and redraws the line in
    the house style, with a label and an ETA the child's bar never had.

    Parsing another project's output is normally a bad trade. It is the right
    one here because the alternative is bootstrapping the child through a
    ``python -c`` preamble, which would stop the command being the byte-identical
    ``hydromt update`` invocation that made rule 1.07+1.08's merge
    behaviour-preserving. A frame that does not match is simply not consumed and
    falls through to the caller unchanged, so a format change upstream costs the
    bar, never the run.

    Elapsed time is measured HERE rather than read out of the frame: the child
    reports ``106.82 ms`` with no total, and an ETA needs a clock we control.
    """

    def __init__(self, label: str = "", *, out=None):
        self._label = label
        self._out = out
        self._bar: DaskProgress | None = None
        self._last_percent = -1

    @property
    def active(self) -> bool:
        return self._bar is not None

    def feed(self, chunk: str) -> bool:
        """Consume ``chunk`` if it is a bar frame; return whether it was.

        A caller streaming a child's output writes anything this refuses.
        """
        match = _DASK_FRAME_RE.match(chunk.strip("\r\n"))
        if match is None:
            return False
        percent = min(int(match.group(1)), 100)
        # A percentage that goes BACKWARDS is a second write starting its own
        # bar, not a stall: hydromt writes one netCDF per forcing file.
        if self._bar is not None and percent < self._last_percent:
            self.close()
        if self._bar is None:
            self._bar = DaskProgress(self._label, out=self._out)
            self._bar.start()
        self._last_percent = percent
        self._bar.update(percent / 100.0, force=percent >= 100)
        return True

    def close(self) -> None:
        """Finish the frame in place and end the line. Idempotent."""
        if self._bar is None:
            return
        bar, self._bar = self._bar, None
        self._last_percent = -1
        bar.finish()


# --------------------------------------------------------------------------
# a Wflow.jl progress frame arriving from a CHILD process
# --------------------------------------------------------------------------

#: One frame emitted by ``shared/wflow_progress.jl``, e.g.
#: ``[cst-progress] rlz_1_st_2 0.24561``. A sentinel rather than a
#: percentage-bearing sentence, so no ordinary log row can match it and so the
#: frame is recognisable without knowing which engine produced it.
_CST_FRAME_RE = re.compile(r"^\[cst-progress\]\s+(\S+)\s+([0-9]*\.?[0-9]+)\s*$")


class WflowFrameRelay:
    """Turn Wflow.jl progress frames into frames of the house bar.

    The counterpart of :class:`DaskFrameRelay`, for the other child process the
    toolbox waits on. It differs in one way that matters: dask's child renders a
    bar and this one reports only a NUMBER, so there is no upstream format to
    break -- ``wflow_progress.jl`` and this regex are two halves of one contract
    we own on both sides.

    Unlike the dask relay it does not write to a stream. It RETURNS the rendered
    text, because its caller is the tee itself (``snake_utils.tee_to_log``),
    which already knows how to stream carriage-return frames to the console and
    collapse them to a single row in the log. Rendering here and writing there
    keeps one implementation of each job.
    """

    def __init__(self, *, width=None):
        self._width = width
        self._label = ""
        self._start_time = 0.0
        self._glyphs = _GLYPHS_UNICODE
        self._active = False
        #: Label of the run whose bar has already been completed, so the
        #: duplicate final frame described in ``feed`` can be recognised.
        self._done_label = None

    @property
    def active(self) -> bool:
        return self._active

    def feed(self, line: str, stream=None) -> str | None:
        """Render ``line`` as a bar frame.

        Returns the rendered text, ``""`` for a frame that is recognised but
        deliberately not drawn, or ``None`` when ``line`` is not a frame at all.
        The three are distinct on purpose: a line that is not ours must reach the
        log untouched, while a frame we chose to swallow must vanish rather than
        appear as raw ``[cst-progress]`` noise. A malformed frame counts as "not
        ours", so it costs the bar rather than the row.
        """
        match = _CST_FRAME_RE.match(line.strip("\r\n"))
        if match is None:
            return None
        label, raw = match.group(1), match.group(2)
        try:
            fraction = float(raw)
        except ValueError:  # unreachable via the regex; belt and braces
            return None
        fraction = min(max(fraction, 0.0), 1.0)

        # A completed run reports 100% TWICE: ProgressLogging emits the last
        # iteration's `i/n == 1.0` and then a `"done"` sentinel, which the Julia
        # side maps to 1.0 as well. Both are genuine frames, so the emitter
        # cannot drop either without guessing which is last -- swallowing the
        # repeat here is what keeps one finished bar from printing as two
        # identical rows (observed on rule 1.14, 2026-08-18).
        if fraction >= 1.0 and not self._active and label == self._done_label:
            return ""

        # A label change means a new run has started -- rule 3.15 runs B members
        # in one process, each with its own progress. Reset the timer so the
        # next member's ETA is not computed from the previous one's start.
        #
        # The incomplete bar of the member being replaced is simply dropped: the
        # tee accumulates `\r` frames and `_cr_overwrite` keeps only the last
        # non-empty segment, so no prefix here could preserve it -- and a bar cut
        # off partway is not a row worth keeping. The `[k/N]` line the batch
        # driver prints on completion is the per-member record.
        if not self._active or label != self._label:
            self._label = label
            self._start_time = time.monotonic()
            self._glyphs = _stream_glyphs(stream if stream is not None else sys.stdout)
            self._active = True
            self._done_label = None

        elapsed = time.monotonic() - self._start_time
        width = self._width if self._width is not None else _bar_width(label)
        text = render_bar(fraction, elapsed, label, width, self._glyphs)
        if fraction >= 1.0:
            self._active = False
            self._done_label = label
            # A newline, so the completed frame stays on screen and becomes the
            # row the log keeps for this run.
            return text + "\n"
        return text + "\r"

    def close(self) -> str | None:
        """Close an unfinished bar's line; ``None`` when there is nothing open.

        Called when the child exits mid-bar -- a crashed run must not leave the
        next row starting halfway across a stale frame.
        """
        if not self._active:
            return None
        self._active = False
        return "\n"
