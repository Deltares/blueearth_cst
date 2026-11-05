"""Hide a step's routine console output; never its warnings or errors.

Shared by WF3's launcher (scripts/generate_scenarios.py) and WF4's freeze
checkpoint. One implementation, because two hand-kept copies are how only one
of them got the Windows console fix.
"""

import contextlib
import io
import logging
import os
import re
import sys
import tempfile
from typing import Any

_ALERT = re.compile(r"warn|error|exception|traceback|fail", re.IGNORECASE)


class Capture:
    """Handle yielded by ``captured_output`` so a caller can flag a failed
    return code (an exception is flagged automatically)."""

    def __init__(self) -> None:
        self.failed = False

    def mark_failed(self) -> None:
        self.failed = True


def _console_log_handlers(*streams) -> dict[logging.StreamHandler, Any]:
    """Every live logging stream handler writing to one of ``streams``."""
    loggers = [logging.getLogger()] + [
        item
        for item in logging.Logger.manager.loggerDict.values()
        if isinstance(item, logging.Logger)
    ]
    return {
        handler: handler.stream
        for logger in loggers
        for handler in logger.handlers
        if isinstance(handler, logging.StreamHandler)
        and not isinstance(handler, logging.FileHandler)
        and any(handler.stream is stream for stream in streams)
    }


@contextlib.contextmanager
def captured_output(replay=None):
    """Redirect stdout/stderr to a temp file; replay it all on failure, and
    only its warning/error lines on success.

    Best-effort: a console-hygiene helper must never crash a scientific step.
    If saving or redirecting the standard fds fails, output is left
    unsuppressed rather than the error propagated. Python-level streams are
    redirected too: on a Windows console they write through WriteConsoleW,
    which fails with WinError 6 on a redirected fd. The replay happens
    only after the real fds are restored, so it reaches the console instead
    of being written back into the very file it came from.

    ``replay`` receives the text to show instead of it going to stderr -- a
    Snakefile at parse time passes :func:`console_style.defer_rows` so the rows
    print under the run header rather than above it.
    """
    try:
        saved_stdout = os.dup(1)
        saved_stderr = os.dup(2)
    except OSError:
        yield Capture()
        return
    capture = tempfile.TemporaryFile()
    handle = Capture()
    try:
        sys.stdout.flush()
        sys.stderr.flush()
        os.dup2(capture.fileno(), 1)
        os.dup2(capture.fileno(), 2)
    except OSError:
        for saved in (saved_stdout, saved_stderr):
            with contextlib.suppress(OSError):
                os.close(saved)
        yield handle
        return
    # On a real Windows console sys.stdout/sys.stderr are _WindowsConsoleIO,
    # whose WriteConsoleW on the redirected fd fails with WinError 6, so the
    # Python-level streams must point at the capture file as well.
    stream = io.TextIOWrapper(
        capture, encoding="utf-8", errors="replace", write_through=True
    )
    real_stdout, real_stderr = sys.stdout, sys.stderr
    sys.stdout = sys.stderr = stream
    # Logging handlers bound the console stream when they were created (HydroMT's
    # among them), so they need the same swap; otherwise their write raises
    # WinError 6 and logging prints `--- Logging error ---` instead of the record.
    console_handlers = _console_log_handlers(real_stdout, real_stderr)
    for handler in console_handlers:
        handler.stream = stream
    try:
        try:
            yield handle
        except BaseException:
            handle.failed = True
            raise
    finally:
        for handler, original in console_handlers.items():
            handler.stream = original
        sys.stdout, sys.stderr = real_stdout, real_stderr
        with contextlib.suppress(OSError, ValueError):
            stream.flush()
        stream.detach()
        for saved, fd in ((saved_stdout, 1), (saved_stderr, 2)):
            with contextlib.suppress(OSError):
                os.dup2(saved, fd)
        for saved in (saved_stdout, saved_stderr):
            with contextlib.suppress(OSError):
                os.close(saved)
        with contextlib.suppress(OSError):
            capture.seek(0)
            text = capture.read().decode(errors="replace")
            if not handle.failed:
                # Success hides only routine output; warnings and errors
                # are never silenced.
                text = "".join(
                    line
                    for line in text.splitlines(keepends=True)
                    if _ALERT.search(line)
                )
            if replay is not None and text:
                replay(text)
            else:
                sys.stderr.write(text)
                sys.stderr.flush()
        capture.close()
