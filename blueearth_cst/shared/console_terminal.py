"""Carry the launcher's terminal capability through diagnostic capture pipes."""

import os
import shutil
import sys

INTERACTIVE_ENV = "CST_CONSOLE_INTERACTIVE"


def terminal_environment() -> dict[str, str]:
    """Keep terminal mode and dimensions across nested managed launchers."""
    interactive = os.environ.get(INTERACTIVE_ENV)
    if interactive is None:
        interactive = "1" if sys.stdout.isatty() else "0"
    result = {INTERACTIVE_ENV: interactive}
    if interactive == "1":
        size = shutil.get_terminal_size(fallback=(100, 24))
        result.update(COLUMNS=str(size.columns), LINES=str(size.lines))
    return result


class _TerminalStream:
    """A pipe relayed to a live terminal; retain its ordinary stream interface."""

    def __init__(self, stream):
        self.stream = stream

    def __getattr__(self, name):
        return getattr(self.stream, name)

    def isatty(self):
        return True


def restore_terminal_streams() -> None:
    """Restore live rendering at child startup, before temporary output capture.

    The launcher owns the physical terminal and relays these streams verbatim.
    Only managed children explicitly inheriting terminal mode are adapted;
    redirected runs and deliberate rule-log tees retain their existing behavior.
    """
    if os.environ.get(INTERACTIVE_ENV) != "1":
        return
    from blueearth_cst.shared.output_capture import _console_log_handlers

    for name in ("stdout", "stderr"):
        stream = getattr(sys, name)
        if stream.isatty():
            continue
        proxy = _TerminalStream(stream)
        for handler in _console_log_handlers(stream):
            handler.stream = proxy
        setattr(sys, name, proxy)
