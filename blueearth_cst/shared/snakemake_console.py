"""Launch Snakemake with routine startup chatter removed from terminal streams.

The owned runners use this entry point before Snakefile hooks are available.
Snakemake's file handlers keep their complete records. Dry runs and verbose
invocations retain native output, and unrecognized lines always pass through.
"""

import logging
import re
import sys
from contextlib import redirect_stderr
from pathlib import Path

_STARTUP = re.compile(
    r"(?:Assuming unrestricted shared filesystem usage\."
    r"|host: \S+"
    r"|Building DAG of jobs\.\.\."
    r"|Provided cores: \d+(?: \([^\n]*\))?"
    r"|Rules claiming more threads will be scaled down\.)"
)
_ANSI = re.compile(r"\x1b\[[0-9;]*m")
_PROFILE = re.compile(
    r"Using (?:workflow specific profile |profiles? ).+"
    r" for setting default command line arguments\."
)


class _StartupFilter(logging.Filter):
    """Mute only known INFO records on the terminal, never warnings or errors."""

    def filter(self, record: logging.LogRecord) -> bool:
        return record.levelno != logging.INFO or not _STARTUP.fullmatch(
            record.getMessage()
        )


class _ProfileStream:
    """Remove the CLI's unlogged profile notice, preserving other stderr writes."""

    def __init__(self, stream):
        self.stream = stream
        self.pending = ""
        self.active = True

    def __getattr__(self, name):
        return getattr(self.stream, name)

    def write(self, text: str) -> int:
        if not self.active:
            return self.stream.write(text)
        self.pending += text
        while "\n" in self.pending:
            line, self.pending = self.pending.split("\n", 1)
            if not _PROFILE.fullmatch(_ANSI.sub("", line).rstrip("\r")):
                self.stream.write(line + "\n")
        return len(text)

    def flush(self) -> None:
        # Preserve unterminated errors and prompts, even if they resemble chatter.
        if self.pending:
            self.stream.write(self.pending)
            self.pending = ""
        self.stream.flush()


def main() -> None:
    # This entry point is also executed by filename, before a Snakefile can
    # put the repository on sys.path or create its terminal handlers.
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from blueearth_cst.shared.windows_socketpair import bounded_windows_socketpair

    with bounded_windows_socketpair():
        _main()


def _main() -> None:
    from blueearth_cst.shared.console_terminal import restore_terminal_streams

    restore_terminal_streams()
    from snakemake.cli import main as snakemake_main
    from snakemake.logging import logger_manager

    if any(arg in {"-n", "--dry-run", "--dryrun", "--verbose"} for arg in sys.argv[1:]):
        snakemake_main()
        return
    # Adapt only the default terminal-handler factory, before logging starts.
    # If upstream changes this private seam, leave its native output intact.
    stderr = _ProfileStream(sys.stderr)
    factory = getattr(logger_manager, "_default_streamhandler", None)
    if callable(factory):

        def filtered_handler():
            # Profile parsing is over. From now on severity-aware filtering
            # handles the logs, including warnings that resemble profile notices.
            stderr.flush()
            stderr.active = False
            handler = factory()
            handler.addFilter(_StartupFilter())
            return handler

        logger_manager._default_streamhandler = filtered_handler
    try:
        with redirect_stderr(stderr):
            snakemake_main()
    finally:
        stderr.flush()
        if callable(factory):
            logger_manager._default_streamhandler = factory


if __name__ == "__main__":
    main()
