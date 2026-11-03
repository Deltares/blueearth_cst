"""Bound Windows fallback socket-pair setup inside the owned Snakemake CLI.

Python 3.12's fallback can wait forever in accept() after a nonblocking
loopback connect. A scheduler doing this while holding its lock cannot finish
other jobs. Keep the workaround here, without changing installed packages or
global network timeouts. Native socketpair implementations are left alone.
"""

from __future__ import annotations

import socket
import sys
import time
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager

_SETUP_TIMEOUT = 5.0
_ATTEMPTS = 3


def _socketpair(
    family: int = socket.AF_INET, type: int = socket.SOCK_STREAM, proto: int = 0
) -> tuple[socket.socket, socket.socket]:
    """Create a peer-checked TCP pair, retrying only setup timeouts.

    Each attempt has a five-second connect/accept budget. Returned sockets
    have normal blocking behavior; no data-transfer timeout is imposed.
    """
    if family not in (socket.AF_INET, socket.AF_INET6):
        raise ValueError("Only AF_INET and AF_INET6 socket families are supported")
    if type != socket.SOCK_STREAM or proto != 0:
        raise ValueError("Only SOCK_STREAM with protocol zero is supported")
    host = "127.0.0.1" if family == socket.AF_INET else "::1"
    for attempt in range(1, _ATTEMPTS + 1):
        try:
            with socket.socket(family, type, proto) as listener, ExitStack() as pair:
                listener.bind((host, 0))
                listener.listen(1)
                client = pair.enter_context(socket.socket(family, type, proto))
                deadline = time.monotonic() + _SETUP_TIMEOUT
                client.settimeout(_SETUP_TIMEOUT)
                # The listening backlog permits connect to complete before accept.
                # Unlike the stdlib fallback, do not ignore an unfinished connect.
                client.connect(listener.getsockname())
                listener.settimeout(max(0.001, deadline - time.monotonic()))
                server, _ = listener.accept()
                pair.enter_context(server)
                # Preserve CPython's peer authentication (CVE-2024-3219).
                if (
                    server.getsockname() != client.getpeername()
                    or client.getsockname() != server.getpeername()
                ):
                    raise ConnectionError("Unexpected socket-pair peer connection")
                server.settimeout(socket.getdefaulttimeout())
                client.settimeout(None)
                pair.pop_all()
                return server, client
        except TimeoutError as exc:
            if attempt == _ATTEMPTS:
                raise TimeoutError(
                    f"Windows socket-pair setup timed out after {_ATTEMPTS} attempts. "
                    "Preserve outputs and resume with --rerun-incomplete; if this "
                    "persists, investigate local loopback filtering or socket resources."
                ) from exc
            print(
                "WARNING: Windows socket-pair setup timed out "
                f"(attempt {attempt}/{_ATTEMPTS}); retrying.",
                file=sys.stderr,
                flush=True,
            )
    raise AssertionError("Socket-pair attempt loop did not return or raise")


@contextmanager
def bounded_windows_socketpair() -> Iterator[None]:
    """Temporarily replace only Windows' Python fallback during CLI execution."""
    original = socket.socketpair
    replace = sys.platform == "win32" and original is getattr(
        socket, "_fallback_socketpair", None
    )
    if replace:
        socket.socketpair = _socketpair
    try:
        yield
    finally:
        if replace:
            socket.socketpair = original
