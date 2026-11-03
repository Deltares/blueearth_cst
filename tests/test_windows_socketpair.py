"""Bound setup failures without weakening loopback peer checks or asyncio."""

import asyncio
import socket
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

from blueearth_cst.shared import windows_socketpair as bounded


@pytest.mark.parametrize("family", [socket.AF_INET, socket.AF_INET6])
def test_pair_transfers_both_ways_without_setup_timeout(family):
    if family == socket.AF_INET6 and not socket.has_ipv6:
        pytest.skip("IPv6 is unavailable")
    original_timeout = socket.getdefaulttimeout()
    server, client = bounded._socketpair(family)
    with server, client:
        assert server.gettimeout() == original_timeout
        assert client.gettimeout() is None
        server.sendall(b"server")
        assert client.recv(6) == b"server"
        client.sendall(b"client")
        assert server.recv(6) == b"client"
    assert socket.getdefaulttimeout() == original_timeout


@pytest.mark.parametrize("stage", ["connect", "accept"])
@pytest.mark.parametrize("persistent", [False, True])
def test_setup_timeout_retries_and_closes_failed_sockets(
    monkeypatch, capsys, stage, persistent
):
    real_socket = socket.socket
    created = []
    connects = []

    class InterruptedSocket(real_socket):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            created.append(self)

        def connect(self, address):
            connects.append(address)
            if persistent or len(connects) == 1:
                if stage == "connect":
                    raise TimeoutError("injected connect failure")
                # Reproduce the recorded unconnected client: accept gets no peer.
                # The old fallback ignores BlockingIOError and waits indefinitely.
                if self.gettimeout() is None:
                    raise BlockingIOError("injected unfinished connect")
                return None
            return super().connect(address)

    monkeypatch.setattr(socket, "socket", InterruptedSocket)
    monkeypatch.setattr(bounded, "_SETUP_TIMEOUT", 0.02)
    started = time.monotonic()
    if persistent:
        with pytest.raises(TimeoutError, match="after 3 attempts.*rerun-incomplete"):
            bounded._socketpair()
        assert len(connects) == 3
    else:
        server, client = bounded._socketpair()
        with server, client:
            client.sendall(b"recovered")
            assert server.recv(9) == b"recovered"
        assert len(connects) == 2
    assert time.monotonic() - started < 2
    assert all(sock.fileno() == -1 for sock in created)
    assert capsys.readouterr().err.count("retrying") == (2 if persistent else 1)


def test_unexpected_peer_fails_without_retry_and_closes_pair(monkeypatch):
    real_socket = socket.socket
    created = []

    class WrongPeerSocket(real_socket):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            created.append(self)

        def getpeername(self):
            return ("127.0.0.2", 1)

    monkeypatch.setattr(socket, "socket", WrongPeerSocket)
    with pytest.raises(ConnectionError, match="Unexpected socket-pair peer"):
        bounded._socketpair()
    assert len(created) == 3
    assert all(sock.fileno() == -1 for sock in created)


@pytest.mark.parametrize(
    "arguments", [(socket.AF_INET, socket.SOCK_DGRAM, 0), (socket.AF_INET, 1, 6), (-1,)]
)
def test_rejects_unsupported_arguments(arguments):
    with pytest.raises(ValueError):
        bounded._socketpair(*arguments)


@pytest.mark.parametrize("platform", ["win32", "linux"])
@pytest.mark.parametrize("native", [False, True])
def test_override_is_windows_fallback_only_and_restored(monkeypatch, platform, native):
    original = socket.socketpair
    monkeypatch.setattr(sys, "platform", platform)
    monkeypatch.setattr(socket, "_fallback_socketpair", None if native else original)
    with pytest.raises(SystemExit):
        with bounded.bounded_windows_socketpair():
            assert socket.socketpair is (
                bounded._socketpair if platform == "win32" and not native else original
            )
            raise SystemExit(7)
    assert socket.socketpair is original


def test_repeated_parallel_asyncio_loops(monkeypatch):
    # Exercise real event-loop wakeup pairs, not just TCP data exchange.
    monkeypatch.setattr(socket, "socketpair", bounded._socketpair)

    async def work(value):
        return await asyncio.to_thread(lambda: value * 2)

    def run(value):
        return asyncio.run(work(value))

    with ThreadPoolExecutor(max_workers=6) as workers:
        assert list(workers.map(run, range(120))) == [value * 2 for value in range(120)]
