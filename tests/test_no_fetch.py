"""The export tests' no-fetch guard records each kind of call it watches."""
import ctypes
import multiprocessing
import os
import socket
import struct
import subprocess
import sys
import threading
import urllib.error
import urllib.request

import pytest

from tests._no_fetch import no_fetch, recording_server


def _events(recorder) -> set[str]:
    return {call.event for call in recorder.calls}


def test_records_a_socket_connect():
    with recording_server() as server:
        with no_fetch() as recorder:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.connect(server.server_address)
        assert "socket.connect" in _events(recorder)
        assert server.hits() == [""]


def _sockaddr_in(*, host: str, port: int) -> bytes:
    """A C ``struct sockaddr_in``; BSD and macOS open it with a length byte."""
    rest = struct.pack("!H", port) + socket.inet_aton(host) + bytes(8)
    if sys.platform == "darwin" or "bsd" in sys.platform:
        return bytes([16, socket.AF_INET]) + rest
    return struct.pack("=H", socket.AF_INET) + rest


@pytest.mark.skipif(os.name == "nt", reason="calls the C library's connect() directly")
def test_the_recording_server_sees_a_native_connect_the_audit_hook_misses():
    """What the server check is for: a connection made from C raises no audit event."""
    libc = ctypes.CDLL(None, use_errno=True)  # loaded before the guard, which records a load
    with recording_server() as server:
        address = _sockaddr_in(host=server.server_name, port=server.server_port)
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            with no_fetch() as recorder:
                result = libc.connect(sock.fileno(), address, len(address))
        assert result == 0, os.strerror(ctypes.get_errno())
        assert recorder.calls == []
        assert server.hits() == [""]


def test_records_a_getaddrinfo():
    with no_fetch() as recorder:
        socket.getaddrinfo("localhost", 80)
    assert "socket.getaddrinfo" in _events(recorder)


def test_records_a_urllib_request():
    with recording_server() as server:
        with no_fetch() as recorder:
            with pytest.raises(urllib.error.HTTPError):
                urllib.request.urlopen(f"{server.url}/fetched", timeout=5)
        assert "urllib.Request" in _events(recorder)
        assert server.hits() == ["GET /fetched HTTP/1.1"]


def test_records_a_subprocess():
    with no_fetch() as recorder:
        subprocess.run([sys.executable, "-c", "pass"], check=True)
    assert "subprocess.Popen" in _events(recorder)


def test_records_a_process_start_by_spawn():
    """Spawn launches through _posixsubprocess, which raises no audit event."""
    context = multiprocessing.get_context("spawn")
    with no_fetch() as recorder:
        process = context.Process(target=int)
        process.start()
        process.join(timeout=60)
    assert process.exitcode == 0
    assert "multiprocessing.start" in _events(recorder)


def test_records_a_library_loaded_through_ctypes():
    with no_fetch() as recorder:
        ctypes.CDLL("kernel32" if os.name == "nt" else None)
    assert "ctypes.dlopen" in _events(recorder)


def test_records_calls_from_another_thread():
    def lookup() -> None:
        socket.getaddrinfo("localhost", 80)

    with no_fetch() as recorder:
        thread = threading.Thread(target=lookup)
        thread.start()
        thread.join()
    assert "socket.getaddrinfo" in _events(recorder)


def test_records_nothing_outside_the_guard():
    with no_fetch() as recorder:
        pass
    socket.getaddrinfo("localhost", 80)
    assert recorder.calls == []


def test_recording_server_sees_no_hit_when_nothing_connects():
    with recording_server() as server:
        assert server.hits() == []
        assert server.hits() == []
