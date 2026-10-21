"""The export tests' no-fetch guard records each kind of call it watches."""
import socket
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
