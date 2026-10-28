"""Catch the network and process calls an export makes.

``no_fetch()`` records every audit event below, from any thread, while the
guard is open, and every ``multiprocessing`` process start: the spawn and
forkserver start methods launch a process through ``_posixsubprocess``,
which raises no audit event. Python's audit hooks never see native code act,
such as libxml2 opening a URL or a ctypes call into ``connect()``; loading a
library through ctypes is recorded, a call into one already loaded is not.
So a test pairs the guard with ``RecordingServer``: a 127.0.0.1 listener
started before the guard opens, which records every connection it accepts,
whatever made it. Point the hostile references at ``server.url`` and assert
both ``recorder.calls`` and ``server.hits()`` are empty. Native code that
reaches any other host is seen by neither.
"""
from __future__ import annotations

import multiprocessing.process
import socket
import sys
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from socketserver import TCPServer

# Audit events that reach the network, start a process or load a native
# library. ``socket.__new__`` is left out: an event loop makes a socket pair
# for itself. ``multiprocessing.start`` is no audit event; the guard's wrapper
# of ``BaseProcess.start`` records it.
WATCHED_EVENTS = frozenset(
    {
        "ctypes.dlopen",
        "multiprocessing.start",
        "socket.connect",
        "socket.bind",
        "socket.sendto",
        "socket.sendmsg",
        "socket.getaddrinfo",
        "socket.gethostbyname",
        "socket.gethostbyaddr",
        "socket.getnameinfo",
        "urllib.Request",
        "http.client.connect",
        "ftplib.connect",
        "smtplib.connect",
        "webbrowser.open",
        "subprocess.Popen",
        "os.system",
        "os.posix_spawn",
        "os.exec",
        "os.spawn",
        "os.fork",
        "os.forkpty",
        "os.startfile",
    }
)

_ARGS_REPR_MAX_CHARS = 200


@dataclass(frozen=True)
class AuditCall:
    event: str
    args: str


@dataclass
class FetchRecorder:
    calls: list[AuditCall] = field(default_factory=list)


_lock = threading.Lock()
_active: list[FetchRecorder] = []
_hook_installed = False
_start_process = multiprocessing.process.BaseProcess.start


def _audit(event: str, args: tuple) -> None:
    if event not in WATCHED_EVENTS or not _active:
        return
    call = AuditCall(event=event, args=repr(args)[:_ARGS_REPR_MAX_CHARS])
    with _lock:
        for recorder in _active:
            recorder.calls.append(call)


def _recording_start(process: multiprocessing.process.BaseProcess) -> None:
    _audit("multiprocessing.start", (process.name,))
    _start_process(process)


@contextmanager
def no_fetch() -> Iterator[FetchRecorder]:
    """Record the watched calls made from any thread until the block ends.

    An audit hook can't be removed, so one hook, and the wrapper of
    ``BaseProcess.start``, stay installed for the whole run and record only
    while a guard is open.
    """
    global _hook_installed
    recorder = FetchRecorder()
    with _lock:
        if not _hook_installed:
            sys.addaudithook(_audit)
            multiprocessing.process.BaseProcess.start = _recording_start
            _hook_installed = True
        _active.append(recorder)
    try:
        yield recorder
    finally:
        with _lock:
            _active.remove(recorder)


class _RecordingHandler(BaseHTTPRequestHandler):
    server: RecordingServer

    def _answer(self) -> None:
        self.server.record_request_line(self.client_address, self.requestline)
        self.send_response(404)
        self.send_header("Content-Length", "0")
        self.send_header("Connection", "close")
        self.end_headers()

    do_GET = do_HEAD = do_POST = do_PUT = do_PATCH = do_DELETE = do_OPTIONS = do_CONNECT = _answer

    def log_message(self, format: str, *args: object) -> None:
        pass


class RecordingServer(ThreadingHTTPServer):
    """A 127.0.0.1 listener that records every connection and answers 404.

    Start it outside ``no_fetch()``. Each hit is the request line, or "" for
    a connection that sent none.
    """

    daemon_threads = True

    def __init__(self) -> None:
        self._hits: dict[tuple[str, int], str] = {}
        self._probes: dict[tuple[str, int], threading.Event] = {}
        self._hits_lock = threading.Lock()
        super().__init__(("127.0.0.1", 0), _RecordingHandler)

    def server_bind(self) -> None:
        # HTTPServer.server_bind looks up the host's name; skip that lookup.
        TCPServer.server_bind(self)
        self.server_name, self.server_port = self.server_address[:2]

    @property
    def url(self) -> str:
        return f"http://{self.server_name}:{self.server_port}"

    def verify_request(self, request: socket.socket, client_address: tuple[str, int]) -> bool:
        with self._hits_lock:
            probe = self._probes.pop(client_address, None)
            if probe is None:
                self._hits[client_address] = ""
        if probe is not None:
            probe.set()
            return False
        return True

    def record_request_line(self, client_address: tuple[str, int], request_line: str) -> None:
        with self._hits_lock:
            self._hits[client_address] = request_line

    def hits(self) -> list[str]:
        """Every connection accepted so far, including any still queued.

        It connects once itself, to know the queue ahead of it is accepted,
        so call it after the ``no_fetch()`` block ends.
        """
        accepted = threading.Event()
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            probe.bind(("127.0.0.1", 0))
            with self._hits_lock:
                self._probes[probe.getsockname()] = accepted
            probe.connect(self.server_address)
            if not accepted.wait(timeout=5):
                raise TimeoutError("the recording server never accepted its probe")
        with self._hits_lock:
            return list(self._hits.values())


@contextmanager
def recording_server() -> Iterator[RecordingServer]:
    server = RecordingServer()
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
