"""Exercise interruption persistence across actual process termination.

A direct registry test cannot detect Uvicorn waiting forever for an open SSE
response before entering the app's shutdown lifespan. Use the command a hosted
pod runs, an open TCP stream, the real app lifespan/producer and a file-backed
database.
"""
from __future__ import annotations

import http.client
import json
import os
from pathlib import Path
import signal
import socket
import sqlite3
import subprocess
import sys
import threading
import time
from urllib.error import URLError
from urllib.request import urlopen

import pytest

from _hosted_command import GRACEFUL_SHUTDOWN_FLAG, graceful_shutdown_seconds, hosted_command
from cowork.handlers.turn_errors import TURN_LOST_CODE, TURN_LOST_MESSAGE


REPO = Path(__file__).resolve().parents[1]
PARTIAL = "persist this partial answer"
pytestmark = pytest.mark.skipif(os.name != "posix", reason="requires SIGTERM and SIGKILL")


def _hosted_command(port: int) -> list[str]:
    command = hosted_command()
    # Preserve all runtime options, particularly the drain timeout. Change only
    # executable, isolated app and loopback bind address.
    command[0] = sys.executable
    command[command.index("cowork.server:app")] = "turn_shutdown_process_app:app"
    command[command.index("--host") + 1] = "127.0.0.1"
    command[command.index("--port") + 1] = str(port)
    command[command.index("--log-config") + 1] = str(REPO / "cowork/logging.json")
    return command


def test_the_hosted_command_bounds_the_request_drain():
    """Without the bound, Uvicorn waits for every open answer stream before
    the lifespan's shutdown runs, so the kubelet's SIGKILL lands first and no
    interrupted turn is saved. The process tests below inherit the command."""
    assert GRACEFUL_SHUTDOWN_FLAG in hosted_command()
    assert graceful_shutdown_seconds(hosted_command()) > 0


@pytest.fixture
def server_process(tmp_path):
    env = os.environ.copy()
    env.update({
        "PYTHONPATH": os.pathsep.join((str(REPO), str(REPO / "tests"))),
        "PYTHONDONTWRITEBYTECODE": "1",
        "DATABASE_URI": f"sqlite:///{tmp_path / 'history.db'}",
        "MASTER_KEY_PATH": str(tmp_path / "master.key"),
        "COWORK_HOME": str(tmp_path),
        "COWORK_PROJECTS_DIR": str(tmp_path / "projects"),
        "COWORK_FILES_DIR": str(tmp_path / "files"),
        "COWORK_SHARED_DIR": str(tmp_path / "shared"),
        "COWORK_STREAMS_DIR": str(tmp_path / "streams"),
        "COWORK_STREAM_BACKEND": "file",
        "COWORK_TENANCY_MODE": "local",
        "COWORK_REQUIRE_AUTH": "false",
        "ENV": "test",
    })
    # Do not let a developer's Uvicorn overrides hide a missing image timeout
    # or launch multiple workers against this single-process test database.
    env = {key: value for key, value in env.items() if not key.startswith("UVICORN_")}
    env.pop("WEB_CONCURRENCY", None)
    children = []

    def start(**overrides: str):
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        log = tmp_path / f"server-{len(children)}.log"
        with log.open("wb") as output:
            process = subprocess.Popen(
                _hosted_command(port), cwd=REPO, env={**env, **overrides},
                stdout=output, stderr=subprocess.STDOUT,
            )
        children.append(process)
        base_url = f"http://127.0.0.1:{port}"
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline and process.poll() is None:
            try:
                with urlopen(f"{base_url}/__shutdown_test/ready", timeout=0.2) as response:
                    if response.status == 200:
                        return process, base_url, log
            except (OSError, URLError):
                time.sleep(0.05)
        pytest.fail(f"server did not become ready:\n{log.read_text()}")

    yield start
    for process in children:
        if process.poll() is None:
            process.kill()
        process.wait(timeout=5)


@pytest.fixture
def fake_redis():
    """A Redis the server process reaches over TCP, and a client that reads it
    after that process has exited."""
    import redis
    from fakeredis import TcpFakeServer
    from fakeredis._tcp_server import Resp2Writer, TCPFakeRequestHandler

    class Resp2Handler(TCPFakeRequestHandler):
        # fakeredis's TCP server answers in RESP3 whatever the client speaks,
        # so a nil reaches a RESP2 client as "_", which redis-py cannot parse.
        # Redis itself answers RESP2 until a client asks otherwise.
        def setup(self) -> None:
            super().setup()
            self.writer = Resp2Writer(self.client_address, self.wfile, self)

    server = TcpFakeServer(("127.0.0.1", 0), server_type="redis")
    port = server.server_address[1]
    server.RequestHandlerClass = Resp2Handler
    # A connection the exited server left in a blocking XREAD holds its
    # handler thread until the block times out; don't wait for it.
    server.daemon_threads = True
    server.block_on_close = False
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    client = redis.Redis(host="127.0.0.1", port=port, decode_responses=True)
    try:
        yield f"redis://127.0.0.1:{port}/0", client
    finally:
        client.close()
        server.shutdown()
        server.server_close()


def _sse_events(body: bytes) -> list[tuple[str, dict]]:
    """(event, data) for each frame, keepalive comments skipped."""
    events = []
    for frame in body.decode().split("\n\n"):
        fields = dict(
            line.split(": ", 1) for line in frame.splitlines() if line and not line.startswith(":")
        )
        if "event" in fields:
            events.append((fields["event"], json.loads(fields.get("data") or "{}")))
    return events


def _history(path: Path, conversation_id: str):
    # Read through an independent connection after the process exits; an
    # in-memory ORM object would not prove anything was committed durably.
    with sqlite3.connect(path) as connection:
        rows = connection.execute(
            "SELECT id, role, content, pending FROM messages "
            "WHERE conversation_id = ? ORDER BY seq",
            (conversation_id.replace("-", ""),),
        ).fetchall()
        events = connection.execute(
            "SELECT e.event_data FROM message_events e JOIN messages m ON e.message_id = m.id "
            "WHERE m.conversation_id = ? ORDER BY e.sequence_number",
            (conversation_id.replace("-", ""),),
        ).fetchall()
    return (
        [(row[0], row[1], json.loads(row[2]), bool(row[3])) for row in rows],
        [json.loads(row[0]) for row in events],
    )


def _assert_interrupted(history):
    messages, events = history
    assert [(role, content, pending) for _, role, content, pending in messages] == [
        ("user", "hello", False), ("assistant", PARTIAL, False),
    ]
    failures = [event for event in events if event.get("type") == "response.failed"]
    assert len(failures) == 1
    assert failures[0]["code"] == TURN_LOST_CODE
    assert failures[0]["error"] == TURN_LOST_MESSAGE


@pytest.mark.parametrize("stop_signal", [signal.SIGTERM, signal.SIGKILL], ids=["sigterm", "sigkill"])
@pytest.mark.parametrize("cancel_on_disconnect", [False, True], ids=["default", "disconnect-opt-in"])
def test_open_stream_survives_process_restart_without_duplicate_history(
    server_process, tmp_path, stop_signal, cancel_on_disconnect,
):
    process, base_url, log = server_process()
    with urlopen(f"{base_url}/__shutdown_test/start?cancel_on_disconnect={str(cancel_on_disconnect).lower()}", timeout=5) as stream:
        conversation_id = stream.headers["X-Test-Conversation-Id"]
        # Receiving the actual delta proves the real producer has committed its
        # pending user row and flushed the partial text into its stream file.
        while True:
            line = stream.readline()
            assert line, f"stream ended before the partial answer:\n{log.read_text()}"
            if PARTIAL.encode() in line:
                break
        process.send_signal(stop_signal)
        try:
            process.wait(timeout=9)
        except subprocess.TimeoutExpired:
            pytest.fail(f"shutdown exceeded Docker's default 10s grace:\n{log.read_text()}")
        # Keep the client connection open throughout the wait. Closing it first
        # would mask the regression by allowing Uvicorn's drain to complete.

    before_restart = _history(tmp_path / "history.db", conversation_id)
    if stop_signal == signal.SIGTERM:
        # This assertion must pass BEFORE boot recovery can repair anything.
        _assert_interrupted(before_restart)
    else:
        assert [(role, content, pending) for _, role, content, pending in before_restart[0]] == [
            ("user", "hello", True),
        ]

    restored = None
    for _ in range(2):
        restarted, _, restart_log = server_process()
        history = _history(tmp_path / "history.db", conversation_id)
        _assert_interrupted(history)
        if restored is not None:
            assert history == restored, "a second boot duplicated or rewrote the recovered turn"
        elif stop_signal == signal.SIGTERM:
            assert history == before_restart, "boot duplicated or rewrote the gracefully persisted turn"
        restored = history
        restarted.terminate()
        try:
            restarted.wait(timeout=9)
        except subprocess.TimeoutExpired:
            pytest.fail(f"restarted server failed to stop:\n{restart_log.read_text()}")


def test_sigterm_ends_an_open_remote_answer_on_its_stream_and_stops_its_controller_turn(
    server_process, fake_redis, tmp_path,
):
    """A hosted replica stopping during a remote answer. The answer is ended
    before Uvicorn's drain could cut its stream: the open stream gets
    response.failed with the interrupted message and then a clean end, the
    controller's cancel flag stops the sandbox turn, the answer is saved, and
    the process exits. Without the SIGTERM pre-drain the bound cuts the stream
    at about 2 s with no terminal frame, which reads below as IncompleteRead.

    Stubbed: the sandbox controller (the harness's stand-in takes the job and
    replies with the partial answer, then nothing) and the auth service's
    turn key and connection list. The registry, relay, Redis job, reply and
    turn streams, save and lifespan are the real ones, on a fakeredis server.
    It runs in local tenancy with no principal, where hosted runs org
    tenancy; the relay, its cancel branch and the save are the same code in
    both.
    """
    from cowork.turnqueue.redis_client import cancel_flag_key

    redis_url, redis_client = fake_redis
    process, base_url, log = server_process(
        COWORK_TURN_BACKEND="remote", COWORK_STREAM_BACKEND="redis", COWORK_TURN_REDIS_URL=redis_url,
    )
    with urlopen(f"{base_url}/__shutdown_test/start_remote", timeout=5) as stream:
        conversation_id = stream.headers["X-Test-Conversation-Id"]
        correlation_id = stream.headers["X-Test-Correlation-Id"]
        before = b""
        while PARTIAL.encode() not in before:
            line = stream.readline()
            assert line, f"stream ended before the partial answer:\n{log.read_text()}"
            before += line
        assert redis_client.get(cancel_flag_key(correlation_id)) is None
        signalled_at = time.monotonic()
        process.send_signal(signal.SIGTERM)
        try:
            after = stream.read()
        except http.client.IncompleteRead as cut:
            pytest.fail(f"the stream was cut with no terminal frame: {cut.partial!r}\n{log.read_text()}")
        ended_after = time.monotonic() - signalled_at
    try:
        process.wait(timeout=9)
    except subprocess.TimeoutExpired:
        pytest.fail(f"shutdown exceeded Docker's default 10s grace:\n{log.read_text()}")

    events = _sse_events(after)
    assert events, f"nothing after the partial answer:\n{log.read_text()}"
    last_event, last_data = events[-1]
    assert last_event == "response.failed"
    assert last_data["error"] == TURN_LOST_MESSAGE
    # Well inside the request drain's bound: the pre-drain wrote it, not the
    # lifespan's shutdown, which runs only after the drain.
    assert ended_after < graceful_shutdown_seconds(hosted_command()), ended_after
    assert redis_client.get(cancel_flag_key(correlation_id)) == "1"
    _assert_interrupted(_history(tmp_path / "history.db", conversation_id))
