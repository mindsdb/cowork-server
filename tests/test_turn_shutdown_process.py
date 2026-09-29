"""Exercise interruption persistence across actual process termination.

A direct registry test cannot detect Uvicorn waiting forever for an open SSE
response before entering the app's shutdown lifespan. Use the image's command,
an open TCP stream, the real app lifespan/producer and a file-backed database.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import signal
import socket
import sqlite3
import subprocess
import sys
import time
from urllib.error import URLError
from urllib.request import urlopen

import pytest


REPO = Path(__file__).resolve().parents[1]
PARTIAL = "persist this partial answer"
INTERRUPTED = "The response was interrupted before it finished. Please try again."
pytestmark = pytest.mark.skipif(os.name != "posix", reason="requires SIGTERM and SIGKILL")


def _image_command(port: int) -> list[str]:
    command = next(
        json.loads(line.removeprefix("CMD "))
        for line in (REPO / "Dockerfile").read_text().splitlines()
        if line.startswith("CMD [")
    )
    # Preserve all runtime options, particularly the actual image's drain
    # timeout. Change only executable, isolated app and loopback bind address.
    command[0] = sys.executable
    command[command.index("cowork.server:app")] = "turn_shutdown_process_app:app"
    command[command.index("--host") + 1] = "127.0.0.1"
    command[command.index("--port") + 1] = str(port)
    return command


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

    def start():
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        log = tmp_path / f"server-{len(children)}.log"
        with log.open("wb") as output:
            process = subprocess.Popen(
                _image_command(port), cwd=REPO, env=env,
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
    assert failures[0]["code"] == "anton_error"
    assert failures[0]["error"] == INTERRUPTED


@pytest.mark.parametrize("stop_signal", [signal.SIGTERM, signal.SIGKILL], ids=["sigterm", "sigkill"])
def test_open_stream_survives_process_restart_without_duplicate_history(server_process, tmp_path, stop_signal):
    process, base_url, log = server_process()
    with urlopen(f"{base_url}/__shutdown_test/start", timeout=5) as stream:
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
