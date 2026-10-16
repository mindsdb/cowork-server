"""Native disconnect cancellation through Uvicorn and real TCP connections.

The routing model and answer model are controlled, but request parsing, routing,
SQLite persistence, detached producers, stream buffers and socket disconnects
are real. ASGITransport does not deliver the disconnects these tests exercise.
"""
from __future__ import annotations

import asyncio
import json
import socket
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field
from uuid import uuid4

import httpx
import pytest
import uvicorn

import cowork.handlers.responses as responses_mod
from cowork.handlers.response_routing import DELEGATED_AGENTIC, RouteDecision
from cowork.server import create_app
from cowork.streaming import RunHandle, registry

from _fakes import PausedHarness


async def _eventually(predicate: Callable[[], bool], *, timeout: float = 5) -> None:
    async with asyncio.timeout(timeout):
        while not predicate():
            await asyncio.sleep(0.01)


@dataclass
class _RoutingGate:
    entered: asyncio.Event = field(default_factory=asyncio.Event)
    release: asyncio.Event = field(default_factory=asyncio.Event)
    cancelled: asyncio.Event = field(default_factory=asyncio.Event)

    async def decide(self, **_kwargs):
        self.entered.set()
        try:
            await self.release.wait()
        except asyncio.CancelledError:
            self.cancelled.set()
            raise
        return RouteDecision(route=DELEGATED_AGENTIC, reason="transport test")


@dataclass
class _Server:
    port: int
    harness: PausedHarness
    gate: _RoutingGate
    cancels: list[RunHandle]
    completion_waiting: asyncio.Event
    send_completion: asyncio.Event
    disconnected: asyncio.Event
    held_request_finished: asyncio.Event
    tail_finished: asyncio.Event

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(base_url=self.url, timeout=10, trust_env=False)

    async def raw_question(self, body: dict):
        reader, writer = await asyncio.open_connection("127.0.0.1", self.port)
        payload = json.dumps(body).encode()
        writer.write(
            f"POST /api/v1/responses/ HTTP/1.1\r\nHost: 127.0.0.1:{self.port}\r\n"
            f"Content-Type: application/json\r\nContent-Length: {len(payload)}\r\n\r\n".encode()
            + payload
        )
        await writer.drain()
        return reader, writer


@pytest.fixture
async def transport_server(monkeypatch, tmp_path):
    monkeypatch.setenv("COWORK_STREAMS_DIR", str(tmp_path / "streams"))
    monkeypatch.setenv("COWORK_STREAM_BACKEND", "file")
    monkeypatch.setenv("COWORK_TURN_BACKEND", "inprocess")
    harness, gate = PausedHarness(), _RoutingGate()
    gate.release.set()
    monkeypatch.setattr(responses_mod, "get_harness", lambda _name: harness)
    monkeypatch.setattr(responses_mod, "decide_route", gate.decide)
    cancels: list[RunHandle] = []
    original_cancel = RunHandle.request_cancel

    def cancel(handle):
        cancels.append(handle)
        return original_cancel(handle)

    monkeypatch.setattr(RunHandle, "request_cancel", cancel)
    completion_waiting, send_completion = asyncio.Event(), asyncio.Event()
    disconnected = asyncio.Event()
    held_request_finished, tail_finished = asyncio.Event(), asyncio.Event()
    app = create_app()

    async def observed_app(scope, receive, send):
        # A controllable socket-write boundary makes the successor-turn race
        # deterministic without replacing either endpoint or producer.
        hold_completion = (b"x-test-hold-completion", b"true") in scope.get("headers", [])

        async def observed_receive():
            message = await receive()
            if message["type"] == "http.disconnect":
                disconnected.set()
            return message

        async def observed_send(message):
            if hold_completion and b"event: response.completed" in message.get("body", b""):
                completion_waiting.set()
                await send_completion.wait()
            await send(message)

        try:
            await app(scope, observed_receive, observed_send)
        finally:
            if hold_completion:
                held_request_finished.set()
            if scope.get("path") == "/api/v1/responses/tail":
                tail_finished.set()

    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(
        observed_app, host="127.0.0.1", port=port, lifespan="off",
        log_config=None, access_log=False, timeout_graceful_shutdown=1,
    ))
    serve = asyncio.create_task(server.serve(sockets=[sock]))
    try:
        await _eventually(lambda: server.started or serve.done())
        assert not serve.done(), "test HTTP server failed to start"
        yield _Server(
            port, harness, gate, cancels, completion_waiting, send_completion,
            disconnected, held_request_finished, tail_finished,
        )
    finally:
        gate.release.set()
        harness.release.set()
        send_completion.set()
        await registry.shutdown()
        server.should_exit = True
        await asyncio.wait_for(serve, timeout=5)
        sock.close()
        registry.reset()


def _question(conversation_id: str, **options) -> dict:
    return {"conversation": conversation_id, "input": "answer this", "stream": True, **options}


async def _created(response: httpx.Response) -> tuple[dict, AsyncIterator[str]]:
    assert response.status_code == 200
    lines = response.aiter_lines()
    async for line in lines:
        if line.startswith("data: "):
            payload = json.loads(line[6:])
            if payload.get("type") == "response.created":
                # Retain this iterator while the response is open: finalizing
                # an abandoned httpx iterator closes its underlying socket.
                return payload, lines
    pytest.fail("stream closed without response.created")


async def _finished(handle: RunHandle) -> None:
    # Observe without awaiting the task: a timeout must not cancel the producer
    # and make a broken disconnect implementation look as if it stopped it.
    await _eventually(lambda: handle.task.done())


async def test_opted_in_midstream_disconnect_cancels_once_and_tail_replays_cancel(transport_server):
    server = transport_server
    conversation_id = str(uuid4())
    async with server.client() as client:
        async with client.stream("POST", "/api/v1/responses/", json=_question(
            conversation_id, cancel_on_disconnect=True,
        )) as response:
            created, _lines = await _created(response)
            handle = registry.get(conversation_id)
            assert created["run_id"] == handle.run_id
            await asyncio.wait_for(server.harness.answering.wait(), timeout=5)
        await _finished(handle)
        replay = await client.get("/api/v1/responses/tail", params={"conversation_id": conversation_id})
    assert server.cancels == [handle]
    assert "event: response.cancelled\n" in replay.text
    assert "event: response.completed\n" not in replay.text


async def test_disconnect_before_headers_stops_routing_without_admitting_a_producer(transport_server):
    server = transport_server
    conversation_id = str(uuid4())
    server.gate.release.clear()
    reader, writer = await server.raw_question(_question(conversation_id, cancel_on_disconnect=True))
    try:
        await asyncio.wait_for(server.gate.entered.wait(), timeout=5)
        assert registry.get(conversation_id) is None
        # No headers or created event can have escaped the held routing gate.
        with pytest.raises(TimeoutError):
            await asyncio.wait_for(reader.read(1), timeout=0.05)
        writer.close()
        await writer.wait_closed()
        await asyncio.wait_for(server.gate.cancelled.wait(), timeout=5)
        assert registry.get(conversation_id) is None
        assert server.harness.started == 0
        assert server.cancels == []
    finally:
        writer.close()
        await writer.wait_closed()
        server.gate.release.set()


async def test_disconnect_after_admission_before_the_first_frame_cancels(transport_server, monkeypatch):
    server = transport_server
    before_created = asyncio.Event()

    class BeforeCreated(PausedHarness):
        async def formatter(self, stream, model, event_sink):
            before_created.set()
            await asyncio.Event().wait()
            yield "unreachable"

    monkeypatch.setattr(responses_mod, "get_harness", lambda _name: BeforeCreated())
    conversation_id = str(uuid4())
    async with server.client() as client:
        async with client.stream("POST", "/api/v1/responses/", json=_question(
            conversation_id, cancel_on_disconnect=True,
        )) as response:
            assert response.status_code == 200
            await asyncio.wait_for(before_created.wait(), timeout=5)
            handle = registry.get(conversation_id)
            assert handle.buffer.latest_seq == 0
        await _finished(handle)
        replay = await client.get("/api/v1/responses/tail", params={"conversation_id": conversation_id})
    assert server.cancels == [handle]
    assert "event: response.cancelled\n" in replay.text


@pytest.mark.parametrize("options", [{}, {"cancel_on_disconnect": False}])
async def test_default_and_false_disconnect_keep_the_producer_running(transport_server, options):
    server = transport_server
    conversation_id = str(uuid4())
    async with server.client() as client:
        async with client.stream("POST", "/api/v1/responses/", json=_question(conversation_id, **options)) as response:
            _created_event, _lines = await _created(response)
            handle = registry.get(conversation_id)
            await asyncio.wait_for(server.harness.answering.wait(), timeout=5)
        await asyncio.wait_for(server.disconnected.wait(), timeout=5)
        assert handle.is_running
        assert server.cancels == []
        server.harness.release.set()
        replay = await client.get("/api/v1/responses/tail", params={"conversation_id": conversation_id})
        await _finished(handle)
    assert "event: response.completed\n" in replay.text
    assert "event: response.cancelled\n" not in replay.text


async def test_opted_in_normal_completion_never_cancels(transport_server):
    server = transport_server
    server.harness.release.set()
    conversation_id = str(uuid4())
    async with server.client() as client:
        response = await client.post("/api/v1/responses/", json=_question(
            conversation_id, cancel_on_disconnect=True,
        ))
    handle = registry.get(conversation_id)
    await _finished(handle)
    assert response.status_code == 200
    assert "event: response.completed\n" in response.text
    assert server.cancels == []


async def test_tail_observer_disconnect_does_not_cancel_an_opted_in_question(transport_server):
    server = transport_server
    conversation_id = str(uuid4())
    async with server.client() as asker, server.client() as observer:
        async with asker.stream("POST", "/api/v1/responses/", json=_question(
            conversation_id, cancel_on_disconnect=True,
        )) as original:
            _created_event, _original_lines = await _created(original)
            handle = registry.get(conversation_id)
            async with observer.stream("GET", "/api/v1/responses/tail", params={
                "conversation_id": conversation_id,
            }) as tail:
                _replayed, _tail_lines = await _created(tail)
            await asyncio.wait_for(server.disconnected.wait(), timeout=5)
            await asyncio.wait_for(server.tail_finished.wait(), timeout=5)
            assert handle.is_running
            assert server.cancels == []
            server.harness.release.set()
            await _finished(handle)
    assert server.cancels == []


async def test_completed_question_disconnect_cannot_cancel_a_successor(transport_server):
    server = transport_server
    conversation_id = str(uuid4())
    async with server.client() as first_client, server.client() as second_client:
        async with first_client.stream("POST", "/api/v1/responses/", json=_question(
            conversation_id, cancel_on_disconnect=True,
        ), headers={"X-Test-Hold-Completion": "true"}) as first:
            _created_event, _first_lines = await _created(first)
            original = registry.get(conversation_id)
            server.harness.release.set()
            await asyncio.wait_for(server.completion_waiting.wait(), timeout=5)
            await _finished(original)
            server.harness.release.clear()
            async with second_client.stream("POST", "/api/v1/responses/", json=_question(conversation_id)) as second:
                _successor_created, _second_lines = await _created(second)
                successor = registry.get(conversation_id)
                assert successor is not original
                await first.aclose()
                await asyncio.wait_for(server.disconnected.wait(), timeout=5)
                server.send_completion.set()
                await asyncio.wait_for(server.held_request_finished.wait(), timeout=5)
                assert successor.is_running
                assert server.cancels == []
                server.harness.release.set()
                await _finished(successor)


@pytest.mark.parametrize("value", ["true", "false", 1, 0, None, [], {}])
async def test_disconnect_option_rejects_non_booleans_before_routing(transport_server, value):
    server = transport_server
    server.harness.release.set()
    async with server.client() as client:
        response = await client.post("/api/v1/responses/", json=_question(
            str(uuid4()), cancel_on_disconnect=value,
        ))
    assert response.status_code == 422
    assert not server.gate.entered.is_set()
    assert server.harness.started == 0


@pytest.mark.parametrize("stream_option", [{"stream": False}, {"stream": None}, {}])
async def test_disconnect_option_requires_streaming_before_handler_work(transport_server, monkeypatch, stream_option):
    from unittest.mock import AsyncMock

    handle = AsyncMock(return_value={"unexpected": "handler was called"})
    monkeypatch.setattr(responses_mod.ResponsesHandler, "handle", handle)
    question = {"input": "answer this", "cancel_on_disconnect": True, **stream_option}
    async with transport_server.client() as client:
        response = await client.post("/api/v1/responses/", json=question)
    assert response.status_code == 422
    handle.assert_not_awaited()


async def test_disconnect_stops_a_real_executing_scratchpad_cell_within_ten_seconds(
    transport_server, monkeypatch, tmp_path,
):
    """A live cell's heartbeat and OS process stop, and no next model step runs.

    Only ChatSession's model loop is scripted. AntonHarness, ScratchpadManager,
    the local scratchpad runtime and the executing Python child are real.
    """
    from anton.core.backends.local import local_scratchpad_runtime_factory
    from anton.core.backends.manager import ScratchpadManager

    from cowork.common.chat_session import drain_scratchpad_closes
    from cowork.harnesses.anton_harness.harness import AntonHarness

    server = transport_server
    conversation_id = str(uuid4())
    heartbeat = tmp_path / "cell-heartbeat.txt"
    next_model_step = asyncio.Event()
    processes: list[asyncio.subprocess.Process] = []
    managers: list[ScratchpadManager] = []
    code = (
        "import pathlib, time\n"
        f"heartbeat = pathlib.Path({str(heartbeat)!r})\n"
        "while True:\n"
        "    with heartbeat.open('a') as output:\n"
        "        output.write('alive\\n')\n"
        "    time.sleep(0.05)\n"
    )

    class CellSession:
        def __init__(self):
            self._scratchpads = ScratchpadManager(
                runtime_factory=local_scratchpad_runtime_factory,
                coding_provider="", coding_model="", coding_api_key="", coding_base_url="",
                workspace_path=tmp_path, session_id=conversation_id,
            )
            managers.append(self._scratchpads)

        async def turn_stream(self, *_args, **_kwargs):
            pad = await self._scratchpads.get_or_create("main")
            processes.append(pad._proc)
            await pad.execute(code)
            next_model_step.set()
            await asyncio.Event().wait()
            yield "unreachable"

    async def build_session(self, conversation, **_kwargs):
        return CellSession(), None, None

    monkeypatch.setattr(AntonHarness, "_build_chat_session", build_session)
    monkeypatch.setattr(responses_mod, "get_harness", lambda _name: AntonHarness())
    try:
        async with server.client() as client:
            async with client.stream("POST", "/api/v1/responses/", json=_question(
                conversation_id, cancel_on_disconnect=True,
            )) as response:
                _created_event, _lines = await _created(response)
                await _eventually(lambda: heartbeat.exists() and heartbeat.stat().st_size > 12, timeout=30)
                process = processes[0]
                assert process.returncode is None
                handle = registry.get(conversation_id)
                disconnected_at = asyncio.get_running_loop().time()
            await _eventually(lambda: process.returncode is not None, timeout=10)
            await _finished(handle)
            assert asyncio.get_running_loop().time() - disconnected_at < 10
            stopped_size = heartbeat.stat().st_size
            await asyncio.sleep(0.2)
            assert heartbeat.stat().st_size == stopped_size
            assert not next_model_step.is_set()
            assert server.cancels == [handle]
            replay = await client.get("/api/v1/responses/tail", params={"conversation_id": conversation_id})
            assert "event: response.cancelled\n" in replay.text
            assert "event: response.completed\n" not in replay.text
    finally:
        # The assertions above precede cleanup: teardown cannot make a broken
        # implementation appear to have killed the cell.
        for manager in managers:
            await manager.close_all()
        await drain_scratchpad_closes()
