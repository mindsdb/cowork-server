"""Server shutdown while a turn is in flight.

`RunRegistry.shutdown` cancels every running turn, marking each handle so
its own `CancelledError` handler persists an interrupted turn instead of
the silent, no-error-card save a user Stop gets. Mirrors
test_discarded_turn_lifecycle.py's fixtures.
"""
from __future__ import annotations

import asyncio
import contextlib
import logging
from types import SimpleNamespace
from uuid import uuid4

import fakeredis.aioredis
import pytest

import cowork.handlers.responses as responses_mod
from cowork.handlers.responses import ResponsesHandler
from cowork.handlers.turn_errors import GENERIC_TURN_ERROR_CODE, INTERRUPTED_TURN_MESSAGE
from cowork.streaming import cancellation
from cowork.streaming.registry import (
    RunHandle, RunIdentity, ServerShuttingDown, TurnLifecycle, registry,
)
from cowork.turnqueue.redis_client import cancel_flag_key

from _fakes import inline_run_db, opens

CID = "conv-shutdown-test"


@pytest.fixture(autouse=True)
def _clean_globals():
    yield
    registry.reset()


class _FakeBuffer:
    def __init__(self) -> None:
        self.records: list[tuple] = []
        self.closed: str | None = None

    @property
    def latest_seq(self) -> int:
        return len(self.records)

    async def append(self, type_, data):
        self.records.append((type_, data))
        return len(self.records)

    async def close(self, reason, extra=None):
        self.closed = reason

    @property
    def is_closed(self) -> bool:
        return self.closed is not None


def _streaming_handler(monkeypatch, saved: dict, started: asyncio.Event):
    """A handler whose turn is mid-stream (a real delta already sent) when
    cancelled — the shape a shutdown actually interrupts."""
    handler = object.__new__(ResponsesHandler)
    handler.principal = object()

    class FakeConversationService:
        def __init__(self, session):
            pass

        def get_conversation(self, conv_id):
            return SimpleNamespace(project=None)

        def save_user_message(self, conv_id, content, *, created_at=None, pending=False):
            saved["user"] = content
            return SimpleNamespace(id=uuid4())

        def clear_pending(self, conv_id, *, message_id=None):
            saved["finalized"] = True
            return True

        def save_assistant_turn(self, conv_id, text, events, harness=None, tool_rows=None):
            saved["assistant"] = text
            saved["events"] = events

    class FakeSession:
        def close(self):
            pass

    async def formatter(stream, model, event_sink):
        event_sink("response.output_text.delta", {"type": "response.output_text.delta", "delta": "partial"})
        yield "event: response.output_text.delta\ndata: {}\n\n"
        started.set()
        await asyncio.sleep(3600)  # never reached without a cancel
        yield "event: response.completed\ndata: {}\n\n"

    monkeypatch.setattr(responses_mod, "ConversationService", FakeConversationService)
    monkeypatch.setattr(responses_mod, "run_db", inline_run_db(FakeSession()))
    monkeypatch.setattr(responses_mod, "scope_from_principal", lambda p: None)
    monkeypatch.setattr(responses_mod, "get_harness", lambda name: SimpleNamespace(
        stream_response=lambda **kwargs: None, formatter=formatter,
    ))
    return handler


async def _start_streaming_turn(monkeypatch, saved, buffer):
    started = asyncio.Event()
    handler = _streaming_handler(monkeypatch, saved, started)
    lifecycle = TurnLifecycle()
    handle = await registry.start(
        conversation_id=CID, turn_id=0, open_buffer=opens(buffer),
        produce=lambda buffer: handler._run_turn(
            conv_id=uuid4(), harness_input=[], original_content="hi", model="anton",
            disabled=None, harness_name="anton", harness_id="anton", buffer=buffer,
            lifecycle=lifecycle,
        ),
        lifecycle=lifecycle,
    )
    await asyncio.wait_for(started.wait(), timeout=5)
    return handle


async def test_shutdown_cancels_and_persists_an_interrupted_turn(monkeypatch):
    saved: dict = {}
    buffer = _FakeBuffer()
    handle = await _start_streaming_turn(monkeypatch, saved, buffer)

    cancelled = await registry.shutdown(timeout_seconds=5)

    assert cancelled == 1
    assert handle.lifecycle.shutting_down is True
    assert handle.task.done()
    assert saved["assistant"] == "partial"
    assert any(
        e.get("code") == GENERIC_TURN_ERROR_CODE and e.get("error") == INTERRUPTED_TURN_MESSAGE
        for e in saved["events"]
    )
    # Distinct from a user Stop, which closes "cancelled" with no error event.
    assert buffer.closed == "interrupted"


async def test_shutdown_is_a_noop_with_nothing_running():
    assert await registry.shutdown(timeout_seconds=5) == 0


async def test_shutdown_reports_a_turn_still_unwinding_past_its_budget():
    async def _slow_to_unwind():
        # Swallows the first cancel (the shutdown's), then keeps running for
        # real past the budget below — a step that would not yield in time.
        with contextlib.suppress(asyncio.CancelledError):
            await asyncio.sleep(1)
        await asyncio.sleep(0.3)

    task = asyncio.create_task(_slow_to_unwind())
    await asyncio.sleep(0)
    lifecycle = TurnLifecycle()
    handle = RunHandle(
        conversation_id=CID, turn_id=0, buffer=_FakeBuffer(), task=task, lifecycle=lifecycle,
    )
    registry._by_cid[CID] = handle

    cancelled = await registry.shutdown(timeout_seconds=0.05)

    assert cancelled == 1
    assert lifecycle.shutting_down is True
    assert not task.done()
    await asyncio.wait_for(task, timeout=2)  # let it finish so nothing leaks past the test


@pytest.mark.parametrize("backend", ["local", "remote"])
@pytest.mark.parametrize("cause", ["watchdog", "shutdown"])
async def test_the_watchdog_and_shutdown_win_over_a_stall(monkeypatch, tmp_path, backend, cause):
    """A stall cancel set the flag, then the server ended the turn on its own
    before the producer unwound. The server's reason wins in both producers:
    the turn saves the interrupted failure, never the stall card. The flag is
    written directly, because a stall through cancel_response would end the
    turn before the second cause could overlap it."""
    import sys

    from cowork.streaming.buffer import FileStreamBuffer, turn_buffer_path
    from test_responses_remote_backend import _remote_handler_with_message_id

    registry_module = sys.modules["cowork.streaming.registry"]
    monkeypatch.setattr(registry_module, "_IDLE_POLL_SECONDS", 0.01)
    monkeypatch.setattr(
        registry_module, "_MAX_TURN_IDLE_SECONDS", 0.05 if cause == "watchdog" else 600,
    )
    saved: dict = {}
    buffer = FileStreamBuffer(turn_buffer_path(tmp_path, CID, 0))
    lifecycle = TurnLifecycle(stalled=True)
    if backend == "local":
        started = asyncio.Event()
        handler = _streaming_handler(monkeypatch, saved, started)

        def produce(buffer):
            return handler._run_turn(
                conv_id=uuid4(), harness_input=[], original_content="hi", model="anton",
                disabled=None, harness_name="anton", harness_id="anton", buffer=buffer,
                lifecycle=lifecycle,
            )
    else:
        started = asyncio.Event()
        handler = _remote_handler_with_message_id(monkeypatch, saved, assistant_message_id=uuid4())

        async def replies(**kwargs):
            yield "turn_delta", {"text": "partial"}
            started.set()
            await asyncio.sleep(3600)

        monkeypatch.setattr(responses_mod, "stream_remote_replies", replies)

        def produce(buffer):
            return handler._produce_remote(
                conv_id=uuid4(), input_text="hi", original_content="hi", model="anton",
                harness_id="anton", buffer=buffer, lifecycle=lifecycle,
            )
    handle = await registry.start(
        conversation_id=CID, turn_id=0, open_buffer=opens(buffer), produce=produce,
        lifecycle=lifecycle,
    )
    await asyncio.wait_for(started.wait(), timeout=5)
    if cause == "shutdown":
        await registry.shutdown()
    await asyncio.wait_for(handle.task, timeout=5)

    assert saved["assistant"] == "partial"
    assert saved["events"][-1]["type"] == "response.failed"
    assert saved["events"][-1]["code"] == GENERIC_TURN_ERROR_CODE
    assert saved["events"][-1]["error"] == INTERRUPTED_TURN_MESSAGE
    assert not [e for e in saved["events"] if e.get("code") == "stalled"]


async def test_a_stop_after_a_swallowed_stall_cancel_saves_as_a_stop():
    """A running cell swallows the UI's stall cancel, so the turn goes on and
    the user clicks Stop. The latest cancel decides: the turn saves as a Stop,
    not as the stall the first cancel asked for."""
    from cowork.api.v1.endpoints.responses import CancelRequest, cancel_response
    from cowork.db.scoped import LOCAL_SCOPE

    async def _swallows_the_first_cancel():
        with contextlib.suppress(asyncio.CancelledError):
            await asyncio.sleep(3600)
        await asyncio.sleep(3600)

    task = asyncio.create_task(_swallows_the_first_cancel())
    await asyncio.sleep(0)
    lifecycle = TurnLifecycle()
    registry._by_cid[CID] = RunHandle(
        conversation_id=CID, turn_id=0, buffer=_FakeBuffer(), task=task, lifecycle=lifecycle,
    )
    stall = asyncio.ensure_future(
        cancel_response(CancelRequest(conversation_id=CID, reason="stalled"), LOCAL_SCOPE)
    )
    try:
        await asyncio.sleep(0.01)
        assert lifecycle.stalled is True
        assert not task.done(), "the turn must have swallowed the stall cancel"

        await asyncio.wait_for(
            cancel_response(CancelRequest(conversation_id=CID), LOCAL_SCOPE), timeout=2,
        )

        assert lifecycle.stalled is False
    finally:
        task.cancel()
        await asyncio.gather(task, stall, return_exceptions=True)


@pytest.mark.parametrize("backend", ["local", "remote"])
@pytest.mark.parametrize("cause", ["watchdog", "shutdown", "user_stop", "ui_stall"])
async def test_only_user_stop_emits_cancelled(monkeypatch, tmp_path, backend, cause):
    import sys

    from cowork.api.v1.endpoints.responses import CancelRequest, cancel_response
    from cowork.db.scoped import LOCAL_SCOPE
    from cowork.handlers.responses import sse_from_buffer
    from cowork.streaming.buffer import FileStreamBuffer, read_records, turn_buffer_path
    from test_responses_remote_backend import _remote_handler_with_message_id

    registry_module = sys.modules["cowork.streaming.registry"]
    monkeypatch.setattr(registry_module, "_IDLE_POLL_SECONDS", 0.01)
    monkeypatch.setattr(registry_module, "_MAX_TURN_IDLE_SECONDS", 0.05 if cause == "watchdog" else 600)
    saved = {}
    buffer = FileStreamBuffer(turn_buffer_path(tmp_path, CID, 0))
    if backend == "local":
        handle = await _start_streaming_turn(monkeypatch, saved, buffer)
    else:
        started = asyncio.Event()
        handler = _remote_handler_with_message_id(monkeypatch, saved, assistant_message_id=uuid4())

        async def replies(**kwargs):
            yield "turn_delta", {"text": "partial"}
            started.set()
            await asyncio.sleep(3600)

        monkeypatch.setattr(responses_mod, "stream_remote_replies", replies)
        lifecycle = TurnLifecycle()
        handle = await registry.start(
            conversation_id=CID, turn_id=0, open_buffer=opens(buffer), lifecycle=lifecycle,
            produce=lambda buffer: handler._produce_remote(
                conv_id=uuid4(), input_text="hi", original_content="hi", model="anton",
                harness_id="anton", buffer=buffer, lifecycle=lifecycle,
            ),
        )
        await asyncio.wait_for(started.wait(), timeout=5)
    if cause == "shutdown":
        await registry.shutdown()
    elif cause == "user_stop":
        await cancel_response(CancelRequest(conversation_id=CID), LOCAL_SCOPE)
    elif cause == "ui_stall":
        # The UI's idle cut: the same endpoint as Stop, with a reason.
        await cancel_response(CancelRequest(conversation_id=CID, reason="stalled"), LOCAL_SCOPE)
    await asyncio.wait_for(handle.task, timeout=5)

    frames = [frame async for frame in sse_from_buffer(buffer)]
    assert saved["assistant"] == "partial"
    assert handle.lifecycle.timed_out is (cause == "watchdog")
    assert handle.lifecycle.shutting_down is (cause == "shutdown")
    records = list(read_records(buffer.path))
    if cause == "user_stop":
        assert frames[-1].startswith("event: response.cancelled\n")
        assert not any(e.get("type") == "response.failed" for e in saved["events"])
        assert records[-1].type == "Cancelled"
    else:
        assert frames[-1].startswith("event: response.failed\n")
        assert not any("response.cancelled" in frame for frame in frames)
        assert records[-1].type == "Interrupted"
    if cause == "ui_stall":
        # Saved as a failure with its own code, so a reload shows the stall
        # card rather than a partial answer that reads like a Stop.
        assert saved["events"][-1]["type"] == "response.failed"
        assert saved["events"][-1]["code"] == "stalled"
        assert saved["events"][-1]["error"] == (
            "The response stalled and was ended. Please try sending again."
        )
        assert handle.lifecycle.stalled is True
    elif cause != "user_stop":
        assert saved["events"][-1]["error"] == INTERRUPTED_TURN_MESSAGE


REGISTRY_LOGGER = "cowork.streaming.registry"
CANCELLATION_LOGGER = "cowork.streaming.cancellation"


@pytest.fixture
def warnings_from(monkeypatch, caplog):
    """The WARNING records of the registry's or the cancellation module's
    logger. Both are turned back on first: the migration tests' alembic env
    calls fileConfig, which disables every logger that existed before it for
    the rest of the session."""
    for name in (REGISTRY_LOGGER, CANCELLATION_LOGGER):
        monkeypatch.setattr(logging.getLogger(name), "disabled", False)
        caplog.set_level(logging.WARNING, logger=name)

    def records(name: str) -> list[logging.LogRecord]:
        return [r for r in caplog.records if r.name == name and r.levelno == logging.WARNING]

    return records


@pytest.fixture
def controller_flags(monkeypatch):
    """The Redis the controller's cancel flags are written to."""
    client = fakeredis.aioredis.FakeRedis(decode_responses=True)
    monkeypatch.setattr(cancellation, "get_redis", lambda: client)
    return client


async def _idle_turn(*, remote: bool, seen_at_cancel: list | None = None, flags=None) -> RunHandle:
    """A running turn registered by hand, whose producer records whether its
    controller flag was already written when its cancel arrived."""
    lifecycle = TurnLifecycle(identity=RunIdentity(remote=remote))

    async def produce():
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            if seen_at_cancel is not None:
                seen_at_cancel.append(
                    await flags.exists(cancel_flag_key(lifecycle.identity.correlation_id))
                )
            raise

    task = asyncio.create_task(produce())
    await asyncio.sleep(0)
    cid = f"{CID}-{lifecycle.identity.run_id}"
    handle = RunHandle(conversation_id=cid, turn_id=0, buffer=_FakeBuffer(), task=task, lifecycle=lifecycle)
    registry._by_cid[cid] = handle
    return handle


@pytest.mark.parametrize("remote", [True, False], ids=["remote", "in-process"])
async def test_shutdown_stops_a_remote_turn_on_its_controller_and_saves_it_interrupted(
    monkeypatch, tmp_path, controller_flags, remote,
):
    """A remote turn runs on the sandbox controller; cancelling its relay alone
    would leave the sandbox turn running with nobody reading its replies. So
    shutdown writes the controller's cancel flag first, keyed by the turn's
    own correlation id, and the relay's cancel branch still saves the turn as
    interrupted. An in-process turn has no controller, and gets no flag."""
    from cowork.streaming.buffer import FileStreamBuffer, read_records, turn_buffer_path
    from test_responses_remote_backend import _remote_handler_with_message_id

    saved: dict = {}
    seen_at_cancel: list[int] = []
    started = asyncio.Event()
    handler = _remote_handler_with_message_id(monkeypatch, saved, assistant_message_id=uuid4())
    lifecycle = TurnLifecycle(identity=RunIdentity(remote=remote))
    flag = cancel_flag_key(lifecycle.identity.correlation_id)

    async def replies(**kwargs):
        yield "turn_delta", {"text": "partial"}
        started.set()
        try:
            await asyncio.sleep(3600)
        except asyncio.CancelledError:
            seen_at_cancel.append(await controller_flags.exists(flag))
            raise

    monkeypatch.setattr(responses_mod, "stream_remote_replies", replies)
    buffer = FileStreamBuffer(turn_buffer_path(tmp_path, CID, 0))
    handle = await registry.start(
        conversation_id=CID, turn_id=0, open_buffer=opens(buffer), lifecycle=lifecycle,
        produce=lambda buffer: handler._produce_remote(
            conv_id=uuid4(), input_text="hi", original_content="hi", model="anton",
            harness_id="anton", buffer=buffer, lifecycle=lifecycle,
        ),
    )
    await asyncio.wait_for(started.wait(), timeout=5)

    assert await registry.shutdown() == 1

    assert handle.task.done()
    assert seen_at_cancel == [1 if remote else 0], "the flag must land before the relay's cancel"
    assert await controller_flags.get(flag) == ("1" if remote else None)
    assert await controller_flags.keys("*") == ([flag] if remote else [])
    assert saved["assistant"] == "partial"
    assert saved["events"][-1]["type"] == "response.failed"
    assert saved["events"][-1]["error"] == INTERRUPTED_TURN_MESSAGE
    assert list(read_records(buffer.path))[-1].type == "Interrupted"


async def test_shutdown_writes_no_flag_for_a_remote_turn_that_is_only_saving(controller_flags):
    """A finished answer whose save is still running has nothing left on the
    controller; completion wins, as it does for cancel_run."""
    handle = await _idle_turn(remote=True)
    await handle.buffer.close("completed")

    assert await registry.shutdown(timeout_seconds=1) == 1

    assert await controller_flags.keys("*") == []


async def test_a_second_shutdown_leaves_the_turns_the_first_ended_alone(controller_flags, monkeypatch):
    """The SIGTERM pre-drain ends the turns, then the lifespan's teardown calls
    shutdown again. That second call must not cancel a turn still saving
    again, nor write its flag again."""
    async def _slow_to_unwind():
        with contextlib.suppress(asyncio.CancelledError):
            await asyncio.sleep(3600)
        await asyncio.sleep(0.3)

    task = asyncio.create_task(_slow_to_unwind())
    await asyncio.sleep(0)
    lifecycle = TurnLifecycle(identity=RunIdentity(remote=True))
    registry._by_cid[CID] = RunHandle(
        conversation_id=CID, turn_id=0, buffer=_FakeBuffer(), task=task, lifecycle=lifecycle,
    )
    writes: list[str] = []
    real_request_cancel = cancellation.request_cancel

    async def counted(correlation_id, **kwargs):
        writes.append(correlation_id)
        await real_request_cancel(correlation_id, **kwargs)

    monkeypatch.setattr(cancellation, "request_cancel", counted)

    assert await registry.shutdown(timeout_seconds=0.05) == 1
    assert not task.done()
    assert await registry.shutdown(timeout_seconds=0.05) == 0

    assert writes == [lifecycle.identity.correlation_id]
    await asyncio.wait_for(task, timeout=2)  # it unwinds once, from the first cancel


async def test_a_failed_flag_write_still_ends_the_turn(monkeypatch, warnings_from):
    """Redis down at shutdown: the controller cannot be told, which is logged,
    and the turn is still cancelled so it saves as interrupted."""
    def unreachable():
        raise ConnectionError("redis is down")

    monkeypatch.setattr(cancellation, "get_redis", unreachable)
    handle = await _idle_turn(remote=True)

    assert await registry.shutdown(timeout_seconds=1) == 1

    assert handle.task.cancelled()
    records = warnings_from(CANCELLATION_LOGGER)
    assert [r.getMessage() for r in records] == [
        f"Could not ask the sandbox controller to stop turn {handle.run_id} for shutdown; "
        "the sandbox turn may run on."
    ]
    assert isinstance(records[0].exc_info[1], ConnectionError)


@pytest.mark.parametrize("backend", ["file", "redis"])
async def test_a_turn_unwinding_past_the_budget_is_reported_with_what_becomes_of_it(
    monkeypatch, warnings_from, backend,
):
    """Boot recovery seals only file buffers, so the warning says what happens
    to an unfinished turn on the backend this server runs."""
    monkeypatch.setenv("COWORK_STREAM_BACKEND", backend)

    async def _slow_to_unwind():
        with contextlib.suppress(asyncio.CancelledError):
            await asyncio.sleep(3600)
        await asyncio.sleep(0.3)

    task = asyncio.create_task(_slow_to_unwind())
    await asyncio.sleep(0)
    registry._by_cid[CID] = RunHandle(
        conversation_id=CID, turn_id=0, buffer=_FakeBuffer(), task=task, lifecycle=TurnLifecycle(),
    )

    await registry.shutdown(timeout_seconds=0.05)
    await asyncio.wait_for(task, timeout=2)

    fate = (
        "boot recovery does not read Redis buffers, so one that never finishes keeps its question "
        "pending, and its stream stays open until a reader's idle cut ends it"
        if backend == "redis" else "one that never finishes is sealed by the next boot's recovery"
    )
    assert [r.getMessage() for r in warnings_from(REGISTRY_LOGGER)] == [
        "1 turn(s) still unwinding past the 0.05s shutdown budget; each saves if it finishes "
        f"before the process exits, and {fate}."
    ]


async def test_a_closed_registry_refuses_a_new_turn_before_opening_anything():
    """Once the server is ending its turns for a shutdown, a new one would
    start behind that and be cut at exit. The refusal comes before a buffer
    or a producer exists, as a turn-in-progress refusal does."""
    opened: list[str] = []
    produced: list[str] = []

    async def open_buffer():
        opened.append("buffer")
        return _FakeBuffer()

    def produce(buffer):
        produced.append("producer")

    registry.close()
    with pytest.raises(ServerShuttingDown):
        await registry.start(conversation_id=CID, turn_id=0, open_buffer=open_buffer, produce=produce)

    assert opened == [] and produced == []
    assert registry.get(CID) is None


@pytest.mark.parametrize("backend", ["file", "redis"])
async def test_a_start_the_shutdown_overtakes_is_refused_before_it_registers(monkeypatch, backend):
    """close() and shutdown() run while a start() that passed the first closed
    check waits on its buffer. shutdown() cannot see that turn, so start()
    must refuse it at its last check rather than register a turn nothing
    ends. On Redis it also removes the run it published to the turn index."""
    from cowork.streaming import turn_index

    monkeypatch.setenv("COWORK_STREAM_BACKEND", backend)
    index = fakeredis.aioredis.FakeRedis(decode_responses=True)
    monkeypatch.setattr(turn_index, "get_redis", lambda: index)
    published: list[str] = []
    real_record_turn = turn_index.record_turn

    async def record_turn(conversation_id, **kwargs):
        published.append(kwargs["run_id"])
        await real_record_turn(conversation_id, **kwargs)

    monkeypatch.setattr(turn_index, "record_turn", record_turn)
    opening = asyncio.Event()
    release = asyncio.Event()
    produced: list[str] = []

    async def open_buffer():
        opening.set()
        await release.wait()
        return _FakeBuffer()

    def produce(buffer):
        produced.append("producer")

    lifecycle = TurnLifecycle(identity=RunIdentity(remote=True))
    start = asyncio.create_task(registry.start(
        conversation_id=CID, turn_id=0, open_buffer=open_buffer, produce=produce, lifecycle=lifecycle,
    ))
    await asyncio.wait_for(opening.wait(), timeout=2)

    registry.close()
    assert await registry.shutdown(timeout_seconds=1) == 0
    release.set()

    with pytest.raises(ServerShuttingDown):
        await asyncio.wait_for(start, timeout=2)
    assert produced == []
    assert registry.get(CID) is None and registry.in_flight() == []
    assert published == ([lifecycle.identity.run_id] if backend == "redis" else [])
    assert await index.keys("*") == []
