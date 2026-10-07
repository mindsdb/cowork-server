"""A second question into a conversation whose turn is still answering is
refused with 409 at once, before anything reads or saves it, and the running
answer goes on."""
from __future__ import annotations

import asyncio
import gc
import warnings
from uuid import uuid4

import httpx
import fakeredis.aioredis
import pytest

import cowork.handlers.responses as responses_mod
from cowork.handlers.response_routing import DELEGATED_AGENTIC, DIRECT_CONTEXT, RouteDecision
from cowork.server import create_app
from cowork.streaming import RunRegistry, TurnInProgress, get_streams_dir, registry
from cowork.streaming.buffer import FileStreamBuffer, read_records, turn_buffer_path
from cowork.streaming import backend as backend_mod, buffer as buffer_mod

from _fakes import PausedHarness, opens

REFUSAL = {
    "detail": (
        "Another question is still being answered in this conversation. "
        "Wait for it to finish, then send yours again."
    ),
    "code": "turn_in_progress",
}


@pytest.fixture(autouse=True)
def _forget_turns():
    yield
    registry.reset()


def _client() -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=create_app()), base_url="http://test")


def _ask(client: httpx.AsyncClient, conversation_id: str, text: str):
    return client.post(
        "/api/v1/responses/",
        json={"input": text, "stream": True, "conversation": conversation_id},
    )


async def _items(client: httpx.AsyncClient, conversation_id: str) -> list[tuple[str, object]]:
    items = (await client.get(f"/api/v1/conversations/{conversation_id}/items")).json()
    return [(item["role"], item["content"]) for item in items]


def _frame_types(body: str) -> list[str]:
    return [line.removeprefix("event: ") for line in body.splitlines() if line.startswith("event: ")]


class _PausedBeforeItsFirstRecord(PausedHarness):
    """Holds a started turn before it writes anything: the window in which its
    buffer file exists and is still empty."""

    def __init__(self) -> None:
        super().__init__()
        self.turn_started = asyncio.Event()
        self.write = asyncio.Event()

    async def formatter(self, stream, model, event_sink):
        self.turn_started.set()
        await self.write.wait()
        async for frame in super().formatter(stream, model, event_sink):
            yield frame


def _never_awaited(caught: list[warnings.WarningMessage]) -> list[str]:
    return [
        str(w.message) for w in caught
        if issubclass(w.category, RuntimeWarning) and "never awaited" in str(w.message)
    ]


async def test_a_second_question_is_refused_at_once_and_the_running_answer_goes_on(monkeypatch):
    gate = PausedHarness()
    monkeypatch.setattr(responses_mod, "get_harness", lambda name: gate)
    routed = []

    async def decide(**kwargs):
        routed.append(kwargs)
        return RouteDecision(route=DELEGATED_AGENTIC, reason="test")

    monkeypatch.setattr(responses_mod, "decide_route", decide)
    conversation_id = str(uuid4())  # handle() adopts an unknown UUID as the new conversation

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        async with _client() as client:
            first = asyncio.create_task(_ask(client, conversation_id, "one"))
            await asyncio.wait_for(gate.answering.wait(), timeout=10)
            running = registry.get(conversation_id)
            try:
                second = await asyncio.wait_for(_ask(client, conversation_id, "two"), timeout=2)
            except asyncio.TimeoutError:
                pytest.fail("the second question got a stream no producer writes")
            still_running = registry.get(conversation_id)
            gates_run, producers_started = len(routed), gate.started

            gate.release.set()
            answered = await asyncio.wait_for(first, timeout=10)
            items = await _items(client, conversation_id)
            # Sent once the first turn has ended. The moment its stream ends,
            # while its producer still unwinds, has a test of its own below.
            follow_up = await asyncio.wait_for(_ask(client, conversation_id, "three"), timeout=10)
        gc.collect()

    assert second.status_code == 409, second.text
    assert second.json() == REFUSAL
    assert still_running is running
    # Refused before the gate ran for it and before any producer started.
    assert (gates_run, producers_started) == (1, 1)
    assert answered.status_code == 200
    assert _frame_types(answered.text)[-1] == "response.completed"
    assert items == [("user", "one"), ("assistant", "ok")]
    assert follow_up.status_code == 200, follow_up.text
    assert _frame_types(follow_up.text)[-1] == "response.completed"
    assert _never_awaited(caught) == []


async def test_a_question_sent_the_moment_a_stream_ends_is_accepted_while_its_producer_unwinds(
    monkeypatch,
):
    """The web UI sends a queued question the moment the stream it waited on
    ends. The producer behind that stream may still be unwinding, but it has
    written its terminal record, so the question is accepted, by the early
    check and by the registry's."""
    gate = PausedHarness()
    gate.release.set()
    monkeypatch.setattr(responses_mod, "get_harness", lambda name: gate)

    async def decide(**_kwargs):
        return RouteDecision(route=DELEGATED_AGENTIC, reason="test")

    monkeypatch.setattr(responses_mod, "decide_route", decide)
    unwinding, finish_unwinding = asyncio.Event(), asyncio.Event()
    seal = responses_mod._seal_unterminated_buffer

    async def held_after_the_terminal_record(buffer, *args, **kwargs):
        if buffer.is_closed and not unwinding.is_set():
            unwinding.set()
            await finish_unwinding.wait()
        return await seal(buffer, *args, **kwargs)

    monkeypatch.setattr(responses_mod, "_seal_unterminated_buffer", held_after_the_terminal_record)
    conversation_id = str(uuid4())

    async with _client() as client:
        first = await asyncio.wait_for(_ask(client, conversation_id, "one"), timeout=10)
        await asyncio.wait_for(unwinding.wait(), timeout=10)
        running = registry.get(conversation_id)
        still_unwinding = running.is_running and running.buffer.is_closed
        try:
            follow_up = await asyncio.wait_for(_ask(client, conversation_id, "two"), timeout=10)
        finally:
            finish_unwinding.set()

    assert still_unwinding
    assert _frame_types(first.text)[-1] == "response.completed"
    assert follow_up.status_code == 200, follow_up.text
    assert _frame_types(follow_up.text)[-1] == "response.completed"


@pytest.mark.parametrize("second_route", [DELEGATED_AGENTIC, DIRECT_CONTEXT])
async def test_two_questions_racing_through_the_gate_leave_the_first_stream_whole(
    monkeypatch, second_route,
):
    """Both questions pass the early check while neither turn has started, read
    the same history and so compute the same turn number, which names the
    turn's buffer file. The second is refused once the first turn has written
    its first record: the registry refuses it under its lock before it opens
    that file, so the first stream keeps every record. The second question
    reaches the registry from the delegated path or the direct one, depending
    on its route."""
    gate = _PausedBeforeItsFirstRecord()
    monkeypatch.setattr(responses_mod, "get_harness", lambda name: gate)
    arrived = [asyncio.Event(), asyncio.Event()]
    go = [asyncio.Event(), asyncio.Event()]
    routes = [
        RouteDecision(route=DELEGATED_AGENTIC, reason="test"),
        RouteDecision(route=second_route, reason="test", model="m", text="direct"),
    ]

    async def decide(**_kwargs):
        index = sum(event.is_set() for event in arrived)
        arrived[index].set()
        await go[index].wait()
        return routes[index]

    monkeypatch.setattr(responses_mod, "decide_route", decide)
    starts = []
    start = registry.start

    async def recording_start(**kwargs):
        starts.append(kwargs["turn_id"])
        return await start(**kwargs)

    monkeypatch.setattr(registry, "start", recording_start)
    conversation_id = str(uuid4())

    async with _client() as client:
        first = asyncio.create_task(_ask(client, conversation_id, "one"))
        await asyncio.wait_for(arrived[0].wait(), timeout=10)
        second = asyncio.create_task(_ask(client, conversation_id, "two"))
        await asyncio.wait_for(arrived[1].wait(), timeout=10)

        go[0].set()
        await asyncio.wait_for(gate.turn_started.wait(), timeout=10)
        running = registry.get(conversation_id)
        path = turn_buffer_path(get_streams_dir(), conversation_id, running.turn_id)
        gate.write.set()
        await asyncio.wait_for(gate.answering.wait(), timeout=10)
        records_before_refusal = [record.type for record in read_records(path)]

        go[1].set()
        try:
            refused = await asyncio.wait_for(second, timeout=5)
        except asyncio.TimeoutError:
            pytest.fail("the second question got a stream no producer writes")
        file_after_refusal = path.exists()
        still_running = registry.get(conversation_id)

        gate.release.set()
        answered = await asyncio.wait_for(first, timeout=10)
        items = await _items(client, conversation_id)

    assert starts == [0, 0]
    assert records_before_refusal == ["sse"]
    assert refused.status_code == 409, refused.text
    assert refused.json() == REFUSAL
    assert file_after_refusal
    assert still_running is running
    assert answered.status_code == 200
    assert _frame_types(answered.text) == [
        "response.created", "response.output_text.delta", "response.completed",
    ]
    assert [record.type for record in read_records(path)] == ["sse", "sse", "sse", "Done"]
    assert gate.started == 1
    assert items == [("user", "one"), ("assistant", "ok")]


async def test_the_registry_refuses_a_second_turn_before_opening_its_buffer(tmp_path):
    runs = RunRegistry()
    path = tmp_path / "turn_000003.jsonl"
    release = asyncio.Event()

    async def answer(buffer):
        await release.wait()
        await buffer.close("completed")

    first = await runs.start(
        conversation_id="c", turn_id=3, open_buffer=opens(FileStreamBuffer(path)), produce=answer,
    )
    opened, produced = [], []

    async def open_again():
        opened.append(path)
        return FileStreamBuffer(path)

    with pytest.raises(TurnInProgress) as refused:
        await runs.start(
            conversation_id="c",
            turn_id=3,
            open_buffer=open_again,
            produce=lambda buffer: produced.append(buffer) or answer(buffer),
        )

    assert (refused.value.conversation_id, refused.value.turn_id) == ("c", 3)
    assert opened == [] and produced == []
    assert runs.get("c") is first and first.is_running
    release.set()
    await first.task
    assert [record.type for record in read_records(path)] == ["Done"]


async def test_a_turn_that_wrote_its_terminal_record_does_not_refuse_the_next(tmp_path):
    """A producer can still be unwinding after its stream ended. The web UI
    sends a queued question the moment the stream ends, so that turn counts as
    finished."""
    runs = RunRegistry()
    wrote_terminal = asyncio.Event()
    unwound = asyncio.Event()

    async def answer_then_unwind(buffer):
        await buffer.close("completed")
        wrote_terminal.set()
        await unwound.wait()

    first = await runs.start(
        conversation_id="c",
        turn_id=0,
        open_buffer=opens(FileStreamBuffer(tmp_path / "turn_000000.jsonl")),
        produce=answer_then_unwind,
    )
    await asyncio.wait_for(wrote_terminal.wait(), timeout=5)
    assert first.is_running

    async def answer(buffer):
        await buffer.close("completed")

    second = await runs.start(
        conversation_id="c",
        turn_id=2,
        open_buffer=opens(FileStreamBuffer(tmp_path / "turn_000002.jsonl")),
        produce=answer,
    )

    assert second is not first
    assert runs.get("c") is second
    unwound.set()
    await asyncio.gather(first.task, second.task)


@pytest.mark.parametrize(
    "frame,reason,task_cancelled",
    [("response.failed", "error", False), ("response.failed", "error", True),
     ("response.cancelled", "cancelled", True)],
)
async def test_a_follow_up_waits_for_the_redis_terminal_instead_of_being_refused(
    monkeypatch, frame, reason, task_cancelled,
):
    redis = fakeredis.aioredis.FakeRedis(decode_responses=True)
    monkeypatch.setattr(buffer_mod, "get_redis", lambda: redis)
    monkeypatch.setenv("COWORK_STREAM_BACKEND", "redis")
    runs = RunRegistry()
    final_written, release = asyncio.Event(), asyncio.Event()
    expire = redis.expire

    async def held_expiry(key, seconds):
        if not final_written.is_set():
            final_written.set()
            await release.wait()
        return await expire(key, seconds)

    monkeypatch.setattr(redis, "expire", held_expiry)

    async def refused(buffer):
        await buffer.append("sse", {"sse": f"event: {frame}\ndata: {{}}\n\n"})
        await buffer.close(reason)
        if task_cancelled:
            raise asyncio.CancelledError()

    first = await runs.start(
        conversation_id="c", turn_id=0,
        open_buffer=lambda: backend_mod.new_buffer("c", 0), produce=refused,
    )
    await asyncio.wait_for(final_written.wait(), timeout=5)

    async def answered(buffer):
        await buffer.append("sse", {"sse": "event: response.completed\ndata: {}\n\n"})
        await buffer.close("completed")

    following = asyncio.create_task(runs.start(
        conversation_id="c", turn_id=0,
        open_buffer=lambda: backend_mod.new_buffer("c", 0), produce=answered,
    ))
    try:
        await asyncio.sleep(0.05)
        assert not first.is_answering
        assert not following.done(), "a queued follow-up must wait rather than receive 409"
    finally:
        release.set()
    second = await asyncio.wait_for(following, timeout=5)
    await asyncio.gather(first.task, second.task, return_exceptions=True)
    records = [record async for record in second.buffer.tail()]
    assert [record.type for record in records] == ["sse", "Done"]
    assert "response.completed" in records[0].data["sse"]
    await redis.aclose()


async def test_a_slow_buffer_open_does_not_block_another_conversation(tmp_path):
    runs = RunRegistry()
    opening, release_open, release_turn = asyncio.Event(), asyncio.Event(), asyncio.Event()

    async def slow_open():
        opening.set()
        await release_open.wait()
        return FileStreamBuffer(tmp_path / "one.jsonl")

    async def answer(buffer):
        await release_turn.wait()
        await buffer.close("completed")

    first = asyncio.create_task(runs.start(
        conversation_id="one", turn_id=0, open_buffer=slow_open, produce=answer,
    ))
    await opening.wait()
    try:
        second = await asyncio.wait_for(runs.start(
            conversation_id="two", turn_id=0,
            open_buffer=opens(FileStreamBuffer(tmp_path / "two.jsonl")), produce=answer,
        ), timeout=1)
    finally:
        release_open.set()
        release_turn.set()
        first_handle = await first
        await first_handle.task
    await second.task
