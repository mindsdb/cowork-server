"""An answer holds a pooled connection only while one of its database units
runs, and never waits for one on the event loop's thread. A save that fails at
the end of the turn ends the stream as failed, and a cancel never splits an
answer from the frame that reports it.

Turns run through POST /api/v1/responses/ and the real AntonHarness, with only
anton's ChatSession replaced (harness.build_chat_session), on the conftest's
SQLite file database, whose QueuePool waits for a connection the way
Postgres's does.
"""
from __future__ import annotations

import asyncio
import json
import shutil
import threading
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from uuid import UUID, uuid4

import anyio
import httpx
import pytest
from sqlalchemy import event
from sqlalchemy.exc import TimeoutError as PoolTimeoutError

import cowork.db.session as db_session
import cowork.handlers.responses as responses_mod
import cowork.services.history_recall as history_recall
from cowork.db import units
from cowork.db.scoped import LOCAL_SCOPE, SYSTEM_SCOPE
from cowork.db.units import unit_session
from cowork.handlers.response_routing import DELEGATED_AGENTIC, RouteDecision
from cowork.handlers.turn_errors import INTERRUPTED_TURN_MESSAGE, SERVER_BUSY_CODE
from cowork.harnesses.anton_harness import harness as harness_mod
from cowork.models.schedule import Schedule, ScheduleRun
from cowork.schemas.schedules import RunStatus
from cowork.server import create_app
from cowork.services.conversations import ConversationService
from cowork.services.projects import GENERAL_PROJECT_ID
from cowork.services.schedules import ScheduleRunService, ScheduleService
from cowork.streaming import discard_conversation, registry
from cowork.streaming.buffer import FileStreamBuffer, read_records

from _fakes import PausedModel

# Seconds; the one_connection_pool fixture's POOL_TIMEOUT.
WAIT = 2


@pytest.fixture(autouse=True)
def _turn_runs_in_this_process(monkeypatch):
    """The real AntonHarness, a gate that delegates at once, and no connector
    vault. The key only lets the harness build its model client: the stand-in
    ChatSession never calls it."""
    monkeypatch.setattr(responses_mod, "get_harness", lambda name: harness_mod.AntonHarness())

    async def delegate(**_kwargs):
        return RouteDecision(route=DELEGATED_AGENTIC, reason="test")

    async def no_vault(_scope):
        return None

    monkeypatch.setattr(responses_mod, "decide_route", delegate)
    monkeypatch.setattr(responses_mod, "register_vault_secrets", no_vault)
    monkeypatch.setattr("anton.core.datasources.data_vault.LocalDataVault", None)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-only-key")
    monkeypatch.setenv("ANTON_SCRATCHPAD_PERSIST_SESSION", "false")
    yield
    registry.reset()


def _app_engine():
    return db_session.get_engine(db_session.settings.database.uri)


def _client() -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=create_app()), base_url="http://test")


def _ask(client: httpx.AsyncClient, conversation_id: UUID, text: str):
    return client.post(
        "/api/v1/responses/",
        json={"input": text, "stream": True, "conversation": str(conversation_id)},
    )


def _frames(body: str) -> list[tuple[str, dict]]:
    frames = []
    for block in body.split("\n\n"):
        lines = block.strip().splitlines()
        kind = next((line.removeprefix("event: ") for line in lines if line.startswith("event: ")), None)
        data = next((line.removeprefix("data: ") for line in lines if line.startswith("data: ")), None)
        if kind and data:
            frames.append((kind, json.loads(data)))
    return frames


@dataclass(frozen=True)
class _Row:
    role: str
    pending: bool
    seq: int


def _rows(conversation_id: UUID) -> list[_Row]:
    with unit_session(scope=LOCAL_SCOPE) as session:
        messages = ConversationService(session).get_ordered_messages(conversation_id, include_pending=True)
        return [
            _Row(role=str(getattr(m.role, "value", m.role)), pending=m.pending, seq=m.seq)
            for m in messages
        ]


def _conversation(*, turns: list[tuple[str, str]] = (), compact_after: int | None = None) -> UUID:
    """A saved conversation holding the given (question, answer) turns. With
    ``compact_after``, a saved summary covers the first that many messages, so
    the recall_history tool has an archive to search."""
    with unit_session(scope=LOCAL_SCOPE) as session:
        service = ConversationService(session)
        conversation = service.create_conversation("pool test", project_id=GENERAL_PROJECT_ID)
        for question, answer in turns:
            service.save_user_message(conversation.id, question)
            service.save_assistant_turn(conversation.id, answer, [], harness="anton")
        if compact_after is not None:
            covered = service.get_ordered_messages(conversation.id)[compact_after - 1]
            service.update_history_compaction(conversation.id, "SUMMARY", covered.id)
        return conversation.id


def _terminal(conversation_id: UUID) -> str:
    handle = registry.get(str(conversation_id))
    records = list(read_records(handle.buffer.path))
    assert records and records[-1].is_terminal, records
    return records[-1].type


async def _until(condition, *, timeout: float = 10.0) -> None:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while not condition():
        assert loop.time() < deadline, "timed out waiting"
        await asyncio.sleep(0.01)


async def _take_the_only_connection(engine):
    """The pool's only connection, taken from a worker thread. A turn that
    holds a connection while its model answers keeps it from us."""
    try:
        return await asyncio.to_thread(engine.connect)
    except PoolTimeoutError:
        pytest.fail("the turn held the pool's only connection while its model answered")


def _state_changes_logged(caplog) -> list[str]:
    return [
        record.getMessage() for record in caplog.records
        if record.exc_info and type(record.exc_info[1]).__name__ == "IllegalStateChangeError"
    ]


# ── T1: no connection held between units ────────────────────────────────────


async def test_an_answer_holds_no_connection_between_its_database_units(monkeypatch):
    """Sampled while the gate's model call is paused, at every frame the turn
    appends, inside a recall_history call once its archive is read, and while
    the agent's model is paused."""
    pool = _app_engine().pool
    conversation_id = _conversation(
        turns=[("the staging password is hunter2", "noted"), ("rename the file", "renamed")],
        compact_after=2,
    )
    held: dict[str, list[int]] = {"gate": [], "appends": [], "recall": [], "model": []}

    gate_entered, gate_release = asyncio.Event(), asyncio.Event()

    async def paused_gate(**_kwargs):
        gate_entered.set()
        await gate_release.wait()
        return RouteDecision(route=DELEGATED_AGENTIC, reason="test")

    monkeypatch.setattr(responses_mod, "decide_route", paused_gate)

    append = FileStreamBuffer.append

    async def sampled_append(self, type_, data):
        held["appends"].append(pool.checkedout())
        return await append(self, type_, data)

    monkeypatch.setattr(FileStreamBuffer, "append", sampled_append)

    search = history_recall.search_turns

    def sampled_search(messages, query, **kwargs):
        held["recall"].append(pool.checkedout())
        return search(messages, query, **kwargs)

    monkeypatch.setattr(history_recall, "search_turns", sampled_search)
    recalled: list[str] = []

    async def recall(session):
        tool = next(t for t in session.config.tools if t.name == "recall_history")
        recalled.append(await tool.handler(None, {"query": "staging password"}))

    model = PausedModel(before_answer=recall)
    monkeypatch.setattr(harness_mod, "build_chat_session", model.build)

    async with _client() as client:
        answer = asyncio.create_task(_ask(client, conversation_id, "what was it again?"))
        await asyncio.wait_for(gate_entered.wait(), timeout=10)
        held["gate"].append(pool.checkedout())
        gate_release.set()
        await asyncio.wait_for(model.answering.wait(), timeout=10)
        held["model"].append(pool.checkedout())
        model.release.set()
        answered = await asyncio.wait_for(answer, timeout=10)

    assert answered.status_code == 200, answered.text
    assert _frames(answered.text)[-1][0] == "response.completed"
    assert recalled and "hunter2" in recalled[0], recalled
    assert held["appends"], "the turn appended no frame"
    assert held == {
        "gate": [0], "appends": [0] * len(held["appends"]), "recall": [0], "model": [0],
    }


# ── T2: no checkout on the event loop, one connection per thread ─────────────


@dataclass(frozen=True)
class _Checkout:
    thread: int
    sites: frozenset[str]


# The harness's end-of-turn writes run in its generator's `finally`, where an
# await is skipped on cancellation, so they stay synchronous on short sessions.
_LOOP_CHECKOUTS_ALLOWED = frozenset({"_persist_history_compaction", "_index_new_slugs"})


async def test_a_turn_checks_out_off_the_event_loop_one_connection_per_thread(monkeypatch, tmp_path):
    """A full turn, with a compaction and a new artifact so that both of the
    harness's end-of-turn writes run: only those two check out on the event
    loop's thread, and no thread ever holds two connections at once."""
    engine = _app_engine()
    loop_thread = threading.get_ident()
    conversation_id = _conversation(turns=[("q1", "a1"), ("q2", "a2")])
    with unit_session(scope=LOCAL_SCOPE) as session:
        project_path = Path(ConversationService(session).get_conversation(conversation_id).project.path)
    slug = f"pool-test-{uuid4().hex[:8]}"
    artifact = project_path / ".anton" / "artifacts" / slug

    async def compact_and_make_an_artifact(session):
        session.last_compaction = {"summary": "SUMMARY", "covered_through": 2}
        artifact.mkdir(parents=True)
        (artifact / "index.html").write_text("<html></html>")
        (artifact / "metadata.json").write_text(json.dumps({"slug": slug, "name": slug, "type": "html-app"}))
        session.artifacts_touched = {slug}

    model = PausedModel(before_answer=compact_and_make_an_artifact)
    model.release.set()
    monkeypatch.setattr(harness_mod, "build_chat_session", model.build)

    checkouts: list[_Checkout] = []
    out: dict[int, int] = {}
    most_at_once: dict[int, int] = {}

    def on_checkout(dbapi_connection, record, proxy):
        thread = threading.get_ident()
        out[thread] = out.get(thread, 0) + 1
        most_at_once[thread] = max(most_at_once.get(thread, 0), out[thread])
        checkouts.append(_Checkout(
            thread=thread, sites=frozenset(frame.name for frame in traceback.extract_stack()),
        ))

    def on_checkin(dbapi_connection, record):
        thread = threading.get_ident()
        out[thread] = out.get(thread, 0) - 1

    event.listen(engine, "checkout", on_checkout)
    event.listen(engine, "checkin", on_checkin)
    try:
        async with _client() as client:
            answered = await asyncio.wait_for(_ask(client, conversation_id, "q3"), timeout=10)
    finally:
        event.remove(engine, "checkout", on_checkout)
        event.remove(engine, "checkin", on_checkin)
        shutil.rmtree(artifact, ignore_errors=True)

    assert answered.status_code == 200, answered.text
    assert _frames(answered.text)[-1][0] == "response.completed"
    on_the_loop = [c for c in checkouts if c.thread == loop_thread]
    unexpected = [c for c in on_the_loop if not c.sites & _LOOP_CHECKOUTS_ALLOWED]
    assert unexpected == [], f"{len(unexpected)} other checkout(s) on the event loop's thread"
    assert max(most_at_once.values()) == 1, most_at_once
    allowed_sites = sorted(site for c in on_the_loop for site in c.sites & _LOOP_CHECKOUTS_ALLOWED)
    assert allowed_sites == sorted(_LOOP_CHECKOUTS_ALLOWED), "both end-of-turn writes should have run"


# ── T5: a failed save ends the stream as failed ──────────────────────────────


@pytest.mark.parametrize("failure", ["no_connection_frees", "the_save_raises"])
async def test_a_save_that_fails_at_turn_end_ends_the_stream_as_failed(
    one_connection_pool, monkeypatch, failure,
):
    """Either no connection frees within POOL_TIMEOUT for the turn-end unit,
    or the save itself raises. The answer was streamed, but it is not in the
    database, so the stream ends with response.failed and an error terminal
    record, the question stays pending, and no assistant row exists."""
    conversation_id = uuid4()
    model = PausedModel()
    monkeypatch.setattr(harness_mod, "build_chat_session", model.build)
    if failure == "the_save_raises":

        def disk_full(*_args, **_kwargs):
            raise RuntimeError("disk full")

        monkeypatch.setattr(ConversationService, "save_assistant_turn", disk_full)

    held = None
    async with _client() as client:
        answer = asyncio.create_task(_ask(client, conversation_id, "hi"))
        await asyncio.wait_for(model.answering.wait(), timeout=10)
        if failure == "no_connection_frees":
            held = await _take_the_only_connection(one_connection_pool)
        model.release.set()
        try:
            answered = await asyncio.wait_for(answer, timeout=5 * WAIT)
        finally:
            if held is not None:
                held.close()

    frames = _frames(answered.text)
    kinds = [kind for kind, _ in frames]
    assert "response.output_text.delta" in kinds
    assert kinds[-1] == "response.failed", kinds
    failed = frames[-1][1]
    if failure == "no_connection_frees":
        assert failed["code"] == SERVER_BUSY_CODE
        assert failed["retry_after"] == WAIT
    else:
        assert failed["code"] == "anton_error"
    assert "assistant_message_id" not in failed
    assert _terminal(conversation_id) == "Error"
    assert _rows(conversation_id) == [_Row(role="user", pending=True, seq=0)]


# ── T6: answers share a one-connection pool ──────────────────────────────────


async def test_three_answers_through_a_one_connection_pool_all_complete(one_connection_pool, monkeypatch):
    """All three turns wait on their model at once, which they can only do if
    none holds the pool's only connection, and then all three are saved."""
    model = PausedModel()
    monkeypatch.setattr(harness_mod, "build_chat_session", model.build)
    conversations = [uuid4() for _ in range(3)]

    async with _client() as client:
        answers = [asyncio.create_task(_ask(client, cid, f"question {i}")) for i, cid in enumerate(conversations)]
        try:
            await _until(lambda: model.turns == 3 or any(a.done() for a in answers), timeout=5 * WAIT)
        finally:
            all_waiting, held_while_waiting = model.turns, one_connection_pool.pool.checkedout()
            model.release.set()
        answered = await asyncio.wait_for(asyncio.gather(*answers), timeout=5 * WAIT)

    assert [a.status_code for a in answered] == [200, 200, 200], [a.text[:200] for a in answered]
    assert all_waiting == 3
    assert held_while_waiting == 0
    assert [_frames(a.text)[-1][0] for a in answered] == ["response.completed"] * 3
    for cid in conversations:
        assert [(r.role, r.pending) for r in _rows(cid)] == [("user", False), ("assistant", False)]


# ── T7: cancels during a unit ────────────────────────────────────────────────


@pytest.mark.parametrize("connection_frees", [True, False], ids=["connection-frees", "pool-stays-held"])
async def test_a_stop_during_the_turn_end_save_leaves_the_stream_and_the_database_agreeing(
    one_connection_pool, monkeypatch, caplog, connection_frees,
):
    """Stop lands while the turn-end unit waits for the pool's only
    connection. The Stop waits for the save: the stream ends completed with
    the answer saved, or failed with nothing saved, and nothing closes a
    session a worker thread is using."""
    conversation_id = uuid4()
    model = PausedModel()
    monkeypatch.setattr(harness_mod, "build_chat_session", model.build)
    slots = units._slots(one_connection_pool)

    async with _client() as client:
        answer = asyncio.create_task(_ask(client, conversation_id, "hi"))
        await asyncio.wait_for(model.answering.wait(), timeout=10)
        held = await _take_the_only_connection(one_connection_pool)
        try:
            model.release.set()
            await _until(lambda: slots.borrowed_tokens == 1)  # the save waits on the pool
            stop = asyncio.create_task(registry.cancel(str(conversation_id)))
            await asyncio.sleep(0.3)
            stop_waited_for_the_save = not stop.done()
            if connection_frees:
                held.close()
            assert await asyncio.wait_for(stop, timeout=5 * WAIT) is True
        finally:
            held.close()
        answered = await asyncio.wait_for(answer, timeout=5 * WAIT)

    assert stop_waited_for_the_save
    assert _state_changes_logged(caplog) == []
    rows = _rows(conversation_id)
    if connection_frees:
        assert _frames(answered.text)[-1][0] == "response.completed"
        assert _terminal(conversation_id) == "Done"
        assert [(r.role, r.pending) for r in rows] == [("user", False), ("assistant", False)]
    else:
        assert _frames(answered.text)[-1][1]["code"] == SERVER_BUSY_CODE
        assert _terminal(conversation_id) == "Error"
        assert [(r.role, r.pending) for r in rows] == [("user", True)]


async def test_shutdown_during_the_turn_start_unit_waits_for_it(monkeypatch, caplog):
    """Shutdown cancels a turn whose first unit is still writing in its
    thread. The cancel waits for the unit, and the turn then ends interrupted
    with no answer saved, matching its stream."""
    conversation_id = uuid4()
    model = PausedModel()
    monkeypatch.setattr(harness_mod, "build_chat_session", model.build)
    entered, proceed = threading.Event(), threading.Event()
    start_turn = responses_mod._start_turn

    def slow_start(session, **kwargs):
        entered.set()
        proceed.wait(timeout=10)
        return start_turn(session, **kwargs)

    monkeypatch.setattr(responses_mod, "_start_turn", slow_start)

    async with _client() as client:
        answer = asyncio.create_task(_ask(client, conversation_id, "hi"))
        await _until(entered.is_set)
        shutdown = asyncio.create_task(registry.shutdown(timeout_seconds=5))
        await asyncio.sleep(0.3)
        shutdown_waited_for_the_unit = not shutdown.done()
        proceed.set()
        assert await asyncio.wait_for(shutdown, timeout=10) == 1
        answered = await asyncio.wait_for(answer, timeout=10)

    assert shutdown_waited_for_the_unit
    assert _state_changes_logged(caplog) == []
    assert model.turns == 0
    last_kind, last = _frames(answered.text)[-1]
    assert (last_kind, last["error"]) == ("response.failed", INTERRUPTED_TURN_MESSAGE)
    assert _terminal(conversation_id) == "Interrupted"
    # The question the unit saved stays pending, out of replayed history, and
    # no answer was saved for it.
    assert [(r.role, r.pending) for r in _rows(conversation_id)] == [("user", True)]


async def test_a_deleted_turns_save_and_the_next_turns_question_never_interleave(monkeypatch):
    """A turn is deleted while its save is mid-write, and the next question
    arrives at once. The next turn's first unit waits for that save, so the
    message numbers stay unique."""
    # As many unit slots as a Postgres pool lends, so only the conversation's
    # write lock keeps the two writes apart.
    units._SLOTS.set(anyio.CapacityLimiter(4))
    conversation_id = uuid4()
    model = PausedModel()
    monkeypatch.setattr(harness_mod, "build_chat_session", model.build)

    order: list[str] = []
    first_save_numbered, finish_first_save = threading.Event(), threading.Event()
    next_seq = ConversationService._next_seq

    def numbered_then_held(self, cid):
        seq = next_seq(self, cid)
        if not first_save_numbered.is_set() and threading.get_ident() != loop_thread:
            if "save_assistant_turn" in {frame.name for frame in traceback.extract_stack()}:
                order.append("deleted turn's save numbered its row")
                first_save_numbered.set()
                finish_first_save.wait(timeout=10)
        return seq

    loop_thread = threading.get_ident()
    monkeypatch.setattr(ConversationService, "_next_seq", numbered_then_held)
    start_turn = responses_mod._start_turn

    def recorded_start(session, **kwargs):
        order.append("a turn saved its question")
        return start_turn(session, **kwargs)

    monkeypatch.setattr(responses_mod, "_start_turn", recorded_start)

    async with _client() as client:
        # Its stream never ends: a deleted turn writes no terminal record.
        first = asyncio.create_task(_ask(client, conversation_id, "one"))
        await asyncio.wait_for(model.answering.wait(), timeout=10)
        deleted_turn = registry.get(str(conversation_id)).task
        model.release.set()
        await _until(first_save_numbered.is_set)
        await asyncio.to_thread(discard_conversation, str(conversation_id))
        second = asyncio.create_task(_ask(client, conversation_id, "two"))
        await asyncio.sleep(0.5)
        order.append("deleted turn's save released")
        finish_first_save.set()
        answered = await asyncio.wait_for(second, timeout=10)
        await asyncio.wait_for(asyncio.gather(deleted_turn, return_exceptions=True), timeout=10)
        first.cancel()

    assert answered.status_code == 200, answered.text
    assert _frames(answered.text)[-1][0] == "response.completed"
    assert order == [
        "a turn saved its question",
        "deleted turn's save numbered its row",
        "deleted turn's save released",
        "a turn saved its question",
    ]
    seqs = [r.seq for r in _rows(conversation_id)]
    assert len(seqs) == len(set(seqs)), seqs


async def test_a_turn_deleted_while_its_save_waits_saves_nothing(one_connection_pool, monkeypatch):
    """A turn delete that lands while the turn-end unit waits for a connection
    stops the save before it writes: no answer lands in the history the
    delete cut, and the turn's deleted buffer is not written again."""
    conversation_id = uuid4()
    model = PausedModel()
    monkeypatch.setattr(harness_mod, "build_chat_session", model.build)
    slots = units._slots(one_connection_pool)

    async with _client() as client:
        answer = asyncio.create_task(_ask(client, conversation_id, "hi"))
        await asyncio.wait_for(model.answering.wait(), timeout=10)
        handle = registry.get(str(conversation_id))
        held = await _take_the_only_connection(one_connection_pool)
        try:
            model.release.set()
            await _until(lambda: slots.borrowed_tokens == 1)  # the save waits on the pool
            await asyncio.to_thread(discard_conversation, str(conversation_id))
            await asyncio.sleep(0.2)
        finally:
            held.close()
        await asyncio.wait_for(asyncio.gather(handle.task, return_exceptions=True), timeout=5 * WAIT)
        answer.cancel()

    assert [(r.role, r.pending) for r in _rows(conversation_id)] == [("user", True)]
    assert not handle.buffer.path.exists()


# ── T8: a scheduled turn ─────────────────────────────────────────────────────


async def test_a_scheduled_turn_completes_and_holds_no_connection_while_it_answers(monkeypatch):
    """Through execute_schedule and the real ResponsesHandler: the run's own
    reads and writes are units too, so nothing is held while the model
    answers, and the run is recorded as a success."""
    from cowork.scheduler import execute_schedule

    pool = _app_engine().pool
    model = PausedModel()
    monkeypatch.setattr(harness_mod, "build_chat_session", model.build)
    with unit_session(scope=SYSTEM_SCOPE) as session:
        schedule_id = ScheduleService(session).create_schedule(
            title="pool test", prompt="do the thing", cadence="daily",
            next_run_at=datetime(2026, 6, 25, 9, 0, tzinfo=timezone.utc),
            model="default", timezone="UTC", project_id=GENERAL_PROJECT_ID, enabled=True,
        ).id

    try:
        run = asyncio.create_task(execute_schedule(schedule_id, is_manual=True))
        await asyncio.wait_for(model.answering.wait(), timeout=10)
        held_while_answering = pool.checkedout()
        model.release.set()
        await asyncio.wait_for(run, timeout=10)

        with unit_session(scope=SYSTEM_SCOPE) as session:
            (finished,) = ScheduleRunService(session).list_runs(schedule_id)
            run_status, conversation_id = finished.status, finished.conversation_id
            last_result = session.get(Schedule, schedule_id).last_result_conversation_id
    finally:
        with unit_session(scope=SYSTEM_SCOPE) as session:
            for leftover in session.exec(session.select(ScheduleRun).where(ScheduleRun.schedule_id == schedule_id)):
                session.delete(leftover)
            ScheduleService(session).delete_schedule(schedule_id)

    assert held_while_answering == 0
    assert run_status == RunStatus.success
    assert last_result == conversation_id
    assert [(r.role, r.pending) for r in _rows(conversation_id)] == [("user", False), ("assistant", False)]
