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
import contextlib
import json
import logging
import shutil
import threading
import traceback
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from uuid import UUID, uuid4

import anyio
import httpx
import pytest
from sqlalchemy import event
from sqlalchemy.exc import TimeoutError as PoolTimeoutError
from sqlalchemy.orm import Session

import cowork.channels.runtime as channel_runtime
import cowork.db.session as db_session
import cowork.handlers.responses as responses_mod
import cowork.scheduler as scheduler
import cowork.services.history_recall as history_recall
from cowork.common.datetime_utils import ensure_utc
from cowork.db import units
from cowork.db.scoped import LOCAL_SCOPE, SYSTEM_SCOPE
from cowork.db.units import DatabaseBusy, conversation_writes, unit_session
from cowork.handlers.response_routing import DELEGATED_AGENTIC, DIRECT_CONTEXT, RouteDecision
from cowork.handlers.turn_errors import INTERRUPTED_TURN_MESSAGE, SERVER_BUSY_CODE
from cowork.harnesses.anton_harness import harness as harness_mod
from cowork.models.message import Message
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


def _logging_on(monkeypatch, logger: logging.Logger) -> None:
    """Turn a module's logger back on for caplog. The migration tests' alembic
    env calls fileConfig, which disables every logger that existed before it
    for the rest of the session."""
    monkeypatch.setattr(logger, "disabled", False)


def _state_changes_logged(caplog) -> list[str]:
    """IllegalStateChangeError tracebacks logged so far: a session closed
    while a worker thread still used it. Callers turn on the loggers that
    would carry one (_logging_on)."""
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


_LOOP_CHECKOUTS_ALLOWED = frozenset()


@dataclass
class _CheckoutLog:
    """Every checkout from a pool while the log was open: the thread it ran
    on and the functions on its stack, and the most connections each thread
    held at once."""

    loop_thread: int
    checkouts: list[_Checkout] = field(default_factory=list)
    most_at_once: dict[int, int] = field(default_factory=dict)

    @property
    def on_the_loop(self) -> list[_Checkout]:
        return [c for c in self.checkouts if c.thread == self.loop_thread]

    @property
    def unexpected_on_the_loop(self) -> list[_Checkout]:
        return [c for c in self.on_the_loop if not c.sites & _LOOP_CHECKOUTS_ALLOWED]


@contextlib.contextmanager
def _logging_checkouts(engine) -> Iterator[_CheckoutLog]:
    log = _CheckoutLog(loop_thread=threading.get_ident())
    out: dict[int, int] = {}

    def on_checkout(dbapi_connection, record, proxy):
        thread = threading.get_ident()
        out[thread] = out.get(thread, 0) + 1
        log.most_at_once[thread] = max(log.most_at_once.get(thread, 0), out[thread])
        log.checkouts.append(_Checkout(
            thread=thread, sites=frozenset(frame.name for frame in traceback.extract_stack()),
        ))

    def on_checkin(dbapi_connection, record):
        thread = threading.get_ident()
        out[thread] = out.get(thread, 0) - 1

    event.listen(engine, "checkout", on_checkout)
    event.listen(engine, "checkin", on_checkin)
    try:
        yield log
    finally:
        event.remove(engine, "checkout", on_checkout)
        event.remove(engine, "checkin", on_checkin)


async def test_a_turn_checks_out_off_the_event_loop_one_connection_per_thread(monkeypatch, tmp_path):
    """A full turn, with compaction and a new artifact: every checkout runs
    off the loop, and no thread holds two connections at once."""
    engine = _app_engine()
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

    try:
        with _logging_checkouts(engine) as log:
            async with _client() as client:
                answered = await asyncio.wait_for(_ask(client, conversation_id, "q3"), timeout=10)
    finally:
        shutil.rmtree(artifact, ignore_errors=True)

    assert answered.status_code == 200, answered.text
    assert _frames(answered.text)[-1][0] == "response.completed"
    unexpected = log.unexpected_on_the_loop
    assert unexpected == [], f"{len(unexpected)} other checkout(s) on the event loop's thread"
    assert max(log.most_at_once.values()) == 1, log.most_at_once
    with unit_session(scope=LOCAL_SCOPE) as session:
        conversation = ConversationService(session).get_conversation(conversation_id)
        assert conversation.history_summary == "SUMMARY"
        from cowork.models.task_object import TaskObject
        assert session.exec(session.select(TaskObject).where(
            TaskObject.conversation_id == conversation_id, TaskObject.ref == slug,
        )).first() is not None


# ── T5: a failed save ends the stream as failed ──────────────────────────────


def _fail_clearing_a_questions_pending_flag(session, flush_context, instances) -> None:
    """A before_flush hook: the write that clears a question's pending flag
    fails, as a lost connection or a full disk would fail it."""
    for row in session.dirty:
        if isinstance(row, Message) and str(getattr(row.role, "value", row.role)) == "user" and not row.pending:
            raise RuntimeError("disk full")


@pytest.mark.parametrize(
    "failure", ["no_connection_frees", "the_save_raises", "clearing_the_flag_fails"],
)
async def test_a_save_that_fails_at_turn_end_ends_the_stream_as_failed(request, monkeypatch, failure):
    """No connection frees within POOL_TIMEOUT for the turn-end unit, or the
    answer's insert raises, or the write that clears the question's pending
    flag raises. The answer was streamed, but it is not in the database, so
    the stream ends with response.failed and an error terminal record, the
    question stays pending, and no assistant row exists. Only the first case
    needs the one-connection pool; the others run on the default pool, as a
    turn does on a deployment."""
    conversation_id = uuid4()
    model = PausedModel()
    monkeypatch.setattr(harness_mod, "build_chat_session", model.build)
    one_connection_pool = None
    if failure == "no_connection_frees":
        one_connection_pool = request.getfixturevalue("one_connection_pool")
    elif failure == "the_save_raises":

        def disk_full(*_args, **_kwargs):
            raise RuntimeError("disk full")

        monkeypatch.setattr(ConversationService, "save_assistant_turn", disk_full)
    else:
        event.listen(Session, "before_flush", _fail_clearing_a_questions_pending_flag)
        request.addfinalizer(
            lambda: event.remove(Session, "before_flush", _fail_clearing_a_questions_pending_flag)
        )

    held = None
    async with _client() as client:
        answer = asyncio.create_task(_ask(client, conversation_id, "hi"))
        await asyncio.wait_for(model.answering.wait(), timeout=10)
        if one_connection_pool is not None:
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
    _logging_on(monkeypatch, responses_mod.logger)
    _logging_on(monkeypatch, units.logger)
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


@pytest.mark.parametrize("cancel", ["stop", "shutdown"])
async def test_a_cancel_during_the_turn_start_unit_waits_for_it_and_finalizes_the_question(
    monkeypatch, caplog, cancel,
):
    """A Stop or a shutdown cancels a turn whose first unit is still saving
    its question in a worker thread. The cancel waits for the unit, and the
    turn then ends as a cancel after its start does: the question is
    finalized rather than left pending for good, and the stream and the
    database agree."""
    _logging_on(monkeypatch, responses_mod.logger)
    _logging_on(monkeypatch, units.logger)
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
        if cancel == "stop":
            cancelling = asyncio.create_task(registry.cancel(str(conversation_id)))
        else:
            cancelling = asyncio.create_task(registry.shutdown(timeout_seconds=5))
        await asyncio.sleep(0.3)
        cancel_waited_for_the_unit = not cancelling.done()
        proceed.set()
        assert await asyncio.wait_for(cancelling, timeout=10) in {True, 1}
        answered = await asyncio.wait_for(answer, timeout=10)

    assert cancel_waited_for_the_unit
    assert _state_changes_logged(caplog) == []
    assert model.turns == 0
    frames = _frames(answered.text)
    rows = [(r.role, r.pending) for r in _rows(conversation_id)]
    if cancel == "shutdown":
        last_kind, last = frames[-1]
        assert (last_kind, last["error"]) == ("response.failed", INTERRUPTED_TURN_MESSAGE)
        assert "assistant_message_id" in last, last
        assert _terminal(conversation_id) == "Interrupted"
        # The interrupted turn's failure is saved as its answer.
        assert rows == [("user", False), ("assistant", False)]
    else:
        assert "response.failed" not in [kind for kind, _ in frames]
        assert _terminal(conversation_id) == "Cancelled"
        # Nothing was answered, so only the question remains, in history.
        assert rows == [("user", False)]


@pytest.mark.parametrize("branch", ["stop", "error"])
async def test_a_second_cancel_during_a_branchs_save_leaves_the_stream_and_the_database_agreeing(
    monkeypatch, caplog, branch,
):
    """A cancel lands while the Stop branch, or the error branch, saves the
    answer: a second Stop, say, or a shutdown after a Stop. It waits for the
    save, so the stream ends the way the turn did, with the frame that names
    the saved row, and no seal has to end it with a generic error."""
    _logging_on(monkeypatch, responses_mod.logger)
    conversation_id = uuid4()

    async def model_fails(_session):
        raise RuntimeError("the model call failed")

    model = PausedModel(before_answer=model_fails if branch == "error" else None)
    monkeypatch.setattr(harness_mod, "build_chat_session", model.build)
    entered, proceed = threading.Event(), threading.Event()
    save = ConversationService.save_assistant_turn

    def slow_save(self, *args, **kwargs):
        entered.set()
        proceed.wait(timeout=10)
        return save(self, *args, **kwargs)

    monkeypatch.setattr(ConversationService, "save_assistant_turn", slow_save)

    async with _client() as client:
        answer = asyncio.create_task(_ask(client, conversation_id, "hi"))
        stops = []
        if branch == "stop":
            await asyncio.wait_for(model.answering.wait(), timeout=10)
            stops.append(asyncio.create_task(registry.cancel(str(conversation_id))))
        await _until(entered.is_set)
        stops.append(asyncio.create_task(registry.cancel(str(conversation_id))))
        await asyncio.sleep(0.2)
        proceed.set()
        await asyncio.wait_for(asyncio.gather(*stops), timeout=10)
        answered = await asyncio.wait_for(answer, timeout=10)

    frames = _frames(answered.text)
    assert [(r.role, r.pending) for r in _rows(conversation_id)] == [("user", False), ("assistant", False)]
    if branch == "stop":
        assert "response.failed" not in [kind for kind, _ in frames], frames
        assert _terminal(conversation_id) == "Cancelled"
    else:
        last_kind, last = frames[-1]
        assert last_kind == "response.failed"
        assert "assistant_message_id" in last, last
        assert _terminal(conversation_id) == "Error"
    sealed = [r for r in caplog.records if "ended without a terminal record" in r.getMessage()]
    assert sealed == []


async def test_a_stop_during_a_direct_answers_save_waits_for_it(monkeypatch):
    """A Stop lands while a direct answer's rows are being saved. It waits
    for the save, so the stream ends completed with the answer the database
    holds, rather than cancelled with nothing shown."""

    async def direct(**_kwargs):
        return RouteDecision(route=DIRECT_CONTEXT, reason="test", model="m", text="direct answer")

    monkeypatch.setattr(responses_mod, "decide_route", direct)
    conversation_id = uuid4()
    entered, proceed = threading.Event(), threading.Event()
    save = ConversationService.save_assistant_turn

    def slow_save(self, *args, **kwargs):
        entered.set()
        proceed.wait(timeout=10)
        return save(self, *args, **kwargs)

    monkeypatch.setattr(ConversationService, "save_assistant_turn", slow_save)

    async with _client() as client:
        answer = asyncio.create_task(_ask(client, conversation_id, "hi"))
        await _until(entered.is_set)
        stop = asyncio.create_task(registry.cancel(str(conversation_id)))
        await asyncio.sleep(0.2)
        stop_waited_for_the_save = not stop.done()
        proceed.set()
        await asyncio.wait_for(stop, timeout=10)
        answered = await asyncio.wait_for(answer, timeout=10)

    assert stop_waited_for_the_save
    assert _frames(answered.text)[-1][0] == "response.completed"
    assert _terminal(conversation_id) == "Done"
    assert [(r.role, r.pending) for r in _rows(conversation_id)] == [("user", False), ("assistant", False)]


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


async def test_a_channel_turns_saves_wait_for_the_conversations_write_lock(monkeypatch):
    """A channel conversation can be continued from the UI, whose turns save
    from worker threads under the conversation's write lock. A channel turn
    saves under the same lock, so the two never number rows at once."""
    runtime = object.__new__(channel_runtime.AntonChannelRuntime)
    conversation = SimpleNamespace(id=uuid4(), messages=[])
    event = SimpleNamespace(message=SimpleNamespace(content="hi", is_group=False, sender_name=None, attachments=[]))
    saved: list[str] = []

    class _Answers:
        def stream_response(self, **_kwargs):
            return None

        async def formatter(self, stream, model, event_sink):
            delta = {"type": "response.output_text.delta", "delta": "ok"}
            event_sink(delta["type"], delta)
            yield "event: response.output_text.delta\ndata: {}\n\n"

    class _RecordingService:
        def __init__(self, session):
            pass

        def save_user_message(self, conversation_id, content, created_at=None):
            saved.append("question")

        def save_assistant_turn(self, conversation_id, text, events, harness=None, tool_rows=None):
            saved.append("answer")

    async def text_only(scoped, adapter, event, text):
        return [{"type": "text", "text": text}]

    monkeypatch.setattr(runtime, "resolve_turn_harness", lambda scoped, conversation: "anton", raising=False)
    monkeypatch.setattr(runtime, "build_input_blocks", text_only, raising=False)
    monkeypatch.setattr(channel_runtime, "get_harness", lambda harness_id: _Answers())
    monkeypatch.setattr(channel_runtime, "ConversationService", _RecordingService)

    lock_held, release = asyncio.Event(), asyncio.Event()

    async def a_ui_turn_saving():
        async with conversation_writes(conversation.id):
            lock_held.set()
            await release.wait()

    ui_turn = asyncio.create_task(a_ui_turn_saving())
    await lock_held.wait()
    channel_turn = asyncio.create_task(runtime._run_anton(None, conversation, event))
    await asyncio.sleep(0.2)
    saved_while_the_ui_turn_saved = list(saved)
    release.set()
    reply, _ = await asyncio.wait_for(channel_turn, timeout=5)
    await ui_turn

    assert saved_while_the_ui_turn_saved == []
    assert saved == ["question", "answer"]
    assert reply == "ok"


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


async def test_a_turn_deleted_before_its_start_writes_does_not_restore_the_question(monkeypatch):
    conversation_id = _conversation(turns=[("old question", "old answer")])
    with unit_session(scope=LOCAL_SCOPE) as session:
        messages = ConversationService(session).get_ordered_messages(conversation_id)
        anchor = messages[-1].id
    entered, proceed = threading.Event(), threading.Event()
    start_turn = responses_mod._start_turn

    def paused_start(session, **kwargs):
        entered.set()
        assert proceed.wait(timeout=10)
        return start_turn(session, **kwargs)

    def delete_history():
        with unit_session(scope=LOCAL_SCOPE) as session:
            ConversationService(session).delete_turn(conversation_id, anchor)

    monkeypatch.setattr(responses_mod, "_start_turn", paused_start)
    model = PausedModel()
    monkeypatch.setattr(harness_mod, "build_chat_session", model.build)
    async with _client() as client:
        answer = asyncio.create_task(_ask(client, conversation_id, "deleted question"))
        try:
            await _until(entered.is_set)
            handle = registry.get(str(conversation_id))
            await asyncio.to_thread(delete_history)
        finally:
            proceed.set()
        await asyncio.wait_for(asyncio.gather(handle.task, return_exceptions=True), timeout=10)
        answer.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await answer

    assert _rows(conversation_id) == []
    assert model.turns == 0
    assert not handle.buffer.path.exists()


async def test_a_stop_whose_final_save_fails_reports_failure(monkeypatch):
    conversation_id = uuid4()
    model = PausedModel()
    monkeypatch.setattr(harness_mod, "build_chat_session", model.build)

    def failed_save(*_args, **_kwargs):
        raise RuntimeError("disk full")

    async with _client() as client:
        answer = asyncio.create_task(_ask(client, conversation_id, "hi"))
        await asyncio.wait_for(model.answering.wait(), timeout=10)
        monkeypatch.setattr(ConversationService, "save_assistant_turn", failed_save)
        assert await registry.cancel(str(conversation_id)) is True
        answered = await asyncio.wait_for(answer, timeout=10)

    assert _frames(answered.text)[-1][0] == "response.failed"
    assert _terminal(conversation_id) == "Error"
    assert _rows(conversation_id) == [_Row(role="user", pending=True, seq=0)]


async def test_stop_waits_for_artifact_cleanup_without_blocking_the_loop(one_connection_pool, monkeypatch):
    conversation_id = _conversation()
    with unit_session(scope=LOCAL_SCOPE) as session:
        project_path = Path(ConversationService(session).get_conversation(conversation_id).project.path)
    slug = f"cleanup-stop-{uuid4().hex[:8]}"
    artifact = project_path / ".anton" / "artifacts" / slug

    async def make_artifact(session):
        artifact.mkdir(parents=True)
        (artifact / "index.html").write_text("<html></html>")
        (artifact / "metadata.json").write_text(json.dumps({"slug": slug, "name": slug, "type": "html-app"}))
        session.artifacts_touched = {slug}

    model = PausedModel(before_answer=make_artifact)
    monkeypatch.setattr(harness_mod, "build_chat_session", model.build)
    slots = units._slots(one_connection_pool)
    try:
        async with _client() as client:
            answer = asyncio.create_task(_ask(client, conversation_id, "make an artifact"))
            await asyncio.wait_for(model.answering.wait(), timeout=10)
            held = await _take_the_only_connection(one_connection_pool)
            try:
                model.release.set()
                await _until(lambda: slots.borrowed_tokens == 1)
                stopping = asyncio.create_task(registry.cancel(str(conversation_id)))
                await asyncio.sleep(0.1)
                assert not stopping.done()
            finally:
                held.close()
            assert await asyncio.wait_for(stopping, timeout=10) is True
            await asyncio.wait_for(answer, timeout=10)
        from cowork.models.task_object import TaskObject
        with unit_session(scope=LOCAL_SCOPE) as session:
            assert session.exec(session.select(TaskObject).where(
                TaskObject.conversation_id == conversation_id, TaskObject.ref == slug,
            )).first() is not None
        assert _terminal(conversation_id) == "Cancelled"
    finally:
        shutil.rmtree(artifact, ignore_errors=True)


# ── A refused turn, and a question sent twice ────────────────────────────────


@pytest.mark.parametrize("route", [DELEGATED_AGENTIC, DIRECT_CONTEXT])
async def test_a_question_sent_again_after_a_refused_turn_streams_its_own_answer(monkeypatch, route):
    """The first turn finds the pool full before it saves anything, so the
    conversation's message count, which numbers its turns and names their
    buffers, does not move. The question sent again streams its own answer,
    not the refused turn's failure, and is saved once. The agent's turn is
    refused at its start, a direct answer at its save."""
    model = PausedModel()
    model.release.set()
    monkeypatch.setattr(harness_mod, "build_chat_session", model.build)

    async def decide(**_kwargs):
        return RouteDecision(route=route, reason="test", model="m", text="direct answer")

    monkeypatch.setattr(responses_mod, "decide_route", decide)
    refused: list[str] = []
    save_user_message = ConversationService.save_user_message

    def full_pool_once(self, *args, **kwargs):
        if not refused:
            refused.append("the question's save")
            raise DatabaseBusy("no database connection freed within POOL_TIMEOUT")
        return save_user_message(self, *args, **kwargs)

    monkeypatch.setattr(ConversationService, "save_user_message", full_pool_once)
    conversation_id = uuid4()

    async with _client() as client:
        first = await asyncio.wait_for(_ask(client, conversation_id, "hi"), timeout=10)
        await asyncio.wait_for(registry.get(str(conversation_id)).task, timeout=10)
        again = await asyncio.wait_for(_ask(client, conversation_id, "hi"), timeout=10)
        await asyncio.wait_for(registry.get(str(conversation_id)).task, timeout=10)

    first_frames, again_frames = _frames(first.text), _frames(again.text)
    assert (first_frames[-1][0], first_frames[-1][1]["code"]) == ("response.failed", SERVER_BUSY_CODE)
    again_kinds = [kind for kind, _ in again_frames]
    assert "response.failed" not in again_kinds, again_frames
    assert again_kinds[-1] == "response.completed"
    assert _terminal(conversation_id) == "Done"
    assert [(r.role, r.pending) for r in _rows(conversation_id)] == [("user", False), ("assistant", False)]


async def test_two_first_sends_for_one_new_conversation_get_one_answer_and_one_refusal(monkeypatch):
    """A double click sends the first question of a new conversation twice.
    With as many unit slots as a Postgres pool lends, both requests look for
    the conversation before either creates it, and both create it. The second
    insert does not fail its request: one question is answered and the other
    is refused as a duplicate send."""
    units._SLOTS.set(anyio.CapacityLimiter(4))
    model = PausedModel()
    monkeypatch.setattr(harness_mod, "build_chat_session", model.build)
    both_looked = threading.Barrier(2, timeout=5)
    create = ConversationService.create_conversation

    def create_once_both_looked(self, *args, **kwargs):
        with contextlib.suppress(threading.BrokenBarrierError):
            both_looked.wait()
        return create(self, *args, **kwargs)

    monkeypatch.setattr(ConversationService, "create_conversation", create_once_both_looked)
    conversation_id = uuid4()

    async with _client() as client:
        sends = [asyncio.create_task(_ask(client, conversation_id, "one")) for _ in range(2)]
        await asyncio.wait(sends, timeout=10, return_when=asyncio.FIRST_COMPLETED)
        model.release.set()
        answered = await asyncio.wait_for(asyncio.gather(*sends), timeout=10)

    assert sorted(a.status_code for a in answered) == [200, 409], [a.text[:200] for a in answered]
    assert [(r.role, r.pending) for r in _rows(conversation_id)] == [("user", False), ("assistant", False)]


# ── T8: scheduled turns ──────────────────────────────────────────────────────


@contextlib.contextmanager
def _a_daily_schedule(*, due: datetime) -> Iterator[UUID]:
    """A daily schedule due at ``due``, deleted again with its runs."""
    with unit_session(scope=SYSTEM_SCOPE) as session:
        schedule_id = ScheduleService(session).create_schedule(
            title="pool test", prompt="do the thing", cadence="daily", next_run_at=due,
            model="default", timezone="UTC", project_id=GENERAL_PROJECT_ID, enabled=True,
        ).id
    try:
        yield schedule_id
    finally:
        with unit_session(scope=SYSTEM_SCOPE) as session:
            for leftover in session.exec(session.select(ScheduleRun).where(ScheduleRun.schedule_id == schedule_id)):
                session.delete(leftover)
            ScheduleService(session).delete_schedule(schedule_id)


@dataclass(frozen=True)
class _FinishedRun:
    status: RunStatus
    conversation_id: UUID | None
    still_active: bool
    last_result: UUID | None
    next_run_at: datetime


def _finished_run(schedule_id: UUID) -> _FinishedRun:
    with unit_session(scope=SYSTEM_SCOPE) as session:
        (finished,) = ScheduleRunService(session).list_runs(schedule_id)
        schedule = session.get(Schedule, schedule_id)
        return _FinishedRun(
            status=finished.status,
            conversation_id=finished.conversation_id,
            still_active=ScheduleRunService(session).has_active_run(schedule_id),
            last_result=schedule.last_result_conversation_id,
            next_run_at=ensure_utc(schedule.next_run_at),
        )


async def test_a_scheduled_turn_completes_and_holds_no_connection_while_it_answers(monkeypatch):
    """Through execute_schedule and the real ResponsesHandler: the run's own
    reads and writes are units too, so nothing is held while the model
    answers, nothing but the harness's two end-of-turn writes checks out on
    the event loop's thread, and the run is recorded as a success."""
    pool = _app_engine().pool
    model = PausedModel()
    monkeypatch.setattr(harness_mod, "build_chat_session", model.build)

    with _a_daily_schedule(due=datetime(2026, 6, 25, 9, 0, tzinfo=timezone.utc)) as schedule_id:
        with _logging_checkouts(_app_engine()) as log:
            run = asyncio.create_task(scheduler.execute_schedule(schedule_id, is_manual=True))
            await asyncio.wait_for(model.answering.wait(), timeout=10)
            held_while_answering = pool.checkedout()
            model.release.set()
            await asyncio.wait_for(run, timeout=10)
        finished = _finished_run(schedule_id)

    assert held_while_answering == 0
    unexpected = log.unexpected_on_the_loop
    assert unexpected == [], f"{len(unexpected)} checkout(s) on the event loop's thread"
    assert finished.status == RunStatus.success
    assert finished.last_result == finished.conversation_id
    assert [(r.role, r.pending) for r in _rows(finished.conversation_id)] == [("user", False), ("assistant", False)]


async def test_a_scheduler_poll_that_finds_the_pool_held_never_stalls_the_loop(
    one_connection_pool, monkeypatch, caplog,
):
    """The scheduler's poll is a unit too. While the pool's only connection
    is held elsewhere, each poll is refused after POOL_TIMEOUT and says so,
    and the event loop keeps running meanwhile."""
    _logging_on(monkeypatch, scheduler.logger)
    monkeypatch.setattr(scheduler, "_POLL_INTERVAL_SECONDS", 0)
    ticks = 0

    async def heartbeat():
        nonlocal ticks
        while True:
            await asyncio.sleep(0.01)
            ticks += 1

    held = await _take_the_only_connection(one_connection_pool)
    beat = asyncio.create_task(heartbeat())
    polling = asyncio.create_task(scheduler._scheduler_loop())
    try:
        with caplog.at_level(logging.WARNING, logger=scheduler.logger.name):
            await asyncio.sleep(1.5 * WAIT)
    finally:
        for task in (polling, beat):
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
        held.close()

    # A free loop ticks about 100 times a second here; a blocked one once.
    assert ticks >= 75 * WAIT, ticks
    assert any("no free database connection" in r.getMessage() for r in caplog.records)


@pytest.mark.parametrize("refused", ["_record_turn_outcome", "_finish_run"])
async def test_a_scheduled_runs_last_writes_land_after_the_pool_refuses_them(monkeypatch, refused):
    """The pool is full at the moment a cron run records how its turn went,
    or finishes the run. The refused write tries again, so the run still ends
    as a success, its slot moves on and is not run a second time, and no run
    is left `running` to stop the schedule firing."""
    monkeypatch.setattr(scheduler, "_BOOKKEEPING_RETRY_PAUSE_SECONDS", 0, raising=False)
    model = PausedModel()
    model.release.set()
    monkeypatch.setattr(harness_mod, "build_chat_session", model.build)
    run_db = scheduler.run_db
    refusals: list[str] = []

    async def full_pool_once(fn, *, scope):
        if getattr(fn, "func", None) is getattr(scheduler, refused) and not refusals:
            refusals.append(refused)
            raise DatabaseBusy("no database connection freed within POOL_TIMEOUT")
        return await run_db(fn, scope=scope)

    monkeypatch.setattr(scheduler, "run_db", full_pool_once)

    with _a_daily_schedule(due=datetime(2026, 6, 25, 9, 0, tzinfo=timezone.utc)) as schedule_id:
        await asyncio.wait_for(scheduler.execute_schedule(schedule_id, is_manual=False), timeout=10)
        finished = _finished_run(schedule_id)

    assert refusals == [refused]
    assert finished.status == RunStatus.success
    assert not finished.still_active
    assert finished.next_run_at > datetime.now(timezone.utc)
