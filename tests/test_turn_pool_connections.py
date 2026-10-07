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
from cowork.handlers.turn_errors import INTERRUPTED_TURN_MESSAGE, SERVER_BUSY_CODE, server_busy_message
from cowork.harnesses.anton_harness import harness as harness_mod
from cowork.models.message import Message
from cowork.models.schedule import Schedule, ScheduleRun
from cowork.models.task_object import TaskObject
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


# The harness's end-of-turn writes run in its generator's `finally`, where an
# await is skipped on cancellation, so they stay synchronous on short sessions.
_LOOP_CHECKOUTS_ALLOWED = frozenset({"_persist_history_compaction", "_index_new_slugs"})


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
    """A full turn, with a compaction and a new artifact so that both of the
    harness's end-of-turn writes run: only those two check out on the event
    loop's thread, and no thread ever holds two connections at once."""
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
    allowed_sites = sorted(site for c in log.on_the_loop for site in c.sites & _LOOP_CHECKOUTS_ALLOWED)
    assert allowed_sites == sorted(_LOOP_CHECKOUTS_ALLOWED), "both end-of-turn writes should have run"


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


# ── T1, remote backend: no connection held while the pod answers ─────────────


def _turns_go_to_the_remote_backend(monkeypatch) -> None:
    """COWORK_TURN_BACKEND=remote, with no per-turn key minted for the gate.
    Each test replaces the pod's replies at stream_remote_replies."""
    monkeypatch.setenv("COWORK_TURN_BACKEND", "remote")

    async def no_minted_key(self):
        return None, None

    monkeypatch.setattr(responses_mod.ResponsesHandler, "_router_binding", no_minted_key)


async def test_a_remote_answer_holds_no_connection_while_the_pod_answers(monkeypatch):
    """Sampled at every frame the turn appends and while the pod's replies
    are awaited. The memory, compaction and artifact the pod reports are
    saved, and every checkout, the end-of-turn artifact index included, runs
    off the event loop's thread, one per thread."""
    _turns_go_to_the_remote_backend(monkeypatch)
    engine = _app_engine()
    pool = engine.pool
    conversation_id = _conversation(turns=[("q1", "a1"), ("q2", "a2")])
    with unit_session(scope=LOCAL_SCOPE) as session:
        service = ConversationService(session)
        project_path = Path(service.get_conversation(conversation_id).project.path)
        # The pod's compaction covers the first two seeded messages.
        covered = service.get_ordered_messages(conversation_id)[1].id
    slug = f"pool-test-{uuid4().hex[:8]}"
    artifact = project_path / ".anton" / "artifacts" / slug
    workspace = project_path / "conversations" / str(conversation_id)
    held: dict[str, list[int]] = {"appends": [], "replies": []}
    memory_saved: list[int] = []
    replying, release = asyncio.Event(), asyncio.Event()

    async def pod_replies(**_kwargs):
        yield "progress", {"phase": "workspace_authorized", "workspace_mode": "persistent"}
        yield "turn_delta", {"text": "o"}
        replying.set()
        await release.wait()
        artifact.mkdir(parents=True)
        (artifact / "index.html").write_text("<html></html>")
        (artifact / "metadata.json").write_text(json.dumps({
            "slug": slug, "name": slug, "type": "html-app",
            "provenance": [{"conversation": str(conversation_id), "turns": []}],
        }))
        yield "turn_memory", {"entries": [{"text": "remember", "kind": "always", "scope": "project"}]}
        yield "turn_compaction", {"summary": "SUMMARY", "covered_through": 2}
        yield "turn_delta", {"text": "k"}
        yield "turn_completed", {}

    monkeypatch.setattr(responses_mod, "stream_remote_replies", pod_replies)

    def apply_turn_memory(scope, path, entries, **_kwargs):
        # The memory write itself is the memory service's; what is pinned
        # here is that it runs in its own unit, off the event loop's thread.
        memory_saved.append(threading.get_ident())
        return len(entries)

    monkeypatch.setattr(responses_mod, "apply_turn_memory", apply_turn_memory)
    append = FileStreamBuffer.append

    async def sampled_append(self, type_, data):
        held["appends"].append(pool.checkedout())
        return await append(self, type_, data)

    monkeypatch.setattr(FileStreamBuffer, "append", sampled_append)

    try:
        with _logging_checkouts(engine) as log:
            async with _client() as client:
                answer = asyncio.create_task(_ask(client, conversation_id, "q3"))
                await asyncio.wait_for(replying.wait(), timeout=10)
                held["replies"].append(pool.checkedout())
                release.set()
                answered = await asyncio.wait_for(answer, timeout=10)
    finally:
        shutil.rmtree(artifact, ignore_errors=True)
        shutil.rmtree(workspace, ignore_errors=True)

    assert answered.status_code == 200, answered.text
    kinds = [kind for kind, _ in _frames(answered.text)]
    assert kinds[-1] == "response.completed", kinds
    assert "response.artifact_created" in kinds
    assert held["appends"], "the turn appended no frame"
    assert held == {"appends": [0] * len(held["appends"]), "replies": [0]}
    assert log.on_the_loop == [], f"{len(log.on_the_loop)} checkout(s) on the event loop's thread"
    assert max(log.most_at_once.values()) == 1, log.most_at_once
    assert memory_saved and threading.get_ident() not in memory_saved
    assert _rows(conversation_id)[-2:] == [
        _Row(role="user", pending=False, seq=4), _Row(role="assistant", pending=False, seq=5),
    ]
    with unit_session(scope=LOCAL_SCOPE) as session:
        conversation = ConversationService(session).get_conversation(conversation_id)
        assert (conversation.history_summary, conversation.history_summary_cutoff_id) == ("SUMMARY", covered)
        indexed = session.exec(
            session.select(TaskObject).where(
                TaskObject.conversation_id == conversation_id, TaskObject.ref == slug,
            )
        ).all()
    assert len(indexed) == 1


async def test_a_remote_turn_that_finds_the_pool_full_ends_as_server_busy(one_connection_pool, monkeypatch):
    """The pool's only connection is taken while the request's gate runs, so
    the remote producer's first unit finds no connection within
    POOL_TIMEOUT. The stream ends with one response.failed frame that says
    so, with the wait, as a refused request's 503 does, and the question is
    not saved."""
    _turns_go_to_the_remote_backend(monkeypatch)

    async def pod_replies(**_kwargs):
        raise AssertionError("the pod must not be asked when the turn found no connection")
        yield  # an async generator, like the real one

    monkeypatch.setattr(responses_mod, "stream_remote_replies", pod_replies)
    gate_entered, gate_release = asyncio.Event(), asyncio.Event()

    async def paused_gate(**_kwargs):
        gate_entered.set()
        await gate_release.wait()
        return RouteDecision(route=DELEGATED_AGENTIC, reason="test")

    monkeypatch.setattr(responses_mod, "decide_route", paused_gate)
    conversation_id = _conversation()

    async with _client() as client:
        answer = asyncio.create_task(_ask(client, conversation_id, "hi"))
        await asyncio.wait_for(gate_entered.wait(), timeout=10)
        held = await _take_the_only_connection(one_connection_pool)
        gate_release.set()
        try:
            answered = await asyncio.wait_for(answer, timeout=5 * WAIT)
        finally:
            held.close()

    assert answered.status_code == 200, answered.text
    frames = _frames(answered.text)
    kinds = [kind for kind, _ in frames]
    assert kinds[-1] == "response.failed", kinds
    failed = frames[-1][1]
    assert failed["code"] == SERVER_BUSY_CODE, failed
    assert failed["retry_after"] == WAIT
    assert _rows(conversation_id) == []



def _artifact_indexed(conversation_id: UUID, slug: str) -> bool:
    with unit_session(scope=LOCAL_SCOPE) as session:
        return session.exec(
            session.select(TaskObject).where(
                TaskObject.conversation_id == conversation_id, TaskObject.ref == slug,
            )
        ).first() is not None


async def test_a_stop_during_a_remote_frame_records_the_pods_artifact_before_the_stream_ends(monkeypatch):
    """The pod wrote an artifact, and the Stop lands while the turn appends a
    frame, so the reply generator is suspended rather than unwinding. The
    artifact is indexed as this conversation's before the stream's terminal
    record, by the turn itself, not later by the abandoned generator's close."""
    _turns_go_to_the_remote_backend(monkeypatch)
    conversation_id = _conversation()
    with unit_session(scope=LOCAL_SCOPE) as session:
        project_path = Path(ConversationService(session).get_conversation(conversation_id).project.path)
    slug = f"pool-test-{uuid4().hex[:8]}"
    artifact = project_path / ".anton" / "artifacts" / slug
    workspace = project_path / "conversations" / str(conversation_id)

    async def pod_replies(**_kwargs):
        yield "progress", {"phase": "workspace_authorized", "workspace_mode": "persistent"}
        artifact.mkdir(parents=True)
        (artifact / "index.html").write_text("<html></html>")
        (artifact / "metadata.json").write_text(json.dumps({
            "slug": slug, "name": slug, "type": "html-app",
            "provenance": [{"conversation": str(conversation_id), "turns": []}],
        }))
        yield "turn_delta", {"text": "partial"}
        await asyncio.sleep(3600)
        yield "turn_completed", {}

    monkeypatch.setattr(responses_mod, "stream_remote_replies", pod_replies)
    appending, never = asyncio.Event(), asyncio.Event()
    append, close = FileStreamBuffer.append, FileStreamBuffer.close
    indexed_at_close: list[bool] = []

    async def held_append(self, type_, data):
        if "partial" in data.get("sse", ""):
            appending.set()
            await never.wait()
        return await append(self, type_, data)

    async def sampled_close(self, reason, extra=None):
        indexed_at_close.append(_artifact_indexed(conversation_id, slug))
        return await close(self, reason, extra)

    monkeypatch.setattr(FileStreamBuffer, "append", held_append)
    monkeypatch.setattr(FileStreamBuffer, "close", sampled_close)

    try:
        async with _client() as client:
            answer = asyncio.create_task(_ask(client, conversation_id, "make a report"))
            await asyncio.wait_for(appending.wait(), timeout=10)
            assert await registry.get(str(conversation_id)).cancel()
            answered = await asyncio.wait_for(answer, timeout=10)
    finally:
        shutil.rmtree(artifact, ignore_errors=True)
        shutil.rmtree(workspace, ignore_errors=True)

    assert _frames(answered.text)[-1][0] == "response.cancelled"
    assert indexed_at_close == [True]
    assert _terminal(conversation_id) == "Cancelled"


def _the_pod_replies(monkeypatch, *replies: tuple[str, dict]) -> None:
    """The pod sends ``replies`` at once."""

    async def pod_replies(**_kwargs):
        for reply in replies:
            yield reply

    monkeypatch.setattr(responses_mod, "stream_remote_replies", pod_replies)


def _paused_unit(monkeypatch, name: str) -> tuple[threading.Event, threading.Event]:
    """Pause the remote producer's unit function ``name`` in its worker
    thread: ``entered`` is set once it runs, and it goes on once ``proceed``
    is set."""
    entered, proceed = threading.Event(), threading.Event()
    unit = getattr(responses_mod, name)

    def paused(session, **kwargs):
        entered.set()
        proceed.wait(timeout=10)
        return unit(session, **kwargs)

    monkeypatch.setattr(responses_mod, name, paused)
    return entered, proceed


async def _stop_while_paused(
    conversation_id: UUID, entered: threading.Event, proceed: threading.Event,
) -> tuple[httpx.Response, bool]:
    """Ask, Stop the turn once its paused unit runs, and let the unit go on.
    The answer, and whether the Stop waited for the unit."""
    async with _client() as client:
        answer = asyncio.create_task(_ask(client, conversation_id, "hi"))
        await _until(entered.is_set)
        stop = asyncio.create_task(registry.cancel(str(conversation_id)))
        await asyncio.sleep(0.3)
        stop_waited_for_the_unit = not stop.done()
        proceed.set()
        assert await asyncio.wait_for(stop, timeout=10) is True
        return await asyncio.wait_for(answer, timeout=10), stop_waited_for_the_unit


async def test_a_stop_during_the_remote_question_save_waits_for_it_and_finalizes_the_question(monkeypatch):
    """A Stop lands while the remote producer's first write, the question,
    runs in its worker thread. The Stop waits for it, so the turn knows the
    question it saved and finalizes it rather than leaving it pending for
    good, and the pod is never asked."""
    _turns_go_to_the_remote_backend(monkeypatch)

    async def pod_replies(**_kwargs):
        raise AssertionError("the pod must not be asked once the turn is stopped")
        yield  # an async generator, like the real one

    monkeypatch.setattr(responses_mod, "stream_remote_replies", pod_replies)
    conversation_id = _conversation()
    entered, proceed = _paused_unit(monkeypatch, "_save_remote_question")

    answered, stop_waited_for_the_unit = await _stop_while_paused(conversation_id, entered, proceed)

    assert stop_waited_for_the_unit
    assert _frames(answered.text)[-1][0] == "response.cancelled"
    assert _terminal(conversation_id) == "Cancelled"
    assert [(r.role, r.pending) for r in _rows(conversation_id)] == [("user", False)]


async def test_a_stop_during_the_remote_answer_save_waits_for_it_and_ends_completed(monkeypatch):
    """A Stop lands while the remote producer saves the pod's finished
    answer. The Stop waits for the save and the frame that reports it, so
    the stream ends completed with the saved row's id, as the database
    holds it, rather than cancelled."""
    _turns_go_to_the_remote_backend(monkeypatch)
    _the_pod_replies(monkeypatch, ("turn_delta", {"text": "done"}), ("turn_completed", {}))
    conversation_id = _conversation()
    entered, proceed = _paused_unit(monkeypatch, "_save_answer")

    answered, stop_waited_for_the_unit = await _stop_while_paused(conversation_id, entered, proceed)

    assert stop_waited_for_the_unit
    last_kind, last = _frames(answered.text)[-1]
    assert last_kind == "response.completed", _frames(answered.text)
    assert last["assistant_message_id"]
    assert _terminal(conversation_id) == "Done"
    assert [(r.role, r.pending) for r in _rows(conversation_id)] == [("user", False), ("assistant", False)]


@pytest.mark.parametrize("failure", ["no_connection_frees", "the_save_raises"])
async def test_a_remote_answer_whose_save_fails_ends_the_stream_as_failed(request, monkeypatch, failure):
    """No connection frees within POOL_TIMEOUT for the remote producer's
    last unit, or the answer's insert raises. The pod's answer was streamed,
    but it is not in the database, so the stream ends with response.failed
    and an error terminal record, never response.completed, and the
    question stays pending."""
    _turns_go_to_the_remote_backend(monkeypatch)
    replying, release = asyncio.Event(), asyncio.Event()

    async def pod_replies(**_kwargs):
        yield "turn_delta", {"text": "done"}
        replying.set()
        await release.wait()
        yield "turn_completed", {}

    monkeypatch.setattr(responses_mod, "stream_remote_replies", pod_replies)
    one_connection_pool = None
    if failure == "no_connection_frees":
        one_connection_pool = request.getfixturevalue("one_connection_pool")
    else:

        def disk_full(*_args, **_kwargs):
            raise RuntimeError("disk full")

        monkeypatch.setattr(ConversationService, "save_assistant_turn", disk_full)
    conversation_id = _conversation()

    held = None
    async with _client() as client:
        answer = asyncio.create_task(_ask(client, conversation_id, "hi"))
        await asyncio.wait_for(replying.wait(), timeout=10)
        if one_connection_pool is not None:
            held = await _take_the_only_connection(one_connection_pool)
        release.set()
        try:
            answered = await asyncio.wait_for(answer, timeout=5 * WAIT)
        finally:
            if held is not None:
                held.close()

    frames = _frames(answered.text)
    kinds = [kind for kind, _ in frames]
    assert "response.output_text.delta" in kinds
    assert "response.completed" not in kinds
    assert kinds[-1] == "response.failed", kinds
    failed = frames[-1][1]
    if failure == "no_connection_frees":
        assert failed["code"] == SERVER_BUSY_CODE
        assert failed["retry_after"] == WAIT
    else:
        assert failed["code"] == "anton_error"
    assert "assistant_message_id" not in failed
    assert _terminal(conversation_id) == "Error"
    assert [(r.role, r.pending) for r in _rows(conversation_id)] == [("user", True)]


# ── The connector form stream: no connection held during the probe ───────────


class _PausedProbe:
    """Stands in for CredentialProbe: it reports a status, waits on
    ``release`` the way a probe waits on its model, then fails the
    credentials. ``probing`` is set while it waits."""

    probing = asyncio.Event()
    release = asyncio.Event()

    def __init__(self, **_kwargs) -> None:
        pass

    async def run(self):
        from cowork.services.connectors.probe import ProbeOutcome

        yield "status", "Connecting…"
        type(self).probing.set()
        await type(self).release.wait()
        yield "verdict", ProbeOutcome(status="failure", error="Password rejected.")


def _probe_runs_against_a_stand_in_form(monkeypatch):
    """A registry connector whose form has no fields, a workspace and a model
    client that are stand-ins, and the probe handler module to patch further."""
    from cowork.handlers import probe as probe_handler

    monkeypatch.setattr(probe_handler.ProbeHandler, "_build_llm_client", staticmethod(lambda settings=None: object()))
    monkeypatch.setattr("anton.workspace.Workspace", lambda path: SimpleNamespace(path=path))
    spec = SimpleNamespace(form=SimpleNamespace(
        form_id="probe-form", methods=None, fields=[], model_dump=lambda: {"form_id": "probe-form"},
    ))
    monkeypatch.setattr(probe_handler.registry, "get_connector", lambda _connector_id: spec)
    return probe_handler


def _form_patches(body: str) -> list[dict]:
    """The form patches a probe stream sent, in order: each is a fenced
    data-vault-form-patch block inside a text delta."""
    text = "".join(data["delta"] for kind, data in _frames(body) if kind == "response.output_text.delta")
    fence = "```data-vault-form-patch\n"
    return [json.loads(block.split("\n```", 1)[0]) for block in text.split(fence)[1:]]


async def test_a_connector_probe_holds_no_connection_while_it_runs(monkeypatch):
    """POST /api/v1/connectors/submissions/ streams a credential probe that
    waits on a model. While it waits no connection is checked out. The
    conversation read and the settings read before it, and the assistant
    turn saved after it, all run off the event loop's thread."""
    probe_handler = _probe_runs_against_a_stand_in_form(monkeypatch)
    engine = _app_engine()
    pool = engine.pool
    conversation_id = _conversation()
    monkeypatch.setattr(_PausedProbe, "probing", asyncio.Event())
    monkeypatch.setattr(_PausedProbe, "release", asyncio.Event())
    monkeypatch.setattr(probe_handler, "CredentialProbe", _PausedProbe)

    with _logging_checkouts(engine) as log:
        async with _client() as client:
            submitted = asyncio.create_task(client.post(
                "/api/v1/connectors/submissions/",
                json={
                    "connector_id": "postgres", "name": "warehouse",
                    "conversation_id": str(conversation_id), "values": {"password": "hunter2"},
                },
            ))
            await asyncio.wait_for(_PausedProbe.probing.wait(), timeout=10)
            held_while_probing = pool.checkedout()
            _PausedProbe.release.set()
            answered = await asyncio.wait_for(submitted, timeout=10)

    assert answered.status_code == 200, answered.text
    completed = _frames(answered.text)[-1]
    assert completed[0] == "response.completed", completed
    assert held_while_probing == 0
    assert log.on_the_loop == [], f"{len(log.on_the_loop)} checkout(s) on the event loop's thread"
    with unit_session(scope=LOCAL_SCOPE) as session:
        (saved,) = ConversationService(session).get_ordered_messages(conversation_id)
    assert saved.role == "assistant" and "Password rejected." in saved.content
    assert completed[1]["assistant_message_id"] == str(saved.id)


async def test_a_connector_probe_that_finds_the_pool_full_tells_the_form_cowork_is_busy(
    one_connection_pool, monkeypatch, caplog,
):
    """The pool's only connection is taken, so the unit that reads the
    probe's settings finds none within POOL_TIMEOUT. The form is told Cowork
    is busy, with the wait, the probe never starts, and the refusal is
    logged as a warning, without a traceback."""
    probe_handler = _probe_runs_against_a_stand_in_form(monkeypatch)
    _logging_on(monkeypatch, probe_handler.logger)

    class _NeverStarts:
        def __init__(self, **_kwargs) -> None:
            raise AssertionError("the probe must not start without its settings")

    monkeypatch.setattr(probe_handler, "CredentialProbe", _NeverStarts)

    held = await _take_the_only_connection(one_connection_pool)
    try:
        async with _client() as client:
            answered = await asyncio.wait_for(client.post(
                "/api/v1/connectors/submissions/",
                json={"connector_id": "postgres", "name": "warehouse", "values": {"password": "hunter2"}},
            ), timeout=5 * WAIT)
    finally:
        held.close()

    assert answered.status_code == 200, answered.text
    completed = _frames(answered.text)[-1]
    assert completed[0] == "response.completed", completed
    assert completed[1]["response"]["status"] == "failed"
    assert _form_patches(answered.text) == [{"form_id": "probe-form", "form_error": server_busy_message(WAIT)}]
    logged = [r for r in caplog.records if r.name == probe_handler.logger.name]
    assert [(r.levelno, r.exc_info) for r in logged] == [(logging.WARNING, None)], [r.getMessage() for r in logged]
