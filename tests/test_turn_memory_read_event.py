"""A turn's record of the learned memory it read.

anton sets ChatSession.memory_read when it builds the turn's prompt, before
its first model call. AntonHarness.stream_response turns it into one
TurnMemoryRead as soon as it is set, so a turn that fails or is stopped after
that carries it too, and the formatter sends it as one `response.memory_read`
event before the answer ends, which the answer's saved events keep. A turn
whose memory layout may not learn records that it read nothing, and an anton
without ChatSession.memory_read records nothing.
"""
from __future__ import annotations

import asyncio
import json
from collections.abc import Sequence
from enum import Enum
from types import SimpleNamespace
from typing import NamedTuple
from uuid import uuid4

import pytest
from sqlmodel import Session

from anton.core.llm.provider import StreamTextDelta
from cowork.common.settings.app_settings import get_app_settings
from cowork.db.scoped import LOCAL_SCOPE, ScopedSession
from cowork.db.session import get_engine
from cowork.harnesses.anton_harness.harness import AntonHarness, BuiltChatSession
from cowork.harnesses.anton_harness.stream_formatter import TurnMemoryRead, format_responses_stream
from cowork.services import task_objects as t
from cowork.services.projects import ProjectService

pytestmark = pytest.mark.usefixtures("cleanup_tmp_projects")


async def _events(*items):
    for item in items:
        yield item


def _parse_sse(chunks: list[str]) -> list[dict]:
    parsed = []
    for chunk in chunks:
        for frame in chunk.strip("\n").split("\n\n"):
            for line in frame.split("\n"):
                if line.startswith("data:"):
                    parsed.append(json.loads(line[len("data:"):].strip()))
    return parsed


class _End(Enum):
    """How a fake turn ends once it has yielded its events."""

    FINISHED = "finished"
    # anton raises, as it does for a provider error its retries don't clear.
    FAILED = "failed"
    # The user stops the turn while it waits on the model.
    STOPPED = "stopped"


class _ProviderDown(Exception):
    """The error a FAILED fake turn raises."""


class _Session:
    """A built session standing in for anton's ChatSession. As anton's,
    ``memory_read`` is None until the turn builds its prompt, which the turn
    does first, setting it to ``read``. ``read`` stands in for anton's
    MemoryRead (anton.core.memory.base), the two fields the harness reads; the
    anton this suite installs predates it. The turn then yields ``events`` and
    ends as ``end`` says."""

    def __init__(self, read, *, events: Sequence[object] = (), end: _End = _End.FINISHED) -> None:
        self.memory_read = None
        self._read = read
        self._events = events
        self._end = end

    async def turn_stream(self, *_args, **_kwargs):
        self.memory_read = self._read
        for event in self._events:
            yield event
        if self._end is _End.FAILED:
            raise _ProviderDown("provider overloaded")
        if self._end is _End.STOPPED:
            # Every turn cancel in this server is a Task.cancel(), which lands
            # on the turn's next await: here, the wait on the model.
            asyncio.current_task().cancel()
            await asyncio.sleep(0)


class _OlderAntonSession:
    """A session from an anton without ChatSession.memory_read."""

    async def turn_stream(self, *_args, **_kwargs):
        return
        yield


class _Turn(NamedTuple):
    # Every event the turn yielded, in order.
    events: list
    # What ended the turn's stream: None when it finished.
    error: BaseException | None


@pytest.fixture
def db():
    with Session(get_engine(get_app_settings().database.uri)) as s:
        yield s


def _run_turn(monkeypatch, tmp_path, db, session, *, learns: bool = True) -> _Turn:
    """One in-process turn with ``session`` built for it, on a memory layout
    whose ``learns`` is given."""
    path = tmp_path / "memory-read"
    path.mkdir()
    project = ProjectService(ScopedSession(db, LOCAL_SCOPE)).create_project("memory-read", path=path)
    conversation = SimpleNamespace(id=uuid4(), project_id=project.id, project=project)

    async def _fake_build(self, conversation, **_kwargs):
        return BuiltChatSession(session=session, turn_scratch_dir=None, seed_info=None, learns=learns)

    async def _drain() -> _Turn:
        events: list = []

        async def consume() -> None:
            async for event in AntonHarness().stream_response(
                conversation=conversation, input=[{"type": "text", "text": "hi"}]
            ):
                events.append(event)

        try:
            # Its own task, as the server runs a turn, so a stop cancels only it.
            await asyncio.create_task(consume())
        except (Exception, asyncio.CancelledError) as error:
            return _Turn(events=events, error=error)
        return _Turn(events=events, error=None)

    monkeypatch.setattr(AntonHarness, "_build_chat_session", _fake_build)
    monkeypatch.setattr(t, "turn_artifact_changes", lambda *_a, **_k: t.ArtifactChanges(created=[], touched=set()))
    monkeypatch.setattr(t, "record_new_artifacts", lambda *_a, **_k: None)
    return asyncio.run(_drain())


def _memory_reads(events: list) -> list[TurnMemoryRead]:
    return [e for e in events if isinstance(e, TurnMemoryRead)]


@pytest.mark.parametrize("learned, entries", [(True, 3), (True, 0), (False, 0)])
def test_a_turn_records_what_its_prompt_read(monkeypatch, tmp_path, db, learned, entries):
    """``read`` is anton's ``learned``, not whether any entry was found: a
    person whose memory is still empty read it, and a turn anton read no
    learned tier for didn't."""
    turn = _run_turn(monkeypatch, tmp_path, db, _Session(SimpleNamespace(learned=learned, entries=entries)))

    assert turn.error is None
    assert _memory_reads(turn.events) == [TurnMemoryRead(read=learned, entries=entries)]


def test_a_turn_on_tiers_that_cannot_learn_records_no_read(monkeypatch, tmp_path, db):
    """A layout that may not learn hands anton two empty scratch tiers, which
    anton reports as read when the server passes it no learned memory scope,
    as on a deployment without COWORK_DEPLOYMENT_KIND."""
    turn = _run_turn(
        monkeypatch, tmp_path, db, _Session(SimpleNamespace(learned=True, entries=0)), learns=False
    )

    assert turn.error is None
    assert _memory_reads(turn.events) == [TurnMemoryRead(read=False, entries=0)]


def test_the_record_goes_out_once_with_the_first_event_after_the_prompt(monkeypatch, tmp_path, db):
    deltas = [StreamTextDelta(text="One"), StreamTextDelta(text=" two"), StreamTextDelta(text=" three")]

    turn = _run_turn(
        monkeypatch, tmp_path, db, _Session(SimpleNamespace(learned=True, entries=2), events=deltas)
    )

    assert turn.error is None
    assert _memory_reads(turn.events) == [TurnMemoryRead(read=True, entries=2)]
    assert turn.events[turn.events.index(deltas[0]) + 1] == TurnMemoryRead(read=True, entries=2)


@pytest.mark.parametrize(
    "events", [[], [StreamTextDelta(text="Looking")]], ids=["before-any-event", "after-an-event"]
)
def test_a_failed_turn_that_built_its_prompt_records_it_before_the_error(
    monkeypatch, tmp_path, db, events
):
    turn = _run_turn(
        monkeypatch,
        tmp_path,
        db,
        _Session(SimpleNamespace(learned=True, entries=3), events=events, end=_End.FAILED),
    )

    assert isinstance(turn.error, _ProviderDown)
    assert _memory_reads(turn.events) == [TurnMemoryRead(read=True, entries=3)]
    assert turn.events[-1] == TurnMemoryRead(read=True, entries=3)


@pytest.mark.parametrize(
    "events", [[], [StreamTextDelta(text="Looking")]], ids=["before-any-event", "after-an-event"]
)
def test_a_turn_stopped_after_building_its_prompt_records_it(monkeypatch, tmp_path, db, events):
    """A stop before any event lands on the first model call, after anton has
    built the prompt. The turn still ends cancelled."""
    turn = _run_turn(
        monkeypatch,
        tmp_path,
        db,
        _Session(SimpleNamespace(learned=True, entries=1), events=events, end=_End.STOPPED),
    )

    assert isinstance(turn.error, asyncio.CancelledError)
    assert _memory_reads(turn.events) == [TurnMemoryRead(read=True, entries=1)]


@pytest.mark.parametrize("end", list(_End))
def test_a_turn_before_its_prompt_was_built_records_nothing(monkeypatch, tmp_path, db, end):
    turn = _run_turn(monkeypatch, tmp_path, db, _Session(None, events=[StreamTextDelta(text="Hi")], end=end))

    assert (turn.error is None) is (end is _End.FINISHED)
    assert _memory_reads(turn.events) == []


def test_an_anton_without_memory_read_records_nothing(monkeypatch, tmp_path, db):
    turn = _run_turn(monkeypatch, tmp_path, db, _OlderAntonSession())

    assert turn.error is None
    assert _memory_reads(turn.events) == []


async def test_the_formatter_sends_one_memory_read_event_before_the_answer_ends():
    saved: list[tuple[str, dict]] = []

    chunks = [
        c async for c in format_responses_stream(
            _events(StreamTextDelta(text="ok"), TurnMemoryRead(read=True, entries=4)),
            model="m",
            event_sink=lambda kind, payload: saved.append((kind, payload)),
        )
    ]

    events = _parse_sse(chunks)
    types = [e.get("type") for e in events]
    memory = [e for e in events if e.get("type") == "response.memory_read"]
    assert len(memory) == 1
    assert {k: v for k, v in memory[0].items() if k not in ("sequence_number", "at_ms")} == {
        "type": "response.memory_read", "read": True, "entries": 4,
    }
    assert isinstance(memory[0]["sequence_number"], int)
    assert types.index("response.memory_read") < types.index("response.completed")
    assert [payload for kind, payload in saved if kind == "response.memory_read"] == memory
