"""A turn's record of the learned memory it read.

After the turn, AntonHarness.stream_response turns anton's
ChatSession.memory_read into a TurnMemoryRead, and the formatter sends it as
one `response.memory_read` event before the answer ends, which the answer's
saved events keep. An anton without ChatSession.memory_read records nothing.
"""
from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from uuid import uuid4

import pytest
from sqlmodel import Session

from anton.core.llm.provider import StreamTextDelta
from cowork.common.settings.app_settings import get_app_settings
from cowork.db.scoped import LOCAL_SCOPE, ScopedSession
from cowork.db.session import get_engine
from cowork.harnesses.anton_harness.harness import AntonHarness
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


class _Session:
    """A built session whose turn streams nothing. ``memory_read`` stands in
    for anton's MemoryRead (anton.core.memory.base), the two fields the
    harness reads; the anton this suite installs predates it."""

    def __init__(self, memory_read) -> None:
        self.memory_read = memory_read

    async def turn_stream(self, *_args, **_kwargs):
        return
        yield


class _OlderAntonSession:
    """A session from an anton without ChatSession.memory_read."""

    async def turn_stream(self, *_args, **_kwargs):
        return
        yield


@pytest.fixture
def db():
    with Session(get_engine(get_app_settings().database.uri)) as s:
        yield s


def _turn_events(monkeypatch, tmp_path, db, session) -> list:
    """Every event one in-process turn yields with ``session`` built for it."""
    path = tmp_path / "memory-read"
    path.mkdir()
    project = ProjectService(ScopedSession(db, LOCAL_SCOPE)).create_project("memory-read", path=path)
    conversation = SimpleNamespace(id=uuid4(), project_id=project.id, project=project)

    async def _fake_build(self, conversation, **_kwargs):
        return session, None, None

    async def _drain():
        return [
            event
            async for event in AntonHarness().stream_response(
                conversation=conversation, input=[{"type": "text", "text": "hi"}]
            )
        ]

    monkeypatch.setattr(AntonHarness, "_build_chat_session", _fake_build)
    monkeypatch.setattr(t, "turn_artifact_changes", lambda *_a, **_k: t.ArtifactChanges(created=[], touched=set()))
    monkeypatch.setattr(t, "record_new_artifacts", lambda *_a, **_k: None)
    return asyncio.run(_drain())


@pytest.mark.parametrize("learned, entries", [(True, 3), (True, 0), (False, 0)])
def test_a_turn_records_what_its_prompt_read(monkeypatch, tmp_path, db, learned, entries):
    """``read`` is anton's ``learned``, not whether any entry was found: a
    person whose memory is still empty read it, and a turn with memory off
    didn't."""
    events = _turn_events(
        monkeypatch, tmp_path, db, _Session(SimpleNamespace(learned=learned, entries=entries))
    )

    assert [e for e in events if isinstance(e, TurnMemoryRead)] == [
        TurnMemoryRead(read=learned, entries=entries)
    ]


def test_a_turn_before_its_prompt_was_built_records_nothing(monkeypatch, tmp_path, db):
    events = _turn_events(monkeypatch, tmp_path, db, _Session(None))

    assert not any(isinstance(e, TurnMemoryRead) for e in events)


def test_an_anton_without_memory_read_records_nothing(monkeypatch, tmp_path, db):
    events = _turn_events(monkeypatch, tmp_path, db, _OlderAntonSession())

    assert not any(isinstance(e, TurnMemoryRead) for e in events)


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
    assert memory[0]["read"] is True and memory[0]["entries"] == 4
    assert types.index("response.memory_read") < types.index("response.completed")
    assert [payload for kind, payload in saved if kind == "response.memory_read"] == memory
