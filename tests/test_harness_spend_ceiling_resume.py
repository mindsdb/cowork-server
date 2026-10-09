"""A reply to anton's spend-ceiling hand-back runs under a raised ceiling.

The hand-back asks the user whether the work is worth more spend. cowork-server
rebuilds the ChatSession every turn, so the session cannot remember that its
previous turn stopped there: the harness records how each turn ended on the
conversation and passes `after_spend_ceiling` into the next turn.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest
from sqlmodel import Session

import cowork.services.artifact_autopublish as autopublish
import cowork.services.task_objects as task_objects
from cowork.common.settings.app_settings import get_app_settings
from cowork.db.scoped import LOCAL_SCOPE, ScopedSession
from cowork.db.session import get_engine
from cowork.harnesses.anton_harness.harness import AntonHarness
from cowork.services.conversations import ConversationService
from cowork.services.projects import GENERAL_PROJECT_ID


class _Session:
    """ChatSession stand-in: records the flag it was given, ends as told."""

    def __init__(self, ends_by: str, *, raises: Exception | None = None) -> None:
        self._ends_by = ends_by
        self._raises = raises
        self.after_spend_ceiling: bool | None = None
        self.last_turn_ended_by: str | None = None

    async def turn_stream(self, _input, *, trace_tags=None, trace_metadata=None,
                          after_spend_ceiling=None):
        self.after_spend_ceiling = after_spend_ceiling
        self.last_turn_ended_by = self._ends_by
        if self._raises is not None:
            raise self._raises
        return
        yield


class _OldSession:
    """An anton build that predates the flag and the property."""

    def __init__(self) -> None:
        self.kwargs: dict | None = None

    async def turn_stream(self, _input, *, trace_tags=None, trace_metadata=None):
        self.kwargs = {"trace_tags": trace_tags, "trace_metadata": trace_metadata}
        return
        yield


@pytest.fixture
def db():
    engine = get_engine(get_app_settings().database.uri)
    with Session(engine) as s:
        yield s


@pytest.fixture
def svc(db):
    return ConversationService(ScopedSession(db, LOCAL_SCOPE))


@pytest.fixture
def turn(monkeypatch, tmp_path, svc, db):
    """Run one harness turn on a saved conversation with the given session."""
    monkeypatch.setattr(task_objects, "snapshot_artifact_state", lambda *_a, **_k: (set(), {}))
    monkeypatch.setattr(
        task_objects, "turn_artifact_changes",
        lambda *_a, **_k: task_objects.ArtifactChanges(created=[], touched=set()),
    )
    monkeypatch.setattr(task_objects, "cards_for_slugs", lambda *_a, **_k: [])

    async def _no_autopublish(*_a, **_k):
        return set()

    monkeypatch.setattr(autopublish, "autopublish_project_artifacts", _no_autopublish)
    saved = svc.create_conversation("topic", project_id=GENERAL_PROJECT_ID)

    async def run(session):
        async def _fake_build(self, conversation, **_kwargs):
            return session, None, None

        monkeypatch.setattr(AntonHarness, "_build_chat_session", _fake_build)
        db.expire_all()
        conversation = SimpleNamespace(
            id=saved.id,
            project_id=saved.project_id,
            project=SimpleNamespace(path=str(tmp_path), name="p"),
            last_turn_ended_by=svc.get_conversation(saved.id).last_turn_ended_by,
        )
        stream = AntonHarness().stream_response(
            conversation=conversation, input=[{"type": "text", "text": "keep going"}]
        )
        async for _ in stream:
            pass

    def stored() -> str | None:
        db.expire_all()
        return svc.get_conversation(saved.id).last_turn_ended_by

    return SimpleNamespace(run=run, stored=stored)


async def test_reply_after_a_ceiling_stop_carries_the_flag(turn):
    first = _Session("spend_ceiling")
    await turn.run(first)
    assert first.after_spend_ceiling is False
    assert turn.stored() == "spend_ceiling"

    reply = _Session("completed")
    await turn.run(reply)
    assert reply.after_spend_ceiling is True
    assert turn.stored() == "completed"

    after = _Session("completed")
    await turn.run(after)
    assert after.after_spend_ceiling is False, "the raise lasts one turn"


async def test_failed_turn_clears_a_stale_ceiling_stop(turn):
    await turn.run(_Session("spend_ceiling"))
    with pytest.raises(RuntimeError):
        await turn.run(_Session("error", raises=RuntimeError("model call failed")))
    assert turn.stored() == "error"


async def test_older_anton_gets_no_flag_and_records_nothing(turn):
    old = _OldSession()
    await turn.run(old)
    assert old.kwargs is not None, "the turn ran without the unknown kwarg"
    assert turn.stored() is None
