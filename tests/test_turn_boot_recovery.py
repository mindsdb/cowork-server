"""Boot recovery for turn buffers left open by a crash/restart.

`seal_orphan_buffers` closes the buffer FILE so a reconnect ends cleanly.
`seal_orphan_turns_in_history` writes the same interruption into the
conversation's own history, so a reload shows it too.
"""
from __future__ import annotations

import asyncio
import json

import pytest
from sqlmodel import Session, select

from cowork.common.settings.app_settings import get_app_settings
from cowork.db.scoped import SYSTEM_SCOPE, ScopedSession
from cowork.db.session import get_engine
from cowork.models.message_event import MessageEvent
from cowork.services.conversations import ConversationService
from cowork.services.projects import GENERAL_PROJECT_ID
from cowork.streaming.buffer import FileStreamBuffer, turn_buffer_path
from cowork.streaming.recovery import seal_orphan_buffers, seal_orphan_turns_in_history

DELTA = 'event: response.output_text.delta\ndata: {{"type":"response.output_text.delta","delta":"{}"}}\n\n'
CREATED = 'event: response.created\ndata: {"type":"response.created"}\n\n'


@pytest.fixture
def session():
    engine = get_engine(get_app_settings().database.uri)
    with Session(engine) as s:
        yield s


@pytest.fixture
def svc(session):
    return ConversationService(ScopedSession(session, SYSTEM_SCOPE))


@pytest.fixture
def conv(svc):
    return svc.create_conversation("topic", project_id=GENERAL_PROJECT_ID)


def _write_orphan_buffer(tmp_path, conversation_id, turn_id, frames):
    """Write a turn buffer with `frames` (raw SSE strings) but never close
    it — the shape a crash mid-turn leaves behind."""
    path = turn_buffer_path(tmp_path, str(conversation_id), turn_id)
    buf = FileStreamBuffer(path)

    async def _write():
        for frame in frames:
            await buf.append("sse", {"sse": frame})

    asyncio.run(_write())
    return path


def _events_for(session, message_id):
    return session.exec(select(MessageEvent).where(MessageEvent.message_id == message_id)).all()


def test_seals_a_crashed_turn_into_history(svc, conv, session, tmp_path):
    svc.save_user_message(conv.id, "hello", pending=True)
    _write_orphan_buffer(tmp_path, conv.id, 0, [CREATED, DELTA.format("Hi"), DELTA.format(" there")])

    sealed = seal_orphan_turns_in_history(ScopedSession(session, SYSTEM_SCOPE), tmp_path)

    assert sealed == 1
    messages = svc.get_ordered_messages(conv.id)
    assert [m.role for m in messages] == ["user", "assistant"]
    assert messages[0].pending is False  # finalized
    assert messages[1].content == "Hi there"
    events = _events_for(session, messages[1].id)
    assert len(events) == 1
    assert events[0].event_data["type"] == "response.failed"
    # Generic code, not a dedicated one — same as the remote path's own
    # TurnInterrupted handling; the message text is what distinguishes it.
    assert events[0].event_data["code"] == "anton_error"
    assert events[0].event_data["error"] == "The response was interrupted before it finished. Please try again."


def test_is_idempotent_once_sealed(svc, conv, session, tmp_path):
    svc.save_user_message(conv.id, "hello", pending=True)
    _write_orphan_buffer(tmp_path, conv.id, 0, [DELTA.format("Hi")])
    scoped = ScopedSession(session, SYSTEM_SCOPE)

    first = seal_orphan_turns_in_history(scoped, tmp_path)
    second = seal_orphan_turns_in_history(scoped, tmp_path)

    assert first == 1
    assert second == 0
    assert len(svc.get_ordered_messages(conv.id)) == 2


def test_skips_a_turn_whose_reply_already_landed(svc, conv, session, tmp_path):
    # The process died between persist() and buffer.close() — history is
    # already correct, only the buffer file's terminal is missing.
    svc.save_user_message(conv.id, "hello", pending=True)
    svc.finalize_pending(conv.id)
    svc.save_assistant_turn(conv.id, "already answered", [])
    _write_orphan_buffer(tmp_path, conv.id, 0, [DELTA.format("stale replay")])

    sealed = seal_orphan_turns_in_history(ScopedSession(session, SYSTEM_SCOPE), tmp_path)

    assert sealed == 0
    messages = svc.get_ordered_messages(conv.id)
    assert len(messages) == 2
    assert messages[1].content == "already answered"


def test_skips_a_turn_with_no_question_row(svc, conv, session, tmp_path):
    # Crashed before even the pending question was committed — nothing to
    # attach an answer to.
    _write_orphan_buffer(tmp_path, conv.id, 0, [DELTA.format("orphaned delta")])

    sealed = seal_orphan_turns_in_history(ScopedSession(session, SYSTEM_SCOPE), tmp_path)

    assert sealed == 0
    assert svc.get_ordered_messages(conv.id, include_pending=True) == []


def test_a_buffer_with_a_terminal_record_is_left_alone(svc, conv, session, tmp_path):
    svc.save_user_message(conv.id, "hello", pending=True)
    path = _write_orphan_buffer(tmp_path, conv.id, 0, [DELTA.format("Hi")])
    # Append a terminal record straight onto the file (mirrors what
    # buffer.close() writes) rather than reopening the buffer, which would
    # restart its seq counter at 0 and duplicate the existing record.
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"seq": 1, "ts": "now", "type": "Done", "data": {"reason": "completed"}}) + "\n")

    sealed = seal_orphan_turns_in_history(ScopedSession(session, SYSTEM_SCOPE), tmp_path)

    assert sealed == 0
    # The buffer already terminated cleanly — the pending question is left
    # exactly as it was, not finalized or turned into an interrupted turn.
    messages = svc.get_ordered_messages(conv.id, include_pending=True)
    assert len(messages) == 1
    assert messages[0].pending is True


def test_returns_zero_when_streams_root_is_missing(session, tmp_path):
    missing = tmp_path / "does-not-exist"
    assert seal_orphan_turns_in_history(ScopedSession(session, SYSTEM_SCOPE), missing) == 0


def test_ignores_a_directory_that_is_not_a_conversation_id(session, tmp_path):
    junk = tmp_path / "not-a-uuid"
    junk.mkdir()
    (junk / "turn_000000.jsonl").write_text(
        json.dumps({"seq": 0, "ts": "now", "type": "sse", "data": {"sse": DELTA.format("hi")}}) + "\n"
    )
    assert seal_orphan_turns_in_history(ScopedSession(session, SYSTEM_SCOPE), tmp_path) == 0


def test_must_run_before_seal_orphan_buffers_or_it_finds_nothing_to_seal(svc, conv, session, tmp_path):
    # seal_orphan_buffers uses the same "no terminal record yet" signal.
    # Running this after it means every buffer already looks cleanly closed
    # and every turn is silently skipped — server.py must call this first.
    svc.save_user_message(conv.id, "hello", pending=True)
    path = _write_orphan_buffer(tmp_path, conv.id, 0, [DELTA.format("Hi")])
    seal_orphan_buffers(tmp_path)  # simulate the wrong order

    sealed = seal_orphan_turns_in_history(ScopedSession(session, SYSTEM_SCOPE), tmp_path)

    assert sealed == 0
    assert svc.get_ordered_messages(conv.id, include_pending=True)[0].pending is True
    # Sanity: a fresh turn DOES seal when this runs first (the right order).
    svc.save_user_message(conv.id, "second", pending=True)
    _write_orphan_buffer(tmp_path, conv.id, 1, [DELTA.format("Hi")])
    assert seal_orphan_turns_in_history(ScopedSession(session, SYSTEM_SCOPE), tmp_path) == 1
