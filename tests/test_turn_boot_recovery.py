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
    svc.save_assistant_turn(conv.id, "already answered", [{"type": "response.completed"}])
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


def test_retries_history_after_buffer_was_sealed(svc, conv, session, tmp_path):
    svc.save_user_message(conv.id, "hello", pending=True)
    _write_orphan_buffer(tmp_path, conv.id, 0, [DELTA.format("Hi")])
    seal_orphan_buffers(tmp_path)

    scoped = ScopedSession(session, SYSTEM_SCOPE)
    assert seal_orphan_turns_in_history(scoped, tmp_path) == 1
    assert seal_orphan_turns_in_history(scoped, tmp_path) == 0
    assert svc.get_ordered_messages(conv.id)[1].content == "Hi"


@pytest.mark.parametrize("failure_at", ["read", "event_write"])
def test_failed_history_recovery_retries_on_next_boot(
    svc, conv, session, tmp_path, monkeypatch, failure_at,
):
    svc.save_user_message(conv.id, "hello", pending=True)
    _write_orphan_buffer(tmp_path, conv.id, 0, [DELTA.format("survives")])
    scoped = ScopedSession(session, SYSTEM_SCOPE)
    with monkeypatch.context() as patch:
        if failure_at == "read":
            def fail_read(*args, **kwargs):
                raise RuntimeError("database unavailable")
            patch.setattr(ConversationService, "get_ordered_messages", fail_read)
        else:
            original_add = scoped.add
            def fail_event_write(row):
                if isinstance(row, MessageEvent):
                    raise RuntimeError("database write failed")
                return original_add(row)
            patch.setattr(scoped, "add", fail_event_write)
        assert seal_orphan_turns_in_history(scoped, tmp_path) == 0
        seal_orphan_buffers(tmp_path)

    assert seal_orphan_turns_in_history(scoped, tmp_path) == 1
    assert seal_orphan_turns_in_history(scoped, tmp_path) == 0
    messages = svc.get_ordered_messages(conv.id)
    assert [m.role for m in messages] == ["user", "assistant"]
    assert messages[1].content == "survives"
    assert _events_for(session, messages[1].id)[0].event_data["type"] == "response.failed"


def test_assistant_and_terminal_event_commit_together(svc, conv, session, monkeypatch):
    svc.save_user_message(conv.id, "hello", pending=True)
    original_add = svc.session.add

    def fail_event_write(row):
        if isinstance(row, MessageEvent):
            raise RuntimeError("killed between assistant and event writes")
        return original_add(row)

    monkeypatch.setattr(svc.session, "add", fail_event_write)
    with pytest.raises(RuntimeError, match="killed between"):
        svc.save_assistant_turn(conv.id, "partial", [{"type": "response.failed"}])
    session.rollback()
    # A fresh session sees committed state, not the failed transaction's objects.
    with Session(session.get_bind()) as fresh:
        messages = ConversationService(ScopedSession(fresh, SYSTEM_SCOPE)).get_ordered_messages(
            conv.id, include_pending=True,
        )
        assert [m.role for m in messages] == ["user"]


@pytest.mark.parametrize("completed_in_buffer", [False, True])
def test_repairs_an_older_assistant_commit_missing_its_events(
    svc, conv, session, tmp_path, completed_in_buffer,
):
    svc.save_user_message(conv.id, "hello", pending=True)
    svc.save_assistant_turn(conv.id, "partial", [])
    frames = [DELTA.format("partial")]
    if completed_in_buffer:
        frames.append('event: response.completed\ndata: {"type":"response.completed"}\n\n')
    _write_orphan_buffer(tmp_path, conv.id, 0, frames)
    scoped = ScopedSession(session, SYSTEM_SCOPE)

    assert seal_orphan_turns_in_history(scoped, tmp_path) == 1
    assert seal_orphan_turns_in_history(scoped, tmp_path) == 0
    messages = svc.get_ordered_messages(conv.id)
    assert [m.role for m in messages] == ["user", "assistant"]
    events = _events_for(session, messages[1].id)
    assert len(events) == 1
    assert events[0].event_data["type"] == (
        "response.completed" if completed_in_buffer else "response.failed"
    )


def test_does_not_attach_orphan_to_a_later_question(svc, conv, session, tmp_path):
    first = svc.save_user_message(conv.id, "first", pending=True)
    second = svc.save_user_message(conv.id, "second", pending=True)
    _write_orphan_buffer(tmp_path, conv.id, 0, [DELTA.format("belongs to first")])
    _write_orphan_buffer(tmp_path, conv.id, 1, [DELTA.format("belongs to second")])

    assert seal_orphan_turns_in_history(ScopedSession(session, SYSTEM_SCOPE), tmp_path) == 1
    messages = svc.get_ordered_messages(conv.id, include_pending=True)
    assert messages[0].id == first.id and messages[0].pending
    assert messages[1].id == second.id and not messages[1].pending
    assert messages[2].content == "belongs to second"
