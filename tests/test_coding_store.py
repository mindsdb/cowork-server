from __future__ import annotations

import json
from pathlib import Path

import pytest

from cowork.coding import store as store_module
from cowork.coding.contracts import (
    CodingEvent,
    CodingSession,
    EventType,
    SessionStatus,
    WorkspaceKind,
)
from cowork.coding.store import CodingStore


def session(session_id: str = "session-1") -> CodingSession:
    return CodingSession(
        id=session_id,
        title="Task",
        engine_id="codex",
        engine_adapter_version="1",
        model="gpt-codex",
        source_path="/source",
        workspace_path="/worktree",
        workspace_kind=WorkspaceKind.git_worktree,
        base_revision="abc",
    )


def test_session_and_event_round_trip_is_monotonic(tmp_path: Path) -> None:
    store = CodingStore(tmp_path)
    item = session()
    store.save_session(item)
    first = store.append_event(item.id, CodingEvent(type=EventType.session, title="Ready"))
    second = store.append_event(item.id, CodingEvent(type=EventType.agent_message, text="Done"))

    assert (first.seq, second.seq) == (1, 2)
    assert store.load_session(item.id).event_count == 2
    assert [event.seq for event in store.events_after(item.id, 1)] == [2]


def test_event_sequence_recovers_when_a_crash_leaves_metadata_behind(tmp_path: Path) -> None:
    store = CodingStore(tmp_path)
    item = session()
    store.save_session(item)
    store.append_event(item.id, CodingEvent(type=EventType.agent_message, text="first"))
    store.append_event(item.id, CodingEvent(type=EventType.agent_message, text="second"))

    metadata = tmp_path / "sessions" / item.id / "session.json"
    raw = json.loads(metadata.read_text(encoding="utf-8"))
    raw["event_count"] = 1
    metadata.write_text(json.dumps(raw), encoding="utf-8")

    restarted = CodingStore(tmp_path)
    third = restarted.append_event(item.id, CodingEvent(type=EventType.agent_message, text="third"))

    assert third.seq == 3
    assert [event.seq for event in restarted.events_after(item.id)] == [1, 2, 3]
    assert restarted.load_session(item.id).event_count == 3


def test_redelivery_reapplies_an_update_a_crash_left_unsaved(tmp_path: Path) -> None:
    store = CodingStore(tmp_path)
    item = session()
    item.status = SessionStatus.running
    item.active_turn_id = "turn-1"
    store.save_session(item)
    metadata = tmp_path / "sessions" / item.id / "session.json"
    before_update = metadata.read_bytes()

    def complete_turn(current: CodingSession) -> None:
        current.status = SessionStatus.completed
        current.active_turn_id = None

    def completed_event() -> CodingEvent:
        return CodingEvent(type=EventType.session, title="Completed", source_event_id="runtime-event-1")

    stored = store.append_event(item.id, completed_event(), complete_turn)
    metadata.write_bytes(before_update)

    redelivered = CodingStore(tmp_path).append_event(item.id, completed_event(), complete_turn)

    restored = store.load_session(item.id)
    assert restored.status == SessionStatus.completed
    assert restored.active_turn_id is None
    assert redelivered.seq == stored.seq
    assert [event.seq for event in store.events_after(item.id)] == [1]


def test_reconcile_marks_orphaned_turn_interrupted(tmp_path: Path) -> None:
    store = CodingStore(tmp_path)
    item = session()
    item.status = SessionStatus.awaiting_approval
    item.active_turn_id = "turn-1"
    store.save_session(item)

    store.reconcile_interrupted()

    restored = store.load_session(item.id)
    assert restored.status == SessionStatus.interrupted
    assert restored.active_turn_id is None
    assert restored.pending_approval is None
    assert store.events_after(item.id)[-1].title == "Task interrupted"


def test_compaction_keeps_recent_events_and_sequence_cursor(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(store_module, "MAX_EVENTS", 4)
    store = CodingStore(tmp_path)
    compact = store._compact_events
    compactions = 0

    def count_compaction(item: CodingSession) -> None:
        nonlocal compactions
        compactions += 1
        compact(item)

    monkeypatch.setattr(store, "_compact_events", count_compaction)
    item = session()
    store.save_session(item)
    for index in range(6):
        store.append_event(item.id, CodingEvent(type=EventType.agent_message, text=str(index)))

    retained = store.events_after(item.id)
    assert [event.seq for event in retained] == [4, 5, 6]
    assert store.load_session(item.id).event_count == 6
    assert compactions == 1


def test_invalid_session_id_cannot_escape_store_root(tmp_path: Path) -> None:
    store = CodingStore(tmp_path)
    with pytest.raises(ValueError, match="invalid coding session id"):
        store.load_session("../outside")


def test_delete_session_removes_only_its_local_record(tmp_path: Path) -> None:
    store = CodingStore(tmp_path)
    first = session("first")
    second = session("second")
    store.save_session(first)
    store.save_session(second)

    store.delete_session("first")

    assert [item.id for item in store.list_sessions()] == ["second"]
    with pytest.raises(FileNotFoundError):
        store.load_session("first")


def test_active_stream_cursor_uses_bounded_recent_event_cache(tmp_path: Path, monkeypatch) -> None:
    store = CodingStore(tmp_path)
    item = session()
    store.save_session(item)
    store.append_event(item.id, CodingEvent(type=EventType.agent_message, text="first"))
    store.append_event(item.id, CodingEvent(type=EventType.agent_message, text="second"))
    monkeypatch.setattr(Path, "read_text", lambda *_args, **_kwargs: pytest.fail("disk rescan"))

    events = store.events_after(item.id, 1)

    assert [event.text for event in events] == ["second"]
