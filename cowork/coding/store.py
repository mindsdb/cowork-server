from __future__ import annotations

import json
import os
import shutil
import threading
from collections import deque
from collections.abc import Callable
from pathlib import Path

from cowork.coding.contracts import CodingEvent, CodingSession, SessionStatus, utc_now

MAX_EVENTS = 6_000
MAX_EVENT_FILE_BYTES = 32 * 1024 * 1024
MAX_RECENT_EVENTS = 512


class CodingStore:
    """Versioned, bounded, crash-safe local persistence for coding sessions."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.sessions_root = root / "sessions"
        self.sessions_root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._changed = threading.Condition(self._lock)
        self._recent_events: deque[tuple[str, CodingEvent]] = deque(maxlen=MAX_RECENT_EVENTS)
        self._last_sequences: dict[str, int] = {}
        self._retained_counts: dict[str, int] = {}
        self._source_event_ids: dict[str, set[str]] = {}

    def _dir(self, session_id: str) -> Path:
        if not session_id or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for c in session_id):
            raise ValueError("invalid coding session id")
        return self.sessions_root / session_id

    def _meta_path(self, session_id: str) -> Path:
        return self._dir(session_id) / "session.json"

    def _events_path(self, session_id: str) -> Path:
        return self._dir(session_id) / "events.jsonl"

    def save_session(self, session: CodingSession, *, touch_updated_at: bool = True) -> None:
        with self._lock:
            target_dir = self._dir(session.id)
            target_dir.mkdir(parents=True, exist_ok=True)
            if touch_updated_at:
                session.updated_at = utc_now()
            target = self._meta_path(session.id)
            temp = target.with_suffix(".tmp")
            data = session.model_dump_json(indent=2)
            temp.write_text(data + "\n", encoding="utf-8")
            os.replace(temp, target)
            self._changed.notify_all()

    def update_session(
        self,
        session_id: str,
        update: Callable[[CodingSession], None],
        *,
        touch_updated_at: bool = True,
    ) -> CodingSession:
        """Apply a state transition to the latest persisted session atomically."""
        with self._lock:
            session = self.load_session(session_id)
            update(session)
            self.save_session(session, touch_updated_at=touch_updated_at)
            return session

    def load_session(self, session_id: str) -> CodingSession:
        with self._lock:
            raw = json.loads(self._meta_path(session_id).read_text(encoding="utf-8"))
            return self._migrate(raw)

    def list_sessions(self) -> list[CodingSession]:
        with self._lock:
            sessions: list[CodingSession] = []
            for path in self.sessions_root.glob("*/session.json"):
                try:
                    sessions.append(self._migrate(json.loads(path.read_text(encoding="utf-8"))))
                except (OSError, ValueError, TypeError, json.JSONDecodeError):
                    continue
            return sorted(sessions, key=lambda item: item.updated_at, reverse=True)

    def delete_session(self, session_id: str) -> None:
        with self._lock:
            target = self._dir(session_id)
            if not target.is_dir():
                raise FileNotFoundError(session_id)
            shutil.rmtree(target)
            self._recent_events = deque(
                ((stored_id, event) for stored_id, event in self._recent_events if stored_id != session_id),
                maxlen=MAX_RECENT_EVENTS,
            )
            self._last_sequences.pop(session_id, None)
            self._retained_counts.pop(session_id, None)
            self._source_event_ids.pop(session_id, None)
            self._changed.notify_all()

    def copy_event_history(self, source_id: str, target: CodingSession) -> None:
        """Copy a task's visible history into a newly persisted fork."""
        with self._lock:
            source = self._events_path(source_id)
            destination = self._events_path(target.id)
            destination.parent.mkdir(parents=True, exist_ok=True)
            if source.exists():
                shutil.copyfile(source, destination)
            target.event_count = self.load_session(source_id).event_count
            self.save_session(target)
            self._last_sequences.pop(target.id, None)
            self._retained_counts.pop(target.id, None)
            self._source_event_ids.pop(target.id, None)

    def append_event(
        self,
        session_id: str,
        event: CodingEvent,
        update: Callable[[CodingSession], None] | None = None,
    ) -> CodingEvent:
        """Append against current metadata so concurrent callbacks cannot regress it.

        An event carrying a ``source_event_id`` already stored for the session is
        a redelivery: ``update`` is still applied, so metadata a crash left
        behind catches up, but the stored event is returned instead of a copy.
        """
        with self._lock:
            session = self.load_session(session_id)
            last_sequence = self._last_sequences.get(session_id)
            retained_count = self._retained_counts.get(session_id)
            source_ids = self._source_event_ids.get(session_id)
            if last_sequence is None or retained_count is None or source_ids is None:
                stored = self.events_after(session_id)
                if last_sequence is None:
                    last_sequence = stored[-1].seq if stored else 0
                if retained_count is None:
                    retained_count = len(stored)
                if source_ids is None:
                    source_ids = {item.source_event_id for item in stored if item.source_event_id}
                    self._source_event_ids[session_id] = source_ids
            if update is not None:
                update(session)
            if event.source_event_id and event.source_event_id in source_ids:
                self.save_session(session)
                return next(item for item in self.events_after(session_id) if item.source_event_id == event.source_event_id)
            event.seq = max(session.event_count, last_sequence) + 1
            session.event_count = event.seq
            target_dir = self._dir(session.id)
            target_dir.mkdir(parents=True, exist_ok=True)
            path = self._events_path(session.id)
            with path.open("a", encoding="utf-8", newline="\n") as handle:
                handle.write(event.model_dump_json() + "\n")
                handle.flush()
            self._last_sequences[session_id] = event.seq
            self._retained_counts[session_id] = retained_count + 1
            if event.source_event_id:
                source_ids.add(event.source_event_id)
            self._recent_events.append((session.id, event))
            if self._retained_counts[session_id] > MAX_EVENTS or path.stat().st_size > MAX_EVENT_FILE_BYTES:
                self._compact_events(session)
            self.save_session(session)
            self._changed.notify_all()
            return event

    def events_after(self, session_id: str, after: int = 0) -> list[CodingEvent]:
        with self._lock:
            recent = [event for stored_id, event in self._recent_events if stored_id == session_id]
            if recent and after >= recent[0].seq - 1:
                return [event for event in recent if event.seq > after]
            path = self._events_path(session_id)
            if not path.exists():
                return []
            events: list[CodingEvent] = []
            for line in path.read_text(encoding="utf-8").splitlines():
                try:
                    event = CodingEvent.model_validate_json(line)
                except ValueError:
                    continue
                if event.seq > after:
                    events.append(event)
            return events

    def wait_for_events(self, session_id: str, after: int, timeout: float = 15.0) -> list[CodingEvent]:
        with self._changed:
            events = self.events_after(session_id, after)
            if events:
                return events
            self._changed.wait(timeout=timeout)
            return self.events_after(session_id, after)

    def reconcile_interrupted(self) -> None:
        """A new server process cannot own a previous process's active turn."""
        for session in self.list_sessions():
            if session.status not in {SessionStatus.running, SessionStatus.awaiting_approval}:
                continue
            self.append_event(
                session.id,
                CodingEvent(
                    type="session",
                    title="Task interrupted",
                    text="The app stopped while this task was running. Send a follow-up to resume the same coding session.",
                    phase="failed",
                ),
                self._mark_interrupted,
            )

    def _compact_events(self, session: CodingSession) -> None:
        path = self._events_path(session.id)
        events = self.events_after(session.id, 0)[-MAX_EVENTS // 2 :]
        # Preserve monotonic sequence ids across compaction; clients can keep
        # their cursor and simply receive the retained tail.
        temp = path.with_suffix(".tmp")
        temp.write_text("".join(event.model_dump_json() + "\n" for event in events), encoding="utf-8")
        os.replace(temp, path)
        self._retained_counts[session.id] = len(events)
        self._source_event_ids[session.id] = {item.source_event_id for item in events if item.source_event_id}
        if events:
            first_retained = events[0].seq
            self._recent_events = deque(
                (
                    (stored_id, event)
                    for stored_id, event in self._recent_events
                    if stored_id != session.id or event.seq >= first_retained
                ),
                maxlen=MAX_RECENT_EVENTS,
            )

    @staticmethod
    def _migrate(raw: dict) -> CodingSession:
        version = raw.get("schema_version", 1)
        if version != 1:
            raise ValueError(f"unsupported coding session schema {version}")
        return CodingSession.model_validate(raw)

    @staticmethod
    def _mark_interrupted(session: CodingSession) -> None:
        session.status = SessionStatus.interrupted
        session.active_turn_id = None
        session.pending_approval = None
