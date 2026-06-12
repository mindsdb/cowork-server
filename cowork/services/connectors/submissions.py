from __future__ import annotations

import time
import uuid
from typing import Any


class SubmissionStore:
    def __init__(self, ttl_seconds: int = 24 * 60 * 60):
        self._ttl = ttl_seconds
        self._store: dict[str, dict[str, Any]] = {}

    def stage(
        self,
        *,
        form_id: str,
        connector_id: str,
        conversation_id: str | None,
        values: dict[str, Any],
        skipped: list[str] | None = None,
        form_spec: dict[str, Any] | None = None,
    ) -> str:
        self._purge_expired()
        submission_id = "sub_" + uuid.uuid4().hex[:12]
        self._store[submission_id] = {
            "submission_id": submission_id,
            "form_id": form_id,
            "connector_id": connector_id,
            "conversation_id": conversation_id,
            "values": dict(values or {}),
            "skipped": list(skipped or []),
            "form_spec": dict(form_spec) if form_spec else None,
            "created_at": time.time(),
            "status": "received",
        }
        # TODO: Is it necessary to return this?
        return submission_id

    def get(self, submission_id: str) -> dict[str, Any] | None:
        self._purge_expired()
        entry = self._store.get(submission_id)
        if entry is None:
            return None
        return {**entry, "values": dict(entry.get("values", {}))}

    def consume(self, submission_id: str) -> dict[str, Any] | None:
        self._purge_expired()
        return self._store.pop(submission_id, None)

    def _purge_expired(self) -> None:
        threshold = time.time() - self._ttl
        stale = [sid for sid, e in self._store.items() if e.get("created_at", 0) < threshold]
        for sid in stale:
            self._store.pop(sid, None)


store = SubmissionStore()
