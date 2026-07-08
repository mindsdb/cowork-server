from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class OAuthStateStore:
    def __init__(self, state_path: str):
        self._path = Path(state_path)

    def _load(self) -> dict[str, Any]:
        if not self._path.exists():
            return {}
        try:
            return json.loads(self._path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return {}

    def _save(self, data: dict[str, Any]) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def set_pending(
        self,
        service: str,
        *,
        state: str,
        verifier: str,
        redirect_uri: str,
        started_at: str,
    ) -> None:
        data = self._load()
        entry = data.get(service) or {}
        entry["pending"] = {
            "state": state,
            "verifier": verifier,
            "redirectUri": redirect_uri,
            "startedAt": started_at,
        }
        entry.setdefault("lastSuccessAt", "")
        entry.setdefault("lastError", "")
        entry.setdefault("lastErrorAt", "")
        data[service] = entry
        self._save(data)

    def get_pending(self, service: str) -> dict[str, Any] | None:
        entry = self._load().get(service) or {}
        pending = entry.get("pending") or {}
        return pending if pending else None

    def clear_pending(self, service: str, *, error: str = "", connection_name: str = "") -> None:
        data = self._load()
        entry = data.get(service) or {}
        entry["pending"] = {}
        now = datetime.now(timezone.utc).isoformat()
        if error:
            entry["lastError"] = error
            entry["lastErrorAt"] = now
        else:
            entry["lastSuccessAt"] = now
            entry["lastError"] = ""
            entry["lastErrorAt"] = ""
            if connection_name:
                entry["connectionName"] = connection_name
        data[service] = entry
        self._save(data)
