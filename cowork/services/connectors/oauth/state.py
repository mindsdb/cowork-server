from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

_log = logging.getLogger(__name__)

_OUTCOME_TTL = timedelta(minutes=20)

_RESERVED_VAULT_KEYS = frozenset({
    "auth_type", "access_token", "refresh_token", "token_type",
    "scope", "expires_at", "account_email", "account_name",
})


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
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        except OSError as exc:
            _log.warning("Could not save OAuth state to %s: %s", self._path, exc)

    def set_pending(
        self,
        service: str,
        *,
        state: str,
        verifier: str,
        redirect_uri: str,
        started_at: str,
        client_id: str = "",
        client_secret: str = "",
        extra_fields: dict[str, str] | None = None,
    ) -> None:
        data = self._load()
        entry = data.get(service) or {}
        entry["pending"] = {
            "state": state,
            "verifier": verifier,
            "redirectUri": redirect_uri,
            "startedAt": started_at,
            "clientId": client_id,
            "clientSecret": client_secret,
            "extraFields": {k: v for k, v in (extra_fields or {}).items() if v and k not in _RESERVED_VAULT_KEYS},
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

    def set_outcome(self, state: str, outcome: dict[str, Any]) -> None:
        data = self._load()
        outcomes = data.setdefault("_outcomes", {})
        outcomes[state] = {**outcome, "_ts": datetime.now(timezone.utc).isoformat()}
        self._save(data)

    def get_outcome(self, state: str) -> dict[str, Any] | None:
        data = self._load()
        entry = data.get("_outcomes", {}).get(state)
        if entry is None:
            return None
        try:
            ts = datetime.fromisoformat(entry["_ts"])
            if datetime.now(timezone.utc) - ts > _OUTCOME_TTL:
                self.clear_outcome(state)
                return None
        except (KeyError, ValueError) as exc:
            _log.debug("Could not parse outcome timestamp for state %r: %s", state, exc)
        return {k: v for k, v in entry.items() if k != "_ts"}

    def clear_outcome(self, state: str) -> None:
        data = self._load()
        data.get("_outcomes", {}).pop(state, None)
        self._save(data)

    def clear_pending(self, service: str, *, error: str = "") -> None:
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
        data[service] = entry
        self._save(data)
