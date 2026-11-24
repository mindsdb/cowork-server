from __future__ import annotations

import json
import logging
import os
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from cowork.common.paths import dir_atomic_write, pinned_dir

_log = logging.getLogger(__name__)

_IS_WINDOWS = os.name == "nt"

_OUTCOME_TTL = timedelta(minutes=20)

_RESERVED_VAULT_KEYS = frozenset({
    "auth_type", "access_token", "refresh_token", "token_type",
    "scope", "expires_at", "account_email", "account_name",
})

# One lock for every store in the process, held across each read and each
# read-change-save. A sign-in start and the status poll run on the event loop,
# and the provider's callback runs on a file-store thread. Each change reads
# the whole file and writes it back, so without the lock one change can drop
# another's. On Windows, a read that holds the file open also makes the save's
# rename over it fail.
_LOCK = threading.Lock()


class OAuthStateStore:
    def __init__(self, state_path: str):
        self._path = Path(state_path)

    def _load(self) -> dict[str, Any]:
        # Only a missing file or folder reads as empty. Any other failed read
        # raises, so the change that called it never saves over entries it
        # could not read. A file that is not JSON is corrupt, since every save
        # renames a whole file into place, and the next save replaces it.
        try:
            text = self._path.read_text(encoding="utf-8")
        except (FileNotFoundError, NotADirectoryError):
            return {}
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return {}

    def _save(self, *, data: dict[str, Any]) -> None:
        # The file holds a connector's client secret, so only the user may
        # read it. dir_atomic_write renames a new 0600 file over the name, so
        # an older, wider file ends at 0600, and a link or a second name
        # planted there is replaced, never written through. The file's own
        # folder, when missing, is created 0700; folders above it take the
        # default mode. An existing folder keeps its mode, since it can be a
        # home folder the user chose. On Windows, Python gives a folder made
        # with 0o700 an owner-only access list that drops the entries the
        # profile passes down, so there the folder takes the default.
        content = json.dumps(data, indent=2).encode("utf-8")
        folder_mode = 0o777 if _IS_WINDOWS else 0o700
        try:
            self._path.parent.mkdir(mode=folder_mode, parents=True, exist_ok=True)
            with pinned_dir(self._path.parent) as folder:
                dir_atomic_write(folder, self._path.name, content=content)
        except (OSError, ValueError) as exc:
            # dir_atomic_write raises ValueError for a path with no file name,
            # such as one that ends in "..".
            _log.warning("Could not save OAuth state to %s: %s", self._path, exc)

    def read_all(self) -> dict[str, Any]:
        """The whole file: each service's entry and the stored outcomes."""
        with _LOCK:
            return self._load()

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
        with _LOCK:
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
            self._save(data=data)

    def get_pending(self, service: str) -> dict[str, Any] | None:
        with _LOCK:
            entry = self._load().get(service) or {}
        pending = entry.get("pending") or {}
        return pending if pending else None

    def set_outcome(self, state: str, outcome: dict[str, Any]) -> None:
        with _LOCK:
            data = self._load()
            outcomes = data.setdefault("_outcomes", {})
            outcomes[state] = {**outcome, "_ts": datetime.now(timezone.utc).isoformat()}
            self._save(data=data)

    def get_outcome(self, state: str) -> dict[str, Any] | None:
        with _LOCK:
            data = self._load()
            outcomes = data.get("_outcomes", {})
            entry = outcomes.get(state)
            if entry is None:
                return None
            try:
                ts = datetime.fromisoformat(entry["_ts"])
                if datetime.now(timezone.utc) - ts > _OUTCOME_TTL:
                    outcomes.pop(state)
                    self._save(data=data)
                    return None
            except (KeyError, ValueError) as exc:
                _log.debug("Could not parse outcome timestamp for state %r: %s", state, exc)
        return {k: v for k, v in entry.items() if k != "_ts"}

    def clear_outcome(self, state: str) -> None:
        with _LOCK:
            data = self._load()
            data.get("_outcomes", {}).pop(state, None)
            self._save(data=data)

    def clear_pending(self, service: str, *, error: str = "") -> None:
        with _LOCK:
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
            self._save(data=data)
