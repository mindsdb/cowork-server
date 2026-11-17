"""MCP connection values the desktop app holds, kept in server memory only.

When the server holds the app's secret (cowork.app_secret), an MCP
connection's access token and tool-access mode don't live in its vault file,
which agent code running as the same OS user can read and edit. The app pushes
them here (``PUT .../app-held``) after every server start, and every server-side
read of the connection lays them over the file's values. A value not pushed yet
falls back to the file's.

Keyed by the vault folder too, so two scopes' connections with the same name
never share an entry.
"""
from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

from cowork.app_secret import get_app_secret
from cowork.services.connectors.layered_vault import FilteredDataVault

MCP_METHOD = "mcp"

# Store key -> the vault field it stands in for.
VAULT_FIELD = {"access_token": "access_token", "expires_at": "expires_at", "access_mode": "_access_mode"}

_entries: dict[tuple[str, str, str], dict[str, str]] = {}
_lock = threading.Lock()


def active() -> bool:
    """Whether MCP values are app-held: only while the server holds the app's secret."""
    return get_app_secret() is not None


def _key(root: Path, engine: str, name: str) -> tuple[str, str, str]:
    return (str(root), engine, name)


def held(root: Path, engine: str, name: str) -> dict[str, str]:
    with _lock:
        return dict(_entries.get(_key(root, engine, name), {}))


def hold(root: Path, engine: str, name: str, values: dict[str, Any]) -> dict[str, str]:
    """Merge the non-empty ``values`` into the entry and return it."""
    with _lock:
        entry = _entries.setdefault(_key(root, engine, name), {})
        for key, value in values.items():
            if key in VAULT_FIELD and value:
                entry[key] = str(value)
        return dict(entry)


def _hold_token_and_default_mode(root: Path, engine: str, name: str, token: str | None, expires_at: str | None) -> None:
    with _lock:
        entry = _entries.setdefault(_key(root, engine, name), {})
        if token:
            entry["access_token"] = token
            if expires_at:
                entry["expires_at"] = str(expires_at)
        entry.setdefault("access_mode", "read")


def release(root: Path, engine: str, name: str) -> None:
    with _lock:
        _entries.pop(_key(root, engine, name), None)


def overlay(root: Path, engine: str, name: str, fields: dict, keys: tuple[str, ...] = tuple(VAULT_FIELD)) -> dict:
    """``fields`` with the held values for ``keys`` laid over them, for an MCP
    connection while app-held; ``fields`` unchanged otherwise."""
    if not active() or fields.get("_method") != MCP_METHOD:
        return fields
    entry = held(root, engine, name)
    if not entry:
        return fields
    out = dict(fields)
    for key in keys:
        if entry.get(key):
            out[VAULT_FIELD[key]] = entry[key]
    return out


class AppHeldVault(FilteredDataVault):
    """A vault whose MCP connections read their app-held values.

    ``load`` carries the token, its expiry and the access mode (anton's MCP
    wiring reads the token from it); ``read_record`` carries only the access
    mode (the wiring reads the mode from it), so the agent's connection listing,
    which reads records, never names a DS_* token variable its environment
    doesn't have. ``env_for``/``inject_env`` read the file alone: the token
    must never reach a Scratchpad's environment.

    The access mode is the held one, else "read", never the file's: agent code
    can write the file, or delete the connection and save it again, while only
    the secret-guarded routes may set a mode. A save of an MCP connection keeps
    the token out of the file, holds it here, ignores any mode it carries and
    holds "read" when no mode is held.
    """

    def __init__(self, *, source: Any, root: Path) -> None:
        super().__init__(source=source, hidden=frozenset())
        self._root = root

    def _held_mode(self, engine: str, name: str, fields: dict) -> dict:
        if not active() or fields.get("_method") != MCP_METHOD:
            return fields
        return {**fields, "_access_mode": held(self._root, engine, name).get("access_mode") or "read"}

    def load(self, engine: str, name: str) -> dict[str, str] | None:
        fields = super().load(engine, name)
        if fields is None:
            return None
        return self._held_mode(engine, name, overlay(self._root, engine, name, fields))

    def read_record(self, engine: str, name: str) -> dict[str, Any] | None:
        record = super().read_record(engine, name)
        if record is None:
            return None
        return {**record, "fields": self._held_mode(engine, name, record.get("fields") or {})}

    def save(
        self,
        engine: str,
        name: str,
        credentials: dict[str, str],
        *,
        secure_keys: list[str] | None = None,
    ) -> object:
        if active() and credentials.get("_method") == MCP_METHOD:
            credentials = dict(credentials)
            token = credentials.pop("access_token", None)
            credentials.pop("_access_mode", None)
            _hold_token_and_default_mode(self._root, engine, name, token, credentials.get("expires_at"))
        return super().save(engine, name, credentials, secure_keys=secure_keys)
