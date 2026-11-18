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

import json
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

from cowork.app_secret import get_app_secret
from cowork.services.connectors.custom_mcp import CUSTOM_MCP_ENGINE, is_custom_mcp, refuse_generic_write
from cowork.services.connectors.layered_vault import LABEL_FIELDS, FilteredDataVault, LayeredDataVault

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
        if is_custom_mcp(engine):
            _custom.pop(_key(root, engine, name), None)


#: A custom MCP connection's held settings and token. Kept apart from the
#: generic entries: none of these may ever come from the vault file.
CUSTOM_FIELDS = (
    "server_url",
    "label",
    "resource",
    "authorization_endpoint",
    "token_endpoint",
    "revocation_endpoint",
    "client_id",
    "allowed_tools",
    "access_token",
    "expires_at",
)

_custom: dict[tuple[str, str, str], dict[str, Any]] = {}


def hold_custom(root: Path, name: str, values: dict[str, Any]) -> dict[str, Any]:
    """Merge ``values`` (CUSTOM_FIELDS keys, None meaning unchanged) into a
    custom MCP connection's held entry and return it. A server URL that
    differs from the held one drops the held token and its expiry, unless the
    same call brings new ones, so a token never pairs with another server."""
    with _lock:
        entry = _custom.setdefault(_key(root, CUSTOM_MCP_ENGINE, name), {})
        url = values.get("server_url")
        if url and entry.get("server_url") and url != entry["server_url"]:
            entry.pop("access_token", None)
            entry.pop("expires_at", None)
        for key in CUSTOM_FIELDS:
            value = values.get(key)
            if value is None:
                continue
            entry[key] = list(value) if key == "allowed_tools" else str(value)
        return _copy(entry)


def held_custom(root: Path, name: str) -> dict[str, Any]:
    """A custom MCP connection's held entry, empty when nothing is held."""
    with _lock:
        return _copy(_custom.get(_key(root, CUSTOM_MCP_ENGINE, name), {}))


def _copy(entry: dict[str, Any]) -> dict[str, Any]:
    """A copy of a held entry that a caller cannot change it through."""
    return {key: list(value) if isinstance(value, list) else value for key, value in entry.items()}


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


def _secure_keys_naming_token(secure_keys: Any, fields: dict) -> list[str] | None:
    """A record's ``secure_keys`` that anton reads as covering a held token.

    anton takes an explicit list as authoritative, so a token it doesn't name
    would be neither scrubbed nor withheld. Agent code can write the file:
    anything but a list of names falls back to anton's name heuristic, which
    treats every field named like a token as secret."""
    if not isinstance(secure_keys, list) or not all(isinstance(key, str) for key in secure_keys):
        return None
    if "access_token" in fields and "access_token" not in secure_keys:
        return [*secure_keys, "access_token"]
    return secure_keys


class AppHeldVault(FilteredDataVault):
    """A vault whose MCP connections read their app-held values.

    ``load`` and ``read_record`` carry the token, its expiry and the access
    mode. The record marks the token secret, so anton adds it to the turn's
    scrub map: a server that echoes its token in a tool result or an error
    gets it redacted. anton withholds an MCP connection's secrets from the
    environment and from the agent's connection listing.
    ``env_for``/``inject_env`` read the file alone: the token must never reach
    a Scratchpad's environment.

    The access mode is the held one, else "read", never the file's: agent code
    can write the file, or delete the connection and save it again, while only
    the secret-guarded routes may set a mode. A save of an MCP connection keeps
    the token out of the file, holds it here, ignores any mode it carries and
    holds "read" when no mode is held.

    A custom MCP connection (``custom_mcp``) reads nothing from the file but
    its existence: its URL, label and allowed tools come from the held entry,
    and only for a record in the user's own layer (``is_own``). With nothing
    held its record carries ``_method`` alone, which anton skips.
    """

    def __init__(self, *, source: Any, root: Path, is_own: Callable[[str, str], bool] | None = None) -> None:
        """``is_own(engine, name)`` says whether the record comes from the
        user's own vault rather than a managed or shared one; without it every
        record counts as the user's own."""
        super().__init__(source=source, hidden=frozenset())
        self._root = root
        self._is_own = is_own or (lambda _engine, _name: True)

    def _held_mode(self, engine: str, name: str, fields: dict) -> dict:
        if not active() or fields.get("_method") != MCP_METHOD:
            return fields
        return {**fields, "_access_mode": held(self._root, engine, name).get("access_mode") or "read"}

    def _custom_fields(self, name: str) -> dict[str, str]:
        """A custom connection's fields, from the held entry only."""
        fields = {"_method": MCP_METHOD}
        if not active() or not self._is_own(CUSTOM_MCP_ENGINE, name):
            return fields
        entry = held_custom(self._root, name)
        if not entry.get("server_url") or "allowed_tools" not in entry:
            return fields
        fields.update(
            {
                "_server_url": entry["server_url"],
                "_allowed_tools": json.dumps(entry["allowed_tools"]),
                "_label": entry.get("label") or name,
                # Which tools run is decided by _allowed_tools; the access mode
                # only has to let the list through.
                "_access_mode": "write",
            }
        )
        if entry.get("access_token"):
            fields["access_token"] = entry["access_token"]
            if entry.get("expires_at"):
                fields["expires_at"] = entry["expires_at"]
        return fields

    def _custom_or_labels(self, name: str, source_fields: dict) -> dict[str, str]:
        """The held fields of a custom connection, or the source's label fields
        when the source turned it off: a connecting view lists a turned-off
        connection by its labels alone, and memory must not refill it. The
        file's own fields never decide which."""
        if self.hides(CUSTOM_MCP_ENGINE, name):
            return {key: value for key, value in source_fields.items() if key in LABEL_FIELDS}
        return self._custom_fields(name)

    def _held_fields(self, engine: str, name: str, fields: dict) -> dict:
        """The fields a turn reads: the held ones for a custom connection, the
        file's with the held token and mode laid over for any other."""
        if is_custom_mcp(engine):
            return self._custom_or_labels(name, fields)
        return self._held_mode(engine, name, overlay(self._root, engine, name, fields))

    def load(self, engine: str, name: str) -> dict[str, str] | None:
        fields = super().load(engine, name)
        if fields is None:
            return None
        return self._held_fields(engine, name, fields)

    def read_record(self, engine: str, name: str) -> dict[str, Any] | None:
        record = super().read_record(engine, name)
        if record is None:
            return None
        fields = self._held_fields(engine, name, record.get("fields") or {})
        if is_custom_mcp(engine):
            # The token is here for the scrub; its expiry would only make the
            # agent's listing name a variable no environment has.
            fields.pop("expires_at", None)
        return {**record, "fields": fields, "secure_keys": _secure_keys_naming_token(record.get("secure_keys"), fields)}

    def for_connecting(self) -> AppHeldVault:
        """The connecting view anton's connect_new_datasource matches against,
        still read through this vault, so a custom MCP record there reads
        from memory and a save of one is refused, as everywhere else."""
        return AppHeldVault(source=super().for_connecting(), root=self._root, is_own=self._is_own)

    def delete(self, engine: str, name: str) -> bool:
        """Delete the record, and for a custom MCP connection its held entry."""
        deleted = super().delete(engine, name)
        if is_custom_mcp(engine):
            release(self._root, engine, name)
        return deleted

    def save(
        self,
        engine: str,
        name: str,
        credentials: dict[str, str],
        *,
        secure_keys: list[str] | None = None,
    ) -> object:
        refuse_generic_write(engine)
        if active() and credentials.get("_method") == MCP_METHOD:
            credentials = dict(credentials)
            token = credentials.pop("access_token", None)
            credentials.pop("_access_mode", None)
            _hold_token_and_default_mode(self._root, engine, name, token, credentials.get("expires_at"))
        return super().save(engine, name, credentials, secure_keys=secure_keys)


def turn_vault(data_vault: Any, *, source_vault: Any, root: Path) -> Any:
    """The vault a turn reads: with the app secret held, ``data_vault`` read
    through AppHeldVault; without it, ``data_vault`` with every custom MCP
    connection hidden, since nothing then holds their settings and their
    vault file is not to be trusted. ``source_vault`` is the scope's vault,
    asked which layer a record comes from."""
    if active():
        return AppHeldVault(
            source=data_vault,
            root=root,
            is_own=source_vault.answers_from_own if isinstance(source_vault, LayeredDataVault) else None,
        )
    return FilteredDataVault(source=data_vault, hidden=frozenset(), hidden_engines=frozenset({CUSTOM_MCP_ENGINE}))
