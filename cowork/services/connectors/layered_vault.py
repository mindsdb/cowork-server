"""Every saved connection one scope reads, and who may change each one.

``cowork.services.connectors.persist.vault_for_scope`` gives every reader this
vault on a deployment that trusts identity headers or has a managed folder:
in-process turns, an artifact's live backend, the connections API and
request-entry secret scrubbing. It satisfies anton's DataVault protocol, so
anton reads it as it reads its own vault.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, NamedTuple

from anton.core.datasources.data_vault import DataVault

from cowork.creators import Changer
from cowork.managed import SharedConnection

SHARED_CONNECTION_REFUSAL = "This connection is shared with you and can't be changed here."
# A create under a name the registry still shares for one of the reader's
# own connections that is gone. The server never writes the registry, so the
# new connection would be shared with everyone without the operator sharing
# it.
STILL_SHARED_CONNECTION_REFUSAL = (
    "This name is still shared for a connection of yours that is gone. "
    "Ask your operator to stop sharing it before you use the name again."
)

# The bookkeeping field that records who created a connection: the person's
# sub. A leading underscore keeps it out of the agent's environment (anton's
# DataVault.env_for and inject_env skip `_` fields) and out of the
# Connected Data Sources listing; ConnectionsService.get drops it from the
# API's fields.
CREATED_BY_FIELD = "_created_by"

# The bookkeeping fields that name a connection, which anton de-duplicates a
# new connection's label against (anton.utils.datasources
# .ensure_unique_user_label).
LABEL_FIELDS = frozenset({"_user_label", "_label"})


class SharedConnectionError(PermissionError):
    """A save or a delete reached a connection the saver can't change, or a
    create a name the registry still shares.

    The server answers it with 403 and ``detail``, SHARED_CONNECTION_REFUSAL
    or STILL_SHARED_CONNECTION_REFUSAL (cowork.server.create_app's exception
    handler).
    """

    def __init__(self, *, engine: str, name: str, detail: str = SHARED_CONNECTION_REFUSAL) -> None:
        super().__init__(detail)
        self.engine = engine
        self.name = name
        self.detail = detail


@dataclass(frozen=True)
class SharedFrom:
    """A registry entry of someone else's, with that person's own vault."""

    entry: SharedConnection
    vault: DataVault


class _Found(NamedTuple):
    vault: DataVault
    record: dict[str, Any] | None


class LayeredDataVault:
    """The connections one scope reads, in order: the managed folder's
    (``managed``), the scope's own (``own``), then the ones other people share
    with it (``shared``), each read live from its creator's own vault. A
    same-named entry earlier in that order hides a later one, and a shared
    entry whose connection is gone is skipped.

    Only the scope's own entries can change: a save or delete of a managed or
    shared one raises SharedConnectionError, and so does a create under a
    name the registry still shares for one of the scope's own (``shared_by_own``)
    that is gone, and a save or delete of an own entry
    the changer may not change where the own vault is deployment-wide
    (Changer.may_change): someone else's for a person, and any for a turn
    with no person. Every other save and delete goes to the own vault,
    a turn's included, so what the agent saves in a turn outlives the turn.
    With a ``changer``, every save records the connection's creator in
    CREATED_BY_FIELD: a create stamps the changer's (Changer.creator), and an
    update keeps the stored value, whatever the saved fields say.
    """

    def __init__(
        self,
        *,
        own: DataVault,
        managed: DataVault | None = None,
        shared: tuple[SharedFrom, ...] = (),
        shared_by_own: frozenset[tuple[str, str]] = frozenset(),
        changer: Changer | None = None,
    ) -> None:
        self._own = own
        self._managed = managed
        self._shared = shared
        self._shared_by_own = shared_by_own
        self._changer = changer

    def _find(self, engine: str, name: str) -> _Found:
        """The vault that answers for (engine, name), with its record; the
        own vault and no record when none holds it."""
        if self._managed is not None:
            record = self._managed.read_record(engine, name)
            if record is not None:
                return _Found(self._managed, record)
        record = self._own.read_record(engine, name)
        if record is not None:
            return _Found(self._own, record)
        for source in self._shared:
            if (source.entry.engine, source.entry.name) == (engine, name):
                record = source.vault.read_record(engine, name)
                if record is not None:
                    return _Found(source.vault, record)
        return _Found(self._own, None)

    @staticmethod
    def _creator(record: dict[str, Any]) -> str | None:
        return str((record.get("fields") or {}).get(CREATED_BY_FIELD) or "") or None

    def _may_change(self, found: _Found) -> bool:
        if found.vault is not self._own:
            return False
        return (
            found.record is None
            or self._changer is None
            or self._changer.may_change(self._creator(found.record))
        )

    def can_change(self, engine: str, name: str) -> bool:
        """Whether a save or delete of (engine, name) goes through: false for a
        managed or shared entry, and for an own entry the changer may not
        change where the own vault is deployment-wide (Changer.may_change).
        True for one nobody holds."""
        return self._may_change(self._find(engine, name))

    def is_shared(self, engine: str, name: str) -> bool:
        """Whether (engine, name) is in the shared-connection registry: one
        someone shares with this scope, or one of the scope's own it shares."""
        found = self._find(engine, name)
        if found.record is None or found.vault is self._managed:
            return False
        if found.vault is self._own:
            return (engine, name) in self._shared_by_own
        return True

    def _stamped(self, found: _Found, credentials: dict[str, str]) -> dict[str, str]:
        """``credentials`` with CREATED_BY_FIELD as the record keeps it: the
        stored creator on an update, the changer's on a create, and none
        where neither names one. Never the value the caller passed."""
        fields = dict(credentials)
        if self._changer is None:
            return fields
        if found.record is not None:
            creator = self._creator(found.record)
        else:
            creator = self._changer.creator
        fields.pop(CREATED_BY_FIELD, None)
        if creator is not None:
            fields[CREATED_BY_FIELD] = creator
        return fields

    def save(
        self,
        engine: str,
        name: str,
        credentials: dict[str, str],
        *,
        secure_keys: list[str] | None = None,
    ) -> object:
        found = self._find(engine, name)
        if not self._may_change(found):
            raise SharedConnectionError(engine=engine, name=name)
        if found.record is None and (engine, name) in self._shared_by_own:
            raise SharedConnectionError(
                engine=engine, name=name, detail=STILL_SHARED_CONNECTION_REFUSAL
            )
        return self._own.save(
            engine, name, self._stamped(found, credentials), secure_keys=secure_keys
        )

    def load(self, engine: str, name: str) -> dict[str, str] | None:
        return self._find(engine, name).vault.load(engine, name)

    def read_record(self, engine: str, name: str) -> dict[str, Any] | None:
        return self._find(engine, name).record

    def delete(self, engine: str, name: str) -> bool:
        if not self._may_change(self._find(engine, name)):
            raise SharedConnectionError(engine=engine, name=name)
        return self._own.delete(engine, name)

    def list_connections(self) -> list[dict[str, str]]:
        listed: list[dict[str, str]] = []
        taken: set[tuple[str, str]] = set()
        for vault in (self._managed, self._own):
            if vault is None:
                continue
            for entry in vault.list_connections():
                key = (entry["engine"], entry["name"])
                if key not in taken:
                    taken.add(key)
                    listed.append(entry)
        for source in self._shared:
            key = (source.entry.engine, source.entry.name)
            if key in taken:
                continue
            record = source.vault.read_record(*key)
            if record is None:
                continue
            taken.add(key)
            listed.append(
                {
                    "engine": source.entry.engine,
                    "name": source.entry.name,
                    "created_at": record.get("created_at", ""),
                }
            )
        return listed

    def env_for(self, engine: str, name: str, *, flat: bool = False) -> dict[str, str] | None:
        return self._find(engine, name).vault.env_for(engine, name, flat=flat)

    def inject_env(self, engine: str, name: str, *, flat: bool = False) -> list[str] | None:
        return self._find(engine, name).vault.inject_env(engine, name, flat=flat)

    def clear_ds_env(self) -> None:
        self._own.clear_ds_env()

    def for_connecting(self) -> ConnectingDataVault:
        """This vault as anton's connect_new_datasource matches a new
        connection against it (ConnectingDataVault)."""
        return ConnectingDataVault(
            own=self._own,
            managed=self._managed,
            shared=self._shared,
            shared_by_own=self._shared_by_own,
            changer=self._changer,
        )


class ConnectingDataVault(LayeredDataVault):
    """A LayeredDataVault whose entries the reader can't change load as their
    labels alone.

    anton's connect_new_datasource (anton.tools.handle_connect_datasource)
    updates the saved connection whose identity fields match the new one's,
    the engine's ``name_from``: the database alone for postgres, mysql, mssql
    and mariadb. Matched to a connection someone shares with the reader, or
    to someone else's where connections are shared, the save over it is
    refused, and the reader's own connection is never saved. Loaded as its
    labels alone, such an entry matches nothing, so the new connection is
    saved as the reader's own, and its label is still de-duplicated against
    every label the reader sees. Every other read is the LayeredDataVault's,
    so the turn's DS_* scrubbing, which reads read_record and env_for, still
    covers every entry.
    """

    def load(self, engine: str, name: str) -> dict[str, str] | None:
        fields = super().load(engine, name)
        if fields is None or self.can_change(engine, name):
            return fields
        return {key: value for key, value in fields.items() if key in LABEL_FIELDS}
