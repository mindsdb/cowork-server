"""Every saved connection one scope reads, and who may change each one.

``cowork.services.connectors.persist.vault_for_scope`` gives every reader this
vault on a deployment that trusts identity headers or has a managed folder:
in-process turns, an artifact's live backend, the connections API and
request-entry secret scrubbing. It satisfies anton's DataVault protocol, so
anton reads it as it reads its own vault.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, NamedTuple

from anton.core.datasources.data_vault import DataVault, _slug_env_prefix

from cowork.creators import Changer
from cowork.managed import SharedConnection

logger = logging.getLogger(__name__)

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


class _PrefixHolder(NamedTuple):
    """The first connection listed under one DS_ env prefix, and its layer."""

    layer: int
    engine: str
    name: str


class LayeredDataVault:
    """The connections one scope reads, in order: the managed folder's
    (``managed``), the scope's own (``own``), then the ones other people share
    with it (``shared``), each read live from its creator's own vault. A
    same-named entry earlier in that order hides a later one, and a shared
    entry whose connection is gone is skipped.

    Only the scope's own entries can change: a save or delete of a managed or
    shared one raises SharedConnectionError. So does a create under a name
    the registry still shares for one of the scope's own that is gone
    (``shared_by_own``), and a save or delete of an own entry the changer may
    not change where the own vault is deployment-wide (Changer.may_change):
    someone else's for a person, and any for a turn with no person. Every
    other save and delete goes to the own vault, a turn's included, so what
    the agent saves in a turn outlives the turn.
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
        """Whether (engine, name) reads as shared with this scope rather than
        the scope's own to change: one someone shares through the registry, one
        of the scope's own it shares, or, in a deployment-wide vault, another
        person's own entry this person didn't create. A managed entry and a
        creator-less base entry stay the operator's (not shared), so the UI
        shows "your operator provides this" for those and "shared with you" for
        a teammate's."""
        found = self._find(engine, name)
        if found.record is None or found.vault is self._managed:
            return False
        if found.vault is self._own:
            if (engine, name) in self._shared_by_own:
                return True
            changer = self._changer
            creator = self._creator(found.record)
            return (
                changer is not None
                and changer.deployment_wide
                and not changer.operator_view
                and changer.user_id is not None
                and creator is not None
                and creator != changer.user_id
            )
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

    def _own_layers(self) -> list[list[dict[str, str]]]:
        """The own vault's entries as listing layers: one layer, or, where
        the vault is deployment-wide, the entries the changer may change
        first and everyone else's after them, so the reader's own entry wins
        a DS_ env prefix over another person's."""
        entries = self._own.list_connections()
        if self._changer is None or not self._changer.deployment_wide:
            return [entries]
        mine: list[dict[str, str]] = []
        others: list[dict[str, str]] = []
        for entry in entries:
            record = self._own.read_record(entry["engine"], entry["name"])
            (mine if self._may_change(_Found(self._own, record)) else others).append(entry)
        return [mine, others]

    def list_connections(self) -> list[dict[str, str]]:
        """Every connection this scope reads, in layers: the managed folder's,
        the scope's own (in a deployment-wide vault, the reader's own before
        everyone else's), then the ones shared with it.

        A same-named entry in an earlier layer hides a later one. So does an
        earlier layer's entry under the same DS_ env prefix: anton builds the
        agent's DS_* environment per listed connection, keyed by
        _slug_env_prefix, which maps '-', '_' and case onto one name, so the
        later entry's credentials would overwrite the earlier one's. Two
        entries of one layer that share a prefix both stay listed, as on the
        desktop.
        """
        listed: list[dict[str, str]] = []
        taken: set[tuple[str, str]] = set()
        prefixes: dict[str, _PrefixHolder] = {}

        def admits(layer: int, engine: str, name: str) -> bool:
            prefix = _slug_env_prefix(engine, name)
            holder = prefixes.get(prefix)
            if holder is None:
                prefixes[prefix] = _PrefixHolder(layer=layer, engine=engine, name=name)
                return True
            if holder.layer == layer:
                return True
            logger.warning(
                "connection %s/%s shares the DS_ env prefix %r with %s/%s, listed "
                "ahead of it; it is hidden from this scope so its credentials never "
                "stand in for that connection's",
                engine, name, prefix, holder.engine, holder.name,
            )
            return False

        layers = [
            *([] if self._managed is None else [self._managed.list_connections()]),
            *self._own_layers(),
        ]
        for layer, entries in enumerate(layers):
            for entry in entries:
                key = (entry["engine"], entry["name"])
                if key in taken or not admits(layer, *key):
                    continue
                taken.add(key)
                listed.append(entry)
        shared_layer = len(layers)
        for source in self._shared:
            key = (source.entry.engine, source.entry.name)
            if key in taken:
                continue
            record = source.vault.read_record(*key)
            if record is None or not admits(shared_layer, *key):
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


class FilteredDataVault:
    """A turn's vault without the connections its conversation turned off.

    Every read skips a ``hidden`` (engine, name) as if nobody had saved it,
    so the agent's environment, its connection listing and anton's match
    of a pasted connection never see it. A save or delete goes to
    ``source`` unchanged, so what the agent saves in the turn lands where it
    would with nothing turned off and outlives the turn, and a
    LayeredDataVault source still refuses a change to a connection the
    asker can't change.
    """

    def __init__(self, *, source: DataVault, hidden: frozenset[tuple[str, str]]) -> None:
        self._source = source
        self._hidden = hidden

    def _hides(self, engine: str, name: str) -> bool:
        return (engine, name) in self._hidden

    def save(
        self,
        engine: str,
        name: str,
        credentials: dict[str, str],
        *,
        secure_keys: list[str] | None = None,
    ) -> object:
        return self._source.save(engine, name, credentials, secure_keys=secure_keys)

    def load(self, engine: str, name: str) -> dict[str, str] | None:
        return None if self._hides(engine, name) else self._source.load(engine, name)

    def read_record(self, engine: str, name: str) -> dict[str, Any] | None:
        return None if self._hides(engine, name) else self._source.read_record(engine, name)

    def delete(self, engine: str, name: str) -> bool:
        return self._source.delete(engine, name)

    def list_connections(self) -> list[dict[str, str]]:
        return [
            entry
            for entry in self._source.list_connections()
            if not self._hides(entry["engine"], entry["name"])
        ]

    def env_for(self, engine: str, name: str, *, flat: bool = False) -> dict[str, str] | None:
        return None if self._hides(engine, name) else self._source.env_for(engine, name, flat=flat)

    def inject_env(self, engine: str, name: str, *, flat: bool = False) -> list[str] | None:
        return None if self._hides(engine, name) else self._source.inject_env(engine, name, flat=flat)

    def clear_ds_env(self) -> None:
        self._source.clear_ds_env()

    def for_connecting(self) -> FilteredDataVault:
        """This vault as anton's connect_new_datasource matches a new
        connection against it: over the source's ConnectingDataVault where
        the source is a LayeredDataVault, with the same connections listed by
        label alone (_ConnectingFilteredDataVault)."""
        source = (
            self._source.for_connecting()
            if isinstance(self._source, LayeredDataVault)
            else self._source
        )
        return _ConnectingFilteredDataVault(source=source, hidden=self._hidden)


class _ConnectingFilteredDataVault(FilteredDataVault):
    """The connecting view of a FilteredDataVault.

    A hidden connection is still listed, so a pasted connection's label is
    de-duplicated against it and the turn's new connection never repeats a
    hidden one's label. It loads as its label fields alone, so it matches no
    identity field anton keys on: the paste is saved as the reader's own
    rather than silently updating the hidden connection. Every other read is
    the filtered vault's, so the turn's environment and listing still hide it.
    """

    def list_connections(self) -> list[dict[str, str]]:
        return self._source.list_connections()

    def load(self, engine: str, name: str) -> dict[str, str] | None:
        fields = self._source.load(engine, name)
        if fields is None or not self._hides(engine, name):
            return fields
        return {key: value for key, value in fields.items() if key in LABEL_FIELDS}

    def read_record(self, engine: str, name: str) -> dict[str, Any] | None:
        record = self._source.read_record(engine, name)
        if record is None or not self._hides(engine, name):
            return record
        fields = {k: v for k, v in (record.get("fields") or {}).items() if k in LABEL_FIELDS}
        return {**record, "fields": fields}
