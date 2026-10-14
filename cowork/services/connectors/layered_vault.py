"""The operator's managed connections layered over one scope's own vault.

Every reader that resolves a scope's saved connections through
``cowork.services.connectors.persist.vault_for_scope`` gets this vault while
COWORK_MANAGED_DIR is set: in-process turns, an artifact's live backend, the
connections API and request-entry secret scrubbing. It satisfies anton's
DataVault protocol, so anton reads it as it reads its own vault.
"""

from __future__ import annotations

from typing import Any

from anton.core.datasources.data_vault import DataVault

from cowork.managed import ManagedConnectionError


class LayeredDataVault:
    """The managed vault's entries first, then the scope's own.

    An (engine, name) the managed vault holds is read from it and hides a
    same-named entry of the scope's own. A save or delete of such an entry
    raises ManagedConnectionError, so it never reaches either vault; every
    other save and delete goes to the scope's own vault, a turn's included,
    so what the agent saves in a turn outlives the turn.
    """

    def __init__(self, *, managed: DataVault, own: DataVault) -> None:
        self._managed = managed
        self._own = own

    def is_managed(self, engine: str, name: str) -> bool:
        return self._managed.read_record(engine, name) is not None

    def _reader(self, engine: str, name: str) -> DataVault:
        return self._managed if self.is_managed(engine, name) else self._own

    def _refuse_managed(self, engine: str, name: str) -> None:
        if self.is_managed(engine, name):
            raise ManagedConnectionError(engine=engine, name=name)

    def save(
        self,
        engine: str,
        name: str,
        credentials: dict[str, str],
        *,
        secure_keys: list[str] | None = None,
    ) -> object:
        self._refuse_managed(engine, name)
        return self._own.save(engine, name, credentials, secure_keys=secure_keys)

    def load(self, engine: str, name: str) -> dict[str, str] | None:
        return self._reader(engine, name).load(engine, name)

    def read_record(self, engine: str, name: str) -> dict[str, Any] | None:
        return self._reader(engine, name).read_record(engine, name)

    def delete(self, engine: str, name: str) -> bool:
        self._refuse_managed(engine, name)
        return self._own.delete(engine, name)

    def list_connections(self) -> list[dict[str, str]]:
        managed = self._managed.list_connections()
        taken = {(entry["engine"], entry["name"]) for entry in managed}
        own = [
            entry
            for entry in self._own.list_connections()
            if (entry["engine"], entry["name"]) not in taken
        ]
        return [*managed, *own]

    def env_for(self, engine: str, name: str, *, flat: bool = False) -> dict[str, str] | None:
        return self._reader(engine, name).env_for(engine, name, flat=flat)

    def inject_env(self, engine: str, name: str, *, flat: bool = False) -> list[str] | None:
        return self._reader(engine, name).inject_env(engine, name, flat=flat)

    def clear_ds_env(self) -> None:
        self._own.clear_ds_env()
