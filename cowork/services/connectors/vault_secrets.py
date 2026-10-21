"""Registers a tenant's saved datasource secrets for history scrubbing.

`scrub_credentials` redacts a free-form secret, such as a DSN password, only by
its registered value. History replayed before a chat session exists (the gate,
the remote seed) needs that registration done at request entry.
"""
from __future__ import annotations

import asyncio
import contextvars
import re
from collections.abc import Callable
from functools import partial
from typing import TYPE_CHECKING, Any

from anton.core.datasources.data_vault import is_secret_key
from anton.utils.datasources import begin_ds_turn_scope, restore_namespaced_env, scrub_credentials

from cowork.common.logger import get_logger
from cowork.services.connectors.persist import vault_for_scope

if TYPE_CHECKING:
    from anton.core.datasources.data_vault import DataVault

    from cowork.db.scoped import TenantScope

logger = get_logger(__name__)


def _hidden_secret_values(vault: DataVault, engine: str, name: str) -> dict[str, str]:
    """Return the secure `_` fields of one connection, keyed by a DS_* style label.

    Every vault's ``env_for`` leaves `_` fields out, so these are the secrets the
    rebuild would otherwise never see. The label only names the scrub marker;
    it follows anton's ``DS_<ENGINE>_<NAME>__<FIELD>`` env naming.
    """
    if not hasattr(vault, "read_record"):
        logger.debug("Vault has no read_record; hidden secrets not registered for %s/%s", engine, name)
        return {}
    record = vault.read_record(engine, name)
    if not record:
        return {}
    secure_keys = record.get("secure_keys")
    prefix = "DS_" + re.sub(r"[^\w]", "_", f"{engine}-{name}").upper()
    return {
        f"{prefix}__{key.upper()}": value
        for key, value in (record.get("fields") or {}).items()
        if key.startswith("_") and isinstance(value, str) and value and is_secret_key(key, secure_keys)
    }


class _ScrubRegistrationVault:
    """A vault as the scrub registration reads it: ``env_for`` also lists each
    secure `_` field, so ``scrub_credentials`` redacts its value.

    Only ever handed to ``restore_namespaced_env``, whose value map feeds the
    scrubber alone. A pad builds its environment from the real vault, so a
    hidden secret still never reaches the agent's DS_* environment.
    """

    def __init__(self, vault: DataVault) -> None:
        """Wrap the real vault whose records the scrub registration reads."""
        self._vault = vault

    def __getattr__(self, attr: str) -> Any:
        """Delegate every other vault attribute to the wrapped vault."""
        return getattr(self._vault, attr)

    def env_for(self, engine: str, name: str, *, flat: bool = False) -> dict[str, str] | None:
        """The real vault's env for a connection plus its hidden secrets, or
        None when the connection does not exist."""
        env = self._vault.env_for(engine, name, flat=flat)
        if env is None:
            return None
        return {**env, **_hidden_secret_values(self._vault, engine, name)}


_KEPT_ENGINE = "connection-check"
_KEPT_NAME = "kept"


class _KeptSecretsVault:
    """The stored secrets one connection check kept, as an in-memory vault's
    only connection, every field marked secure.

    Only ever handed to ``restore_namespaced_env``, so the scrubber learns the
    values; nothing builds a pad environment from it.
    """

    def __init__(self, secrets: tuple[str, ...]) -> None:
        """Hold each secret as one secure field."""
        self._fields = {f"kept_{index}": value for index, value in enumerate(secrets)}

    def _is_kept(self, engine: str, name: str) -> bool:
        """Whether ``engine``/``name`` names this vault's one connection."""
        return (engine, name) == (_KEPT_ENGINE, _KEPT_NAME)

    def list_connections(self) -> list[dict[str, str]]:
        """The one connection, in the vault listing's shape."""
        return [{"engine": _KEPT_ENGINE, "name": _KEPT_NAME, "created_at": ""}]

    def read_record(self, engine: str, name: str) -> dict[str, Any] | None:
        """The connection's record with every field secure, or None for any other name."""
        if not self._is_kept(engine, name):
            return None
        return {"fields": dict(self._fields), "secure_keys": list(self._fields)}

    def load(self, engine: str, name: str) -> dict[str, str] | None:
        """The connection's fields, or None for any other name."""
        return dict(self._fields) if self._is_kept(engine, name) else None

    def env_for(self, engine: str, name: str, *, flat: bool = False) -> dict[str, str] | None:
        """The connection's DS_* names and values, named as anton names them, or None."""
        if not self._is_kept(engine, name):
            return None
        prefix = "DS_" + re.sub(r"[^\w]", "_", f"{engine}-{name}").upper()
        return {f"{prefix}__{key.upper()}": value for key, value in self._fields.items()}


def _register_kept_secrets(secrets: tuple[str, ...]) -> None:
    """Open a DS_* scope in the current context and register ``secrets`` in it."""
    begin_ds_turn_scope()
    restore_namespaced_env(_KeptSecretsVault(secrets))


def kept_secret_scrubber(secrets: tuple[str, ...]) -> Callable[[str], str]:
    """A function that hides ``secrets`` in a text the way ``scrub_credentials`` does.

    The secrets are registered in a context of their own, so every printed or
    encoded form anton matches is hidden, and neither the caller's scope nor
    a scope a later turn opens is touched. Each call scrubs in that context.
    Raises whatever the registration raises; it never logs a value.
    """
    context = contextvars.Context()
    context.run(_register_kept_secrets, secrets)
    return partial(context.run, scrub_credentials)


def _restore_from_vault(scope: TenantScope | None) -> None:
    vault = vault_for_scope(scope)
    # Nothing to register, and the rebuild re-parses anton's datasource
    # registry every time, which is most of its per-turn cost.
    if not vault.list_connections():
        return
    restore_namespaced_env(_ScrubRegistrationVault(vault))


async def register_vault_secrets(scope: TenantScope | None) -> None:
    """Register this request's DS_* secrets so `scrub_credentials` redacts
    them by exact value, not only by API-key shape.

    Call at request entry, before any history is scrubbed. The scope it opens
    is this request's own and reaches every task spawned from it. A failure
    must not fail the turn: scrubbing falls back to the shape-based regex.
    """
    # The scope opens here, in the request's context, and the vault reads and
    # rebuild run in a worker thread, off the event loop. The thread gets a
    # copy of the context, which holds the same containers, and anton fills
    # them in place, so the rebuild lands in this request's scope.
    begin_ds_turn_scope()
    try:
        await asyncio.to_thread(_restore_from_vault, scope)
    except Exception:
        logger.warning(
            "Could not register vault secrets; this turn's history is scrubbed "
            "by key shape only",
            exc_info=True,
        )
