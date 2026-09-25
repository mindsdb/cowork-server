"""Registers a tenant's saved datasource secrets for history scrubbing.

`scrub_credentials` redacts a free-form secret, such as a DSN password, only by
its registered value. History replayed before a chat session exists (the gate,
the remote seed) needs that registration done at request entry.
"""
from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from anton.utils.datasources import begin_ds_turn_scope, restore_namespaced_env

from cowork.common.logger import get_logger
from cowork.services.connectors.persist import vault_for_scope

if TYPE_CHECKING:
    from cowork.db.scoped import TenantScope

logger = get_logger(__name__)


def _restore_from_vault(scope: TenantScope | None) -> None:
    vault = vault_for_scope(scope)
    # Nothing to register, and the rebuild re-parses anton's datasource
    # registry every time, which is most of its per-turn cost.
    if not vault.list_connections():
        return
    restore_namespaced_env(vault)


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
