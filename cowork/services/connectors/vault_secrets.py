"""Registers a tenant's saved datasource secrets for history scrubbing.

`scrub_credentials` redacts a free-form secret, such as a DSN password, only by
its registered value. History replayed before a chat session exists (the gate,
the remote seed) needs that registration done at request entry.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from anton.utils.datasources import begin_ds_turn_scope, restore_namespaced_env

from cowork.common.logger import get_logger
from cowork.services.connectors.persist import vault_for_scope

if TYPE_CHECKING:
    from cowork.db.scoped import TenantScope

logger = get_logger(__name__)


def register_vault_secrets(scope: TenantScope | None) -> None:
    """Register this request's DS_* secrets so `scrub_credentials` redacts
    them by exact value, not only by API-key shape.

    Call at request entry, before any history is scrubbed. The scope it opens
    is this request's own and reaches every task spawned from it. A failure
    must not fail the turn: scrubbing falls back to the shape-based regex.
    """
    begin_ds_turn_scope()
    try:
        restore_namespaced_env(vault_for_scope(scope))
    except Exception:
        logger.warning(
            "Could not register vault secrets; this turn's history is scrubbed "
            "by key shape only",
            exc_info=True,
        )
