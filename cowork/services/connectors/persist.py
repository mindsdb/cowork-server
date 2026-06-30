"""Single place that writes a connection to the vault.

Shared by every save path (the probe/form flow and the OAuth flows) so they all
get the same behavior: an identity-derived readable slug (with random fallback),
modify-flow sentinel resolution, explicit ``secure_keys``, and a non-destructive
save that never overwrites a different account.
"""
from __future__ import annotations

import uuid
from pathlib import Path

from cowork.common.settings.app_settings import ConnectorSettings
from cowork.services.connectors.identity import (
    derive_connection_name,
    resolve_keep_sentinels,
    resolve_unique_slug,
    secure_keys_for,
)


def persist_connection(
    connector_id: str,
    method: str | None,
    name: str,
    credentials: dict,
    *,
    vault=None,
) -> str:
    """Persist a connection and return the slug used.

    ``name`` (explicit) wins; otherwise the connector's identity-derived slug
    (e.g. gmail → ``user-gmail-com``); otherwise a random fallback. An edit
    (a save carrying keep-sentinels) updates the named record in place; every
    other save is non-destructive — a different account gets a ``-N`` suffix.
    """
    if vault is None:
        from anton.core.datasources.data_vault import LocalDataVault

        vault = LocalDataVault(Path(ConnectorSettings().vault_dir))

    base_slug = (
        (name or "").strip()
        or derive_connection_name(connector_id, method, credentials)
        or f"{connector_id}-{uuid.uuid4().hex[:6]}"
    )
    # Resolve modify-flow "keep" sentinels against the record being updated, so
    # an unchanged secret keeps its stored value instead of persisting the
    # literal sentinel.
    target = vault.read_record(connector_id, base_slug)
    credentials, is_edit = resolve_keep_sentinels(credentials, target)
    payload = {**credentials, "_connector_id": connector_id}
    if method:
        payload["_method"] = method
    secure_keys = secure_keys_for(connector_id, method, payload)
    if is_edit:
        slug = base_slug  # an edit targets the named connection — update in place
    else:
        slug = resolve_unique_slug(vault, connector_id, base_slug, payload, secure_keys)
    vault.save(connector_id, slug, payload, secure_keys=secure_keys)
    return slug
