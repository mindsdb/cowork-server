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


def _default_vault():
    from anton.core.datasources.data_vault import LocalDataVault

    return LocalDataVault(Path(ConnectorSettings().vault_dir))


def persist_connection(
    connector_id: str,
    method: str | None,
    name: str,
    credentials: dict,
    *,
    label: str | None = None,
    vault=None,
) -> str:
    """Persist a connection and return the slug used.

    ``name`` (explicit) wins; otherwise the connector's identity-derived slug
    (e.g. gmail → ``user-gmail-com``); otherwise a random fallback. An edit
    (a save carrying keep-sentinels) updates the named record in place; every
    other save is non-destructive — a different account gets a ``-N`` suffix.

    A human ``label`` ("Support", "Personal") — passed explicitly or as a
    ``label`` / ``_label`` field in ``credentials`` — is stored as the non-secret
    ``_label`` so it can name the connection without changing its identity/slug.
    An existing label is preserved when a later save doesn't set one.
    """
    if vault is None:
        vault = _default_vault()

    cred = dict(credentials)
    label = str(
        label or cred.pop("label", "") or cred.pop("_label", "") or ""
    ).strip()

    base_slug = (
        (name or "").strip()
        or derive_connection_name(connector_id, method, cred)
        or f"{connector_id}-{uuid.uuid4().hex[:6]}"
    )
    # Resolve modify-flow "keep" sentinels against the record being updated, so
    # an unchanged secret keeps its stored value instead of persisting the
    # literal sentinel.
    target = vault.read_record(connector_id, base_slug)
    cred, is_edit = resolve_keep_sentinels(cred, target)
    payload = {**cred, "_connector_id": connector_id}
    if method:
        payload["_method"] = method
    secure_keys = secure_keys_for(connector_id, method, payload)
    if is_edit:
        slug = base_slug  # an edit targets the named connection — update in place
    else:
        slug = resolve_unique_slug(vault, connector_id, base_slug, payload, secure_keys)
    # Carry an existing label forward when this save didn't set one (a full save
    # overwrites the record), so editing other fields doesn't drop the label.
    if not label:
        existing = target if slug == base_slug else vault.read_record(connector_id, slug)
        label = str((existing or {}).get("fields", {}).get("_label", "")).strip()
    if label:
        payload["_label"] = label
    vault.save(connector_id, slug, payload, secure_keys=secure_keys)
    return slug


def set_connection_label(engine: str, name: str, label: str, *, vault=None) -> bool:
    """Set the human label on an existing connection in place. Returns False if
    the connection doesn't exist. Used by the agent's learn-and-persist flow
    (e.g. after the user confirms which address is "Support") — updates only the
    non-secret ``_label`` and leaves the identity/slug and secrets untouched.
    """
    if vault is None:
        vault = _default_vault()
    record = vault.read_record(engine, name)
    if record is None:
        return False
    fields = dict(record.get("fields") or {})
    fields["_label"] = str(label or "").strip()
    vault.save(engine, name, fields, secure_keys=record.get("secure_keys"))
    return True
