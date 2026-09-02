"""Single place that writes a connection to the vault.

Shared by every save path (the probe/form flow and the OAuth flows) so they all
get the same behavior: an identity-derived readable slug (with random fallback),
modify-flow sentinel resolution, explicit ``secure_keys``, and a non-destructive
save that never overwrites a different account.
"""
from __future__ import annotations

import uuid
from pathlib import Path
from typing import TYPE_CHECKING

from anton.utils.datasources import default_user_label, ensure_unique_user_label

from cowork.common.settings.app_settings import ConnectorSettings
from cowork.services.connectors.identity import (
    derive_connection_name,
    resolve_keep_sentinels,
    resolve_unique_slug,
    secure_keys_for,
)
from cowork.services.connectors.vault_lock import lock_for

if TYPE_CHECKING:
    from cowork.db.scoped import TenantScope


def vault_for_scope(scope: "TenantScope | None" = None):
    from anton.core.datasources.data_vault import LocalDataVault
    from cowork.db.scoped import scoped_storage_root

    # Org mode keys the persisted vault per org, the same as SkillService and
    # FileService. Desktop passes no scope and gets vault_dir unchanged; on an
    # org deployment a missing scope RAISES rather than falling back, because
    # the fallback path is the shared namespace root and saved credentials
    # must never land there.
    return LocalDataVault(scoped_storage_root(Path(ConnectorSettings().vault_dir), scope, store="data-vault"))


def persist_connection(
    connector_id: str,
    method: str | None,
    name: str,
    credentials: dict,
    *,
    label: str | None = None,
    user_label: str | None = None,
    default_label: str | None = None,
    vault=None,
    scope: "TenantScope | None" = None,
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

    ``user_label`` — passed explicitly or as a ``user_label`` / ``_user_label``
    field in ``credentials`` — is the newer, globally-unique, de-duplicated
    replacement for ``label``; stored as ``_user_label`` and carried forward
    the same way. A brand-new connection that doesn't set one gets a computed
    default: ``default_label`` if the caller supplied one (e.g. an OAuth
    connector's fetched account/org/workspace name), else the engine id.

    ``default_label`` only ever applies to a genuinely new connection (see the
    ``existing is None`` branch below) — unlike ``user_label``, it can never
    overwrite a name the user already set on a reconnect/re-save.
    """
    if vault is None:
        vault = vault_for_scope(scope)

    cred = dict(credentials)
    label = str(
        label or cred.pop("label", "") or cred.pop("_label", "") or ""
    ).strip()
    user_label = str(
        user_label or cred.pop("user_label", "") or cred.pop("_user_label", "") or ""
    ).strip()

    base_slug = (
        (name or "").strip()
        or derive_connection_name(connector_id, method, cred)
        or f"{connector_id}-{uuid.uuid4().hex[:8]}"
    )
    # Locked for the same reason ConnectionsService's merge_picked_files/
    # remove_picked_file/patch_token are: this is a read-modify-write against
    # the vault record at (connector_id, base_slug) (the edit case — a brand
    # new connection can't race anything since nothing else references it
    # yet), and an unlocked interleaving with one of those would silently
    # revert whichever side saves first.
    with lock_for(connector_id, base_slug):
        # Resolve modify-flow "keep" sentinels against the record being
        # updated, so an unchanged secret keeps its stored value instead of
        # persisting the literal sentinel.
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
        # Same reasoning applies to _picked_files (Google Picker grants) below — a
        # full save here must not silently revoke files the user already granted
        # access to.
        existing = target if slug == base_slug else vault.read_record(connector_id, slug)
        if not label:
            label = str((existing or {}).get("fields", {}).get("_label", "")).strip()
        if label:
            payload["_label"] = label
        if not user_label:
            user_label = str((existing or {}).get("fields", {}).get("_user_label", "")).strip()
        if not user_label and existing is None:
            # Genuinely new connection — nothing existed at this slug before
            # this save, nothing explicit was passed, nothing to carry
            # forward. Compute the same default anton's CLI prompt would show
            # (engine id, de-duplicated); otherwise a connection created via
            # cowork without an explicit label ends up with none at all,
            # while every anton-created connection always gets one.
            #
            # Checked against `existing is None`, not `is_edit` — `is_edit`
            # is only true when the request carried a GUI modify-flow
            # keep-sentinel; a same-account re-save that reaches this
            # function some other way (no sentinel) still resolves to the
            # pre-existing record via `resolve_unique_slug()`'s
            # `is_same_account()` check, and `existing` correctly reflects
            # that (non-None) even though `is_edit` would be False.
            user_label = str(default_label or "").strip() or default_user_label(vault, connector_id)
        if user_label:
            payload["_user_label"] = ensure_unique_user_label(
                vault, user_label, exclude=(connector_id, slug)
            )
        existing_picked_files = (existing or {}).get("fields", {}).get("_picked_files")
        if existing_picked_files:
            payload.setdefault("_picked_files", existing_picked_files)
        vault.save(connector_id, slug, payload, secure_keys=secure_keys)
        return slug


def set_connection_label(engine: str, name: str, label: str, *, vault=None,
                         scope: "TenantScope | None" = None) -> str | None:
    """Set the human label on an existing connection in place. Returns the
    stored value (post-deduplication — may differ from the requested `label`),
    or None if the connection doesn't exist or `label` is blank. Used by the
    agent's learn-and-persist flow (e.g. after the user confirms which address
    is "Support") — updates only the non-secret ``_user_label`` and leaves the
    identity/slug and secrets untouched.
    """
    clean_label = str(label or "").strip()
    if not clean_label:
        return None
    if vault is None:
        vault = vault_for_scope(scope)
    with lock_for(engine, name):
        record = vault.read_record(engine, name)
        if record is None:
            return None
        fields = dict(record.get("fields") or {})
        stored = ensure_unique_user_label(vault, clean_label, exclude=(engine, name))
        fields["_user_label"] = stored
        vault.save(engine, name, fields, secure_keys=record.get("secure_keys"))
        return stored
