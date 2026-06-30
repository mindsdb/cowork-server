"""Connection identity helpers.

A saved connection's *name* (slug) and its *secret classification* are derived
here so both probe save-paths agree:

- ``derive_connection_name`` turns a connector's natural identity field(s) — the
  spec's ``name_from`` (e.g. gmail → ``email``) — into a readable, stable slug,
  so a connection is ``user-gmail-com`` instead of a random
  ``gmail-548bdb``. Same identity → same slug → re-connecting updates in place
  (dedup) instead of leaving a stale duplicate. Returns ``None`` when the
  connector declares no ``name_from`` or the field wasn't provided, so callers
  fall back to the random slug.

- ``spec_secret_fields`` reads the per-field ``secret`` flags the connector spec
  already declares, so the saved record can carry an explicit ``secure_keys``
  list (the email stays readable; the password is masked) rather than relying on
  the name-matching heuristic at read time.
"""
from __future__ import annotations

import re

from anton.core.datasources.data_vault import is_secret_key

from cowork.services.connectors.specs._registry import registry


def _spec_name_from(connector_id: str, method: str | None):
    """Resolve the ``name_from`` declaration for a connector/method.

    Method-level ``name_from`` wins (the identity field can differ per auth
    method — gmail's app-password method identifies by ``email``, its
    service-account method by ``impersonate_email``); falls back to a
    form-level or top-level declaration.
    """
    raw = registry.get_connectors().get(connector_id)
    if not raw:
        return None
    form = raw.get("form") or {}
    if method:
        for m in form.get("methods", []) or []:
            if m.get("id") == method and m.get("name_from"):
                return m["name_from"]
    return form.get("name_from") or raw.get("name_from")


def _identity_fields(connector_id: str, method: str | None, credentials: dict) -> list[str]:
    """Field name(s) to build the connection slug from.

    1. The connector's explicit ``name_from`` (curated, authoritative).
    2. Otherwise a *narrow* heuristic limited to **credential-unique** fields:
       - ``email`` (one address = one account), or
       - ``host`` (+ ``database`` + ``username`` when present) for databases.

    Deliberately NOT included: ``project_id`` / ``tenant_id`` / ``subdomain`` /
    ``account_id`` (identify a tenant/project, not the specific credential — two
    accounts can share them) and ``base_url`` / ``client_id`` / config fields
    (constant or opaque). Those stay on the random fallback until curated, so the
    auto-derived slug can't silently collapse two distinct accounts.
    """
    name_from = _spec_name_from(connector_id, method)
    if name_from:
        return [name_from] if isinstance(name_from, str) else list(name_from)
    if str(credentials.get("email", "")).strip():
        return ["email"]
    if str(credentials.get("host", "")).strip():
        return [f for f in ("host", "database", "username") if str(credentials.get(f, "")).strip()]
    return []


def derive_connection_name(
    connector_id: str, method: str | None, credentials: dict
) -> str | None:
    """Readable, stable slug from the connector's identity field(s), or None.

    Uses the connector's ``name_from`` if declared, else a narrow
    credential-unique heuristic (see ``_identity_fields``). Returns ``None`` when
    no identity field applies or its value is absent — the caller then keeps its
    random fallback.
    """
    fields = _identity_fields(connector_id, method, credentials)
    parts = [
        str(credentials.get(f, "")).strip()
        for f in fields
        if str(credentials.get(f, "")).strip()
    ]
    if not parts:
        return None
    slug = re.sub(r"[^\w]+", "-", "-".join(parts)).strip("-").lower()
    return slug or None


def spec_secret_fields(connector_id: str, method: str | None) -> list[str]:
    """Field names the connector spec marks ``secret: true`` (for the method)."""
    raw = registry.get_connectors().get(connector_id) or {}
    form = raw.get("form") or {}
    secret: set[str] = set()
    for m in form.get("methods", []) or []:
        if method and m.get("id") != method:
            continue
        for f in m.get("fields", []) or []:
            if f.get("secret") and f.get("name"):
                secret.add(f["name"])
    for f in form.get("fields", []) or []:
        if f.get("secret") and f.get("name"):
            secret.add(f["name"])
    return sorted(secret)


def secure_keys_for(connector_id: str, method: str | None, fields: dict) -> list[str]:
    """The ``secure_keys`` to persist: spec-marked secrets ∪ name-heuristic.

    Union so we never *under*-mask: an explicit spec flag classifies a field
    whose name the heuristic would miss, and the heuristic catches any extra
    secret-shaped field the spec didn't mark.
    """
    spec_secrets = set(spec_secret_fields(connector_id, method))
    return sorted(
        {k for k in fields if k in spec_secrets or is_secret_key(k, None)}
    )


def _nonsecret_identity(fields: dict, secure_keys: list[str]) -> dict:
    """Non-secret, non-bookkeeping fields — the part that identifies an account.

    Drops secrets (so a rotated password still reads as the same account) and
    ``_``-prefixed meta (``_connector_id`` / ``_method``).
    """
    secure = set(secure_keys or [])
    return {
        k: v
        for k, v in (fields or {}).items()
        if not k.startswith("_")
        and k not in secure
        and not is_secret_key(k, secure_keys)
    }


def is_same_account(existing_record: dict | None, payload: dict, secure_keys: list[str]) -> bool:
    """True when ``payload`` is the *same account* as an existing record.

    Compares only the non-secret identity fields, so a rotated secret / edited
    connection still matches (update in place) while a genuinely different
    account does not.

    Caveat: two accounts that share **every** non-secret field and differ only
    in the secret are indistinguishable from a rotation here. Only slugs derived
    from a credential-unique field (e.g. ``email``, or ``host``+``database``+
    ``username``) are fully collision-proof — which is why auto-derivation is
    kept to those fields and everything else falls back to a random slug.
    """
    if not existing_record:
        return False
    return _nonsecret_identity(existing_record.get("fields", {}), secure_keys) == \
        _nonsecret_identity(payload, secure_keys)


def resolve_unique_slug(
    vault, engine: str, base_slug: str, payload: dict, secure_keys: list[str]
) -> str:
    """Slug to save under, without ever overwriting a *different* account.

    Reuses ``base_slug`` when it's free or already holds the same account
    (update in place); otherwise returns the next free ``base_slug-N`` — so
    naming two distinct accounts the same thing (or a non-unique derived slug)
    yields ``support`` / ``support-2`` instead of silently clobbering the first.
    """
    rec = vault.read_record(engine, base_slug)
    if rec is None or is_same_account(rec, payload, secure_keys):
        return base_slug
    n = 2
    while True:
        candidate = f"{base_slug}-{n}"
        rec = vault.read_record(engine, candidate)
        if rec is None or is_same_account(rec, payload, secure_keys):
            return candidate
        n += 1
