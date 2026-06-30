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


def derive_connection_name(
    connector_id: str, method: str | None, credentials: dict
) -> str | None:
    """Readable, stable slug from the connector's identity field(s), or None.

    Returns ``None`` when the connector declares no ``name_from`` or none of the
    identity fields were supplied — the caller then keeps its random fallback.
    """
    name_from = _spec_name_from(connector_id, method)
    if not name_from:
        return None
    fields = [name_from] if isinstance(name_from, str) else list(name_from)
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
