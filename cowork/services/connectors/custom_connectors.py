"""Stored custom connector definitions: built from a handcrafted form, kept per scope."""
from __future__ import annotations

import re
from typing import Any
from urllib.parse import urlsplit

from pydantic import ValidationError

from cowork.db.scoped import ScopedSession
from cowork.models.custom_connector import CustomConnector
from cowork.schemas.connectors import ConnectorForm

# Top-level keys a handcrafted form may carry that ConnectorForm doesn't model
# but the connect form still renders.
_KEPT_FORM_KEYS = ("how_to", "help_url")
_COLOR_PATTERN = re.compile(r"#[0-9a-fA-F]{3}(?:[0-9a-fA-F]{3})?")
# Matches the cap agent-facing usage notes have on built-in specs.
USAGE_NOTES_MAX = 1600


class UnstorableFormError(ValueError):
    """A handcrafted form that cannot be kept as a custom connector."""


def _require_https(value: Any, what: str) -> None:
    if value is None:
        return
    parts = urlsplit(str(value))
    if parts.scheme != "https" or not parts.netloc:
        raise UnstorableFormError(f"{what} must be an https:// URL.")


def check_handcrafted_form(form_spec: dict[str, Any]) -> None:
    """Refuse a handcrafted form that would be unsafe or invalid to keep.

    The form is model-written and, once stored, is served on every later
    connect: its links open in the browser and its OAuth endpoints receive
    the user's client secret, so those must be https.

    Raises:
        UnstorableFormError: with a message the model can act on.
    """
    try:
        ConnectorForm.model_validate(form_spec)
    except ValidationError as e:
        raise UnstorableFormError(f"The connection form is not valid: {e.errors()[0]['msg']}") from e
    _require_https(form_spec.get("help_url"), "help_url")
    for m in form_spec.get("methods") or []:
        if not isinstance(m, dict):
            continue
        _require_https(m.get("help_url"), "A method's help_url")
        oauth = m.get("oauth") if isinstance(m.get("oauth"), dict) else {}
        for key in ("auth_url", "token_url", "revoke_url"):
            _require_https(oauth.get(key), f"oauth.{key}")
    color = form_spec.get("logo_color")
    if color is not None and not _COLOR_PATTERN.fullmatch(str(color)):
        raise UnstorableFormError("logo_color must be a hex color such as #3a7 or #33aa77.")
    meta = form_spec.get("connector") if isinstance(form_spec.get("connector"), dict) else {}
    if len(str(meta.get("usage_notes") or "")) > USAGE_NOTES_MAX:
        raise UnstorableFormError(f"connector.usage_notes must be at most {USAGE_NOTES_MAX} characters.")


def _text(value: Any, limit: int | None = None) -> str | None:
    """Return ``value`` as stripped text cut to ``limit``, or None when empty."""
    if value is None:
        return None
    text = str(value).strip()
    if limit is not None:
        text = text[:limit]
    return text or None


def stored_form(connector_id: str, form_spec: dict[str, Any]) -> dict[str, Any]:
    """Return the connection form to store for a handcrafted spec.

    Args:
        connector_id: The validated engine id the connector is saved under.
        form_spec: The handcrafted form as submitted.

    Returns:
        The form validated as ``ConnectorForm``, with a stable form id and the
        renderable top-level help keys. Chat bookkeeping is dropped.

    Raises:
        UnstorableFormError: when the form is invalid or unsafe to keep.
    """
    check_handcrafted_form(form_spec)
    form = ConnectorForm.model_validate(form_spec).model_dump(exclude_none=True)
    form["form_id"] = f"{connector_id}-connector"
    for key in _KEPT_FORM_KEYS:
        if isinstance(form_spec.get(key), str):
            form[key] = form_spec[key]
    return form


class CustomConnectorService:
    """Create, read, change and delete one scope's custom connector definitions."""

    def __init__(self, session: ScopedSession) -> None:
        """Bind the service to a tenant-scoped session.

        Args:
            session: The scope's session; org mode filters and stamps org_id.
        """
        self.session = session

    def get(self, connector_id: str) -> CustomConnector | None:
        """Return the scope's definition for ``connector_id``, or None."""
        stmt = self.session.select(CustomConnector).where(CustomConnector.connector_id == connector_id)
        # ScopedSession only filters in org mode; a local scope must not read
        # or change rows that belong to an org.
        if not self.session.scope.org_mode:
            stmt = stmt.where(CustomConnector.org_id.is_(None))
        return self.session.exec(stmt).first()

    def upsert_from_form(self, connector_id: str, form_spec: dict[str, Any]) -> CustomConnector:
        """Save a handcrafted form as the scope's definition for ``connector_id``.

        Args:
            connector_id: The engine id the connection was saved under.
            form_spec: The handcrafted form, with its optional ``connector``
                block (label, description, category, usage_notes).

        Returns:
            The stored row. An existing definition is updated in place; one an
            admin unfeatured stays unfeatured.

        Side effects:
            Commits the session.
        """
        meta = form_spec.get("connector") if isinstance(form_spec.get("connector"), dict) else {}
        values = {
            "label": _text(meta.get("label"), 255) or _text(form_spec.get("title"), 255) or connector_id,
            "description": _text(meta.get("description")) or _text(form_spec.get("subtitle")) or "",
            "category": _text(meta.get("category"), 64),
            "logo_color": _text(form_spec.get("logo_color"), 32),
            "usage_notes": _text(meta.get("usage_notes")),
            "spec": stored_form(connector_id, form_spec),
        }
        row = self.get(connector_id)
        if row is None:
            row = CustomConnector(connector_id=connector_id, **values)
        else:
            for key, value in values.items():
                setattr(row, key, value)
        self.session.add(row)
        self.session.commit()
        self.session.refresh(row)
        return row

    def update(self, connector_id: str, changes: dict[str, Any]) -> CustomConnector | None:
        """Change a definition's display fields, featured flag or form.

        Args:
            connector_id: The definition to change.
            changes: Any of ``label``, ``description``, ``category``,
                ``featured`` and ``spec`` (a full connection form).

        Returns:
            The updated row, or None when the scope has no such definition.

        Raises:
            UnstorableFormError: when ``spec`` is invalid or unsafe to keep.

        Side effects:
            Commits the session.
        """
        row = self.get(connector_id)
        if row is None:
            return None
        if "spec" in changes:
            row.spec = stored_form(connector_id, changes["spec"])
        for key in ("label", "description", "category", "featured"):
            if key in changes:
                setattr(row, key, changes[key])
        self.session.add(row)
        self.session.commit()
        self.session.refresh(row)
        return row

    def delete(self, connector_id: str) -> bool:
        """Delete a definition. Saved connections that use it are kept.

        Returns:
            True when a definition was deleted, False when there was none.

        Side effects:
            Commits the session.
        """
        row = self.get(connector_id)
        if row is None:
            return False
        self.session.delete(row)
        self.session.commit()
        return True
