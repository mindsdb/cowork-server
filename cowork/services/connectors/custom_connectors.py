"""Stored custom connector definitions: built from a handcrafted form, kept per scope."""
from __future__ import annotations

from typing import Any

from cowork.db.scoped import ScopedSession
from cowork.models.custom_connector import CustomConnector
from cowork.schemas.connectors import ConnectorForm

# Top-level keys a handcrafted form may carry that ConnectorForm doesn't model
# but the connect form still renders.
_KEPT_FORM_KEYS = ("how_to", "help_url")


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
        pydantic.ValidationError: when the form is not a valid connection form.
    """
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
            pydantic.ValidationError: when ``spec`` is not a valid connection form.

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
