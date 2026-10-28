"""Stored custom connector definitions: built from a handcrafted form, kept per scope."""
from __future__ import annotations

import re
from typing import Any
from urllib.parse import urlsplit

from anton.utils.datasources import scrub_output
from pydantic import ValidationError
from sqlalchemy import false

from cowork.db.personal_resources import PersonalResource
from cowork.db.scoped import ScopedSelect, ScopedSession, TenantScope, tenant_scope_required
from cowork.models.custom_connector import CustomConnector
from cowork.schemas.connectors import ConnectorForm, CustomConnectorUpdate

# Top-level keys a handcrafted form may carry that ConnectorForm doesn't model
# but the connect form still renders.
_KEPT_FORM_KEYS = ("how_to", "help_url")
_COLOR_PATTERN = re.compile(r"#[0-9a-fA-F]{3}(?:[0-9a-fA-F]{3})?")
# Matches the cap agent-facing usage notes have on built-in specs.
USAGE_NOTES_MAX = 1600


class UnstorableFormError(ValueError):
    """A handcrafted form that cannot be kept as a custom connector."""


def handcrafted_form_allowed(*, known: bool, custom: bool, extends: bool) -> bool:
    """Whether a connector id may take a handcrafted form instead of its stored one.

    Args:
        known: The id is a built-in or a custom connector in the scope.
        custom: It is a custom connector.
        extends: The form is a follow-up step that adds fields to a saved
            connection.

    Returns:
        True for an unknown id, and for a follow-up step of a custom
        connector. A known id is otherwise checked and saved against its
        stored form, so a handcrafted one would skip that check.
    """
    return not known or (custom and extends)


def _require_https(value: Any, what: str) -> None:
    """Raise UnstorableFormError unless ``value`` is None or an https:// URL
    with a host; a value urlsplit cannot parse is refused the same way."""
    if value is None:
        return
    try:
        parts = urlsplit(str(value))
    except ValueError as e:
        raise UnstorableFormError(f"{what} must be an https:// URL.") from e
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
    notes = str(meta.get("usage_notes") or "")
    if len(notes) > USAGE_NOTES_MAX:
        raise UnstorableFormError(f"connector.usage_notes must be at most {USAGE_NOTES_MAX} characters.")
    # Hygiene for the builder's own prompt: anton's output scrubber knows the key
    # and token shapes. No one else reads a definition (scope_is_one_persons).
    if scrub_output(notes) != notes:
        raise UnstorableFormError(
            "connector.usage_notes looks like it holds a credential value. "
            "Say how to authenticate (header name, scheme) without the value."
        )


def scope_is_one_persons(scope: TenantScope) -> bool:
    """Whether ``scope``'s custom connector definitions belong to one person.

    True on a single-user install, and in a personal scope whose connections
    are personal. False in org mode, where an organization shares them, in a
    personal scope with shared connections, and in the operator view of a
    deployment that trusts identity headers.

    Everything in a definition (label, description, help text, the form,
    usage notes) is model-written and may hold what its builder typed, in
    any encoding. Custom connectors are therefore built only where this is
    True, and served outside org mode only there, so none reaches another
    person. Sharing them across people (org mode included) is not built yet.
    """
    if scope.personal:
        return scope.partitioned_for(PersonalResource.CONNECTIONS)
    return not scope.org_mode and not tenant_scope_required()


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


def scope_definitions(session: ScopedSession) -> ScopedSelect:
    """Select the custom connector definitions ``session``'s scope sees.

    Where connections are partitioned (org mode, or a personal scope whose
    connections are personal), the scoped session already keys rows by the
    scope's org_id. Elsewhere the session adds no filter, so only the shared
    rows (org_id NULL) are the scope's; another scope's rows are left out.
    Outside org mode, a scope that is not one person's sees none
    (scope_is_one_persons), rows written before that rule included.
    """
    scope = session.scope
    stmt = session.select(CustomConnector)
    if not scope.org_mode and not scope_is_one_persons(scope):
        return stmt.where(false())
    if not scope.partitioned_for(PersonalResource.CONNECTIONS):
        stmt = stmt.where(CustomConnector.org_id.is_(None))
    return stmt


class CustomConnectorService:
    """Create, read, change and delete one scope's custom connector definitions."""

    def __init__(self, session: ScopedSession) -> None:
        """Bind the service to a tenant-scoped session.

        Args:
            session: The scope's session; org mode filters and stamps org_id.
        """
        self.session = session

    def _kept_notes(self, notes: Any) -> str | None:
        """``notes`` as stored: the text in one person's scope, else None.
        In org mode a definition is served to every member, and its notes
        were typed by one of them."""
        return _text(notes) if scope_is_one_persons(self.session.scope) else None

    def get(self, connector_id: str) -> CustomConnector | None:
        """Return the scope's definition for ``connector_id``, or None."""
        stmt = scope_definitions(self.session).where(CustomConnector.connector_id == connector_id)
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

        Raises:
            UnstorableFormError: when the form is invalid or unsafe to keep,
                or the scope is shared by several people, org mode included
                (scope_is_one_persons), where a definition would reach them all.

        Side effects:
            Commits the session.
        """
        scope = self.session.scope
        if not scope_is_one_persons(scope):
            raise UnstorableFormError(
                "custom connectors are kept only for one person's own connections on this deployment."
            )
        meta = form_spec.get("connector") if isinstance(form_spec.get("connector"), dict) else {}
        values = {
            "label": _text(meta.get("label"), 255) or _text(form_spec.get("title"), 255) or connector_id,
            "description": _text(meta.get("description")) or _text(form_spec.get("subtitle")) or "",
            "category": _text(meta.get("category"), 64),
            "logo_color": _text(form_spec.get("logo_color"), 32),
            "usage_notes": self._kept_notes(meta.get("usage_notes")),
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

    def update(self, connector_id: str, changes: CustomConnectorUpdate) -> CustomConnector | None:
        """Change a definition's display fields, featured flag or form.

        Args:
            connector_id: The definition to change.
            changes: The fields the caller set. A null ``category`` or
                ``description`` clears it; a null ``label``, ``featured`` or
                ``spec`` leaves it as it is, since none of them can be empty.
                A ``spec`` with a ``connector`` block also sets the usage
                notes, and any of label, description and category not sent
                on their own, as the first save does.

        Returns:
            The updated row, or None when the scope has no such definition.

        Raises:
            UnstorableFormError: when ``spec`` is invalid or unsafe to keep.

        Side effects:
            Commits the session.
        """
        sent = changes.model_fields_set
        row = self.get(connector_id)
        if row is None:
            return None
        if changes.spec is not None:
            row.spec = stored_form(connector_id, changes.spec)
            # The directory tile reads the column, so a new form color moves it too.
            if changes.spec.get("logo_color"):
                row.logo_color = _text(changes.spec["logo_color"], 32)
            meta = changes.spec.get("connector")
            if isinstance(meta, dict):
                if "usage_notes" in meta:
                    row.usage_notes = self._kept_notes(meta["usage_notes"])
                for key, limit in (("label", 255), ("description", None), ("category", 64)):
                    if key not in sent and _text(meta.get(key), limit):
                        setattr(row, key, _text(meta[key], limit))
        if changes.label is not None:
            row.label = changes.label
        if changes.featured is not None:
            row.featured = changes.featured
        if "description" in sent:
            row.description = changes.description or ""
        if "category" in sent:
            row.category = changes.category
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
