"""Stored custom connector definitions: built from a handcrafted form, kept per scope."""
from __future__ import annotations

import re
from typing import Any
from urllib.parse import urlsplit

from anton.utils.datasources import scrub_output
import httpx
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


#: What a member reads when a custom connector was saved before connection tests.
NEEDS_REBUILD_FOR_MEMBER = (
    "This connector was saved before connection tests existed. Ask the agent to build it again."
)
#: What the agent reads instead of a form it cannot use.
NEEDS_REBUILD_FOR_AGENT = (
    "{connector_id!r} was saved before connection tests existed and cannot be connected as it is. "
    "Build a handcrafted form for it again, with engine {connector_id!r} and a test_request; it "
    "replaces the old definition once its test passes."
)


class UnstorableFormError(ValueError):
    """A handcrafted form that cannot be kept as a custom connector."""


def handcrafted_form_allowed(
    *, known: bool, custom: bool, extends: bool, stored_test_request: bool | None,
) -> bool:
    """Whether a connector id may take a handcrafted form instead of its stored one.

    Args:
        known: The id is a built-in or a custom connector in the scope.
        custom: It is a custom connector.
        extends: The form is a follow-up step that adds fields to a saved
            connection.
        stored_test_request: Whether the stored custom definition carries a
            test request; None when the id is not a stored custom connector.
            Required, so no caller can leave the stored definition out.

    Returns:
        True for an unknown id, for a follow-up step of a custom connector,
        and, on a single-user install, for a custom connector saved before
        connection tests existed, which can only be connected again once a
        new form replaces it. A known id is otherwise checked and saved
        against its stored form, so a handcrafted one would skip that check.
    """
    if not known or (custom and extends):
        return True
    return custom and stored_test_request is False and not tenant_scope_required()


def test_request_for(form_spec: dict[str, Any], method: str | None) -> dict[str, Any] | None:
    """The test request a submission of ``method`` runs: the form's own on a
    form with no methods, else that method's; None when there is none."""
    methods = [m for m in form_spec.get("methods") or [] if isinstance(m, dict)]
    if not methods:
        request = form_spec.get("test_request")
    else:
        chosen = next((m for m in methods if m.get("id") == method), None)
        request = (chosen or {}).get("test_request")
    return request if isinstance(request, dict) else None


def has_test_requests(form_spec: dict[str, Any]) -> bool:
    """Whether every method of ``form_spec`` carries a test request."""
    methods = [m for m in form_spec.get("methods") or [] if isinstance(m, dict)]
    holders = methods or [form_spec]
    return all(isinstance(h.get("test_request"), dict) for h in holders)


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
    # Self-hosted and org mode keep the model probe until their own milestone.
    if not tenant_scope_required():
        check_test_requests(form_spec)


#: A `{{name}}` placeholder in a test request; the runner fills the same pattern.
PLACEHOLDER = re.compile(r"\{\{\s*([A-Za-z_][A-Za-z0-9_]*)\s*\}\}")
_ANY_BRACES = re.compile(r"\{\{|\}\}")
_FORBIDDEN_HEADERS = frozenset({"host", "connection", "transfer-encoding", "content-length", "te", "upgrade"})
_FORBIDDEN_HEADER_PREFIXES = ("proxy-", "x-forwarded-")
_NEEDS_TEST_REQUEST = (
    "The connection form needs a test_request: the request Cowork sends to "
    "check that the API returns data. Give each method one."
)


def _is_oauth_grant_method(method: dict[str, Any]) -> bool:
    """Whether ``method`` is a browser OAuth grant (its values arrive from the provider)."""
    return method.get("submit_action") == "oauth_launch" and isinstance(method.get("oauth"), dict)


def _form_methods(form_spec: dict[str, Any]) -> list[tuple[str, dict[str, Any], dict[str, Any] | None]]:
    """Each method as (where, holder of its test_request, method or None).

    A form with no ``methods`` is one method: the form itself holds it.
    """
    # Hidden methods too: a client can still submit one.
    methods = [m for m in form_spec.get("methods") or [] if isinstance(m, dict)]
    if not methods:
        return [("the form", form_spec, None)]
    return [(f"method {m.get('id')!r}", m, m) for m in methods]


def _placeholders(text: str) -> set[str]:
    """Placeholder names in ``text``; raises when braces are left over that are not one."""
    if _ANY_BRACES.search(PLACEHOLDER.sub("", text)):
        raise UnstorableFormError("test_request has a malformed {{placeholder}}.")
    return set(PLACEHOLDER.findall(text))


def _check_literal(text: str, where: str) -> None:
    """Refuse literal text in a test request that looks like a credential value."""
    literal = PLACEHOLDER.sub("", text)
    if scrub_output(literal) != literal:
        raise UnstorableFormError(
            f"test_request {where} looks like it holds a credential value. Use a {{{{field}}}} placeholder."
        )


def _json_strings(value: Any, where: str) -> list[str]:
    """Every leaf string of a JSON body; placeholders are refused in object keys."""
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        out: list[str] = []
        for key, item in value.items():
            if _placeholders(str(key)):
                raise UnstorableFormError(f"test_request {where} may not use a placeholder in a JSON key.")
            out += _json_strings(item, where)
        return out
    if isinstance(value, list):
        return [s for item in value for s in _json_strings(item, where)]
    return []


def _scheme_and_authority(url: str) -> str:
    """The text up to the end of ``url``'s host and port, or "" when it has no
    "scheme://". Plain string search: the url is form text of any length."""
    scheme, sep, rest = url.partition("://")
    if not sep or any(c in scheme for c in ":/?#"):
        return ""
    end = min((i for i in (rest.find(c) for c in "/?#") if i != -1), default=len(rest))
    return f"{scheme}://{rest[:end]}"


def _check_step(step: dict[str, Any], where: str) -> set[str]:
    """Check one test step's shape rules and return the placeholder names it uses."""
    url = str(step.get("url") or "")
    # The raw scheme and authority, before any placeholder could add a "/".
    if _ANY_BRACES.search(_scheme_and_authority(url)):
        raise UnstorableFormError(f"test_request {where} url may use placeholders only in its path and query.")
    try:
        parts = urlsplit(PLACEHOLDER.sub("x", url))
    except ValueError as e:
        raise UnstorableFormError(f"test_request {where} url is not a valid URL.") from e
    if parts.scheme not in ("http", "https") or not parts.hostname:
        raise UnstorableFormError(f"test_request {where} url must be an http or https URL with a host.")
    if "@" in parts.netloc:
        raise UnstorableFormError(f"test_request {where} url must not carry user info; use basic_auth.")
    # The runner dials with httpx, so the form is checked with the same parser.
    filled = PLACEHOLDER.sub("x", url)
    try:
        parsed = httpx.URL(filled)
        parsed.raw_host.decode("ascii")
    except (httpx.InvalidURL, UnicodeError) as e:
        raise UnstorableFormError(f"test_request {where} url is not a valid URL.") from e
    if parsed.port is not None and not 1 <= parsed.port <= 65535:
        raise UnstorableFormError(f"test_request {where} url has a port outside 1 to 65535.")
    used = _placeholders(url)
    _check_literal(url, f"{where} url")
    for name, value in (step.get("headers") or {}).items():
        if any(c in f"{name}{value}" for c in "\r\n"):
            raise UnstorableFormError(f"test_request {where} has a header with a line break.")
        lowered = name.strip().lower()
        if lowered in _FORBIDDEN_HEADERS or lowered.startswith(_FORBIDDEN_HEADER_PREFIXES):
            raise UnstorableFormError(f"test_request {where} may not set the {name} header.")
        if _placeholders(name):
            raise UnstorableFormError(f"test_request {where} may not use a placeholder in a header name.")
        used |= _placeholders(value)
        _check_literal(value, f"{where} header {name}")
    for item in step.get("basic_auth") or []:
        names = _placeholders(item)
        if not names or PLACEHOLDER.sub("", item).strip():
            raise UnstorableFormError(f"test_request {where} basic_auth takes two {{{{field}}}} placeholders.")
        used |= names
    if step.get("form") is not None and step.get("json_body") is not None:
        raise UnstorableFormError(f"test_request {where} takes a form or a json_body, not both.")
    for key, value in (step.get("form") or {}).items():
        if _placeholders(key):
            raise UnstorableFormError(f"test_request {where} may not use a placeholder in a form key.")
        used |= _placeholders(value)
        _check_literal(value, f"{where} form value")
    for text in _json_strings(step.get("json_body"), where):
        used |= _placeholders(text)
        _check_literal(text, f"{where} json_body")
    return used


def check_test_requests(form_spec: dict[str, Any]) -> None:
    """Refuse a handcrafted form whose methods do not each carry a valid test_request.

    A single-method form holds one at the top level; a multi-method form holds
    one on every method, hidden ones included, and none at the top level. Placeholders
    may name only the method's own fields, ``token`` in the data step after a
    token step, and ``access_token``/``refresh_token`` on an OAuth grant
    method. A follow-up step (``_extends_connection``) may also name the saved
    connection's fields, which the probe handler checks once it has read them.

    Raises:
        UnstorableFormError: with a message the model can act on.
    """
    multi = bool([m for m in form_spec.get("methods") or [] if isinstance(m, dict)])
    if multi and form_spec.get("test_request") is not None:
        raise UnstorableFormError(
            "A multi-method form puts a test_request on each method, not at the top level."
        )
    extends = bool(form_spec.get("_extends_connection"))
    for where, holder, method in _form_methods(form_spec):
        request = holder.get("test_request")
        if not isinstance(request, dict):
            raise UnstorableFormError(f"{_NEEDS_TEST_REQUEST} {where.capitalize()} has none.")
        fields = (method or form_spec).get("fields") or []
        field_names = {f.get("name") for f in fields if isinstance(f, dict) and f.get("name")}
        secret_names = {
            f["name"] for f in fields
            if isinstance(f, dict) and f.get("name") and (f.get("secret") or f.get("type") == "password")
        }
        grant_names = {"access_token", "refresh_token"} if method and _is_oauth_grant_method(method) else set()
        token_step = request.get("token")
        allowed = field_names | grant_names
        if isinstance(token_step, dict):
            unknown = _check_step(token_step, f"{where} token step") - allowed
            if unknown and not extends:
                raise UnstorableFormError(
                    f"test_request {where} token step names {', '.join(sorted(unknown))}, which are not form fields."
                )
        data_step = request.get("data") or {}
        data_allowed = allowed | ({"token"} if isinstance(token_step, dict) else set())
        data_used = _check_step(data_step, f"{where} data step")
        unknown = data_used - data_allowed
        if unknown and not extends:
            raise UnstorableFormError(
                f"test_request {where} data step names {', '.join(sorted(unknown))}, which are not form fields."
            )
        if not (data_used & ({"token", "access_token"} | secret_names) or data_step.get("basic_auth") or extends):
            raise UnstorableFormError(
                f"test_request {where} data step must send a credential: {{{{token}}}}, "
                "basic_auth or a secret field."
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
