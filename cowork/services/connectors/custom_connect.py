"""Connecting a member's custom API connector: the address they type, the values, and the check.

The template says which path to GET and, in its auth section, how the
member's values go on a request; the member types the API's address, and
only that origin is dialed, through the pinned public-only helper. The
values are filled in by anton's shared rules, as the gateway fills them.
The check must first be refused without the placed credentials, then pass
with them. Nothing here logs or raises a submitted value: errors name a
field or a code, and a failure's excerpt has every submitted value, and
every form of it that went on the request, replaced before it leaves.
"""

from __future__ import annotations

import base64
import logging
import re
import unicodedata
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any
from urllib.parse import quote, quote_plus

import httpx
from anton.core.tools import connector_auth_rules
from anton.core.tools.connector_draft import ConnectorDraftError, parse_connector_draft
from redis.asyncio import Redis
from redis.exceptions import RedisError

from cowork.models.custom_connector import CustomConnector
from cowork.services.connectors import egress

logger = logging.getLogger(__name__)

CONNECT_ATTEMPTS_MAX = 10
CONNECT_WINDOW_SECONDS = 600
CHECK_SECONDS = 10.0
CHECK_BODY_BYTES = 16 * 1024
EXCERPT_MAX = 300
VALUE_MAX = 4096
BASE_URL_FIELD = "base_url"

_HOST_LABEL = re.compile(r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?")
_TOP_LABEL = re.compile(r"[a-z]{2,63}")
_NO_CHECK_NEXT = "If the API takes a key, ask Anton to build it again; if it takes none, Anton can use it without a connector."
_ADDRESS_HINT = "type only the address, like api.example.com"
# A query-key API often refuses a missing key as a bad request. The two sends
# differ only by the placed values, so a refusal here and a 2xx with them still proves the key.
_REFUSED_WITHOUT_KEY = frozenset({400, 401, 403, 422})
_VERSION_HINT = (
    "If your provider's address includes a path or version such as /v2, "
    "ask Anton to build the connector again with it in the connector's paths."
)


class ConnectRefused(Exception):
    """The connect was refused before or by the check; ``code`` says why, never with a value."""

    def __init__(self, code: str, status_code: int, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.status_code = status_code
        self.message = message


@dataclass(frozen=True, repr=False)
class CheckFailure:
    """A check that ran and did not pass: its code, the provider's status and a scrubbed excerpt."""

    code: str
    provider_status: int | None
    excerpt: str


@dataclass(frozen=True)
class StoredTemplate:
    """What connect needs of a saved template, parsed again: its check path and auth section."""

    path: str
    auth: dict[str, Any]


def parse_stored(row: CustomConnector) -> StoredTemplate | None:
    """Parse the row's fields, auth section and check again with anton's parser.

    Returns None for a template saved without an auth section, without a
    check, or in a shape the parser no longer accepts.
    """
    fields = [
        {
            "name": f["name"],
            "label": f.get("label") or f["name"],
            "secret": bool(f.get("secret")),
            "required": f.get("required", True),
        }
        for f in row.spec.get("fields", [])
    ]
    draft = {
        "connector_id": row.connector_id,
        "label": row.label,
        "fields": fields,
        "auth": row.connection_auth,
        "check": row.connection_check,
    }
    try:
        parsed = parse_connector_draft(draft)
    except ConnectorDraftError:
        return None
    if parsed.check is None:
        return None
    return StoredTemplate(path=parsed.check.path, auth=parsed.auth.to_dict())


def stored_template(row: CustomConnector) -> StoredTemplate:
    """The row's template as connect uses it.

    Raises:
        ConnectRefused: 409 ``no_check`` when it has no check or auth
            section, or one the parser no longer accepts.
    """
    template = parse_stored(row)
    if template is None:
        raise ConnectRefused("no_check", 409, "This connector cannot be connected as saved. " + _NO_CHECK_NEXT)
    return template


def _refuse_address(message: str = "The API address is not valid.") -> ConnectRefused:
    """A 422 about the typed address."""
    return ConnectRefused("invalid_address", 422, message)


def normalize_address(text: Any, refused_domains: Iterable[str]) -> str:
    """Return the member's typed API address as ``https://host``, or refuse it.

    Accepts a bare host or an ``https://`` origin, port 443 or none, with an
    optional trailing slash. The host is two or more ASCII labels, the last
    letters only, no ``xn--`` label, and not one of
    ``refused_domains`` or under one.

    Raises:
        ConnectRefused: 422 ``invalid_address``, or ``own_domain``.
    """
    # Whitespace and non-ASCII need no test of their own: httpx refuses them
    # or escapes them into the host, where the label rules refuse them.
    if not isinstance(text, str) or not text:
        raise _refuse_address()
    raw = text if "://" in text else f"https://{text}"
    # A URL pasted from the docs carries a query too; it still gets the path hint.
    remainder = raw.split("://", 1)[1].split("?", 1)[0].split("#", 1)[0].removesuffix("/")
    if "/" in remainder:
        raise _refuse_address(f"The API address cannot include a path; {_ADDRESS_HINT}. {_VERSION_HINT}")
    if any(c in text for c in "@?#\\"):
        raise _refuse_address()
    try:
        url = httpx.URL(raw)
    except httpx.InvalidURL as exc:
        raise _refuse_address() from exc
    if url.scheme != "https" or url.userinfo or url.port not in (None, 443):
        raise _refuse_address()
    host = url.host.removesuffix(".")
    # A last label of letters only also refuses every IP literal.
    labels = host.split(".")
    if (
        len(host) > 253
        or len(labels) < 2
        or not all(_HOST_LABEL.fullmatch(label) for label in labels)
        or any(label.startswith("xn--") for label in labels)
        or not _TOP_LABEL.fullmatch(labels[-1])
    ):
        raise _refuse_address()
    for domain in refused_domains:
        domain = domain.strip().lower().removesuffix(".")
        if domain and (host == domain or host.endswith(f".{domain}")):
            raise ConnectRefused("own_domain", 422, "That address belongs to this service and cannot be used.")
    return f"https://{host}"


def check_values(fields: list[dict[str, Any]], values: Any) -> dict[str, str]:
    """Return the non-empty values for the template's own fields, or refuse.

    Raises:
        ConnectRefused: 422 ``invalid_values`` naming the field, never the value.
    """
    if not isinstance(values, Mapping):
        raise ConnectRefused("invalid_values", 422, "values must be an object.")
    known = {f["name"]: f for f in fields}
    if any(name not in known for name in values):
        raise ConnectRefused("invalid_values", 422, "values has a field this connector does not declare.")
    accepted: dict[str, str] = {}
    for name in known:
        value = values.get(name)
        if value is None or value == "":
            raise ConnectRefused("invalid_values", 422, f"{name} is required.")
        if not isinstance(value, str) or len(value) > VALUE_MAX or not all(0x20 <= ord(c) <= 0x7E for c in value):
            raise ConnectRefused(
                "invalid_values", 422, f"{name} must be printable ASCII text of at most {VALUE_MAX} characters."
            )
        # The HTTP library refuses such a header value with an error that quotes it.
        if value != value.strip():
            raise ConnectRefused("invalid_values", 422, f"{name} cannot start or end with a space.")
        accepted[name] = value
    return accepted


@dataclass(frozen=True, repr=False)
class PlacedRequest:
    """What the check sends: the credential-free part first, then everything ``render`` placed."""

    bare_headers: dict[str, str]
    bare_params: list[tuple[str, str]]
    headers: dict[str, str]
    params: list[tuple[str, str]]


def place_values(auth: Mapping[str, Any], values: Mapping[str, str]) -> PlacedRequest:
    """Fill the auth section with the values by anton's shared rules, as the gateway will.

    The bare request keeps only the headers and params whose template has no
    ``{field}`` slot; ``basic`` has none.

    Raises:
        ConnectRefused: 422 ``invalid_values`` when the rules refuse a value.
    """
    try:
        headers, params = connector_auth_rules.render(auth, values)
    except connector_auth_rules.AuthRuleError as exc:
        raise ConnectRefused("invalid_values", 422, "A value cannot be placed on the request.") from exc
    bare_headers = dict(_slot_free(auth.get("headers", {}))) if auth["type"] == "header" else {}
    bare_params = _slot_free(auth.get("params", {})) if auth["type"] == "query" else []
    return PlacedRequest(bare_headers, bare_params, headers, params)


def auth_summary(auth: Mapping[str, Any]) -> dict[str, Any]:
    """The auth type and the header or parameter names that carry a value, for the connect form.

    Names only: a template's text can be model-written, so none of it is shown.
    """
    if auth["type"] == "basic":
        return {"type": "basic", "names": []}
    templates = auth.get("params" if auth["type"] == "query" else "headers", {})
    slot_free = {name for name, _ in _slot_free(templates)}
    return {"type": auth["type"], "names": [name for name in templates if name not in slot_free]}


def _slot_free(templates: Mapping[str, str]) -> list[tuple[str, str]]:
    """The entries whose template places no value."""
    return [(name, text) for name, text in templates.items() if not connector_auth_rules.PLACEHOLDER.search(text)]


def sent_forms(auth: Mapping[str, Any], request: PlacedRequest, values: Iterable[str]) -> set[str]:
    """Every form of a submitted value that went on the request, for the excerpt scrub.

    The values themselves, their percent-encodings, each placed header and
    param value, and for ``basic`` the bare base64 credential with and without
    its padding.
    """
    forms = {value for value in values if value}
    forms |= {quote(value, safe="") for value in list(forms)} | {quote_plus(value) for value in list(forms)}
    forms |= {value for value in request.headers.values() if value}
    forms |= {value for _, value in request.params if value}
    if auth["type"] == "basic":
        token = request.headers["Authorization"].removeprefix("Basic ")
        forms |= {token, token.rstrip("=")}
    # A slot-free header or param is template text, not a member's value.
    forms -= set(request.bare_headers.values()) | {value for _, value in request.bare_params}
    return {form for form in forms if form}


def scrub_excerpt(body: bytes, values: Iterable[str]) -> str:
    """The provider's answer as text a member may see: controls removed, values replaced, cut short.

    In order: decoded as UTF-8 with replacement, Cc and Cf characters
    removed, every non-empty value replaced with ``***`` longest first, then
    cut to EXCERPT_MAX. A value the provider re-encodes or splits is not caught.
    """
    text = body.decode("utf-8", errors="replace")
    text = "".join(c for c in text if unicodedata.category(c) not in ("Cc", "Cf"))
    for value in sorted({v for v in values if v}, key=len, reverse=True):
        text = text.replace(value, "***")
    return text[:EXCERPT_MAX]


async def take_attempt(redis: Redis, key: str) -> None:
    """Count one connect attempt for ``key``; every attempt in the window pushes it out again.

    Raises:
        ConnectRefused: 429 ``rate_limited`` past CONNECT_ATTEMPTS_MAX, or
            503 ``unavailable`` when Redis cannot be asked.
    """
    try:
        async with redis.pipeline(transaction=True) as pipe:
            pipe.incr(key)
            pipe.expire(key, CONNECT_WINDOW_SECONDS)
            count, _ = await pipe.execute()
    except (RedisError, OSError) as exc:
        logger.warning("custom connect: attempt counter unavailable (%s)", type(exc).__name__)
        raise ConnectRefused("unavailable", 503, "Connecting is unavailable right now; try again later.") from exc
    if count > CONNECT_ATTEMPTS_MAX:
        raise ConnectRefused("rate_limited", 429, "Too many connection attempts; wait a few minutes and try again.")


async def _send(
    origin: str, path: str, headers: dict[str, str], params: list[tuple[str, str]]
) -> egress.PinnedResponse | CheckFailure:
    """One pinned GET of the check, or the failure its error stands for, logged by type only."""
    try:
        return await egress.pinned_get(
            origin, path, headers, total_seconds=CHECK_SECONDS, max_body_bytes=CHECK_BODY_BYTES, params=params
        )
    except egress.EgressRefused as exc:
        code, error = "address_refused", exc
    except egress.EgressHostUnresolved as exc:
        code, error = "host_unresolved", exc
    except egress.EgressHostNotPublic as exc:
        code, error = "host_not_public", exc
    except egress.EgressTimeout as exc:
        code, error = "timed_out", exc
    except (httpx.HTTPError, UnicodeError) as exc:
        code, error = "unreachable", exc
    logger.info("custom connect: check not sent or not answered (%s)", type(error).__name__)
    return CheckFailure(code, None, "")


async def run_check(origin: str, template: StoredTemplate, values: Mapping[str, str]) -> CheckFailure | None:
    """Send the check without the placed credentials, then with them; None when it passed.

    Without them it must answer 400, 401, 403 or 422, or a passing check would prove
    nothing. With them, 2xx passes, 401 or 403 is ``authentication_failed``
    and any other status ``provider_error``. A 404 to either send is
    ``check_path_not_found``. Only the answer to the request that carried
    the credentials is excerpted.

    Raises:
        ConnectRefused: 422 ``invalid_values`` when a value cannot be placed.
    """
    placed = place_values(template.auth, values)
    unprotected = await _send(origin, template.path, placed.bare_headers, placed.bare_params)
    if isinstance(unprotected, CheckFailure):
        return unprotected
    if unprotected.status_code not in _REFUSED_WITHOUT_KEY:
        # Its body is never shown: an excerpt would let connect read any page our egress can reach.
        code = "check_path_not_found" if unprotected.status_code == 404 else "check_not_protected"
        return CheckFailure(code, unprotected.status_code, "")
    answer = await _send(origin, template.path, placed.headers, placed.params)
    if isinstance(answer, CheckFailure):
        return answer
    if 200 <= answer.status_code < 300:
        return None
    if answer.status_code in (401, 403):
        code = "authentication_failed"
    elif answer.status_code == 404:
        code = "check_path_not_found"
    else:
        code = "provider_error"
    scrubbed = scrub_excerpt(answer.body, sent_forms(template.auth, placed, values.values()))
    return CheckFailure(code, answer.status_code, scrubbed)
