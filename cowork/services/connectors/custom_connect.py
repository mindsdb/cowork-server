"""Connecting a member's custom API connector: the address they type, the values, and the check.

The template only says which path to GET and how to lay out the headers; the
member types the API's address, and only that origin is dialed, through the
pinned public-only helper. The check must first be refused without the
credential headers, then pass with them. Nothing here logs or raises a
submitted value: errors name a field or a code, and a failure's excerpt has
every submitted value replaced before it leaves.
"""

from __future__ import annotations

import logging
import re
import unicodedata
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

import httpx
from anton.core.tools.connector_draft import ConnectorDraftCheck, ConnectorDraftError, parse_connector_draft
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
_PLACEHOLDER = re.compile(r"\{([a-z][a-z0-9_]{0,39})\}")
_NO_CHECK_NEXT = "If the API takes a key, ask Anton to build it again; if it takes none, Anton can use it without a connector."
_ADDRESS_HINT = "type only the address, like api.example.com"


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


def stored_check(row: CustomConnector) -> ConnectorDraftCheck:
    """Parse the row's stored check again with anton's parser, against the row's own fields.

    Raises:
        ConnectRefused: 409 ``no_check`` when the template has none, or one
            that no longer passes the parser.
    """
    fields = [
        {"name": f["name"], "label": f.get("label") or f["name"], "secret": bool(f.get("secret")), "required": bool(f.get("required"))}
        for f in row.spec.get("fields", [])
    ]
    draft = {"connector_id": row.connector_id, "label": row.label, "fields": fields, "check": row.connection_check}
    try:
        parsed = parse_connector_draft(draft)
    except ConnectorDraftError as exc:
        raise ConnectRefused("no_check", 409, "This connector's connection check is not usable, so it cannot be connected. " + _NO_CHECK_NEXT) from exc
    if parsed.check is None:
        raise ConnectRefused("no_check", 409, "This connector has no connection check, so it cannot be connected. " + _NO_CHECK_NEXT)
    return parsed.check


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
    if any(c in text for c in "@?#\\"):
        raise _refuse_address()
    raw = text if "://" in text else f"https://{text}"
    remainder = raw.split("://", 1)[1].removesuffix("/")
    if "/" in remainder:
        raise _refuse_address(f"The API address cannot include a path; {_ADDRESS_HINT}.")
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
    for name, field in known.items():
        value = values.get(name)
        if value is None or value == "":
            if field.get("required"):
                raise ConnectRefused("invalid_values", 422, f"{name} is required.")
            continue
        if not isinstance(value, str) or len(value) > VALUE_MAX or not all(0x20 <= ord(c) <= 0x7E for c in value):
            raise ConnectRefused(
                "invalid_values", 422, f"{name} must be printable ASCII text of at most {VALUE_MAX} characters."
            )
        accepted[name] = value
    return accepted


def check_headers(check: ConnectorDraftCheck, values: Mapping[str, str]) -> tuple[dict[str, str], dict[str, str]]:
    """Split the check's headers into the fixed ones and the ones that place a value, filled in.

    A placeholder is replaced by an exact ``{name}`` match only.

    Raises:
        ConnectRefused: 422 ``invalid_values`` when a placed field has no value.
    """
    fixed: dict[str, str] = {}
    placed: dict[str, str] = {}
    for name, template in check.headers:
        names = _PLACEHOLDER.findall(template)
        if not names:
            fixed[name] = template
            continue
        missing = [field for field in names if field not in values]
        if missing:
            raise ConnectRefused("invalid_values", 422, f"{missing[0]} is needed to check the connection.")
        placed[name] = _PLACEHOLDER.sub(lambda m: values[m.group(1)], template)
    return fixed, placed


def secure_keys(fields: list[dict[str, Any]], check: ConnectorDraftCheck, values: Mapping[str, str]) -> list[str]:
    """The fields stored as secrets: the secret fields given, plus every field the check places."""
    placed = {name for _, template in check.headers for name in _PLACEHOLDER.findall(template)}
    secret = {f["name"] for f in fields if f.get("secret")}
    return sorted(name for name in values if name in secret or name in placed)


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


async def _send(origin: str, path: str, headers: dict[str, str]) -> egress.PinnedResponse | CheckFailure:
    """One pinned GET of the check, or the failure its error stands for, logged by type only."""
    try:
        return await egress.pinned_get(
            origin, path, headers, total_seconds=CHECK_SECONDS, max_body_bytes=CHECK_BODY_BYTES
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


async def run_check(origin: str, check: ConnectorDraftCheck, values: Mapping[str, str]) -> CheckFailure | None:
    """Send the check without its credential headers, then with them; None when it passed.

    Without them it must answer 401 or 403, or a passing check would prove
    nothing. With them, 2xx passes, 401 or 403 is ``authentication_failed``
    and any other status ``provider_error``.
    """
    fixed, placed = check_headers(check, values)
    unprotected = await _send(origin, check.path, fixed)
    if isinstance(unprotected, CheckFailure):
        return unprotected
    if unprotected.status_code not in (401, 403):
        return CheckFailure(
            "check_not_protected", unprotected.status_code, scrub_excerpt(unprotected.body, values.values())
        )
    answer = await _send(origin, check.path, {**fixed, **placed})
    if isinstance(answer, CheckFailure):
        return answer
    if 200 <= answer.status_code < 300:
        return None
    code = "authentication_failed" if answer.status_code in (401, 403) else "provider_error"
    return CheckFailure(code, answer.status_code, scrub_excerpt(answer.body, values.values()))
