"""Datasource connection input, parsed and checked before the relay to auth.

These rules are a subset of auth's canonicalize, not a second copy of it:
enough to refuse a bad connection before a password crosses a service
boundary. Auth re-parses everything and owns the storage contract, so where
this is looser (it checks PEM framing but does not parse X.509, for example)
the result is a rejection one hop later, never an accepted bad connection.

Every rejection message is a fixed string. The DSN, the password and the CA
are never interpolated into an error, a log or a response.
"""

from __future__ import annotations

import ipaddress
import re
import socket
from typing import Any
from urllib.parse import parse_qsl, unquote, urlsplit

from fastapi import HTTPException, status

from cowork.schemas.connectors import DatasourceCreateRequest
from cowork.services.connectors.datasource_capabilities import load_datasource_capabilities

#: Connector to its single supported cloud method, mirroring auth.
SUPPORTED_METHODS = {"postgres": "host-port", "mysql": "host-password"}
DEFAULT_PORTS = {"postgres": 5432, "mysql": 3306}
MAX_CA_PEM_BYTES = 64 * 1024

_HOST_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9.-]{0,252}[A-Za-z0-9]$")
_STRUCTURED_FIELDS = ("host", "port", "database", "username", "password", "tls")
_CERTIFICATE_RE = re.compile(r"-----BEGIN CERTIFICATE-----.+?-----END CERTIFICATE-----", re.DOTALL)


class InvalidDatasourceInput(HTTPException):
    """400 whose message is fixed text, never the value that was rejected."""

    def __init__(self, message: str) -> None:
        super().__init__(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"code": "invalid_connection", "message": message},
        )


class UnsupportedDatasourceCapability(HTTPException):
    """409 for a method this deployment does not run as a cloud datasource.

    Raised by both capture paths, the management routes and the submission
    relay, so a client branching on `code` sees one answer whichever door it
    knocked on.
    """

    def __init__(self) -> None:
        super().__init__(
            status_code=status.HTTP_409_CONFLICT,
            detail={
                "code": "unsupported_capability",
                "message": "This connector method is not available in cloud.",
            },
        )


def require_cloud_method_enabled(connector_id: str | None, method: str | None) -> None:
    """Refuse a method the capability policy has not enabled.

    Runs before the body is parsed, so a disabled method answers the same way
    whether or not the connection itself is well formed. Reads and deletes do
    not call this: a connection captured while a method was enabled must stay
    visible and removable after it is switched off.
    """
    capabilities = load_datasource_capabilities()
    if not capabilities.is_available((connector_id or "").strip().lower(), (method or "").strip()):
        raise UnsupportedDatasourceCapability()


def _canonical_host(value: Any) -> str:
    """Normalize a hostname, rejecting multi-host lists, sockets and IPv6 literals.

    An address literal is refused when it points somewhere only the pod can
    reach: the stored host is dialed from a cluster pod later, where
    169.254.169.254 is the cloud metadata service. RFC1918 stays allowed
    because a VPC-peered private address is a legitimate customer database.

    Parsing goes through inet_aton, which is what the resolver itself accepts,
    so the numeric spellings that reach the same address (2852039166 and
    0xa9fea9fe are both 169.254.169.254) are caught rather than mistaken for
    DNS names, and the literal is stored in its canonical form.

    This is an early filter, not the wall. A DNS name that resolves to a
    link-local address passes here, and no literal check can catch that or
    DNS rebinding; only resolving at connect time can, and that code lives
    with whatever dials the connection.
    """
    host = str(value or "").strip().rstrip(".").lower()
    if not host or len(host) > 253 or not _HOST_RE.fullmatch(host) or "/" in host or ":" in host:
        raise InvalidDatasourceInput("host must be a single hostname or IP address")
    if host == "localhost" or host.endswith(".localhost"):
        raise InvalidDatasourceInput("host must be a reachable database server, not a local address")
    try:
        packed = socket.inet_aton(host)
    except OSError:
        return host
    address = ipaddress.ip_address(packed)
    if (
        address.is_loopback
        or address.is_link_local
        or address.is_unspecified
        or address.is_reserved
        or address.is_multicast
    ):
        raise InvalidDatasourceInput("host must be a reachable database server, not a local address")
    return str(address)


def _canonical_port(value: Any, connector_id: str) -> int:
    """Resolve the port, defaulting to the connector's standard one.

    The bound is auth's storage contract, 1..65535. Whether a release supports
    only the default port belongs to the capability policy and the adapter, not
    to this normalizer: refusing it here would reject the managed poolers that
    auth accepts and store.
    """
    if value in (None, ""):
        return DEFAULT_PORTS[connector_id]
    try:
        port = int(value)
    except (TypeError, ValueError) as exc:
        raise InvalidDatasourceInput("port must be an integer") from exc
    if not 1 <= port <= 65535:
        raise InvalidDatasourceInput("port must be between 1 and 65535")
    return port


def _canonical_tls(tls: Any) -> dict[str, Any]:
    """Check the TLS block's framing and bound; auth re-parses the certificates."""
    if tls is None:
        return {"mode": "system", "ca_pem": None}
    mode = tls.mode
    ca_pem = tls.ca_pem
    if mode == "system":
        if ca_pem not in (None, ""):
            raise InvalidDatasourceInput("system TLS cannot include a CA certificate")
        return {"mode": "system", "ca_pem": None}
    if not isinstance(ca_pem, str) or not ca_pem.strip():
        raise InvalidDatasourceInput("custom_ca TLS requires a CA certificate")
    ca_pem = ca_pem.strip()
    if len(ca_pem.encode()) > MAX_CA_PEM_BYTES:
        raise InvalidDatasourceInput("the CA certificate exceeds the 64 KiB limit")
    blocks = _CERTIFICATE_RE.findall(ca_pem)
    if not blocks or re.sub(r"\s+", "", ca_pem) != "".join(re.sub(r"\s+", "", block) for block in blocks):
        raise InvalidDatasourceInput("the CA certificate must contain only X.509 certificate blocks")
    return {"mode": "custom_ca", "ca_pem": ca_pem}


def parse_datasource_dsn(dsn: str, connector_id: str) -> dict[str, Any]:
    """Split a DSN into the approved connection fields.

    Driver options are refused rather than forwarded: anything beyond
    PostgreSQL's sslmode=verify-full can change how the client connects.
    """
    if not isinstance(dsn, str) or not dsn.strip():
        raise InvalidDatasourceInput("a connection string is required")
    try:
        parsed = urlsplit(dsn.strip())
    except ValueError as exc:
        raise InvalidDatasourceInput("the connection string is malformed") from exc

    allowed_schemes = {"postgres", "postgresql"} if connector_id == "postgres" else {"mysql"}
    try:
        hostname = parsed.hostname
    except ValueError as exc:
        raise InvalidDatasourceInput("the connection string host is invalid") from exc
    if parsed.scheme.lower() not in allowed_schemes or not hostname:
        raise InvalidDatasourceInput("the connection string scheme or host is unsupported")
    if hostname.count("@") or parsed.netloc.count("@") != 1:
        raise InvalidDatasourceInput("the connection string user information is ambiguous")
    if parsed.username is None or parsed.password is None:
        raise InvalidDatasourceInput("the connection string must include a username and password")
    if parsed.path.count("/") > 1 or not parsed.path.strip("/"):
        raise InvalidDatasourceInput("the connection string must name exactly one database")

    query = parse_qsl(parsed.query, keep_blank_values=True)
    if len({key for key, _ in query}) != len(query):
        raise InvalidDatasourceInput("the connection string repeats a query option")
    if connector_id == "postgres":
        if any(key != "sslmode" or value != "verify-full" for key, value in query):
            raise InvalidDatasourceInput("only sslmode=verify-full is accepted in a PostgreSQL connection string")
    elif query:
        raise InvalidDatasourceInput("a MySQL connection string cannot carry query options")

    try:
        port = parsed.port
    except ValueError as exc:
        raise InvalidDatasourceInput("the connection string port is invalid") from exc

    return {
        "host": hostname,
        "port": port,
        "database": unquote(parsed.path.lstrip("/")),
        "username": unquote(parsed.username),
        "password": unquote(parsed.password),
        "tls": None,
    }


def normalize_datasource_input(model: DatasourceCreateRequest) -> dict[str, Any]:
    """Turn a create or edit body into the payload auth accepts.

    Returns only the canonical fields: never `dsn`, never `input_mode`, and
    never `expected_version`, which travels as its own relay argument.
    """
    connector_id = (model.connector_id or "").strip().lower()
    method = (model.method or "").strip()
    if connector_id not in SUPPORTED_METHODS or method != SUPPORTED_METHODS[connector_id]:
        raise InvalidDatasourceInput("this connector method is not supported for cloud connections")

    if model.input_mode == "dsn":
        supplied = [name for name in _STRUCTURED_FIELDS if getattr(model, name) is not None]
        if supplied:
            raise InvalidDatasourceInput("a connection string cannot be combined with individual fields")
        fields = parse_datasource_dsn(model.dsn or "", connector_id)
        tls = None
    else:
        if model.dsn is not None:
            raise InvalidDatasourceInput("a connection string requires input_mode=dsn")
        fields = {name: getattr(model, name) for name in _STRUCTURED_FIELDS}
        tls = model.tls

    database = str(fields.get("database") or "").strip()
    username = str(fields.get("username") or "").strip()
    password = fields.get("password") or ""
    if not database or not username or not password:
        raise InvalidDatasourceInput("database, username and password are required")

    name = model.name.strip()
    if not name:
        raise InvalidDatasourceInput("name is required")

    return {
        "connector_id": connector_id,
        "method": method,
        "name": name,
        "host": _canonical_host(fields.get("host")),
        "port": _canonical_port(fields.get("port"), connector_id),
        "database": database,
        "username": username,
        "password": password,
        "tls": _canonical_tls(tls),
    }
