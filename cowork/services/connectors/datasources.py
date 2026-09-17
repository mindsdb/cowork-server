"""Datasource connection input, parsed and checked before the relay to auth.

The rules here mirror auth's canonicalize so a bad connection is refused
before a password crosses a service boundary. Auth validates again and owns
the storage contract; this is not the only gate, it is the early one.

Every rejection message is a fixed string. The DSN, the password and the CA
are never interpolated into an error, a log or a response.
"""

from __future__ import annotations

import re
from typing import Any
from urllib.parse import parse_qsl, unquote, urlsplit

from fastapi import HTTPException, status

from cowork.schemas.connectors import DatasourceCreateRequest

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


def _canonical_host(value: Any) -> str:
    """Normalize a hostname, rejecting multi-host lists, sockets and IPv6 literals."""
    host = str(value or "").strip().rstrip(".").lower()
    if not host or len(host) > 253 or not _HOST_RE.fullmatch(host) or "/" in host or ":" in host:
        raise InvalidDatasourceInput("host must be a single hostname or IP address")
    return host


def _canonical_port(value: Any, connector_id: str) -> int:
    """Resolve the port, which this release pins to the connector default.

    Auth accepts 1..65535; the connector spec accepts only the default in this
    release, so the stricter of the two is applied before relay.
    """
    default = DEFAULT_PORTS[connector_id]
    if value in (None, ""):
        return default
    try:
        port = int(value)
    except (TypeError, ValueError) as exc:
        raise InvalidDatasourceInput("port must be an integer") from exc
    if port != default:
        raise InvalidDatasourceInput("only the connector's default port is accepted in this release")
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

    return {
        "connector_id": connector_id,
        "method": method,
        "name": model.name.strip(),
        "host": _canonical_host(fields.get("host")),
        "port": _canonical_port(fields.get("port"), connector_id),
        "database": database,
        "username": username,
        "password": password,
        "tls": _canonical_tls(tls),
    }
