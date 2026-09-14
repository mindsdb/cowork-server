"""Validate credentials for Code's first-class developer connectors.

The generic direct-save endpoint is intentionally probe-free because its OAuth
callers have already exchanged and verified a token.  Personal tokens entered
from Code need a separate boundary: validate them with the provider before a
vault record is created, otherwise an invalid token would appear connected.
"""
from __future__ import annotations

import ipaddress
import socket
import threading
from dataclasses import dataclass
from typing import Any, Callable
from urllib.parse import urlparse

import httpx
from httpx._utils import get_environment_proxies


class DeveloperCredentialError(ValueError):
    """The submitted credential or connector configuration is invalid."""


class DeveloperProviderUnavailable(RuntimeError):
    """The provider could not be reached to validate a credential."""


@dataclass(frozen=True)
class ValidatedDeveloperIdentity:
    account_email: str

    def as_fields(self) -> dict[str, str]:
        return {"account_email": self.account_email}


_METHODS: dict[str, frozenset[str]] = {
    "github": frozenset({"fine-grained-pat", "classic-pat"}),
    "linear": frozenset({"personal-api-key"}),
}


def validate_developer_connection(
    connector_id: str,
    method: str | None,
    values: dict[str, Any],
    *,
    transport: httpx.BaseTransport | None = None,
    resolver: Callable[..., list[tuple]] = socket.getaddrinfo,
) -> ValidatedDeveloperIdentity:
    """Validate a GitHub or Linear personal credential and return its identity."""
    allowed = _METHODS.get(connector_id)
    if allowed is None:
        raise DeveloperCredentialError("This connector does not support personal-token setup in Code.")
    if method not in allowed:
        raise DeveloperCredentialError("Choose a supported connection method.")

    egress = PinnedHostTransport(
        transport or httpx.HTTPTransport(),
        resolver,
        pinned_transport_factory=(lambda: transport) if transport is not None else httpx.HTTPTransport,
    )
    client = egress.client(timeout=15.0, follow_redirects=False, trust_env=transport is None)
    try:
        if connector_id == "github":
            return _validate_github(values, client, egress)
        return _validate_linear(values, client)
    except PinnedHostError as exc:
        raise exc.reason from exc
    except httpx.HTTPError as exc:
        raise DeveloperProviderUnavailable(
            f"{connector_id.title()} could not be reached. Try again in a moment."
        ) from exc
    finally:
        client.close()


def _validate_github(
    values: dict[str, Any],
    client: httpx.Client,
    egress: PinnedHostTransport,
) -> ValidatedDeveloperIdentity:
    token = str(values.get("access_token") or "").strip()
    if not token:
        raise DeveloperCredentialError("Enter a GitHub personal access token.")

    base_url = str(values.get("base_url") or "https://github.com").strip().rstrip("/")
    parsed = urlparse(base_url)
    if (
        parsed.scheme != "https"
        or not parsed.hostname
        or parsed.username
        or parsed.password
        or parsed.query
        or parsed.fragment
    ):
        raise DeveloperCredentialError("Enter a valid GitHub base URL.")

    if base_url.casefold() != "https://github.com":
        egress.pin(parsed.hostname)

    api_url = "https://api.github.com" if base_url.casefold() == "https://github.com" else f"{base_url}/api/v3"
    response = client.get(
        f"{api_url}/user",
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        },
    )
    if response.status_code in {401, 403}:
        raise DeveloperCredentialError("GitHub rejected this token. Check the token and its repository access.")
    if response.is_error:
        raise DeveloperProviderUnavailable(f"GitHub returned HTTP {response.status_code}. Try again in a moment.")

    payload = _json_object(response, "GitHub")
    identity = str(payload.get("email") or payload.get("login") or "").strip()
    if not identity:
        raise DeveloperCredentialError("GitHub accepted the token but did not return an account identity.")
    return ValidatedDeveloperIdentity(account_email=identity)


def require_public_host(
    hostname: str,
    resolver: Callable[..., list[tuple]],
) -> list[ipaddress.IPv4Address | ipaddress.IPv6Address]:
    """Reject private/custom GitHub endpoints before attaching credentials; return the validated addresses."""

    try:
        literal = ipaddress.ip_address(hostname)
        addresses = [literal]
    except ValueError:
        try:
            records = resolver(hostname, 443, type=socket.SOCK_STREAM)
        except OSError as exc:
            raise DeveloperProviderUnavailable(
                "The GitHub Enterprise host could not be resolved. Check the base URL."
            ) from exc
        addresses = []
        for record in records:
            try:
                addresses.append(ipaddress.ip_address(record[4][0]))
            except (IndexError, ValueError):
                continue
    if not addresses or any(not address.is_global for address in addresses):
        raise DeveloperCredentialError(
            "GitHub Enterprise must use a publicly routable HTTPS host."
        )
    return addresses


class PinnedHostError(httpx.ConnectError):
    """A pinned host failed the public-address check when the request resolved it."""

    def __init__(self, reason: Exception, request: httpx.Request) -> None:
        super().__init__(str(reason), request=request)
        self.reason = reason


class PinnedHostTransport(httpx.BaseTransport):
    """Connect pinned hosts to an address validated at resolution time.

    Resolving once and connecting to the validated address closes the window
    in which a host that passed the public-address check rebinds to a private
    one before the connection is made. The original hostname is kept for the
    ``Host`` header and for SNI/certificate verification.
    """

    def __init__(
        self,
        inner: httpx.BaseTransport,
        resolver: Callable[..., list[tuple]],
        hosts: set[str] | None = None,
        pinned_transport_factory: Callable[[], httpx.BaseTransport] | None = None,
    ) -> None:
        self._inner = inner
        self._resolver = resolver
        self._hosts = set() if hosts is None else hosts
        self._pinned_transport_factory = pinned_transport_factory or (lambda: inner)
        self._pinned_transports: dict[str, httpx.BaseTransport] = {}
        self._transport_lock = threading.Lock()
        self._closed = False

    def pin(self, host: str) -> None:
        self._hosts.add(host)

    def client(self, *, trust_env: bool = True, **options: Any) -> httpx.Client:
        """Build a client whose environment-proxy routes pin the same hosts as the direct one."""
        proxies = get_environment_proxies() if trust_env else {}
        mounts = {
            pattern: None if proxy is None else PinnedHostTransport(
                httpx.HTTPTransport(proxy=proxy),
                self._resolver,
                self._hosts,
                pinned_transport_factory=lambda proxy=proxy: httpx.HTTPTransport(proxy=proxy),
            )
            for pattern, proxy in proxies.items()
        }
        return httpx.Client(transport=self, mounts=mounts, **options)

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        host = (request.url.host or "").casefold()
        if host not in self._hosts:
            return self._inner.handle_request(request)
        try:
            addresses = require_public_host(host, self._resolver)
        except (DeveloperCredentialError, DeveloperProviderUnavailable) as exc:
            raise PinnedHostError(exc, request) from exc
        transport = self._transport_for(host)
        for address in addresses[:-1]:
            try:
                return transport.handle_request(self._pinned(request, host, str(address)))
            except httpx.ConnectError:
                continue
        return transport.handle_request(self._pinned(request, host, str(addresses[-1])))

    def _transport_for(self, host: str) -> httpx.BaseTransport:
        # HTTP connection pools key connections by the rewritten IP origin.
        # Two Enterprise hosts can legitimately resolve to one address, but
        # they must never share a pool: a keep-alive connection authenticated
        # for one virtual host could otherwise carry the other host's token.
        with self._transport_lock:
            if self._closed:
                raise RuntimeError("Pinned host transport is closed")
            transport = self._pinned_transports.get(host)
            if transport is None:
                transport = self._pinned_transport_factory()
                self._pinned_transports[host] = transport
            return transport

    @staticmethod
    def _pinned(request: httpx.Request, host: str, address: str) -> httpx.Request:
        return httpx.Request(
            request.method,
            request.url.copy_with(host=address),
            headers=request.headers,
            stream=request.stream,
            extensions={**request.extensions, "sni_hostname": host},
        )

    def close(self) -> None:
        with self._transport_lock:
            if self._closed:
                return
            self._closed = True
            transports = [self._inner, *self._pinned_transports.values()]
            self._pinned_transports.clear()
        closed: set[int] = set()
        for transport in transports:
            identity = id(transport)
            if identity in closed:
                continue
            closed.add(identity)
            transport.close()


def _validate_linear(values: dict[str, Any], client: httpx.Client) -> ValidatedDeveloperIdentity:
    token = str(values.get("api_key") or "").strip()
    if not token:
        raise DeveloperCredentialError("Enter a Linear personal API key.")

    response = client.post(
        "https://api.linear.app/graphql",
        headers={"Authorization": token, "Content-Type": "application/json"},
        json={"query": "query CodeConnectorViewer { viewer { id name email } }"},
    )
    if response.status_code in {401, 403}:
        raise DeveloperCredentialError("Linear rejected this API key. Check the key and try again.")
    if response.is_error:
        raise DeveloperProviderUnavailable(f"Linear returned HTTP {response.status_code}. Try again in a moment.")

    payload = _json_object(response, "Linear")
    if payload.get("errors"):
        raise DeveloperCredentialError("Linear rejected this API key. Check the key and try again.")
    viewer = (payload.get("data") or {}).get("viewer") or {}
    identity = str(viewer.get("email") or viewer.get("name") or viewer.get("id") or "").strip()
    if not identity:
        raise DeveloperCredentialError("Linear accepted the key but did not return an account identity.")
    return ValidatedDeveloperIdentity(account_email=identity)


def _json_object(response: httpx.Response, provider: str) -> dict[str, Any]:
    try:
        payload = response.json()
    except ValueError as exc:
        raise DeveloperProviderUnavailable(f"{provider} returned an unreadable response. Try again in a moment.") from exc
    if not isinstance(payload, dict):
        raise DeveloperProviderUnavailable(f"{provider} returned an unreadable response. Try again in a moment.")
    return payload
