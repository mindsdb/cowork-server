"""Validate credentials for Code's first-class developer connectors.

The generic direct-save endpoint is intentionally probe-free because its OAuth
callers have already exchanged and verified a token.  Personal tokens entered
from Code need a separate boundary: validate them with the provider before a
vault record is created, otherwise an invalid token would appear connected.
"""
from __future__ import annotations

import ipaddress
import socket
from dataclasses import dataclass
from typing import Any, Callable
from urllib.parse import urlparse

import httpx


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
    client: httpx.Client | None = None,
    resolver: Callable[..., list[tuple]] = socket.getaddrinfo,
) -> ValidatedDeveloperIdentity:
    """Validate a GitHub or Linear personal credential and return its identity."""
    allowed = _METHODS.get(connector_id)
    if allowed is None:
        raise DeveloperCredentialError("This connector does not support personal-token setup in Code.")
    if method not in allowed:
        raise DeveloperCredentialError("Choose a supported connection method.")

    owns_client = client is None
    active_client = client or httpx.Client(timeout=15.0, follow_redirects=False)
    try:
        if connector_id == "github":
            return _validate_github(values, active_client, resolver)
        return _validate_linear(values, active_client)
    except httpx.HTTPError as exc:
        raise DeveloperProviderUnavailable(
            f"{connector_id.title()} could not be reached. Try again in a moment."
        ) from exc
    finally:
        if owns_client:
            active_client.close()


def _validate_github(
    values: dict[str, Any],
    client: httpx.Client,
    resolver: Callable[..., list[tuple]],
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
        require_public_host(parsed.hostname, resolver)

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


def require_public_host(hostname: str, resolver: Callable[..., list[tuple]]) -> None:
    """Reject private/custom GitHub endpoints before attaching credentials."""

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
