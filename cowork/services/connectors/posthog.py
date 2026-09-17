"""PostHog-specific connection helpers.

Project discovery happens before the generic credential probe: users know their
PostHog project by name, while the downstream engine needs its numeric ID.

Discovery forwards the caller's personal API key, so the destination is the
security boundary: a host the caller chooses freely turns this into a
credentialed fetch against anything the server can reach. In org mode it
therefore reaches ``CLOUD_ORIGINS`` only. A self-hosted host stays
desktop-only until a guarded custom-host path is reviewed, and the connector
form still offers "Self-hosted (enter URL)" in cloud, so the refusal message
is the only thing that tells a caller no URL will do.
"""
from __future__ import annotations

import asyncio
import socket
from collections.abc import Callable
from dataclasses import dataclass
from urllib.parse import urlparse

import httpx

from cowork.common.settings.app_settings import get_app_settings
from cowork.services.connectors.egress import (
    EgressHostNotPublic,
    EgressHostUnresolved,
    vetted_public_addresses,
)

#: The origins org mode may reach. Server-owned: a caller selects one, it is
#: not a value a caller supplies. tests/test_posthog_discovery_org_mode.py
#: keeps this in step with the spec's `host` options.
CLOUD_ORIGINS: tuple[str, ...] = ("https://us.posthog.com", "https://eu.posthog.com")

#: The resolver a request through the route uses, since a route cannot pass one.
_RESOLVER: Callable[..., list[tuple]] = socket.getaddrinfo

_TIMEOUT_SECONDS = 15.0
_CLOUD_ONLY = (
    "Cowork cloud reaches PostHog US Cloud or EU Cloud only. "
    "Use the desktop app to connect a self-hosted PostHog host."
)
_UNREACHABLE = "Could not reach PostHog. Check the selected host and try again."


class PostHogDiscoveryError(Exception):
    """A user-safe PostHog project discovery failure."""


@dataclass(frozen=True)
class PostHogProject:
    id: str
    name: str


def resolve_host(host: object, custom_host: object = None) -> str:
    """Validate and normalize the selected cloud or self-hosted PostHog host."""
    selected = str(custom_host if host == "custom" else host or "").strip().rstrip("/")
    parsed = urlparse(selected)
    if parsed.scheme != "https" or not parsed.netloc or parsed.username or parsed.password:
        raise PostHogDiscoveryError("Choose a valid HTTPS PostHog host before finding projects.")
    return selected


def cloud_origin(selected: str) -> str:
    """Return the allowlisted origin ``selected`` names, or refuse it.

    Compares the whole origin, so a port, a path, userinfo or a lookalike host
    cannot match, and returns the server's own constant, so nothing the caller
    sent reaches the request. Non-ASCII is refused rather than folded: casefold
    maps U+017F to "s" and U+212A to "k", which would make two different
    hostnames compare equal.
    """
    if not selected.isascii():
        raise PostHogDiscoveryError(_CLOUD_ONLY)
    lowered = selected.lower()
    for origin in CLOUD_ORIGINS:
        if lowered == origin:
            return origin
    raise PostHogDiscoveryError(_CLOUD_ONLY)


async def _fetch_cloud(
    origin: str,
    api_key: str,
    *,
    transport: httpx.AsyncBaseTransport | None,
    resolver: Callable[..., list[tuple]] | None,
) -> httpx.Response:
    """Request an allowlisted origin at an address vetted as globally routable.

    Resolving once and dialing the address as a literal is what closes the
    window between the check and the connection. The hostname stays on ``Host``
    and on SNI, so the certificate is still verified against it. Addresses are
    tried in the resolver's order, because a dual-stack answer on a host with
    no egress for one family would otherwise fail outright.
    """
    url = httpx.URL(f"{origin}/api/projects/")
    hostname = url.host
    try:
        addresses = await asyncio.to_thread(vetted_public_addresses, hostname, resolver or _RESOLVER)
    except (EgressHostUnresolved, EgressHostNotPublic) as exc:
        raise PostHogDiscoveryError(_UNREACHABLE) from exc

    headers = {"Authorization": f"Bearer {api_key}", "Host": hostname}
    extensions = {"sni_hostname": hostname}
    # trust_env stays off: an environment proxy tunnels to the pinned address
    # and httpcore verifies the certificate against the tunnel's own origin,
    # ignoring sni_hostname, so a proxy would defeat the pin instead of
    # carrying it. No deployment sets one.
    async with httpx.AsyncClient(
        timeout=_TIMEOUT_SECONDS,
        follow_redirects=False,
        trust_env=False,
        transport=transport,
    ) as client:
        for address in addresses[:-1]:
            try:
                return await client.get(url.copy_with(host=str(address)), headers=headers, extensions=extensions)
            except httpx.ConnectError:
                continue
        return await client.get(url.copy_with(host=str(addresses[-1])), headers=headers, extensions=extensions)


async def _fetch_direct(
    base_url: str,
    api_key: str,
    *,
    transport: httpx.AsyncBaseTransport | None,
) -> httpx.Response:
    """Request the selected host as entered, for desktop's self-hosted PostHog."""
    async with httpx.AsyncClient(timeout=_TIMEOUT_SECONDS, follow_redirects=False, transport=transport) as client:
        return await client.get(
            f"{base_url}/api/projects/",
            headers={"Authorization": f"Bearer {api_key}"},
        )


async def discover_projects(
    *,
    personal_api_key: object,
    host: object,
    custom_host: object = None,
    transport: httpx.AsyncBaseTransport | None = None,
    resolver: Callable[..., list[tuple]] | None = None,
) -> list[PostHogProject]:
    """Return accessible projects without exposing credentials or upstream bodies."""
    api_key = str(personal_api_key or "").strip()
    if not api_key:
        raise PostHogDiscoveryError("Enter your PostHog personal API key before finding projects.")
    base_url = resolve_host(host, custom_host)
    org_mode = get_app_settings().tenancy_mode == "org"
    if org_mode:
        base_url = cloud_origin(base_url)
    try:
        if org_mode:
            response = await _fetch_cloud(base_url, api_key, transport=transport, resolver=resolver)
        else:
            response = await _fetch_direct(base_url, api_key, transport=transport)
    except httpx.HTTPError as exc:
        raise PostHogDiscoveryError(_UNREACHABLE) from exc

    if response.status_code in {401, 403}:
        raise PostHogDiscoveryError("PostHog rejected that personal API key. Check its access and try again.")
    if response.status_code >= 400:
        raise PostHogDiscoveryError("PostHog could not list projects for that host. Try again or enter a project ID manually.")
    try:
        payload = response.json()
    except ValueError as exc:
        raise PostHogDiscoveryError("PostHog returned an invalid project list. Try again or enter a project ID manually.") from exc

    entries = payload.get("results", payload) if isinstance(payload, dict) else payload
    if not isinstance(entries, list):
        raise PostHogDiscoveryError("PostHog returned an invalid project list. Try again or enter a project ID manually.")
    projects = []
    for entry in entries:
        if not isinstance(entry, dict) or entry.get("id") is None:
            continue
        project_id = str(entry["id"]).strip()
        if project_id:
            projects.append(PostHogProject(id=project_id, name=str(entry.get("name") or f"Project {project_id}")))
    return projects
