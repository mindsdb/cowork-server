"""PostHog-specific connection helpers.

Project discovery happens before the generic credential probe: users know their
PostHog project by name, while the downstream engine needs its numeric ID.
"""
from __future__ import annotations

from dataclasses import dataclass
from urllib.parse import urlparse

import httpx


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
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise PostHogDiscoveryError("Choose a valid PostHog host before finding projects.")
    return selected


async def discover_projects(*, personal_api_key: object, host: object, custom_host: object = None) -> list[PostHogProject]:
    """Return accessible projects without exposing credentials or upstream bodies."""
    api_key = str(personal_api_key or "").strip()
    if not api_key:
        raise PostHogDiscoveryError("Enter your PostHog personal API key before finding projects.")
    base_url = resolve_host(host, custom_host)
    try:
        async with httpx.AsyncClient(timeout=15.0, follow_redirects=True) as client:
            response = await client.get(
                f"{base_url}/api/projects/",
                headers={"Authorization": f"Bearer {api_key}"},
            )
    except httpx.HTTPError as exc:
        raise PostHogDiscoveryError("Could not reach PostHog. Check the selected host and try again.") from exc

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
