"""Storing a member's custom API connection in auth, after its check passed.

auth accepts a custom API connection only from this server, on its internal
route, with the producer credential and the member's own credential
forwarded: the member is whoever that credential names, and the organization
this server acted in must be the one active on it. Nothing here logs or
raises a submitted value; a refusal carries auth's code only.
"""

from __future__ import annotations

from typing import Any

import httpx

from cowork.common.settings.app_settings import TurnQueueSettings

_TIMEOUT_SECONDS = 10.0


class CustomConnectionUnavailable(RuntimeError):
    """auth could not be asked, or answered with something this server cannot read."""


class CustomConnectionRefused(RuntimeError):
    """auth refused the connection; ``code`` is its machine-readable reason."""

    def __init__(self, code: str, status_code: int) -> None:
        super().__init__(code)
        self.code = code
        self.status_code = status_code


def _producer_headers(settings: TurnQueueSettings, credential: str) -> dict[str, str]:
    """The producer credential plus the member's own; refuses when either is missing."""
    if not settings.auth_internal_base_url or not settings.datasource_producer_key_id:
        raise CustomConnectionUnavailable("producer identity is not configured")
    if not settings.datasource_producer_key:
        raise CustomConnectionUnavailable("producer identity is not configured")
    if not credential:
        raise CustomConnectionUnavailable("the caller presented no credential")
    return {
        "X-Datasource-Service-Key-Id": settings.datasource_producer_key_id,
        "X-Datasource-Service-Key": settings.datasource_producer_key,
        "Authorization": credential,
    }


async def create_custom_connection(
    *,
    credential: str,
    organization_id: str,
    payload: dict[str, Any],
    dry_run: bool,
    settings: TurnQueueSettings | None = None,
) -> dict[str, Any]:
    """Ask auth to store the member's verified connection, or with ``dry_run`` to check it would.

    Args:
        credential: The member's own Authorization header value.
        organization_id: The organization this server acted in.
        payload: ``connector_id``, ``name``, ``fields`` and ``secure_keys``.
        dry_run: Run auth's checks and store nothing.

    Returns:
        auth's answer: the stored connection's metadata, or ``{"dry_run": true}``.

    Raises:
        CustomConnectionRefused: auth refused it; ``code`` says why.
        CustomConnectionUnavailable: auth could not be asked or its answer read.
    """
    settings = settings or TurnQueueSettings()
    headers = _producer_headers(settings, credential)
    url = f"{settings.auth_internal_base_url.rstrip('/')}/internal/custom-connections/create/"
    body = {**payload, "organization_id": organization_id, "dry_run": dry_run}
    try:
        async with httpx.AsyncClient(timeout=_TIMEOUT_SECONDS, follow_redirects=False) as client:
            response = await client.post(url, json=body, headers=headers)
    except (httpx.HTTPError, TimeoutError) as exc:
        raise CustomConnectionUnavailable("auth is unreachable") from exc
    try:
        answer = response.json()
    except ValueError as exc:
        raise CustomConnectionUnavailable("auth answered with a body this server cannot read") from exc
    if response.status_code >= 400:
        code = answer.get("code") if isinstance(answer, dict) else None
        raise CustomConnectionRefused(str(code)[:64] if isinstance(code, str) else "uncoded", response.status_code)
    if not isinstance(answer, dict):
        raise CustomConnectionUnavailable("auth answered with a body this server cannot read")
    return answer
