"""Validation of a captured cloud datasource connection.

Auth stores the credential encrypted and marks the connection pending. Only a
probe moves it to verified or failed, and only this server may ask for one:
auth mints a single-use capability to the producer identity when the owner's
own credential is presented with it, and the gateway dials the database with
that capability and reports the verdict back to auth itself. Neither hop
returns a credential to this process, and the capability never leaves it.

Two addresses, on purpose. The mint is a cluster-only auth route reached with
the producer role key. The probe is a cluster-only gateway route reached by
Service name, because the public datasource ingress admits execution alone and
its identity gate would refuse a capability, which is not an API key.

Validation never fails the request that triggered it. The connection exists
either way, and one that could not be probed stays pending and can be retried.
"""

from __future__ import annotations

import logging
import uuid

import httpx

from cowork.common.settings.app_settings import TurnQueueSettings

logger = logging.getLogger(__name__)

#: The gateway's wire version for a probe body.
PROTOCOL_VERSION = 1
_MINT_TIMEOUT_SECONDS = 5.0
#: Longer than the mint: the gateway dials a customer database behind this,
#: under its own deadline, and answers once auth has recorded the verdict.
_PROBE_TIMEOUT_SECONDS = 45.0


class _ValidationUnavailable(Exception):
    """This deployment cannot validate, or a hop refused. Never leaves the module."""


def _producer_headers(settings: TurnQueueSettings) -> dict[str, str]:
    if not settings.auth_internal_base_url or not settings.datasource_producer_key_id:
        raise _ValidationUnavailable("producer identity is not configured")
    if not settings.datasource_producer_key:
        raise _ValidationUnavailable("producer identity is not configured")
    return {
        "X-Datasource-Service-Key-Id": settings.datasource_producer_key_id,
        "X-Datasource-Service-Key": settings.datasource_producer_key,
    }


async def _mint_capability(
    connection_id: int, credential: str, settings: TurnQueueSettings
) -> tuple[str, str, int]:
    """Ask auth for a single-use probe capability for one connection.

    Takes the caller's own credential as well as the producer key: auth issues
    the capability to the connection's owner, so a service key alone cannot ask
    for one.
    """
    headers = _producer_headers(settings)
    if not credential:
        raise _ValidationUnavailable("the caller presented no credential")
    headers["Authorization"] = credential
    url = f"{settings.auth_internal_base_url.rstrip('/')}/internal/datasources/connections/{connection_id}/probes/"
    try:
        async with httpx.AsyncClient(timeout=_MINT_TIMEOUT_SECONDS, follow_redirects=False) as client:
            response = await client.post(url, json={}, headers=headers)
    except (httpx.HTTPError, TimeoutError) as exc:
        raise _ValidationUnavailable("auth is unreachable") from exc
    if response.status_code >= 400:
        raise _ValidationUnavailable(f"auth refused the capability with {response.status_code}")
    try:
        body = response.json()
    except ValueError as exc:
        raise _ValidationUnavailable("auth answered the mint with a body this server cannot read") from exc
    probe_id = body.get("id") if isinstance(body, dict) else None
    capability = body.get("capability") if isinstance(body, dict) else None
    version = body.get("credential_version") if isinstance(body, dict) else None
    if not isinstance(probe_id, str) or not isinstance(capability, str) or not isinstance(version, int):
        raise _ValidationUnavailable("auth answered the mint without a usable capability")
    return probe_id, capability, version


async def _run_probe(
    probe_id: str, capability: str, credential_version: int, settings: TurnQueueSettings
) -> None:
    """Spend the capability on the gateway's fixed connection probe.

    Returning normally means the gateway answered, not that the database
    accepted the connection. The gateway records its own verdict with auth
    before it answers, including on the refusals a caller can act on, so a
    failed dial is a connection this server then re-reads rather than an
    attempt that never happened.
    """
    if not settings.datasource_gateway_base_url:
        raise _ValidationUnavailable("the datasource gateway address is not configured")
    url = f"{settings.datasource_gateway_base_url.rstrip('/')}/v1/datasources/probe"
    body = {
        "protocol_version": PROTOCOL_VERSION,
        "probe_id": probe_id,
        "credential_version": credential_version,
        "correlation_id": str(uuid.uuid4()),
    }
    try:
        async with httpx.AsyncClient(timeout=_PROBE_TIMEOUT_SECONDS, follow_redirects=False) as client:
            response = await client.post(url, json=body, headers={"Authorization": f"Bearer {capability}"})
    except (httpx.HTTPError, TimeoutError) as exc:
        raise _ValidationUnavailable("the datasource gateway is unreachable") from exc
    if response.status_code >= 400:
        logger.info("[datasources] probe %s answered %s", probe_id, response.status_code)


async def validate_connection(connection_id: int, credential: str, settings: TurnQueueSettings | None = None) -> bool:
    """Run one validation attempt for a connection, and say whether it ran.

    True means the gateway answered, so auth may have recorded a verdict and
    the caller should re-read the connection. False means the attempt could not
    be made at all, and the connection stands exactly as auth left it.
    """
    settings = settings or TurnQueueSettings()
    try:
        probe_id, capability, credential_version = await _mint_capability(connection_id, credential, settings)
        await _run_probe(probe_id, capability, credential_version, settings)
    except _ValidationUnavailable as exc:
        logger.warning("[datasources] connection %s was not validated: %s", connection_id, exc)
        return False
    return True
