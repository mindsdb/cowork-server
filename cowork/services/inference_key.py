"""Keeps a turn key for the desktop's mounted organization.

Auth bills a session token against whatever organization Keycloak has active,
so a switch on another device would move this desktop's billing. A turn key
pins the organization. It is re-minted on a token hand-over (about every nine
minutes) once it has less than 20 minutes left.
"""

from __future__ import annotations

import logging
import threading
import uuid
from datetime import datetime, timedelta, timezone

import httpx

from cowork.common.settings import runtime_credential
from cowork.common.settings.runtime_credential import InferenceKey

logger = logging.getLogger(__name__)

_TTL = timedelta(minutes=30)
# More than one missed hand-over of headroom, so a late push never finds it expired.
_REMINT_WHEN_LEFT = timedelta(minutes=20)
_TIMEOUT_S = 10.0
# A failed mint stops the desktop's LLM calls, so retry well before the next hand-over.
_RETRY_S = 60.0
_retry_timer: threading.Timer | None = None


def _turn_keys_url() -> str:
    from cowork.common.settings.app_settings import default_minds_auth_host

    return f"{default_minds_auth_host().rstrip('/')}/v1/turn-keys/"


def refresh_inference_key(token: str, organization_id: str) -> None:
    """Mint a key for ``organization_id`` unless a fresh one is already held.

    A failed mint keeps a still-valid key for the same organization and
    retries in a minute. The old key is never revoked here: the scratchpad and
    coding sessions keep using it until it expires.
    """
    current = runtime_credential.get_inference_key()
    now = datetime.now(timezone.utc)
    if (
        current
        and current.organization_id == organization_id
        and current.expires_at - now > _REMINT_WHEN_LEFT
    ):
        return

    instance_id = f"desktop-{uuid.uuid4()}"
    expires_at = now + _TTL
    try:
        response = httpx.post(
            _turn_keys_url(),
            json={
                "organization_id": organization_id,
                "instance_id": instance_id,
                "expiry_date": expires_at.isoformat(),
            },
            headers={"Authorization": f"Bearer {token}"},
            timeout=_TIMEOUT_S,
        )
        response.raise_for_status()
        key = response.json()["key"]
    except Exception as exc:
        logger.warning(
            "Could not mint a turn key for organization %s: %s", organization_id, exc
        )
        if current and current.organization_id != organization_id:
            runtime_credential.set_inference_key(None)
        _schedule_retry()
        return

    runtime_credential.set_inference_key(
        InferenceKey(
            value=key,
            organization_id=organization_id,
            instance_id=instance_id,
            expires_at=expires_at,
        )
    )


def _schedule_retry() -> None:
    global _retry_timer
    if _retry_timer:
        _retry_timer.cancel()
    _retry_timer = threading.Timer(_RETRY_S, _retry)
    _retry_timer.daemon = True
    _retry_timer.start()


def _retry() -> None:
    # The latest hand-over's token and organization; nothing if signed out since.
    token = runtime_credential.get_minds_credential()
    organization_id = runtime_credential.get_organization()
    if token and organization_id:
        refresh_inference_key(token, organization_id)


def revoke_inference_key(token: str, key: InferenceKey) -> None:
    """Best effort: an unrevoked key still expires on its own."""
    try:
        httpx.delete(
            f"{_turn_keys_url()}{key.instance_id}/",
            headers={"Authorization": f"Bearer {token}"},
            timeout=_TIMEOUT_S,
        )
    except Exception as exc:
        logger.info("Could not revoke turn key %s: %s", key.instance_id, exc)
