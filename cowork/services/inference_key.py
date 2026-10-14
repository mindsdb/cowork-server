"""Keeps a turn key for the desktop's mounted organization.

Auth bills a session token against whatever organization Keycloak has active,
so a switch on another device would move this desktop's billing. A turn key
pins the organization. It is re-minted on a token hand-over (about every nine
minutes) once it has less than 20 minutes left.
"""

from __future__ import annotations

import logging
import threading
import time
import uuid
from datetime import datetime, timedelta, timezone

import httpx

from cowork.common.settings import runtime_credential
from cowork.common.settings.runtime_credential import InferenceKey

logger = logging.getLogger(__name__)

_TTL = timedelta(minutes=30)
# More than one missed hand-over of headroom, so a late push never finds it expired.
_REMINT_WHEN_LEFT = timedelta(minutes=20)
# The whole mint, lock wait included. The hand-over waits on it and the desktop
# gives that PUT 10s; past this the hand-over confirms and the retry takes over.
_MINT_BUDGET_S = 6.0
_CONNECT_S = 1.5
_REVOKE_TIMEOUT_S = 3.0
# A failed mint stops the desktop's LLM calls, so retry well before the next hand-over.
_RETRY_S = 60.0
_retry_timer: threading.Timer | None = None
# The hand-over, a workspace switch and the retry timer mint on different threads;
# without this a slower, older mint could install over a newer one.
_mint_lock = threading.Lock()


def _turn_keys_url() -> str:
    from cowork.common.settings.app_settings import default_minds_auth_host

    return f"{default_minds_auth_host().rstrip('/')}/v1/turn-keys/"


def desktop_workspace_pick() -> str | None:
    """The workspace picked in the desktop's menu; the console's pick never counts."""
    from cowork.common.settings.user_settings import get_user_settings

    return get_user_settings().hub_workspace_id or None


def refresh_inference_key(token: str, organization_id: str) -> None:
    """Mint a key for ``organization_id`` and the desktop's workspace pick, unless a fresh one is held.

    A failed mint keeps a still-valid key only for the same organization and
    workspace, and retries in a minute. The old key is never revoked here: the
    scratchpad and coding sessions keep using it until it expires.
    """
    deadline = time.monotonic() + _MINT_BUDGET_S
    if not _mint_lock.acquire(timeout=_MINT_BUDGET_S):
        # Another mint is running past the budget; don't hold up the hand-over for it.
        _drop_key_unless_for(organization_id)
        _schedule_retry()
        return
    try:
        _refresh(token, organization_id, deadline)
    finally:
        _mint_lock.release()


def _refresh(token: str, organization_id: str, deadline: float) -> None:
    current = runtime_credential.get_inference_key()
    workspace_id = desktop_workspace_pick()
    now = datetime.now(timezone.utc)
    if (
        current
        and current.organization_id == organization_id
        and current.workspace_id == workspace_id
        and current.expires_at - now > _REMINT_WHEN_LEFT
    ):
        return

    instance_id = f"desktop-{uuid.uuid4()}"
    # Auth stamps the real expiry on its own clock; this one only decides when to re-mint.
    expires_at = now + _TTL
    body = {
        "organization_id": organization_id,
        "instance_id": instance_id,
        "ttl_seconds": int(_TTL.total_seconds()),
    }
    if workspace_id:
        body["workspace_id"] = workspace_id
    try:
        response = _post(body, token, deadline)
        if workspace_id and _code(response) == "workspace_not_found":
            # A stale pick (grant removed, workspace deleted) bills the Default, as the menu shows.
            response = _post(
                {k: v for k, v in body.items() if k != "workspace_id"}, token, deadline
            )
        response.raise_for_status()
        key = response.json()["key"]
    except Exception as exc:
        logger.warning(
            "Could not mint a turn key for organization %s: %s", organization_id, exc
        )
        _drop_key_unless_for(organization_id)
        _schedule_retry()
        return

    runtime_credential.set_inference_key(
        InferenceKey(
            value=key,
            organization_id=organization_id,
            instance_id=instance_id,
            expires_at=expires_at,
            workspace_id=workspace_id,
        )
    )


def _drop_key_unless_for(organization_id: str) -> None:
    # A key for another org or workspace would bill the wrong one while the menu shows the new pick.
    current = runtime_credential.get_inference_key()
    if current and (current.organization_id, current.workspace_id) != (
        organization_id,
        desktop_workspace_pick(),
    ):
        runtime_credential.set_inference_key(None)


def _post(body: dict, token: str, deadline: float) -> httpx.Response:
    left = deadline - time.monotonic()
    if left <= 0:
        raise TimeoutError("turn-key mint ran out of time")
    # ponytail: httpx bounds each phase, not their sum, so a slow connect can run
    # about _CONNECT_S over; still well inside the desktop's 10s.
    return httpx.post(
        _turn_keys_url(),
        json=body,
        headers={"Authorization": f"Bearer {token}"},
        timeout=httpx.Timeout(left, connect=min(_CONNECT_S, left)),
    )


def _code(response: httpx.Response) -> str | None:
    try:
        return response.json().get("code")
    except Exception:
        return None


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
            timeout=_REVOKE_TIMEOUT_S,
        )
    except Exception as exc:
        logger.info("Could not revoke turn key %s: %s", key.instance_id, exc)
