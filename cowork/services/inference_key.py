"""Keeps a turn key for the desktop's mounted organization.

Auth bills a session token against whatever organization Keycloak has active,
so a switch on another device would move this desktop's billing. A turn key
pins the organization. It is re-minted on a token hand-over (about every nine
minutes) once it has less than 45 minutes left.
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

# Auth's bound for a desktop key; an older anton's scratchpad keeps the key it was built with.
_TTL = timedelta(minutes=60)
# More than one missed hand-over of headroom, so a late push never finds it expired.
_REMINT_WHEN_LEFT = timedelta(minutes=45)
# Refusals a retry can't fix: a bad body (e.g. a TTL past auth's bound), not a
# member, no product.execute, or the key cap.
_PERMANENT_REFUSALS = {400, 403, 409}
# The whole mint, lock wait included. The hand-over waits on it and the desktop
# gives that PUT 10s; past this the hand-over confirms and the retry takes over.
_MINT_BUDGET_S = 6.0
_CONNECT_S = 1.5
_REVOKE_TIMEOUT_S = 3.0
# A failed mint stops the desktop's LLM calls, so retry well before the next hand-over,
# backing off while auth stays down.
_RETRY_S = 60.0
_RETRY_MAX_S = 600.0
_retry_delay = _RETRY_S
_retry_timer: threading.Timer | None = None
# The busy-lock path schedules outside _mint_lock, so the timer has its own.
_retry_lock = threading.Lock()
# Every key installed by this process, so sign-out can revoke the ones a scratchpad may still hold.
_installed: dict[str, InferenceKey] = {}
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
        _schedule_retry()
        _drop_key_unless_for(organization_id)
        return
    try:
        _refresh(token, organization_id, deadline)
    except Exception as exc:
        # A failed settings read must not fail the hand-over or end the retries.
        # Fields, not %s args: the structured logger drops positional values.
        logger.warning(
            "Could not refresh the turn key",
            extra={"event": "turn_key_refresh_failed", "organization_id": organization_id, "error_type": type(exc).__name__},
        )
        _schedule_retry()
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
        if response.status_code in _PERMANENT_REFUSALS and (response.status_code == 400 or _code(response)):
            # A retry a minute later gets the same answer; the next hand-over tries again.
            logger.warning(
                "Auth refused a turn key",
                extra={
                    "event": "turn_key_refused",
                    "organization_id": organization_id,
                    "status_code": response.status_code,
                    "reason": _code(response),
                },
            )
            _drop_key_unless_for(organization_id)
            return
        response.raise_for_status()
        key = response.json()["key"]
    except Exception as exc:
        logger.warning(
            "Could not mint a turn key",
            extra={
                "event": "turn_key_mint_failed",
                "organization_id": organization_id,
                "error_type": type(exc).__name__,
                "status_code": getattr(getattr(exc, "response", None), "status_code", None),
            },
        )
        _schedule_retry()
        _drop_key_unless_for(organization_id)
        return

    minted = InferenceKey(
        value=key,
        organization_id=organization_id,
        instance_id=instance_id,
        expires_at=expires_at,
        workspace_id=workspace_id,
    )
    # Sign-out doesn't wait for this lock, so the hand-over may be gone by now.
    if (
        not runtime_credential.get_minds_credential()
        or runtime_credential.get_organization() != organization_id
    ):
        revoke_inference_key(token, minted)
        return
    _installed[minted.instance_id] = minted
    runtime_credential.set_inference_key(minted)
    # Sign-out doesn't take this lock either: one that cleared the hand-over meanwhile
    # may have missed this key, so revoke it here.
    if not runtime_credential.get_minds_credential():
        runtime_credential.set_inference_key(None)
        revoke_inference_key(token, minted)
        return
    _reset_retry_backoff()


def _drop_key_unless_for(organization_id: str) -> None:
    # A key for another org or workspace would bill the wrong one while the menu shows the new pick.
    current = runtime_credential.get_inference_key()
    if not current:
        return
    try:
        pick = desktop_workspace_pick()
    except Exception:
        # Unreadable pick: only the organization can be checked, and it's what billing hinges on.
        pick = current.workspace_id
    if (current.organization_id, current.workspace_id) != (organization_id, pick):
        runtime_credential.set_inference_key(None)


def _post(body: dict, token: str, deadline: float) -> httpx.Response:
    left = deadline - time.monotonic()
    if left <= 0:
        raise TimeoutError("turn-key mint ran out of time")
    # httpx restarts its read timeout on every chunk, so a slow-dripping auth could
    # outlast it; the join is what holds the whole request to the time left.
    # A daemon thread, so an abandoned request can't hold up shutdown.
    result: dict = {}
    # Whichever side finishes second owns the response: the caller once it has it,
    # else the worker, which revokes a key nobody will hold.
    handoff = threading.Lock()
    state = {"abandoned": False, "taken": False}

    def call() -> None:
        try:
            result["response"] = httpx.post(
                _turn_keys_url(),
                json=body,
                headers={"Authorization": f"Bearer {token}"},
                timeout=httpx.Timeout(left, connect=min(_CONNECT_S, left)),
            )
        except Exception as exc:
            result["error"] = exc
            return
        with handoff:
            orphaned = state["abandoned"] and not state["taken"]
        if orphaned and result["response"].is_success:
            _revoke_instance(token, body["instance_id"])

    thread = threading.Thread(target=call, daemon=True)
    thread.start()
    thread.join(left)
    with handoff:
        state["taken"] = not thread.is_alive()
        state["abandoned"] = not state["taken"]
    if state["abandoned"]:
        raise TimeoutError("turn-key mint ran out of time")
    if "error" in result:
        raise result["error"]
    return result["response"]


def _code(response: httpx.Response) -> str | None:
    try:
        return response.json().get("code")
    except Exception:
        return None


def _reset_retry_backoff() -> None:
    global _retry_delay
    with _retry_lock:
        _retry_delay = _RETRY_S


def _schedule_retry() -> None:
    global _retry_timer, _retry_delay
    with _retry_lock:
        if _retry_timer:
            _retry_timer.cancel()
        _retry_timer = threading.Timer(_retry_delay, _retry)
        _retry_delay = min(_retry_delay * 2, _RETRY_MAX_S)
        _retry_timer.daemon = True
        _retry_timer.start()


def _cancel_retry() -> None:
    global _retry_timer, _retry_delay
    with _retry_lock:
        if _retry_timer:
            _retry_timer.cancel()
        _retry_timer = None
        _retry_delay = _RETRY_S


def _retry() -> None:
    # The latest hand-over's token and organization; nothing if signed out since.
    token = runtime_credential.get_minds_credential()
    organization_id = runtime_credential.get_organization()
    if token and organization_id:
        refresh_inference_key(token, organization_id)


def revoke_inference_key(token: str, key: InferenceKey) -> None:
    """Best effort: an unrevoked key still expires on its own."""
    _installed.pop(key.instance_id, None)
    _revoke_instance(token, key.instance_id)


def revoke_all_inference_keys(token: str) -> None:
    """Sign-out: revoke every unexpired key this process installed, not just the held one.

    In parallel and within one revoke timeout, so the sign-out PUT stays inside the desktop's 10s.
    """
    _cancel_retry()
    now = datetime.now(timezone.utc)
    live = []
    while _installed:
        # Popped one at a time: a mint installing meanwhile is revoked by its own post-install check.
        _, key = _installed.popitem()
        if key.expires_at > now:
            live.append(key)
    threads = [
        threading.Thread(target=_revoke_instance, args=(token, key.instance_id), daemon=True)
        for key in live
    ]
    for thread in threads:
        thread.start()
    deadline = time.monotonic() + _REVOKE_TIMEOUT_S + 0.5
    for thread in threads:
        thread.join(max(0.0, deadline - time.monotonic()))


def _revoke_instance(token: str, instance_id: str) -> None:
    try:
        response = httpx.delete(
            f"{_turn_keys_url()}{instance_id}/",
            headers={"Authorization": f"Bearer {token}"},
            timeout=_REVOKE_TIMEOUT_S,
        )
    except Exception as exc:
        logger.warning(
            "Could not revoke a turn key",
            extra={"event": "turn_key_revoke_failed", "instance_id": instance_id, "error_type": type(exc).__name__},
        )
        return
    if not response.is_success:
        # It stays live until it expires; worth seeing, since a scratchpad may still hold it.
        logger.warning(
            "Auth refused to revoke a turn key",
            extra={"event": "turn_key_revoke_refused", "instance_id": instance_id, "status_code": response.status_code},
        )
