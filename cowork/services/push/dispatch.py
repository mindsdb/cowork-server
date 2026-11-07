"""Send a push to every phone a person registered, without ever holding up
the work that caused it.

``spawn`` runs a push as a detached task, kept in ``_background_tasks`` so it
is not collected mid-send, and ``drain_background_tasks`` waits for the ones
still going at shutdown, as cowork.channels.webhooks does for inbound
messages. ``notify_user`` itself never raises: a push that fails is logged
and dropped.

Push runs only in org mode (COWORK_TENANCY_MODE=org), with
COWORK_PUSH_ENABLED on and at least one platform's credentials set. A
platform whose credentials are missing skips its devices.
"""
from __future__ import annotations

import asyncio
import logging
from collections.abc import Coroutine
from dataclasses import dataclass
from functools import partial
from typing import Any

from cowork.common.settings.app_settings import get_app_settings, get_push_settings
from cowork.db.scoped import SYSTEM_SCOPE
from cowork.db.units import run_db
from cowork.services.push import devices as device_store
from cowork.services.push.apns import ApnsSender, Delivery, read_key
from cowork.services.push.devices import Device
from cowork.services.push.fcm import FcmSender, read_service_account
from cowork.services.push.payloads import PushPayload

logger = logging.getLogger(__name__)

# Bounds one person's whole send, every device included.
_NOTIFY_TIMEOUT_SECONDS = 20.0


@dataclass(frozen=True)
class Senders:
    apns: ApnsSender | None = None
    fcm: FcmSender | None = None

    def __bool__(self) -> bool:
        return self.apns is not None or self.fcm is not None


_senders: Senders | None = None
_background_tasks: set[asyncio.Task[Any]] = set()


def _build_senders() -> Senders:
    settings = get_push_settings()
    apns = fcm = None
    key = settings.apns_key_p8.get_secret_value() if settings.apns_key_p8 else ""
    if key.strip() and settings.apns_key_id and settings.apns_team_id:
        try:
            apns = ApnsSender(
                key_pem=read_key(key), key_id=settings.apns_key_id,
                team_id=settings.apns_team_id, topic=settings.apns_topic,
            )
        except ValueError as exc:
            logger.warning(str(exc), extra={"event": "push_apns_misconfigured"})
    account = settings.fcm_service_account_json.get_secret_value() if settings.fcm_service_account_json else ""
    if account.strip():
        try:
            fcm = FcmSender(read_service_account(account))
        except ValueError as exc:
            logger.warning(str(exc), extra={"event": "push_fcm_misconfigured"})
    return Senders(apns=apns, fcm=fcm)


def get_senders() -> Senders:
    global _senders
    if _senders is None:
        _senders = _build_senders()
    return _senders


def set_senders(senders: Senders | None) -> None:
    """Replace the senders (tests); None rebuilds them from settings on next use."""
    global _senders
    _senders = senders


def push_enabled() -> bool:
    """Whether a push could go anywhere: org mode, the kill switch on, and
    some platform configured."""
    try:
        if get_app_settings().tenancy_mode != "org" or not get_push_settings().enabled:
            return False
        return bool(get_senders())
    except Exception:
        logger.warning("Could not read push settings", exc_info=True, extra={"event": "push_settings_unreadable"})
        return False


async def _deliver(senders: Senders, device: Device, payload: PushPayload) -> None:
    if device.platform == "ios":
        if senders.apns is None:
            return
        result = await senders.apns.send(device, payload)
        delivery = result.delivery
        if result.environment is not None:
            await run_db(
                partial(device_store.set_environment, token=device.token, environment=result.environment),
                scope=SYSTEM_SCOPE,
            )
    elif device.platform == "android":
        if senders.fcm is None:
            return
        delivery = await senders.fcm.send(device, payload)
    else:
        return
    if delivery == Delivery.DEAD:
        await run_db(partial(device_store.forget_token, token=device.token), scope=SYSTEM_SCOPE)
        logger.info("Removed a push device its platform no longer reaches", extra={"event": "push_device_removed"})


async def _deliver_logged(senders: Senders, device: Device, payload: PushPayload) -> None:
    try:
        await _deliver(senders, device, payload)
    except Exception:
        logger.warning(
            "Push to a device failed", exc_info=True,
            extra={"event": "push_send_failed", "platform": device.platform, "kind": str(payload.kind)},
        )


async def notify_user(org_id: str, user_id: str, payload: PushPayload) -> None:
    """Send ``payload`` to each phone ``user_id`` registered in ``org_id``.
    Never raises."""
    try:
        if not org_id or not user_id or not push_enabled():
            return
        senders = get_senders()
        async with asyncio.timeout(_NOTIFY_TIMEOUT_SECONDS):
            devices = await run_db(
                partial(device_store.devices_for, org_id=org_id, user_id=user_id), scope=SYSTEM_SCOPE,
            )
            if devices:
                await asyncio.gather(*(_deliver_logged(senders, device, payload) for device in devices))
    except Exception:
        logger.warning(
            "Push notification failed", exc_info=True,
            extra={"event": "push_notify_failed", "kind": str(payload.kind)},
        )


def spawn(coro: Coroutine[Any, Any, None]) -> asyncio.Task[Any] | None:
    """Run ``coro`` detached. Returns None (and closes it) when no event loop
    is running."""
    try:
        task = asyncio.get_running_loop().create_task(coro)
    except RuntimeError:
        coro.close()
        return None
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)
    return task


async def drain_background_tasks(*, timeout: float = 3.0) -> None:
    """Wait for pushes still sending. Called at shutdown; best effort."""
    tasks = list(_background_tasks)
    if not tasks:
        return
    try:
        await asyncio.wait_for(asyncio.gather(*tasks, return_exceptions=True), timeout=timeout)
    except asyncio.TimeoutError:
        logger.warning(
            "Push notifications still sending at shutdown",
            extra={"event": "push_shutdown_unfinished", "count": len(tasks)},
        )
