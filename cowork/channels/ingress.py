from __future__ import annotations

import asyncio
import contextlib
import logging
import uuid
from collections.abc import Awaitable, Callable
from typing import Any

from cowork.channels import ingress_lease
from cowork.channels.webhooks import intake_events
from cowork.common.settings.app_settings import get_app_settings

log = logging.getLogger(__name__)

InboundSink = Callable[[str, Any, "str | None"], Awaitable[None]]

# Pause before reconnecting/retrying after an ingress loop ends or raises, so a
# persistent failure (network down, bad token) doesn't spin a tight loop.
_ERROR_BACKOFF_S = 3.0


class IngressManager:
    """Runs background ingress loops for channels that fetch their own inbound
    instead of receiving it via a public webhook. Two adapter shapes are
    supported — both feed the same ``intake_events`` path as the webhook routes,
    so dedupe/logging/runtime behaviour is identical regardless of source:

      - ``stream_events()`` — async-iterates batches of events over a persistent
        connection (e.g. the Discord Gateway websocket). Reconnected on exit.
      - ``poll(*, offset)`` — one request/response cycle returning
        ``(events, next_offset)`` (e.g. Telegram getUpdates long-poll).

    A channel opts in just by exposing one of these methods on its live adapter
    (duck-typed, like the other optional bridge hooks).

    Tasks are keyed by (channel_type, org_id). org_id=None is local/desktop —
    one task per channel_type, no lease, exactly today's behavior. A real
    org_id goes through a Redis lease (cowork.channels.ingress_lease) first,
    since multiple cowork-server replicas could otherwise all try to run the
    same org's Gateway connection at once."""

    def __init__(self, *, sink: InboundSink) -> None:
        self._sink = sink
        self._tasks: dict[tuple[str, str | None], asyncio.Task[None]] = {}
        self._renewals: dict[tuple[str, str | None], asyncio.Task[None]] = {}
        self._owner_id = uuid.uuid4().hex

    def is_running(self, channel_type: str, org_id: str | None = None) -> bool:
        task = self._tasks.get((channel_type, org_id))
        return task is not None and not task.done()

    def running_keys(self) -> list[tuple[str, str | None]]:
        return [key for key, task in self._tasks.items() if not task.done()]

    async def start(self, channel_type: str, bridge: Any, org_id: str | None = None) -> None:
        """Begin ingress for a channel, optionally scoped to one org. No-op if
        the adapter can't ingest this way, a loop is already running for this
        key, or (org mode) another replica already holds this org's lease."""
        if not self._can_ingest(bridge) or self.is_running(channel_type, org_id):
            return
        if org_id is not None:
            acquired = await ingress_lease.acquire(channel_type, org_id, self._owner_id)
            if not acquired:
                log.info("channel %s org %s: lease held by another replica", channel_type, org_id)
                return
        key = (channel_type, org_id)
        self._tasks[key] = asyncio.create_task(self._loop(channel_type, org_id, bridge))
        if org_id is not None:
            self._renewals[key] = asyncio.create_task(self._renew_loop(channel_type, org_id))
        log.info("channel %s: started background ingress (org=%s)", channel_type, org_id)

    async def stop(self, channel_type: str, org_id: str | None = None) -> None:
        key = (channel_type, org_id)
        renewal = self._renewals.pop(key, None)
        if renewal is not None:
            renewal.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await renewal
        task = self._tasks.pop(key, None)
        if task is None:
            return
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task
        if org_id is not None:
            await ingress_lease.release(channel_type, org_id, self._owner_id)
        log.info("channel %s: stopped background ingress (org=%s)", channel_type, org_id)

    async def stop_all(self) -> None:
        for channel_type, org_id in list(self._tasks):
            await self.stop(channel_type, org_id)

    @staticmethod
    def _can_ingest(bridge: Any) -> bool:
        return callable(getattr(bridge, "stream_events", None)) or callable(getattr(bridge, "poll", None))

    async def _renew_loop(self, channel_type: str, org_id: str) -> None:
        key = (channel_type, org_id)
        while True:
            await asyncio.sleep(ingress_lease.RENEW_INTERVAL_S)
            ok = await ingress_lease.renew(channel_type, org_id, self._owner_id)
            if not ok:
                log.warning("channel %s org %s: lease lost; stopping local ingress", channel_type, org_id)
                task = self._tasks.pop(key, None)
                if task is not None:
                    task.cancel()
                self._renewals.pop(key, None)
                return

    async def _loop(self, channel_type: str, org_id: str | None, bridge: Any) -> None:
        if callable(getattr(bridge, "stream_events", None)):
            await self._stream_loop(channel_type, org_id, bridge)
        else:
            await self._poll_loop(channel_type, org_id, bridge)

    async def _stream_loop(self, channel_type: str, org_id: str | None, bridge: Any) -> None:
        # One stream_events() call is a single connection lifecycle; when it
        # ends (clean close or error) we back off and reconnect.
        while True:
            try:
                async for events in bridge.stream_events():
                    if events:
                        log.info("channel %s: stream delivered %d event(s)", channel_type, len(events))
                        intake_events(channel_type, bridge, events, sink=self._sink, org_id=org_id)
            except asyncio.CancelledError:
                raise
            except Exception:
                log.exception("channel %s: ingress stream failed; reconnecting", channel_type)
            await asyncio.sleep(_ERROR_BACKOFF_S)

    async def _poll_loop(self, channel_type: str, org_id: str | None, bridge: Any) -> None:
        offset: int | None = None
        while True:
            try:
                events, offset = await bridge.poll(offset=offset)
            except asyncio.CancelledError:
                raise
            except Exception:
                log.exception("channel %s: poll cycle failed; backing off", channel_type)
                await asyncio.sleep(_ERROR_BACKOFF_S)
                continue
            if events:
                log.info("channel %s: poll fetched %d event(s)", channel_type, len(events))
                intake_events(channel_type, bridge, events, sink=self._sink, org_id=org_id)


async def sync_channel_ingress(
    manager: IngressManager | None, adapters: Any, channel_type: str, org_id: str | None = None
) -> None:
    """Reconcile a channel's background ingress after any lifecycle change.

    A streaming adapter (e.g. Discord Gateway) runs whenever the channel is
    active — the Gateway is its primary inbound path, independent of any
    webhook. A polling adapter (e.g. Telegram) runs only when no public base URL
    is configured, since otherwise the webhook owns ingress and the two are
    mutually exclusive at the platform. Idempotent."""
    if manager is None or adapters is None:
        return
    adapter = adapters.get(channel_type, org_id)
    if adapter is None:
        await manager.stop(channel_type, org_id)
        return
    if callable(getattr(adapter, "stream_events", None)):
        await manager.start(channel_type, adapter, org_id)
        return
    if callable(getattr(adapter, "poll", None)):
        has_public_url = bool((get_app_settings().public_base_url or "").strip())
        if has_public_url:
            await manager.stop(channel_type, org_id)
        else:
            await manager.start(channel_type, adapter, org_id)
        return
    await manager.stop(channel_type, org_id)
