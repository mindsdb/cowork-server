from __future__ import annotations

import asyncio
import contextlib
import logging
from collections.abc import Awaitable, Callable
from typing import Any

from cowork.channels.webhooks import intake_events
from cowork.common.settings.app_settings import get_app_settings

log = logging.getLogger(__name__)

InboundSink = Callable[[str, Any], Awaitable[None]]

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
    (duck-typed, like the other optional bridge hooks)."""

    def __init__(self, *, sink: InboundSink) -> None:
        self._sink = sink
        self._tasks: dict[str, asyncio.Task[None]] = {}

    def is_running(self, channel_type: str) -> bool:
        task = self._tasks.get(channel_type)
        return task is not None and not task.done()

    async def start(self, channel_type: str, bridge: Any) -> None:
        """Begin ingress for a channel. No-op if the adapter can't ingest this
        way or a loop is already running for it."""
        if not self._can_ingest(bridge) or self.is_running(channel_type):
            return
        self._tasks[channel_type] = asyncio.create_task(self._loop(channel_type, bridge))
        log.info("channel %s: started background ingress", channel_type)

    async def stop(self, channel_type: str) -> None:
        task = self._tasks.pop(channel_type, None)
        if task is None:
            return
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task
        log.info("channel %s: stopped background ingress", channel_type)

    async def stop_all(self) -> None:
        for channel_type in list(self._tasks):
            await self.stop(channel_type)

    @staticmethod
    def _can_ingest(bridge: Any) -> bool:
        return callable(getattr(bridge, "stream_events", None)) or callable(getattr(bridge, "poll", None))

    async def _loop(self, channel_type: str, bridge: Any) -> None:
        if callable(getattr(bridge, "stream_events", None)):
            await self._stream_loop(channel_type, bridge)
        else:
            await self._poll_loop(channel_type, bridge)

    async def _stream_loop(self, channel_type: str, bridge: Any) -> None:
        # One stream_events() call is a single connection lifecycle; when it
        # ends (clean close or error) we back off and reconnect.
        failing = False
        while True:
            try:
                async for events in bridge.stream_events():
                    if failing:
                        log.info("channel %s: ingress stream recovered", channel_type)
                        failing = False
                    if events:
                        log.info("channel %s: stream delivered %d event(s)", channel_type, len(events))
                        intake_events(channel_type, bridge, events, sink=self._sink)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                failing = _log_loop_failure(channel_type, "ingress stream", exc, failing)
            await asyncio.sleep(_ERROR_BACKOFF_S)

    async def _poll_loop(self, channel_type: str, bridge: Any) -> None:
        offset: int | None = None
        failing = False
        while True:
            try:
                events, offset = await bridge.poll(offset=offset)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                failing = _log_loop_failure(channel_type, "poll cycle", exc, failing)
                await asyncio.sleep(_ERROR_BACKOFF_S)
                continue
            if failing:
                log.info("channel %s: poll cycle recovered", channel_type)
                failing = False
            if events:
                log.info("channel %s: poll fetched %d event(s)", channel_type, len(events))
                intake_events(channel_type, bridge, events, sink=self._sink)


def _log_loop_failure(channel_type: str, what: str, exc: Exception, failing: bool) -> bool:
    """Log an ingress-loop failure once per outage and return the new failing
    state. Transient transport errors (platform unreachable, DNS, timeout) are
    expected operational noise: emit one human-readable WARNING when a streak
    starts, then stay quiet until it recovers (logged by the caller). Anything
    unexpected still gets a full ERROR traceback every time so real bugs surface."""
    if isinstance(exc, (ConnectionError, TimeoutError)):
        if not failing:
            log.warning(
                "channel %s: %s failed (%s); retrying every %.0fs until it recovers",
                channel_type, what, exc, _ERROR_BACKOFF_S,
            )
        return True
    log.exception("channel %s: %s failed; backing off", channel_type, what)
    return failing


async def sync_channel_ingress(manager: IngressManager | None, adapters: Any, channel_type: str) -> None:
    """Reconcile a channel's background ingress after any lifecycle change.

    A streaming adapter (e.g. Discord Gateway) runs whenever the channel is
    active — the Gateway is its primary inbound path, independent of any
    webhook. A polling adapter (e.g. Telegram) runs only when no public base URL
    is configured, since otherwise the webhook owns ingress and the two are
    mutually exclusive at the platform. Idempotent."""
    if manager is None or adapters is None:
        return
    adapter = adapters.get(channel_type)
    if adapter is None:
        await manager.stop(channel_type)
        return
    if callable(getattr(adapter, "stream_events", None)):
        await manager.start(channel_type, adapter)
        return
    if callable(getattr(adapter, "poll", None)):
        has_public_url = bool((get_app_settings().public_base_url or "").strip())
        if has_public_url:
            await manager.stop(channel_type)
        else:
            await manager.start(channel_type, adapter)
        return
    await manager.stop(channel_type)
