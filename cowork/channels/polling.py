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

# Pause before retrying after a poll cycle raises, so a persistent error
# (network down, bad token) doesn't spin a tight failure loop.
_POLL_ERROR_BACKOFF_S = 3.0


class PollManager:
    """Runs long-poll ingress loops for channels that support polling, feeding
    events through the same ``intake_events`` path as the webhook routes.

    This is the tunnel-free ingress mode: when no public webhook URL is
    configured, the server pulls updates from the platform itself instead of
    waiting for the platform to POST to a webhook it cannot reach (localhost).
    A channel opts in simply by exposing an ``async poll(*, offset)`` method on
    its live adapter (duck-typed, like the other optional bridge hooks)."""

    def __init__(self, *, sink: InboundSink) -> None:
        self._sink = sink
        self._tasks: dict[str, asyncio.Task[None]] = {}

    def is_polling(self, channel_type: str) -> bool:
        task = self._tasks.get(channel_type)
        return task is not None and not task.done()

    async def start(self, channel_type: str, bridge: Any) -> None:
        """Begin polling a channel. No-op if the adapter can't poll or a loop is
        already running for it."""
        if not callable(getattr(bridge, "poll", None)) or self.is_polling(channel_type):
            return
        self._tasks[channel_type] = asyncio.create_task(self._loop(channel_type, bridge))
        log.info("channel %s: started server-side polling", channel_type)

    async def stop(self, channel_type: str) -> None:
        task = self._tasks.pop(channel_type, None)
        if task is None:
            return
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task
        log.info("channel %s: stopped server-side polling", channel_type)

    async def stop_all(self) -> None:
        for channel_type in list(self._tasks):
            await self.stop(channel_type)

    async def _loop(self, channel_type: str, bridge: Any) -> None:
        offset: int | None = None
        while True:
            try:
                events, offset = await bridge.poll(offset=offset)
            except asyncio.CancelledError:
                raise
            except Exception:
                log.exception("channel %s: poll cycle failed; backing off", channel_type)
                await asyncio.sleep(_POLL_ERROR_BACKOFF_S)
                continue
            if events:
                log.info("channel %s: poll fetched %d event(s)", channel_type, len(events))
                intake_events(channel_type, bridge, events, sink=self._sink)


async def sync_channel_polling(poller: PollManager | None, adapters: Any, channel_type: str) -> None:
    """Reconcile a channel's polling state after any lifecycle change. Polls iff
    the channel is active, its adapter supports polling, and no public base URL
    is configured (so there is no webhook to receive instead); otherwise stops.
    Idempotent — safe to call after setup/teardown/reload/config changes."""
    if poller is None or adapters is None:
        return
    adapter = adapters.get(channel_type)
    has_public_url = bool((get_app_settings().public_base_url or "").strip())
    if adapter is not None and not has_public_url and callable(getattr(adapter, "poll", None)):
        await poller.start(channel_type, adapter)
    else:
        await poller.stop(channel_type)
