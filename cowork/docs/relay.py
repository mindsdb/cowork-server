"""Doc events between replicas, over Redis pub/sub.

Where several replicas serve one shared tree (``COWORK_STREAM_BACKEND=redis``),
an editor's stream and the write it should hear of can sit on different
replicas. Each replica runs one relay: it publishes this replica's hub events
to ``cowork:docs:<topic>`` and listens on ``cowork:docs:*`` for the others'.

Delivery is best effort. A message lost to a full queue, a Redis error or a
dropped subscription costs only latency: a doc stream also polls the doc
file's mtime (``cowork.api.v1.endpoints.docs._sse``). Desktop and self-hosted
servers keep the hub in-process and run no relay.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
from typing import Any

from redis.asyncio import Redis

from cowork.common.settings.app_settings import get_app_settings
from cowork.docs.hub import DocHub, hub
from cowork.streaming.backend import get_backend
from cowork.turnqueue.redis_client import get_redis, redis_deadline

logger = logging.getLogger(__name__)

CHANNEL_PREFIX = "cowork:docs:"
_QUEUE_MAX = 256
# A nudge that waits longer than this is worth less than the next mtime poll.
_PUBLISH_BUDGET_S = 2.0
_READ_TIMEOUT_S = 1.0
# get_message can answer None without waiting, as on a health-check reply, so a
# short pause keeps the read loop from spinning.
_IDLE_PAUSE_S = 0.05
_BACKOFF_MIN_S = 0.5
_BACKOFF_MAX_S = 10.0


def channel(topic: str) -> str:
    return CHANNEL_PREFIX + topic


class DocRelay:
    """One replica's link: a publisher task and a subscriber task on one client."""

    def __init__(self, doc_hub: DocHub, client: Redis, *, queue_max: int = _QUEUE_MAX) -> None:
        self._hub = doc_hub
        self._client = client
        self._queue_max = queue_max
        self._loop: asyncio.AbstractEventLoop | None = None
        self._queue: asyncio.Queue | None = None
        self._tasks: list[asyncio.Task] = []
        # Counters for logs and tests: messages read from Redis (this replica's own
        # included) and messages this replica never published.
        self.received = 0
        self.dropped = 0

    def start(self) -> None:
        """Start both tasks on the running loop and take the hub's events."""
        loop = asyncio.get_running_loop()
        self._loop = loop
        self._queue = asyncio.Queue(maxsize=self._queue_max)
        self._tasks = [
            loop.create_task(self._publish_loop(), name="doc-relay-publish"),
            loop.create_task(self._subscribe_loop(), name="doc-relay-subscribe"),
        ]
        self._hub.attach(self.offer)

    async def stop(self) -> None:
        self._hub.detach(self.offer)
        self._loop = None
        for task in self._tasks:
            task.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks = []

    def offer(self, message: dict[str, Any]) -> None:
        """Queue ``message`` for publishing; safe from any thread, never waits on Redis."""
        loop = self._loop
        if loop is None:
            return
        try:
            loop.call_soon_threadsafe(self._enqueue, message)
        except RuntimeError:  # the loop has closed: the server is stopping
            logger.debug("doc event not queued for the relay: its loop is closed")

    def _enqueue(self, message: dict[str, Any]) -> None:
        if self._queue is None:
            return
        try:
            self._queue.put_nowait(message)
        except asyncio.QueueFull:
            self.dropped += 1
            logger.debug("doc relay queue full; event dropped")

    async def _publish_loop(self) -> None:
        assert self._queue is not None
        while True:
            message = await self._queue.get()
            try:
                payload = json.dumps(message, ensure_ascii=False)
                with redis_deadline(budget_s=_PUBLISH_BUDGET_S):
                    await self._client.publish(channel(message["topic"]), payload)
            except asyncio.CancelledError:
                raise
            except Exception:
                self.dropped += 1
                logger.debug("doc event not published", exc_info=True)

    async def _subscribe_loop(self) -> None:
        delay = _BACKOFF_MIN_S
        failing = False
        while True:
            pubsub = self._client.pubsub(ignore_subscribe_messages=True)
            try:
                await pubsub.psubscribe(CHANNEL_PREFIX + "*")
                if failing:
                    logger.info("doc relay subscribed again", extra={"event": "doc_relay_resubscribed"})
                failing = False
                delay = _BACKOFF_MIN_S
                while True:
                    message = await pubsub.get_message(timeout=_READ_TIMEOUT_S)
                    if message is None:
                        await asyncio.sleep(_IDLE_PAUSE_S)
                        continue
                    self._take(message)
            except asyncio.CancelledError:
                raise
            except Exception:
                # Once per outage; the retries after it are quiet.
                if not failing:
                    logger.warning("doc relay subscription lost; retrying", exc_info=True,
                                   extra={"event": "doc_relay_subscribe_failed"})
                failing = True
            finally:
                with contextlib.suppress(Exception):
                    await pubsub.aclose()
            await asyncio.sleep(delay)
            delay = min(delay * 2, _BACKOFF_MAX_S)

    def _take(self, message: dict[str, Any]) -> None:
        self.received += 1
        try:
            data = json.loads(message["data"])
            if isinstance(data, dict):
                self._hub.deliver_remote(data)
        except Exception:  # one bad message must not end the subscription
            logger.debug("doc relay message skipped", exc_info=True)


def start_doc_relay() -> DocRelay | None:
    """Start the relay where replicas share Redis and cloud turns write docs;
    None everywhere else. Runs on the server's loop.

    Without COWORK_REMOTE_DOCS no doc is written there, and the relay would
    hold a pub/sub connection per replica for nothing.
    """
    if get_backend() != "redis" or not get_app_settings().remote_docs_available():
        return None
    relay = DocRelay(hub, get_redis())
    relay.start()
    return relay


async def stop_doc_relay(relay: DocRelay | None) -> None:
    if relay is not None:
        await relay.stop()
