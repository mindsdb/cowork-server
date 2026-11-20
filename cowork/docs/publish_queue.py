"""The org republish queue: a shared doc's page follows its edits across replicas.

A write puts the doc in a Redis sorted set, scored by when to publish:
4 s after the last write, and never later than 30 s after the first write of
a burst. Every replica polls the set once a second; the one whose ZREM takes
the member publishes (cowork.services.live_publish). A retry goes back with
ZADD NX, so it never pushes back the due time a fresher write already set.
A claim lost with its replica is not retried here: the next write or the
turn-end autopublish catches the page up.
"""
from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable

from redis.asyncio import Redis

from cowork.common.settings.app_settings import get_app_settings
from cowork.services.live_publish import (
    DocPublishJob,
    Outcome,
    PublishResult,
    drain,
    drain_cleanup,
    is_permanent,
    publish_doc_page,
    track,
)
from cowork.streaming.backend import get_backend
from cowork.turnqueue.redis_client import get_redis

logger = logging.getLogger(__name__)

DUE_KEY = "cowork:doc-publish:due"
_DOC_PREFIX = "cowork:doc-publish:doc:"
_FIELDS = ("project_id", "artifacts_base", "slug", "org_id")
DEBOUNCE_S = 4.0
CAP_S = 30.0
HASH_TTL_S = 86400
TICK_S = 1.0
SLOTS = 4
MAX_ATTEMPTS = 5
BACKOFF_S = (5.0, 15.0, 60.0, 300.0)
# One deadline for the whole stop. The server runs it alongside the
# scratchpad closes, so it must stay within their timeout: the pod's grace
# period has no room for a wait of its own (tests/test_chart_values.py).
STOP_BUDGET_S = 3.0

_enqueues: set[asyncio.Task] = set()

Publish = Callable[..., Awaitable[PublishResult]]


def doc_key(artifact_id: str) -> str:
    return _DOC_PREFIX + artifact_id


def _attempts(value: str | None) -> int:
    # A corrupted count must not stop the publish, or the claim would re-queue every tick.
    try:
        return int(value or 0)
    except ValueError:
        return 0


def _level(result: PublishResult) -> int:
    quiet = result.outcome is Outcome.UNCHANGED or (
        result.outcome is Outcome.SKIPPED and result.reason == "unpublished") or (
        result.outcome is Outcome.RETRY and result.reason == "lock_busy")
    return logging.DEBUG if quiet else logging.WARNING


def _record(result: PublishResult, job: DocPublishJob, *, attempts: int = 0, error_type: str | None = None) -> None:
    # Only fields the JSON formatter keeps (cowork.common.structured_logging.EVENT_FIELDS).
    logger.log(_level(result), "doc_publish", extra={
        "event": "doc_publish", "outcome": result.outcome.value, "reason": result.reason,
        "artifact_slug": job.slug, "project_id": job.project_id, "organization_id": job.org_id,
        "attempts": attempts, "error_type": error_type,
    })


async def enqueue(job: DocPublishJob, *, client: Redis | None = None, now: float | None = None) -> None:
    """Ask for ``job``'s page to be published once the burst of writes settles. Never raises."""
    now = time.time() if now is None else now
    key = doc_key(job.artifact_id)
    try:
        r = client if client is not None else get_redis()
        pipe = r.pipeline(transaction=True)
        pipe.hset(key, mapping={f: getattr(job, f) for f in _FIELDS})
        pipe.hsetnx(key, "first", now)
        pipe.expire(key, HASH_TTL_S)
        pipe.hget(key, "first")
        *_, first = await pipe.execute()
        due = min(now + DEBOUNCE_S, float(first) + CAP_S)
        await r.zadd(DUE_KEY, {job.artifact_id: due})
    except Exception:  # noqa: BLE001 - the write stands; the page catches up on a later one
        logger.warning("doc_publish", exc_info=True, extra={
            "event": "doc_publish", "outcome": "enqueue_failed", "artifact_slug": job.slug,
            "project_id": job.project_id})


def schedule_enqueue(job: DocPublishJob) -> bool:
    """Run ``enqueue`` in the background; False with no running loop or no Redis backend."""
    if get_backend() != "redis":
        return False
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        logger.warning("doc_publish", extra={
            "event": "doc_publish", "outcome": "not_queued", "reason": "no_loop", "artifact_slug": job.slug})
        return False
    track(loop.create_task(enqueue(job)), _enqueues)
    return True


async def drain_enqueues(*, timeout: float) -> None:
    await drain(_enqueues, timeout=timeout, outcome="enqueue_unfinished")


class DocPublishWorker:
    """One replica's poll of the queue."""

    def __init__(self, client: Redis, *, slots: int = SLOTS, tick_s: float = TICK_S,
                 publish: Publish = publish_doc_page, clock: Callable[[], float] = time.time) -> None:
        self._client = client
        self._free = slots
        self._tick_s = tick_s
        self._publish = publish
        self._clock = clock
        self._loop_task: asyncio.Task | None = None
        # artifact_id -> its _run while it goes on: a member due again meanwhile stays in DUE_KEY.
        self._running: dict[str, asyncio.Task] = {}

    @property
    def free_slots(self) -> int:
        return self._free

    def start(self) -> None:
        self._loop_task = asyncio.get_running_loop().create_task(self._loop(), name="doc-publish-worker")

    async def _loop(self) -> None:
        while True:
            try:
                await self.tick()
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001 - one bad tick must not end the poll
                logger.warning("doc_publish", exc_info=True, extra={"event": "doc_publish", "outcome": "tick_failed"})
            await asyncio.sleep(self._tick_s)

    async def tick(self) -> None:
        if self._free <= 0:
            return
        r = self._client
        # Enough to fill the free slots even when every doc still publishing is among them.
        members = await r.zrangebyscore(DUE_KEY, "-inf", self._clock(), start=0,
                                        num=self._free + len(self._running))
        for artifact_id in members:
            if self._free <= 0:
                return
            if artifact_id in self._running:
                continue
            if not await r.zrem(DUE_KEY, artifact_id):
                continue  # another replica took it
            try:
                await self._claim(artifact_id)
            except BaseException:
                # Claimed but not started (a stop, a Redis error): nobody else would publish it.
                await self._requeue(artifact_id)
                raise

    async def _claim(self, artifact_id: str) -> None:
        r = self._client
        key = doc_key(artifact_id)
        # A write from here on starts a new burst.
        pipe = r.pipeline(transaction=False)
        pipe.hgetall(key)
        pipe.hdel(key, "first")
        fields, _ = await pipe.execute()
        if not all(fields.get(name) for name in _FIELDS):
            # Expired, or recreated by a retry's HSET after it expired.
            await r.delete(key)
            logger.debug("doc_publish", extra={"event": "doc_publish", "outcome": "skipped", "reason": "no_fields"})
            return
        job = DocPublishJob(artifact_id=artifact_id, **{f: fields[f] for f in _FIELDS})
        task = asyncio.get_running_loop().create_task(self._run(job, _attempts(fields.get("attempts"))))
        self._free -= 1
        self._running[artifact_id] = task
        task.add_done_callback(lambda _task: self._running.pop(artifact_id, None))

    async def _requeue(self, artifact_id: str) -> None:
        try:
            await self._client.zadd(DUE_KEY, {artifact_id: self._clock()}, nx=True)
        except Exception:  # noqa: BLE001 - the caller re-raises its own error
            logger.warning("doc_publish", exc_info=True, extra={
                "event": "doc_publish", "outcome": "requeue_failed"})

    async def _run(self, job: DocPublishJob, attempts: int) -> None:
        freed = False

        def on_done() -> None:
            nonlocal freed
            if not freed:
                freed = True
                self._free += 1

        error_type = None
        try:
            result = await self._publish(job, on_done=on_done)
        except asyncio.CancelledError:
            # Shutdown: the next replica up publishes it.
            await self._client.zadd(DUE_KEY, {job.artifact_id: self._clock()}, nx=True)
            raise
        except Exception as exc:  # noqa: BLE001 - classified below
            result = PublishResult(Outcome.DROPPED if is_permanent(exc) else Outcome.RETRY)
            error_type = type(exc).__name__
        key = doc_key(job.artifact_id)
        claimed = attempts
        try:
            if result.outcome is Outcome.RETRY and result.reason == "lock_busy":
                # Another publish of this page is still running: not an attempt of ours.
                await self._client.zadd(DUE_KEY, {job.artifact_id: self._clock() + DEBOUNCE_S}, nx=True)
                _record(result, job, attempts=attempts, error_type=error_type)
                return
            if result.outcome is Outcome.RETRY:
                attempts += 1
                if attempts >= MAX_ATTEMPTS:
                    result = PublishResult(Outcome.DROPPED, result.reason)
                else:
                    pipe = self._client.pipeline(transaction=True)
                    pipe.hset(key, "attempts", attempts)
                    pipe.expire(key, HASH_TTL_S)
                    await pipe.execute()
                    delay = BACKOFF_S[min(attempts, len(BACKOFF_S)) - 1]
                    await self._client.zadd(DUE_KEY, {job.artifact_id: self._clock() + delay}, nx=True)
            if claimed and result.outcome is not Outcome.RETRY:
                await self._client.hdel(key, "attempts")
            _record(result, job, attempts=attempts, error_type=error_type)
        except Exception:  # noqa: BLE001 - the next write queues the doc again
            logger.warning("doc_publish", exc_info=True, extra={
                "event": "doc_publish", "outcome": "record_failed", "artifact_slug": job.slug})

    async def pause(self) -> None:
        """Stop claiming; publishes already running go on."""
        task, self._loop_task = self._loop_task, None
        if task is not None:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    async def stop(self, *, timeout: float) -> None:
        await self.pause()
        tasks = list(self._running.values())
        for task in tasks:
            task.cancel()
        await drain(tasks, timeout=timeout, outcome="stop_unfinished")


def start_doc_publish_worker() -> DocPublishWorker | None:
    """Start the poll on an org deployment with the Redis backend; None anywhere else. Runs on the server's loop."""
    if get_app_settings().tenancy_mode != "org":
        return None
    if get_backend() != "redis":
        logger.warning("doc_publish", extra={"event": "doc_publish", "outcome": "not_started", "reason": "no_redis"})
        return None
    worker = DocPublishWorker(get_redis())
    worker.start()
    return worker


async def pause_doc_publish_worker(worker: DocPublishWorker | None) -> None:
    """Stop claiming at the start of shutdown; what is already publishing goes on."""
    if worker is not None:
        await worker.pause()


async def stop_doc_publish_worker(worker: DocPublishWorker | None, *, budget_s: float = STOP_BUDGET_S) -> None:
    """Shutdown, within one budget: queue what turns just wrote, stop the poll,
    then let go of what finished uploads held."""
    loop = asyncio.get_running_loop()
    deadline = loop.time() + budget_s
    await drain_enqueues(timeout=deadline - loop.time())
    if worker is not None:
        await worker.stop(timeout=max(0.0, deadline - loop.time()))
    await drain_cleanup(timeout=max(0.0, deadline - loop.time()))
