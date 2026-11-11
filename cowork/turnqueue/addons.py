"""Hand a message sent mid-task (an "addon", ENG-4194) to a remote (pod) turn,
or take one back.

Rides the queue ask_user answers use (``answers.py``): the entry is pushed
onto ``answer_queue_key(correlation_id)``, scratchpad-controller writes it to
the pod's stdin, and the pod alone decides whether its turn takes it. It says
so with an ``addon`` turn_step naming the entry's ``addon_id`` (anton's
``cloud_turn/addons.py``), which this module waits for. The same steps reach
the app as ``response.addon`` events (``producer.step_stream_events``).

Entries, compact JSON with ensure_ascii=False:

- ``{"kind":"addon","addon_id":<hex>,"text":...,"decided_by":"jev"|"user"}``,
  answered with state ``added`` or ``refused``;
- ``{"kind":"addon_withdraw","addon_id":...,"request_id":<hex>}``, answered
  with state ``withdrawn`` or ``already_read`` naming the ``request_id``.

The messages a turn took are kept per correlation id (``addon_index_key``),
so Jev judges a follow-up with the ones before it and a withdraw finds its
addon from any replica.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from collections.abc import Callable
from enum import StrEnum

import redis.asyncio as aioredis
from redis.exceptions import RedisError

from cowork.turnqueue.answers import (
    _READ_GRACE_S,
    _TAIL_COUNT,
    _TERMINAL,
    _WRAP_HEADROOM_BYTES,
    ANSWER_TTL_S,
    MAX_ANSWER_BYTES,
    _own_reply,
)
from cowork.turnqueue.redis_client import (
    REDIS_BLOCK_MS,
    addon_index_key,
    answer_queue_key,
    get_redis,
    redis_deadline,
    reply_stream_key,
    xread_blocking,
)

logger = logging.getLogger(__name__)

#: How long /addon and /addon/withdraw wait for the pod's verdict. The person
#: is waiting on the composer, and a pod that takes longer than this is busy
#: in a way that makes the message better off queued.
ADDON_ACK_TIMEOUT_S = 5.0
#: The added messages of a turn are kept this long after the last one: past
#: any turn's length, and gone on their own once nobody asks.
ADDON_INDEX_TTL_S = 6 * 3600
#: How far back the reply stream is read for the turn's recent tool calls.
_PROGRESS_SCAN = 200
#: The recent steps Jev reads, as in-process (response_routing.ADDON_PROGRESS_STEPS).
_PROGRESS_STEPS = 8
#: The turn_step the pod's addon events arrive as.
ADDON_STEP = "addon"


class RemoteAddonResult(StrEnum):
    ADDED = "added"
    REFUSED = "refused"
    NOT_FOUND = "not_found"
    TIMEOUT = "timeout"
    TOO_LARGE = "too_large"


class RemoteWithdrawResult(StrEnum):
    WITHDRAWN = "withdrawn"
    ALREADY_READ = "already_read"
    NOT_FOUND = "not_found"
    UNAVAILABLE = "unavailable"


def _dumps(body: dict) -> str:
    # Compact and ensure_ascii=False: the size bound tracks the raw UTF-8 size
    # the controller measures (answers.py says why).
    return json.dumps(body, separators=(",", ":"), ensure_ascii=False)


def _fits(entry: str) -> bool:
    return len(entry.encode("utf-8")) + _WRAP_HEADROOM_BYTES <= MAX_ANSWER_BYTES


def addon_entry(*, addon_id: str, text: str, decided_by: str) -> str:
    return _dumps({"kind": "addon", "addon_id": addon_id, "text": text, "decided_by": decided_by})


def withdraw_entry(*, addon_id: str, request_id: str) -> str:
    return _dumps({"kind": "addon_withdraw", "addon_id": addon_id, "request_id": request_id})


def is_addon_step(data: dict) -> bool:
    return data.get("step") == ADDON_STEP


async def _push(r: aioredis.Redis, correlation_id: str, entry: str) -> None:
    key = answer_queue_key(correlation_id)
    async with r.pipeline(transaction=True) as pipe:
        pipe.rpush(key, entry)
        pipe.expire(key, ANSWER_TTL_S)
        await pipe.execute()


async def _tail(r: aioredis.Redis, stream: str, correlation_id: str) -> tuple[str, bool]:
    """Where to read the pod's verdict from, and whether the turn has already
    sent its terminal reply (then it never answers)."""
    tail = await r.xrevrange(stream, count=_TAIL_COUNT)
    last_id = tail[0][0] if tail else "0-0"
    for _entry_id, fields in tail:  # newest first
        reply = _own_reply(fields, correlation_id)
        if reply is None:
            continue
        return last_id, reply.kind in _TERMINAL
    return last_id, False


_ENDED = object()


async def _wait_for(
    r: aioredis.Redis,
    *,
    stream: str,
    last_id: str,
    correlation_id: str,
    match: Callable[[dict], bool],
    timeout_s: float,
) -> dict | object | None:
    """The first addon step of this turn *match* accepts, ``_ENDED`` when the
    turn ends first, None on timeout. Raises RedisError."""
    deadline = time.monotonic() + timeout_s
    while (left := deadline - time.monotonic()) > 0:
        block_ms = max(1, min(REDIS_BLOCK_MS, int(left * 1000)))
        with redis_deadline(budget_s=left + _READ_GRACE_S):
            resp = await xread_blocking(r=r, streams={stream: last_id}, count=50, block_ms=block_ms)
        for _stream, entries in resp or []:
            for entry_id, fields in entries:
                last_id = entry_id
                reply = _own_reply(fields, correlation_id)
                if reply is None:
                    continue
                if reply.kind in _TERMINAL:
                    return _ENDED
                data = reply.data or {}
                if reply.kind == "turn_step" and is_addon_step(data) and match(data):
                    return data
    return None


async def _withdraw_quietly(r: aioredis.Redis, correlation_id: str, addon_id: str) -> None:
    """After a verdict that never came: so a late add can't also run as the
    queued next task, unless the model already read it."""
    try:
        await _push(r, correlation_id, withdraw_entry(addon_id=addon_id, request_id=uuid.uuid4().hex))
    except RedisError:
        logger.warning("addon withdraw after a missing verdict could not be queued correlation_id=%s "
                       "addon_id=%s", correlation_id, addon_id, exc_info=True)


async def submit_remote_addon(
    *,
    conversation_id: str,
    correlation_id: str,
    text: str,
    decided_by: str,
    r: aioredis.Redis | None = None,
    ack_timeout_s: float = ADDON_ACK_TIMEOUT_S,
) -> tuple[RemoteAddonResult, str]:
    """Queue *text* for the pod's turn and return its verdict with the addon id.

    TIMEOUT covers a verdict that never came and a Redis error once the entry
    may be queued; both also queue a withdraw of it. An older
    scratchpad-controller drops the entry, which reads as TIMEOUT."""
    r = r or get_redis()
    addon_id = uuid.uuid4().hex
    entry = addon_entry(addon_id=addon_id, text=text, decided_by=decided_by)
    if not _fits(entry):
        return RemoteAddonResult.TOO_LARGE, addon_id
    stream = reply_stream_key(conversation_id)
    try:
        last_id, ended = await _tail(r, stream, correlation_id)
        if ended:
            return RemoteAddonResult.NOT_FOUND, addon_id
        await _push(r, correlation_id, entry)
        verdict = await _wait_for(
            r, stream=stream, last_id=last_id, correlation_id=correlation_id,
            match=lambda data: data.get("addon_id") == addon_id
            and data.get("state") in ("added", "read", "refused"),
            timeout_s=ack_timeout_s,
        )
    except RedisError:
        logger.warning("addon verdict wait hit a Redis error conversation=%s correlation_id=%s addon_id=%s",
                       conversation_id, correlation_id, addon_id, exc_info=True)
        verdict = None
    if verdict is _ENDED:
        return RemoteAddonResult.NOT_FOUND, addon_id
    if verdict is None:
        logger.warning("addon not confirmed within %.0fs conversation=%s correlation_id=%s addon_id=%s",
                       ack_timeout_s, conversation_id, correlation_id, addon_id)
        await _withdraw_quietly(r, correlation_id, addon_id)
        return RemoteAddonResult.TIMEOUT, addon_id
    if verdict.get("state") == "refused":
        return RemoteAddonResult.REFUSED, addon_id
    await _remember(r, correlation_id, addon_id, text=text, decided_by=decided_by)
    return RemoteAddonResult.ADDED, addon_id


async def _remember(r: aioredis.Redis, correlation_id: str, addon_id: str, *, text: str, decided_by: str) -> None:
    key = addon_index_key(correlation_id)
    try:
        async with r.pipeline(transaction=True) as pipe:
            pipe.hset(key, addon_id, _dumps({"text": text, "decided_by": decided_by, "at": time.time()}))
            pipe.expire(key, ADDON_INDEX_TTL_S)
            await pipe.execute()
    except RedisError:
        # The turn has it either way; only a follow-up's judgement and a
        # withdraw (404) miss it.
        logger.warning("added message not recorded correlation_id=%s addon_id=%s",
                       correlation_id, addon_id, exc_info=True)


async def remote_added_texts(correlation_id: str, *, r: aioredis.Redis | None = None) -> list[str]:
    """The messages the turn took so far, oldest first. Raises RedisError."""
    r = r or get_redis()
    rows = await r.hgetall(addon_index_key(correlation_id))
    entries = []
    for raw in rows.values():
        try:
            entry = json.loads(raw)
            entries.append((float(entry.get("at") or 0), str(entry["text"])))
        except (ValueError, KeyError, TypeError):
            continue
    return [text for _at, text in sorted(entries)]


async def remote_progress(
    *, conversation_id: str, correlation_id: str, r: aioredis.Redis | None = None,
) -> list[str]:
    """The turn's last steps in the in-process channel's format
    (response_routing.progress_line), read from the replies the pod relayed:
    `tool_start` opens a step, `tool_end` carries its args (and so its
    description), `tool_result` closes it. Raises RedisError."""
    from cowork.handlers.response_routing import progress_line, step_description

    r = r or get_redis()
    tail = await r.xrevrange(reply_stream_key(conversation_id), count=_PROGRESS_SCAN)
    steps: dict[str, dict] = {}  # insertion order is start order
    for _entry_id, fields in reversed(tail):  # oldest first
        reply = _own_reply(fields, correlation_id)
        if reply is None or reply.kind != "turn_step":
            continue
        data = reply.data or {}
        step, tool_id = data.get("step"), str(data.get("id") or "")
        if step == "tool_start" and data.get("name"):
            steps[tool_id or f"_{len(steps)}"] = {"name": data["name"], "description": "", "done": False}
        elif step == "tool_end" and tool_id in steps:
            steps[tool_id]["description"] = step_description(data.get("args") or "")
        elif step == "tool_result" and tool_id in steps:
            steps[tool_id]["done"] = True
    lines = [
        progress_line(s["name"], s["description"], in_progress=not s["done"])
        for s in steps.values()
    ]
    return lines[-_PROGRESS_STEPS:]


async def withdraw_remote_addon(
    *,
    conversation_id: str,
    correlation_id: str,
    addon_id: str,
    r: aioredis.Redis | None = None,
    ack_timeout_s: float = ADDON_ACK_TIMEOUT_S,
) -> RemoteWithdrawResult:
    """Take *addon_id* back from the pod's turn before the model reads it.

    NOT_FOUND for an addon this turn never took (or one already withdrawn)
    and for a turn that ended, which returned every unread message.
    UNAVAILABLE when the pod's verdict did not come or Redis failed: the
    withdraw may still land, and the app keeps the message where it is."""
    r = r or get_redis()
    key = addon_index_key(correlation_id)
    stream = reply_stream_key(conversation_id)
    try:
        if not await r.hexists(key, addon_id):
            return RemoteWithdrawResult.NOT_FOUND
        last_id, ended = await _tail(r, stream, correlation_id)
        if ended:
            return RemoteWithdrawResult.NOT_FOUND
        request_id = uuid.uuid4().hex
        await _push(r, correlation_id, withdraw_entry(addon_id=addon_id, request_id=request_id))
        verdict = await _wait_for(
            r, stream=stream, last_id=last_id, correlation_id=correlation_id,
            match=lambda data: data.get("request_id") == request_id
            and data.get("state") in ("withdrawn", "already_read"),
            timeout_s=ack_timeout_s,
        )
    except RedisError:
        logger.warning("addon withdraw hit a Redis error conversation=%s correlation_id=%s addon_id=%s",
                       conversation_id, correlation_id, addon_id, exc_info=True)
        return RemoteWithdrawResult.UNAVAILABLE
    if verdict is _ENDED:
        return RemoteWithdrawResult.NOT_FOUND
    if verdict is None:
        logger.warning("addon withdraw not confirmed within %.0fs conversation=%s correlation_id=%s addon_id=%s",
                       ack_timeout_s, conversation_id, correlation_id, addon_id)
        return RemoteWithdrawResult.UNAVAILABLE
    if verdict.get("state") == "already_read":
        return RemoteWithdrawResult.ALREADY_READ
    try:
        await r.hdel(key, addon_id)
    except RedisError:
        logger.warning("withdrawn addon left in the index correlation_id=%s addon_id=%s",
                       correlation_id, addon_id, exc_info=True)
    return RemoteWithdrawResult.WITHDRAWN
