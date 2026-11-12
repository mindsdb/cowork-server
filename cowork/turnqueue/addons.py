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
addon from any replica. Each is recorded there, as ``pending``, before it is
pushed to the pod, and the hash is also what the relay trusts: a pod's
``addon`` step reaches the app only for an addon this server recorded for
that turn and did not give up on (``relay_addon_step``). An addon /addon
answered queued for after the push (no verdict in time, a refusal, the turn
ending) is marked ``abandoned``, so a late verdict from the pod can't show
it as added while the app also runs it as the next task.

A pod whose anton takes messages mid-task says so once at turn start, with
an ``addon`` step in state ``ready`` (``addon_ready_key``); a turn without
it is never offered one (``not_supported``).
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
    ANSWER_TTL_S,
    READ_GRACE_S,
    TAIL_COUNT,
    TERMINAL_KINDS,
    fits_stdin_line,
    own_reply,
)
from cowork.turnqueue.redis_client import (
    REDIS_BLOCK_MS,
    addon_index_key,
    addon_rate_key,
    addon_ready_key,
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
#: The turn_step the pod's addon events arrive as.
ADDON_STEP = "addon"
#: The pod's once-per-turn "this turn takes messages mid-task" state.
READY_STATE = "ready"
#: The pod's addon states the app renders. "refused" and "already_read" are
#: verdicts on one POST /addon or /addon/withdraw, which answers the person
#: itself; "withdrawn" is also saved so a reload shows the message moved.
STREAMED_STATES = frozenset({"added", "read", "returned", "withdrawn"})

# Where each recorded addon stands (the index entry's "status").
_PENDING = "pending"        # recorded, pushed or about to be; no verdict yet
_ADDED = "added"            # the pod took it and /addon said so
_WITHDRAWN = "withdrawn"    # taken back by /addon/withdraw
_ABANDONED = "abandoned"    # /addon answered queued after the push
# The ones the turn holds: what Jev reads as already added, and what a
# withdraw can take back. An entry from before statuses existed counts as added.
_HELD = frozenset({_PENDING, _ADDED})


class RemoteAddonResult(StrEnum):
    ADDED = "added"
    REFUSED = "refused"
    NOT_FOUND = "not_found"
    TIMEOUT = "timeout"
    TOO_LARGE = "too_large"
    # Redis failed before anything reached the pod.
    REDIS_ERROR = "redis_error"


class RemoteWithdrawResult(StrEnum):
    WITHDRAWN = "withdrawn"
    ALREADY_READ = "already_read"
    NOT_FOUND = "not_found"
    UNAVAILABLE = "unavailable"


def _dumps(body: dict) -> str:
    # Compact and ensure_ascii=False: the size bound tracks the raw UTF-8 size
    # the controller measures (answers.py says why).
    return json.dumps(body, separators=(",", ":"), ensure_ascii=False)


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
    tail = await r.xrevrange(stream, count=TAIL_COUNT)
    last_id = tail[0][0] if tail else "0-0"
    for _entry_id, fields in tail:  # newest first
        reply = own_reply(fields, correlation_id)
        if reply is None:
            continue
        return last_id, reply.kind in TERMINAL_KINDS
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
        with redis_deadline(budget_s=left + READ_GRACE_S):
            resp = await xread_blocking(r=r, streams={stream: last_id}, count=50, block_ms=block_ms)
        for _stream, entries in resp or []:
            for entry_id, fields in entries:
                last_id = entry_id
                reply = own_reply(fields, correlation_id)
                if reply is None:
                    continue
                if reply.kind in TERMINAL_KINDS:
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
    local_key: str | None = None,
    deadline: float | None = None,
    r: aioredis.Redis | None = None,
    ack_timeout_s: float = ADDON_ACK_TIMEOUT_S,
) -> tuple[RemoteAddonResult, str]:
    """Queue *text* for the pod's turn and return its verdict with the addon id.

    The addon is recorded in the turn's index before it is pushed, so the
    relay can trust the pod's steps for it from the first one. REDIS_ERROR
    when Redis fails before the push (nothing reached the pod). TIMEOUT
    covers a verdict that never came and a Redis error once the entry may be
    queued; both also queue a withdraw of it. An older scratchpad-controller
    drops the entry, which reads as TIMEOUT. Every answer but ADDED after the
    push marks the addon abandoned, so a late pod step for it is dropped.

    *deadline* (time.monotonic()) is the request's own: the wait for the
    pod ends by it too, as TIMEOUT, and one already past before the push
    pushes nothing."""
    r = r or get_redis()
    addon_id = uuid.uuid4().hex
    entry = addon_entry(addon_id=addon_id, text=text, decided_by=decided_by)
    if not fits_stdin_line(entry):
        return RemoteAddonResult.TOO_LARGE, addon_id
    stream = reply_stream_key(conversation_id)
    try:
        last_id, ended = await _tail(r, stream, correlation_id)
        if ended:
            return RemoteAddonResult.NOT_FOUND, addon_id
        await _record(r, correlation_id, addon_id, text=text, decided_by=decided_by,
                      local_key=local_key, status=_PENDING)
    except RedisError:
        logger.warning("addon not queued: Redis failed conversation=%s correlation_id=%s",
                       conversation_id, correlation_id, exc_info=True)
        return RemoteAddonResult.REDIS_ERROR, addon_id
    if deadline is not None:
        ack_timeout_s = min(ack_timeout_s, deadline - time.monotonic())
        if ack_timeout_s <= 0:
            await _set_status_quietly(r, correlation_id, addon_id, _ABANDONED)
            return RemoteAddonResult.TIMEOUT, addon_id
    try:
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
        await _set_status_quietly(r, correlation_id, addon_id, _ABANDONED)
        return RemoteAddonResult.NOT_FOUND, addon_id
    if verdict is None:
        logger.warning("addon not confirmed within %.0fs conversation=%s correlation_id=%s addon_id=%s",
                       ack_timeout_s, conversation_id, correlation_id, addon_id)
        await _set_status_quietly(r, correlation_id, addon_id, _ABANDONED)
        await _withdraw_quietly(r, correlation_id, addon_id)
        return RemoteAddonResult.TIMEOUT, addon_id
    if verdict.get("state") == "refused":
        await _set_status_quietly(r, correlation_id, addon_id, _ABANDONED)
        return RemoteAddonResult.REFUSED, addon_id
    await _set_status_quietly(r, correlation_id, addon_id, _ADDED)
    return RemoteAddonResult.ADDED, addon_id


async def _record(
    r: aioredis.Redis, correlation_id: str, addon_id: str, *,
    text: str, decided_by: str, local_key: str | None, status: str,
) -> None:
    """Write *addon_id*'s index entry. Raises RedisError."""
    entry = {"text": text, "decided_by": decided_by, "at": time.time(), "status": status}
    if local_key is not None:
        entry["local_key"] = local_key
    key = addon_index_key(correlation_id)
    async with r.pipeline(transaction=True) as pipe:
        pipe.hset(key, addon_id, _dumps(entry))
        pipe.expire(key, ADDON_INDEX_TTL_S)
        await pipe.execute()


def _load(raw: str | None) -> dict | None:
    if not raw:
        return None
    try:
        entry = json.loads(raw)
    except ValueError:
        return None
    return entry if isinstance(entry, dict) and isinstance(entry.get("text"), str) else None


def _status(entry: dict) -> str:
    return str(entry.get("status") or _ADDED)


async def _set_status_quietly(r: aioredis.Redis, correlation_id: str, addon_id: str, status: str) -> None:
    """Move *addon_id*'s entry to *status*. Read-modify-write: only the
    /addon and /addon/withdraw holding the turn's decision lock write it."""
    key = addon_index_key(correlation_id)
    try:
        entry = _load(await r.hget(key, addon_id))
        if entry is None:
            return
        entry["status"] = status
        await r.hset(key, addon_id, _dumps(entry))
    except RedisError:
        # A pending entry left so reads as held: Jev judges a follow-up with
        # it, and a late pod step for it is still relayed.
        logger.warning("addon status not recorded correlation_id=%s addon_id=%s status=%s",
                       correlation_id, addon_id, status, exc_info=True)


async def _entries(r: aioredis.Redis, correlation_id: str) -> list[tuple[float, dict]]:
    rows = await r.hgetall(addon_index_key(correlation_id))
    entries = []
    for raw in rows.values():
        entry = _load(raw)
        if entry is None:
            continue
        try:
            at = float(entry.get("at") or 0)
        except (TypeError, ValueError):
            at = 0.0
        entries.append((at, entry))
    return sorted(entries, key=lambda pair: pair[0])


async def remote_added_texts(correlation_id: str, *, r: aioredis.Redis | None = None) -> list[str]:
    """The messages the turn holds so far, oldest first. Raises RedisError."""
    r = r or get_redis()
    return [entry["text"] for _at, entry in await _entries(r, correlation_id) if _status(entry) in _HELD]


async def remote_addon_totals(correlation_id: str, *, r: aioredis.Redis | None = None) -> tuple[int, int]:
    """How many messages the turn took and their UTF-8 bytes, withdrawn ones
    included, for the per-turn limits (streaming.addons.turn_limit_reason).
    Raises RedisError."""
    r = r or get_redis()
    taken = [entry for _at, entry in await _entries(r, correlation_id) if _status(entry) != _ABANDONED]
    return len(taken), sum(len(entry["text"].encode("utf-8")) for entry in taken)


async def remote_addons_ready(correlation_id: str, *, r: aioredis.Redis | None = None) -> bool:
    """Whether the pod's turn said it takes messages mid-task. Raises RedisError."""
    r = r or get_redis()
    return bool(await r.exists(addon_ready_key(correlation_id)))


async def remote_rate_limited(
    user_key: str, *, limit: int, window_s: int, r: aioredis.Redis | None = None,
) -> bool:
    """Count one /addon for *user_key* in the current fixed window, shared by
    every replica, and say whether it is over *limit*. Raises RedisError."""
    r = r or get_redis()
    key = addon_rate_key(user_key, int(time.time() // window_s))
    async with r.pipeline(transaction=True) as pipe:
        pipe.incr(key)
        pipe.expire(key, window_s * 2)
        count, _ = await pipe.execute()
    return int(count) > limit


async def relay_addon_step(
    *, correlation_id: str, data: dict, r: aioredis.Redis | None = None,
) -> list:
    """The ``TurnAddon`` events a pod's ``addon`` step becomes on the app's
    stream, which saves them with the answer.

    The pod is semi-trusted: a step is relayed only for an addon this server
    recorded for this turn and did not abandon, and its text, decider and
    ``local_key`` come from that record, not the pod. "ready" is recorded
    for /addon's support check and never relayed. Never raises: a step that
    can't be checked is dropped."""
    from cowork.streaming.addons import TurnAddon

    r = r or get_redis()
    state = data.get("state")
    if state == READY_STATE:
        try:
            await r.set(addon_ready_key(correlation_id), "1", ex=ADDON_INDEX_TTL_S)
        except RedisError:
            logger.warning("addon ready marker not recorded correlation_id=%s", correlation_id, exc_info=True)
        return []
    addon_id = data.get("addon_id")
    if not isinstance(addon_id, str) or not addon_id or state not in STREAMED_STATES:
        return []
    try:
        entry = _load(await r.hget(addon_index_key(correlation_id), addon_id))
    except RedisError:
        logger.warning("addon step dropped: Redis failed correlation_id=%s addon_id=%s",
                       correlation_id, addon_id, exc_info=True)
        return []
    if entry is None or _status(entry) == _ABANDONED:
        logger.info("addon step dropped: not an addon this turn holds correlation_id=%s addon_id=%s state=%s",
                    correlation_id, addon_id, state)
        return []
    decided_by = entry.get("decided_by")
    local_key = entry.get("local_key")
    return [TurnAddon(
        addon_id=addon_id,
        state=state,
        text=entry["text"],
        decided_by=decided_by if decided_by in ("jev", "user") else None,
        local_key=local_key if isinstance(local_key, str) else None,
    )]


async def remote_progress(
    *, conversation_id: str, correlation_id: str, r: aioredis.Redis | None = None,
) -> list[str]:
    """The turn's last steps in the in-process channel's format
    (response_routing.progress_line), read from the replies the pod relayed:
    `tool_start` opens a step, `tool_end` carries its args (and so its
    description), `tool_result` closes it. Raises RedisError."""
    from cowork.handlers.response_routing import ADDON_PROGRESS_STEPS, progress_line, step_description

    r = r or get_redis()
    tail = await r.xrevrange(reply_stream_key(conversation_id), count=_PROGRESS_SCAN)
    steps: dict[str, dict] = {}  # insertion order is start order
    for _entry_id, fields in reversed(tail):  # oldest first
        reply = own_reply(fields, correlation_id)
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
    return lines[-ADDON_PROGRESS_STEPS:]


async def withdraw_remote_addon(
    *,
    conversation_id: str,
    correlation_id: str,
    addon_id: str,
    r: aioredis.Redis | None = None,
    ack_timeout_s: float = ADDON_ACK_TIMEOUT_S,
) -> RemoteWithdrawResult:
    """Take *addon_id* back from the pod's turn before the model reads it.

    NOT_FOUND for an addon this turn never took (or one already withdrawn,
    or one /addon answered queued) and for a turn that ended, which returned
    every unread message.
    UNAVAILABLE when the pod's verdict did not come or Redis failed: the
    withdraw may still land, and the app keeps the message where it is."""
    r = r or get_redis()
    key = addon_index_key(correlation_id)
    stream = reply_stream_key(conversation_id)
    try:
        entry = _load(await r.hget(key, addon_id))
        if entry is None or _status(entry) not in _HELD:
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
    # Kept, not deleted: the pod's "withdrawn" step is relayed (and saved)
    # against it, and it still counts toward the turn's limits.
    await _set_status_quietly(r, correlation_id, addon_id, _WITHDRAWN)
    return RemoteWithdrawResult.WITHDRAWN
