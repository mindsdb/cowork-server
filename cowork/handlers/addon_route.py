"""Where a message sent while a turn runs goes: into that turn, or the queue
(ENG-4194).

POST /responses/addon settles authorization and the turn, then hands the
message here. Jev decides with ``jev_route.ADDON_ROUTE_QUESTION``; the person
can also pick "Add to current task" themselves (``force``), which skips Jev.
Every failure queues the message, which is what happens without this feature,
so nothing here raises.

A turn that runs in a pod (``COWORK_TURN_BACKEND=remote``) has no in-process
channel: ``route_remote_addon`` reads what Jev needs from the database and
Redis, then hands the message to the pod (``turnqueue/addons.py``).

The checks before any of that (the per-person rate limit, the app's
``response_id`` and ``local_key``, valid text) are here too, for the endpoint.
"""
from __future__ import annotations

import asyncio
import itertools
import json
import logging
import math
import re
import time
from collections import deque
from contextlib import asynccontextmanager
from dataclasses import dataclass
from functools import partial
from typing import Any, Literal
from uuid import UUID, uuid4

from redis.exceptions import RedisError
from sqlalchemy.exc import SQLAlchemyError

from cowork.common.history_scrub import scrub_credentials
from cowork.common.settings.app_settings import TurnQueueSettings
from cowork.db.scoped import TenantScope
from cowork.handlers import jev_route
from cowork.streaming.addons import MAX_ADDON_CHARS, AddonChannel, turn_limit_reason
from cowork.turnqueue.addons import (
    RemoteAddonResult,
    remote_added_texts,
    remote_addon_totals,
    remote_progress,
    remote_rate_limited,
    submit_remote_addon,
)

logger = logging.getLogger(__name__)

# Why POST /addon queued a message rather than add it to the running turn.
# The app keeps a queued message as the next task either way; the reason is
# for its copy and for logs.
QueuedReason = Literal[
    # The request itself.
    "empty",           # blank text
    "too_large",       # over MAX_ADDON_CHARS, over the turn's total added
                       # text, or (remote) over the pod's stdin line
    # The turn.
    "disabled",        # COWORK_TURN_ADDON_ROUTING_ENABLED is off
    "no_turn",         # no running turn, a response_id that isn't it, or the
                       # turn stopped taking messages (it is ending)
    "not_supported",   # the turn's anton can't take messages mid-task (no
                       # channel in-process; no "ready" from the pod)
    "too_many",        # the turn already took MAX_ADDONS_PER_TURN
    "rate_limited",    # over ADDON_RATE_LIMIT for this person
    "busy",            # another message for the turn is still being decided
    # Jev.
    "jev_off",         # no decision model to ask
    "jev_timeout",     # Jev did not answer in time
    "jev_error",       # Jev answered with an error or nothing usable
    "low_confidence",  # Jev leaned add, below COWORK_TURN_ADDON_MIN_P_ADD
    "next_task",       # Jev chose the queue
    # Remote (pod) turns: the pod did not answer in time.
    "pod_timeout",
    # Failures outside Jev.
    "redis_error",     # Redis failed
    "internal",        # anything else (settings, minting a key, anton raised)
]

#: The whole of one POST /addon, from the endpoint to its answer: lock wait,
#: Jev (with any key mint) and, remote, the pod's ack. The app gives up
#: after 10 s, so the server answers first: a stage the budget runs out in
#: answers queued with that stage's reason (busy, jev_timeout, pod_timeout).
ADDON_BUDGET_S = 8.0
#: How long a message waits for the one before it on the same turn to be
#: decided, at most: half the budget, leaving the rest for its own decision.
DECIDE_LOCK_TIMEOUT_S = 4.0
#: POST /addon calls one person may make per window, across their turns.
ADDON_RATE_LIMIT = 30
ADDON_RATE_WINDOW_S = 60
#: A minted key outlives the Jev call it is for by this much, at least
#: _MINTED_KEY_MIN_TTL_S in all; it is revoked right after anyway.
_MINTED_KEY_HEADROOM_S = 10
_MINTED_KEY_MIN_TTL_S = 30
#: The app's own id for a message (POST /addon's local_key).
_LOCAL_KEY = re.compile(r"[A-Za-z0-9_-]{1,64}")
#: How many of the turn's first records are searched for its response.created.
_RESPONSE_ID_SCAN = 64


def queued(reason: QueuedReason) -> dict:
    return {"status": "queued", "reason": reason}


def added(addon_id: str, decided_by: str) -> dict:
    return {"status": "added", "addon_id": addon_id, "decided_by": decided_by}


@dataclass(frozen=True)
class _MintKey:
    """Jev on MindsHub, on a key minted for this one decision: a hosted org
    keeps no stored MindsHub key (its turns mint one), like the route gate's
    `_router_binding`. Carries what the mint needs from the caller's settings."""
    workspace_id: str | None
    model: str


def _jev_target(
    scope: TenantScope, queue_settings: TurnQueueSettings | None = None,
) -> tuple[jev_route.JevTarget, str] | _MintKey | None:
    """Who to ask and with which model, from the caller's settings: the
    organization's own decision model, else Jev on MindsHub on the stored key,
    else, on a hosted (remote) deployment, on a key minted for the decision.
    None when none applies (the LLM gate's providers, Jev turned off, no
    MindsHub key to use or mint). Sync: it reads settings through a session of
    its own. *queue_settings* is the request's; read here when not given."""
    from cowork.common.settings.user_settings import Provider, get_user_settings, provider_api_key_str
    from cowork.services.providers import minds_chat_base_url

    settings = get_user_settings(scope)
    queue_settings = queue_settings or TurnQueueSettings()
    mode = jev_route.gate_mode(settings)
    if mode == "own":
        target = jev_route.own_decision_target(settings)
        return (target, settings.decision_model or queue_settings.jev_route_model) if target else None
    if mode != "jev":
        return None
    key = provider_api_key_str(settings, Provider.MINDS_CLOUD)
    if key:
        return jev_route.JevTarget.mindshub(minds_chat_base_url(settings.minds_url), key), queue_settings.jev_route_model
    if queue_settings.is_remote:
        return _MintKey(
            workspace_id=getattr(settings, "hub_workspace_id", "") or None,
            model=queue_settings.jev_route_model,
        )
    return None


# ── Checks before a turn is asked ──────────────────────────────────────────


def valid_text(text: str) -> bool:
    """Whether *text* is valid Unicode the turn can carry: a lone surrogate
    can't be encoded for Jev, Redis or the pod's stdin. Request validation
    already refuses one (422) and body parsing refuses bytes that are not
    UTF-8 (400); this guards the handler should a parser ever let one by."""
    try:
        text.encode("utf-8")
    except UnicodeEncodeError:
        return False
    return True


def clean_local_key(value: str | None) -> str | None:
    """The app's ``local_key`` when it is 1-64 of [A-Za-z0-9_-], else None:
    an invalid one is ignored, not an error, so the message still goes."""
    return value if isinstance(value, str) and _LOCAL_KEY.fullmatch(value) else None


async def turn_response_ids(buffer: Any) -> set[str] | None:
    """The ids of the ``response.created`` frames among the turn's first
    records: the response id the app saw for it. None for a buffer this
    can't read (then there is nothing to compare). Raises RedisError."""
    from cowork.streaming.buffer import FileStreamBuffer, RedisStreamBuffer, read_records
    from cowork.turnqueue.redis_client import get_redis

    if isinstance(buffer, RedisStreamBuffer):
        entries = await get_redis().xrange(buffer.key, count=_RESPONSE_ID_SCAN)
        records = [RedisStreamBuffer._record(fields) for _entry_id, fields in entries]
    elif isinstance(buffer, FileStreamBuffer):
        records = await asyncio.to_thread(
            lambda: list(itertools.islice(read_records(buffer.path), _RESPONSE_ID_SCAN))
        )
    else:
        return None
    ids: set[str] = set()
    for record in records:
        sse = record.data.get("sse") if record.type == "sse" else None
        if not isinstance(sse, str) or not sse.startswith("event: response.created"):
            continue
        for line in sse.splitlines():
            if not line.startswith("data:"):
                continue
            try:
                response_id = json.loads(line[len("data:"):])["response"]["id"]
            except (ValueError, KeyError, TypeError):
                continue
            if isinstance(response_id, str):
                ids.add(response_id)
    return ids


class _RateWindow:
    """ADDON_RATE_LIMIT in one process, for a server with no shared Redis
    (the in-process backend): a sliding window of each person's calls."""

    def __init__(self) -> None:
        self._calls: dict[str, deque[float]] = {}

    def hit(self, user_key: str, *, limit: int, window_s: float) -> bool:
        """Count one call and say whether it is over *limit*."""
        now = time.monotonic()
        calls = self._calls.setdefault(user_key, deque())
        while calls and calls[0] <= now - window_s:
            calls.popleft()
        if len(calls) >= limit:
            return True
        calls.append(now)
        return False

    def reset(self) -> None:
        """For tests: this object is a process global."""
        self._calls.clear()


_local_rate = _RateWindow()


async def rate_limit_refusal(scope: TenantScope, *, remote: bool) -> dict | None:
    """The queued response for a person over ADDON_RATE_LIMIT, else None.
    Shared across replicas through Redis on the remote backend."""
    user_key = f"{scope.org_id or '-'}:{scope.user_id or '-'}"
    if not remote:
        over = _local_rate.hit(user_key, limit=ADDON_RATE_LIMIT, window_s=ADDON_RATE_WINDOW_S)
        return queued("rate_limited") if over else None
    try:
        over = await remote_rate_limited(user_key, limit=ADDON_RATE_LIMIT, window_s=ADDON_RATE_WINDOW_S)
    except RedisError:
        logger.warning("[addon] rate limit check failed; queueing", exc_info=True)
        return queued("redis_error")
    return queued("rate_limited") if over else None


def budget_deadline() -> float:
    """The loop-clock time one POST /addon must answer by (ADDON_BUDGET_S)."""
    return time.monotonic() + ADDON_BUDGET_S


def _left(deadline: float) -> float:
    return deadline - time.monotonic()


async def _staged(deadline: float, reason: QueuedReason, make) -> dict | None:
    """Run the stage ``make()`` in what is left of the budget; queued with
    *reason* when the budget runs out first."""
    left = _left(deadline)
    if left <= 0:
        return queued(reason)
    try:
        async with asyncio.timeout(left) as stage:
            return await make()
    except TimeoutError:
        if not stage.expired():
            raise
        logger.info("[addon] the request's budget ran out; queueing as %s", reason)
        return queued(reason)


class _Busy(Exception):
    """The turn's decision lock did not free in time."""


@asynccontextmanager
async def _holding(lock: asyncio.Lock, deadline: float):
    """Hold *lock*, waiting at most DECIDE_LOCK_TIMEOUT_S for it, and never
    past *deadline* (_Busy)."""
    try:
        async with asyncio.timeout(min(DECIDE_LOCK_TIMEOUT_S, _left(deadline))):
            await lock.acquire()
    except TimeoutError:
        raise _Busy from None
    try:
        yield
    finally:
        lock.release()


# ── In-process turns ───────────────────────────────────────────────────────


async def route_addon(
    *, channel: AddonChannel, text: str, force: bool, scope: TenantScope, local_key: str | None = None,
    deadline: float | None = None, queue_settings: TurnQueueSettings | None = None,
) -> dict:
    """Add *text* to the turn *channel* belongs to, or say why it is queued,
    by *deadline* (budget_deadline() when not given). *queue_settings* is the
    request's (the endpoint reads it once); read here when not given. Never
    raises."""
    deadline = budget_deadline() if deadline is None else deadline
    queue_settings = queue_settings or TurnQueueSettings()
    try:
        async with _holding(channel.deciding, deadline):
            limit = channel.limit_reason(text)
            if limit is not None:
                return queued(limit)
            if force:
                addon_id = channel.add(text, decided_by="user", local_key=local_key)
                return added(addon_id, "user") if addon_id else queued("no_turn")
            return await _ask_jev(
                channel=channel, text=text, scope=scope, local_key=local_key, deadline=deadline,
                queue_settings=queue_settings,
            )
    except _Busy:
        return queued("busy")
    except Exception:
        # Kept broad: the endpoint's contract is that nothing raises (a
        # failure queues the message, as without this feature), and anton
        # raising from add_user_input lands here. Not Jev: its failures come
        # back as a verdict.
        logger.exception("[addon] routing failed; queueing", extra={"conversation_id": channel.conversation_id})
        return queued("internal")


async def _ask_jev(
    *, channel: AddonChannel, text: str, scope: TenantScope, local_key: str | None, deadline: float,
    queue_settings: TurnQueueSettings,
) -> dict:
    refusal = await _staged(deadline, "jev_timeout", partial(
        _jev_refusal,
        conversation_id=channel.conversation_id,
        task=channel.task,
        progress=channel.progress(),
        added=channel.added_texts(),
        text=text,
        scope=scope,
        deadline=deadline,
        queue_settings=queue_settings,
    ))
    if refusal is not None:
        return refusal
    addon_id = channel.add(text, decided_by="jev", local_key=local_key)
    return added(addon_id, "jev") if addon_id else queued("no_turn")


async def _jev_refusal(
    *, conversation_id: str, task: str, progress: list[str], added: list[str], text: str, scope: TenantScope,
    deadline: float | None = None, queue_settings: TurnQueueSettings | None = None,
) -> dict | None:
    """Ask Jev whether *text* belongs to the running task. None when it does,
    else the queued response saying why not. *queue_settings* is the
    request's; read here when not given."""
    from cowork.db.units import run_on_unit_threads

    queue_settings = queue_settings or TurnQueueSettings()
    try:
        found = await run_on_unit_threads(partial(_jev_target, scope, queue_settings))
    except SQLAlchemyError:
        # The settings read failed. Anything else is a bug: it reaches the
        # route's own handler, which logs it as an error and queues it as
        # internal too.
        logger.warning("[addon] could not resolve the decision target; queueing", exc_info=True)
        return queued("internal")
    if found is None:
        return queued("jev_off")
    # The correlation id of a key minted for this decision, set before the
    # mint: one that failed or was cancelled may still have issued a key, and
    # revoking by correlation id is a no-op when none was.
    minted_for: str | None = None
    try:
        if isinstance(found, _MintKey):
            from cowork.turnqueue import producer

            minted_for = str(uuid4())
            try:
                block = await producer.mint_llm_block(
                    org_id=scope.org_id, user_id=scope.user_id, correlation_id=minted_for,
                    settings=queue_settings, workspace_id=found.workspace_id,
                    ttl_seconds=minted_key_ttl(queue_settings),
                )
            except Exception:
                # Kept broad: the mint calls the auth service, whose failures
                # (transport, a refusal, a bad answer) are no one family, and
                # this function answers for itself (it is called directly, not
                # only from the routes). The finally below revokes whatever it
                # may have issued.
                logger.warning("[addon] could not mint a key to ask Jev with; queueing", exc_info=True)
                return queued("internal")
            target, model = jev_route.JevTarget.mindshub(block["base_url"], block["api_key"]), found.model
        else:
            target, model = found
        timeout_s = queue_settings.jev_route_timeout_seconds
        if deadline is not None:
            # Jev's own timeout, cut to the request's budget, so a slow Jev
            # comes back as its own timeout verdict rather than a cancel.
            timeout_s = max(0.001, min(timeout_s, _left(deadline)))
        return await _ask_jev_with(
            target=target, model=model, conversation_id=conversation_id,
            task=task, progress=progress, added=added, text=text, queue_settings=queue_settings,
            timeout_s=timeout_s,
        )
    finally:
        if minted_for is not None:
            from cowork.turnqueue import producer

            # Used for this one decision only, and revoked on every way out,
            # cancellation included. Shielded, so a second cancel can't stop
            # the revoke; a failed one leaves the key to its short expiry.
            await asyncio.shield(producer.revoke_turn_key_quietly(minted_for))


def minted_key_ttl(settings: TurnQueueSettings) -> int:
    """How long a key minted for one decision lives: just past the Jev call's
    own timeout, so a revoke that fails leaves it usable for seconds."""
    return max(_MINTED_KEY_MIN_TTL_S, math.ceil(settings.jev_route_timeout_seconds) + _MINTED_KEY_HEADROOM_S)


async def _ask_jev_with(
    *, target: jev_route.JevTarget, model: str, conversation_id: str, task: str,
    progress: list[str], added: list[str], text: str, queue_settings: TurnQueueSettings, timeout_s: float,
) -> dict | None:
    # Scrubbed like the route gate's history: the state leaves this server.
    # The task and each added message are cut to MAX_ADDON_CHARS (the new
    # message already fits it), so a long question can't blow up the call.
    state = jev_route.addon_state(
        task=scrub_credentials(task[:MAX_ADDON_CHARS]),
        # Step descriptions are the model's own words, so scrubbed too.
        progress=[scrub_credentials(line) for line in progress],
        added=[scrub_credentials(t[:MAX_ADDON_CHARS]) for t in added],
        message=scrub_credentials(text),
    )
    from anton.core.llm.tracing import TraceContext, reset_trace_context, set_trace_context

    # Attributed to the conversation like the route gate's decisions, so the
    # gateway doesn't stamp it as a direct API call on the user's key.
    token = set_trace_context(TraceContext(
        session_id=conversation_id, harness="anton", tags=(jev_route.JEV_ROUTE_TAG,),
    ))
    try:
        verdict = await jev_route.decide_addon(
            target=target,
            state=state,
            model=model,
            timeout_seconds=timeout_s,
            min_p_add=queue_settings.addon_min_p_add,
        )
    finally:
        reset_trace_context(token)
    logger.info(
        "[addon] jev verdict add=%s p_add=%s ms=%d error=%s",
        verdict.add, verdict.p_add, verdict.ms, verdict.error or "-",
        extra={"conversation_id": conversation_id},
    )
    if verdict.error:
        return queued("jev_timeout" if verdict.error == "timeout" else "jev_error")
    if not verdict.add:
        return queued("low_confidence" if verdict.chose_add else "next_task")
    return None


# ── Remote (pod) turns ─────────────────────────────────────────────────────

# One decision at a time per turn, as AddonChannel.deciding does in-process,
# so a follow-up is judged with the one before it already added. Per process:
# two replicas deciding for one turn at once is a person sending from two
# tabs within a Jev call, and the worst case is a follow-up judged without
# its predecessor.
@dataclass
class _TurnLock:
    lock: asyncio.Lock
    # Holders and waiters: the entry goes only when none remain, so a waiter
    # never ends up on a lock a newcomer no longer finds.
    users: int = 0


_remote_deciding: dict[str, _TurnLock] = {}


@asynccontextmanager
async def _deciding(correlation_id: str, deadline: float):
    entry = _remote_deciding.get(correlation_id)
    if entry is None:
        entry = _remote_deciding[correlation_id] = _TurnLock(asyncio.Lock())
    entry.users += 1
    try:
        async with _holding(entry.lock, deadline):
            yield
    finally:
        entry.users -= 1
        if entry.users == 0 and _remote_deciding.get(correlation_id) is entry:
            del _remote_deciding[correlation_id]


def _content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, dict):
        content = [content]
    if isinstance(content, list):
        return " ".join(
            block["text"] for block in content
            if isinstance(block, dict) and isinstance(block.get("text"), str)
        )
    return ""


def _running_task(session, *, conversation_id: UUID) -> str:
    """The text of the question the in-flight turn runs: the latest pending
    user message (the producer saves it pending at turn start)."""
    from cowork.models.message import Message

    message = session.exec(
        session.select(Message)
        .where(Message.conversation_id == conversation_id)
        .where(Message.role == "user")
        .where(Message.pending == True)  # noqa: E712 - SQL boolean column
        .order_by(Message.seq.desc())
        .limit(1)
    ).first()
    return _content_text(message.content) if message is not None else ""


async def _remote_jev_inputs(
    *, conversation_id: str, correlation_id: str, scope: TenantScope,
) -> tuple[str, list[str], list[str]]:
    """What Jev reads about a pod's turn without an in-process channel: its
    question from the database, its recent tool calls from the replies the
    pod relayed, and the messages it already took. A part that can't be read
    is left empty rather than failing the decision; the already-added
    messages are the exception, since without them a follow-up is judged
    as if it were the first (raises RedisError)."""
    from cowork.db.units import run_db

    # Both reads below are kept broad: each only adds context for Jev, and a
    # decision made without it beats none (the docstring's contract).
    task = ""
    try:
        task = await run_db(partial(_running_task, conversation_id=UUID(conversation_id)), scope=scope)
    except Exception:
        logger.warning("[addon] could not read the running task", exc_info=True,
                       extra={"conversation_id": conversation_id})
    progress: list[str] = []
    try:
        progress = await remote_progress(conversation_id=conversation_id, correlation_id=correlation_id)
    except Exception:
        logger.warning("[addon] could not read the running turn's progress", exc_info=True,
                       extra={"conversation_id": conversation_id})
    already = await remote_added_texts(correlation_id)
    return task, progress, already


_REMOTE_ADD_FAILURES: dict[RemoteAddonResult, QueuedReason] = {
    RemoteAddonResult.REFUSED: "no_turn",
    RemoteAddonResult.NOT_FOUND: "no_turn",
    RemoteAddonResult.TIMEOUT: "pod_timeout",
    RemoteAddonResult.TOO_LARGE: "too_large",
    RemoteAddonResult.REDIS_ERROR: "redis_error",
}


async def _remote_jev_refusal(
    *, conversation_id: str, correlation_id: str, text: str, scope: TenantScope, deadline: float,
    queue_settings: TurnQueueSettings,
) -> dict | None:
    task, progress, already = await _remote_jev_inputs(
        conversation_id=conversation_id, correlation_id=correlation_id, scope=scope,
    )
    return await _jev_refusal(
        conversation_id=conversation_id, task=task, progress=progress,
        added=already, text=text, scope=scope, deadline=deadline, queue_settings=queue_settings,
    )


async def route_remote_addon(
    *, conversation_id: str, correlation_id: str, text: str, force: bool, scope: TenantScope,
    local_key: str | None = None, deadline: float | None = None,
    queue_settings: TurnQueueSettings | None = None,
) -> dict:
    """``route_addon`` for a turn that runs in a pod: decide with Jev here
    (``force`` skips it), then hand the message to the pod and return its
    verdict (``turnqueue/addons.py``), all by *deadline* (budget_deadline()
    when not given). *queue_settings* is the request's; read here when not
    given. Never raises."""
    deadline = budget_deadline() if deadline is None else deadline
    queue_settings = queue_settings or TurnQueueSettings()
    try:
        async with _deciding(correlation_id, deadline):
            count, total_bytes = await remote_addon_totals(correlation_id)
            limit = turn_limit_reason(count=count, total_bytes=total_bytes, text=text)
            if limit is not None:
                return queued(limit)
            if force:
                decided_by = "user"
            else:
                # What Jev reads, a minted key and Jev itself: one stage.
                refusal = await _staged(deadline, "jev_timeout", partial(
                    _remote_jev_refusal, conversation_id=conversation_id, correlation_id=correlation_id,
                    text=text, scope=scope, deadline=deadline, queue_settings=queue_settings,
                ))
                if refusal is not None:
                    return refusal
                decided_by = "jev"
            result, addon_id = await submit_remote_addon(
                conversation_id=conversation_id, correlation_id=correlation_id,
                text=text, decided_by=decided_by, local_key=local_key, deadline=deadline,
            )
    except _Busy:
        return queued("busy")
    except RedisError:
        logger.warning("[addon] remote routing hit a Redis error; queueing", exc_info=True,
                       extra={"conversation_id": conversation_id})
        return queued("redis_error")
    except Exception:
        # Kept broad, as in route_addon: nothing here raises to the endpoint.
        logger.exception("[addon] remote routing failed; queueing", extra={"conversation_id": conversation_id})
        return queued("internal")
    if result is RemoteAddonResult.ADDED:
        return added(addon_id, decided_by)
    return queued(_REMOTE_ADD_FAILURES[result])
