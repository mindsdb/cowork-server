"""Preview checks for remote (pod) turns (ENG-4227).

A hosted turn runs anton in a scratchpad pod, so the in-process
``preview_checker`` never reaches it. The pod's anton asks instead: it writes
an ``artifact_check`` step (``check_id``, ``artifact_slug``, ``timeout_s``)
on its stdout, which scratchpad-controller relays as a ``turn_step``. The
turn producer records the check here (``register_remote_check``) with the
turn's identity and owner, and puts the same ``response.artifact_check``
frame an in-process turn sends on the conversation stream. The app answers
with the existing ``POST /api/v1/artifacts/checks/{check_id}``; the endpoint
finds the check here when its own process holds no such check
(``submit_remote_check``), authorizes the caller against the turn's owner,
and queues the answer for the pod on the queue ask_user answers take
(``answer_queue_key``): ``{"kind": "check_result", "check_id", "body"}``,
which scratchpad-controller writes to the pod's stdin.

In Redis rather than the in-process broker for the reason ask_user answers
are (turnqueue/answers.py): the app's POST may reach any replica, not the
one whose producer relays the turn. The pod keeps its own timeout, and
resolves an unanswered check to "unverified" when the turn ends, so nothing
here waits.
"""

from __future__ import annotations

import json
import logging
import math
import re

import redis.asyncio as aioredis
from redis.exceptions import RedisError

from cowork.streaming.preview_checks import CheckSubmitResult
from cowork.streaming.turn_index import TurnOwner
from cowork.turnqueue.answers import ANSWER_TTL_S, fits_stdin_line
from cowork.turnqueue.redis_client import answer_queue_key, get_redis

logger = logging.getLogger(__name__)

__all__ = [
    "ARTIFACT_CHECK_STEP",
    "RemoteCheck",
    "check_result_entry",
    "queue_check_result",
    "register_remote_check",
    "submit_remote_check",
]

#: The pod's step (scratchpad-controller relays it as turn_step "artifact_check").
ARTIFACT_CHECK_STEP = "artifact_check"
#: An id the pod mints the way the in-process broker does: "acheck:" + 128 bits.
CHECK_ID = re.compile(r"^acheck:[0-9a-f]{24,64}$")
_MIN_TIMEOUT_S = 5
_MAX_TIMEOUT_S = 120
#: Past the pod's own timeout, so a slow answer still finds its record.
_RECORD_GRACE_S = 60
_MAX_SLUG = 200


def _record_key(check_id: str) -> str:
    return f"cowork:acheck:{check_id}"


def _answered_key(check_id: str) -> str:
    return f"cowork:acheck:{check_id}:answered"


class RemoteCheck(dict):
    """A recorded check: ``conversation_id``, ``correlation_id``,
    ``artifact_slug``, ``timeout_s``, ``owner`` (TurnOwner)."""


def _timeout(value: object) -> int:
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return _MIN_TIMEOUT_S
    if math.isnan(number):
        return _MIN_TIMEOUT_S
    if math.isinf(number):
        return _MAX_TIMEOUT_S if number > 0 else _MIN_TIMEOUT_S
    return min(max(math.ceil(number), _MIN_TIMEOUT_S), _MAX_TIMEOUT_S)


async def register_remote_check(
    data: dict,
    *,
    conversation_id: str,
    correlation_id: str,
    org_id: str | None,
    user_id: str | None,
    r: aioredis.Redis | None = None,
) -> RemoteCheck | None:
    """Record the pod's check for the app's answer; the check to relay, or
    None for one this server won't relay (a malformed id or slug, an id
    already recorded, Redis down). Never raises: the pod then times out and
    reads the check as unverified."""
    check_id = data.get("check_id")
    slug = str(data.get("artifact_slug") or "").strip()
    if not isinstance(check_id, str) or not CHECK_ID.fullmatch(check_id) or not slug or len(slug) > _MAX_SLUG:
        logger.info("preview check step dropped: malformed correlation_id=%s", correlation_id)
        return None
    timeout = _timeout(data.get("timeout_s"))
    record = {
        "conversation_id": conversation_id,
        "correlation_id": correlation_id,
        "org_id": org_id,
        "user_id": user_id,
        "artifact_slug": slug,
        "timeout_s": timeout,
    }
    r = r or get_redis()
    try:
        stored = await r.set(_record_key(check_id), json.dumps(record), nx=True, ex=timeout + _RECORD_GRACE_S)
    except RedisError:
        logger.warning("preview check not recorded: Redis failed correlation_id=%s", correlation_id, exc_info=True)
        return None
    if not stored:
        # The pod named an id it already used: relaying it twice would let
        # one answer reach two checks.
        return None
    return _check(record, check_id)


def _check(record: dict, check_id: str) -> RemoteCheck:
    return RemoteCheck(
        check_id=check_id,
        conversation_id=str(record.get("conversation_id") or ""),
        correlation_id=str(record.get("correlation_id") or ""),
        artifact_slug=str(record.get("artifact_slug") or ""),
        timeout_s=int(record.get("timeout_s") or _MIN_TIMEOUT_S),
        owner=TurnOwner(org_id=record.get("org_id"), user_id=record.get("user_id")),
    )


async def load_remote_check(check_id: str, *, r: aioredis.Redis | None = None) -> RemoteCheck | None:
    """The recorded check, or None (never recorded, or expired)."""
    if not CHECK_ID.fullmatch(check_id or ""):
        return None
    r = r or get_redis()
    raw = await r.get(_record_key(check_id))
    if raw is None:
        return None
    try:
        record = json.loads(raw)
    except ValueError:
        return None
    return _check(record, check_id) if isinstance(record, dict) else None


def check_result_entry(check_id: str, body: dict | None) -> str:
    """The queue entry scratchpad-controller writes to the pod's stdin. A body
    too large for the pod's line keeps its verdict and sheds diagnostics."""
    candidate = dict(body) if isinstance(body, dict) else None
    while True:
        entry = json.dumps(
            {"kind": "check_result", "check_id": check_id, "body": candidate},
            separators=(",", ":"), ensure_ascii=False,
        )
        if fits_stdin_line(entry) or candidate is None:
            return entry
        errors = list(candidate.get("errors") or [])
        if not errors:
            candidate = {"status": candidate.get("status"), "errors": [], "ms": candidate.get("ms", 0)}
            entry = json.dumps({"kind": "check_result", "check_id": check_id, "body": candidate},
                               separators=(",", ":"), ensure_ascii=False)
            return entry
        candidate["errors"] = errors[: len(errors) // 2]


async def queue_check_result(
    *, correlation_id: str, check_id: str, body: dict | None, r: aioredis.Redis | None = None,
) -> None:
    """Queue ``body`` (None: no answer) for the pod's stdin."""
    r = r or get_redis()
    async with r.pipeline(transaction=True) as pipe:
        pipe.rpush(answer_queue_key(correlation_id), check_result_entry(check_id, body))
        pipe.expire(answer_queue_key(correlation_id), ANSWER_TTL_S)
        await pipe.execute()


async def submit_remote_check(
    check: RemoteCheck, answer: dict, *, r: aioredis.Redis | None = None,
) -> CheckSubmitResult:
    """Deliver the app's answer to the pod: once per check. The caller has
    already authorized the caller against ``check["owner"]``."""
    r = r or get_redis()
    first = await r.set(_answered_key(check["check_id"]), "1", nx=True, ex=check["timeout_s"] + _RECORD_GRACE_S)
    if not first:
        return CheckSubmitResult.ALREADY_ANSWERED
    await queue_check_result(correlation_id=check["correlation_id"], check_id=check["check_id"], body=answer, r=r)
    return CheckSubmitResult.ACCEPTED
