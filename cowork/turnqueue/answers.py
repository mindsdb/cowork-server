"""Deliver an ask_user answer to a question a remote (pod) turn is blocked on.

The answer is queued in Redis for scratchpad-controller, which writes it to
the pod's stdin. The pod alone decides whether it closes the question and says
so on the reply stream, naming the answer's `answer_id`; this module waits for
that verdict so /answer can return the same statuses as the in-process broker.
Works from any replica: nothing here depends on owning the turn's producer.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from enum import StrEnum

import redis.asyncio as aioredis
from redis.exceptions import RedisError

from cowork.turnqueue.models import TurnReply
from cowork.turnqueue.redis_client import (
    REDIS_BLOCK_MS,
    answer_queue_key,
    get_redis,
    redis_deadline,
    reply_stream_key,
    xread_blocking,
)

logger = logging.getLogger(__name__)

#: How long /answer waits for the pod's verdict. Past it the answer stays
#: queued and the call reports it accepted: an error would make the client
#: retry, get 409 and resend a typed answer as a new message.
ACK_TIMEOUT_S = 15.0
#: A queued answer nobody pops (the turn ended) must not linger.
ANSWER_TTL_S = 60
#: How long past its BLOCK a verdict read may wait for Redis's reply. The
#: client cuts a read's wait to the time left before its deadline, so a read
#: whose deadline fell with its BLOCK would time out just as Redis answered
#: an empty read, and the wait would end as a Redis error.
READ_GRACE_S = 0.1
#: scratchpad-controller drops any stdin line over this many raw UTF-8 bytes
#: (its own MAX_ANSWER_BYTES); past that /answer would wait the full ack
#: timeout for a verdict that never comes. Reject it here instead.
MAX_ANSWER_BYTES = 64 * 1024
#: Upper bound on the bytes the controller's re-wrap adds to an entry
#: ("kind":"answer", fields it defaults, the trailing newline).
WRAP_HEADROOM_BYTES = 256
#: The reply kinds that end a turn: after one, the pod answers nothing more.
TERMINAL_KINDS = frozenset({"turn_completed", "turn_failed"})
#: The scan stops at this turn's latest reply and skips other turns'. Turns on
#: one conversation are serialised by the controller, so at most one later turn
#: can have started — and then the turn index already names it, so /answer never
#: gets here with the old correlation id. 10 is plenty; do not raise it "to be safe".
TAIL_COUNT = 10


class RemoteAnswerResult(StrEnum):
    ACCEPTED = "accepted"
    INVALID_OPTION = "invalid_option"
    NOT_FOUND = "not_found"
    ALREADY_ANSWERED = "already_answered"
    TOO_LARGE = "too_large"


_REJECTIONS = {
    "invalid_option": RemoteAnswerResult.INVALID_OPTION,
    "not_found": RemoteAnswerResult.NOT_FOUND,
    "already_answered": RemoteAnswerResult.ALREADY_ANSWERED,
}


def fits_stdin_line(entry: str) -> bool:
    """Whether *entry*, once the controller re-wraps it, fits the pod's stdin
    line (MAX_ANSWER_BYTES). Shared with turnqueue/addons.py."""
    return len(entry.encode("utf-8")) + WRAP_HEADROOM_BYTES <= MAX_ANSWER_BYTES


def own_reply(fields: dict, correlation_id: str) -> TurnReply | None:
    """This turn's reply in a reply-stream entry, or None for another turn's
    or one this build can't parse. Shared with turnqueue/addons.py."""
    # pydantic's ValidationError is a ValueError: an entry with a kind this
    # build's `TurnReply.kind` Literal does not know is skipped, not raised.
    try:
        reply = TurnReply.model_validate_json(fields["payload"])
    except (KeyError, ValueError):
        return None
    return reply if reply.correlation_id == correlation_id else None


async def submit_remote_answer(
    *,
    conversation_id: str,
    correlation_id: str,
    question_id: str,
    payload: dict,
    r: aioredis.Redis | None = None,
    ack_timeout_s: float = ACK_TIMEOUT_S,
) -> RemoteAnswerResult:
    r = r or get_redis()
    stream = reply_stream_key(conversation_id)

    # Before pushing: the read position must precede the pod's verdict, and a
    # turn whose terminal reply is already on the stream (buffer not yet
    # closed, so the turn index still says in flight) will never answer.
    tail = await r.xrevrange(stream, count=TAIL_COUNT)
    last_id = tail[0][0] if tail else "0-0"
    for _entry_id, fields in tail:  # newest first
        reply = own_reply(fields, correlation_id)
        if reply is None:
            continue
        if reply.kind in TERMINAL_KINDS:
            return RemoteAnswerResult.NOT_FOUND
        break

    answer_id = uuid.uuid4().hex
    key = answer_queue_key(correlation_id)
    # Compact and ensure_ascii=False, so the size here tracks the raw UTF-8
    # size the controller measures (the default \uXXXX escaping inflates
    # non-Latin text). The headroom covers what the controller adds when it
    # re-wraps the entry as the pod's stdin line, so an entry accepted here is
    # never silently dropped there.
    entry = json.dumps(
        {"question_id": question_id, "answer_id": answer_id, **payload},
        separators=(",", ":"), ensure_ascii=False,
    )
    if not fits_stdin_line(entry):
        return RemoteAnswerResult.TOO_LARGE
    # Retried like any other call. A transaction resent after its reply was
    # lost queues this entry twice, with one answer_id: the pod answers the
    # question once, and nothing renders its verdict on the copy. Raised, the
    # lost reply would fail /answer, and the client would resend the typed
    # text as a new message.
    async with r.pipeline(transaction=True) as pipe:
        pipe.rpush(key, entry)
        pipe.expire(key, ANSWER_TTL_S)
        await pipe.execute()

    try:
        deadline = time.monotonic() + ack_timeout_s
        while (left := deadline - time.monotonic()) > 0:
            block_ms = max(1, min(REDIS_BLOCK_MS, int(left * 1000)))
            # A read's retries end with the wait, not after it.
            with redis_deadline(budget_s=left + READ_GRACE_S):
                resp = await xread_blocking(r=r, streams={stream: last_id}, count=50, block_ms=block_ms)
            for _stream, entries in resp or []:
                for entry_id, fields in entries:
                    last_id = entry_id
                    reply = own_reply(fields, correlation_id)
                    if reply is None:
                        continue
                    if reply.kind in TERMINAL_KINDS:
                        return RemoteAnswerResult.NOT_FOUND
                    data = reply.data or {}
                    if reply.kind != "turn_step" or data.get("answer_id") != answer_id:
                        continue
                    if data.get("step") == "ask_user_answered":
                        return RemoteAnswerResult.ACCEPTED
                    if data.get("step") == "ask_user_answer_rejected":
                        reason = data.get("reason")
                        if reason not in _REJECTIONS:
                            logger.warning(
                                "ask_user answer rejected with an unrecognised reason, treating as "
                                "invalid_option conversation=%s correlation_id=%s question_id=%s "
                                "answer_id=%s reason=%r",
                                conversation_id, correlation_id, question_id, answer_id, reason,
                            )
                        return _REJECTIONS.get(reason, RemoteAnswerResult.INVALID_OPTION)
    except RedisError:
        # The answer is already queued (RPUSH above succeeded): propagating
        # here would 500 the request, the client would retry, get 409 (a
        # second RPUSH the pod never asked for) and resend the typed text as
        # a new message — a visible duplicate of an answer that may well have
        # been delivered. Report it accepted instead, same as a plain ack
        # timeout; the card settles from SSE once the pod's verdict lands.
        logger.warning(
            "ask_user ack wait hit a Redis error, reporting accepted conversation=%s "
            "correlation_id=%s question_id=%s answer_id=%s",
            conversation_id, correlation_id, question_id, answer_id, exc_info=True,
        )
        return RemoteAnswerResult.ACCEPTED
    logger.warning(
        "ask_user answer not confirmed within %.0fs conversation=%s correlation_id=%s "
        "question_id=%s answer_id=%s",
        ack_timeout_s, conversation_id, correlation_id, question_id, answer_id,
    )
    return RemoteAnswerResult.ACCEPTED
