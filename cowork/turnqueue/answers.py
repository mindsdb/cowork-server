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

from cowork.turnqueue.models import TurnReply
from cowork.turnqueue.redis_client import answer_queue_key, get_redis

logger = logging.getLogger(__name__)

#: How long /answer waits for the pod's verdict. Past it the answer stays
#: queued and the call reports it accepted: an error would make the client
#: retry, get 409 and resend a typed answer as a new message.
ACK_TIMEOUT_S = 15.0
#: A queued answer nobody pops (the turn ended) must not linger.
ANSWER_TTL_S = 60
_TERMINAL = frozenset({"turn_completed", "turn_failed"})
#: The scan stops at this turn's latest reply and skips other turns'. Turns on
#: one conversation are serialised by the controller, so at most one later turn
#: can have started — and then the turn index already names it, so /answer never
#: gets here with the old correlation id. 10 is plenty; do not raise it "to be safe".
_TAIL_COUNT = 10


class RemoteAnswerResult(StrEnum):
    ACCEPTED = "accepted"
    INVALID_OPTION = "invalid_option"
    NOT_FOUND = "not_found"
    ALREADY_ANSWERED = "already_answered"


_REJECTIONS = {
    "invalid_option": RemoteAnswerResult.INVALID_OPTION,
    "not_found": RemoteAnswerResult.NOT_FOUND,
    "already_answered": RemoteAnswerResult.ALREADY_ANSWERED,
}


def _own_reply(fields: dict, correlation_id: str) -> TurnReply | None:
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
    stream = f"scratchpad:reply:{conversation_id}"

    # Before pushing: the read position must precede the pod's verdict, and a
    # turn whose terminal reply is already on the stream (buffer not yet
    # closed, so the turn index still says in flight) will never answer.
    tail = await r.xrevrange(stream, count=_TAIL_COUNT)
    last_id = tail[0][0] if tail else "0-0"
    for _entry_id, fields in tail:  # newest first
        reply = _own_reply(fields, correlation_id)
        if reply is None:
            continue
        if reply.kind in _TERMINAL:
            return RemoteAnswerResult.NOT_FOUND
        break

    answer_id = uuid.uuid4().hex
    key = answer_queue_key(correlation_id)
    entry = json.dumps({"question_id": question_id, "answer_id": answer_id, **payload})
    async with r.pipeline(transaction=True) as pipe:
        pipe.rpush(key, entry)
        pipe.expire(key, ANSWER_TTL_S)
        await pipe.execute()

    deadline = time.monotonic() + ack_timeout_s
    while (left := deadline - time.monotonic()) > 0:
        resp = await r.xread({stream: last_id}, count=50, block=max(1, int(left * 1000)))
        for _stream, entries in resp or []:
            for entry_id, fields in entries:
                last_id = entry_id
                reply = _own_reply(fields, correlation_id)
                if reply is None:
                    continue
                if reply.kind in _TERMINAL:
                    return RemoteAnswerResult.NOT_FOUND
                data = reply.data or {}
                if reply.kind != "turn_step" or data.get("answer_id") != answer_id:
                    continue
                if data.get("step") == "ask_user_answered":
                    return RemoteAnswerResult.ACCEPTED
                if data.get("step") == "ask_user_answer_rejected":
                    return _REJECTIONS.get(data.get("reason"), RemoteAnswerResult.INVALID_OPTION)
    logger.warning(
        "ask_user answer not confirmed within %.0fs conversation=%s correlation_id=%s "
        "question_id=%s answer_id=%s",
        ack_timeout_s, conversation_id, correlation_id, question_id, answer_id,
    )
    return RemoteAnswerResult.ACCEPTED
