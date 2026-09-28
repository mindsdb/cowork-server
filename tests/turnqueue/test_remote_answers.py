"""submit_remote_answer: queue an answer for a pod turn, return the pod's verdict."""

from __future__ import annotations

import asyncio
import json
import logging
from contextlib import contextmanager

import fakeredis.aioredis
import pytest
from redis.exceptions import RedisError

from cowork.turnqueue.answers import MAX_ANSWER_BYTES, RemoteAnswerResult, submit_remote_answer

STREAM = "scratchpad:reply:conv-1"
KEY = "cowork:answer:corr-1"


@pytest.fixture
def r():
    return fakeredis.aioredis.FakeRedis(decode_responses=True)


def _reply(kind, data, corr="corr-1"):
    return {"payload": json.dumps({"correlation_id": corr, "kind": kind, "data": data})}


async def _queued_answer(r):
    for _ in range(300):
        raw = await r.lindex(KEY, 0)
        if raw:
            return json.loads(raw)
        await asyncio.sleep(0.01)
    raise AssertionError("no answer was queued")


async def _pod_says(r, step, corr="corr-1", **data):
    answer = await _queued_answer(r)
    await r.xadd(STREAM, _reply("turn_step", {"step": step, "id": answer["question_id"],
                                              "answer_id": answer["answer_id"], **data}, corr))


def _submit(r, timeout=2.0, payload=None):
    return submit_remote_answer(conversation_id="conv-1", correlation_id="corr-1",
                                question_id="ask:1", payload=payload or {"values": ["pg"]},
                                r=r, ack_timeout_s=timeout)


@contextmanager
def _records_from(name: str, level: int = logging.WARNING):
    """Capture one logger's records, with that logger forced back on.

    Not caplog: the alembic env used by the migration tests calls fileConfig,
    which defaults to disable_existing_loggers=True and leaves every logger
    built before it with disabled=True for the rest of the session. A log
    assertion running after one of those silently captures nothing, so this
    clears the flag for the duration and puts it back. Same trap and fix as
    tests/test_turn_errors.py's ``_records_from``.
    """
    captured: list[logging.LogRecord] = []

    class _Capture(logging.Handler):
        def emit(self, record):
            captured.append(record)

    target = logging.getLogger(name)
    handler = _Capture(level)
    restore = (target.level, target.propagate, target.disabled)
    target.setLevel(level)
    target.propagate = False
    target.disabled = False
    target.addHandler(handler)
    try:
        yield captured
    finally:
        target.removeHandler(handler)
        target.level, target.propagate, target.disabled = restore


async def test_accepted_when_the_pod_closes_the_question(r):
    pod = asyncio.create_task(_pod_says(r, "ask_user_answered", status="answered"))
    assert await _submit(r) is RemoteAnswerResult.ACCEPTED
    await pod
    entry = json.loads(await r.lindex(KEY, 0))
    assert entry["question_id"] == "ask:1"
    assert entry["values"] == ["pg"]
    assert len(entry["answer_id"]) == 32
    assert 0 < await r.ttl(KEY) <= 60


@pytest.mark.parametrize("reason,expected", [
    ("invalid_option", RemoteAnswerResult.INVALID_OPTION),
    ("not_found", RemoteAnswerResult.NOT_FOUND),
    ("already_answered", RemoteAnswerResult.ALREADY_ANSWERED),
])
async def test_rejections_map_to_their_result(r, reason, expected):
    pod = asyncio.create_task(_pod_says(r, "ask_user_answer_rejected", reason=reason))
    assert await _submit(r) is expected
    await pod


async def test_an_unrecognised_rejection_reason_falls_back_to_invalid_option_and_warns(r):
    pod = asyncio.create_task(_pod_says(r, "ask_user_answer_rejected", reason="some_future_reason"))
    with _records_from("cowork.turnqueue.answers") as records:
        assert await _submit(r) is RemoteAnswerResult.INVALID_OPTION
    await pod
    text = "\n".join(record.getMessage() for record in records)
    assert "some_future_reason" in text
    assert "pg" not in text  # answer content never logged


async def test_the_turn_ending_first_is_not_found(r):
    async def turn_ends():
        await _queued_answer(r)
        await r.xadd(STREAM, _reply("turn_completed", {}))
    task = asyncio.create_task(turn_ends())
    assert await _submit(r) is RemoteAnswerResult.NOT_FOUND
    await task


async def test_a_terminal_already_on_the_stream_is_not_found_and_nothing_is_queued(r):
    await r.xadd(STREAM, _reply("turn_step", {"step": "tool_start"}))
    await r.xadd(STREAM, _reply("turn_failed", {"error": "boom"}))
    await r.xadd(STREAM, _reply("turn_delta", {"text": "x"}, corr="other-turn"))
    assert await _submit(r) is RemoteAnswerResult.NOT_FOUND
    assert await r.llen(KEY) == 0


async def test_a_previous_turns_terminal_does_not_count(r):
    await r.xadd(STREAM, _reply("turn_completed", {}, corr="previous-turn"))
    await r.xadd(STREAM, _reply("turn_step", {"step": "ask_user", "id": "ask:1"}))
    pod = asyncio.create_task(_pod_says(r, "ask_user_answered", status="answered"))
    assert await _submit(r) is RemoteAnswerResult.ACCEPTED
    await pod


async def test_acks_for_another_answer_or_turn_are_ignored(r):
    async def noise():
        answer = await _queued_answer(r)
        await r.xadd(STREAM, _reply("turn_step", {"step": "ask_user_answered", "id": "ask:1",
                                                  "answer_id": "someone-else"}))
        await r.xadd(STREAM, _reply("turn_step", {"step": "ask_user_answered", "id": "ask:1",
                                                  "answer_id": answer["answer_id"]}, corr="other"))
    task = asyncio.create_task(noise())
    with _records_from("cowork.turnqueue.answers") as records:
        # No verdict for OUR answer within the window: queued, reported as accepted.
        assert await _submit(r, timeout=0.3) is RemoteAnswerResult.ACCEPTED
    await task
    text = "\n".join(record.getMessage() for record in records)
    assert "not confirmed" in text
    assert "pg" not in text  # answer content never logged


async def test_an_oversized_entry_is_rejected_and_nothing_is_queued(r):
    # ensure_ascii=False keeps a UTF-8 char at its real width; padding with
    # plain ASCII is enough to clear MAX_ANSWER_BYTES either way.
    big_text = "a" * (MAX_ANSWER_BYTES + 1)
    result = await _submit(r, payload={"text": big_text})
    assert result is RemoteAnswerResult.TOO_LARGE
    assert await r.llen(KEY) == 0


async def test_redis_error_during_ack_wait_is_reported_accepted(r, monkeypatch):
    async def boom(*args, **kwargs):
        raise RedisError("connection lost")

    monkeypatch.setattr(r, "xread", boom)
    with _records_from("cowork.turnqueue.answers") as records:
        assert await _submit(r) is RemoteAnswerResult.ACCEPTED
    # The answer was already queued (RPUSH ran before the ack wait began).
    entry = json.loads(await r.lindex(KEY, 0))
    assert entry["values"] == ["pg"]
    text = "\n".join(record.getMessage() for record in records)
    assert "Redis error" in text
    assert "pg" not in text  # answer content never logged
