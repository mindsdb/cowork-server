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


async def test_an_entry_under_the_limit_but_over_it_once_rewrapped_is_rejected(r):
    """scratchpad-controller doesn't check our raw queued entry — it checks the
    stdin line it re-wraps from it (kind/question_id/answer_id/values/text/
    skipped, compact separators, plus a trailing newline), which is a bit
    bigger. An entry that clears MAX_ANSWER_BYTES raw but not once re-wrapped
    would otherwise be accepted here and then silently dropped there."""
    question_id, answer_id = "ask:1", "a" * 32
    raw_overhead = len(
        json.dumps({"question_id": question_id, "answer_id": answer_id, "text": ""},
                   ensure_ascii=False).encode("utf-8")
    )
    big_text = "a" * (MAX_ANSWER_BYTES - raw_overhead)  # raw entry lands exactly at the limit
    raw_entry = json.dumps(
        {"question_id": question_id, "answer_id": answer_id, "text": big_text}, ensure_ascii=False
    )
    assert len(raw_entry.encode("utf-8")) <= MAX_ANSWER_BYTES  # confirms the raw size alone would pass

    result = await submit_remote_answer(
        conversation_id="conv-1", correlation_id="corr-1", question_id=question_id,
        payload={"text": big_text}, r=r, ack_timeout_s=2.0,
    )
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


async def test_an_unblocked_read_during_the_ack_wait_reads_again(r, monkeypatch):
    """A primary demoted during the wait answers ``-UNBLOCKED``. Read as a Redis
    error, it would report accepted and lose the pod's rejection."""
    from redis.exceptions import ResponseError

    real_xread = r.xread
    reads = 0

    async def xread(*args, **kwargs):
        nonlocal reads
        reads += 1
        if reads == 1:
            raise ResponseError(
                "UNBLOCKED force unblock from blocking operation, instance state changed (master -> replica?)")
        return await real_xread(*args, **kwargs)

    monkeypatch.setattr(r, "xread", xread)
    pod = asyncio.create_task(_pod_says(r, "ask_user_answer_rejected", reason="invalid_option"))
    assert await _submit(r) is RemoteAnswerResult.INVALID_OPTION
    await pod
    assert reads >= 2


async def test_the_answer_is_queued_at_most_once(r, monkeypatch):
    """An RPUSH resent after its reply was lost queues the answer twice."""
    from cowork.turnqueue import answers

    sending_once = False
    sent: list[bool] = []

    @contextmanager
    def recording_send_once():
        nonlocal sending_once
        sending_once = True
        try:
            yield
        finally:
            sending_once = False

    real_pipeline = r.pipeline

    def pipeline(*args, **kwargs):
        pipe = real_pipeline(*args, **kwargs)
        real_execute = pipe.execute

        async def execute(*a, **kw):
            sent.append(sending_once)
            return await real_execute(*a, **kw)

        pipe.execute = execute
        return pipe

    monkeypatch.setattr(answers, "send_once", recording_send_once)
    monkeypatch.setattr(r, "pipeline", pipeline)
    pod = asyncio.create_task(_pod_says(r, "ask_user_answered", status="answered"))
    assert await _submit(r) is RemoteAnswerResult.ACCEPTED
    await pod

    assert sent == [True]
    assert await r.llen(KEY) == 1


async def test_the_ack_wait_blocks_for_no_longer_than_the_shared_block(r, monkeypatch):
    """redis-py cuts a read at the socket timeout, so a longer BLOCK reads an
    idle stream as a dead socket."""
    from cowork.turnqueue.redis_client import REDIS_BLOCK_MS

    real_xread = r.xread
    blocks: list[int] = []

    async def xread(*args, **kwargs):
        blocks.append(kwargs["block"])
        return await real_xread(*args, **kwargs)

    monkeypatch.setattr(r, "xread", xread)
    pod = asyncio.create_task(_pod_says(r, "ask_user_answered", status="answered"))
    assert await _submit(r, timeout=4.0) is RemoteAnswerResult.ACCEPTED
    await pod

    assert blocks and max(blocks) <= REDIS_BLOCK_MS


async def test_the_ack_wait_stops_retrying_a_read_when_the_wait_ends(r, monkeypatch):
    """The client retries a read for its whole call budget, 30 s by default.
    A Redis that drops every read near the end of the wait would hold /answer
    that long past its wait without a deadline that ends with the wait."""
    import time

    from redis.exceptions import ConnectionError as RedisConnectionError

    from cowork.turnqueue.redis_client import _AsyncDeadlineRetry

    retry = _AsyncDeadlineRetry(budget_s=5)
    reads_ended: list[float] = []

    async def dropped():
        raise RedisConnectionError("Connection closed by server.")

    async def fail(_error):
        return None

    async def xread(*args, **kwargs):
        try:
            return await retry.call_with_retry(dropped, fail)
        finally:
            reads_ended.append(time.monotonic())

    monkeypatch.setattr(r, "xread", xread)
    started = time.monotonic()
    assert await _submit(r, timeout=1.0) is RemoteAnswerResult.ACCEPTED

    # One read, retried until the wait ends and no longer.
    assert reads_ended == [pytest.approx(started + 1.0, abs=0.3)]
