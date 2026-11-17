"""A remote turn's Redis traffic through a Redis blip or failover.

The job that queues the turn is sent at most once and read back when its
reply is lost, a reply the controller wrote twice is applied once, a reply
that skips a number or comes from a second run is reported, and an
``-UNBLOCKED`` read is read again.
"""
import json
import logging
from contextlib import contextmanager

import pytest
from redis.exceptions import ConnectionError as RedisConnectionError
from redis.exceptions import ResponseError

from cowork.streaming import cancellation
from cowork.turnqueue import producer as prod
from cowork.turnqueue.redis_client import REDIS_BLOCK_MS, cancel_flag_key
from tests._logging_assertions import assert_log_fields

from _fakes import QueuedPipeline

STREAM = "scratchpad:reply:conv-1"


@pytest.fixture(autouse=True)
def _stub_turn_setup(monkeypatch):
    async def fake_mint(**_kwargs):
        return "mdb_turnkey"

    monkeypatch.setattr(prod, "mint_turn_key", fake_mint)
    monkeypatch.setattr(prod, "_new_correlation_id", lambda: "r")


def _reply(*, kind, data, n=None):
    payload = {"correlation_id": "r", "kind": kind, "data": data}
    if n is not None:
        payload["n"] = n
    return {"payload": json.dumps(payload)}


class _Redis:
    """One reply per XREAD, after any ``-UNBLOCKED`` reads, recording each write."""

    def __init__(self, *, replies, unblocked_reads=0):
        self.writes: list[tuple[str, bool]] = []
        self.blocks: list[int | None] = []
        self.sending_once = False
        self._replies = list(replies)
        self._unblocked_reads = unblocked_reads

    async def delete(self, *keys):
        self.writes.append(("delete", self.sending_once))
        return len(keys)

    async def xrevrange(self, key, count=None):
        return []

    async def exists(self, *keys):
        return 0

    async def sadd(self, key, member):
        self.writes.append(("sadd", self.sending_once))
        return 1

    async def hset(self, key, mapping=None):
        self.writes.append(("hset", self.sending_once))
        return 1

    async def expire(self, key, seconds):
        self.writes.append(("expire", self.sending_once))
        return 1

    async def xadd(self, stream, fields):
        self.writes.append(("xadd", self.sending_once))
        return "1-0"

    def pipeline(self, transaction=True):
        return QueuedPipeline(self)

    async def xread(self, streams, count=None, block=None):
        self.blocks.append(block)
        if self._unblocked_reads:
            self._unblocked_reads -= 1
            raise ResponseError(
                "UNBLOCKED force unblock from blocking operation, instance state changed (master -> replica?)")
        if self._replies:
            return [[STREAM, [["9-0", self._replies.pop(0)]]]]
        return None


def _authorized(*, n=None):
    return _reply(kind="progress", data={"phase": "workspace_authorized", "workspace_mode": "persistent"}, n=n)


async def _turn(*, monkeypatch, fake):
    monkeypatch.setattr(prod, "get_redis", lambda: fake)
    return [
        item async for item in prod.stream_remote_replies(
            conversation_id="conv-1", org_id=None, user_id=None, input_text="hi", model="m",
        )
        if item[0] != "progress"
    ]


async def test_the_job_that_queues_a_turn_is_sent_at_most_once(monkeypatch):
    """A job XADD resent after its reply was lost queues the turn twice."""
    fake = _Redis(replies=[_authorized(), _reply(kind="turn_completed", data={})])

    @contextmanager
    def recording_send_once():
        fake.sending_once = True
        try:
            yield
        finally:
            fake.sending_once = False

    monkeypatch.setattr(prod, "send_once", recording_send_once)
    await _turn(monkeypatch=monkeypatch, fake=fake)

    assert [once for command, once in fake.writes if command == "xadd"] == [True]
    # The rest are safe to resend, and keep their retries.
    assert not any(once for command, once in fake.writes if command != "xadd")


class _RetiringRedis(_Redis):
    """Retires the conversation's queue right after its first registration,
    as a controller's retirement script can between the SADD and the XADD."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.registered: set[str] = set()
        self._retired = False

    async def sadd(self, key, member):
        await super().sadd(key, member)
        if key.endswith(":queues"):
            self.registered.add(member)
            if not self._retired:
                self._retired = True
                self.registered.discard(member)
        return 1


async def test_a_queue_retired_between_registering_and_queueing_is_registered_again(monkeypatch):
    """A job XADD into a stream the registry no longer names is read by no controller."""
    fake = _RetiringRedis(replies=[_authorized(), _reply(kind="turn_completed", data={})])

    await _turn(monkeypatch=monkeypatch, fake=fake)

    assert fake.registered == {"conv-1"}


async def test_a_reply_that_repeats_the_last_number_is_dropped(monkeypatch):
    """The controller resends a reply whose XADD reply it lost, and a doubled
    turn_delta would double its text in the answer."""
    fake = _Redis(replies=[
        _authorized(n=1),
        _reply(kind="turn_delta", data={"text": "he"}, n=2),
        _reply(kind="turn_delta", data={"text": "he"}, n=2),
        _reply(kind="turn_delta", data={"text": "llo"}, n=3),
        _reply(kind="turn_completed", data={}, n=4),
    ])

    items = await _turn(monkeypatch=monkeypatch, fake=fake)

    assert items == [("turn_delta", {"text": "he"}), ("turn_delta", {"text": "llo"}), ("turn_completed", {})]


def _producer_warnings(caplog, event: str) -> list[logging.LogRecord]:
    return [
        record for record in caplog.records
        if record.name == prod.logger.name and record.levelno == logging.WARNING
        and getattr(record, "event", None) == event
    ]


async def test_a_reply_that_skips_a_number_is_applied_and_logged(monkeypatch, caplog, owned_logger):
    """The controller numbers each reply one above the last before its XADD,
    so a skipped number is a reply that never reached the stream: a failover
    or an eviction lost it, and the answer lacks it without a word."""
    logged = owned_logger(prod.logger.name, level=logging.WARNING)
    fake = _Redis(replies=[
        _authorized(n=1),
        _reply(kind="turn_delta", data={"text": "a"}, n=2),
        _reply(kind="turn_delta", data={"text": "c"}, n=4),
        _reply(kind="turn_completed", data={}, n=5),
    ])

    items = await _turn(monkeypatch=monkeypatch, fake=fake)

    assert items == [("turn_delta", {"text": "a"}), ("turn_delta", {"text": "c"}), ("turn_completed", {})]
    assert [(r.last_reply_n, r.reply_n) for r in _producer_warnings(caplog, "remote_reply_gap")] == [(2, 4)]
    assert_log_fields(logged.output(), event="remote_reply_gap", last_reply_n=2, reply_n=4)


async def test_replies_from_a_second_run_of_the_job_fail_the_turn(monkeypatch, caplog, owned_logger):
    """Two runs of one job number their replies from 1 under one correlation
    id. Dropping every number at or below the last one spliced their deltas,
    and could drop the first run's failure, so a failed turn read as
    completed with mixed text."""
    logged = owned_logger(prod.logger.name, level=logging.WARNING)
    fake = _Redis(replies=[
        _authorized(n=1),
        _reply(kind="turn_delta", data={"text": "first "}, n=2),
        _reply(kind="turn_delta", data={"text": "run"}, n=3),
        _authorized(n=1),
        _reply(kind="turn_delta", data={"text": "second"}, n=2),
        _reply(kind="turn_delta", data={"text": " again"}, n=4),
        _reply(kind="turn_completed", data={}, n=5),
    ])

    items = await _turn(monkeypatch=monkeypatch, fake=fake)

    assert items[:2] == [("turn_delta", {"text": "first "}), ("turn_delta", {"text": "run"})]
    kind, data = items[2]
    assert (kind, data["error"]) == ("turn_failed", prod.SECOND_RUN_ERROR)
    assert data["code"] and data["message"]
    assert len(items) == 3
    assert [(r.last_reply_n, r.reply_n) for r in _producer_warnings(caplog, "remote_reply_second_run")] == [(3, 1)]
    assert_log_fields(logged.output(), event="remote_reply_second_run", last_reply_n=3, reply_n=1)


async def test_replies_with_no_number_are_all_applied(monkeypatch):
    """A controller that numbers nothing: two equal deltas are two deltas."""
    fake = _Redis(replies=[
        _authorized(),
        _reply(kind="turn_delta", data={"text": "ha"}),
        _reply(kind="turn_delta", data={"text": "ha"}),
        _reply(kind="turn_completed", data={}),
    ])

    items = await _turn(monkeypatch=monkeypatch, fake=fake)

    assert items == [("turn_delta", {"text": "ha"}), ("turn_delta", {"text": "ha"}), ("turn_completed", {})]


async def test_the_relay_reads_again_after_an_unblocked_reply(monkeypatch):
    """Raised, ``-UNBLOCKED`` would fail a turn the worker is still running."""
    fake = _Redis(
        replies=[_authorized(), _reply(kind="turn_delta", data={"text": "hi"}), _reply(kind="turn_completed", data={})],
        unblocked_reads=2,
    )

    items = await _turn(monkeypatch=monkeypatch, fake=fake)

    assert items == [("turn_delta", {"text": "hi"}), ("turn_completed", {})]


async def test_the_relay_blocks_for_the_shared_block(monkeypatch):
    fake = _Redis(replies=[_authorized(), _reply(kind="turn_completed", data={})])

    await _turn(monkeypatch=monkeypatch, fake=fake)

    assert fake.blocks and set(fake.blocks) == {REDIS_BLOCK_MS}


JOB_STREAM = "scratchpad:requests:conv-1"


class _LostJobReplyRedis(_Redis):
    """Loses the reply to the job XADD, after Redis applied it or before.
    ``readback_fails`` makes the job stream unreadable too."""

    def __init__(self, *, applied: bool, readback_fails: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.applied = applied
        self.readback_fails = readback_fails
        self.jobs: list[tuple[str, dict]] = []
        self.flags: list[str] = []
        self.registrations = 0

    async def sadd(self, key, member):
        if key.endswith(":queues"):
            self.registrations += 1
        return await super().sadd(key, member)

    async def xadd(self, stream, fields):
        if stream != JOB_STREAM:
            return await super().xadd(stream, fields)
        self.writes.append(("xadd", self.sending_once))
        if self.applied:
            self.jobs.append((f"{len(self.jobs) + 1}-0", fields))
        raise RedisConnectionError("Connection closed by server.")

    async def xrevrange(self, key, count=None):
        if key != JOB_STREAM:
            return []
        if self.readback_fails:
            raise RedisConnectionError("Connection closed by server.")
        return [("0-1", {"payload": json.dumps({"correlation_id": "another-turn"})}), *reversed(self.jobs)][:count]

    async def set(self, key, value, ex=None):
        self.flags.append(key)
        return True


async def test_a_job_whose_xadd_reply_was_lost_after_it_was_queued_runs_its_turn(monkeypatch):
    """Redis applied the job XADD and the reply was lost, or came later than
    the socket timeout. The turn used to end as failed while the controller
    ran its job with nobody reading the replies."""
    fake = _LostJobReplyRedis(applied=True, replies=[_authorized(), _reply(kind="turn_completed", data={})])
    monkeypatch.setattr(cancellation, "get_redis", lambda: fake)

    items = await _turn(monkeypatch=monkeypatch, fake=fake)

    assert items == [("turn_completed", {})]
    # Registered again after the job, as a confirmed XADD is.
    assert fake.registrations == 2
    assert fake.flags == []


@pytest.mark.parametrize(
    "applied, readback_fails", [(False, False), (True, True)], ids=["not-queued", "read-back-fails"],
)
async def test_a_job_whose_xadd_reply_was_lost_and_is_not_found_queued_is_canceled(
    monkeypatch, applied, readback_fails,
):
    """The job is not on the stream, or the stream cannot be read either. The
    turn fails, and the job's cancel flag stops a copy that did land before
    the controller runs it with nobody reading."""
    fake = _LostJobReplyRedis(applied=applied, readback_fails=readback_fails, replies=[])
    monkeypatch.setattr(cancellation, "get_redis", lambda: fake)

    with pytest.raises(RedisConnectionError):
        await _turn(monkeypatch=monkeypatch, fake=fake)

    assert fake.flags == [cancel_flag_key("r")]
    assert fake.registrations == 1
