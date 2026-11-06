"""A remote turn's Redis traffic through a Redis blip or failover.

The job that queues the turn is sent at most once, a reply the controller wrote
twice is applied once, and an ``-UNBLOCKED`` read is read again.
"""
import json
from contextlib import contextmanager

import pytest
from redis.exceptions import ResponseError

from cowork.turnqueue import producer as prod
from cowork.turnqueue.redis_client import REDIS_BLOCK_MS

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


async def test_a_reply_whose_number_is_not_above_the_last_applied_is_dropped(monkeypatch):
    """The controller resends a reply whose XADD reply it lost, and a doubled
    turn_delta would double its text in the answer."""
    fake = _Redis(replies=[
        _authorized(n=1),
        _reply(kind="turn_delta", data={"text": "he"}, n=2),
        _reply(kind="turn_delta", data={"text": "he"}, n=2),
        _reply(kind="turn_delta", data={"text": "stale"}, n=1),
        _reply(kind="turn_delta", data={"text": "llo"}, n=3),
        _reply(kind="turn_completed", data={}, n=4),
    ])

    items = await _turn(monkeypatch=monkeypatch, fake=fake)

    assert items == [("turn_delta", {"text": "he"}), ("turn_delta", {"text": "llo"}), ("turn_completed", {})]


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
