"""The doc relay between replicas, on fakeredis pub/sub: one DocHub and one
DocRelay per simulated replica, all on one FakeServer."""
from __future__ import annotations

import asyncio
import json

import fakeredis
import fakeredis.aioredis as fakeaioredis
import pytest

from cowork.docs import relay as relay_mod
from cowork.docs.hub import DocHub, conv_topic, doc_topic
from cowork.docs.relay import CHANNEL_PREFIX, DocRelay


async def _until(predicate, timeout: float = 3.0) -> bool:
    deadline = asyncio.get_running_loop().time() + timeout
    while not predicate():
        if asyncio.get_running_loop().time() >= deadline:
            return False
        await asyncio.sleep(0.01)
    return True


def _drain(queue: asyncio.Queue) -> list[dict]:
    events = []
    while not queue.empty():
        events.append(queue.get_nowait())
    return events


@pytest.fixture
def server():
    return fakeredis.FakeServer()


def _client(server):
    return fakeaioredis.FakeRedis(server=server, decode_responses=True)


async def _replica(server, **kwargs) -> tuple[DocHub, DocRelay]:
    hub = DocHub()
    relay = DocRelay(hub, _client(server), **kwargs)
    relay.start()
    return hub, relay


async def _subscribed(relay: DocRelay, server) -> None:
    """Wait until the relay's PSUBSCRIBE is live, so a test's first message reaches it."""
    probe = _client(server)
    before = relay.received
    for _ in range(100):
        await probe.publish(CHANNEL_PREFIX + "probe", json.dumps({"origin": "probe", "topic": "probe"}))
        # Longer than the subscriber's idle pause, so one probe is read before the next goes out.
        if await _until(lambda: relay.received > before, timeout=0.2):
            return
    raise AssertionError("the relay never subscribed")


async def test_publish_returns_before_the_relay_touches_redis(server, tmp_path, monkeypatch):
    from cowork.turnqueue import redis_client

    monkeypatch.setattr(redis_client, "get_sync_redis", lambda: pytest.fail("publish must not use the blocking client"))
    calls: list[str] = []
    client = _client(server)
    real_publish = client.publish

    async def recording_publish(channel, payload):
        calls.append(channel)
        return await real_publish(channel, payload)

    client.publish = recording_publish
    hub = DocHub()
    relay = DocRelay(hub, client)
    relay.start()
    try:
        topic = doc_topic(tmp_path)
        hub.publish(topic, {"type": "doc.updated", "rev": 2, "clientId": "tab-1", "doc": {"rev": 2}})
        # Nothing awaited yet: publish only queued the message.
        assert calls == []
        assert await _until(lambda: calls == [CHANNEL_PREFIX + topic])
    finally:
        await relay.stop()


async def test_with_redis_down_publish_loses_the_message_quietly_and_later_ones_go_out(server, monkeypatch):
    monkeypatch.setattr(relay_mod, "_BACKOFF_MIN_S", 0.01)
    hub_a, relay_a = await _replica(server)
    hub_b, relay_b = await _replica(server)
    try:
        await _subscribed(relay_b, server)
        topic = conv_topic("c-1")
        queue = hub_b.subscribe(topic)
        server.connected = False
        hub_a.publish(topic, {"type": "doc.updated", "rev": 1}, keep=True)
        assert await _until(lambda: relay_a.dropped == 1)
        assert all(not task.done() for task in relay_a._tasks)
        server.connected = True
        await _subscribed(relay_b, server)
        hub_a.publish(topic, {"type": "doc.updated", "rev": 2}, keep=True)
        assert await _until(lambda: not queue.empty())
        assert [e["rev"] for e in _drain(queue)] == [2]
    finally:
        await relay_a.stop()
        await relay_b.stop()


async def test_a_full_queue_drops_messages_without_blocking_the_writer(server, tmp_path):
    stuck = asyncio.Event()
    client = _client(server)

    async def hanging_publish(channel, payload):
        await stuck.wait()

    client.publish = hanging_publish
    hub = DocHub()
    relay = DocRelay(hub, client, queue_max=2)
    relay.start()
    try:
        for rev in range(1, 6):
            hub.publish(doc_topic(tmp_path), {"type": "doc.updated", "rev": rev})
        assert await _until(lambda: relay.dropped >= 2)
    finally:
        stuck.set()
        await relay.stop()


async def test_a_conversation_event_reaches_the_other_replica_with_its_card_and_replay(server):
    hub_a, relay_a = await _replica(server)
    hub_b, relay_b = await _replica(server)
    try:
        await _subscribed(relay_b, server)
        topic = conv_topic("c-1")
        queue = hub_b.subscribe(topic, replay=True)
        event = {"type": "doc.created", "artifactId": "a-1", "conversationId": "c-1", "title": "Plan",
                 "rev": 1, "by": "agent", "card": {"kind": "Doc", "title": "Plan"}}
        hub_a.publish(topic, event, keep=True)
        assert await _until(lambda: not queue.empty())
        assert _drain(queue) == [event]
        assert _drain(hub_b.subscribe(topic, replay=True)) == [{**event, "replayed": True}]
    finally:
        await relay_a.stop()
        await relay_b.stop()


async def test_a_replicas_own_message_comes_back_from_redis_and_is_not_delivered_twice(
    server, tmp_path, monkeypatch
):
    hub, relay = await _replica(server)
    try:
        await _subscribed(relay, server)
        topics = [doc_topic(tmp_path), conv_topic("c-1")]
        queues = [hub.subscribe(t) for t in topics]
        came_back: list[dict] = []
        deliver = hub.deliver_remote

        def recording(message):
            if message.get("topic") in topics:  # a late probe is not one of them
                came_back.append(message)
            deliver(message)

        monkeypatch.setattr(hub, "deliver_remote", recording)
        hub.publish(topics[0], {"type": "doc.updated", "rev": 2, "doc": {"rev": 2}})
        hub.publish(topics[1], {"type": "doc.updated", "rev": 2}, keep=True)
        # Both messages came back on this replica's own subscription.
        assert await _until(lambda: len(came_back) == 2)
        assert [[e["type"] for e in _drain(q)] for q in queues] == [["doc.updated"], ["doc.updated"]]
    finally:
        await relay.stop()


async def test_the_subscription_comes_back_after_redis_drops_it(server, monkeypatch):
    monkeypatch.setattr(relay_mod, "_BACKOFF_MIN_S", 0.01)
    hub_a, relay_a = await _replica(server)
    hub_b, relay_b = await _replica(server)
    try:
        await _subscribed(relay_b, server)
        server.connected = False
        await asyncio.sleep(0.1)
        server.connected = True
        await _subscribed(relay_b, server)
        topic = conv_topic("c-2")
        queue = hub_b.subscribe(topic)
        hub_a.publish(topic, {"type": "doc.updated", "rev": 3}, keep=True)
        assert await _until(lambda: not queue.empty())
    finally:
        await relay_a.stop()
        await relay_b.stop()


async def test_stop_detaches_the_relay_and_ends_its_tasks(server, tmp_path):
    hub, relay = await _replica(server)
    tasks = list(relay._tasks)
    assert hub._relay == relay.offer
    await relay.stop()
    assert all(task.done() for task in tasks)
    assert hub._relay is None
    hub.publish(doc_topic(tmp_path), {"type": "doc.updated", "rev": 1})  # no relay, no error


async def test_no_relay_without_the_redis_backend(monkeypatch):
    monkeypatch.setattr(relay_mod, "get_backend", lambda: "file")
    monkeypatch.setattr(relay_mod, "get_redis", lambda: pytest.fail("no Redis client without the redis backend"))
    assert relay_mod.start_doc_relay() is None
    await relay_mod.stop_doc_relay(None)


async def test_the_redis_backend_starts_the_relay_on_the_process_hub(monkeypatch, server):
    from cowork.docs import hub as hub_mod

    monkeypatch.setattr(relay_mod, "get_backend", lambda: "redis")
    monkeypatch.setattr(relay_mod, "get_redis", lambda: _client(server))
    relay = relay_mod.start_doc_relay()
    try:
        assert relay is not None
        assert hub_mod.hub._relay == relay.offer
    finally:
        await relay_mod.stop_doc_relay(relay)
    assert hub_mod.hub._relay is None

