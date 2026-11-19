"""DocHub's side of the relay between replicas: what it hands the relay, and
what it does with another replica's message. No Redis here: the relay is a list."""
from __future__ import annotations

import asyncio

from cowork.docs.hub import NUDGE, DocHub, conv_topic, doc_topic


def _drain(queue: asyncio.Queue) -> list[dict]:
    events = []
    while not queue.empty():
        events.append(queue.get_nowait())
    return events


def test_publish_hands_the_relay_a_doc_message_without_the_doc(tmp_path):
    hub = DocHub()
    sent: list[dict] = []
    hub.attach(sent.append)
    topic = doc_topic(tmp_path)
    hub.publish(topic, {"type": "doc.updated", "artifactId": "a-1", "rev": 4, "by": "user",
                        "clientId": "tab-1", "touched": {"t": ["b"]}, "removed": {}, "doc": {"rev": 4}})
    assert sent == [{"type": "doc.updated", "rev": 4, "clientId": "tab-1", "origin": hub.origin, "topic": topic}]


def test_publish_hands_the_relay_a_conversation_event_whole():
    hub = DocHub()
    sent: list[dict] = []
    hub.attach(sent.append)
    topic = conv_topic("C-1")
    event = {"type": "doc.created", "artifactId": "a-1", "conversationId": "c-1", "title": "Plan",
             "rev": 1, "by": "agent", "card": {"kind": "Doc", "title": "Plan"}}
    hub.publish(topic, event, keep=True)
    assert sent == [{**event, "origin": hub.origin, "topic": topic}]


def test_a_relay_that_fails_does_not_fail_the_publish(tmp_path):
    async def scenario():
        hub = DocHub()

        def broken(_message):
            raise RuntimeError("relay down")

        hub.attach(broken)
        topic = doc_topic(tmp_path)
        queue = hub.subscribe(topic)
        hub.publish(topic, {"type": "doc.updated", "rev": 2})
        await asyncio.sleep(0)
        return _drain(queue)

    assert asyncio.run(scenario()) == [{"type": "doc.updated", "rev": 2}]


def test_detach_leaves_a_newer_relay_in_place():
    hub = DocHub()
    first: list[dict] = []
    second: list[dict] = []
    hub.attach(first.append)
    hub.attach(second.append)
    hub.detach(first.append)
    hub.publish(conv_topic("c"), {"type": "doc.updated", "rev": 1})
    assert first == [] and len(second) == 1
    hub.detach(second.append)
    hub.publish(conv_topic("c"), {"type": "doc.updated", "rev": 2})
    assert len(second) == 1


def test_another_replicas_doc_message_wakes_the_docs_streams(tmp_path):
    async def scenario():
        hub = DocHub()
        topic = doc_topic(tmp_path)
        queue = hub.subscribe(topic)
        hub.deliver_remote({"type": "doc.updated", "rev": 5, "clientId": "tab-9", "origin": "other", "topic": topic})
        await asyncio.sleep(0)
        return _drain(queue)

    assert asyncio.run(scenario()) == [{"type": NUDGE, "rev": 5, "clientId": "tab-9"}]


def test_this_replicas_own_message_is_ignored(tmp_path):
    async def scenario():
        hub = DocHub()
        topics = [doc_topic(tmp_path), conv_topic("c-1")]
        queues = [hub.subscribe(t, replay=True) for t in topics]
        for t in topics:
            hub.deliver_remote({"type": "doc.updated", "rev": 5, "origin": hub.origin, "topic": t})
        await asyncio.sleep(0)
        late = hub.subscribe(topics[1], replay=True)
        return [_drain(q) for q in [*queues, late]]

    assert asyncio.run(scenario()) == [[], [], []]


def test_another_replicas_conversation_event_reaches_streams_and_the_replay():
    async def scenario():
        hub = DocHub()
        topic = conv_topic("c-1")
        queue = hub.subscribe(topic, replay=True)
        hub.deliver_remote({"type": "doc.created", "artifactId": "a-1", "conversationId": "c-1", "title": "Plan",
                            "rev": 1, "by": "agent", "card": {"kind": "Doc"}, "origin": "other", "topic": topic})
        await asyncio.sleep(0)
        live = _drain(queue)
        # A chat that subscribes a moment late still hears of the doc.
        late = _drain(hub.subscribe(topic, replay=True))
        return live, late

    live, late = asyncio.run(scenario())
    event = {"type": "doc.created", "artifactId": "a-1", "conversationId": "c-1", "title": "Plan",
             "rev": 1, "by": "agent", "card": {"kind": "Doc"}}
    assert live == [event]
    assert late == [{**event, "replayed": True}]


def test_a_message_without_a_known_topic_is_ignored():
    async def scenario():
        hub = DocHub()
        queue = hub.subscribe("other:x")
        hub.deliver_remote({"type": "doc.updated", "rev": 1, "origin": "other"})
        hub.deliver_remote({"type": "doc.updated", "rev": 1, "origin": "other", "topic": "other:x"})
        hub.deliver_remote({"type": "doc.updated", "rev": 1, "origin": "other", "topic": 7})
        await asyncio.sleep(0)
        return _drain(queue)

    assert asyncio.run(scenario()) == []


def test_a_message_with_malformed_fields_is_ignored(tmp_path):
    """The type becomes an SSE event name and the rev is compared as a number."""
    async def scenario():
        hub = DocHub()
        topics = [doc_topic(tmp_path), conv_topic("c-1")]
        queues = [hub.subscribe(t) for t in topics]
        bad = [
            {"type": "doc.updated\nevent: x", "rev": 1},
            {"type": "doc.updated", "rev": "1"},
            {"type": "doc.updated", "rev": True},
            {"type": "doc.updated", "rev": 1, "clientId": 7},
            {"type": "doc.updated", "rev": 1, "clientId": "x" * 101},
        ]
        for t in topics:
            for message in bad:
                hub.deliver_remote({**message, "origin": "other", "topic": t})
        await asyncio.sleep(0)
        return [_drain(q) for q in queues]

    assert asyncio.run(scenario()) == [[], []]


def test_another_replicas_doc_published_reaches_the_docs_streams_as_itself(tmp_path):
    from cowork.docs.hub import DOC_PUBLISHED

    async def scenario():
        hub = DocHub()
        topic = doc_topic(tmp_path)
        queue = hub.subscribe(topic)
        hub.deliver_remote({"type": DOC_PUBLISHED, "rev": 0, "clientId": None, "origin": "other", "topic": topic})
        await asyncio.sleep(0)
        return _drain(queue)

    assert asyncio.run(scenario()) == [{"type": "doc.published", "rev": 0, "clientId": None}]


def test_a_doc_published_event_goes_to_the_relay():
    from cowork.docs.hub import DOC_PUBLISHED

    hub = DocHub()
    sent = []
    hub.attach(sent.append)
    hub.publish("doc:x", {"type": DOC_PUBLISHED, "rev": 0})
    assert sent and sent[0]["type"] == "doc.published" and sent[0]["rev"] == 0
