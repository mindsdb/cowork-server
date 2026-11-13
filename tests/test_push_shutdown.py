"""The shutdown drain owns the APNs client's lifetime, including idle clients."""
from __future__ import annotations

import asyncio

import httpx
import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ec

from cowork.services.push import dispatch, payloads
from cowork.services.push.apns import ApnsSender, Delivery
from cowork.services.push.devices import Device


@pytest.fixture
async def sender_factory(monkeypatch):
    # Exercise the real HTTP client without sockets or push credentials.
    # Keep this test's tasks and sender isolated from process-global state.
    tasks = set()
    senders = []
    monkeypatch.setattr(dispatch, "_background_tasks", tasks)
    monkeypatch.setattr(dispatch, "_senders", None)

    def create(handler):
        key = ec.generate_private_key(ec.SECP256R1()).private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        ).decode()
        sender = ApnsSender(
            key_pem=key, key_id="TEST_KEY", team_id="TEST_TEAM",
            topic="ai.mindshub.cowork", transport=httpx.MockTransport(handler),
        )
        senders.append(sender)
        dispatch.set_senders(dispatch.Senders(apns=sender))
        return sender

    yield create

    pending = list(tasks)
    for task in pending:
        task.cancel()
    await asyncio.gather(*pending, return_exceptions=True)
    for sender in senders:
        await sender.aclose()


_DEVICE = Device("ios", "test-device", None)
_PAYLOAD = payloads.task_finished(conversation_id="test-conversation", title="Task", succeeded=True)


async def test_shutdown_closes_cached_apns_client_without_pending_tasks(sender_factory):
    sender = sender_factory(lambda request: httpx.Response(200))
    await sender.send(_DEVICE, _PAYLOAD)
    client = sender._client()
    assert not client.is_closed
    assert not dispatch._background_tasks

    await dispatch.drain_background_tasks()

    assert client.is_closed


async def test_shutdown_waits_for_pending_send_before_closing_apns_client(sender_factory):
    started = asyncio.Event()
    release = asyncio.Event()

    async def respond(request):
        started.set()
        await release.wait()
        assert not client.is_closed
        return httpx.Response(200)

    sender = sender_factory(respond)
    client = sender._client()
    result = []

    async def send():
        result.append(await sender.send(_DEVICE, _PAYLOAD))

    sending = dispatch.spawn(send())
    draining = None
    try:
        await asyncio.wait_for(started.wait(), timeout=5)
        draining = asyncio.create_task(dispatch.drain_background_tasks(timeout=5))
        # Let the drain reach its wait while the transport remains blocked.
        await asyncio.sleep(0)
        assert not draining.done()
        assert not client.is_closed
        release.set()
        await asyncio.wait_for(draining, timeout=5)

        assert sending.done() and not sending.cancelled()
        assert result[0].delivery == Delivery.SENT
        assert client.is_closed
    finally:
        release.set()
        if draining is not None and not draining.done():
            draining.cancel()
            await asyncio.gather(draining, return_exceptions=True)


async def test_shutdown_cancels_overdue_send_before_closing_apns_client(sender_factory, caplog):
    started = asyncio.Event()
    cancelled = asyncio.Event()

    async def respond(request):
        started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            assert not client.is_closed
            cancelled.set()
            raise

    sender = sender_factory(respond)
    client = sender._client()

    async def send():
        await sender.send(_DEVICE, _PAYLOAD)

    sending = dispatch.spawn(send())
    await asyncio.wait_for(started.wait(), timeout=5)

    await asyncio.wait_for(dispatch.drain_background_tasks(timeout=0.01), timeout=5)

    assert cancelled.is_set()
    assert sending.cancelled()
    assert client.is_closed
    assert any(getattr(record, "event", None) == "push_shutdown_unfinished" for record in caplog.records)
