"""A delayed cancellation can target only the run instance that requested it."""
from __future__ import annotations

import asyncio
from dataclasses import FrozenInstanceError
from uuid import UUID, uuid4

import fakeredis.aioredis
import pytest
from pydantic import ValidationError

from cowork.api.v1.endpoints import responses
from cowork.db.scoped import TenantScope
from cowork.schemas.responses import ResponsesRequest
from cowork.streaming import backend, buffer, cancellation, turn_index
from cowork.streaming.registry import RunIdentity, RunRegistry, TurnLifecycle

LOCAL = TenantScope(org_mode=False)


@pytest.fixture
def redis(monkeypatch):
    client = fakeredis.aioredis.FakeRedis(decode_responses=True)
    for module in (responses, buffer, cancellation, turn_index):
        monkeypatch.setattr(module, "get_redis", lambda: client)
    monkeypatch.setattr(cancellation, "CANCEL_POLL_SECONDS", 0.01)
    monkeypatch.setenv("COWORK_STREAM_BACKEND", "redis")
    return client


async def start(runs, *, cid="c", turn=0, org=None, user=None):
    ready = asyncio.Event()
    lifecycle = TurnLifecycle()

    async def producer(stream):
        try:
            await stream.append("sse", {"sse": 'event: response.created\ndata: {}\n\n'})
            ready.set()
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            await stream.close("cancelled")
            raise

    handle = await runs.start(
        conversation_id=cid, turn_id=turn, open_buffer=lambda: backend.new_buffer(cid, turn),
        produce=producer, lifecycle=lifecycle, org_id=org, user_id=user,
    )
    await asyncio.wait_for(ready.wait(), 1)
    return handle


async def settled(handle):
    await asyncio.wait_for(asyncio.gather(handle.task, return_exceptions=True), 1)


@pytest.mark.parametrize("value", [None, 0, 1, "true", "false", [], {}])
def test_disconnect_opt_in_requires_a_json_boolean(value):
    with pytest.raises(ValidationError):
        ResponsesRequest(stream=True, cancel_on_disconnect=value)


def test_disconnect_opt_in_defaults_off_and_requires_streaming():
    assert not ResponsesRequest().cancel_on_disconnect
    assert ResponsesRequest(stream=True, cancel_on_disconnect=True).cancel_on_disconnect
    for stream in (False, None):
        with pytest.raises(ValidationError, match="requires stream=true"):
            ResponsesRequest(stream=stream, cancel_on_disconnect=True)


def test_run_identity_is_unique_and_immutable():
    identity = RunIdentity()
    assert str(UUID(identity.run_id)) == identity.run_id
    assert identity.run_id != RunIdentity().run_id
    with pytest.raises(FrozenInstanceError):
        identity.run_id = str(uuid4())


async def test_reused_turn_number_cannot_match_an_old_cancellation(redis, monkeypatch):
    runs = RunRegistry()
    monkeypatch.setattr(responses, "registry", runs)
    old = await start(runs)
    await old.cancel()
    new = await start(runs)  # no saved question: the turn number is legitimately reused
    try:
        reply = await responses.cancel_response(responses.CancelRequest(
            conversation_id="c", expected_run_id=old.run_id, reason="client_gone",
        ), LOCAL)
        assert new.is_answering, "a stale cancellation stopped the successor that reused its turn number"
        assert reply.status_code == 404
        assert old.run_id != new.run_id
        assert not await redis.exists(cancellation.cancel_flag_key(new.lifecycle.identity.correlation_id))
    finally:
        await runs.shutdown()


async def test_exact_cancel_uses_captured_handle_not_a_new_shared_index(redis, monkeypatch):
    old_owner, new_owner = RunRegistry(), RunRegistry()
    old = await start(old_owner, turn=0)
    new = await start(new_owner, turn=1)
    monkeypatch.setattr(responses, "registry", old_owner)
    try:
        reply = await responses.cancel_response(responses.CancelRequest(
            conversation_id="c", expected_run_id=old.run_id, reason="client_gone",
        ), LOCAL)
        assert reply["cancelled"] is True
        await settled(old)
        assert new.is_answering
        assert await redis.exists(cancellation.cancel_flag_key(old.lifecycle.identity.correlation_id))
        assert not await redis.exists(cancellation.cancel_flag_key(new.lifecycle.identity.correlation_id))
    finally:
        await old_owner.shutdown()
        await new_owner.shutdown()


@pytest.mark.parametrize("reason", [None, "stalled"])
@pytest.mark.parametrize("legacy_index", [False, True])
async def test_ordinary_cancel_reaches_a_successor_when_this_replica_retains_the_predecessor(
    redis, monkeypatch, reason, legacy_index,
):
    old_owner, new_owner = RunRegistry(), RunRegistry()
    old = await start(old_owner, turn=0)
    await old.cancel()
    new = await start(new_owner, turn=1)
    monkeypatch.setattr(responses, "registry", old_owner)
    try:
        if legacy_index:
            # Older replicas and channel turns publish a correlation without a run ID.
            await turn_index.record_turn(
                "c", turn_id=new.turn_id, correlation_id=new.lifecycle.identity.correlation_id,
                org_id=None, user_id=None,
            )
        reply = await responses.cancel_response(responses.CancelRequest(
            conversation_id="c", reason=reason,
        ), LOCAL)
        assert reply["cancelled"] is True
        assert await redis.exists(cancellation.cancel_flag_key(new.lifecycle.identity.correlation_id))
        assert not await redis.exists(cancellation.cancel_flag_key(old.lifecycle.identity.correlation_id))
        await settled(new)
        assert new.task.cancelled()
        assert new.lifecycle.stalled is (reason == "stalled")
    finally:
        await old_owner.shutdown()
        await new_owner.shutdown()


@pytest.mark.parametrize("org,user", [("org", "bob"), ("other", "alice")])
async def test_retained_predecessor_does_not_authorize_ordinary_cancel_of_a_foreign_successor(
    redis, monkeypatch, org, user,
):
    old_owner, new_owner = RunRegistry(), RunRegistry()
    old = await start(old_owner, turn=0, org="org", user="alice")
    await old.cancel()
    new = await start(new_owner, turn=1, org=org, user=user)
    monkeypatch.setattr(responses, "registry", old_owner)
    try:
        reply = await responses.cancel_response(
            responses.CancelRequest(conversation_id="c"),
            TenantScope(org_mode=True, org_id="org", user_id="alice"),
        )
        assert reply.status_code == 404
        assert new.is_answering
        assert not await redis.exists(cancellation.cancel_flag_key(new.lifecycle.identity.correlation_id))
        assert not await redis.exists(cancellation.cancel_flag_key(old.lifecycle.identity.correlation_id))
    finally:
        await old_owner.shutdown()
        await new_owner.shutdown()


async def test_ordinary_cancel_keeps_the_finished_current_runs_false_result(redis, monkeypatch):
    owner = RunRegistry()
    handle = await start(owner)
    await handle.cancel()
    monkeypatch.setattr(responses, "registry", owner)
    reply = await responses.cancel_response(responses.CancelRequest(conversation_id="c"), LOCAL)
    assert reply == {"cancelled": False, "conversation_id": "c"}


async def test_another_replica_cancels_the_owner_of_a_direct_or_local_run(redis, monkeypatch):
    owner, receiver = RunRegistry(), RunRegistry()
    handle = await start(owner)
    monkeypatch.setattr(responses, "registry", receiver)
    try:
        reply = await responses.cancel_response(responses.CancelRequest(
            conversation_id="c", expected_run_id=handle.run_id, reason="client_gone",
        ), LOCAL)
        assert reply["cancelled"] is True
        await settled(handle)
        assert handle.task.cancelled()
        assert await redis.get(cancellation.cancel_cause_key(handle.lifecycle.identity.correlation_id)) == "client_gone"
    finally:
        await owner.shutdown()


@pytest.mark.parametrize("other_replica", [False, True])
async def test_completion_wins_over_automatic_cancel(redis, monkeypatch, other_replica):
    owner = RunRegistry()
    handle = await start(owner)
    monkeypatch.setattr(responses, "registry", RunRegistry() if other_replica else owner)
    try:
        await handle.buffer.append("sse", {"sse": 'event: response.completed\ndata: {}\n\n'})
        reply = await responses.cancel_response(responses.CancelRequest(
            conversation_id="c", expected_run_id=handle.run_id, reason="client_gone",
        ), LOCAL)
        assert reply["cancelled"] is False
        assert not await redis.exists(cancellation.cancel_flag_key(handle.lifecycle.identity.correlation_id))
        assert handle.is_running
    finally:
        await owner.shutdown()


@pytest.mark.parametrize("other_replica", [False, True])
async def test_an_exact_run_id_does_not_bypass_ownership(redis, monkeypatch, other_replica):
    owner = RunRegistry()
    handle = await start(owner, org="org", user="alice")
    monkeypatch.setattr(responses, "registry", RunRegistry() if other_replica else owner)
    try:
        for scope in (TenantScope(org_mode=True, org_id="org", user_id="bob"),
                      TenantScope(org_mode=True, org_id="other", user_id="alice")):
            reply = await responses.cancel_response(responses.CancelRequest(
                conversation_id="c", expected_run_id=handle.run_id,
            ), scope)
            assert reply.status_code == 404
        assert handle.is_answering
        assert not await redis.exists(cancellation.cancel_flag_key(handle.lifecycle.identity.correlation_id))
    finally:
        await owner.shutdown()


async def test_shutdown_does_not_become_client_cancellation(redis):
    owner = RunRegistry()
    handle = await start(owner)
    try:
        handle.lifecycle.shutting_down = True
        assert await cancellation.cancel_run(handle, reason="client_gone") is False
        assert not await redis.exists(cancellation.cancel_flag_key(handle.lifecycle.identity.correlation_id))
    finally:
        await owner.shutdown()


async def test_cancelled_admission_removes_its_index_without_starting_work(redis, monkeypatch):
    published, release = asyncio.Event(), asyncio.Event()
    record = turn_index.record_turn
    started = False

    async def held_record(*args, **kwargs):
        await record(*args, **kwargs)
        published.set()
        await release.wait()

    async def producer(stream):
        nonlocal started
        started = True

    monkeypatch.setattr(turn_index, "record_turn", held_record)
    owner = RunRegistry()
    admitting = asyncio.create_task(owner.start(
        conversation_id="c", turn_id=0, open_buffer=lambda: backend.new_buffer("c", 0), produce=producer,
    ))
    await published.wait()
    admitting.cancel()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await admitting
    assert await turn_index.get_turn("c") is None
    assert owner.get("c") is None
    assert started is False


async def test_abandoned_admission_cleanup_does_not_remove_a_successor(redis):
    old, new = str(uuid4()), str(uuid4())
    await turn_index.record_turn("c", turn_id=0, correlation_id="new", run_id=new, org_id=None, user_id=None)
    await turn_index.forget_run("c", old)
    assert (await turn_index.get_turn("c"))["run_id"] == new


async def test_remote_cancellation_signals_its_worker_with_a_file_buffer(redis, monkeypatch, tmp_path):
    from cowork.streaming.buffer import FileStreamBuffer

    monkeypatch.setenv("COWORK_STREAM_BACKEND", "file")
    lifecycle = TurnLifecycle(identity=RunIdentity(remote=True))
    owner = RunRegistry()
    ready = asyncio.Event()

    async def producer(stream):
        ready.set()
        await asyncio.Event().wait()

    async def open_buffer():
        return FileStreamBuffer(tmp_path / "turn.jsonl")

    handle = await owner.start(
        conversation_id="c", turn_id=0, open_buffer=open_buffer, produce=producer, lifecycle=lifecycle,
    )
    await ready.wait()
    try:
        assert await cancellation.cancel_run(handle, reason="client_gone") is True
        await settled(handle)
        assert await redis.exists(cancellation.cancel_flag_key(lifecycle.identity.correlation_id))
    finally:
        await owner.shutdown()
