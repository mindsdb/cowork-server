"""Which events push, and to whom (cowork.services.push.hooks).

- A task's owner hears it finished, or didn't, unless someone watched it end.
- A scheduled run's owner hears how it ended; a trial run and a cancelled
  one send nothing, and a run another replica reaped is pushed by the reaper.
- An artifact share pushes the people the sharer asked to notify.
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace
from uuid import uuid4

import pytest

from cowork.common.settings.app_settings import get_app_settings
from cowork.services.push import dispatch, hooks
from cowork.services.push.payloads import PushKind
from cowork.streaming import watchers
from cowork.streaming.buffer import FileStreamBuffer
from cowork.streaming.registry import RunHandle, RunRegistry, TurnLifecycle

ORG = "11111111-1111-1111-1111-111111111111"
OWNER = "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"
OTHER = "bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb"


@pytest.fixture
def pushes(monkeypatch):
    """Org mode with a sender configured; every push lands here instead."""
    monkeypatch.setenv("COWORK_TENANCY_MODE", "org")
    get_app_settings.cache_clear()
    dispatch.set_senders(dispatch.Senders(apns=SimpleNamespace()))  # type: ignore[arg-type]
    sent: list[tuple[str, str, object]] = []

    async def record(org_id, user_id, payload):
        sent.append((org_id, user_id, payload))

    monkeypatch.setattr(hooks, "notify_user", record)
    watchers.reset()
    yield sent
    watchers.reset()
    dispatch.set_senders(None)
    get_app_settings.cache_clear()


async def _drained() -> None:
    await dispatch.drain_background_tasks(timeout=5)


async def _handle(tmp_path, reason, **lifecycle) -> RunHandle:
    conversation_id = str(uuid4())
    buffer = FileStreamBuffer(tmp_path / f"{conversation_id}.jsonl")
    await buffer.append("sse", {"sse": "data: hi\n\n"})
    if reason is not None:
        await buffer.close(reason)
    done = asyncio.get_running_loop().create_future()
    done.set_result(None)
    return RunHandle(
        conversation_id=conversation_id, turn_id=1, buffer=buffer, task=done,  # type: ignore[arg-type]
        org_id=ORG, user_id=OWNER, lifecycle=TurnLifecycle(**lifecycle),
    )


# ── a task the person started ─────────────────────────────────────────


@pytest.mark.parametrize(("reason", "succeeded"), [("completed", True), ("error", False), ("interrupted", False)])
async def test_a_finished_task_pushes_its_owner(tmp_path, pushes, reason, succeeded):
    handle = await _handle(tmp_path, reason)
    hooks.turn_ended(handle)
    await _drained()
    ((org_id, user_id, payload),) = pushes
    assert (org_id, user_id) == (ORG, OWNER)
    assert payload.kind == PushKind.TASK_FINISHED
    assert payload.path == f"/c/{handle.conversation_id}"
    # No such conversation in the database: the title falls back.
    assert payload.title == "Your task"
    assert payload.body == ("Finished." if succeeded else "Didn't finish. Tap to see what happened.")


@pytest.mark.parametrize("lifecycle", [{"timed_out": True}, {"stalled": True}])
async def test_a_reaped_or_stalled_task_did_not_finish(tmp_path, pushes, lifecycle):
    hooks.turn_ended(await _handle(tmp_path, "cancelled", **lifecycle))
    await _drained()
    ((_, _, payload),) = pushes
    assert payload.body.startswith("Didn't finish")


@pytest.mark.parametrize(("reason", "lifecycle"), [
    ("cancelled", {}),
    ("completed", {"cancel_requested": True}),
    ("restart", {}),
    (None, {}),
    ("completed", {"interactive": False}),
    ("completed", {"discarded": True}),
    ("completed", {"shutting_down": True}),
])
async def test_these_endings_push_nothing(tmp_path, pushes, monkeypatch, reason, lifecycle):
    import cowork.services.push.hooks as hooks_mod

    monkeypatch.setattr(hooks_mod, "_TERMINAL_READ_TIMEOUT_SECONDS", 0.2)
    hooks.turn_ended(await _handle(tmp_path, reason, **lifecycle))
    await _drained()
    assert pushes == []


async def test_a_task_someone_watched_end_pushes_nothing(tmp_path, pushes):
    from cowork.handlers.responses import sse_from_buffer

    handle = await _handle(tmp_path, "completed")
    frames = [frame async for frame in sse_from_buffer(
        handle.buffer, 0, watching=(handle.conversation_id, handle.turn_id),
    )]
    assert frames, "the reader streamed the turn to its end"
    hooks.turn_ended(handle)
    await _drained()
    assert pushes == []


async def test_with_push_off_a_stream_records_no_watcher(tmp_path, monkeypatch):
    from cowork.handlers.responses import sse_from_buffer

    monkeypatch.setenv("COWORK_TENANCY_MODE", "org")
    get_app_settings.cache_clear()
    dispatch.set_senders(dispatch.Senders())  # no platform configured: push is off
    watchers.reset()
    try:
        handle = await _handle(tmp_path, "completed")
        frames = [frame async for frame in sse_from_buffer(
            handle.buffer, 0, watching=(handle.conversation_id, handle.turn_id),
        )]
        assert frames
        assert not await watchers.is_watched(handle.conversation_id, handle.turn_id)
    finally:
        watchers.reset()
        dispatch.set_senders(None)
        get_app_settings.cache_clear()


async def test_a_reader_that_left_before_the_end_no_longer_counts(tmp_path, pushes):
    handle = await _handle(tmp_path, "completed")
    async with watchers.watch(handle.conversation_id, handle.turn_id):
        assert await watchers.is_watched(handle.conversation_id, handle.turn_id)
    assert not await watchers.is_watched(handle.conversation_id, handle.turn_id)
    hooks.turn_ended(handle)
    await _drained()
    assert len(pushes) == 1


async def test_a_reader_that_saw_the_end_counts_for_a_grace_period(monkeypatch):
    watchers.reset()
    async with watchers.watch("c", 1) as held:
        held.saw_end = True
    assert await watchers.is_watched("c", 1)
    monkeypatch.setattr(watchers.time, "monotonic", lambda: 10.0**9)
    assert not await watchers.is_watched("c", 1)
    watchers.reset()


async def test_the_registry_hands_every_finished_turn_to_the_hook(tmp_path, monkeypatch):
    ended: list[RunHandle] = []
    monkeypatch.setattr(hooks, "turn_ended", ended.append)
    registry = RunRegistry()

    async def produce(buffer):
        await buffer.close("completed")

    async def open_buffer():
        return FileStreamBuffer(tmp_path / "turn.jsonl")

    handle = await registry.start(
        conversation_id=str(uuid4()), turn_id=0, open_buffer=open_buffer, produce=produce,
        org_id=ORG, user_id=OWNER, lifecycle=TurnLifecycle(interactive=False),
    )
    await handle.task
    await asyncio.sleep(0)
    assert ended == [handle]
    assert ended[0].lifecycle.interactive is False


async def test_a_failing_hook_never_reaches_the_turn(tmp_path, monkeypatch):
    def broken(_handle):
        raise RuntimeError("push down")

    monkeypatch.setattr(hooks, "turn_ended", broken)
    registry = RunRegistry()

    async def produce(buffer):
        await buffer.close("completed")

    async def open_buffer():
        return FileStreamBuffer(tmp_path / "turn.jsonl")

    handle = await registry.start(
        conversation_id=str(uuid4()), turn_id=0, open_buffer=open_buffer, produce=produce,
    )
    await handle.task
    await asyncio.sleep(0)
    assert handle.task.exception() is None


async def test_scheduled_turns_are_started_non_interactive():
    from cowork.handlers.responses import ResponsesHandler

    assert ResponsesHandler(interactive=False).interactive is False


async def test_no_push_outside_org_mode(tmp_path, pushes, monkeypatch):
    monkeypatch.setenv("COWORK_TENANCY_MODE", "local")
    get_app_settings.cache_clear()
    hooks.turn_ended(await _handle(tmp_path, "completed"))
    hooks.scheduled_run_ended(org_id=ORG, user_id=OWNER, schedule_name="s", conversation_id=None, succeeded=True)
    hooks.artifact_shared(
        org_id=ORG, sharer_id=OWNER, sharer_name=None, recipient_ids=[OTHER], artifact_title="a", url=None,
    )
    await _drained()
    assert pushes == []


# ── a share ───────────────────────────────────────────────────────────


async def test_a_share_pushes_each_named_person_but_never_the_sharer(pushes):
    hooks.artifact_shared(
        org_id=ORG, sharer_id=OWNER, sharer_name="owner@example.com",
        recipient_ids=[OTHER, OWNER, OTHER], artifact_title="Q3", url="https://view.example/r/1",
    )
    await _drained()
    ((org_id, user_id, payload),) = pushes
    assert (org_id, user_id) == (ORG, OTHER)
    assert payload.kind == PushKind.ARTIFACT_SHARED
    assert payload.title == "owner@example.com shared “Q3” with you"
    assert payload.url == "https://view.example/r/1" and payload.path is None


async def test_a_share_with_no_one_to_notify_pushes_nothing(pushes):
    hooks.artifact_shared(
        org_id=ORG, sharer_id=OWNER, sharer_name=None, recipient_ids=[], artifact_title="Q3", url=None,
    )
    await _drained()
    assert pushes == []


async def test_a_scheduled_run_pushes_its_owner(pushes):
    hooks.scheduled_run_ended(
        org_id=ORG, user_id=OWNER, schedule_name="Daily report", conversation_id="c1", succeeded=False,
    )
    await _drained()
    ((_, user_id, payload),) = pushes
    assert user_id == OWNER
    assert (payload.kind, payload.title, payload.body, payload.path) == (
        PushKind.SCHEDULED_RUN_FINISHED, "Daily report", "Scheduled run didn't finish.", "/c/c1",
    )


# ── watchers across replicas (Redis stream backend) ───────────────────


@pytest.fixture
def shared_redis(monkeypatch):
    import fakeredis

    client = fakeredis.aioredis.FakeRedis(decode_responses=True)
    monkeypatch.setattr("cowork.turnqueue.redis_client.get_redis", lambda: client)
    monkeypatch.setenv("COWORK_STREAM_BACKEND", "redis")
    watchers.reset()
    yield client
    watchers.reset()


async def test_a_reader_on_another_replica_counts_as_watching(shared_redis):
    async with watchers.watch("c", 2) as held:
        # Another replica has no local entry; only Redis says someone reads.
        watchers.reset()
        assert await watchers.is_watched("c", 2)
        held.saw_end = True
    watchers.reset()
    assert await watchers.is_watched("c", 2), "the reader saw the end: it counts for the grace period"


async def test_a_reader_on_another_replica_that_left_early_stops_counting(shared_redis):
    async with watchers.watch("c", 3):
        pass
    watchers.reset()
    assert not await watchers.is_watched("c", 3)


async def test_a_dead_replicas_reader_lapses(shared_redis, monkeypatch):
    async with watchers.watch("c", 4):
        now = watchers.time.time()
        monkeypatch.setattr(watchers.time, "time", lambda: now + watchers.WATCH_LEASE_SECONDS + 1)
        watchers.reset()
        assert not await watchers.is_watched("c", 4)


async def test_a_broken_redis_reads_as_unwatched(monkeypatch):
    def broken():
        raise ConnectionError("redis down")

    monkeypatch.setattr("cowork.turnqueue.redis_client.get_redis", broken)
    monkeypatch.setenv("COWORK_STREAM_BACKEND", "redis")
    watchers.reset()
    async with watchers.watch("c", 5):
        watchers.reset()
        assert not await watchers.is_watched("c", 5)
