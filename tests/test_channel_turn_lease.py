"""Channel turns of one conversation run one at a time on every replica.

Each replica builds its own AntonChannelRuntime, whose thread locks order
turns inside that process only. A second message that reaches another
replica waits on the conversation's lease in Redis
(runtime.conversation_turn_lease) before it stages the workspace, so it never
rewrites the project files' copy while the turn before it still reads it.

Two runtimes on one fakeredis stand in for two replicas. Everything around
the lease is stubbed down to the staging call and the remote turn.
"""
from __future__ import annotations

import asyncio
import json
import logging
from types import SimpleNamespace
from uuid import uuid4

import fakeredis
import fakeredis.aioredis
import pytest

import cowork.channels.runtime as runtime_mod
from cowork.channels.runtime import BUSY_NOTICE, AntonChannelRuntime, LiveAdapterRegistry, _TurnSetup, lease_key
from cowork.common.settings.user_settings import UserSettings
from cowork.db.units import FilesystemBusy
from cowork.streaming import turn_claim
from cowork.streaming.turn_claim import TurnClaimUnavailable

from _fakes import UnreachableRedis, inline_run_db

ORG = "org-1"
CONVERSATION_ID = uuid4()


class Turns:
    """The staging calls and remote turns both replicas run, in order. The
    first remote turn waits on ``first_may_end``."""

    def __init__(self) -> None:
        self.steps: list[str] = []
        self.first_running = asyncio.Event()
        self.first_may_end = asyncio.Event()
        self.stage_error: Exception | None = None

    async def stage(self, *, conv_id, scope) -> None:
        self.steps.append("stage")
        if self.stage_error is not None:
            raise self.stage_error

    async def remote_turn_events(self, **_kwargs):
        first = "turn starts" not in self.steps
        self.steps.append("turn starts")
        if first:
            self.first_running.set()
            await self.first_may_end.wait()
        self.steps.append("turn ends")
        return
        yield  # pragma: no cover


class StreamingHarness:
    """get_harness() stand-in whose formatter reads the turn's stream to its end."""

    async def formatter(self, stream, harness_id, event_sink):
        async for chunk in stream:
            yield chunk


class RecordingAdapter:
    def __init__(self) -> None:
        self.delivered: list[str] = []

    async def deliver(self, message) -> None:
        self.delivered.append(message.text)


@pytest.fixture
def redis(monkeypatch):
    client = fakeredis.aioredis.FakeRedis(server=fakeredis.FakeServer(), decode_responses=True)
    monkeypatch.setattr(runtime_mod, "get_redis", lambda: client)
    monkeypatch.setenv("COWORK_STREAM_BACKEND", "redis")
    monkeypatch.setattr(runtime_mod, "LEASE_POLL_SECONDS", 0.01)
    return client


@pytest.fixture
def turns(monkeypatch) -> Turns:
    """A remote channel turn in CONVERSATION_ID with no database and no pod."""
    monkeypatch.setenv("COWORK_TURN_BACKEND", "remote")
    turns = Turns()
    binding = SimpleNamespace(
        id=uuid4(), anton_project_id=None, trigger_rule="always", display_name=None, instructions=None,
    )
    conversation = SimpleNamespace(id=CONVERSATION_ID, project=None)
    monkeypatch.setattr(runtime_mod, "run_db", inline_run_db(None))
    monkeypatch.setattr(AntonChannelRuntime, "_find_binding", staticmethod(lambda _session, **_kw: binding))
    monkeypatch.setattr(AntonChannelRuntime, "_open_conversation", lambda self, _session, **_kw: conversation)
    monkeypatch.setattr(
        AntonChannelRuntime, "_read_turn_setup",
        lambda self, _session, **_kw: _TurnSetup(harness_id="anton", settings=UserSettings()),
    )

    async def no_save(**_kwargs) -> None:
        return None

    monkeypatch.setattr(AntonChannelRuntime, "_save_turn_row", staticmethod(no_save))
    monkeypatch.setattr(runtime_mod, "get_harness", lambda _harness_id: StreamingHarness())
    monkeypatch.setattr(runtime_mod, "stage_remote_workspace", turns.stage)
    monkeypatch.setattr(runtime_mod, "remote_turn_events", turns.remote_turn_events)
    return turns


def replica(adapter: RecordingAdapter | None = None) -> AntonChannelRuntime:
    adapters = LiveAdapterRegistry()
    if adapter is not None:
        adapters._cache[("slack", ORG)] = adapter
    return AntonChannelRuntime(adapters)


def message(text: str = "what changed in data.csv?") -> SimpleNamespace:
    return SimpleNamespace(
        address=SimpleNamespace(channel_type="slack", platform_id="C1", thread_id="1700000000.000100"),
        message=SimpleNamespace(
            content=text, is_mention=True, is_group=False, attachments=[], sender_name=None,
        ),
    )


def lease_waits(caplog) -> int:
    return sum(1 for r in caplog.records if getattr(r, "event", None) == "channel_turn_lease_wait")


async def test_a_second_replica_stages_only_after_the_first_replicas_turn_ends(redis, turns, caplog):
    caplog.set_level(logging.INFO, logger=runtime_mod.__name__)
    first = asyncio.create_task(replica().handle("slack", message(), ORG))
    await asyncio.wait_for(turns.first_running.wait(), timeout=5)

    second = asyncio.create_task(replica().handle("slack", message("and now?"), ORG))
    await asyncio.sleep(0.2)
    assert turns.steps == ["stage", "turn starts"]
    assert lease_waits(caplog) == 1

    turns.first_may_end.set()
    await asyncio.wait_for(asyncio.gather(first, second), timeout=5)
    assert turns.steps == ["stage", "turn starts", "turn ends", "stage", "turn starts", "turn ends"]
    assert await redis.get(lease_key(CONVERSATION_ID)) is None


async def test_a_turn_longer_than_the_leases_ttl_keeps_it_by_renewing(redis, turns, monkeypatch):
    monkeypatch.setattr(turn_claim, "CLAIM_TTL_SECONDS", 1)
    monkeypatch.setattr(turn_claim, "RENEW_INTERVAL_SECONDS", 0.2)
    first = asyncio.create_task(replica().handle("slack", message(), ORG))
    await asyncio.wait_for(turns.first_running.wait(), timeout=5)

    second = asyncio.create_task(replica().handle("slack", message("and now?"), ORG))
    await asyncio.sleep(1.6)
    assert turns.steps == ["stage", "turn starts"]

    turns.first_may_end.set()
    await asyncio.wait_for(asyncio.gather(first, second), timeout=5)
    assert turns.steps[2:4] == ["turn ends", "stage"]


async def test_a_refused_turn_releases_the_lease_for_the_next_one(redis, turns):
    adapter = RecordingAdapter()
    turns.stage_error = FilesystemBusy("no shared file store slot freed within POOL_TIMEOUT")
    with pytest.raises(FilesystemBusy):
        await replica(adapter).handle("slack", message(), ORG)
    assert adapter.delivered == [BUSY_NOTICE]
    assert await redis.get(lease_key(CONVERSATION_ID)) is None

    turns.stage_error = None
    turns.first_may_end.set()
    await asyncio.wait_for(replica().handle("slack", message("again"), ORG), timeout=5)
    assert turns.steps == ["stage", "stage", "turn starts", "turn ends"]


async def test_a_refused_turns_warning_names_the_limit_and_the_share(redis, turns, caplog):
    """A channel turn whose copy of the project's files found no copy slot
    logs which limit ran out and whose share it was, as fields the JSON log
    keeps."""
    from cowork.common.structured_logging import make_formatter

    turns.stage_error = FilesystemBusy(
        "no project files copy slot freed within POOL_TIMEOUT", limiter="run_efs_copy", share=f"org:{ORG}:copy",
    )
    with caplog.at_level(logging.WARNING, logger=runtime_mod.log.name), pytest.raises(FilesystemBusy):
        await replica(RecordingAdapter()).handle("slack", message(), ORG)

    [record] = [r for r in caplog.records if getattr(r, "event", None) == "channel_turn_busy"]
    line = json.loads(make_formatter().format(record))
    assert (line["limiter"], line["share"], line["conversation_id"]) == (
        "run_efs_copy", f"org:{ORG}:copy", str(CONVERSATION_ID),
    )


async def test_a_redis_that_cannot_hand_out_the_lease_refuses_the_turn_with_the_busy_notice(turns, monkeypatch):
    monkeypatch.setenv("COWORK_STREAM_BACKEND", "redis")
    monkeypatch.setattr(runtime_mod, "get_redis", lambda: UnreachableRedis())
    adapter = RecordingAdapter()
    turns.first_may_end.set()
    with pytest.raises(TurnClaimUnavailable):
        await asyncio.wait_for(replica(adapter).handle("slack", message(), ORG), timeout=5)
    assert adapter.delivered == [BUSY_NOTICE]
    assert turns.steps == []


async def test_the_file_backend_takes_no_lease(turns, monkeypatch):
    """One process serves the file backend, and its thread locks order its turns."""
    monkeypatch.delenv("COWORK_STREAM_BACKEND", raising=False)
    monkeypatch.setattr(runtime_mod, "get_redis", lambda: UnreachableRedis())
    turns.first_may_end.set()
    await asyncio.wait_for(replica().handle("slack", message(), ORG), timeout=5)
    assert turns.steps == ["stage", "turn starts", "turn ends"]
