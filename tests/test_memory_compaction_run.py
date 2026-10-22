"""A cloud memory tier compacted on the server: the key, the model call, the
write, and what each failure leaves behind."""
from __future__ import annotations

import asyncio
import logging
from types import SimpleNamespace

import fakeredis.aioredis as fakeaioredis
import pytest
from anton.core.memory.hippocampus import Hippocampus

from cowork.harnesses.memory.registry import MemorySlot
from cowork.services import memory_compaction
from cowork.services.memory_compaction import MemoryTier, record_attempt
from cowork.services.product_permissions import ProductPermissionUnavailable

SCOPE = SimpleNamespace(org_id="org-1", user_id="user-1")
PERSON = MemoryTier(org_id="org-1", kind="person", owner_id="user-1")
SLOTS = [MemorySlot.LESSONS, MemorySlot.RULES]
COUNT_KEY = "cowork:memory-compaction:count:org-1:person:user-1"
LOCK_KEY = "cowork:memory-compaction:lock:org-1:person:user-1"


class _Model:
    """Stands in for the coding model: keeps entry 1, or does what `answer` says."""

    def __init__(self, answer=None) -> None:
        self.calls = 0
        self._answer = answer

    async def generate_object_code(self, schema, *, system, messages, max_tokens):
        self.calls += 1
        if self._answer is not None:
            await self._answer()
        return schema(keep=[1])


@pytest.fixture()
def redis_client(monkeypatch):
    client = fakeaioredis.FakeRedis(decode_responses=True)
    monkeypatch.setattr(memory_compaction, "get_redis", lambda: client)
    return client


@pytest.fixture()
def world(monkeypatch, redis_client, tmp_path):
    """Memory on, a mint that records its calls, and a model the test can swap."""
    state = SimpleNamespace(
        settings=SimpleNamespace(memory_enabled=True, memory_mode="autopilot", hub_workspace_id="ws-1"),
        mints=[],
        model=_Model(),
        root=tmp_path / "memory",
        redis=redis_client,
    )

    async def mint(**kwargs):
        state.mints.append(kwargs)
        return {"provider": "minds-cloud", "api_key": "turn-key", "base_url": "http://minds", "coding_model": "m"}

    monkeypatch.setattr(memory_compaction, "get_user_settings", lambda scope: state.settings)
    monkeypatch.setattr(memory_compaction, "_mint_llm_block", mint)
    monkeypatch.setattr(memory_compaction, "LLMClient", lambda **kwargs: state.model)
    return state


def _lessons(root, count: int) -> Hippocampus:
    hc = Hippocampus(root)
    for i in range(count):
        hc.encode_lesson(f"Fact number {i}", topic=f"t{i}")
    return hc


async def _compact(world) -> None:
    await memory_compaction._compact_tier(PERSON, scope=SCOPE, target=(world.root, SLOTS))


async def test_a_due_tier_is_compacted_and_its_count_stored(world):
    hc = _lessons(world.root, 25)

    await _compact(world)

    assert len(hc.get_lessons()) == 1
    assert world.mints[0]["user_id"] == "user-1"
    assert world.mints[0]["org_id"] == "org-1"
    assert world.mints[0]["workspace_id"] == "ws-1"
    assert await world.redis.get(COUNT_KEY) == "1"
    assert await world.redis.get(LOCK_KEY) is None


async def test_a_tier_that_is_not_due_mints_nothing(world):
    hc = _lessons(world.root, 20)

    await _compact(world)

    assert world.mints == []
    assert len(hc.get_lessons()) == 20


async def test_memory_off_mints_nothing(world):
    _lessons(world.root, 25)
    world.settings.memory_mode = "off"

    await _compact(world)

    assert world.mints == []
    assert await world.redis.get(COUNT_KEY) is None


async def test_a_failed_model_call_keeps_the_file_and_counts_the_attempt(world, caplog):
    hc = _lessons(world.root, 25)
    before = hc._lessons_path.read_bytes()

    async def fail():
        raise RuntimeError("Fact number 3 echoed back by the provider")

    world.model = _Model(answer=fail)
    with caplog.at_level(logging.INFO):
        await _compact(world)

    assert hc._lessons_path.read_bytes() == before
    assert await world.redis.get(COUNT_KEY) == "25"
    assert not any("Fact number" in record.getMessage() for record in caplog.records)


async def test_a_failed_mint_keeps_the_file_and_counts_the_attempt(world, monkeypatch, caplog):
    hc = _lessons(world.root, 25)
    before = hc._lessons_path.read_bytes()

    async def refuse(**kwargs):
        raise ProductPermissionUnavailable()

    monkeypatch.setattr(memory_compaction, "_mint_llm_block", refuse)
    with caplog.at_level(logging.WARNING, logger="cowork.services.memory_compaction"):
        await _compact(world)

    assert hc._lessons_path.read_bytes() == before
    assert await world.redis.get(COUNT_KEY) == "25"
    assert "ProductPermissionUnavailable" in caplog.text


async def test_the_deadline_stops_a_slow_model(world, monkeypatch):
    hc = _lessons(world.root, 25)
    before = hc._lessons_path.read_bytes()
    monkeypatch.setattr(memory_compaction, "COMPACTION_DEADLINE_SECONDS", 0.05)

    async def hang():
        await asyncio.sleep(5)

    world.model = _Model(answer=hang)
    await _compact(world)

    assert hc._lessons_path.read_bytes() == before
    assert await world.redis.get(COUNT_KEY) == "25"


async def test_a_cancel_mid_call_keeps_the_file_and_the_attempt(world):
    """A shutdown during the model call must not use up the attempt."""
    hc = _lessons(world.root, 25)
    before = hc._lessons_path.read_bytes()
    reached = asyncio.Event()

    async def wait_forever():
        reached.set()
        await asyncio.Event().wait()

    world.model = _Model(answer=wait_forever)
    task = asyncio.create_task(_compact(world))
    await reached.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert hc._lessons_path.read_bytes() == before
    assert await world.redis.get(COUNT_KEY) is None
    assert await world.redis.get(LOCK_KEY) is None


async def test_two_triggers_at_once_make_one_model_call(world):
    _lessons(world.root, 25)
    reached = asyncio.Event()
    release = asyncio.Event()

    async def held():
        reached.set()
        await release.wait()

    world.model = _Model(answer=held)
    first = asyncio.create_task(_compact(world))
    await reached.wait()
    await _compact(world)
    release.set()
    await first

    assert len(world.mints) == 1
    assert world.model.calls == 1


async def test_a_folder_removed_during_the_model_call_stays_removed(world):
    """A reset during the call wins; writing now would bring the folder back."""
    _lessons(world.root, 25)

    async def reset():
        for path in world.root.iterdir():
            path.unlink()
        world.root.rmdir()

    world.model = _Model(answer=reset)
    await _compact(world)

    assert not world.root.exists()


async def test_the_profile_is_left_alone(world):
    hc = _lessons(world.root, 25)
    hc.rewrite_identity(["Name: Ada"])
    before = hc._profile_path.read_bytes()

    await _compact(world)

    assert hc._profile_path.read_bytes() == before


async def test_start_compaction_returns_before_the_model_answers(world, monkeypatch):
    _lessons(world.root, 25)
    reached = asyncio.Event()
    release = asyncio.Event()

    async def held():
        reached.set()
        await release.wait()

    world.model = _Model(answer=held)
    monkeypatch.setattr(memory_compaction, "GlobalMemoryStore", lambda scope: SimpleNamespace(root=world.root))
    memory_compaction.start_compaction(
        scope=SCOPE, principal=None, project_id=None, personal_applied=1, project_applied=0
    )
    await reached.wait()
    assert len(memory_compaction._compaction_tasks) == 1

    await memory_compaction.cancel_compactions()

    assert memory_compaction._compaction_tasks == set()


async def test_a_stored_count_from_an_earlier_attempt_is_respected(world):
    _lessons(world.root, 25)
    await record_attempt(PERSON, 20)

    await _compact(world)

    assert world.mints == []
