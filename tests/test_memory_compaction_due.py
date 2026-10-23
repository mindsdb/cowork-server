"""When a cloud memory tier is due for compaction, against fakeredis."""
from __future__ import annotations

import fakeredis.aioredis as fakeaioredis
import pytest

from cowork.services import memory_compaction
from cowork.services.memory_compaction import MemoryTier, is_due, record_attempt, tier_lock

PERSON = MemoryTier(org_id="org-1", kind="person", owner_id="user-1")


@pytest.fixture()
def redis_client(monkeypatch):
    client = fakeaioredis.FakeRedis(decode_responses=True)
    monkeypatch.setattr(memory_compaction, "get_redis", lambda: client)
    return client


async def test_a_tier_is_due_only_above_the_threshold(redis_client):
    assert await is_due(PERSON, 20) is False
    assert await is_due(PERSON, 21) is True


async def test_an_attempt_waits_for_ten_more_entries(redis_client):
    await record_attempt(PERSON, 30)
    assert await is_due(PERSON, 39) is False
    assert await is_due(PERSON, 40) is True


async def test_a_hand_prune_lowers_the_bar(redis_client):
    """After an attempt at 600 and a prune to 50, growth counts from 50."""
    await record_attempt(PERSON, 600)
    assert await is_due(PERSON, 50) is False
    assert await is_due(PERSON, 59) is False
    assert await is_due(PERSON, 60) is True


async def test_the_count_key_expires(redis_client):
    await record_attempt(PERSON, 30)
    ttl = await redis_client.ttl("cowork:memory-compaction:count:org-1:person:user-1")
    assert 0 < ttl <= 90 * 24 * 3600


async def test_the_same_user_in_two_orgs_keeps_two_counts(redis_client):
    other_org = MemoryTier(org_id="org-2", kind="person", owner_id="user-1")
    await record_attempt(PERSON, 30)
    assert await is_due(other_org, 30) is True
    async with tier_lock(PERSON) as first, tier_lock(other_org) as second:
        assert first is True and second is True


async def test_a_person_and_a_project_with_the_same_id_keep_two_counts(redis_client):
    project = MemoryTier(org_id="org-1", kind="project", owner_id="user-1")
    await record_attempt(PERSON, 30)
    assert await is_due(project, 30) is True


async def test_only_one_holder_of_a_tier_lock(redis_client):
    async with tier_lock(PERSON) as first:
        async with tier_lock(PERSON) as second:
            assert first is True
            assert second is False
    async with tier_lock(PERSON) as again:
        assert again is True


async def test_the_lock_expires_if_its_holder_dies(redis_client):
    async with tier_lock(PERSON):
        ttl = await redis_client.ttl("cowork:memory-compaction:lock:org-1:person:user-1")
    assert ttl > memory_compaction.COMPACTION_DEADLINE_SECONDS


async def test_a_late_release_leaves_the_next_holder_alone(redis_client):
    """An attempt that outran its TTL must not free the next attempt's lock."""
    key = "cowork:memory-compaction:lock:org-1:person:user-1"
    async with tier_lock(PERSON):
        await redis_client.set(key, "the-next-attempt")
    assert await redis_client.get(key) == "the-next-attempt"
