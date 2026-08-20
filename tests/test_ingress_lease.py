"""Redis-backed per-org ingress lease: acquire/renew/release semantics,
tested against fakeredis (no EVAL support, so the module must not use Lua)."""
from __future__ import annotations

import asyncio

import fakeredis
import fakeredis.aioredis as fakeaioredis
import pytest

from cowork.channels import ingress_lease


@pytest.fixture()
def redis_client(monkeypatch):
    client = fakeaioredis.FakeRedis(decode_responses=True)
    monkeypatch.setattr(ingress_lease, "get_redis", lambda: client)
    return client


def test_acquire_succeeds_when_free(redis_client):
    assert asyncio.run(ingress_lease.acquire("discord", "org-1", "owner-a")) is True


def test_acquire_fails_when_already_held(redis_client):
    asyncio.run(ingress_lease.acquire("discord", "org-1", "owner-a"))
    assert asyncio.run(ingress_lease.acquire("discord", "org-1", "owner-b")) is False


def test_renew_succeeds_for_the_true_owner(redis_client):
    asyncio.run(ingress_lease.acquire("discord", "org-1", "owner-a"))
    assert asyncio.run(ingress_lease.renew("discord", "org-1", "owner-a")) is True


def test_renew_fails_for_a_non_owner(redis_client):
    asyncio.run(ingress_lease.acquire("discord", "org-1", "owner-a"))
    assert asyncio.run(ingress_lease.renew("discord", "org-1", "owner-b")) is False


def test_renew_fails_once_the_lease_is_gone(redis_client):
    # No one holds it — nothing to renew.
    assert asyncio.run(ingress_lease.renew("discord", "org-1", "owner-a")) is False


def test_release_is_a_noop_for_a_non_owner(redis_client):
    asyncio.run(ingress_lease.acquire("discord", "org-1", "owner-a"))
    asyncio.run(ingress_lease.release("discord", "org-1", "owner-b"))
    # Still held by owner-a — a second acquire by owner-b must still fail.
    assert asyncio.run(ingress_lease.acquire("discord", "org-1", "owner-b")) is False


def test_release_by_the_true_owner_frees_it_for_others(redis_client):
    asyncio.run(ingress_lease.acquire("discord", "org-1", "owner-a"))
    asyncio.run(ingress_lease.release("discord", "org-1", "owner-a"))
    assert asyncio.run(ingress_lease.acquire("discord", "org-1", "owner-b")) is True


def test_lease_can_be_reacquired_by_someone_else_after_it_expires(monkeypatch, redis_client):
    monkeypatch.setattr(ingress_lease, "LEASE_TTL_S", 0.05)
    asyncio.run(ingress_lease.acquire("discord", "org-1", "owner-a"))
    asyncio.run(asyncio.sleep(0.15))
    assert asyncio.run(ingress_lease.acquire("discord", "org-1", "owner-b")) is True


def test_lease_keys_are_isolated_per_org_and_channel_type(redis_client):
    asyncio.run(ingress_lease.acquire("discord", "org-1", "owner-a"))
    # A different org, and a different channel_type for the same org, are
    # both untouched by org-1's discord lease.
    assert asyncio.run(ingress_lease.acquire("discord", "org-2", "owner-b")) is True
    assert asyncio.run(ingress_lease.acquire("slack", "org-1", "owner-b")) is True
