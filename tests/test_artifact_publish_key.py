"""One publish key per reconciliation: minted lazily, revoked afterwards.

The key decides who owns the published artifact on view.mindshub.ai, so it must be
scoped to the acting user. It is NOT keyed on the turn's correlation id: that id
does not exist at this call site, and auth returns 200 with `key: None` when an
active turn key already exists for the same (user, org, instance_id).
"""
from __future__ import annotations

import pytest

from cowork.services.artifact_publish_key import MAX_PUBLISH_KEY_TTL_S, PublishKey


@pytest.fixture
def mint_calls(monkeypatch):
    calls = []

    async def fake_mint(*, user_id, org_id, correlation_id, ttl_seconds, settings):
        calls.append({"user_id": user_id, "org_id": org_id,
                      "instance_id": correlation_id, "ttl_seconds": ttl_seconds})
        return "turnkey-1"

    monkeypatch.setattr("cowork.services.artifact_publish_key.mint_turn_key", fake_mint)
    return calls


@pytest.fixture
def revoke_calls(monkeypatch):
    calls = []

    async def fake_revoke(*, instance_id, settings):
        calls.append(instance_id)

    monkeypatch.setattr("cowork.services.artifact_publish_key.revoke_turn_key", fake_revoke)
    return calls


async def test_no_mint_until_first_get(mint_calls):
    PublishKey("u-1", "o-1", min_ttl_s=60)
    assert mint_calls == []


async def test_key_is_minted_once_and_reused(mint_calls):
    key = PublishKey("u-1", "o-1", min_ttl_s=60)

    first = await key.get()
    second = await key.get()

    assert first == second == "turnkey-1"
    assert len(mint_calls) == 1


async def test_mint_carries_user_and_org_from_scope(mint_calls):
    key = PublishKey("u-1", "o-1", min_ttl_s=60)
    await key.get()

    assert mint_calls[0]["user_id"] == "u-1"
    assert mint_calls[0]["org_id"] == "o-1"


async def test_instance_id_is_a_fresh_uuid_not_a_turn_id(mint_calls):
    key = PublishKey("u-1", "o-1", min_ttl_s=60)
    await key.get()

    assert mint_calls[0]["instance_id"] == key.instance_id
    assert len(key.instance_id) == 36  # uuid4 string form


async def test_ttl_is_at_least_the_requested_floor(mint_calls):
    key = PublishKey("u-1", "o-1", min_ttl_s=200)
    await key.get()

    # The key must outlive an abandoned upload thread, otherwise it reaches
    # /upload already expired.
    assert mint_calls[0]["ttl_seconds"] >= 200


async def test_ttl_is_clamped_below_the_auth_cap(mint_calls):
    # auth rejects an expiry beyond turn_key_max_ttl_seconds with 400, which
    # would surface as "no publishing at all" with no obvious cause.
    key = PublishKey("u-1", "o-1", min_ttl_s=999_999)
    await key.get()

    assert mint_calls[0]["ttl_seconds"] == MAX_PUBLISH_KEY_TTL_S


async def test_none_from_auth_is_reported_as_no_key(monkeypatch):
    async def fake_mint(**kwargs):
        return None  # idempotent hit: auth answers 200 without plaintext

    monkeypatch.setattr("cowork.services.artifact_publish_key.mint_turn_key", fake_mint)
    key = PublishKey("u-1", "o-1", min_ttl_s=60)

    assert await key.get() is None


async def test_mint_failure_is_swallowed_and_reported_as_no_key(monkeypatch):
    async def fake_mint(**kwargs):
        raise RuntimeError("401 Unauthorized")

    monkeypatch.setattr("cowork.services.artifact_publish_key.mint_turn_key", fake_mint)
    key = PublishKey("u-1", "o-1", min_ttl_s=60)

    assert await key.get() is None


async def test_auth_400_on_ttl_is_reported_as_no_key(monkeypatch):
    async def reject(**kwargs):
        raise RuntimeError("400 expiry_date exceeds the max turn-key TTL")

    monkeypatch.setattr("cowork.services.artifact_publish_key.mint_turn_key", reject)
    key = PublishKey("u-1", "o-1", min_ttl_s=60)

    assert await key.get() is None


async def test_failed_mint_is_not_retried_within_one_reconciliation(monkeypatch):
    attempts = []

    async def fake_mint(**kwargs):
        attempts.append(1)
        raise RuntimeError("503 Service Unavailable")

    monkeypatch.setattr("cowork.services.artifact_publish_key.mint_turn_key", fake_mint)
    key = PublishKey("u-1", "o-1", min_ttl_s=60)

    await key.get()
    await key.get()

    assert len(attempts) == 1


async def test_revoke_targets_the_minted_instance_id(mint_calls, revoke_calls):
    key = PublishKey("u-1", "o-1", min_ttl_s=60)
    await key.get()

    await key.revoke()

    assert revoke_calls == [key.instance_id]


async def test_revoke_is_a_noop_when_nothing_was_minted(revoke_calls):
    key = PublishKey("u-1", "o-1", min_ttl_s=60)
    await key.revoke()
    assert revoke_calls == []


async def test_revoke_failure_never_raises(mint_calls, monkeypatch):
    async def boom(**kwargs):
        raise RuntimeError("network down")

    monkeypatch.setattr("cowork.services.artifact_publish_key.revoke_turn_key", boom)
    key = PublishKey("u-1", "o-1", min_ttl_s=60)
    await key.get()

    await key.revoke()  # must not raise — the turn already succeeded
