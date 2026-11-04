"""A web turn's time limit: sent to the pod, its key sized and extended to match,
and the key revoked once the turn is over.

COWORK_TURN_TIME_LIMIT_SECONDS is off by default; a turn then carries no time
limit and its key keeps the old 20-minute life.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from unittest.mock import AsyncMock

import httpx
import pytest

from cowork.common.settings.app_settings import TurnQueueSettings
from cowork.turnqueue import auth_keys
from cowork.turnqueue import producer as prod

from tests.turnqueue.test_producer import FakeRedis, _drain, _reply


@pytest.fixture
def time_limit(monkeypatch):
    monkeypatch.setenv("COWORK_TURN_TIME_LIMIT_SECONDS", "3600")


@pytest.fixture
def auth_configured(monkeypatch):
    monkeypatch.setenv("COWORK_TURN_AUTH_INTERNAL_BASE_URL", "http://auth.internal")
    monkeypatch.setenv("COWORK_TURN_AUTH_INTERNAL_SECRET", "shh")


@pytest.fixture
def remote_turn(monkeypatch):
    """One remote turn that completes; returns the fake Redis and the mints and revokes seen."""
    fake = FakeRedis(replies=[("scratchpad:reply:conv-1", _reply("turn_completed", {}))])
    minted, revoked = [], []

    async def _mint(**kwargs):
        minted.append(kwargs)
        return "mdb_turnkey"

    async def _revoke(*, instance_id, settings):
        revoked.append(instance_id)

    monkeypatch.setattr(prod, "get_redis", lambda: fake)
    monkeypatch.setattr(prod, "_new_correlation_id", lambda: "r")
    monkeypatch.setattr(prod, "_mint_oauth_block", AsyncMock(return_value=None))
    monkeypatch.setattr(prod, "mint_turn_key", _mint)
    monkeypatch.setattr(prod, "revoke_turn_key", _revoke)
    return fake, minted, revoked


def _run(**kwargs):
    return prod.stream_remote_replies(
        conversation_id="conv-1", org_id=None, user_id=None, input_text="hi", model="m", **kwargs,
    )


def _params(fake):
    return json.loads(fake.added[0][1]["payload"])["params"]


def test_the_key_lives_one_time_limit_window_and_the_old_twenty_minutes_without_one(monkeypatch):
    assert TurnQueueSettings().turn_key_lifetime_seconds == 1200
    monkeypatch.setenv("COWORK_TURN_TIME_LIMIT_SECONDS", "3600")
    assert TurnQueueSettings().turn_key_lifetime_seconds == 3600 + 1800


@pytest.mark.asyncio
async def test_a_turn_carries_the_time_limit_and_a_key_that_covers_it(time_limit, remote_turn):
    fake, minted, _ = remote_turn
    await _drain(_run())
    assert _params(fake)["limits"] == {"max_turn_seconds": 3600}
    assert minted[0]["ttl_seconds"] == 5400


@pytest.mark.asyncio
async def test_without_a_time_limit_nothing_changes(remote_turn):
    fake, minted, _ = remote_turn
    await _drain(_run())
    assert "limits" not in _params(fake)
    assert minted[0]["ttl_seconds"] == 1200


@pytest.mark.asyncio
async def test_the_key_is_revoked_once_the_turn_is_over(auth_configured, remote_turn):
    _, _, revoked = remote_turn
    await _drain(_run())
    assert revoked == ["r"]


@pytest.mark.asyncio
async def test_the_key_is_revoked_when_the_caller_stops_reading(auth_configured, remote_turn):
    """Stop, a dropped stream or a turn_completed `break` closes the generator
    early; the key must not outlive it."""
    _, _, revoked = remote_turn
    gen = _run()
    await gen.__anext__()
    await gen.aclose()
    assert revoked == ["r"]


@pytest.mark.asyncio
async def test_a_failed_revoke_does_not_fail_the_turn(auth_configured, remote_turn, monkeypatch):
    async def _down(**_kwargs):
        raise httpx.ConnectError("auth down")

    monkeypatch.setattr(prod, "revoke_turn_key", _down)
    items = await _drain(_run())
    assert items[-1][0] == "turn_completed"


# ── /answer: Keep going extends the key ─────────────────────────────────────


@pytest.mark.parametrize(
    ("question_id", "values", "extends"),
    [
        ("limit:abc", ["continue"], True),
        ("limit:abc", ["stop"], False),
        ("q-1", ["continue"], False),
        ("limit:abc", ["continue", "stop"], False),
    ],
)
def test_only_keep_going_at_a_limit_extends(question_id, values, extends):
    from cowork.api.v1.endpoints.responses import _keeps_going_past_a_limit

    assert _keeps_going_past_a_limit(question_id, values) is extends


@pytest.mark.asyncio
async def test_keep_going_gives_the_key_another_window(time_limit, monkeypatch):
    from cowork.api.v1.endpoints import responses as endpoint

    extend = AsyncMock(return_value=True)
    monkeypatch.setattr(endpoint, "extend_turn_key", extend)
    await endpoint._extend_turn_key_window("corr-1")
    assert extend.call_args.kwargs["instance_id"] == "corr-1"
    assert extend.call_args.kwargs["ttl_seconds"] == 5400


@pytest.mark.asyncio
async def test_without_a_time_limit_keep_going_leaves_the_key_alone(monkeypatch):
    from cowork.api.v1.endpoints import responses as endpoint

    extend = AsyncMock(return_value=True)
    monkeypatch.setattr(endpoint, "extend_turn_key", extend)
    await endpoint._extend_turn_key_window("corr-1")
    extend.assert_not_called()


# ── the extend client ───────────────────────────────────────────────────────


class _Settings(TurnQueueSettings):
    auth_internal_base_url: str = "http://auth.internal"
    auth_internal_secret: str = "shh"


def _client_answering(status_code, captured, body=None):
    class _Resp:
        def __init__(self):
            self.status_code = status_code

        def json(self):
            return body or {}

    class _Client:
        def __init__(self, *a, **k):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def patch(self, url, json, headers):
            captured.update(url=url, json=json, headers=headers)
            return _Resp()

    return _Client


@pytest.mark.asyncio
async def test_extend_patches_the_turns_key_on_the_cluster_only_route(monkeypatch):
    captured = {}
    monkeypatch.setattr(httpx, "AsyncClient", _client_answering(200, captured))
    before = datetime.now(timezone.utc)
    assert await auth_keys.extend_turn_key(instance_id="corr-1", ttl_seconds=5400, settings=_Settings())
    assert captured["url"] == "http://auth.internal/internal/turn-keys/corr-1/"
    assert captured["headers"] == {"X-Internal-Auth": "shh"}
    expiry = datetime.fromisoformat(captured["json"]["expiry_date"])
    assert 5390 <= (expiry - before).total_seconds() <= 5410


@pytest.mark.asyncio
async def test_a_refused_extension_reports_false(monkeypatch):
    monkeypatch.setattr(
        httpx, "AsyncClient", _client_answering(400, {}, {"code": "turn_key_lifetime_exceeded"}),
    )
    assert not await auth_keys.extend_turn_key(instance_id="corr-1", ttl_seconds=5400, settings=_Settings())


@pytest.mark.asyncio
async def test_an_unreachable_auth_reports_false(monkeypatch):
    class _Down:
        def __init__(self, *a, **k):
            pass

        async def __aenter__(self):
            raise httpx.ConnectError("down")

        async def __aexit__(self, *a):
            return False

    monkeypatch.setattr(httpx, "AsyncClient", _Down)
    assert not await auth_keys.extend_turn_key(instance_id="corr-1", ttl_seconds=5400, settings=_Settings())


@pytest.mark.asyncio
async def test_extend_without_auth_configured_does_nothing(monkeypatch):
    called = []
    monkeypatch.setattr(httpx, "AsyncClient", lambda *a, **k: called.append(1))
    assert not await auth_keys.extend_turn_key(
        instance_id="corr-1", ttl_seconds=5400, settings=TurnQueueSettings(),
    )
    assert called == []
