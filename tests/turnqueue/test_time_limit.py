"""A web turn's time limit: sent to the pod, and its key sized and extended to match.

COWORK_TURN_TIME_LIMIT_SECONDS is off by default; a turn then carries no time
limit and its key keeps the old 20-minute life.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
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
    assert TurnQueueSettings().turn_key_lifetime_seconds == 3600 + 2400


@pytest.mark.asyncio
async def test_a_turn_carries_the_time_limit_and_a_key_that_covers_it(time_limit, remote_turn):
    fake, minted, _ = remote_turn
    await _drain(_run())
    # One block: the time limit sits beside the tenant's per-task limits.
    limits = _params(fake)["limits"]
    assert limits["max_turn_seconds"] == 3600
    assert set(limits) == {"max_tool_rounds", "max_continuations", "max_turn_tokens", "max_turn_seconds"}
    assert minted[0]["ttl_seconds"] == 6000


@pytest.mark.asyncio
async def test_without_a_time_limit_nothing_changes(remote_turn):
    fake, minted, _ = remote_turn
    await _drain(_run())
    assert "max_turn_seconds" not in _params(fake)["limits"]
    assert minted[0]["ttl_seconds"] == 1200


# ── /answer: Keep going extends the key ─────────────────────────────────────


@pytest.mark.parametrize(
    ("question_id", "values", "extends"),
    [
        ("limit:time:abc", ["continue"], True),
        ("limit:time:abc", ["stop"], False),
        ("limit:abc", ["continue"], False),
        ("q-1", ["continue"], False),
        ("limit:time:abc", ["continue", "stop"], False),
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
    assert extend.call_args.kwargs["ttl_seconds"] == 6000


@pytest.mark.asyncio
async def test_a_refused_extension_is_tried_again_after_a_pause(time_limit, monkeypatch):
    """One blip in auth must not turn the user's Keep going into Stop here, so
    the second try waits a moment rather than following straight on."""
    import time

    from cowork.api.v1.endpoints import responses as endpoint

    tries = []

    async def extend(**_kwargs):
        tries.append(time.monotonic())
        return len(tries) == 2

    monkeypatch.setattr(endpoint, "extend_turn_key", extend)
    monkeypatch.setattr(endpoint, "_KEY_EXTENSION_RETRY_PAUSE_SECONDS", 0.05)
    assert await endpoint._extend_turn_key_window("corr-1") is True
    assert len(tries) == 2
    assert tries[1] - tries[0] >= 0.05


@pytest.mark.asyncio
async def test_without_a_time_limit_keep_going_leaves_the_key_alone(monkeypatch):
    from cowork.api.v1.endpoints import responses as endpoint

    extend = AsyncMock(return_value=True)
    monkeypatch.setattr(endpoint, "extend_turn_key", extend)
    assert await endpoint._extend_turn_key_window("corr-1") is True
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
async def test_an_extension_auth_clamped_to_the_keys_lifetime_still_counts(monkeypatch):
    """auth clamps the last window to the key's lifetime, which covers the
    turn's whole time budget, and answers 200 with the expiry it kept. The
    third Keep going always lands there; refusing it ended the turn an hour
    early."""
    clamped = (datetime.now(timezone.utc) + timedelta(seconds=300)).isoformat()
    monkeypatch.setattr(httpx, "AsyncClient", _client_answering(200, {}, {"expiry_date": clamped}))
    assert await auth_keys.extend_turn_key(instance_id="corr-1", ttl_seconds=6000, settings=_Settings())


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


@pytest.mark.asyncio
async def test_the_datasource_mint_sizes_the_key_to_the_time_limit(time_limit, monkeypatch):
    """The gate and producer mint through this path when datasources are on."""
    seen = {}

    async def _details(**kwargs):
        seen.update(kwargs)
        return auth_keys.MintedTurnKey(key="mdb_turnkey", prefix="mdb_tk")

    monkeypatch.setattr(prod, "mint_turn_key_details", _details)
    await prod._mint_llm_block_with_turn_key_id(
        org_id="o", user_id="u", correlation_id="c", settings=TurnQueueSettings(),
    )
    assert seen["ttl_seconds"] == 3600 + 2400


def test_a_time_limit_whose_key_window_auth_would_refuse_is_rejected(monkeypatch):
    """Every web turn's mint would fail with permission_unavailable."""
    from pydantic import ValidationError

    from cowork.common.settings.app_settings import AUTH_TURN_KEY_MAX_TTL_SECONDS, AUTH_TURN_KEY_SKEW_SECONDS

    # A limit the time-limit cap allows, with a margin that takes its window
    # past auth's per-window cap: only the key-window check refuses it.
    monkeypatch.setenv("COWORK_TURN_TIME_LIMIT_SECONDS", "3600")
    monkeypatch.setenv("COWORK_TURN_TIME_LIMIT_KEY_MARGIN_SECONDS", "2701")
    with pytest.raises(ValidationError):
        TurnQueueSettings()
    monkeypatch.setenv("COWORK_TURN_TIME_LIMIT_KEY_MARGIN_SECONDS", "2700")
    assert TurnQueueSettings().turn_key_lifetime_seconds == AUTH_TURN_KEY_MAX_TTL_SECONDS - AUTH_TURN_KEY_SKEW_SECONDS


def test_a_time_limit_longer_than_the_backstop_and_key_lifetime_cover_is_rejected(monkeypatch):
    """The controller's backstop and auth's key lifetime are sized for four
    60-minute windows. 3700 s fits auth's per-window cap, but its fourth
    window would end on the backstop or a dead key, with no summary."""
    from pydantic import ValidationError

    monkeypatch.setenv("COWORK_TURN_TIME_LIMIT_SECONDS", "3700")
    with pytest.raises(ValidationError):
        TurnQueueSettings()


@pytest.mark.asyncio
async def test_a_failed_extension_is_retried_once(time_limit, monkeypatch):
    from cowork.api.v1.endpoints import responses as endpoint

    extend = AsyncMock(side_effect=[False, True])
    monkeypatch.setattr(endpoint, "extend_turn_key", extend)
    assert await endpoint._extend_turn_key_window("corr-1") is True
    assert extend.call_count == 2
