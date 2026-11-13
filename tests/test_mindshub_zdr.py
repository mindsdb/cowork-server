"""Whether ZDR is on for the caller (cowork.services.mindshub_zdr), read from
``zdr_mode`` on their cached entitlements."""
from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from cowork.services import mindshub_teams
from cowork.services.mindshub_client import MindsHubAnswer, MindsHubReply
from cowork.services.mindshub_zdr import zdr_on

_BROWSER = {"permitted": True, "enabled": True, "daily_bytes": 1073741824, "disabled_reason": None}


def _entitlements(**fields):
    return {"permissions": ["skills.share_org"], "browser": _BROWSER, **fields}


@pytest.fixture
def mindshub(monkeypatch):
    """MindsHub's entitlements answer; ``reply`` overrides it."""
    state = SimpleNamespace(asked=[], reply=MindsHubReply(MindsHubAnswer.OK, payload=_entitlements(zdr_mode="off")))

    async def reply(path, bearer):
        state.asked.append((path, bearer))
        return state.reply

    mindshub_teams.reset_caches_for_tests()
    monkeypatch.setattr(mindshub_teams, "get_mindshub_reply", reply)
    yield state
    mindshub_teams.reset_caches_for_tests()


def _zdr_on(bearer="jwt-alice"):
    return asyncio.run(zdr_on(bearer_token=bearer, org_id="o", user_id="u"))


@pytest.mark.parametrize(("payload", "on"), [
    (_entitlements(zdr_mode="off"), False),
    (_entitlements(zdr_mode="standard"), True),
    (_entitlements(zdr_mode="strict"), True),
    # An auth build from before zdr_mode: off.
    (_entitlements(), False),
    # A mode that isn't a string can't be read: on.
    (_entitlements(zdr_mode=None), True),
    # auth's outage answer, which can't say: on.
    ({"permissions": [], "browser": None}, True),
])
def test_zdr_follows_the_entitlements_mode(mindshub, payload, on):
    mindshub.reply = MindsHubReply(MindsHubAnswer.OK, payload=payload)

    assert _zdr_on() is on


@pytest.mark.parametrize("reply", [
    MindsHubReply(MindsHubAnswer.UNREACHABLE),
    MindsHubReply(MindsHubAnswer.REFUSED, status_code=401),
    MindsHubReply(MindsHubAnswer.THROTTLED, status_code=429),
])
def test_zdr_counts_as_on_when_entitlements_cant_be_read(mindshub, reply):
    mindshub.reply = reply

    assert _zdr_on() is True


def test_zdr_counts_as_on_without_a_credential(mindshub):
    assert _zdr_on(bearer="") is True
    assert mindshub.asked == []


def test_zdr_shares_the_permissions_read_and_its_cache(mindshub):
    assert _zdr_on() is False
    asyncio.run(mindshub_teams.fetch_mindshub_permissions(bearer_token="jwt-alice", org_id="o", user_id="u"))
    assert _zdr_on() is False

    assert mindshub.asked == [("/entitlements/me/", "jwt-alice")]


def test_an_unreadable_answer_is_kept_only_as_long_as_a_failure(mindshub):
    mindshub.reply = MindsHubReply(MindsHubAnswer.UNREACHABLE)
    assert _zdr_on() is True

    key = mindshub_teams.CallerCacheKey.for_caller(org_id="o", user_id="u", bearer_token="jwt-alice")
    cache = mindshub_teams._permissions_cache
    entry = cache._entries[key]
    assert entry.fresh_until - entry.stored_at == pytest.approx(cache.ttl_fail_s)
