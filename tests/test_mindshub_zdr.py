"""Whether ZDR is on for the caller (cowork.services.mindshub_zdr), read from
``zdr_mode`` on their cached entitlements."""
from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from cowork.services import mindshub_teams, mindshub_zdr
from cowork.services.mindshub_client import MindsHubAnswer, MindsHubReply
from cowork.services.mindshub_zdr import ZdrState, zdr_on, zdr_on_for, zdr_state

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


@pytest.mark.parametrize(
    ("org_mode", "org_bearer", "turn_key", "expected"),
    [
        (True, "jwt-alice", "turn-key", "jwt-alice"),
        (True, "", "turn-key", "turn-key"),
        (False, "jwt-alice", "turn-key", "turn-key"),
        (False, "", "", "mdb_stored"),
    ],
)
def test_zdr_on_for_asks_with_the_callers_credential(mindshub, monkeypatch, org_mode, org_bearer, turn_key, expected):
    # The web caller's bearer first, else the turn's minted key, else the stored MindsHub key.
    import cowork.common.settings.user_settings as user_settings

    monkeypatch.setattr(user_settings, "provider_api_key_str", lambda _settings, _provider: "mdb_stored")
    scope = SimpleNamespace(org_mode=org_mode, org_id="o", user_id="u")
    asyncio.run(zdr_on_for(scope=scope, settings=None, org_bearer=org_bearer, turn_key=turn_key))
    assert mindshub.asked[-1][1] == expected


@pytest.mark.parametrize(("needs", "read", "expected", "asked"), [
    (False, True, False, False),
    (True, True, True, True),
    (True, False, False, True),
    (True, RuntimeError("auth down"), True, True),
])
@pytest.mark.asyncio
async def test_the_browser_asks_zdr_only_when_jev_would_pick_its_steps(monkeypatch, needs, read, expected, asked):
    from cowork.handlers import jev_route

    monkeypatch.setattr(jev_route, "browser_needs_zdr", lambda settings: needs)
    calls = []

    async def zdr_on_for(**kwargs):
        calls.append(kwargs)
        if isinstance(read, Exception):
            raise read
        return read

    monkeypatch.setattr(mindshub_zdr, "zdr_on_for", zdr_on_for)
    assert await mindshub_zdr.browser_zdr_on(scope=None, settings=None, org_bearer="tok") is expected
    assert bool(calls) is asked


@pytest.mark.parametrize(("reply", "state"), [
    (MindsHubReply(MindsHubAnswer.OK, payload=_entitlements(zdr_mode="off")), ZdrState.OFF),
    (MindsHubReply(MindsHubAnswer.OK, payload=_entitlements(zdr_mode="strict")), ZdrState.ON),
    (MindsHubReply(MindsHubAnswer.OK, payload=_entitlements(zdr_mode=None)), ZdrState.UNKNOWN),
    (MindsHubReply(MindsHubAnswer.OK, payload={"permissions": [], "browser": None}), ZdrState.UNKNOWN),
    (MindsHubReply(MindsHubAnswer.UNREACHABLE), ZdrState.UNKNOWN),
    (MindsHubReply(MindsHubAnswer.REFUSED, status_code=401), ZdrState.UNKNOWN),
])
def test_zdr_state_tells_an_unreadable_answer_from_zdr_on(mindshub, reply, state):
    mindshub.reply = reply

    assert asyncio.run(zdr_state(bearer_token="jwt-alice", org_id="o", user_id="u")) is state


def test_zdr_state_is_unknown_without_a_credential(mindshub):
    assert asyncio.run(zdr_state(bearer_token="", org_id="o", user_id="u")) is ZdrState.UNKNOWN
    assert mindshub.asked == []


@pytest.mark.parametrize(("needs", "read", "expected"), [
    (False, ZdrState.ON, ZdrState.OFF),
    (True, ZdrState.ON, ZdrState.ON),
    (True, ZdrState.UNKNOWN, ZdrState.UNKNOWN),
    (True, RuntimeError("auth down"), ZdrState.UNKNOWN),
])
@pytest.mark.asyncio
async def test_the_browsers_zdr_state_marks_a_failed_read_unknown(monkeypatch, needs, read, expected):
    from cowork.handlers import jev_route

    monkeypatch.setattr(jev_route, "browser_needs_zdr", lambda settings: needs)

    async def zdr_state_for(**kwargs):
        if isinstance(read, Exception):
            raise read
        return read

    monkeypatch.setattr(mindshub_zdr, "zdr_state_for", zdr_state_for)
    assert await mindshub_zdr.browser_zdr_state(scope=None, settings=None, org_bearer="tok") is expected
