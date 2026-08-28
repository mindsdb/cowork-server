"""Google Drive Picker session handoff (org mode).

A `window.open()` popup navigation can't carry an Authorization header, so
the live access token is minted at `POST .../picker/session` time (a real
`fetch()`, which does carry it) and handed to the `GET .../picker` route —
served to the header-less popup — via an opaque, single-use session id
cached in Redis. These tests cover that handoff directly, without a
TestClient/app fixture (matches this repo's existing convention — see
test_connectors_endpoints.py).
"""
from __future__ import annotations

import asyncio

import fakeredis.aioredis
import pytest
from fastapi import HTTPException

from cowork.api.v1.endpoints.connectors import oauth as oauth_endpoints
from cowork.db.scoped import TenantScope
from cowork.services.connectors.oauth import picker_session
from tests._fakes import FakeRequest

ORG_SCOPE = TenantScope(org_mode=True, org_id="org-1", user_id="user-1")
LOCAL_SCOPE = TenantScope(org_mode=False)


@pytest.fixture
def org_mode(monkeypatch):
    monkeypatch.setenv("COWORK_TENANCY_MODE", "org")
    from cowork.common.settings.app_settings import get_app_settings
    get_app_settings.cache_clear()
    yield
    get_app_settings.cache_clear()


@pytest.fixture
def fake_redis(monkeypatch):
    client = fakeredis.aioredis.FakeRedis(decode_responses=True)
    monkeypatch.setattr(picker_session, "get_redis", lambda: client)
    return client


@pytest.mark.asyncio
async def test_consume_uses_atomic_getdel(fake_redis, monkeypatch):
    """A separate GET-then-DELETE lets two concurrent consumes of the same
    session id both read the live token before either deletes it — GETDEL
    (a single atomic Redis op) is required, not just "get, then delete"."""
    calls = []
    real_getdel = fake_redis.getdel

    async def tracking_getdel(key):
        calls.append(key)
        return await real_getdel(key)

    def fail_get(*a, **k):
        raise AssertionError("consume() must not call GET separately from GETDEL")

    def fail_delete(*a, **k):
        raise AssertionError("consume() must not call DELETE separately from GETDEL")

    monkeypatch.setattr(fake_redis, "getdel", tracking_getdel)
    monkeypatch.setattr(fake_redis, "get", fail_get)
    monkeypatch.setattr(fake_redis, "delete", fail_delete)

    await fake_redis.set("oauth:picker:session:s1", '{"access_token": "t"}')
    result = await picker_session.consume("s1")
    assert result == {"access_token": "t"}
    assert calls == ["oauth:picker:session:s1"]


@pytest.mark.asyncio
async def test_concurrent_consumes_only_one_wins(fake_redis):
    session_id = await picker_session.create({"access_token": "t"})

    results = await asyncio.gather(
        picker_session.consume(session_id),
        picker_session.consume(session_id),
    )
    winners = [r for r in results if r is not None]
    assert len(winners) == 1
    assert winners[0] == {"access_token": "t"}


@pytest.fixture
def picker_ready(monkeypatch):
    """Stubs auth_proxy.proxy_token so no real network call happens, and a
    Picker API key so the "not fully configured" 503 branch isn't hit."""
    async def fake_proxy_token(engine, request, settings, *, name=""):
        assert engine == "google_drive"
        return {
            "access_token": "live-token-abc",
            "account_email": "a@b.com",
            "picker_api_key": "picker-key-xyz",
            "app_id": "app-123",
        }

    monkeypatch.setattr(oauth_endpoints.auth_proxy, "proxy_token", fake_proxy_token)


@pytest.mark.asyncio
async def test_create_session_mints_token_while_bearer_present(org_mode, fake_redis, picker_ready):
    request = FakeRequest(headers={"authorization": "Bearer real-user-token"})

    result = await oauth_endpoints.create_picker_session(
        "google_drive", request, ORG_SCOPE, body={"account_email": "a@b.com", "file_ids": ["f1"]},
    )

    assert result["url"].startswith("/api/v1/connectors/oauth/google_drive/picker?session=")
    session_id = result["url"].rsplit("session=", 1)[1]
    assert await fake_redis.exists(f"oauth:picker:session:{session_id}")


@pytest.mark.asyncio
async def test_create_session_forwards_the_connection_name(org_mode, fake_redis, monkeypatch):
    """auth auto-resolves a lone connection without `name`, but 400s on an
    ambiguous one — an org with two Drive connections needs this forwarded."""
    seen = {}

    async def fake_proxy_token(engine, request, settings, *, name=""):
        seen["name"] = name
        return {"access_token": "t", "picker_api_key": "k", "app_id": "a"}

    monkeypatch.setattr(oauth_endpoints.auth_proxy, "proxy_token", fake_proxy_token)
    request = FakeRequest(headers={"authorization": "Bearer real-user-token"})
    await oauth_endpoints.create_picker_session(
        "google_drive", request, ORG_SCOPE, body={"name": "work", "account_email": "a@b.com"},
    )
    assert seen["name"] == "work"


@pytest.mark.asyncio
async def test_get_picker_embeds_session_token_and_is_single_use(org_mode, fake_redis, picker_ready):
    request = FakeRequest(headers={"authorization": "Bearer real-user-token"})
    session = await oauth_endpoints.create_picker_session("google_drive", request, ORG_SCOPE, body={})
    session_id = session["url"].rsplit("session=", 1)[1]

    # The popup's GET carries no Authorization header at all — the whole
    # point of this handoff.
    response = await oauth_endpoints.oauth_picker("google_drive", FakeRequest(), ORG_SCOPE, session=session_id)
    assert response.status_code == 200
    body = response.body.decode()
    assert "live-token-abc" in body
    assert "picker-key-xyz" in body

    # Single-use: the same session id must not work a second time.
    replay = await oauth_endpoints.oauth_picker("google_drive", FakeRequest(), ORG_SCOPE, session=session_id)
    assert replay.status_code == 404


@pytest.mark.asyncio
async def test_get_picker_unknown_session_returns_styled_404(org_mode, fake_redis):
    response = await oauth_endpoints.oauth_picker("google_drive", FakeRequest(), ORG_SCOPE, session="does-not-exist")
    assert response.status_code == 404
    assert "expired" in response.body.decode().lower()


@pytest.mark.asyncio
async def test_create_session_rejects_engine_without_picker_support(org_mode, fake_redis):
    with pytest.raises(HTTPException) as exc_info:
        await oauth_endpoints.create_picker_session("gmail", FakeRequest(), ORG_SCOPE, body={})
    assert exc_info.value.status_code == 404


@pytest.mark.asyncio
async def test_create_session_404s_outside_org_mode(fake_redis, picker_ready):
    with pytest.raises(HTTPException) as exc_info:
        await oauth_endpoints.create_picker_session("google_drive", FakeRequest(), LOCAL_SCOPE, body={})
    assert exc_info.value.status_code == 404
