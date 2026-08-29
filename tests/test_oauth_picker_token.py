"""Google Drive Picker token mint (org mode).

Replaces the old two-step session handoff: a `window.open()` popup navigation
can't carry an Authorization header, so the previous design minted the live
token at POST time and handed it to a header-less `GET .../picker` route via
an opaque, single-use Redis-backed ticket. The Picker now renders in-page in
the SPA itself, so there is no header-less hop left — `mint_picker_token` just
returns the live token directly to the caller's own (authenticated) `fetch()`.
"""
from __future__ import annotations

import pytest
from fastapi import HTTPException

from cowork.api.v1.endpoints.connectors import oauth as oauth_endpoints
from cowork.db.scoped import TenantScope
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
async def test_mint_token_returns_the_live_token_while_bearer_present(org_mode, picker_ready):
    request = FakeRequest(headers={"authorization": "Bearer real-user-token"})

    result = await oauth_endpoints.mint_picker_token(
        "google_drive", request, ORG_SCOPE, body={"account_email": "a@b.com"},
    )

    assert result == {
        "access_token": "live-token-abc",
        "account_email": "a@b.com",
        "api_key": "picker-key-xyz",
        "app_id": "app-123",
    }


@pytest.mark.asyncio
async def test_mint_token_forwards_the_connection_name(org_mode, monkeypatch):
    """auth auto-resolves a lone connection without `name`, but 400s on an
    ambiguous one — an org with two Drive connections needs this forwarded."""
    seen = {}

    async def fake_proxy_token(engine, request, settings, *, name=""):
        seen["name"] = name
        return {"access_token": "t", "picker_api_key": "k", "app_id": "a"}

    monkeypatch.setattr(oauth_endpoints.auth_proxy, "proxy_token", fake_proxy_token)
    request = FakeRequest(headers={"authorization": "Bearer real-user-token"})
    await oauth_endpoints.mint_picker_token(
        "google_drive", request, ORG_SCOPE, body={"name": "work", "account_email": "a@b.com"},
    )
    assert seen["name"] == "work"


@pytest.mark.asyncio
async def test_mint_token_503s_when_not_fully_configured(org_mode, monkeypatch):
    async def fake_proxy_token(engine, request, settings, *, name=""):
        return {"access_token": "t"}  # no picker_api_key, no app_id, and no configured fallback

    monkeypatch.setattr(oauth_endpoints.auth_proxy, "proxy_token", fake_proxy_token)
    with pytest.raises(HTTPException) as exc_info:
        await oauth_endpoints.mint_picker_token("google_drive", FakeRequest(), ORG_SCOPE, body={})
    assert exc_info.value.status_code == 503


@pytest.mark.asyncio
async def test_mint_token_rejects_engine_without_picker_support(org_mode):
    with pytest.raises(HTTPException) as exc_info:
        await oauth_endpoints.mint_picker_token("gmail", FakeRequest(), ORG_SCOPE, body={})
    assert exc_info.value.status_code == 404


@pytest.mark.asyncio
async def test_mint_token_404s_outside_org_mode(picker_ready):
    with pytest.raises(HTTPException) as exc_info:
        await oauth_endpoints.mint_picker_token("google_drive", FakeRequest(), LOCAL_SCOPE, body={})
    assert exc_info.value.status_code == 404
