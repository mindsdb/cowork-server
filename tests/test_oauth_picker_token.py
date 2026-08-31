"""Google Drive Picker token mint (org mode).

The SPA builds the Picker itself, so this endpoint hands the live token
straight back to the caller's own authenticated fetch. Its defining
property is that it is stateless — no ticket, no store, nothing to expire.
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


def _stub_token(monkeypatch, payload: dict):
    async def fake_proxy_token(engine, request, settings, *, name=""):
        return payload

    monkeypatch.setattr(oauth_endpoints.auth_proxy, "proxy_token", fake_proxy_token)


@pytest.mark.asyncio
async def test_mint_token_returns_the_live_token_while_bearer_present(picker_ready):
    request = FakeRequest(headers={"authorization": "Bearer real-user-token"})

    result = await oauth_endpoints.mint_picker_token(
        "google_drive", request, ORG_SCOPE, body={},
    )

    assert result.access_token == "live-token-abc"
    # auth calls it picker_api_key; the client reads api_key. The rename is
    # real, so pin it here rather than only in the client's hand-written type.
    assert result.api_key == "picker-key-xyz"
    assert result.app_id == "app-123"
    assert result.account_email == "a@b.com"


@pytest.mark.asyncio
async def test_mint_token_forwards_the_connection_name(monkeypatch):
    """auth auto-resolves a lone connection without `name`, but 400s on an
    ambiguous one — an org with two Drive connections needs this forwarded."""
    seen = {}

    async def fake_proxy_token(engine, request, settings, *, name=""):
        seen["name"] = name
        return {"access_token": "t", "picker_api_key": "k", "app_id": "a"}

    monkeypatch.setattr(oauth_endpoints.auth_proxy, "proxy_token", fake_proxy_token)
    request = FakeRequest(headers={"authorization": "Bearer real-user-token"})
    await oauth_endpoints.mint_picker_token(
        "google_drive", request, ORG_SCOPE, body={"name": "work"},
    )
    assert seen["name"] == "work"


@pytest.mark.asyncio
async def test_account_email_comes_only_from_auth_never_from_the_request_body():
    """The client already knows what it sent; echoing it back would round-trip
    unvalidated input through a credential response for nothing."""
    async def fake_proxy_token(engine, request, settings, *, name=""):
        return {"access_token": "t", "picker_api_key": "k", "app_id": "a"}

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(oauth_endpoints.auth_proxy, "proxy_token", fake_proxy_token)
        result = await oauth_endpoints.mint_picker_token(
            "google_drive", FakeRequest(), ORG_SCOPE,
            body={"account_email": "attacker@example.com"},
        )
    assert result.account_email == ""


@pytest.mark.asyncio
@pytest.mark.parametrize("missing", ["access_token", "picker_api_key", "app_id"])
async def test_mint_token_503s_when_any_required_field_is_missing(monkeypatch, missing):
    """One case per field. A single combined case goes green off whichever
    field happens to be empty, so the other guards can rot unnoticed — the
    api_key half in particular, since a deployment-configured
    GOOGLE_PICKER_API_KEY fills it in from settings."""
    # Cleared explicitly: OAuthSettings reads an .env chain, so a machine with
    # this key set would otherwise satisfy the api_key guard from settings.
    monkeypatch.setenv("GOOGLE_PICKER_API_KEY", "")
    payload = {"access_token": "t", "picker_api_key": "k", "app_id": "a"}
    payload[missing] = ""
    _stub_token(monkeypatch, payload)

    with pytest.raises(HTTPException) as exc_info:
        await oauth_endpoints.mint_picker_token("google_drive", FakeRequest(), ORG_SCOPE, body={})
    assert exc_info.value.status_code == 503


@pytest.mark.asyncio
async def test_mint_token_ignores_a_legacy_file_ids_body(picker_ready):
    """An old client still posting file_ids must be served, not rejected —
    the picker takes them client-side now. Pins the permissive body against a
    later switch to a typed model with extra="forbid"."""
    result = await oauth_endpoints.mint_picker_token(
        "google_drive", FakeRequest(), ORG_SCOPE,
        body={"file_ids": ["1a2B3c"], "project_name": "proj"},
    )
    assert result.access_token == "live-token-abc"


@pytest.mark.asyncio
async def test_mint_token_touches_no_redis(picker_ready, monkeypatch):
    """Statelessness is the whole point of the change — the old flow kept a
    single-use ticket in Redis to bridge to a header-less popup navigation."""
    import cowork.turnqueue.redis_client as redis_client

    def fail(*a, **k):
        raise AssertionError("mint_picker_token must not reach Redis")

    monkeypatch.setattr(redis_client, "get_redis", fail)
    result = await oauth_endpoints.mint_picker_token("google_drive", FakeRequest(), ORG_SCOPE, body={})
    assert result.access_token == "live-token-abc"


@pytest.mark.asyncio
async def test_mint_token_rejects_engine_without_picker_support():
    with pytest.raises(HTTPException) as exc_info:
        await oauth_endpoints.mint_picker_token("gmail", FakeRequest(), ORG_SCOPE, body={})
    assert exc_info.value.status_code == 404


@pytest.mark.asyncio
async def test_mint_token_404s_outside_org_mode(picker_ready):
    with pytest.raises(HTTPException) as exc_info:
        await oauth_endpoints.mint_picker_token("google_drive", FakeRequest(), LOCAL_SCOPE, body={})
    assert exc_info.value.status_code == 404


@pytest.mark.asyncio
async def test_retired_session_route_tells_a_stale_tab_to_reload():
    """410, not 404: a browser tab still running the previous bundle would
    otherwise report the generic "could not start the file picker"."""
    with pytest.raises(HTTPException) as exc_info:
        await oauth_endpoints.create_picker_session("google_drive", ORG_SCOPE)
    assert exc_info.value.status_code == 410
    assert "reload" in exc_info.value.detail.lower()
