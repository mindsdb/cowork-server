"""Org-mode branch of GET /connectors/specs/ — the full ~230-connector
registry is a desktop concept; org mode must restrict it to whatever auth's
catalogue authorizes (same allow-list list_connections already trusts).
Direct function calls, no TestClient/app fixture — matches this repo's
existing convention (see test_connections_org_mode.py).
"""
from __future__ import annotations

import pytest

from cowork.api.v1.endpoints.connectors import specs as specs_endpoints
from cowork.db.scoped import LOCAL_SCOPE, TenantScope
from tests._fakes import FakeRequest

ORG_SCOPE = TenantScope(org_mode=True, org_id="org-1", user_id="user-1")


@pytest.mark.asyncio
async def test_local_mode_returns_the_full_registry():
    result = await specs_endpoints.list_connector_specs(LOCAL_SCOPE, FakeRequest())
    assert len(result) > 100
    assert any(c.id == "postgres" for c in result)


@pytest.mark.asyncio
async def test_org_mode_restricts_to_auths_catalogue(monkeypatch):
    async def fake_proxy_catalogue(request, settings):
        return {"items": [{"id": "google_drive"}, {"id": "gmail"}]}

    monkeypatch.setattr(specs_endpoints.auth_proxy, "proxy_catalogue", fake_proxy_catalogue)

    result = await specs_endpoints.list_connector_specs(ORG_SCOPE, FakeRequest())

    ids = {c.id for c in result}
    assert ids == {"google_drive", "gmail"}


@pytest.mark.asyncio
async def test_org_mode_with_an_empty_catalogue_returns_no_connectors(monkeypatch):
    async def fake_proxy_catalogue(request, settings):
        return {"items": []}

    monkeypatch.setattr(specs_endpoints.auth_proxy, "proxy_catalogue", fake_proxy_catalogue)

    result = await specs_endpoints.list_connector_specs(ORG_SCOPE, FakeRequest())

    assert result == []
