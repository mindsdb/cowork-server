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


@pytest.mark.asyncio
async def test_org_mode_include_unavailable_returns_all_flagged(monkeypatch):
    """`include_unavailable` keeps the whole registry but marks what cloud
    can't run, so the directory can list those under a desktop-only group
    instead of hiding them."""

    async def fake_proxy_catalogue(request, settings):
        return {"items": [{"id": "google_drive"}, {"id": "gmail"}]}

    monkeypatch.setattr(specs_endpoints.auth_proxy, "proxy_catalogue", fake_proxy_catalogue)

    result = await specs_endpoints.list_connector_specs(
        ORG_SCOPE, FakeRequest(), include_unavailable=True
    )

    assert len(result) > 100
    by_id = {c.id: c for c in result}
    assert by_id["gmail"].cloud_available is True
    assert by_id["google_drive"].cloud_available is True
    assert by_id["postgres"].cloud_available is False


@pytest.mark.asyncio
async def test_local_mode_reports_every_connector_available():
    """Desktop runs the whole registry, so nothing is ever desktop-only there."""
    result = await specs_endpoints.list_connector_specs(LOCAL_SCOPE, FakeRequest())

    assert all(c.cloud_available for c in result)
