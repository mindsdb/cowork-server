"""Whether the caller may build a custom connector, as the picker asks it.

The route sits beside /connectors/specs/{connector_id}, so one test goes
through the app to prove the path reaches it rather than a connector lookup.
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

import cowork.server as cowork_server
from cowork.api.v1.endpoints.connectors import specs as specs_endpoints
from cowork.common.settings.app_settings import get_app_settings
from cowork.db.personal_resources import ALL_PERSONAL_RESOURCES, PersonalResource
from cowork.db.scoped import LOCAL_SCOPE, TenantScope

TOKEN = "custom-connector-availability-token"


@pytest.fixture()
def client(monkeypatch):
    """The app, with a known auth token; the settings cache is cleared after
    the test whatever its outcome, so the token never leaks into later tests."""
    monkeypatch.setenv("COWORK_AUTH_TOKEN", TOKEN)
    get_app_settings.cache_clear()
    yield TestClient(
        cowork_server.create_app(), base_url="http://127.0.0.1:26866", client=("127.0.0.1", 54321)
    )
    get_app_settings.cache_clear()


def test_the_route_answers_on_a_single_user_install(client):
    response = client.get(
        "/api/v1/connectors/specs/custom-availability", headers={"Authorization": f"Bearer {TOKEN}"},
    )

    assert response.status_code == 200
    assert response.json() == {"can_build": True}


@pytest.mark.parametrize("scope, can_build", [
    (LOCAL_SCOPE, True),
    (TenantScope(org_id="p-a", user_id="a", personal=True, personal_resources=ALL_PERSONAL_RESOURCES), True),
    (TenantScope(
        org_id="p-a", user_id="a", personal=True,
        personal_resources=ALL_PERSONAL_RESOURCES - {PersonalResource.CONNECTIONS},
    ), False),
    (TenantScope(org_mode=True, org_id="org-a", user_id="a"), False),
], ids=["single-user install", "personal connections", "shared connections", "org mode"])
@pytest.mark.asyncio
async def test_building_is_offered_only_in_one_persons_scope(scope, can_build):
    result = await specs_endpoints.custom_connector_availability(scope)

    assert result.can_build is can_build


@pytest.mark.asyncio
async def test_the_operator_view_is_not_offered_building(personal_deployment):
    result = await specs_endpoints.custom_connector_availability(LOCAL_SCOPE)

    assert result.can_build is False
