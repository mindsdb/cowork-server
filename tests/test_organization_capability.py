from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from cowork.api.v1.router import api_router
from cowork.common.settings.app_settings import get_app_settings
from cowork.principal import Principal, get_principal

PATH = "/api/v1/capabilities/organization-switch"
PRINCIPAL = Principal(
    user_id="0f7f0b6a-3f0f-4c58-9e0c-6dbb3ac0f0a1",
    org_id="6ba7b810-9dad-11d1-80b4-00c04fd430c8",
)
JWT = "eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJ1c2VyIn0.signature"


@pytest.fixture(autouse=True)
def _reset_app_settings():
    get_app_settings.cache_clear()
    yield
    get_app_settings.cache_clear()


def _client(principal: Principal | None = PRINCIPAL) -> TestClient:
    app = FastAPI()
    app.include_router(api_router)
    app.dependency_overrides[get_principal] = lambda: principal
    return TestClient(app)


def _configure(
    monkeypatch,
    *,
    tenancy_mode: str,
    identity_enforce: str,
    boundary_mode: str,
    switch_enabled: bool,
) -> None:
    monkeypatch.setenv("COWORK_TENANCY_MODE", tenancy_mode)
    monkeypatch.setenv("COWORK_IDENTITY_ENFORCE", identity_enforce)
    monkeypatch.setenv("COWORK_ORGANIZATION_BOUNDARY_MODE", boundary_mode)
    monkeypatch.setenv(
        "COWORK_ORGANIZATION_SWITCH_ENABLED", "true" if switch_enabled else "false"
    )


@pytest.mark.parametrize(
    (
        "tenancy_mode",
        "identity_enforce",
        "boundary_mode",
        "switch_enabled",
        "expected_enforced",
        "expected_enabled",
    ),
    [
        ("local", "enforce", "enforce", True, False, False),
        ("org", "audit", "enforce", True, False, False),
        ("org", "enforce", "audit", True, False, False),
        ("org", "enforce", "enforce", False, True, False),
        ("org", "enforce", "enforce", True, True, True),
    ],
)
def test_organization_switch_capability_is_fail_closed(
    monkeypatch,
    tenancy_mode,
    identity_enforce,
    boundary_mode,
    switch_enabled,
    expected_enforced,
    expected_enabled,
):
    _configure(
        monkeypatch,
        tenancy_mode=tenancy_mode,
        identity_enforce=identity_enforce,
        boundary_mode=boundary_mode,
        switch_enabled=switch_enabled,
    )

    response = _client().get(PATH)

    assert response.status_code == 200
    assert response.json() == {
        "protocolVersion": 1,
        "expectedOrganizationEnforced": expected_enforced,
        "enabled": expected_enabled,
    }
    assert response.headers.get("cache-control") == "no-store"


@pytest.mark.parametrize("tenancy_mode", ["local", "org"])
def test_organization_switch_capability_requires_a_principal(monkeypatch, tenancy_mode):
    from cowork.server import create_app

    _configure(
        monkeypatch,
        tenancy_mode=tenancy_mode,
        identity_enforce="enforce",
        boundary_mode="enforce",
        switch_enabled=True,
    )

    response = TestClient(create_app()).get(PATH)

    assert response.status_code == 401
    assert response.json() == {"detail": "Unauthorized"}
    assert response.headers.get("cache-control") == "no-store"


def test_real_app_advertises_enabled_protocol_through_the_boundary(monkeypatch):
    from cowork.server import create_app

    _configure(
        monkeypatch,
        tenancy_mode="org",
        identity_enforce="enforce",
        boundary_mode="enforce",
        switch_enabled=True,
    )

    response = TestClient(create_app()).get(
        PATH,
        headers={
            "Authorization": f"Bearer {JWT}",
            "X-User-Id": PRINCIPAL.user_id,
            "X-Organization-Id": PRINCIPAL.org_id,
            "X-Cowork-Expected-Organization-Id": PRINCIPAL.org_id,
        },
    )

    assert response.status_code == 200
    assert response.json() == {
        "protocolVersion": 1,
        "expectedOrganizationEnforced": True,
        "enabled": True,
    }
    assert response.headers.get("cache-control") == "no-store"
