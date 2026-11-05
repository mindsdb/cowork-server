"""SharePoint is saved only by its own routes, which check the key first.

The generic connector save paths (form submission, direct save, validate and
save) refuse a SharePoint connection however it is named, stage nothing and
store nothing, and say where to set it up instead.
"""
from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from cowork.api.v1.router import api_router
from cowork.common.settings.app_settings import get_app_settings
from cowork.db.scoped import LOCAL_SCOPE
from cowork.principal import Principal, get_principal
from cowork.services.connectors.persist import vault_for_scope
from cowork.services.connectors.submissions import store

KEY = {
    "tenant_id": "contoso.onmicrosoft.com",
    "client_id": "11111111-2222-3333-4444-555555555555",
    "client_secret": "sp-generic-secret-91c2",
    "site_urls": "https://contoso.sharepoint.com/sites/team",
}
DESKTOP = "Update Cowork, then set up SharePoint in Customize, under Connections, where its key is checked."
ORG = "SharePoint is available in the desktop app."
ORG_ID = "6ba7b810-9dad-11d1-80b4-00c04fd430c8"
USER = "11111111-1111-4111-8111-111111111111"


@pytest.fixture
def stores(monkeypatch, tmp_path):
    monkeypatch.setenv("COWORK_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("COWORK_SHARED_DIR", str(tmp_path / "shared"))
    monkeypatch.setenv("COWORK_VAULT_DIR", str(tmp_path / "home" / "data-vault"))
    get_app_settings.cache_clear()
    yield tmp_path
    get_app_settings.cache_clear()


def _client(app: FastAPI | None = None) -> TestClient:
    if app is None:
        app = FastAPI()
        app.include_router(api_router)
    return TestClient(app, base_url="http://127.0.0.1:26866", client=("127.0.0.1", 54321))


def _sharepoint_records() -> list:
    return [c for c in vault_for_scope(LOCAL_SCOPE).list_connections() if c.get("engine") == "sharepoint"]


@pytest.mark.parametrize(
    "body",
    [
        {"connector_id": "sharepoint", "method": "app-key", "values": KEY},
        {"form_spec": {"_connector_id": "sharepoint", "fields": []}, "values": KEY},
        {"form_spec": {"engine": "sharepoint", "fields": []}, "values": KEY},
        {"form_id": "sharepoint-connector", "form_spec": {"fields": []}, "values": KEY},
    ],
    ids=["connector_id", "stamped form", "handcrafted engine", "form_id"],
)
def test_a_form_submission_for_sharepoint_is_refused_and_stages_nothing(stores, body):
    staged_before = len(store._store)

    response = _client().post("/api/v1/connectors/submissions/", json=body)

    assert response.status_code == 409
    assert response.json()["detail"] == DESKTOP
    assert len(store._store) == staged_before
    assert KEY["client_secret"] not in response.text
    assert _sharepoint_records() == []


@pytest.mark.parametrize("route", ["/api/v1/connectors/connections/save", "/api/v1/connectors/connections/validate-and-save"])
def test_a_direct_save_for_sharepoint_is_refused_and_stores_nothing(stores, route):
    response = _client().post(route, json={"connector_id": "sharepoint", "method": "app-key", "name": "corp", "values": KEY})

    assert response.status_code == 409
    assert response.json()["detail"] == DESKTOP
    assert _sharepoint_records() == []


def test_in_org_mode_the_refusal_points_to_the_desktop_app(stores, monkeypatch):
    monkeypatch.setenv("COWORK_TENANCY_MODE", "org")
    get_app_settings.cache_clear()
    app = FastAPI()
    app.include_router(api_router)
    app.dependency_overrides[get_principal] = lambda: Principal(user_id=USER, org_id=ORG_ID)
    staged_before = len(store._store)

    response = _client(app).post(
        "/api/v1/connectors/submissions/",
        json={"form_spec": {"_connector_id": "sharepoint", "fields": []}, "values": KEY},
    )

    assert response.status_code == 409
    assert response.json()["detail"] == ORG
    assert len(store._store) == staged_before


def test_other_connectors_still_save_through_the_generic_path(stores):
    values = {"host": "db.example.com", "user": "demo", "password": "pg-generic-secret-5e1a"}

    response = _client().post(
        "/api/v1/connectors/connections/save",
        json={"connector_id": "postgres", "method": "host-port", "name": "corp", "values": values},
    )

    assert response.status_code == 200, response.text
    record = vault_for_scope(LOCAL_SCOPE).read_record("postgres", "corp")
    assert record is not None
    assert record["fields"]["host"] == "db.example.com"
