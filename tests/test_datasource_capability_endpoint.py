"""GET /capabilities/datasources — what this deployment may run, for the client.

It answers what the policy knows, so a client can tell "this deployment does
not run databases" from "it runs them and this one is off". Authenticated,
like the organization-switch capability beside it: a caller with no principal
has nothing to be told about.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from cowork.common.settings.app_settings import get_app_settings
from cowork.principal import Principal, get_principal
from cowork.server import create_app

PATH = "/api/v1/capabilities/datasources"
PRINCIPAL = Principal(
    user_id="0f7f0b6a-3f0f-4c58-9e0c-6dbb3ac0f0a1",
    org_id="6ba7b810-9dad-11d1-80b4-00c04fd430c8",
)


@pytest.fixture(autouse=True)
def _reset_app_settings():
    get_app_settings.cache_clear()
    yield
    get_app_settings.cache_clear()


def _client(principal: Principal | None = PRINCIPAL) -> TestClient:
    # The real application, not a bare router: the no-store guarantee below is
    # a middleware on the /api/v1/capabilities prefix, and a hand-built app
    # would pin only the header the handler sets on a success.
    app = create_app()
    app.dependency_overrides[get_principal] = lambda: principal
    return TestClient(app)


def test_a_caller_without_a_principal_is_refused():
    assert _client(principal=None).get(PATH).status_code == 401


def test_the_default_reports_the_methods_it_knows_and_none_of_them_available():
    body = _client().get(PATH).json()

    assert body["manifestVersion"] == 1
    assert body["datasources"]["postgres"]["methods"]["host-port"]["available"] is False
    assert body["datasources"]["mysql"]["methods"]["host-password"]["available"] is False
    # A method with no cloud block is not a capability at all.
    assert "connection-string" not in body["datasources"]["postgres"]["methods"]


def test_an_enabled_pair_is_the_only_one_that_reads_as_available(monkeypatch, adapter_verified_datasources):
    monkeypatch.setenv("COWORK_DATASOURCE_CAPABILITIES", '{"manifest_version": 1, "enabled": ["postgres:host-port"]}')

    body = _client().get(PATH).json()

    assert body["datasources"]["postgres"]["methods"]["host-port"]["available"] is True
    assert body["datasources"]["mysql"]["methods"]["host-password"]["available"] is False


def test_a_method_the_specs_call_unverified_stays_off_however_it_is_configured(monkeypatch):
    # No adapter_verified fixture: this is what the release ships today.
    monkeypatch.setenv("COWORK_DATASOURCE_CAPABILITIES", '{"manifest_version": 1, "enabled": ["postgres:host-port"]}')

    body = _client().get(PATH).json()

    assert body["datasources"]["postgres"]["methods"]["host-port"]["available"] is False


def test_a_manifest_from_another_version_reports_everything_unavailable(monkeypatch):
    monkeypatch.setenv("COWORK_DATASOURCE_CAPABILITIES", '{"manifest_version": 2, "enabled": ["postgres:host-port"]}')

    body = _client().get(PATH).json()

    assert body["manifestVersion"] == 1
    every = [
        method["available"]
        for connector in body["datasources"].values()
        for method in connector["methods"].values()
    ]
    assert every and not any(every)


def test_the_answer_is_never_cached(monkeypatch):
    monkeypatch.setenv("COWORK_DATASOURCE_CAPABILITIES", '{"manifest_version": 1, "enabled": ["postgres:host-port"]}')

    response = _client().get(PATH)

    assert response.headers["cache-control"] == "no-store"
