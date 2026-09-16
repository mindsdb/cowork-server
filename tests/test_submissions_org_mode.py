"""POST /connectors/submissions/ in org mode, through create_app().

Org mode has no encrypted relay for static credentials yet. A submission
that got through staged every value in the process-global SubmissionStore
for 24 hours, then either wrote a plaintext env file for a probe that org
mode refuses anyway, or saved a handcrafted form into a local vault on the
pod. The router declares DesktopOnly, which refuses before FastAPI validates
the body, so none of those paths run. Desktop keeps the whole flow.

Same shape as test_front_door_http.py: the real router stack, gateway
identity headers, function-scoped clients, settings cache cleared around
every test.
"""
from __future__ import annotations

import logging
from types import SimpleNamespace

import pytest
from anton.core.datasources import data_vault
from fastapi.testclient import TestClient

from cowork.api.v1.endpoints.connectors import submissions as submissions_endpoints
from cowork.common.settings.app_settings import get_app_settings
from cowork.handlers import probe as probe_handlers
from cowork.server import create_app
from cowork.services.connectors import persist
from cowork.services.connectors.connections import ConnectionsService
from cowork.services.connectors.specs._registry import registry
from cowork.services.connectors.submissions import store

SUBMISSIONS = "/api/v1/connectors/submissions/"
REFUSAL = "not available in org deployments"

#: Valid gateway-injected identity, so TrustedHeaderMiddleware lets the
#: request reach the route's own declaration.
MEMBER_HEADERS = {
    "X-User-Id": "11111111-1111-4111-8111-111111111111",
    "X-Organization-Id": "22222222-2222-4222-8222-222222222222",
}

# Sentinels, not credentials. Distinct so a leak is attributable.
PASSWORD = "sentinel-postgres-password-7f3a"
API_KEY = "sentinel-handcrafted-api-key-9c1d"
TOKEN = "sentinel-malformed-token-2b8e"

HANDCRAFTED_CONNECTOR = "acme-internal"


def _registry_body() -> dict:
    return {
        "connector_id": "postgres",
        "method": "host-port",
        "name": "",
        "conversation_id": None,
        "values": {
            "host": "db.example.com",
            "port": "5432",
            "database": "app",
            "username": "reader",
            "password": PASSWORD,
        },
        "skipped": [],
        "form_spec": None,
    }


def _handcrafted_body() -> dict:
    return {
        "connector_id": HANDCRAFTED_CONNECTOR,
        "name": "acme",
        "conversation_id": None,
        "values": {"api_key": API_KEY, "region": "eu"},
        "skipped": [],
        "form_spec": {
            "form_id": f"{HANDCRAFTED_CONNECTOR}-connector",
            "title": "Acme",
            "fields": [
                {"name": "api_key", "label": "API key", "type": "password", "required": True, "secret": True},
                {"name": "region", "label": "Region", "type": "text", "required": False},
            ],
        },
    }


@pytest.fixture(autouse=True)
def _reset_app_settings():
    get_app_settings.cache_clear()
    yield
    get_app_settings.cache_clear()


@pytest.fixture()
def org_client(monkeypatch):
    monkeypatch.setenv("COWORK_TENANCY_MODE", "org")
    monkeypatch.setenv("COWORK_IDENTITY_ENFORCE", "enforce")
    get_app_settings.cache_clear()
    return TestClient(create_app())


@pytest.fixture()
def forbid_local_paths(monkeypatch):
    """Every local credential path raises if reached, and the process-global
    staging store ends the test with exactly the keys it started with."""

    def refuse(name: str):
        def _raise(*_args, **_kwargs):
            raise AssertionError(f"{name} must not be reached by an org-mode submission")

        return _raise

    staged_before = set(store._store)
    monkeypatch.setattr(store, "stage", refuse("store.stage"))
    monkeypatch.setattr(submissions_endpoints, "ProbeHandler", refuse("ProbeHandler"))
    monkeypatch.setattr(probe_handlers, "CredentialProbe", refuse("CredentialProbe"))
    monkeypatch.setattr(probe_handlers, "vault_for_scope", refuse("vault_for_scope"))
    monkeypatch.setattr(persist, "vault_for_scope", refuse("vault_for_scope"))
    monkeypatch.setattr(data_vault.LocalDataVault, "__init__", refuse("LocalDataVault"))
    monkeypatch.setattr(ConnectionsService, "__init__", refuse("ConnectionsService"))
    yield
    assert set(store._store) == staged_before


def test_registry_submission_is_refused_before_staging(org_client, forbid_local_paths, caplog):
    caplog.set_level(logging.DEBUG)

    res = org_client.post(SUBMISSIONS, json=_registry_body(), headers=MEMBER_HEADERS)

    assert res.status_code == 403
    assert res.json()["detail"] == REFUSAL
    assert PASSWORD not in res.text
    assert PASSWORD not in caplog.text


def test_handcrafted_submission_is_refused_before_the_local_vault(org_client, forbid_local_paths, caplog):
    # The handcrafted branch only exists for connectors the registry does not
    # know; if this id ever lands in the registry the test stops testing it.
    assert registry.get_connector(HANDCRAFTED_CONNECTOR) is None
    caplog.set_level(logging.DEBUG)

    res = org_client.post(SUBMISSIONS, json=_handcrafted_body(), headers=MEMBER_HEADERS)

    assert res.status_code == 403
    assert res.json()["detail"] == REFUSAL
    assert API_KEY not in res.text
    assert API_KEY not in caplog.text


def test_malformed_submission_is_refused_not_echoed(org_client, forbid_local_paths, caplog):
    """A body of the wrong shape would answer 422 with the offending input
    echoed in the response, and the open database session logs that same
    validation error at ERROR with the input in it. The refusal has to come
    before body validation, so the token appears in neither place."""
    caplog.set_level(logging.DEBUG)
    body = {**_registry_body(), "values": TOKEN}

    res = org_client.post(SUBMISSIONS, json=body, headers=MEMBER_HEADERS)

    assert res.status_code == 403
    assert TOKEN not in res.text
    assert TOKEN not in caplog.text


def test_incomplete_submission_is_refused_before_field_validation(org_client, forbid_local_paths):
    """Missing a required field used to answer 400 from the handler's own
    check. In org mode the handler never runs, so neither does that check."""
    body = _registry_body()
    del body["values"]["password"]

    res = org_client.post(SUBMISSIONS, json=body, headers=MEMBER_HEADERS)

    assert res.status_code == 403
    assert "Missing required fields" not in res.text


@pytest.fixture()
def local_client(monkeypatch):
    monkeypatch.setenv("COWORK_TENANCY_MODE", "local")
    get_app_settings.cache_clear()
    return TestClient(create_app())


@pytest.fixture()
def leave_store_as_found():
    """The handler stages into the process-global store and never consumes;
    drop whatever this test staged so later tests see the store they expect."""
    staged_before = set(store._store)
    yield
    for submission_id in set(store._store) - staged_before:
        store.consume(submission_id)


def test_desktop_registry_submission_still_stages_and_probes(local_client, leave_store_as_found, monkeypatch):
    seen: dict = {}

    class FakeProbeHandler:
        def __init__(self, scoped):
            pass

        async def run(self, submission_id, connector_id, method, name, conversation_id):
            seen.update(submission_id=submission_id, connector_id=connector_id, method=method)
            yield "event: response.created\ndata: {}\n\n"

    monkeypatch.setattr(submissions_endpoints, "ProbeHandler", FakeProbeHandler)

    res = local_client.post(SUBMISSIONS, json=_registry_body())

    assert res.status_code == 200
    assert res.headers["content-type"].startswith("text/event-stream")
    assert res.headers["cache-control"] == "no-store"
    assert seen["connector_id"] == "postgres"
    assert seen["method"] == "host-port"
    staged = store.get(seen["submission_id"])
    assert staged is not None
    assert staged["values"]["password"] == PASSWORD


def test_desktop_handcrafted_submission_still_saves_to_the_local_vault(
    local_client, leave_store_as_found, monkeypatch, tmp_path
):
    """No probe for a connector the registry does not know: the real handler
    saves straight into the local vault, and the record it wrote is the
    desktop behavior to keep."""
    assert registry.get_connector(HANDCRAFTED_CONNECTOR) is None
    vault_dir = tmp_path / "vault"
    monkeypatch.setattr(persist, "ConnectorSettings", lambda: SimpleNamespace(vault_dir=str(vault_dir)))

    res = local_client.post(SUBMISSIONS, json=_handcrafted_body())

    assert res.status_code == 200
    assert "Saved as" in res.text
    record = data_vault.LocalDataVault(vault_dir).read_record(HANDCRAFTED_CONNECTOR, "acme")
    assert record is not None
    assert record["fields"]["api_key"] == API_KEY
    assert "api_key" in record["secure_keys"]
