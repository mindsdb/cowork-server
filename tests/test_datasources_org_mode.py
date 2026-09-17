"""The datasource management routes are relays and nothing else.

Every case runs with the local credential paths booby-trapped: if a handler
stages a submission, builds a probe or opens a vault, the spy raises and the
test fails. That has to hold on the error paths too, which is where a
fallback would hide.
"""
from __future__ import annotations

import json

import httpx
import pytest
from fastapi.testclient import TestClient

from cowork.server import create_app

BEARER = "Bearer sentinel-caller-credential"
PASSWORD = "s3ntinel-p4ssw0rd-do-not-echo"

MEMBER_HEADERS = {
    "X-User-Id": "11111111-1111-4111-8111-111111111111",
    "X-Organization-Id": "22222222-2222-4222-8222-222222222222",
}
AUTH_HEADERS = {**MEMBER_HEADERS, "Authorization": BEARER}

#: Shaped like auth's DatasourceConnectionResponseSerializer.
CONNECTION = {
    "id": 7,
    "connector_id": "postgres",
    "method": "host-port",
    "name": "prod reporting",
    "status": "pending",
    "credential_version": 1,
    "host_masked": "db***om",
    "port": 5432,
    "database": "appdb",
    "username": "dbuser",
    "tls_mode": "system",
    "validation_error": None,
    "created_at": "2026-09-17T10:00:00Z",
    "updated_at": "2026-09-17T10:00:00Z",
}

CREATE_BODY = {
    "connector_id": "postgres",
    "method": "host-port",
    "name": "prod reporting",
    "host": "db.example.com",
    "port": 5432,
    "database": "appdb",
    "username": "dbuser",
    "password": PASSWORD,
}


@pytest.fixture(autouse=True)
def _reset_app_settings():
    from cowork.common.settings.app_settings import get_app_settings

    get_app_settings.cache_clear()
    yield
    get_app_settings.cache_clear()


@pytest.fixture(autouse=True)
def forbid_local_paths(monkeypatch):
    """Booby-trap every local credential path these routes must never reach."""
    from anton.core.datasources import data_vault
    from cowork.handlers import probe as probe_module
    from cowork.services.connectors import connections as connections_module
    from cowork.services.connectors import persist as persist_module
    from cowork.services.connectors.submissions import store

    def forbidden(name):
        def _raise(*args, **kwargs):
            raise AssertionError(f"a cloud datasource route reached the local path {name}")

        return _raise

    monkeypatch.setattr(store, "stage", forbidden("store.stage"))
    monkeypatch.setattr(persist_module, "vault_for_scope", forbidden("persist.vault_for_scope"))
    monkeypatch.setattr(probe_module, "vault_for_scope", forbidden("probe.vault_for_scope"))
    monkeypatch.setattr(probe_module, "CredentialProbe", forbidden("CredentialProbe"))
    monkeypatch.setattr(data_vault.LocalDataVault, "__init__", forbidden("LocalDataVault"))
    monkeypatch.setattr(connections_module.ConnectionsService, "__init__", forbidden("ConnectionsService"))

    before = set(store._store)
    yield
    assert set(store._store) == before, "a cloud datasource route left something in the staging store"


@pytest.fixture()
def relay(monkeypatch):
    """Record what is relayed and script auth's answer."""
    from cowork.services.connectors.oauth import auth_proxy

    recorded: list[httpx.Request] = []
    scripted: dict = {"status": 200, "body": CONNECTION, "error": None}

    def handle(request: httpx.Request) -> httpx.Response:
        recorded.append(request)
        if scripted["error"]:
            raise scripted["error"]
        if scripted["status"] == 204:
            return httpx.Response(204)
        return httpx.Response(scripted["status"], json=scripted["body"])

    real_client = httpx.AsyncClient
    monkeypatch.setattr(
        auth_proxy.httpx,
        "AsyncClient",
        lambda **kw: real_client(transport=httpx.MockTransport(handle), **kw),
    )
    return recorded, scripted


@pytest.fixture()
def org_client(monkeypatch) -> TestClient:
    monkeypatch.setenv("COWORK_TENANCY_MODE", "org")
    monkeypatch.setenv("AUTH_SERVICE_BASE_URL", "https://auth.example.com")
    from cowork.common.settings.app_settings import get_app_settings

    get_app_settings.cache_clear()
    return TestClient(create_app())


@pytest.fixture()
def local_client(monkeypatch) -> TestClient:
    monkeypatch.setenv("COWORK_TENANCY_MODE", "local")
    from cowork.common.settings.app_settings import get_app_settings

    get_app_settings.cache_clear()
    return TestClient(create_app())


def test_create_forwards_the_callers_bearer_and_the_canonical_payload(org_client, relay):
    recorded, scripted = relay
    scripted["status"] = 201

    res = org_client.post("/api/v1/connectors/datasources/", json=CREATE_BODY, headers=AUTH_HEADERS)

    assert res.status_code == 201
    assert res.json()["id"] == 7
    assert len(recorded) == 1
    sent = recorded[0]
    assert str(sent.url) == "https://auth.example.com/v1/datasources"
    assert sent.headers["authorization"] == BEARER
    body = json.loads(sent.content)
    # Owner and org come from the bearer. Nothing identity-shaped may ride along.
    assert "user_id" not in body and "organization_id" not in body
    assert "user_id" not in str(sent.url) and "organization_id" not in str(sent.url)
    assert body["password"] == PASSWORD
    assert body["tls"] == {"mode": "system", "ca_pem": None}


def test_a_dsn_create_relays_structured_fields_and_never_the_dsn(org_client, relay):
    recorded, scripted = relay
    scripted["status"] = 201
    dsn = f"postgres://dbuser:{PASSWORD}@db.example.com:5432/appdb?sslmode=verify-full"

    res = org_client.post(
        "/api/v1/connectors/datasources/",
        json={
            "connector_id": "postgres",
            "method": "host-port",
            "name": "prod reporting",
            "input_mode": "dsn",
            "dsn": dsn,
        },
        headers=AUTH_HEADERS,
    )

    assert res.status_code == 201
    body = json.loads(recorded[0].content)
    assert body["host"] == "db.example.com"
    assert body["database"] == "appdb"
    assert "dsn" not in body
    assert dsn not in recorded[0].content.decode()


def test_an_unknown_driver_option_is_refused_before_anything_is_relayed(org_client, relay):
    recorded, _ = relay

    res = org_client.post(
        "/api/v1/connectors/datasources/",
        json={
            "connector_id": "postgres",
            "method": "host-port",
            "name": "prod reporting",
            "input_mode": "dsn",
            "dsn": f"postgres://dbuser:{PASSWORD}@db.example.com:5432/appdb?options=-c%20x%3D1",
        },
        headers=AUTH_HEADERS,
    )

    assert res.status_code == 400
    assert res.json()["detail"]["code"] == "invalid_connection"
    assert PASSWORD not in res.text
    assert recorded == [], "a rejected connection must not reach auth"


def test_list_unwraps_auths_items_envelope(org_client, relay):
    recorded, scripted = relay
    scripted["body"] = {"items": [CONNECTION]}

    res = org_client.get("/api/v1/connectors/datasources/", headers=AUTH_HEADERS)

    assert res.status_code == 200
    assert isinstance(res.json(), list)
    assert res.json()[0]["id"] == 7
    assert str(recorded[0].url) == "https://auth.example.com/v1/datasources"


def test_detail_edit_delete_and_retry_hit_the_exact_auth_paths(org_client, relay):
    recorded, scripted = relay

    assert org_client.get("/api/v1/connectors/datasources/7", headers=AUTH_HEADERS).status_code == 200

    edit = org_client.patch(
        "/api/v1/connectors/datasources/7",
        json={**CREATE_BODY, "expected_version": 3},
        headers=AUTH_HEADERS,
    )
    assert edit.status_code == 200
    assert json.loads(recorded[1].content)["expected_version"] == 3

    scripted["status"] = 204
    assert org_client.delete("/api/v1/connectors/datasources/7", headers=AUTH_HEADERS).status_code == 204

    scripted["status"] = 200
    retry = org_client.post("/api/v1/connectors/datasources/7/validation-retry", headers=AUTH_HEADERS)
    assert retry.status_code == 200

    assert [f"{r.method} {r.url.path}" for r in recorded] == [
        "GET /v1/datasources/7",
        "PATCH /v1/datasources/7",
        "DELETE /v1/datasources/7",
        "POST /v1/datasources/7/validation-retry",
    ]


@pytest.mark.parametrize("status_code", [400, 401, 404, 409, 429, 503])
def test_auth_failures_relay_without_a_local_fallback(org_client, relay, status_code):
    recorded, scripted = relay
    scripted["status"] = status_code
    scripted["body"] = {"detail": "auth said no"}

    res = org_client.get("/api/v1/connectors/datasources/7", headers=AUTH_HEADERS)

    assert res.status_code == status_code
    assert len(recorded) == 1


def test_a_coded_conflict_keeps_its_code_and_a_codeless_one_does_not_invent_one(org_client, relay):
    _, scripted = relay
    scripted["status"] = 409
    scripted["body"] = {"detail": "Connection was modified.", "code": "stale_version"}

    coded = org_client.patch(
        "/api/v1/connectors/datasources/7",
        json={**CREATE_BODY, "expected_version": 3},
        headers=AUTH_HEADERS,
    )
    assert coded.status_code == 409
    assert coded.json()["detail"]["code"] == "stale_version"

    # A duplicate-name conflict is also 409 but carries no code.
    scripted["body"] = {"detail": "A connection with this name already exists."}
    codeless = org_client.patch(
        "/api/v1/connectors/datasources/7",
        json={**CREATE_BODY, "expected_version": 3},
        headers=AUTH_HEADERS,
    )
    assert codeless.status_code == 409
    assert codeless.json()["detail"] == "A connection with this name already exists."


def test_the_disabled_feature_503_reaches_the_caller_intact(org_client, relay):
    _, scripted = relay
    scripted["status"] = 503
    scripted["body"] = {"detail": "Datasource connections are not enabled."}

    res = org_client.post("/api/v1/connectors/datasources/", json=CREATE_BODY, headers=AUTH_HEADERS)

    assert res.status_code == 503
    assert res.json()["detail"] == "Datasource connections are not enabled."


def test_an_unreachable_auth_answers_502_and_touches_nothing_local(org_client, relay):
    _, scripted = relay
    scripted["error"] = httpx.ConnectError("auth is down")

    res = org_client.post("/api/v1/connectors/datasources/", json=CREATE_BODY, headers=AUTH_HEADERS)

    assert res.status_code == 502
    assert PASSWORD not in res.text


def test_a_request_without_a_bearer_forwards_no_authorization_header(org_client, relay):
    recorded, scripted = relay
    scripted["status"] = 401
    scripted["body"] = {"detail": "Authentication credentials were not provided."}

    res = org_client.get("/api/v1/connectors/datasources/", headers=MEMBER_HEADERS)

    assert res.status_code == 401
    assert "authorization" not in recorded[0].headers


def test_an_org_request_without_identity_headers_never_reaches_auth(org_client, relay):
    recorded, _ = relay

    res = org_client.get("/api/v1/connectors/datasources/")

    assert res.status_code == 401
    assert recorded == []


@pytest.mark.parametrize(
    "method,path,body",
    [
        ("post", "/api/v1/connectors/datasources/", CREATE_BODY),
        ("get", "/api/v1/connectors/datasources/", None),
        ("get", "/api/v1/connectors/datasources/7", None),
        ("patch", "/api/v1/connectors/datasources/7", {**CREATE_BODY, "expected_version": 1}),
        ("delete", "/api/v1/connectors/datasources/7", None),
        ("post", "/api/v1/connectors/datasources/7/validation-retry", None),
    ],
)
def test_every_route_is_absent_in_local_mode(local_client, relay, method, path, body):
    recorded, _ = relay

    res = local_client.request(method.upper(), path, json=body)

    assert res.status_code == 404
    assert recorded == [], "a desktop caller must not reach auth"
