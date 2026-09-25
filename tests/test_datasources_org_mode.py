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
    "revision": 6,
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


ENABLED = '{"manifest_version": 1, "enabled": ["postgres:host-port"]}'


def _org_client(monkeypatch, capabilities: str) -> TestClient:
    monkeypatch.setenv("COWORK_TENANCY_MODE", "org")
    monkeypatch.setenv("AUTH_SERVICE_BASE_URL", "https://auth.example.com")
    monkeypatch.setenv("COWORK_DATASOURCE_CAPABILITIES", capabilities)
    from cowork.common.settings.app_settings import get_app_settings

    get_app_settings.cache_clear()
    return TestClient(create_app())


@pytest.fixture()
def org_client(monkeypatch, adapter_verified_datasources) -> TestClient:
    """A deployment that runs PostgreSQL connections."""
    return _org_client(monkeypatch, ENABLED)


@pytest.fixture()
def default_org_client(monkeypatch, adapter_verified_datasources) -> TestClient:
    """A deployment before anyone enabled a datasource method."""
    return _org_client(monkeypatch, "")


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
    assert body["tls"] == {"mode": "prefer", "ca_pem": None}


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
        json={**CREATE_BODY, "expected_revision": 6},
        headers=AUTH_HEADERS,
    )
    assert edit.status_code == 200
    sent = json.loads(recorded[1].content)
    assert sent["expected_revision"] == 6
    assert "expected_version" not in sent
    assert (edit.json()["revision"], edit.json()["credential_version"]) == (6, 1)

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
        json={**CREATE_BODY, "expected_revision": 3},
        headers=AUTH_HEADERS,
    )
    assert coded.status_code == 409
    assert coded.json()["detail"]["code"] == "stale_version"

    # A duplicate-name conflict is also 409 but carries no code.
    scripted["body"] = {"detail": "A connection with this name already exists."}
    codeless = org_client.patch(
        "/api/v1/connectors/datasources/7",
        json={**CREATE_BODY, "expected_revision": 3},
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
        ("patch", "/api/v1/connectors/datasources/7", {**CREATE_BODY, "expected_revision": 1}),
        ("delete", "/api/v1/connectors/datasources/7", None),
        ("post", "/api/v1/connectors/datasources/7/validation-retry", None),
    ],
)
def test_every_route_is_absent_in_local_mode(local_client, relay, method, path, body):
    recorded, _ = relay

    res = local_client.request(method.upper(), path, json=body)

    assert res.status_code == 404
    assert recorded == [], "a desktop caller must not reach auth"


@pytest.mark.parametrize(
    "method,path,body",
    [
        ("post", "/api/v1/connectors/datasources/", {}),
        ("post", "/api/v1/connectors/datasources/", {"nope": 1}),
        ("patch", "/api/v1/connectors/datasources/7", {}),
    ],
)
def test_a_malformed_body_is_still_absent_in_local_mode(local_client, relay, method, path, body):
    """The org refusal has to beat body validation, not follow it.

    FastAPI validates the body before the handler runs, so an org check made
    inside the handler answers 422 here instead and names the schema's fields
    to a desktop caller the surface is supposed to be absent for.
    """
    recorded, _ = relay

    res = local_client.request(method.upper(), path, json=body)

    assert res.status_code == 404
    assert "connector_id" not in res.text, "a 422 here would advertise the schema"
    assert recorded == [], "a desktop caller must not reach auth"


@pytest.mark.parametrize(
    "method,path,body",
    [
        ("post", "/api/v1/connectors/datasources/", CREATE_BODY),
        ("get", "/api/v1/connectors/datasources/", None),
        ("get", "/api/v1/connectors/datasources/7", None),
        ("patch", "/api/v1/connectors/datasources/7", {**CREATE_BODY, "expected_revision": 1}),
        ("delete", "/api/v1/connectors/datasources/7", None),
        ("post", "/api/v1/connectors/datasources/7/validation-retry", None),
    ],
)
def test_every_route_relays_the_callers_credential_and_no_substitute_identity(
    org_client, relay, method, path, body
):
    """Owner and org are auth's to derive from the bearer.

    Pinned on every route, not just create: a per-route regression that added
    a user_id parameter or dropped the caller's header would otherwise pass.
    """
    recorded, scripted = relay
    scripted["status"] = 201 if method == "post" and path.endswith("datasources/") else 200
    if method == "delete":
        scripted["status"] = 204
    if path.endswith("datasources/") and method == "get":
        scripted["body"] = {"items": [CONNECTION]}

    res = org_client.request(method.upper(), path, json=body, headers=AUTH_HEADERS)

    assert res.status_code == scripted["status"], "the route must relay cleanly, not 500 on its own response"
    assert len(recorded) == 1
    sent = recorded[0]
    assert sent.headers["authorization"] == BEARER
    # The gateway's identity headers are cowork-server's, not auth's. Auth
    # derives owner and org from the bearer, so forwarding these would hand it
    # a second, unverified opinion about who is calling.
    assert "x-user-id" not in sent.headers
    assert "x-organization-id" not in sent.headers
    for identity in ("user_id", "organization_id", "org_id"):
        assert identity not in str(sent.url)
        assert identity not in sent.content.decode()


def test_capture_is_refused_when_the_deployment_has_not_enabled_the_method(default_org_client, relay):
    """The capability response says unavailable, so the capture route must agree."""
    recorded, _ = relay

    res = default_org_client.post("/api/v1/connectors/datasources/", json=CREATE_BODY, headers=AUTH_HEADERS)

    assert res.status_code == 409
    assert res.json()["detail"]["code"] == "unsupported_capability"
    assert PASSWORD not in res.text
    assert recorded == [], "a method this deployment does not run must not reach auth"


def test_an_edit_is_refused_on_a_method_the_deployment_does_not_run(default_org_client, relay):
    recorded, _ = relay

    res = default_org_client.patch(
        "/api/v1/connectors/datasources/7",
        json={**CREATE_BODY, "expected_revision": 1},
        headers=AUTH_HEADERS,
    )

    assert res.status_code == 409
    assert res.json()["detail"]["code"] == "unsupported_capability"
    assert recorded == []


@pytest.mark.parametrize(
    "method,path",
    [
        ("get", "/api/v1/connectors/datasources/"),
        ("get", "/api/v1/connectors/datasources/7"),
        ("delete", "/api/v1/connectors/datasources/7"),
    ],
)
def test_switching_a_method_off_never_traps_a_connection_already_captured(
    default_org_client, relay, method, path
):
    """Reading and deleting stay open, or a disabled method would strand its rows."""
    recorded, scripted = relay
    scripted["status"] = 204 if method == "delete" else 200
    if path.endswith("datasources/"):
        scripted["body"] = {"items": [CONNECTION]}

    res = default_org_client.request(method.upper(), path, headers=AUTH_HEADERS)

    assert res.status_code == scripted["status"]
    assert len(recorded) == 1
