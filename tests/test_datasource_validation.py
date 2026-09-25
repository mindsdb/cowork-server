"""Capture drives one validation attempt, and answers with what auth recorded.

A connection auth stores is pending until a probe says otherwise, and the
interface excludes every connection that is not verified, so a capture that
skipped validation would hand the user something they cannot use. These tests
follow both hops: auth mints a single-use capability to the producer identity
when the owner's own credential comes with it, and the gateway spends it.

The failure direction matters as much: the credential is already stored by the
time validation runs, so nothing here may turn a stored connection into a
failed request.
"""

from __future__ import annotations

import json
import logging

import httpx
import pytest
from fastapi.testclient import TestClient

from cowork.server import create_app

CREATE = "/api/v1/connectors/datasources/"
BEARER = "Bearer sentinel-caller-credential"
PASSWORD = "s3ntinel-p4ssw0rd-do-not-echo"
CAPABILITY = "sentinel-probe-capability-never-echoed"
PRODUCER_KEY = "sentinel-producer-key"
GATEWAY = "http://mindshub-inference"
AUTH_INTERNAL = "http://auth"
ENABLED = '{"manifest_version": 1, "enabled": ["postgres:host-port"]}'

MEMBER_HEADERS = {
    "X-User-Id": "11111111-1111-4111-8111-111111111111",
    "X-Organization-Id": "22222222-2222-4222-8222-222222222222",
}
AUTH_HEADERS = {**MEMBER_HEADERS, "Authorization": BEARER}

PENDING = {
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
VERIFIED = {**PENDING, "status": "verified"}
FAILED = {**PENDING, "status": "failed", "validation_error": "The database refused the connection."}
MINT = {
    "id": "33333333-3333-4333-8333-333333333333",
    "capability": CAPABILITY,
    "connection_id": 7,
    "credential_version": 1,
    "expires_at": "2026-09-17T10:15:00Z",
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


@pytest.fixture()
def cluster(monkeypatch):
    """Record every request and script each hop by its path.

    `relay` is auth's user-facing datasource API, `mint` its cluster-only
    capability route, `probe` the gateway's. Both modules share one httpx
    module, so one transport covers all three.
    """
    from cowork.services.connectors import datasource_validation
    from cowork.services.connectors.oauth import auth_proxy

    recorded: list[httpx.Request] = []
    scripted: dict = {
        "relay": (201, PENDING),
        "detail": (200, VERIFIED),
        "mint": (201, MINT),
        "probe": (200, {"protocol_version": 1, "operation": "describe", "connection_id": 7,
                        "credential_version": 1, "columns": []}),
        "error": None,
    }

    def handle(request: httpx.Request) -> httpx.Response:
        recorded.append(request)
        path = request.url.path
        if path.endswith("/probes/"):
            key = "mint"
        elif path.endswith("/datasources/probe"):
            key = "probe"
        elif request.method == "GET":
            key = "detail"
        else:
            key = "relay"
        if scripted["error"] and key in ("mint", "probe"):
            raise scripted["error"]
        status, body = scripted[key]
        return httpx.Response(status, json=body)

    real_client = httpx.AsyncClient
    client = lambda **kw: real_client(transport=httpx.MockTransport(handle), **kw)  # noqa: E731
    monkeypatch.setattr(auth_proxy.httpx, "AsyncClient", client)
    monkeypatch.setattr(datasource_validation.httpx, "AsyncClient", client)
    return recorded, scripted


def _client(monkeypatch, *, producer: bool = True, gateway: bool = True) -> TestClient:
    from cowork.common.settings.app_settings import get_app_settings

    monkeypatch.setenv("COWORK_TENANCY_MODE", "org")
    monkeypatch.setenv("AUTH_SERVICE_BASE_URL", "https://auth.example.com")
    monkeypatch.setenv("COWORK_DATASOURCE_CAPABILITIES", ENABLED)
    monkeypatch.setenv("COWORK_TURN_AUTH_INTERNAL_BASE_URL", AUTH_INTERNAL)
    if producer:
        monkeypatch.setenv("COWORK_TURN_DATASOURCE_PRODUCER_KEY_ID", "producer_v1")
        monkeypatch.setenv("COWORK_TURN_DATASOURCE_PRODUCER_KEY", PRODUCER_KEY)
    if gateway:
        monkeypatch.setenv("COWORK_TURN_DATASOURCE_GATEWAY_BASE_URL", GATEWAY)
    get_app_settings.cache_clear()
    return TestClient(create_app())


@pytest.fixture()
def org_client(monkeypatch, adapter_verified_datasources) -> TestClient:
    return _client(monkeypatch)


def _by_path(recorded: list[httpx.Request], fragment: str) -> httpx.Request | None:
    return next((r for r in recorded if fragment in str(r.url)), None)


def test_a_captured_connection_is_probed_and_answers_with_its_terminal_state(org_client, cluster):
    recorded, _ = cluster

    res = org_client.post(CREATE, json=CREATE_BODY, headers=AUTH_HEADERS)

    assert res.status_code == 201
    assert res.json()["status"] == "verified", "the caller must not be told pending after a passing probe"

    mint = _by_path(recorded, "/internal/datasources/connections/7/probes/")
    assert mint is not None, "auth was never asked for a capability"
    assert str(mint.url) == f"{AUTH_INTERNAL}/internal/datasources/connections/7/probes/"
    # Both factors: the producer role key and the owner's own credential.
    assert mint.headers["x-datasource-service-key-id"] == "producer_v1"
    assert mint.headers["x-datasource-service-key"] == PRODUCER_KEY
    assert mint.headers["authorization"] == BEARER

    probe = _by_path(recorded, "/v1/datasources/probe")
    assert probe is not None, "the gateway was never asked to dial"
    assert str(probe.url) == f"{GATEWAY}/v1/datasources/probe"
    assert probe.headers["authorization"] == f"Bearer {CAPABILITY}"
    body = json.loads(probe.content)
    assert body["protocol_version"] == 1
    assert body["probe_id"] == MINT["id"]
    assert body["credential_version"] == 1
    assert body["correlation_id"]
    # The gateway takes a connection reference, never the connection itself.
    assert "password" not in body and "host" not in body


def test_a_refused_probe_leaves_the_gateways_own_reason_in_the_log(org_client, cluster, caplog):
    """The connection only ever says that validation failed, never why."""
    _, scripted = cluster
    scripted["probe"] = (
        502,
        {"code": "tls_verification_failed", "detail": "The database's certificate was not trusted.",
         "request_id": "req-sentinel-1234"},
    )
    scripted["detail"] = (200, FAILED)

    with caplog.at_level(logging.INFO):
        res = org_client.post(CREATE, json=CREATE_BODY, headers=AUTH_HEADERS)

    assert res.status_code == 201
    assert "tls_verification_failed" in caplog.text
    assert "req-sentinel-1234" in caplog.text
    assert PASSWORD not in caplog.text
    assert CAPABILITY not in caplog.text


def test_a_body_that_is_not_the_gateways_shape_is_named_not_echoed(org_client, cluster, caplog):
    _, scripted = cluster
    scripted["probe"] = (502, {"unexpected": PASSWORD})
    scripted["detail"] = (200, FAILED)

    with caplog.at_level(logging.INFO):
        res = org_client.post(CREATE, json=CREATE_BODY, headers=AUTH_HEADERS)

    assert res.status_code == 201
    assert "uncoded" in caplog.text
    assert PASSWORD not in caplog.text


def test_the_gateways_reason_reaches_the_caller_on_a_failed_capture(org_client, cluster):
    """Auth records a verdict, not a cause, so this is the only reason a client
    could act on: it is what lets the form offer a weaker trust choice."""
    _, scripted = cluster
    scripted["probe"] = (502, {"code": "tls_failed", "detail": "The certificate could not be verified.",
                               "request_id": "req-1"})
    scripted["detail"] = (200, FAILED)

    res = org_client.post(CREATE, json=CREATE_BODY, headers=AUTH_HEADERS)

    assert res.status_code == 201
    body = res.json()
    assert body["status"] == "failed"
    assert body["validation_code"] == "tls_failed"


def test_a_passing_capture_carries_no_reason(org_client, cluster):
    res = org_client.post(CREATE, json=CREATE_BODY, headers=AUTH_HEADERS)

    assert res.json()["status"] == "verified"
    assert res.json()["validation_code"] is None


def test_a_failing_probe_is_reported_as_the_connection_auth_recorded(org_client, cluster):
    recorded, scripted = cluster
    scripted["probe"] = (502, {"code": "connect_failed", "detail": "The database refused the connection."})
    scripted["detail"] = (200, FAILED)

    res = org_client.post(CREATE, json=CREATE_BODY, headers=AUTH_HEADERS)

    assert res.status_code == 201, "a failed database is a failed connection, not a failed request"
    assert res.json()["status"] == "failed"
    assert res.json()["validation_error"] == "The database refused the connection."
    assert _by_path(recorded, "/v1/datasources/probe") is not None


def test_neither_the_capability_nor_the_password_reaches_the_caller_or_the_log(org_client, cluster, caplog):
    with caplog.at_level(logging.DEBUG):
        res = org_client.post(CREATE, json=CREATE_BODY, headers=AUTH_HEADERS)

    assert res.status_code == 201
    assert CAPABILITY not in res.text
    assert CAPABILITY not in caplog.text
    assert PRODUCER_KEY not in res.text
    assert PRODUCER_KEY not in caplog.text
    assert PASSWORD not in res.text
    assert PASSWORD not in caplog.text


def test_an_unreachable_gateway_leaves_the_stored_connection_pending(org_client, cluster, caplog):
    recorded, scripted = cluster
    scripted["error"] = httpx.ConnectError("the gateway is down")

    with caplog.at_level(logging.WARNING):
        res = org_client.post(CREATE, json=CREATE_BODY, headers=AUTH_HEADERS)

    assert res.status_code == 201, "auth already stored the credential; the request stands"
    assert res.json()["status"] == "pending"
    assert "not validated" in caplog.text
    assert CAPABILITY not in caplog.text


def test_a_refused_capability_leaves_the_stored_connection_pending(org_client, cluster):
    recorded, scripted = cluster
    scripted["mint"] = (403, {"code": "identity_denied", "detail": "refused"})

    res = org_client.post(CREATE, json=CREATE_BODY, headers=AUTH_HEADERS)

    assert res.status_code == 201
    assert res.json()["status"] == "pending"
    assert _by_path(recorded, "/v1/datasources/probe") is None, "a refused mint must not be spent"


@pytest.mark.parametrize("missing", ["producer", "gateway"])
def test_a_deployment_that_cannot_validate_captures_without_trying(monkeypatch, cluster, adapter_verified_datasources, missing):
    recorded, _ = cluster
    client = _client(monkeypatch, producer=missing != "producer", gateway=missing != "gateway")

    res = client.post(CREATE, json=CREATE_BODY, headers=AUTH_HEADERS)

    assert res.status_code == 201
    assert res.json()["status"] == "pending"
    assert _by_path(recorded, "/probes/") is None or missing == "gateway"
    assert _by_path(recorded, "/v1/datasources/probe") is None


def test_a_retry_runs_a_fresh_validation_attempt(org_client, cluster):
    recorded, scripted = cluster
    scripted["relay"] = (200, PENDING)

    res = org_client.post("/api/v1/connectors/datasources/7/validation-retry", headers=AUTH_HEADERS)

    assert res.status_code == 200
    assert res.json()["status"] == "verified"
    assert _by_path(recorded, "/internal/datasources/connections/7/probes/") is not None
    assert _by_path(recorded, "/v1/datasources/probe") is not None


def test_a_rename_that_leaves_the_connection_verified_is_not_probed_again(org_client, cluster):
    recorded, scripted = cluster
    scripted["relay"] = (200, {**VERIFIED, "name": "renamed", "revision": 7})

    res = org_client.patch(
        "/api/v1/connectors/datasources/7",
        json={**CREATE_BODY, "name": "renamed", "expected_revision": 6},
        headers=AUTH_HEADERS,
    )

    assert res.status_code == 200
    assert (res.json()["status"], res.json()["name"], res.json()["revision"]) == ("verified", "renamed", 7)
    assert _by_path(recorded, "/internal/datasources/connections/7/probes/") is None
    assert _by_path(recorded, "/v1/datasources/probe") is None


def test_an_edit_revalidates_the_new_credential(org_client, cluster):
    recorded, scripted = cluster
    scripted["relay"] = (200, {**PENDING, "credential_version": 2, "revision": 6})
    scripted["mint"] = (201, {**MINT, "credential_version": 2})

    res = org_client.patch(
        "/api/v1/connectors/datasources/7",
        json={**CREATE_BODY, "expected_revision": 5},
        headers=AUTH_HEADERS,
    )

    assert res.status_code == 200
    assert res.json()["status"] == "verified"
    # The edit is guarded by the revision, the probe by the credential version.
    probe = _by_path(recorded, "/v1/datasources/probe")
    assert json.loads(probe.content)["credential_version"] == 2, "the probe must name the version it is checking"
