"""Connecting a member's own custom connector: address, values, the two-send check, auth and the attempt limit."""

from __future__ import annotations

import asyncio
import json
import socket

import fakeredis
import httpx
import pytest
from anton.core.tools.connector_draft import parse_connector_draft
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy.pool import StaticPool
from sqlmodel import Session, SQLModel, create_engine

from cowork.api.v1.endpoints.connectors import custom as custom_endpoints
from cowork.db.scoped import LOCAL_SCOPE, ScopedSession, TenantScope, get_scoped_session, get_tenant_scope
from cowork.models.custom_connector import CustomConnector
from cowork.principal import Principal, get_principal
from cowork.services.connectors import custom_connect, egress
from cowork.services.connectors.custom_connectors import CustomConnectorService
from tests._fakes import inline_run_db

BUILDER = TenantScope(org_mode=True, org_id="org-a", user_id="builder-a")
SAME_ORG = TenantScope(org_mode=True, org_id="org-a", user_id="member-a2")
OTHER_ORG = TenantScope(org_mode=True, org_id="org-b", user_id="builder-a")
BEARER = "Bearer sentinel-member-credential"
SECRET = "sk_live_planted_value_never_echoed"

DRAFT = {
    "connector_id": "acme_crm",
    "label": "Acme CRM",
    "fields": [
        {"name": "api_key", "label": "API key", "secret": True, "required": True},
        {"name": "workspace", "label": "Workspace", "secret": False, "required": False},
    ],
    "check": {"path": "/v1/me", "headers": {"Authorization": "Bearer {api_key}", "Accept": "application/json"}},
}
STORED = {
    "id": 7, "connector_id": "acme_crm", "name": "Acme", "secure_keys": ["api_key"], "status": "verified",
    "credential_version": 1, "created_at": "2026-11-09T10:00:00Z", "updated_at": "2026-11-09T10:00:00Z",
}


def _scoped(engine, scope) -> ScopedSession:
    return ScopedSession(Session(engine), scope)


@pytest.fixture()
def engine():
    """One database holding the builder's two templates: one with a check, one without."""
    eng = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    SQLModel.metadata.create_all(eng)
    service = CustomConnectorService(_scoped(eng, BUILDER))
    service.save_member_draft(parse_connector_draft(DRAFT))
    service.save_member_draft(parse_connector_draft({k: v for k, v in DRAFT.items() if k != "check"} | {"connector_id": "no_check_api"}))
    return eng


@pytest.fixture()
def world(engine, monkeypatch):
    """Settings on, Redis faked, and auth and the provider behind recording transports."""
    for name, value in {
        "COWORK_TURN_BACKEND": "remote",
        "COWORK_TURN_CONNECTOR_DRAFTS_ENABLED": "true",
        "COWORK_TURN_AUTH_INTERNAL_BASE_URL": "http://auth.internal",
        "COWORK_TURN_DATASOURCE_PRODUCER_KEY_ID": "producer",
        "COWORK_TURN_DATASOURCE_PRODUCER_KEY": "producer-secret",
    }.items():
        monkeypatch.setenv(name, value)
    state = {
        "auth": [], "provider": [], "redis": fakeredis.FakeAsyncRedis(),
        "auth_answer": (201, STORED), "dry_answer": (200, {"dry_run": True}),
        "unprotected": (401, b'{"error":"no key"}'), "protected": (200, b'{"id":"me"}'),
    }

    def auth(request: httpx.Request) -> httpx.Response:
        state["auth"].append(request)
        status, body = state["dry_answer"] if json.loads(request.content)["dry_run"] else state["auth_answer"]
        return httpx.Response(status, json=body)

    async def provider(request: httpx.Request) -> httpx.Response:
        state["provider"].append(request)
        answer = state["protected"] if "authorization" in request.headers else state["unprotected"]
        if callable(answer):
            return await answer(request)
        status, body = answer
        return httpx.Response(status, stream=httpx.ByteStream(body))

    real_client = httpx.AsyncClient
    auth_transport = httpx.MockTransport(auth)
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kw: real_client(**{"transport": auth_transport, **kw}))
    monkeypatch.setattr(egress, "_TRANSPORT", httpx.MockTransport(provider))
    monkeypatch.setattr(
        egress, "_RESOLVER", lambda host, *a, **k: [(socket.AF_INET, socket.SOCK_STREAM, 0, "", ("93.184.216.34", 443))]
    )
    monkeypatch.setattr(custom_endpoints, "get_redis", lambda: state["redis"])
    return state


def _client(engine, monkeypatch, scope=BUILDER) -> TestClient:
    monkeypatch.setattr(custom_endpoints, "run_db", inline_run_db(_scoped(engine, scope)))
    app = FastAPI()
    app.include_router(custom_endpoints.router, prefix="/custom")
    app.dependency_overrides[get_principal] = lambda: Principal(user_id=scope.user_id, org_id=scope.org_id)
    app.dependency_overrides[get_scoped_session] = lambda: _scoped(engine, scope)
    app.dependency_overrides[get_tenant_scope] = lambda: scope
    return TestClient(app, headers={"Authorization": BEARER})


def _connect(client, connector_id="acme_crm", **body):
    payload = {"name": "Acme", "address": "api.acme.example", "values": {"api_key": SECRET}} | body
    return client.post(f"/custom/{connector_id}/connect", json=payload)


def test_a_passing_check_stores_the_connection_verified_with_the_typed_address(engine, world, monkeypatch):
    res = _connect(_client(engine, monkeypatch), address="https://API.Acme.Example./", values={"api_key": SECRET, "workspace": "w1"})

    assert res.status_code == 201, res.text
    assert res.json()["id"] == 7
    unprotected, protected = world["provider"]
    assert str(unprotected.url) == "https://93.184.216.34/v1/me"
    assert unprotected.headers["host"] == "api.acme.example"
    assert "authorization" not in unprotected.headers and unprotected.headers["accept"] == "application/json"
    assert protected.headers["authorization"] == f"Bearer {SECRET}"
    dry, create = world["auth"]
    sent = json.loads(create.content)
    assert json.loads(dry.content)["dry_run"] is True and sent["dry_run"] is False
    assert sent["fields"] == {"api_key": SECRET, "workspace": "w1", "base_url": "https://api.acme.example"}
    assert sent["secure_keys"] == ["api_key"] and sent["organization_id"] == "org-a"
    assert create.headers["authorization"] == BEARER
    assert SECRET not in res.text


def test_a_field_the_check_places_is_stored_secret_whatever_the_template_says(engine, world, monkeypatch):
    draft = {**DRAFT, "connector_id": "tenant_api"}
    draft["check"] = {"path": "/me", "headers": {"Authorization": "Bearer {api_key}", "X-Workspace": "{workspace}"}}
    CustomConnectorService(_scoped(engine, BUILDER)).save_member_draft(parse_connector_draft(draft))
    world["unprotected"] = (403, b"")

    res = _connect(_client(engine, monkeypatch), "tenant_api", values={"api_key": SECRET, "workspace": "w1"})

    assert res.status_code == 201
    assert json.loads(world["auth"][-1].content)["secure_keys"] == ["api_key", "workspace"]
    assert "x-workspace" not in world["provider"][0].headers


@pytest.mark.parametrize("scope", [SAME_ORG, OTHER_ORG], ids=["same-org-member", "other-org"])
def test_another_members_template_is_not_found_and_nothing_is_sent(engine, world, monkeypatch, scope):
    res = _connect(_client(engine, monkeypatch, scope))

    assert res.status_code == 404
    assert world["auth"] == [] and world["provider"] == []


def test_a_template_without_a_check_cannot_connect(engine, world, monkeypatch):
    res = _connect(_client(engine, monkeypatch), "no_check_api")

    assert (res.status_code, res.json()["detail"]["code"]) == (409, "no_check")
    message = res.json()["detail"]["message"]
    assert "ask Anton to build it again" in message
    assert "if it takes none, Anton can use it without a connector" in message
    assert world["auth"] == [] and world["provider"] == []


def test_a_stored_check_that_no_longer_parses_cannot_connect(engine, world, monkeypatch):
    session = _scoped(engine, BUILDER)
    row = CustomConnectorService(session).get("acme_crm")
    row.connection_check = {"path": "/v1/me", "headers": {"Authorization": "Bearer {undeclared}"}}
    session.add(row)
    session.commit()

    res = _connect(_client(engine, monkeypatch))

    assert (res.status_code, res.json()["detail"]["code"]) == (409, "no_check")
    assert world["auth"] == [] and world["provider"] == []


@pytest.mark.parametrize(
    "address",
    [
        "http://api.acme.example", "https://api.acme.example/v1", "api.acme.example/v1", "https://api.acme.example:8443",
        "https://user@api.acme.example", "api.acme.example?x=1", "api.acme.example#", "api .acme.example",
        "api.acme.example\n", "localhost", "api.acme.123", "api..acme.example", "api_x.acme.example",
        "xn--80ak6aa92e.com", "api.xn--80ak6aa92e.com", "ａpi.acme.example", "bücher.example", "api.acme.example ", "1.1.1.1", "[2606:4700::1111]", "0x01010101",
        "https:api.acme.example", "//api.acme.example", "",
    ],
)
def test_an_address_outside_the_rules_is_refused_before_anything_is_sent(engine, world, monkeypatch, address):
    res = _connect(_client(engine, monkeypatch), address=address)

    assert res.status_code == 422
    assert world["auth"] == [] and world["provider"] == []


@pytest.mark.parametrize("address", ["mindshub.ai", "auth.mindshub.ai", "API.MDB.AI.", "svc.cluster.local"])
def test_our_own_domains_are_refused(engine, world, monkeypatch, address):
    res = _connect(_client(engine, monkeypatch), address=address)

    assert (res.status_code, res.json()["detail"]["code"]) == (422, "own_domain")
    assert world["provider"] == []


def test_a_name_that_only_contains_our_domain_is_not_ours(engine, world, monkeypatch):
    assert _connect(_client(engine, monkeypatch), address="mindshub.ai.acme.example").status_code == 201


@pytest.mark.parametrize(
    "values",
    [
        {}, {"api_key": ""}, {"api_key": SECRET, "other": "x"}, {"api_key": SECRET + "\n"},
        {"api_key": SECRET + "é"}, {"api_key": "a" * 4097}, {"api_key": 12345},
    ],
    ids=["missing", "empty", "unknown-field", "newline", "non-ascii", "too-long", "not-text"],
)
def test_a_value_outside_the_rules_is_refused_by_field_never_by_value(engine, world, monkeypatch, values):
    res = _connect(_client(engine, monkeypatch), values=values)

    assert (res.status_code, res.json()["detail"]["code"]) == (422, "invalid_values")
    assert SECRET not in res.text and "x" * 10 not in res.text
    assert world["auth"] == [] and world["provider"] == []


@pytest.mark.parametrize(
    ("fields", "values"),
    [
        # A required field the check does not place.
        ([{"name": "api_key", "label": "API key"}, {"name": "region", "label": "Region", "secret": False}], {"api_key": SECRET}),
        # An optional field the check places.
        ([{"name": "api_key", "label": "API key"}, {"name": "tenant", "label": "Tenant", "required": False}], {"api_key": SECRET}),
    ],
    ids=["required-not-placed", "optional-but-placed"],
)
def test_every_needed_value_is_present_before_anything_is_sent(engine, world, monkeypatch, fields, values):
    headers = {"Authorization": "Bearer {api_key}"} | ({"X-Tenant": "{tenant}"} if fields[1]["name"] == "tenant" else {})
    draft = {"connector_id": "needs_api", "label": "Needs", "fields": fields, "check": {"path": "/me", "headers": headers}}
    CustomConnectorService(_scoped(engine, BUILDER)).save_member_draft(parse_connector_draft(draft))

    res = _connect(_client(engine, monkeypatch), "needs_api", values=values)

    assert (res.status_code, res.json()["detail"]["code"]) == (422, "invalid_values")
    assert world["auth"] == [] and world["provider"] == []


def test_a_check_that_ignores_the_credential_proves_nothing(engine, world, monkeypatch):
    world["unprotected"] = (200, b"welcome")

    res = _connect(_client(engine, monkeypatch))

    assert res.status_code == 422
    assert res.json()["detail"] == {"code": "check_not_protected", "provider_status": 200, "excerpt": "welcome"}
    assert len(world["provider"]) == 1
    assert [json.loads(r.content)["dry_run"] for r in world["auth"]] == [True]


@pytest.mark.parametrize(("status", "code"), [(401, "authentication_failed"), (403, "authentication_failed"), (500, "provider_error")])
def test_a_failed_check_shows_the_status_and_a_scrubbed_excerpt(engine, world, monkeypatch, status, code):
    world["protected"] = (status, f'{{"error":"bad key {SECRET}"}}'.encode())

    res = _connect(_client(engine, monkeypatch))

    assert res.status_code == 422
    assert res.json()["detail"] == {"code": code, "provider_status": status, "excerpt": '{"error":"bad key ***"}'}
    assert [json.loads(r.content)["dry_run"] for r in world["auth"]] == [True]


def test_the_check_has_one_deadline(engine, world, monkeypatch):
    async def stall(request):
        await asyncio.sleep(5)

    world["protected"] = stall
    monkeypatch.setattr(custom_connect, "CHECK_SECONDS", 0.2)

    res = _connect(_client(engine, monkeypatch))

    assert res.json()["detail"]["code"] == "timed_out"


def test_an_address_that_resolves_privately_is_not_dialed(engine, world, monkeypatch):
    monkeypatch.setattr(
        egress, "_RESOLVER", lambda host, *a, **k: [(socket.AF_INET, socket.SOCK_STREAM, 0, "", ("10.0.0.5", 443))]
    )

    res = _connect(_client(engine, monkeypatch))

    assert res.json()["detail"]["code"] == "host_not_public"
    assert world["provider"] == []


@pytest.mark.parametrize(
    ("status", "code", "says"),
    [
        (409, "conflict", "already have a connection with this name"),
        (429, "quota_exceeded", "limit of custom connections"),
        (409, "organization_mismatch", "organization changed"),
        (503, "disabled", "turned off on this deployment"),
        (400, "invalid_connection", "could not be stored"),
    ],
)
def test_auths_dry_run_refusal_answers_before_anything_reaches_the_provider(engine, world, monkeypatch, status, code, says):
    world["dry_answer"] = (status, {"code": code, "detail": "auth's own words"})

    res = _connect(_client(engine, monkeypatch))

    assert (res.status_code, res.json()["detail"]["code"]) == (status, code)
    assert says in res.json()["detail"]["message"]
    assert world["provider"] == []


def test_the_eleventh_attempt_in_the_window_is_refused_before_anything_is_sent(engine, world, monkeypatch):
    client = _client(engine, monkeypatch)
    for _ in range(custom_connect.CONNECT_ATTEMPTS_MAX):
        assert _connect(client, address="localhost").status_code == 422

    res = _connect(client)

    assert (res.status_code, res.json()["detail"]["code"]) == (429, "rate_limited")
    assert world["auth"] == [] and world["provider"] == []
    ttl = asyncio.run(world["redis"].ttl("cowork:custom-connect:org-a:builder-a"))
    assert 0 < ttl <= custom_connect.CONNECT_WINDOW_SECONDS


def test_attempts_are_counted_per_member(engine, world, monkeypatch):
    for _ in range(custom_connect.CONNECT_ATTEMPTS_MAX):
        _connect(_client(engine, monkeypatch, SAME_ORG))

    assert _connect(_client(engine, monkeypatch)).status_code == 201


def test_a_redis_outage_refuses_before_anything_is_sent(engine, world, monkeypatch):
    server = fakeredis.FakeServer()
    server.connected = False
    world["redis"] = fakeredis.FakeAsyncRedis(server=server)

    res = _connect(_client(engine, monkeypatch))

    assert (res.status_code, res.json()["detail"]["code"]) == (503, "unavailable")
    assert world["auth"] == [] and world["provider"] == []


def test_without_redis_the_deployment_cannot_connect(engine, world, monkeypatch):
    monkeypatch.setenv("COWORK_TURN_BACKEND", "inprocess")

    res = _connect(_client(engine, monkeypatch))

    assert (res.status_code, res.json()["detail"]["code"]) == (503, "unavailable")
    assert world["provider"] == []


@pytest.mark.parametrize("setting", ["org", "flag"])
def test_outside_org_mode_or_with_the_setting_off_the_route_is_404(engine, world, monkeypatch, setting):
    scope = LOCAL_SCOPE if setting == "org" else BUILDER
    if setting == "org":
        # A local definition with a check, so only the org-mode gate stands in the way.
        row = CustomConnectorService(_scoped(engine, BUILDER)).get("acme_crm")
        with Session(engine) as session:
            session.add(CustomConnector(
                connector_id="acme_crm", label="Acme CRM", spec=row.spec, connection_check=row.connection_check,
            ))
            session.commit()
    if setting == "flag":
        monkeypatch.setenv("COWORK_TURN_CONNECTOR_DRAFTS_ENABLED", "false")

    assert _connect(_client(engine, monkeypatch, scope)).status_code == 404
    assert world["provider"] == []


def test_the_excerpt_removes_controls_then_replaces_values_longest_first_then_cuts():
    hidden = SECRET[:6] + "​" + SECRET[6:]
    assert custom_connect.scrub_excerpt(f"key {hidden}\x07!".encode(), [SECRET]) == "key ***!"
    assert custom_connect.scrub_excerpt(b"abcdef abc", ["abc", "abcdef"]) == "*** ***"
    near_the_end = ("a" * 295 + SECRET).encode()
    assert custom_connect.scrub_excerpt(near_the_end, [SECRET]) == "a" * 295 + "***"
    assert custom_connect.scrub_excerpt(b"\xff" + b"b" * 400, []) == "�" + "b" * 299
