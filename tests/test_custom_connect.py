"""Connecting a member's own custom connector: address, values, the two-send check, auth and the attempt limit."""

from __future__ import annotations

import asyncio
import base64
import json
import socket
from urllib.parse import quote, quote_plus

import fakeredis
import httpx
import pytest
from anton.core.tools import connector_auth_rules
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
        {"name": "api_key", "label": "API key", "secret": True},
        {"name": "workspace", "label": "Workspace", "secret": False},
    ],
    "auth": {
        "type": "header",
        "headers": {"Authorization": "Bearer {api_key}", "X-Workspace": "{workspace}", "X-Api-Version": "2024-01"},
    },
    "check": {"path": "/v1/me"},
}
VALUES = {"api_key": SECRET, "workspace": "w1"}
BASIC_DRAFT = {
    "connector_id": "basic_api",
    "label": "Basic API",
    "fields": [{"name": "email", "label": "Email", "secret": False}, {"name": "api_token", "label": "Token"}],
    "auth": {"type": "basic", "username": "{email}/token", "password": "{api_token}"},
    "check": {"path": "/api/v2/me"},
}
QUERY_DRAFT = {
    "connector_id": "query_api",
    "label": "Query API",
    "fields": [{"name": "access_key", "label": "Access key"}],
    "auth": {"type": "query", "params": {"access_key": "{access_key}", "output": "json"}},
    "check": {"path": "/v1/status?pretty"},
}
TWO_SLOT_DRAFT = {
    "connector_id": "two_slot_api",
    "label": "Two Slot API",
    "fields": [{"name": "access_key", "label": "Access key"}, {"name": "account", "label": "Account"}],
    "auth": {"type": "query", "params": {"access_key": "{access_key}", "account": "{account}"}},
    "check": {"path": "/v1/status"},
}
PAIRED_DRAFT = {
    "connector_id": "paired_api",
    "label": "Paired API",
    "fields": [{"name": "s", "label": "Short"}, *({"name": f"f{i}", "label": f"Part {i}"} for i in range(10))],
    "auth": {
        "type": "header",
        "headers": {f"X-Acme-Part-{i}": f"{{s}}{{f{2 * i}}}{{f{2 * i + 1}}}" for i in range(5)},
    },
    "check": {"path": "/v1/status"},
}
HEADER_KEY_DRAFT = {
    "connector_id": "header_key_api",
    "label": "Header Key API",
    "fields": [{"name": "api_key", "label": "API key"}],
    "auth": {"type": "header", "headers": {"X-Api-Key": "{api_key}", "X-Api-Version": "2024-01"}},
    "check": {"path": "/v1/status"},
}
STORED = {
    "id": 7, "connector_id": "acme_crm", "name": "Acme", "secure_keys": ["api_key"], "status": "verified",
    "credential_version": 1, "created_at": "2026-11-09T10:00:00Z", "updated_at": "2026-11-09T10:00:00Z",
}


def _scoped(engine, scope) -> ScopedSession:
    return ScopedSession(Session(engine), scope)


@pytest.fixture()
def engine():
    """One database holding the builder's templates: one per auth type, and one without a check."""
    eng = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    SQLModel.metadata.create_all(eng)
    service = CustomConnectorService(_scoped(eng, BUILDER))
    for draft in (DRAFT, BASIC_DRAFT, QUERY_DRAFT, TWO_SLOT_DRAFT, HEADER_KEY_DRAFT, PAIRED_DRAFT):
        service.save_member_draft(parse_connector_draft(draft))
    service.save_member_draft(parse_connector_draft({k: v for k, v in DRAFT.items() if k != "check"} | {"connector_id": "no_check_api"}))
    return eng


def _carries_the_credential(request: httpx.Request) -> bool:
    """Whether the member's key is on the request, wherever the template put it."""
    if SECRET in request.url.raw_path.decode() or quote(SECRET, safe="") in request.url.raw_path.decode():
        return True
    for name, value in request.headers.items():
        if SECRET in value:
            return True
        if name == "authorization" and value.startswith("Basic "):
            return SECRET in base64.b64decode(value.removeprefix("Basic ")).decode()
    return False


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
        answer = state["protected"] if _carries_the_credential(request) else state["unprotected"]
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
    payload = {"name": "Acme", "address": "api.acme.example", "values": VALUES} | body
    return client.post(f"/custom/{connector_id}/connect", json=payload)


def test_a_passing_check_stores_the_connection_verified_with_the_typed_address(engine, world, monkeypatch):
    res = _connect(_client(engine, monkeypatch), address="https://API.Acme.Example./")

    assert res.status_code == 201, res.text
    assert res.json()["id"] == 7
    unprotected, protected = world["provider"]
    assert str(unprotected.url) == "https://93.184.216.34/v1/me"
    assert unprotected.headers["host"] == "api.acme.example"
    # The first send keeps the slot-free header and leaves out every placed one.
    assert unprotected.headers["x-api-version"] == "2024-01"
    assert "authorization" not in unprotected.headers and "x-workspace" not in unprotected.headers
    assert protected.headers["authorization"] == f"Bearer {SECRET}" and protected.headers["x-workspace"] == "w1"
    dry, create = world["auth"]
    sent = json.loads(create.content)
    assert json.loads(dry.content)["dry_run"] is True and sent["dry_run"] is False
    assert sent["fields"] == {"api_key": SECRET, "workspace": "w1", "base_url": "https://api.acme.example"}
    assert sent["secure_keys"] == ["api_key", "workspace"] and sent["organization_id"] == "org-a"
    assert sent["auth"] == DRAFT["auth"]
    assert create.headers["authorization"] == BEARER
    assert SECRET not in res.text


def test_the_auth_section_sent_is_one_auths_create_accepts_as_sent(engine, world, monkeypatch):
    for connector_id, values in (("acme_crm", VALUES), ("basic_api", {"email": "me@x.io", "api_token": SECRET}),
                                 ("query_api", {"access_key": SECRET})):
        assert _connect(_client(engine, monkeypatch), connector_id, values=values).status_code == 201
    for create in world["auth"][1::2]:
        sent = json.loads(create.content)
        held = [value for name, value in sent["fields"].items() if name != "base_url" and len(value) >= 8]
        placed = [(name, True) for name in sent["fields"] if name != "base_url"]

        # The arguments auth's create passes to the same shared rules.
        rebuilt = connector_auth_rules.parse_auth(sent["auth"], placed, lambda text: any(v in text for v in held))

        assert rebuilt == sent["auth"]


def test_a_basic_template_signs_in_only_on_the_second_send(engine, world, monkeypatch):
    res = _connect(_client(engine, monkeypatch), "basic_api", values={"email": "me@x.io", "api_token": SECRET})

    assert res.status_code == 201, res.text
    unprotected, protected = world["provider"]
    assert "authorization" not in unprotected.headers
    token = base64.b64encode(f"me@x.io/token:{SECRET}".encode()).decode()
    assert protected.headers["authorization"] == f"Basic {token}"
    assert unprotected.url.raw_path == protected.url.raw_path == b"/api/v2/me"


def test_a_query_template_adds_its_parameters_after_the_paths_own_query(engine, world, monkeypatch):
    res = _connect(_client(engine, monkeypatch), "query_api", values={"access_key": SECRET})

    assert res.status_code == 201, res.text
    unprotected, protected = world["provider"]
    assert unprotected.url.raw_path == b"/v1/status?pretty&output=json"
    assert protected.url.raw_path == f"/v1/status?pretty&access_key={quote(SECRET, safe='')}&output=json".encode()
    assert "authorization" not in protected.headers


@pytest.mark.parametrize("scope", [SAME_ORG, OTHER_ORG], ids=["same-org-member", "other-org"])
def test_another_members_template_is_not_found_and_nothing_is_sent(engine, world, monkeypatch, scope):
    res = _connect(_client(engine, monkeypatch, scope))

    assert res.status_code == 404
    assert world["auth"] == [] and world["provider"] == []


def test_a_template_without_a_check_cannot_connect(engine, world, monkeypatch):
    res = _connect(_client(engine, monkeypatch), "no_check_api")

    assert (res.status_code, res.json()["detail"]["code"]) == (409, "no_check")
    assert world["auth"] == [] and world["provider"] == []


@pytest.mark.parametrize(
    "change",
    [
        {"connection_check": {"path": "/v1/me", "headers": {"Authorization": "Bearer {api_key}"}}},
        {"connection_auth": None},
        {"connection_auth": {"type": "header", "headers": {"Authorization": "Bearer {undeclared}"}}},
    ],
    ids=["old-check-shape", "saved-without-auth", "auth-no-longer-parses"],
)
def test_a_template_saved_in_a_shape_the_parser_refuses_cannot_connect(engine, world, monkeypatch, change):
    session = _scoped(engine, BUILDER)
    row = CustomConnectorService(session).get("acme_crm")
    for column, value in change.items():
        setattr(row, column, value)
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


@pytest.mark.parametrize("address", [
    "mindshub.ai", "auth.mindshub.ai", "API.MDB.AI.", "svc.cluster.local",
    "draft.mindshub.co", "mindscloud.ai", "api.minds-db.com",
])
def test_our_own_domains_are_refused(engine, world, monkeypatch, address):
    res = _connect(_client(engine, monkeypatch), address=address)

    assert (res.status_code, res.json()["detail"]["code"]) == (422, "own_domain")
    assert world["provider"] == []


def test_a_name_that_only_contains_our_domain_is_not_ours(engine, world, monkeypatch):
    assert _connect(_client(engine, monkeypatch), address="mindshub.ai.acme.example").status_code == 201


@pytest.mark.parametrize(
    "values",
    [
        {}, {"api_key": SECRET}, {**VALUES, "api_key": ""}, {**VALUES, "other": "x"}, {**VALUES, "api_key": SECRET + "\n"},
        {**VALUES, "api_key": SECRET + "é"}, {**VALUES, "api_key": "a" * 4097}, {**VALUES, "api_key": 12345},
        {**VALUES, "api_key": " " + SECRET}, {**VALUES, "api_key": SECRET + " "},
    ],
    ids=[
        "none", "one-field-missing", "empty", "unknown-field", "newline", "non-ascii", "too-long", "not-text",
        "leading-space", "trailing-space",
    ],
)
def test_a_value_outside_the_rules_is_refused_by_field_never_by_value(engine, world, monkeypatch, values):
    res = _connect(_client(engine, monkeypatch), values=values)

    assert (res.status_code, res.json()["detail"]["code"]) == (422, "invalid_values")
    assert SECRET not in res.text and "x" * 10 not in res.text
    assert world["auth"] == [] and world["provider"] == []


def test_a_missing_value_is_named_by_its_field(engine, world, monkeypatch):
    res = _connect(_client(engine, monkeypatch), values={"api_key": SECRET})

    assert res.json()["detail"] == {"code": "invalid_values", "message": "workspace is required."}


def test_a_value_our_rules_pass_but_the_shared_rules_refuse_is_a_422_before_any_send(engine, world, monkeypatch):
    real = custom_connect.check_values
    monkeypatch.setattr(custom_connect, "check_values", lambda fields, values: {**real(fields, values), "workspace": "w\x01"})

    res = _connect(_client(engine, monkeypatch))

    assert (res.status_code, res.json()["detail"]["code"]) == (422, "invalid_values")
    assert world["auth"] == [] and world["provider"] == []


def test_a_failed_basic_check_replaces_the_echoed_sign_in_token(engine, world, monkeypatch):
    token = base64.b64encode(f"me@x.io/token:{SECRET}".encode()).decode()
    world["protected"] = (401, f'{{"seen": "{token.rstrip("=")}"}}'.encode())

    res = _connect(_client(engine, monkeypatch), "basic_api", values={"email": "me@x.io", "api_token": SECRET})

    assert res.json()["detail"] == {"code": "authentication_failed", "provider_status": 401, "excerpt": '{"seen": "***"}'}


def test_a_value_the_shared_rules_cannot_place_is_a_422_not_an_error():
    auth = DRAFT["auth"]

    with pytest.raises(custom_connect.ConnectRefused) as refused:
        custom_connect.place_values(auth, {"api_key": SECRET})

    assert (refused.value.status_code, refused.value.code) == (422, "invalid_values")
    assert SECRET not in refused.value.message


def test_a_check_that_ignores_the_credential_proves_nothing(engine, world, monkeypatch):
    world["unprotected"] = (200, b"internal status page")

    res = _connect(_client(engine, monkeypatch))

    assert res.status_code == 422
    assert res.json()["detail"] == {"code": "check_not_protected", "provider_status": 200, "excerpt": ""}
    assert "internal status page" not in res.text
    assert len(world["provider"]) == 1
    assert [json.loads(r.content)["dry_run"] for r in world["auth"]] == [True]


@pytest.mark.parametrize(("token_length", "refused"), [(2035, False), (2038, True)])
def test_a_rendered_value_over_the_gateways_8_kib_is_refused_before_any_send(
    engine, world, monkeypatch, token_length, refused
):
    # Basic sends "Basic " plus base64 of "{email}/token:{api_token}": 6138 bytes encode to 8190, 6141 to 8194.
    values = {"email": "e" * custom_connect.VALUE_MAX, "api_token": "t" * token_length}

    res = _connect(_client(engine, monkeypatch), "basic_api", values=values)

    if refused:
        assert res.status_code == 422
        assert res.json()["detail"]["code"] == "invalid_values"
        assert world["provider"] == []
    else:
        sent = world["provider"][-1].headers["authorization"]
        assert len(sent) == 8190
        assert res.json()["detail"]["code"] != "invalid_values"


@pytest.mark.parametrize(("short", "refused"), [("abcdefgh", False), ("abc", True)])
def test_values_the_gateway_could_not_search_for_together_are_refused_before_any_send(
    engine, world, monkeypatch, short, refused
):
    # Every header stays under 8 KiB. A short field makes the gateway also search each whole
    # header value, which takes the searched bytes past what its 1.5 MiB of forms allows.
    values = {"s": short, **{f"f{i}": chr(ord("a") + i) * 4090 for i in range(10)}}

    res = _connect(_client(engine, monkeypatch), "paired_api", values=values)

    if refused:
        assert res.status_code == 422
        assert res.json()["detail"]["code"] == "invalid_values"
        assert world["provider"] == []
    else:
        assert world["provider"]
        assert res.json()["detail"]["code"] != "invalid_values"


def test_the_connection_name_is_stored_trimmed_and_a_blank_one_is_refused(engine, world, monkeypatch):
    client = _client(engine, monkeypatch)

    assert _connect(client, name="   ").status_code == 422
    assert world["auth"] == []
    assert _connect(client, name="  Rates  ").status_code == 201
    assert json.loads(world["auth"][-1].content)["name"] == "Rates"


@pytest.mark.parametrize("status", [401, 403])
@pytest.mark.parametrize(("connector_id", "values"), [("acme_crm", VALUES), ("query_api", {"access_key": SECRET})])
def test_a_check_refused_without_the_key_then_answered_with_it_connects(engine, world, monkeypatch, status, connector_id, values):
    world["unprotected"] = (status, b'{"error":"no key"}')

    res = _connect(_client(engine, monkeypatch), connector_id, values=values)

    assert res.status_code == 201, res.text
    assert len(world["provider"]) == 2


@pytest.mark.parametrize("status", [400, 422])
@pytest.mark.parametrize(("connector_id", "values"), [
    ("query_api", {"access_key": SECRET}),
    ("header_key_api", {"api_key": SECRET}),
])
def test_a_bad_request_without_the_key_proves_it_when_the_key_is_the_one_value_placed(
    engine, world, monkeypatch, status, connector_id, values
):
    world["unprotected"] = (status, b'{"error_message":"Bad Request. Variable access_key is not set."}')

    res = _connect(_client(engine, monkeypatch), connector_id, values=values)

    assert res.status_code == 201, res.text
    assert len(world["provider"]) == 2


@pytest.mark.parametrize("status", [400, 422])
@pytest.mark.parametrize(("connector_id", "values"), [
    ("acme_crm", VALUES),
    ("basic_api", {"email": "me@x.io", "api_token": SECRET}),
    ("two_slot_api", {"access_key": SECRET, "account": "a1"}),
])
def test_a_bad_request_without_the_key_proves_nothing_when_another_value_is_missing(
    engine, world, monkeypatch, status, connector_id, values
):
    # The keyless send also lacks the other placed value, which alone could be the request's fault.
    world["unprotected"] = (status, b'{"error":"missing account"}')

    res = _connect(_client(engine, monkeypatch), connector_id, values=values)

    assert res.json()["detail"] == {"code": "check_not_protected", "provider_status": status, "excerpt": ""}
    assert len(world["provider"]) == 1


@pytest.mark.parametrize("status", [200, 429, 500])
def test_a_check_that_answers_otherwise_without_the_key_is_not_protected(engine, world, monkeypatch, status):
    world["unprotected"] = (status, b"internal status page")

    res = _connect(_client(engine, monkeypatch))

    assert res.json()["detail"] == {"code": "check_not_protected", "provider_status": status, "excerpt": ""}
    assert len(world["provider"]) == 1


def test_a_check_path_the_provider_does_not_have_says_so_before_the_key_is_sent(engine, world, monkeypatch):
    world["unprotected"] = (404, b'{"message":"Route not found"}')

    res = _connect(_client(engine, monkeypatch))

    assert res.status_code == 422
    assert res.json()["detail"] == {"code": "check_path_not_found", "provider_status": 404, "excerpt": ""}
    assert "Route not found" not in res.text
    assert len(world["provider"]) == 1
    assert [json.loads(r.content)["dry_run"] for r in world["auth"]] == [True]


@pytest.mark.parametrize(
    "address",
    ["api.acme.example/v2", "https://api.acme.example/v2/", "https://api.acme.example/v2/eod?symbols=AAPL", "api.acme.example/v2#top"],
)
def test_an_address_with_a_path_says_where_the_version_goes(engine, world, monkeypatch, address):
    res = _connect(_client(engine, monkeypatch), address=address)

    assert (res.status_code, res.json()["detail"]["code"]) == (422, "invalid_address")
    assert "ask Anton to build the connector again" in res.json()["detail"]["message"]
    assert world["auth"] == [] and world["provider"] == []


@pytest.mark.parametrize("address", ["api.acme.example?next=/v2", "api.acme.example#/v2"])
def test_a_slash_only_in_the_query_or_fragment_is_not_a_path(engine, world, monkeypatch, address):
    res = _connect(_client(engine, monkeypatch), address=address)

    assert (res.status_code, res.json()["detail"]["message"]) == (422, "The API address is not valid.")


@pytest.mark.parametrize(("status", "code"), [
    (401, "authentication_failed"), (403, "authentication_failed"), (404, "check_path_not_found"), (500, "provider_error"),
])
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
                connection_auth=row.connection_auth,
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


def test_the_excerpt_also_replaces_every_form_of_a_value_that_went_on_the_request():
    basic = BASIC_DRAFT["auth"]
    values = {"email": "me@x.io", "api_token": "hunter2+hunter2/x"}
    placed = custom_connect.place_values(basic, values)
    token = base64.b64encode(b"me@x.io/token:hunter2+hunter2/x").decode()
    forms = custom_connect.sent_forms(basic, placed, values.values())

    body = f"bare {token} unpadded {token.rstrip('=')} encoded {quote(values['api_token'], safe='')}"

    assert custom_connect.scrub_excerpt(body.encode(), forms) == "bare *** unpadded *** encoded ***"


def test_the_excerpt_replaces_a_value_in_its_form_encoded_spelling():
    auth = QUERY_DRAFT["auth"]
    values = {"access_key": "a b+c0123456"}
    placed = custom_connect.place_values(auth, values)
    forms = custom_connect.sent_forms(auth, placed, values.values())

    body = f"echo access_key={quote_plus(values['access_key'])}"

    assert custom_connect.scrub_excerpt(body.encode(), forms) == "echo access_key=***"


def test_the_excerpt_keeps_template_text_that_places_no_value():
    auth = DRAFT["auth"]
    placed = custom_connect.place_values(auth, VALUES)

    forms = custom_connect.sent_forms(auth, placed, VALUES.values())

    assert "2024-01" not in forms
    assert {SECRET, f"Bearer {SECRET}", "w1"} <= forms
