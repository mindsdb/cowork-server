"""A member's custom API connections: the auth relay for list and delete, and the internal create."""

from __future__ import annotations

import json

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from cowork.api.v1.endpoints.connectors import custom_connections as custom_connections_endpoints
from cowork.api.v1.permissions import AuthenticatedInOrgMode, AuthenticatedOrgAdmin
from cowork.api.v1.route_walker import declared_permissions
from cowork.common.settings.app_settings import TurnQueueSettings, get_app_settings
from cowork.db.scoped import LOCAL_SCOPE, get_scoped_session, get_tenant_scope
from cowork.principal import Principal, get_principal
from cowork.server import create_app
from cowork.services.connectors import custom_connections
from cowork.services.connectors.oauth import auth_proxy
from tests._fakes import inline_run_db
from tests import test_custom_connector_management as management

ORG_A = management.ORG_A
ORG_A_OTHER = management.ORG_A_OTHER
_scoped = management._scoped
#: The management tests' database: two httpbin templates, one of them ORG_A's own.
engine = management.engine

BEARER = "Bearer sentinel-member-credential"
SECRET = "sk_live_planted_value_never_echoed"

#: Shaped like auth's CustomApiConnectionResponseSerializer.
CONNECTION = {
    "id": 7,
    "connector_id": "httpbin",
    "name": "Acme",
    "secure_keys": ["api_key"],
    "status": "verified",
    "credential_version": 1,
    "created_at": "2026-11-09T10:00:00Z",
    "updated_at": "2026-11-09T10:00:00Z",
}


@pytest.fixture()
def auth(monkeypatch):
    """Record what reaches auth and script its answer."""
    recorded: list[httpx.Request] = []
    scripted: dict = {"status": 200, "body": {"items": [CONNECTION]}, "error": None}

    def handle(request: httpx.Request) -> httpx.Response:
        recorded.append(request)
        if scripted["error"]:
            raise scripted["error"]
        if scripted["status"] == 204:
            return httpx.Response(204)
        return httpx.Response(scripted["status"], json=scripted["body"])

    real_client = httpx.AsyncClient
    for module in (auth_proxy.httpx, custom_connections.httpx):
        monkeypatch.setattr(module, "AsyncClient", lambda **kw: real_client(transport=httpx.MockTransport(handle), **kw))
    monkeypatch.setenv("AUTH_SERVICE_BASE_URL", "https://auth.example.com")
    get_app_settings.cache_clear()
    yield recorded, scripted
    get_app_settings.cache_clear()


def _client(engine, scope, inline_db=None) -> TestClient:
    if inline_db is not None:
        inline_db(scope)
    app = FastAPI()
    app.include_router(custom_connections_endpoints.router, prefix="/custom-connections")
    app.dependency_overrides[get_principal] = lambda: Principal(user_id=scope.user_id, org_id=scope.org_id)
    app.dependency_overrides[get_scoped_session] = lambda: _scoped(engine, scope)
    app.dependency_overrides[get_tenant_scope] = lambda: scope
    return TestClient(app, headers={"Authorization": BEARER})


@pytest.fixture(autouse=True)
def inline_db(engine, monkeypatch):
    """Run each route's unit of work on the test database, inline, as the scope the client acts in."""

    def use(scope):
        monkeypatch.setattr(custom_connections_endpoints, "run_db", inline_run_db(_scoped(engine, scope)))

    return use


@pytest.fixture()
def org_mode(monkeypatch):
    monkeypatch.setenv("COWORK_TENANCY_MODE", "org")
    get_app_settings.cache_clear()
    yield
    get_app_settings.cache_clear()


def test_the_list_forwards_the_members_own_bearer_and_marks_a_deleted_template(engine, inline_db, auth, org_mode):
    recorded, scripted = auth
    # ORG_A keeps the fixture's httpbin template; the second connection's template is gone.
    scripted["body"] = {"items": [CONNECTION, {**CONNECTION, "id": 8, "connector_id": "gone_api"}]}

    res = _client(engine, ORG_A, inline_db).get("/custom-connections/")

    assert res.status_code == 200
    assert [(item["id"], item["template_missing"]) for item in res.json()] == [(7, False), (8, True)]
    [sent] = recorded
    assert str(sent.url) == "https://auth.example.com/v1/custom-connections"
    assert sent.headers["authorization"] == BEARER
    assert "user_id" not in str(sent.url) and "organization_id" not in str(sent.url)


def test_another_members_template_does_not_count_as_theirs(engine, inline_db, auth, org_mode):
    """ORG_A_OTHER keeps no httpbin template of their own, so their httpbin connection's template is gone."""
    res = _client(engine, ORG_A_OTHER, inline_db).get("/custom-connections/")

    assert [item["template_missing"] for item in res.json()] == [True]


def test_delete_forwards_and_passes_auths_answer_through(engine, inline_db, auth, org_mode):
    recorded, scripted = auth
    scripted["status"] = 204

    assert _client(engine, ORG_A, inline_db).delete("/custom-connections/7").status_code == 204
    assert str(recorded[0].url) == "https://auth.example.com/v1/custom-connections/7"
    assert recorded[0].method == "DELETE"
    assert recorded[0].headers["authorization"] == BEARER

    scripted.update(status=404, body={"detail": "connection_not_found"})
    assert _client(engine, ORG_A, inline_db).delete("/custom-connections/9").status_code == 404


def test_list_and_delete_stay_open_while_the_setting_is_off(engine, inline_db, auth, org_mode, monkeypatch):
    recorded, scripted = auth
    monkeypatch.setenv("COWORK_TURN_CONNECTOR_DRAFTS_ENABLED", "false")
    get_app_settings.cache_clear()

    assert _client(engine, ORG_A, inline_db).get("/custom-connections/").status_code == 200
    scripted["status"] = 204
    assert _client(engine, ORG_A, inline_db).delete("/custom-connections/7").status_code == 204


def test_outside_org_mode_the_surface_is_404_and_auth_is_never_asked(engine, inline_db, auth):
    recorded, _ = auth

    assert _client(engine, LOCAL_SCOPE, inline_db).get("/custom-connections/").status_code == 404
    assert _client(engine, LOCAL_SCOPE, inline_db).delete("/custom-connections/7").status_code == 404
    assert recorded == []


def test_both_routes_need_an_authenticated_member_not_an_admin():
    routes = [r for r in create_app().routes if getattr(r, "path", "").startswith("/api/v1/connectors/custom-connections")]

    assert {tuple(sorted(r.methods)) for r in routes} == {("GET",), ("DELETE",)}
    for route in routes:
        assert AuthenticatedInOrgMode in declared_permissions(route)
        assert AuthenticatedOrgAdmin not in declared_permissions(route)


# ── the internal create ─────────────────────────────────────────────────────


def _settings(**overrides) -> TurnQueueSettings:
    values = {
        "auth_internal_base_url": "http://auth.internal",
        "datasource_producer_key_id": "producer",
        "datasource_producer_key": "producer-secret",
    }
    values.update(overrides)
    return TurnQueueSettings(**values)


_PAYLOAD = {"connector_id": "acme_crm", "name": "Acme", "fields": {"api_key": SECRET}, "secure_keys": ["api_key"]}


async def _create(**kwargs):
    return await custom_connections.create_custom_connection(
        credential=kwargs.pop("credential", BEARER),
        organization_id="org-a",
        payload=_PAYLOAD,
        dry_run=kwargs.pop("dry_run", False),
        settings=kwargs.pop("settings", _settings()),
    )


async def test_create_sends_the_producer_and_member_credentials_and_the_organization(auth):
    recorded, scripted = auth
    scripted.update(status=201, body=CONNECTION)

    answer = await _create(dry_run=True)

    assert answer == CONNECTION
    [sent] = recorded
    assert str(sent.url) == "http://auth.internal/internal/custom-connections/create/"
    assert sent.headers["x-datasource-service-key-id"] == "producer"
    assert sent.headers["x-datasource-service-key"] == "producer-secret"
    assert sent.headers["authorization"] == BEARER
    assert json.loads(sent.content) == {**_PAYLOAD, "organization_id": "org-a", "dry_run": True}


async def test_a_refusal_carries_auths_code_and_never_a_value(auth):
    _, scripted = auth
    scripted.update(status=409, body={"code": "conflict", "detail": "a connection with this name already exists"})

    with pytest.raises(custom_connections.CustomConnectionRefused) as refused:
        await _create()

    assert (refused.value.code, refused.value.status_code) == ("conflict", 409)
    assert SECRET not in str(refused.value)


async def test_a_refusal_without_a_code_reads_as_uncoded(auth):
    _, scripted = auth
    scripted.update(status=500, body=["not", "an", "object"])

    with pytest.raises(custom_connections.CustomConnectionRefused) as refused:
        await _create()

    assert refused.value.code == "uncoded"


@pytest.mark.parametrize(
    "problem",
    ["unreachable", "unreadable", "not-an-object", "no-base-url", "no-key-id", "no-key", "no-credential"],
)
async def test_create_is_unavailable_without_a_usable_answer_or_identity(auth, problem):
    _, scripted = auth
    kwargs: dict = {}
    if problem == "unreachable":
        scripted["error"] = httpx.ConnectError("down")
    elif problem == "unreadable":
        scripted["body"] = None
        scripted["status"] = 200
    elif problem == "not-an-object":
        scripted["body"] = [SECRET]
    elif problem == "no-base-url":
        kwargs["settings"] = _settings(auth_internal_base_url="")
    elif problem == "no-key-id":
        kwargs["settings"] = _settings(datasource_producer_key_id="")
    elif problem == "no-key":
        kwargs["settings"] = _settings(datasource_producer_key="")
    else:
        kwargs["credential"] = ""
    if problem == "unreadable":
        real = custom_connections.httpx.Response.json

        def broken(self, **_):
            raise ValueError("not json")

        custom_connections.httpx.Response.json = broken
    try:
        with pytest.raises(custom_connections.CustomConnectionUnavailable) as unavailable:
            await _create(**kwargs)
    finally:
        if problem == "unreadable":
            custom_connections.httpx.Response.json = real

    assert SECRET not in str(unavailable.value)
