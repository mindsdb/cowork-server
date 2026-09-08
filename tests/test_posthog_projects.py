import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from cowork.api.v1.router import api_router
from cowork.principal import Principal, get_principal
from cowork.services.connectors.posthog import PostHogDiscoveryError, discover_projects

PATH = "/api/v1/connectors/posthog/projects"
BODY = {"personal_api_key": "secret-key", "host": "https://us.posthog.com"}
PRINCIPAL = Principal(user_id="u1", org_id="o1")


def _client(principal: Principal | None) -> TestClient:
    app = FastAPI()
    app.include_router(api_router)
    app.dependency_overrides[get_principal] = lambda: principal
    return TestClient(app)


class _Response:
    def __init__(self, status_code, payload):
        self.status_code = status_code
        self._payload = payload

    def json(self):
        if isinstance(self._payload, Exception):
            raise self._payload
        return self._payload


class _Client:
    response = None
    calls = []

    def __init__(self, **_kwargs):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args):
        return None

    async def get(self, url, **kwargs):
        self.calls.append((url, kwargs))
        return self.response


@pytest.mark.asyncio
async def test_discovers_posthog_project_choices(monkeypatch):
    _Client.calls = []
    _Client.response = _Response(200, {"results": [{"id": 12, "name": "Production"}]})
    monkeypatch.setattr(httpx, "AsyncClient", _Client)

    projects = await discover_projects(
        personal_api_key="secret-key", host="https://us.posthog.com"
    )

    assert projects[0].id == "12"
    assert projects[0].name == "Production"
    assert _Client.calls == [(
        "https://us.posthog.com/api/projects/",
        {"headers": {"Authorization": "Bearer secret-key"}},
    )]


@pytest.mark.asyncio
async def test_hides_posthog_auth_failure(monkeypatch):
    _Client.response = _Response(401, {})
    monkeypatch.setattr(httpx, "AsyncClient", _Client)

    with pytest.raises(PostHogDiscoveryError, match="rejected"):
        await discover_projects(personal_api_key="secret-key", host="https://eu.posthog.com")


@pytest.mark.asyncio
async def test_rejects_invalid_posthog_host():
    with pytest.raises(PostHogDiscoveryError, match="valid HTTPS PostHog host"):
        await discover_projects(personal_api_key="secret-key", host="not a URL")


@pytest.fixture(autouse=True)
def _reset_app_settings():
    from cowork.common.settings.app_settings import get_app_settings

    get_app_settings.cache_clear()
    yield
    get_app_settings.cache_clear()


def test_route_requires_identity_in_org_mode(monkeypatch):
    monkeypatch.setenv("COWORK_TENANCY_MODE", "org")

    resp = _client(principal=None).post(PATH, json=BODY)

    assert resp.status_code == 401


def test_route_allows_an_authenticated_member_in_org_mode(monkeypatch):
    monkeypatch.setenv("COWORK_TENANCY_MODE", "org")
    _Client.response = _Response(200, {"results": [{"id": 12, "name": "Production"}]})
    monkeypatch.setattr(httpx, "AsyncClient", _Client)

    resp = _client(principal=PRINCIPAL).post(PATH, json=BODY)

    assert resp.status_code == 200
    assert resp.json() == {"projects": [{"id": "12", "name": "Production"}]}


def test_route_is_unchanged_in_local_mode_with_no_principal(monkeypatch):
    # tenancy_mode defaults to "local" — no COWORK_TENANCY_MODE set.
    _Client.response = _Response(200, {"results": [{"id": 12, "name": "Production"}]})
    monkeypatch.setattr(httpx, "AsyncClient", _Client)

    resp = _client(principal=None).post(PATH, json=BODY)

    assert resp.status_code == 200
    assert resp.json() == {"projects": [{"id": "12", "name": "Production"}]}
