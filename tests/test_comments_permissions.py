"""HTTP-layer proof that comments.py's two routes require identity in org
mode (ENG-2094) — the business logic (local journal vs. cloud proxy) is
stubbed out so these tests isolate the permission layer, already covered by
test_comments_routing.py / test_comments_proxy_org.py.
"""
from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from cowork.api.v1.endpoints import comments as comments_module
from cowork.api.v1.router import api_router
from cowork.principal import Principal, get_principal

STREAM_PATH = "/api/v1/artifact-comments/artifact/e9267de0-3fa4-4c17-b964-dc63216311cf/stream"
REST_PATH = "/api/v1/artifact-comments/artifact/e9267de0-3fa4-4c17-b964-dc63216311cf/threads"
PRINCIPAL = Principal(user_id="u1", org_id="o1")


@pytest.fixture(autouse=True)
def _reset_app_settings():
    from cowork.common.settings.app_settings import get_app_settings

    get_app_settings.cache_clear()
    yield
    get_app_settings.cache_clear()


@pytest.fixture(autouse=True)
def _stub_business_logic(monkeypatch):
    """Every route resolves to the cloud-proxy branch, which is stubbed to a
    trivial success — isolates the permission dependency from artifact
    resolution and the real proxy/local-journal logic."""
    monkeypatch.setattr(comments_module, "resolve_comments_route", lambda user_dir, report_id: ("u", "r"))

    async def fake_stream(request, user_dir, report_id):
        return {"ok": "stream"}

    async def fake_rest(request, user_dir, report_id, subpath):
        return {"ok": "rest"}

    monkeypatch.setattr(comments_module, "forward_comments_stream", fake_stream)
    monkeypatch.setattr(comments_module, "forward_comments_rest", fake_rest)


def _client(principal: Principal | None) -> TestClient:
    app = FastAPI()
    app.include_router(api_router)
    app.dependency_overrides[get_principal] = lambda: principal
    return TestClient(app)


def test_stream_requires_identity_in_org_mode(monkeypatch):
    monkeypatch.setenv("COWORK_TENANCY_MODE", "org")
    resp = _client(principal=None).get(STREAM_PATH)
    assert resp.status_code == 401


def test_rest_requires_identity_in_org_mode(monkeypatch):
    monkeypatch.setenv("COWORK_TENANCY_MODE", "org")
    resp = _client(principal=None).get(REST_PATH)
    assert resp.status_code == 401


def test_stream_allows_an_authenticated_member_in_org_mode(monkeypatch):
    monkeypatch.setenv("COWORK_TENANCY_MODE", "org")
    resp = _client(principal=PRINCIPAL).get(STREAM_PATH)
    assert resp.status_code == 200
    assert resp.json() == {"ok": "stream"}


def test_rest_allows_an_authenticated_member_in_org_mode(monkeypatch):
    monkeypatch.setenv("COWORK_TENANCY_MODE", "org")
    resp = _client(principal=PRINCIPAL).get(REST_PATH)
    assert resp.status_code == 200
    assert resp.json() == {"ok": "rest"}


def test_stream_is_unchanged_in_local_mode_with_no_principal():
    # tenancy_mode defaults to "local" — no COWORK_TENANCY_MODE set.
    resp = _client(principal=None).get(STREAM_PATH)
    assert resp.status_code == 200
    assert resp.json() == {"ok": "stream"}


def test_rest_is_unchanged_in_local_mode_with_no_principal():
    resp = _client(principal=None).get(REST_PATH)
    assert resp.status_code == 200
    assert resp.json() == {"ok": "rest"}
