"""The runtime control-plane router is refused in org mode.

Remote computers are a desktop control-plane feature. Hosted/org activation
needs the tenant-bound service resolver and SQL store, and until that boundary
is wired the whole router fails closed rather than sharing desktop-global
state.

Asserted through a request rather than by looking for a particular callable in
``router.dependencies``: the guard moved behind the ``DesktopOnly`` permission
(ENG-2094) with no change to what a caller experiences, and a test that names
the function rather than the refusal fails on the refactor while passing on a
regression that swaps one guard for a weaker one.
"""
from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from cowork.api.v1.endpoints import coding_runtime


@pytest.fixture(autouse=True)
def _reset_app_settings():
    from cowork.common.settings.app_settings import get_app_settings

    get_app_settings.cache_clear()
    yield
    get_app_settings.cache_clear()


def _client() -> TestClient:
    app = FastAPI()
    app.include_router(coding_runtime.router, prefix="/api/v1/coding/runtime")
    return TestClient(app)


def test_runtime_router_is_fail_closed_in_org_mode(monkeypatch) -> None:
    monkeypatch.setenv("COWORK_TENANCY_MODE", "org")

    resp = _client().post("/api/v1/coding/runtime/register", json={})

    assert resp.status_code == 403
    assert resp.json()["detail"] == "not available in org deployments"


def test_runtime_router_is_reachable_on_desktop() -> None:
    # tenancy_mode defaults to "local". Past the tenancy guard the route's own
    # credential check takes over, so anything but a 403 proves the router
    # itself did not refuse.
    resp = _client().post("/api/v1/coding/runtime/register", json={})

    assert resp.status_code != 403
