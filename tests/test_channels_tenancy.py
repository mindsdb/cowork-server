"""The channel config surface refuses org mode, and says so with a 403.

`_require_local_channels` guards every route that reads or writes channel
credentials, the shared adapter, or the harness setting. Those live in
deployment-global rows, so in org mode one tenant could otherwise read or
delete another's. The runtime side already fails closed; the guard closes the
config side.

The refusal is a 403 rather than a 501 because it turns a caller away at a
tenant boundary instead of admitting a missing capability. The two lifecycle
501s in the same module are the other kind, and the last test here keeps them
apart.

TestClient is enough: the guard reads the tenancy setting at request time, so
the app's build-time mode does not matter. The scope resolver reads it at
request time too, so these requests carry an org scope with no org id rather
than LOCAL_SCOPE. The guard still refuses first on every guarded route, because
it is the first statement of each handler body and building a ScopedSession
touches no table. Move a guard below a `scoped.select(...)` and that route
answers 401 from the scope layer instead.
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from cowork.common.settings.app_settings import get_app_settings

DETAIL = "channels are not available in org deployments yet"

# Every route behind `_require_local_channels`. Seven call the guard directly;
# setup and teardown reach it through `_lifecycle_service`, which is why a grep
# over the route decorators alone comes up two short.
GUARDED = [
    ("GET", "/api/v1/channels/status", None),
    ("GET", "/api/v1/channels/agent", None),
    ("PUT", "/api/v1/channels/agent", {"harness": "anton"}),
    ("GET", "/api/v1/channels/slack/config", None),
    ("PUT", "/api/v1/channels/slack/config", {"values": {}}),
    ("DELETE", "/api/v1/channels/slack/config", None),
    ("POST", "/api/v1/channels/slack/reload", None),
    ("POST", "/api/v1/channels/slack/setup", None),
    ("POST", "/api/v1/channels/slack/teardown", None),
]

# The catalogue and bindings routes sit outside the guard on purpose: they carry
# a tenant scope of their own and expose no credentials. They are the negative
# case, so the guard cannot quietly grow to cover the whole router.
#
# Their statuses are pinned rather than asserted "not 403". A TestClient request
# carries no gateway identity headers, so the fail-closed scope layer refuses the
# two org-scoped routes with a 401; "not 403" would pass on that refusal and read
# as though those routes were reachable.
UNGUARDED = [
    ("/api/v1/channels/plugins", 200),  # catalogue only, no tenant data
    ("/api/v1/channels/installations", 401),
    ("/api/v1/channels/bindings", 401),
]


@pytest.fixture
def org_mode(monkeypatch):
    monkeypatch.setenv("COWORK_TENANCY_MODE", "org")
    get_app_settings.cache_clear()
    yield
    get_app_settings.cache_clear()


@pytest.fixture
def local_mode(monkeypatch):
    monkeypatch.delenv("COWORK_TENANCY_MODE", raising=False)
    get_app_settings.cache_clear()
    yield
    get_app_settings.cache_clear()


def _call(method: str, path: str, body):
    from cowork.server import app

    kwargs = {"json": body} if body is not None else {}
    return TestClient(app).request(method, path, **kwargs)


@pytest.mark.parametrize("method,path,body", GUARDED)
def test_channel_config_routes_are_403_in_org_mode(org_mode, method, path, body):
    res = _call(method, path, body)
    assert res.status_code == 403
    # The detail is load-bearing: it tells this refusal apart from the other
    # 403s the app raises, and it tells the caller why.
    assert res.json()["detail"] == DETAIL


@pytest.mark.parametrize("path,expected", UNGUARDED)
def test_unguarded_channel_routes_are_not_refused_by_the_guard(org_mode, path, expected):
    from cowork.server import app

    assert TestClient(app).get(path).status_code == expected


@pytest.mark.parametrize("action", ["setup", "teardown"])
def test_lifecycle_refusals_keep_their_501(local_mode, action):
    """A channel plugin that ships no setup or teardown is a real missing
    capability, so those two raises keep their 501 while the tenancy guard moves
    to 403. `slack` ships no lifecycle, so it is the live case; a blanket status
    sweep over the module turns these into 403 and fails here.

    Builds its own app instead of importing the module-level one: whichever test
    imports `cowork.server` first fixes the app's build-time tenancy mode for the
    session, and an app built in org mode loads no plugins, which would answer
    404 here rather than 501.
    """
    from cowork.server import create_app

    res = TestClient(create_app()).post(f"/api/v1/channels/slack/{action}")
    assert res.status_code == 501
    assert res.json()["detail"] == f"{action} not implemented for channel: slack"
