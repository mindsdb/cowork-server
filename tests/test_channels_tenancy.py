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
the app's build-time mode does not matter. The principal middleware is only
wired when the process itself started in org mode, so these requests run under
LOCAL_SCOPE and the guard, not the scope resolver, is what refuses them.
"""
from __future__ import annotations

import inspect
import re

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
UNGUARDED = [
    "/api/v1/channels/plugins",
    "/api/v1/channels/installations",
    "/api/v1/channels/bindings",
]


@pytest.fixture
def org_mode(monkeypatch):
    monkeypatch.setenv("COWORK_TENANCY_MODE", "org")
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


@pytest.mark.parametrize("path", UNGUARDED)
def test_unguarded_channel_routes_are_not_refused_in_org_mode(org_mode, path):
    from cowork.server import app

    assert TestClient(app).get(path).status_code != 403


def test_lifecycle_refusals_keep_their_501():
    """A channel plugin that ships no setup or teardown is a real missing
    capability, so those two raises keep their 501 while the tenancy guard moves
    to 403. This is the tripwire for a blanket status sweep over the module."""
    from cowork.api.v1.endpoints import channels

    src = inspect.getsource(channels)
    for action in ("setup", "teardown"):
        pattern = (
            r"HTTP_501_NOT_IMPLEMENTED,\s*\n\s*detail=f\"" + action + r" not implemented for channel"
        )
        assert re.search(pattern, src), f"the {action} lifecycle refusal is no longer a 501"
    # The tenancy guard is the first statement of the helper both lifecycle
    # routes use, so an org-mode caller never reaches either 501.
    assert "_require_local_channels()" in inspect.getsource(channels._lifecycle_service)
