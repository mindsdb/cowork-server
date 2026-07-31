"""`_require_streaming_scope` on the streaming endpoints.

In org mode with `COWORK_IDENTITY_ENFORCE=audit`, a request that carries no
identity headers reaches the endpoint with `scope.org_mode=True` and
`scope.org_id=None`. `_authorized_handle` compares `handle.org_id ==
scope.org_id`, so `None == None` **succeeds** — an identity-less caller would
otherwise match every handle a run started without a principal registered
(which is exactly what audit mode produces). `_require_streaming_scope` is the
only thing standing between such a caller and a real blocked agent's question,
so each endpoint that touches the registry is pinned here.

Mutation proof (recorded 2026-07-31): deleting the
`raise MissingTenantScopeError(...)` body of `_require_streaming_scope`
(cowork/api/v1/endpoints/responses.py) turns every case below from 401 into
200/404 — /answer returns 200 {"accepted": true} on somebody else's question.
`tests/test_ask_user_endpoint.py::test_foreign_org_is_404` does NOT catch it:
it sends real identity headers, so `scope.org_id` is populated and the guard
is a no-op there.
"""

from __future__ import annotations

import asyncio

import pytest
from fastapi.testclient import TestClient

from anton.core.interaction.elicit import AskOption, AskRequest
from cowork.common.settings.app_settings import get_app_settings
from cowork.server import create_app
from cowork.streaming.answers import broker
from cowork.streaming.registry import RunHandle, registry

CID = "conv-scope-guard"
QID = "ask:1"

_REQUEST = AskRequest(
    prompt="Which database?",
    options=(
        AskOption(value="pg", label="postgres"),
        AskOption(value="my", label="mysql"),
    ),
)


@pytest.fixture(autouse=True)
def _org_mode_without_identity(monkeypatch):
    """Org mode, audit enforcement, and no identity headers on the request."""
    monkeypatch.setenv("COWORK_TENANCY_MODE", "org")
    monkeypatch.setenv("COWORK_IDENTITY_ENFORCE", "audit")
    get_app_settings.cache_clear()
    yield
    broker.reset()
    registry.reset()
    get_app_settings.cache_clear()


@pytest.fixture()
async def orphan_run():
    """A handle with `org_id=None` — a run started without a principal, the
    shape an identity-less caller would collide with."""

    async def _forever():
        await asyncio.sleep(3600)

    handle = RunHandle(
        conversation_id=CID,
        turn_id=1,
        buffer=None,
        task=asyncio.create_task(_forever()),
        org_id=None,
    )
    registry._by_cid[CID] = handle
    broker.open(CID, QID, _REQUEST)
    yield handle
    handle.task.cancel()


@pytest.fixture()
def client():
    return TestClient(create_app())


async def test_answer_without_identity_is_401(client, orphan_run):
    resp = client.post(
        "/api/v1/responses/answer",
        json={"conversation_id": CID, "question_id": QID, "values": ["pg"]},
    )
    assert resp.status_code == 401
    # The answer must not have reached the blocked agent.
    assert not broker._pending[(CID, QID)].future.done()


async def test_cancel_without_identity_is_401(client, orphan_run):
    resp = client.post("/api/v1/responses/cancel", json={"conversation_id": CID})
    assert resp.status_code == 401
    assert orphan_run.is_running


async def test_tail_without_identity_is_401(client, orphan_run):
    resp = client.get("/api/v1/responses/tail", params={"conversation_id": CID})
    assert resp.status_code == 401


async def test_in_flight_without_identity_is_401(client, orphan_run):
    resp = client.get("/api/v1/responses/in-flight", params={"conversation_id": CID})
    assert resp.status_code == 401


async def test_in_flight_list_without_identity_is_401(client, orphan_run):
    resp = client.get("/api/v1/responses/in-flight-list")
    assert resp.status_code == 401
