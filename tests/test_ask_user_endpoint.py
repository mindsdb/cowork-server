"""POST /api/v1/responses/answer.

Authorization copies /cancel exactly: same _require_streaming_scope, same
_authorized_handle, same 404 shape — so a foreign-org id is
indistinguishable from an unknown one and cannot leak existence.

The router is mounted at /api/v1 (cowork/api/v1/router.py:56), which is why
every path below carries that prefix.
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

CID = "conv-answer-test"
QID = "ask:1"

_REQUEST = AskRequest(
    prompt="Which database?",
    options=(
        AskOption(value="pg", label="postgres"),
        AskOption(value="my", label="mysql"),
    ),
)


@pytest.fixture(autouse=True)
def _clean_globals():
    """Both the registry and the broker outlive a single test."""
    yield
    broker._pending.clear()
    registry._by_cid.clear()
    get_app_settings.cache_clear()


async def _register(org_id: str | None = None) -> RunHandle:
    """An in-flight run for CID plus one open question on it."""

    async def _forever():
        await asyncio.sleep(3600)

    handle = RunHandle(
        conversation_id=CID,
        turn_id=1,
        buffer=None,
        task=asyncio.create_task(_forever()),
        org_id=org_id,
    )
    registry._by_cid[CID] = handle
    broker.open(CID, QID, _REQUEST)
    return handle


@pytest.fixture()
async def run():
    handle = await _register()
    yield handle
    handle.task.cancel()


@pytest.fixture()
def client():
    return TestClient(create_app())


def _post(client, **body):
    return client.post(
        "/api/v1/responses/answer",
        json={"conversation_id": CID, "question_id": QID, **body},
    )


async def test_accepts_a_valid_selection(client, run):
    resp = _post(client, values=["pg"])
    assert resp.status_code == 200
    assert resp.json() == {"accepted": True}


async def test_accepts_values_and_text_together(client, run):
    assert _post(client, values=["pg", "my"], text="and duckdb").status_code == 200


async def test_accepts_skipped(client, run):
    assert _post(client, skipped=True).status_code == 200


@pytest.mark.parametrize(
    "body,expected_status",
    [
        ({}, "empty_answer"),
        ({"values": []}, "empty_answer"),
        ({"text": "  "}, "empty_answer"),
        ({"skipped": True, "values": ["pg"]}, "ambiguous_answer"),
    ],
    ids=["nothing", "empty-values", "blank-text", "skipped-plus-values"],
)
async def test_rejects_a_malformed_body(client, run, body, expected_status):
    resp = _post(client, **body)
    assert resp.status_code == 400
    assert resp.json() == {"status": expected_status}


async def test_rejects_an_option_that_was_never_offered(client, run):
    resp = _post(client, values=["sqlite"])
    assert resp.status_code == 400
    assert resp.json()["status"] == "invalid_option"


async def test_unknown_question_is_404(client, run):
    resp = client.post(
        "/api/v1/responses/answer",
        json={"conversation_id": CID, "question_id": "ask:nope", "values": ["pg"]},
    )
    assert resp.status_code == 404
    assert resp.json() == {"status": "not_found"}


def test_unknown_conversation_is_404(client):
    resp = client.post(
        "/api/v1/responses/answer",
        json={"conversation_id": "no-such-conv", "question_id": QID, "values": ["pg"]},
    )
    assert resp.status_code == 404
    assert resp.json() == {"status": "not_found"}


async def test_duplicate_answer_is_409(client, run):
    assert _post(client, values=["pg"]).status_code == 200
    second = _post(client, values=["my"])
    assert second.status_code == 409
    assert second.json() == {"accepted": False, "status": "already_answered"}


async def test_foreign_org_is_404(monkeypatch):
    """A conversation_id is not an authorization token."""
    monkeypatch.setenv("COWORK_TENANCY_MODE", "org")
    monkeypatch.setenv("COWORK_IDENTITY_ENFORCE", "audit")
    get_app_settings.cache_clear()
    owner_org = "6ba7b810-9dad-11d1-80b4-00c04fd430c8"
    intruder_org = "0f7f0b6a-3f0f-4c58-9e0c-6dbb3ac0f0a1"
    user_id = "11111111-1111-1111-1111-111111111111"
    handle = await _register(org_id=owner_org)
    try:
        client = TestClient(create_app())
        resp = client.post(
            "/api/v1/responses/answer",
            json={"conversation_id": CID, "question_id": QID, "values": ["pg"]},
            headers={"X-Organization-Id": intruder_org, "X-User-Id": user_id},
        )
        assert resp.status_code == 404
        assert resp.json() == {"status": "not_found"}
    finally:
        handle.task.cancel()
