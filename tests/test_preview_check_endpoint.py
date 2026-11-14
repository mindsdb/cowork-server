"""POST /api/v1/artifacts/checks/{check_id}.

Authorization copies POST /responses/answer (tests/test_ask_user_endpoint.py):
the check's conversation must have a turn the caller may touch, and a foreign
org's check is indistinguishable from an unknown one.
"""

from __future__ import annotations

import asyncio

import pytest
from fastapi.testclient import TestClient

from cowork.api.v1.endpoints.artifact_checks import MAX_ERRORS
from cowork.common.settings.app_settings import get_app_settings
from cowork.server import create_app
from cowork.streaming.preview_checks import preview_checks
from cowork.streaming.registry import RunHandle, registry

CID = "conv-preview-check-endpoint"


@pytest.fixture(autouse=True)
def _clean_globals():
    yield
    preview_checks.reset()
    registry.reset()
    get_app_settings.cache_clear()


async def _register(org_id: str | None = None):
    """An in-flight run for CID plus one open check on it."""

    async def _forever():
        await asyncio.sleep(3600)

    handle = RunHandle(
        conversation_id=CID, turn_id=1, buffer=None, task=asyncio.create_task(_forever()), org_id=org_id,
    )
    registry._by_cid[CID] = handle
    check_id, future = preview_checks.open(CID)
    return handle, check_id, future


@pytest.fixture()
async def run():
    handle, check_id, future = await _register()
    yield check_id, future
    handle.task.cancel()


@pytest.fixture()
def client():
    return TestClient(create_app())


def _post(client, check_id, body):
    return client.post(f"/api/v1/artifacts/checks/{check_id}", json=body)


async def test_a_failed_verdict_reaches_the_waiting_turn(client, run):
    check_id, future = run
    resp = _post(client, check_id, {
        "status": "failed",
        "errors": [
            {"type": "http", "message": "GET /api/data → 500: boom", "file": "", "line": 0, "url": "/api/data"},
            {"type": "error", "message": "TypeError: x", "file": "app.js", "line": 12, "url": "", "col": 3},
        ],
        "ms": 4100,
    })
    assert resp.status_code == 200
    assert resp.json() == {"accepted": True}
    assert future.result() == {
        "status": "failed",
        "errors": [
            {"type": "http", "message": "GET /api/data → 500: boom", "file": "", "line": 0, "url": "/api/data"},
            # Fields outside PreviewDiagnostic (col) are dropped.
            {"type": "error", "message": "TypeError: x", "file": "app.js", "line": 12, "url": ""},
        ],
        "ms": 4100,
    }


async def test_unavailable_with_nothing_else_is_accepted(client, run):
    check_id, future = run
    assert _post(client, check_id, {"status": "unavailable"}).status_code == 200
    assert future.result() == {"status": "unavailable", "errors": [], "ms": 0}


async def test_long_strings_are_cut_not_rejected(client, run):
    """One huge minified error line must not turn a real failure into a 422."""
    check_id, future = run
    resp = _post(client, check_id, {
        "status": "failed",
        "errors": [{"type": "console", "message": "x" * 50_000, "file": None, "url": "u" * 5000}],
    })
    assert resp.status_code == 200
    [error] = future.result()["errors"]
    assert len(error["message"]) == 1000 and len(error["url"]) == 1000 and error["file"] == ""


@pytest.mark.parametrize(
    "body",
    [
        {},
        {"status": "maybe"},
        {"status": "failed", "errors": "boom"},
        {"status": "failed", "errors": [{"type": "error", "message": "x"}] * (MAX_ERRORS + 1)},
        {"status": "failed", "errors": [{"message": "x", "line": -1}]},
        {"status": "passed", "ms": -5},
        {"status": "passed", "ms": 10**9},
    ],
    ids=["empty", "unknown-status", "errors-not-a-list", "too-many-errors", "negative-line", "negative-ms", "huge-ms"],
)
async def test_a_malformed_body_is_422_and_leaves_the_check_open(client, run, body):
    check_id, future = run
    assert _post(client, check_id, body).status_code == 422
    assert not future.done()


async def test_an_over_long_id_is_rejected(client, run):
    assert _post(client, "acheck:" + "a" * 200, {"status": "passed"}).status_code == 422


async def test_unknown_id_is_404(client, run):
    resp = _post(client, "acheck:" + "0" * 32, {"status": "passed"})
    assert resp.status_code == 404
    assert resp.json() == {"status": "not_found"}


async def test_an_expired_check_is_404(client, run):
    check_id, _ = run
    preview_checks.close_conversation(CID)
    assert _post(client, check_id, {"status": "passed"}).status_code == 404


async def test_a_check_whose_turn_is_gone_is_404(client, run):
    check_id, future = run
    registry.reset()
    assert _post(client, check_id, {"status": "passed"}).status_code == 404
    assert not future.done()


async def test_a_second_answer_is_409(client, run):
    check_id, future = run
    assert _post(client, check_id, {"status": "passed"}).status_code == 200
    second = _post(client, check_id, {"status": "failed"})
    assert second.status_code == 409
    assert second.json() == {"accepted": False, "status": "already_answered"}
    assert future.result()["status"] == "passed"


async def test_foreign_org_is_404(monkeypatch):
    """A check id is not an authorization token either."""
    monkeypatch.setenv("COWORK_TENANCY_MODE", "org")
    monkeypatch.setenv("COWORK_IDENTITY_ENFORCE", "audit")
    get_app_settings.cache_clear()
    owner_org = "6ba7b810-9dad-11d1-80b4-00c04fd430c8"
    intruder_org = "0f7f0b6a-3f0f-4c58-9e0c-6dbb3ac0f0a1"
    user_id = "11111111-1111-1111-1111-111111111111"
    handle, check_id, future = await _register(org_id=owner_org)
    try:
        client = TestClient(create_app())
        resp = client.post(
            f"/api/v1/artifacts/checks/{check_id}",
            json={"status": "passed"},
            headers={"X-Organization-Id": intruder_org, "X-User-Id": user_id},
        )
        assert resp.status_code == 404
        assert resp.json() == {"status": "not_found"}
        assert not future.done()
    finally:
        handle.task.cancel()
