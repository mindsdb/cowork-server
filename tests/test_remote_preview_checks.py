"""Preview checks for remote (pod) turns (ENG-4227): the pod's artifact_check
step becomes the app's response.artifact_check frame, and the app's answer
reaches the pod's stdin as a check_result entry on the answer queue."""
from __future__ import annotations

import json

import fakeredis.aioredis
import pytest
from fastapi.testclient import TestClient

from cowork.common.settings.app_settings import get_app_settings
from cowork.turnqueue import preview_checks as remote
from cowork.turnqueue.answers import fits_stdin_line
from cowork.turnqueue.redis_client import answer_queue_key

OWNER_ORG = "6ba7b810-9dad-11d1-80b4-00c04fd430c8"
OWNER = "11111111-1111-1111-1111-111111111111"
CHECK = "acheck:" + "ab" * 12  # anton's cloud_turn mints 12 random bytes
STEP = {"step": "artifact_check", "check_id": CHECK, "artifact_slug": "sales", "timeout_s": 20}


@pytest.fixture
def redis_client(monkeypatch):
    client = fakeredis.aioredis.FakeRedis(decode_responses=True)
    monkeypatch.setattr(remote, "get_redis", lambda: client)
    return client


async def _register(redis_client, data=STEP):
    return await remote.register_remote_check(
        data, conversation_id="conv-1", correlation_id="corr-1", org_id=OWNER_ORG, user_id=OWNER, r=redis_client,
    )


async def test_a_pod_s_check_is_recorded_with_the_turn_s_owner(redis_client):
    check = await _register(redis_client)
    assert (check["check_id"], check["artifact_slug"], check["timeout_s"]) == (CHECK, "sales", 20)
    loaded = await remote.load_remote_check(CHECK, r=redis_client)
    assert loaded["owner"].org_id == OWNER_ORG and loaded["owner"].user_id == OWNER
    assert loaded["correlation_id"] == "corr-1"
    # The same id twice is never relayed twice.
    assert await _register(redis_client) is None


@pytest.mark.parametrize(
    "bad",
    [
        {**STEP, "check_id": "acheck:../../x"},
        {**STEP, "check_id": "not-a-check"},
        {**STEP, "artifact_slug": ""},
        {**STEP, "artifact_slug": "s" * 201},
    ],
)
async def test_a_malformed_step_is_dropped(redis_client, bad):
    assert await _register(redis_client, bad) is None


async def test_the_timeout_is_bounded(redis_client):
    check = await _register(redis_client, {**STEP, "timeout_s": 10_000})
    assert check["timeout_s"] == 120


def test_the_formatter_writes_the_in_process_frame_and_never_records_it():
    import asyncio

    from cowork.harnesses.anton_harness.stream_formatter import PreviewCheckRequest, format_responses_stream

    recorded = []

    async def events():
        yield PreviewCheckRequest(check_id=CHECK, artifact_slug="sales", timeout_s=20)

    async def run():
        return [f async for f in format_responses_stream(events(), "m", lambda t, d: recorded.append(t))]

    frames = asyncio.run(run())
    frame = next(f for f in frames if "response.artifact_check" in f)
    assert frame.startswith("event: response.artifact_check\n")
    payload = json.loads(frame.split("data: ", 1)[1])
    assert payload == {"type": "response.artifact_check", "check_id": CHECK, "artifact_slug": "sales", "timeout_s": 20}
    assert "response.artifact_check" not in recorded


def test_a_large_answer_sheds_diagnostics_to_fit_the_pod_s_line():
    errors = [{"type": "error", "message": "x" * 1000, "file": "f" * 500, "line": 1, "url": "u" * 1000}] * 50
    entry = remote.check_result_entry(CHECK, {"status": "failed", "errors": errors, "ms": 9})
    assert fits_stdin_line(entry)
    parsed = json.loads(entry)
    assert parsed["kind"] == "check_result" and parsed["body"]["status"] == "failed"
    assert 0 < len(parsed["body"]["errors"]) < 50
    assert json.loads(remote.check_result_entry(CHECK, None)) == {"kind": "check_result", "check_id": CHECK, "body": None}


@pytest.fixture
def remote_app(monkeypatch, redis_client):
    from cowork.server import create_app

    monkeypatch.setenv("COWORK_TENANCY_MODE", "org")
    monkeypatch.setenv("COWORK_IDENTITY_ENFORCE", "audit")
    monkeypatch.setenv("COWORK_TURN_BACKEND", "remote")
    get_app_settings.cache_clear()
    yield TestClient(create_app())
    get_app_settings.cache_clear()


def _post(client, user=OWNER, org=OWNER_ORG, body=None):
    return client.post(
        f"/api/v1/artifacts/checks/{CHECK}",
        json=body or {"status": "failed", "errors": [{"type": "error", "message": "boom"}], "ms": 12},
        headers={"X-Organization-Id": org, "X-User-Id": user},
    )


async def test_the_app_s_answer_reaches_the_pod_s_stdin_queue(remote_app, redis_client):
    await _register(redis_client)
    response = _post(remote_app)
    assert response.status_code == 200, response.text
    assert response.json() == {"accepted": True}
    queued = await redis_client.lrange(answer_queue_key("corr-1"), 0, -1)
    assert len(queued) == 1
    entry = json.loads(queued[0])
    assert entry["kind"] == "check_result" and entry["check_id"] == CHECK
    assert entry["body"]["status"] == "failed" and entry["body"]["errors"][0]["message"] == "boom"
    # Once only, as for a turn's check in process.
    again = _post(remote_app)
    assert again.status_code == 409 and again.json() == {"accepted": False, "status": "already_answered"}


@pytest.mark.parametrize(
    "org,user",
    [(OWNER_ORG, "22222222-2222-2222-2222-222222222222"), ("0f7f0b6a-3f0f-4c58-9e0c-6dbb3ac0f0a1", OWNER)],
)
async def test_only_the_turn_s_owner_may_answer(remote_app, redis_client, org, user):
    await _register(redis_client)
    response = _post(remote_app, user=user, org=org)
    assert response.status_code == 404 and response.json() == {"status": "not_found"}
    assert await redis_client.llen(answer_queue_key("corr-1")) == 0


async def test_an_unknown_or_expired_check_is_404(remote_app, redis_client):
    assert _post(remote_app).status_code == 404


async def test_a_check_with_no_preview_reaches_the_pod_as_unavailable(remote_app, redis_client):
    await _register(redis_client)
    assert _post(remote_app, body={"status": "unavailable"}).status_code == 200
    entry = json.loads((await redis_client.lrange(answer_queue_key("corr-1"), 0, -1))[0])
    assert entry["body"]["status"] == "unavailable"
