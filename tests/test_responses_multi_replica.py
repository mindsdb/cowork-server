"""Endpoints must answer for a turn this replica did not start.

With two replicas behind a load balancer, a reconnect or a stop lands on
whichever one the balancer picks. Before this, the replica that did not start
the turn had no handle and no buffer, so it reported the turn as missing.
"""

from __future__ import annotations

import fakeredis.aioredis
import pytest
from fastapi.testclient import TestClient

from cowork.server import create_app
from cowork.streaming import buffer as buffer_mod
from cowork.streaming import turn_index
from cowork.streaming.buffer import RedisStreamBuffer
from cowork.streaming.registry import registry


@pytest.fixture
def fake_redis(monkeypatch):
    """One Redis, no local handles: this replica is the one that did not start
    the turn."""
    client = fakeredis.aioredis.FakeRedis(decode_responses=True)
    for mod in (turn_index, buffer_mod):
        monkeypatch.setattr(mod, "get_redis", lambda: client)
    monkeypatch.setattr(
        "cowork.api.v1.endpoints.responses.get_redis", lambda: client)
    monkeypatch.setenv("COWORK_STREAM_BACKEND", "redis")
    registry.reset()
    yield client
    registry.reset()


@pytest.fixture
def client():
    return TestClient(create_app())


def _sse(text: str) -> dict:
    """The record shape producers write: readers emit data["sse"] verbatim."""
    return {"sse": f"event: response.output_text.delta\ndata: {{\"delta\": \"{text}\"}}\n\n"}


async def _running_turn(conversation_id: str, turn_id: int, correlation_id: str):
    await turn_index.record_turn(
        conversation_id, turn_id=turn_id, correlation_id=correlation_id,
        org_id=None, user_id=None,
    )
    buf = RedisStreamBuffer(conversation_id=conversation_id, turn_id=turn_id)
    await buf.append("sse", _sse("hi"))
    return buf


@pytest.mark.asyncio
async def test_in_flight_reports_a_turn_started_elsewhere(client, fake_redis):
    """Answering False would make the UI think the turn had finished."""
    await _running_turn("c1", 4, "corr-1")

    body = client.get("/api/v1/responses/in-flight?conversation_id=c1").json()

    assert body["in_flight"] is True
    assert body["has_buffer"] is True
    assert body["latest_seq"] == 1
    assert body["turn_id"] == 4


@pytest.mark.asyncio
async def test_in_flight_is_false_once_the_buffer_is_closed(client, fake_redis):
    """Liveness comes from the terminal record, not from a process."""
    buf = await _running_turn("c2", 1, "corr-2")
    await buf.close("completed")

    body = client.get("/api/v1/responses/in-flight?conversation_id=c2").json()

    assert body["in_flight"] is False
    assert body["has_buffer"] is True


@pytest.mark.asyncio
async def test_in_flight_list_includes_turns_started_elsewhere(client, fake_redis):
    await _running_turn("c3", 1, "corr-3")

    body = client.get("/api/v1/responses/in-flight-list").json()

    assert [row["conversation_id"] for row in body["in_flight"]] == ["c3"]


@pytest.mark.asyncio
async def test_in_flight_list_skips_finished_turns(client, fake_redis):
    buf = await _running_turn("c4", 1, "corr-4")
    await buf.close("completed")

    body = client.get("/api/v1/responses/in-flight-list").json()

    assert body["in_flight"] == []


@pytest.mark.asyncio
async def test_tail_streams_a_turn_started_elsewhere(client, fake_redis):
    buf = await _running_turn("c5", 1, "corr-5")
    await buf.append("sse", _sse("hello-from-elsewhere"))
    await buf.close("completed")

    with client.stream("GET", "/api/v1/responses/tail?conversation_id=c5") as resp:
        assert resp.status_code == 200
        body = "".join(resp.iter_text())

    assert "hello-from-elsewhere" in body


@pytest.mark.asyncio
async def test_cancel_sets_the_flag_the_controller_reads(client, fake_redis):
    """The pod is where the tokens are spent, and only the flag reaches it."""
    await _running_turn("c6", 1, "corr-6")

    body = client.post(
        "/api/v1/responses/cancel", json={"conversation_id": "c6"}).json()

    assert body["cancelled"] is True
    assert await fake_redis.exists("cowork:cancel:corr-6") == 1


@pytest.mark.asyncio
async def test_cancel_404s_when_no_turn_is_recorded(client, fake_redis):
    resp = client.post("/api/v1/responses/cancel", json={"conversation_id": "ghost"})
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_cancel_404s_for_a_finished_turn(client, fake_redis):
    buf = await _running_turn("c7", 1, "corr-7")
    await buf.close("completed")

    resp = client.post("/api/v1/responses/cancel", json={"conversation_id": "c7"})

    assert resp.status_code == 404
    assert await fake_redis.exists("cowork:cancel:corr-7") == 0


@pytest.mark.asyncio
async def test_a_just_enqueued_turn_reads_as_running(client, fake_redis):
    """The index entry is written when the job is enqueued; the first record
    only lands once the pod answers. An empty stream in between means "not
    started yet", not "finished"."""
    await turn_index.record_turn(
        "c8", turn_id=1, correlation_id="corr-8", org_id=None, user_id=None)

    body = client.get("/api/v1/responses/in-flight?conversation_id=c8").json()

    assert body["in_flight"] is True
    assert body["latest_seq"] == 0


@pytest.mark.asyncio
async def test_an_old_entry_with_no_stream_reads_as_finished(client, fake_redis):
    """Past the grace period an empty stream is a truncated conversation, whose
    buffers were deleted while the index entry lived on."""
    await turn_index.record_turn(
        "c9", turn_id=1, correlation_id="corr-9", org_id=None, user_id=None)
    await fake_redis.hset("cowork:turn:c9", "started_at", "1")   # 1970

    body = client.get("/api/v1/responses/in-flight?conversation_id=c9").json()

    assert body["in_flight"] is False


@pytest.mark.asyncio
async def test_cancel_works_on_a_turn_that_has_not_spoken_yet(client, fake_redis):
    """Stop must work while the pod is still starting, which is exactly when a
    user is most likely to press it."""
    await turn_index.record_turn(
        "c10", turn_id=1, correlation_id="corr-10", org_id=None, user_id=None)

    body = client.post(
        "/api/v1/responses/cancel", json={"conversation_id": "c10"}).json()

    assert body["cancelled"] is True
    assert await fake_redis.exists("cowork:cancel:corr-10") == 1


@pytest.mark.asyncio
async def test_a_stall_cancel_writes_its_cause_before_the_flag(client, fake_redis, monkeypatch):
    """The controller deletes the flag as soon as it reports the cancel, so
    the cause has to be in Redis before the flag can start that clock."""
    await _running_turn("c11", 1, "corr-11")
    writes: list[str] = []
    real_set = fake_redis.set

    async def recording_set(key, value, *args, **kwargs):
        writes.append(key)
        return await real_set(key, value, *args, **kwargs)

    monkeypatch.setattr(fake_redis, "set", recording_set)

    body = client.post(
        "/api/v1/responses/cancel",
        json={"conversation_id": "c11", "reason": "stalled"},
    ).json()

    assert body["cancelled"] is True
    assert writes == ["cowork:cancel_cause:corr-11", "cowork:cancel:corr-11"]
    assert await fake_redis.get("cowork:cancel_cause:corr-11") == "stalled"
    assert 0 < await fake_redis.ttl("cowork:cancel_cause:corr-11") <= 300


@pytest.mark.asyncio
@pytest.mark.parametrize("reason", [None, "user", "something-newer"])
async def test_only_a_stall_writes_a_cancel_cause(client, fake_redis, reason):
    """A Stop, and any reason this server does not know, cancels exactly as
    before and leaves no cause behind."""
    await _running_turn("c12", 1, "corr-12")
    payload = {"conversation_id": "c12"} if reason is None else {
        "conversation_id": "c12", "reason": reason,
    }

    resp = client.post("/api/v1/responses/cancel", json=payload)

    assert resp.status_code == 200
    assert await fake_redis.exists("cowork:cancel:corr-12") == 1
    assert await fake_redis.exists("cowork:cancel_cause:corr-12") == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("stalled", [True, False])
async def test_the_owner_saves_a_stall_another_replica_reported(monkeypatch, fake_redis, stalled):
    """The replica that owns the turn learns of the cancel only from the
    controller's turn_failed, after the controller deleted the flag. The cause
    key is what tells it a stall from a Stop."""
    import cowork.handlers.responses as responses_mod
    from test_responses_remote_backend import _RecBuffer, _remote_handler

    monkeypatch.setattr(responses_mod, "get_redis", lambda: fake_redis)
    if stalled:
        await fake_redis.set("cowork:cancel_cause:corr-13", "stalled", ex=300)
    saved: dict = {}
    handler = _remote_handler(monkeypatch, saved)

    async def replies(**kwargs):
        yield "progress", {"phase": "workspace_authorized", "workspace_mode": "persistent"}
        yield "turn_delta", {"text": "partial"}
        yield "turn_failed", {"error": "cancelled"}

    monkeypatch.setattr(responses_mod, "stream_remote_replies", replies)
    buffer = _RecBuffer()

    await handler._produce_remote(
        conv_id="c13", input_text="hi", original_content="hi", model="anton",
        harness_id="anton", buffer=buffer, turn_llm={"correlation_id": "corr-13"},
    )

    assert saved["assistant"] == "partial"
    if stalled:
        assert buffer.frames[-1] == "CLOSE:interrupted"
        failed = [f for f in buffer.frames if f.startswith("event: response.failed")]
        assert len(failed) == 1 and '"code": "stalled"' in failed[0]
        assert saved["events"][-1]["code"] == "stalled"
        assert saved["events"][-1]["request_id"] == "corr-13"
    else:
        # A Stop: partial answer, no error row, closed cancelled.
        assert buffer.frames[-1] == "CLOSE:cancelled"
        assert not any("response.failed" in f for f in buffer.frames)
        assert not any(e.get("type") == "response.failed" for e in saved["events"])
