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
