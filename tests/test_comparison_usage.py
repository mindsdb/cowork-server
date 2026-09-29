"""What each side of a comparison cost: the gateway's calls filed under the side's turns."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import patch
from urllib.parse import urlparse
from uuid import UUID

import httpx
import pytest
from fastapi.testclient import TestClient

from cowork.db.session import get_open_session
from cowork.models.message import Message
from cowork.services import comparison_usage
from cowork.services.comparison_usage import side_usage

T0 = datetime(2026, 9, 29, 12, 0, 0, tzinfo=timezone.utc)


def _call(seconds: float, cost: float | None = 0.01, **tokens) -> dict:
    return {
        "timestamp": (T0 + timedelta(seconds=seconds)).isoformat(),
        "input_tokens": tokens.get("input", 100),
        "output_tokens": tokens.get("output", 10),
        "cached_input_tokens": tokens.get("cached", 0),
        "cache_write_tokens": tokens.get("written", 0),
        "estimated_cost_usd": cost,
    }


def _starts(*seconds: float) -> list[datetime]:
    return [T0 + timedelta(seconds=s) for s in seconds]


def test_each_call_is_filed_under_the_turn_that_started_before_it():
    # A turn's first call usually lands in the same second as its message,
    # which is stored to the second: that call belongs to the new turn.
    payload = {"requests": [_call(0.4, 0.02), _call(20, 0.03), _call(60, 0.5), _call(90, 0.25)]}

    usage = side_usage(_starts(0, 60), payload)

    assert usage.available is True
    assert [t.turn for t in usage.turns] == [1, 2]
    assert [t.estimated_cost_usd for t in usage.turns] == pytest.approx([0.05, 0.75])
    assert usage.estimated_cost_usd == pytest.approx(0.8)
    assert [t.input_tokens for t in usage.turns] == [200, 200]
    assert usage.input_tokens == 400


def test_unpriced_calls_are_counted_and_left_out_of_the_cost():
    payload = {"requests": [_call(1, 0.1), _call(2, None), _call(61, None)]}

    usage = side_usage(_starts(0, 60), payload)

    assert [t.estimated_cost_usd for t in usage.turns] == pytest.approx([0.1, None])
    assert [t.unpriced_calls for t in usage.turns] == [1, 1]
    assert usage.unpriced_calls == 2
    assert usage.estimated_cost_usd == pytest.approx(0.1)
    # Tokens still count even when the price is unknown.
    assert usage.turns[1].input_tokens == 100


def test_a_continued_side_counts_only_the_turns_the_comparison_owns():
    payload = {"requests": [_call(1, 0.1), _call(61, 0.2), _call(120, 4.0)]}

    usage = side_usage(_starts(0, 60, 120), payload, turn_limit=2)

    assert [t.turn for t in usage.turns] == [1, 2]
    assert usage.estimated_cost_usd == pytest.approx(0.3)


def test_naive_timestamps_read_as_utc():
    # SQLite hands message times back without a zone.
    naive = [T0.replace(tzinfo=None), (T0 + timedelta(seconds=60)).replace(tzinfo=None)]

    usage = side_usage(naive, {"requests": [_call(61, 0.2)]})

    assert [t.estimated_cost_usd for t in usage.turns] == pytest.approx([None, 0.2])


@pytest.mark.parametrize(
    ("starts", "payload"),
    [
        pytest.param(_starts(0), None, id="gateway had nothing"),
        pytest.param([], {"requests": [_call(1)]}, id="no turns yet"),
    ],
)
def test_no_usage_reads_as_unavailable(starts, payload):
    assert side_usage(starts, payload).available is False


def test_the_gateway_cap_marks_the_totals_as_a_floor():
    assert side_usage(_starts(0), {"requests": [_call(1)], "truncated": True}).truncated is True


# ── The route ──────────────────────────────────────────────────────────────


@pytest.fixture()
def client():
    from cowork.server import create_app

    return TestClient(create_app())


@pytest.fixture()
def gateway():
    """The gateway, faked at the HTTP transport, recording what it was sent."""
    state = {"sent": [], "respond": lambda request: httpx.Response(404)}

    def handler(request: httpx.Request) -> httpx.Response:
        state["sent"].append(request)
        return state["respond"](request)

    real = httpx.AsyncClient

    def client_factory(**kwargs):
        return real(transport=httpx.MockTransport(handler), **kwargs)

    with patch.object(comparison_usage.httpx, "AsyncClient", side_effect=client_factory):
        yield state


def _comparison(client) -> dict:
    resp = client.post(
        "/api/v1/comparisons/",
        json={"title": "t", "sides": [{"model": "kimi"}, {"model": "qwen"}]},
    )
    assert resp.status_code == 201
    return resp.json()


def _user_turn(conversation_id: str, seq: int, at: datetime, content="task") -> None:
    session = get_open_session()
    try:
        session.add(Message(conversation_id=UUID(conversation_id), role="user", content=content, seq=seq, created_at=at))
        session.commit()
    finally:
        session.close()


def test_reads_each_side_from_this_deployments_gateway_as_the_caller(client, gateway):
    comparison = _comparison(client)
    side_a, side_b = comparison["sides"]
    _user_turn(side_a["conversationId"], 1, T0)
    # A tool result is a user-role row, not a turn.
    _user_turn(side_a["conversationId"], 2, T0 + timedelta(seconds=5),
               content=[{"type": "tool_result", "tool_use_id": "t1", "content": "found"}])
    _user_turn(side_b["conversationId"], 1, T0)

    def respond(request):
        cost = 0.4 if side_a["conversationId"] in request.url.path else 0.1
        return httpx.Response(200, json={"requests": [_call(1, cost), _call(9, cost)]})

    gateway["respond"] = respond

    resp = client.get(
        f"/api/v1/comparisons/{comparison['id']}/usage",
        headers={"X-MindsHub-Authorization": "Bearer caller-jwt"},
    )

    assert resp.status_code == 200
    sides = resp.json()["sides"]
    assert [len(sides["a"]["turns"]), len(sides["b"]["turns"])] == [1, 1]
    assert sides["a"]["estimatedCostUsd"] == pytest.approx(0.8)
    assert sides["b"]["estimatedCostUsd"] == pytest.approx(0.2)
    from cowork.common.settings.app_settings import default_turn_minds_api_host

    host = urlparse(default_turn_minds_api_host()).hostname
    assert {r.url.host for r in gateway["sent"]} == {host}
    assert {r.url.path for r in gateway["sent"]} == {
        f"/v1/usage/sessions/{side_a['conversationId']}",
        f"/v1/usage/sessions/{side_b['conversationId']}",
    }
    assert {r.headers["authorization"] for r in gateway["sent"]} == {"Bearer caller-jwt"}


@pytest.mark.parametrize(
    "respond",
    [
        pytest.param(lambda r: httpx.Response(404, json={"detail": "Not Found"}), id="gateway without the route"),
        pytest.param(lambda r: httpx.Response(500), id="gateway error"),
        pytest.param(lambda r: httpx.Response(200, text="not json"), id="not json"),
        pytest.param(
            lambda r: httpx.Response(302, headers={"location": "https://elsewhere.example/v1/usage"}),
            id="redirect",
        ),
    ],
)
def test_a_failed_read_hides_the_figure_without_failing(client, gateway, respond):
    comparison = _comparison(client)
    for side in comparison["sides"]:
        _user_turn(side["conversationId"], 1, T0)
    gateway["respond"] = respond

    resp = client.get(
        f"/api/v1/comparisons/{comparison['id']}/usage",
        headers={"X-MindsHub-Authorization": "Bearer caller-jwt"},
    )

    assert resp.status_code == 200
    assert [s["available"] for s in resp.json()["sides"].values()] == [False, False]
    # A redirect is not followed, so the bearer never reaches another host.
    assert {r.url.host for r in gateway["sent"]} != {"elsewhere.example"}
    assert len(gateway["sent"]) == 2


def test_without_a_hub_credential_nothing_is_sent(client, gateway):
    comparison = _comparison(client)
    for side in comparison["sides"]:
        _user_turn(side["conversationId"], 1, T0)

    resp = client.get(f"/api/v1/comparisons/{comparison['id']}/usage")

    assert resp.status_code == 200
    assert [s["available"] for s in resp.json()["sides"].values()] == [False, False]
    assert gateway["sent"] == []


def test_an_unknown_comparison_is_not_found(client, gateway):
    assert client.get("/api/v1/comparisons/00000000-0000-0000-0000-000000000000/usage").status_code == 404
