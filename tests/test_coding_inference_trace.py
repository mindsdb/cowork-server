from __future__ import annotations

import json

import httpx
import pytest
from starlette.requests import Request

from cowork.coding import inference_proxy as inference_proxy_module
from cowork.coding import inference_trace
from cowork.coding.engines.base import EngineCredentials
from cowork.coding.inference_proxy import proxy_inference
from cowork.coding.inference_trace import TurnOrdinals, trace_headers

THREAD = "01a0cb52-cefa-7f01-a0d9-ad665320f62d"
TURN_1 = "01a0cb52-cf12-7541-9cfd-c20a0bd4f3d2"
TURN_2 = "01a0cb53-0000-7000-8000-000000000002"


def _request(headers: dict[str, str], body: bytes = b"") -> Request:
    delivered = False

    async def receive() -> dict[str, object]:
        nonlocal delivered
        if delivered:
            return {"type": "http.request", "body": b"", "more_body": False}
        delivered = True
        return {"type": "http.request", "body": body, "more_body": False}

    return Request(
        {
            "type": "http",
            "method": "POST",
            "path": "/",
            "headers": [(k.encode(), v.encode()) for k, v in headers.items()],
        },
        receive=receive,
    )


def _codex_headers(turn_id: str | None = TURN_1, thread_id: str = THREAD) -> dict[str, str]:
    # The shape Codex 0.147.0 sends on every /responses call.
    headers = {"thread-id": thread_id, "session-id": thread_id}
    if turn_id is not None:
        headers["x-codex-turn-metadata"] = json.dumps(
            {"session_id": thread_id, "thread_id": thread_id, "turn_id": turn_id, "request_kind": "turn"}
        )
    return headers


@pytest.fixture(autouse=True)
def _desktop(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(inference_trace, "surface", lambda: "desktop")


def test_codex_identity_becomes_a_labelled_langfuse_turn() -> None:
    headers = trace_headers(_request(_codex_headers()), TurnOrdinals())

    assert headers["Langfuse-Session-Id"] == THREAD
    assert headers["Langfuse-Tags"] == "codex,surface:desktop"
    assert json.loads(headers["Langfuse-Metadata"]) == {"harness": "codex", "surface": "desktop", "turn_id": 1}


def test_requests_in_one_turn_share_a_number_and_the_next_turn_takes_the_next() -> None:
    turns = TurnOrdinals()

    def turn_of(turn_id: str, thread_id: str = THREAD) -> int:
        return json.loads(trace_headers(_request(_codex_headers(turn_id, thread_id)), turns)["Langfuse-Metadata"])["turn_id"]

    assert turn_of(TURN_1) == 1
    assert turn_of(TURN_1) == 1  # a retry or compaction inside the same turn
    assert turn_of(TURN_2) == 2
    assert turn_of(TURN_1, thread_id="other-thread") == 1


def test_turn_ordinals_stay_bounded() -> None:
    turns = TurnOrdinals(max_threads=2, turns_per_thread=2)

    assert [turns.ordinal("a", t) for t in ("t1", "t2", "t3")] == [1, 2, 3]
    turns.ordinal("b", "t1")
    turns.ordinal("c", "t1")

    # "a" was evicted as the least recently used thread, so it restarts.
    assert turns.ordinal("a", "t9") == 1


def test_a_request_without_codex_identity_gets_no_trace_headers() -> None:
    assert trace_headers(_request({"content-type": "application/json"}), TurnOrdinals()) == {}


@pytest.mark.parametrize(
    "metadata",
    ["not-json", "[1, 2]", json.dumps({"turn_id": 7}), json.dumps({"turn_id": "bad id!"}), "x" * 5000],
)
def test_unreadable_turn_metadata_keeps_the_session_but_drops_the_turn(metadata: str) -> None:
    headers = trace_headers(_request({"thread-id": THREAD, "x-codex-turn-metadata": metadata}), TurnOrdinals())

    assert headers["Langfuse-Session-Id"] == THREAD
    assert "turn_id" not in json.loads(headers["Langfuse-Metadata"])


def test_a_malformed_thread_id_is_not_forwarded() -> None:
    assert trace_headers(_request(_codex_headers(thread_id="x" * 200)), TurnOrdinals()) == {}
    assert trace_headers(_request(_codex_headers(thread_id="a\tb")), TurnOrdinals()) == {}


def test_session_id_is_used_when_thread_id_is_absent() -> None:
    headers = trace_headers(_request({"session-id": THREAD}), TurnOrdinals())

    assert headers["Langfuse-Session-Id"] == THREAD


def test_an_unknown_surface_is_left_out(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(inference_trace, "surface", lambda: None)

    headers = trace_headers(_request(_codex_headers()), TurnOrdinals())

    assert headers["Langfuse-Tags"] == "codex"
    assert "surface" not in json.loads(headers["Langfuse-Metadata"])


@pytest.mark.asyncio
async def test_proxy_forwards_the_trace_headers_upstream(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, str] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen.update(request.headers)
        return httpx.Response(200, content=b"data: [DONE]\n\n", headers={"content-type": "text/event-stream"})

    real_client = httpx.AsyncClient
    monkeypatch.setattr(
        inference_proxy_module.httpx,
        "AsyncClient",
        lambda **kwargs: real_client(transport=httpx.MockTransport(handler), **kwargs),
    )
    request = _request({"content-type": "application/json", **_codex_headers()}, b'{"model": "gpt"}')

    response = await proxy_inference(request, "responses", EngineCredentials(minds_url="https://api.example", minds_api_key="mdb_key"))
    await response.body_iterator.aclose()

    assert seen["authorization"] == "Bearer mdb_key"
    assert seen["langfuse-session-id"] == THREAD
    assert json.loads(seen["langfuse-metadata"])["harness"] == "codex"
    # Codex's own transport headers still stop at the proxy.
    assert "x-codex-turn-metadata" not in seen
    assert "thread-id" not in seen
