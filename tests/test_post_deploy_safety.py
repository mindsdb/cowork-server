"""Regression tests for post-deploy test helpers that must fail safely."""

import json
from collections.abc import Callable, Iterator

import httpx
import pytest

from cowork.handlers.turn_errors import response_failed_sse
from cowork.schemas.responses import (
    IncompleteDetails,
    Response,
    ResponseOutput,
    ResponseOutputContent,
    ResponseStatus,
)
from cowork.streaming.sse import sse_frame
from tests.integration import test_post_deploy

_CONVERSATION_ID = "conversation-1"
_USER_MESSAGE_ID = "6f0f1b8e-0000-4000-8000-000000000001"
_ASSISTANT_MESSAGE_ID = "6f0f1b8e-0000-4000-8000-000000000002"
_ANSWER = "pong"
_KEEPALIVE = ": keepalive\n\n"


class _Response:
    def __init__(self, payload: dict) -> None:
        self.payload = payload

    def json(self) -> dict:
        return self.payload


class _Peer:
    def __init__(self, probes: list[dict]) -> None:
        self.probes = probes
        self.calls = 0

    def get(self, path: str, *, params: dict) -> _Response:
        assert path == "/api/v1/responses/in-flight"
        assert params == {"conversation_id": "conversation-1"}
        payload = self.probes[min(self.calls, len(self.probes) - 1)]
        self.calls += 1
        return _Response(payload)


def test_identity_repr_never_renders_the_api_key():
    identity = test_post_deploy._Identity({
        "api_key": "mdb_do-not-print-me",
        "user_id": "user-1",
        "organization_id": "org-1",
    })

    rendered = repr(identity)

    assert "mdb_do-not-print-me" not in rendered
    assert "'api_key': '***'" in rendered
    assert "user-1" in rendered


@pytest.mark.parametrize("event", ["response.failed", "error"])
def test_failed_turn_assertion_includes_the_server_explanation(event):
    message = "Your current permissions could not be verified. Try again."
    code = "permission_unavailable"
    if event == "response.failed":
        failure = response_failed_sse(message, code)
    else:
        payload = json.dumps({"code": code, "message": message})
        failure = f"event: error\ndata: {payload}\n\n"
    body = "event: response.created\ndata: {}\n\n" + failure
    transport = httpx.MockTransport(
        lambda _request: httpx.Response(
            200, text=body, headers={"Content-Type": "text/event-stream"},
        )
    )

    with httpx.Client(base_url="https://cowork.example", transport=transport) as api:
        with pytest.raises(AssertionError, match="turn did not complete") as error:
            test_post_deploy.test_a_turn_runs_end_to_end(api, "conversation-1")

    assert f"code={code!r} message={message!r}" in str(error.value)


def test_peer_probe_retries_until_the_shared_index_is_visible(monkeypatch):
    peer = _Peer([
        {"has_buffer": False},
        {"has_buffer": False},
        {"has_buffer": True, "latest_seq": 1},
    ])
    monkeypatch.setattr(test_post_deploy.time, "sleep", lambda _seconds: None)

    probe = test_post_deploy._await_shared_buffer(peer, "conversation-1")

    assert probe == {"has_buffer": True, "latest_seq": 1}
    assert peer.calls == 3


def test_peer_probe_returns_the_last_response_at_timeout(monkeypatch):
    peer = _Peer([{"has_buffer": False, "latest_seq": 0}])
    monkeypatch.setattr(test_post_deploy.time, "sleep", lambda _seconds: None)

    probe = test_post_deploy._await_shared_buffer(
        peer, "conversation-1", timeout_s=-1.0,
    )

    assert probe == {"has_buffer": False, "latest_seq": 0}
    assert peer.calls == 1


def _created_frame() -> str:
    """The `response.created` frame as the streaming handler injects it."""
    return sse_frame("response.created", {
        "type": "response.created",
        "sequence_number": 1,
        "conversation_id": _CONVERSATION_ID,
        "user_message_id": _USER_MESSAGE_ID,
        "response": Response(status=ResponseStatus.created, model="test-model").model_dump(),
    })


def _delta_frame(text: str, sequence_number: int = 2) -> str:
    """One answer text delta as the stream formatter emits it."""
    return sse_frame("response.output_text.delta", {
        "type": "response.output_text.delta",
        "sequence_number": sequence_number,
        "item_id": "item-1",
        "delta": text,
    })


def _completed_frame(text: str = _ANSWER, *, ended_by: str | None = None) -> str:
    """The terminal frame, with the saved assistant id injected at its root."""
    response = Response(
        model="test-model",
        status=ResponseStatus.incomplete if ended_by else ResponseStatus.completed,
        incomplete_details=IncompleteDetails(reason=ended_by) if ended_by else None,
        output=[ResponseOutput(
            id="item-1",
            status=ResponseStatus.completed,
            content=[ResponseOutputContent(text=text)],
        )],
    )
    return sse_frame("response.completed", {
        "type": "response.completed",
        "sequence_number": 9,
        "response": response.model_dump(),
        "assistant_message_id": _ASSISTANT_MESSAGE_ID,
    })


_CANCELLED_FRAME = sse_frame("response.cancelled", {"type": "response.cancelled"})
_FAILED_FRAME = response_failed_sse("The model is unavailable.", "model_unavailable")
_STARTED = _created_frame() + _KEEPALIVE + _delta_frame("po") + _delta_frame("ng", 3)
_SUCCESSFUL_TURN = _STARTED + _completed_frame()


def _chunks(body: str, size: int = 7) -> Iterator[bytes]:
    """Split a body at arbitrary byte offsets, as a proxy may deliver it."""
    raw = body.encode()
    for start in range(0, len(raw), size):
        yield raw[start:start + size]


def _sse(body: str) -> httpx.Response:
    """A streaming SSE response that delivers `body` in small chunks."""
    return httpx.Response(
        200, content=_chunks(body), headers={"Content-Type": "text/event-stream"},
    )


def _saved_turn(answer: str = _ANSWER) -> list[dict]:
    """Conversation items in the shape `_hydrate_message_items` serves."""
    return [
        {"id": _USER_MESSAGE_ID, "role": "user", "content": "question",
         "created_at": "2026-11-02T10:00:00", "events": []},
        {"id": _ASSISTANT_MESSAGE_ID, "role": "assistant", "content": answer,
         "created_at": "2026-11-02T10:00:01", "events": [], "harness": "anton"},
    ]


def _deployment(
    turn_body: str,
    tail_body: str,
    *,
    in_flight: bool = True,
    saved: list[dict] | None = None,
    seen: list[httpx.Request] | None = None,
) -> httpx.MockTransport:
    """A fake deployment answering the routes the turn smoke tests call."""
    items = _saved_turn() if saved is None else saved

    def handle(request: httpx.Request) -> httpx.Response:
        if seen is not None:
            seen.append(request)
        route = (request.method, request.url.path)
        if route == ("POST", "/api/v1/responses/"):
            return _sse(turn_body)
        if route == ("GET", "/api/v1/responses/in-flight"):
            return httpx.Response(200, json={"in_flight": in_flight, "has_buffer": True})
        if route == ("GET", "/api/v1/responses/tail"):
            assert request.url.params["from_seq"] == "0"
            return _sse(tail_body)
        if route == ("GET", f"/api/v1/conversations/{_CONVERSATION_ID}/items"):
            return httpx.Response(200, json=items)
        return httpx.Response(404, json={"unexpected_route": list(route)})

    return httpx.MockTransport(handle)


def _run_end_to_end(body: str, **deployment) -> None:
    """Run the end-to-end smoke test against a deployment streaming `body`."""
    transport = _deployment(turn_body=body, tail_body=body, **deployment)
    with httpx.Client(base_url="https://cowork.example", transport=transport) as api:
        test_post_deploy.test_a_turn_runs_end_to_end(api, _CONVERSATION_ID)


def _run_ingress_reconnect(body: str, **deployment) -> None:
    """Run the ingress reconnect smoke test with `body` as the replayed turn."""
    transport = _deployment(turn_body=_STARTED, tail_body=body, **deployment)
    with httpx.Client(base_url="https://cowork.example", transport=transport) as api:
        test_post_deploy.test_reconnect_replays_a_turn_in_progress(api, _CONVERSATION_ID)


_DIRECT_IDENTITY = test_post_deploy._Identity({
    "api_key": "mdb_never-sent-to-a-pod",
    "user_id": "6f0f1b8e-0000-4000-8000-0000000000aa",
    "organization_id": "6f0f1b8e-0000-4000-8000-0000000000bb",
    "email": "postdeploy@example.com",
})


def _run_replica_reconnect(
    body: str, *, seen: list[httpx.Request] | None = None, **deployment,
) -> None:
    """Run the A to B reconnect smoke test with `body` as replica B's replay.

    Both replica clients share one fake; `seen` collects every request so a
    caller can check which host and headers each one used.
    """
    requests: list[httpx.Request] = [] if seen is None else seen
    transport = _deployment(turn_body=_STARTED, tail_body=body, seen=requests, **deployment)
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("COWORK_BASE_URL_A", "http://replica-a:9010")
        patch.setenv("COWORK_BASE_URL_B", "http://replica-b:9010/")
        patch.setattr(
            test_post_deploy,
            "_replica_client",
            lambda base_url, headers: httpx.Client(
                base_url=base_url, headers=headers, transport=transport,
            ),
        )
        test_post_deploy.test_reconnect_works_on_the_other_replica(
            _CONVERSATION_ID, _DIRECT_IDENTITY,
        )


_SMOKE_RUNNERS: dict[str, Callable[..., None]] = {
    "end_to_end": _run_end_to_end,
    "ingress_reconnect": _run_ingress_reconnect,
    "replica_reconnect": _run_replica_reconnect,
}

_RECONNECT_RUNNERS = ["ingress_reconnect", "replica_reconnect"]

_UNSUCCESSFUL_TURNS = {
    "failed": (
        _STARTED + _FAILED_FRAME,
        "turn did not complete: stream reported response.failed",
    ),
    "cancelled": (
        _STARTED + _CANCELLED_FRAME,
        "turn did not complete: stream reported response.cancelled",
    ),
    "early_eof": (_STARTED, "turn did not complete: no response.completed frame"),
    "created_only": (_created_frame(), "turn did not complete: no response.completed frame"),
    "duplicate_completion": (
        _SUCCESSFUL_TURN + _completed_frame(),
        "turn completed more than once",
    ),
    "handed_back": (
        _STARTED + _completed_frame(ended_by="round_cap"),
        "turn ended with status 'incomplete'",
    ),
    "empty_answer": (
        _created_frame() + _completed_frame(text="  "),
        "turn completed with no answer text",
    ),
}


@pytest.mark.parametrize("runner", list(_SMOKE_RUNNERS))
def test_smoke_turn_accepts_a_complete_answer_in_arbitrary_chunks(runner):
    _SMOKE_RUNNERS[runner](_SUCCESSFUL_TURN)


@pytest.mark.parametrize("case", list(_UNSUCCESSFUL_TURNS))
@pytest.mark.parametrize("runner", list(_SMOKE_RUNNERS))
def test_smoke_turn_rejects_an_unsuccessful_stream(runner, case):
    body, reason = _UNSUCCESSFUL_TURNS[case]

    with pytest.raises(AssertionError, match=reason):
        _SMOKE_RUNNERS[runner](body)


@pytest.mark.parametrize("runner", _RECONNECT_RUNNERS)
def test_reconnect_rejects_a_replay_that_differs_from_the_saved_answer(runner):
    with pytest.raises(AssertionError, match="replayed answer differs from the saved one"):
        _SMOKE_RUNNERS[runner](_SUCCESSFUL_TURN, saved=_saved_turn("pon"))


@pytest.mark.parametrize("runner", _RECONNECT_RUNNERS)
def test_reconnect_rejects_a_replay_whose_answer_was_not_saved(runner):
    with pytest.raises(AssertionError, match="replayed answer was not saved"):
        _SMOKE_RUNNERS[runner](_SUCCESSFUL_TURN, saved=_saved_turn()[:1])


@pytest.mark.parametrize("runner", _RECONNECT_RUNNERS)
def test_reconnect_skips_only_after_a_good_replay_when_the_turn_already_ended(runner):
    with pytest.raises(pytest.skip.Exception, match="ended before the connection dropped"):
        _SMOKE_RUNNERS[runner](_SUCCESSFUL_TURN, in_flight=False)


@pytest.mark.parametrize("runner", _RECONNECT_RUNNERS)
def test_reconnect_rejects_a_broken_replay_even_when_the_turn_already_ended(runner):
    # A skip is not an AssertionError, so pytest.raises alone would let it
    # through and report this test as skipped rather than failed.
    try:
        with pytest.raises(AssertionError, match="stream reported response.failed"):
            _SMOKE_RUNNERS[runner](_STARTED + _FAILED_FRAME, in_flight=False)
    except pytest.skip.Exception as skipped:
        pytest.fail(f"a broken replay was skipped instead of rejected: {skipped}")


def test_replica_reconnect_starts_on_a_and_replays_from_b_with_direct_identity():
    seen: list[httpx.Request] = []

    _run_replica_reconnect(_SUCCESSFUL_TURN, seen=seen)

    hosts = {(request.method, request.url.path): request.url.host for request in seen}
    assert hosts[("POST", "/api/v1/responses/")] == "replica-a"
    assert hosts[("GET", "/api/v1/responses/tail")] == "replica-b"
    assert hosts[("GET", f"/api/v1/conversations/{_CONVERSATION_ID}/items")] == "replica-b"
    for request in seen:
        assert "authorization" not in request.headers
        assert request.headers["x-user-id"] == _DIRECT_IDENTITY["user_id"]
        assert request.headers["x-organization-id"] == _DIRECT_IDENTITY["organization_id"]
