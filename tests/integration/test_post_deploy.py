"""Smoke tests against a deployed cowork-server.

These talk to a running deployment over HTTP, so they exercise cowork-server,
Redis, the scratchpad controller and a real scratchpad pod together.

Skipped unless COWORK_BASE_URL is set, so a normal `pytest` run ignores them.

    COWORK_BASE_URL=https://cowork.staging.example.com \\
    COWORK_TEST_ORG_ID=<uuid> COWORK_TEST_USER_ID=<uuid> \\
    uv run pytest tests/integration/test_post_deploy.py -v

Reconnect-across-replicas needs to reach individual pods rather than the
service, since the load balancer may send both requests to the same one:

    kubectl port-forward pod/cowork-server-aaa 8081:8000
    kubectl port-forward pod/cowork-server-bbb 8082:8000
    COWORK_BASE_URL=http://localhost:8081 COWORK_BASE_URL_B=http://localhost:8082 ...
"""

from __future__ import annotations

import json
import os
import time
import uuid

import httpx
import pytest

pytestmark = pytest.mark.postdeploy

# A turn that runs for a while, so there is something to cancel and something
# to reconnect to. Short enough that a failing test does not hang CI.
SLOW_PROMPT = (
    "Count slowly from 1 to 40, one number per line, "
    "pausing to think between each one."
)
QUICK_PROMPT = "Reply with exactly the word: pong"

TURN_TIMEOUT_S = 180.0
CANCEL_VISIBLE_S = 45.0


def _base_url() -> str:
    url = os.environ.get("COWORK_BASE_URL")
    if not url:
        pytest.skip("COWORK_BASE_URL not set; post-deploy tests only run against a deployment")
    return url.rstrip("/")


def _headers() -> dict[str, str]:
    """Identity headers the gateway normally injects (see cowork/principal.py).

    Sent by hand here because these tests bypass the gateway. Both must be
    UUIDs or TrustedHeaderMiddleware rejects them.
    """
    org_id = os.environ.get("COWORK_TEST_ORG_ID")
    user_id = os.environ.get("COWORK_TEST_USER_ID")
    if not (org_id and user_id):
        pytest.skip("COWORK_TEST_ORG_ID and COWORK_TEST_USER_ID are required")
    headers = {
        "X-Organization-Id": org_id,
        "X-User-Id": user_id,
        "X-User-Email": os.environ.get("COWORK_TEST_USER_EMAIL", "postdeploy@example.com"),
    }
    token = os.environ.get("COWORK_AUTH_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


@pytest.fixture
def api():
    with httpx.Client(base_url=_base_url(), headers=_headers(), timeout=30.0) as client:
        yield client


@pytest.fixture
def conversation_id():
    return f"postdeploy-{uuid.uuid4()}"


def _sse_events(text: str) -> list[str]:
    """Event names from an SSE body, in order."""
    return [
        line.removeprefix("event:").strip()
        for line in text.splitlines()
        if line.startswith("event:")
    ]


def _stream_turn(client: httpx.Client, conversation_id: str, prompt: str, *,
                 read_timeout: float = TURN_TIMEOUT_S) -> list[str]:
    """POST a turn and drain its SSE response, returning the event names."""
    events: list[str] = []
    with client.stream(
        "POST", "/api/v1/responses/",
        json={"input": prompt, "conversation": conversation_id, "stream": True},
        timeout=httpx.Timeout(read_timeout, connect=10.0),
    ) as resp:
        assert resp.status_code == 200, resp.read()[:500]
        for line in resp.iter_lines():
            if line.startswith("event:"):
                events.append(line.removeprefix("event:").strip())
    return events


def test_a_turn_runs_end_to_end(api, conversation_id):
    """A turn posted over HTTP reaches a pod and its replies reach the client."""
    events = _stream_turn(api, conversation_id, QUICK_PROMPT)

    assert "response.created" in events
    assert "response.completed" in events, f"turn did not complete: {events}"


def test_reconnect_replays_a_turn_in_progress(api, conversation_id):
    """Close the stream mid-turn and tail it back, the page-reload path."""
    with api.stream(
        "POST", "/api/v1/responses/",
        json={"input": SLOW_PROMPT, "conversation": conversation_id, "stream": True},
        timeout=httpx.Timeout(TURN_TIMEOUT_S, connect=10.0),
    ) as resp:
        assert resp.status_code == 200
        for line in resp.iter_lines():          # drop the connection early
            if line.startswith("event:"):
                break

    probe = api.get("/api/v1/responses/in-flight",
                    params={"conversation_id": conversation_id}).json()
    assert probe["has_buffer"] is True, probe

    with api.stream(
        "GET", "/api/v1/responses/tail",
        params={"conversation_id": conversation_id, "from_seq": 0},
        timeout=httpx.Timeout(TURN_TIMEOUT_S, connect=10.0),
    ) as resp:
        assert resp.status_code == 200
        replayed = _sse_events(resp.read().decode())

    assert "response.created" in replayed, replayed


def test_reconnect_works_on_the_other_replica(conversation_id):
    """A turn started on one replica can be probed and tailed from another.

    Requires COWORK_BASE_URL_B pointing at a different pod, since the load
    balancer may otherwise send both requests to the same one.
    """
    url_b = os.environ.get("COWORK_BASE_URL_B")
    if not url_b:
        pytest.skip("COWORK_BASE_URL_B not set; needs a second replica to be meaningful")

    headers = _headers()
    with httpx.Client(base_url=_base_url(), headers=headers, timeout=30.0) as replica_a:
        with replica_a.stream(
            "POST", "/api/v1/responses/",
            json={"input": SLOW_PROMPT, "conversation": conversation_id, "stream": True},
            timeout=httpx.Timeout(TURN_TIMEOUT_S, connect=10.0),
        ) as resp:
            assert resp.status_code == 200
            for line in resp.iter_lines():
                if line.startswith("event:"):
                    break

    with httpx.Client(base_url=url_b.rstrip("/"), headers=headers, timeout=30.0) as replica_b:
        probe = replica_b.get("/api/v1/responses/in-flight",
                              params={"conversation_id": conversation_id}).json()
        # Before the Redis backend this replica had no handle and no buffer,
        # and answered has_buffer=False.
        assert probe["has_buffer"] is True, probe

        with replica_b.stream(
            "GET", "/api/v1/responses/tail",
            params={"conversation_id": conversation_id, "from_seq": 0},
            timeout=httpx.Timeout(TURN_TIMEOUT_S, connect=10.0),
        ) as resp:
            assert resp.status_code == 200
            replayed = _sse_events(resp.read().decode())

    assert "response.created" in replayed, replayed


def test_cancel_ends_the_turn(api, conversation_id):
    """After POST /cancel, /in-flight reports the turn as finished."""
    with api.stream(
        "POST", "/api/v1/responses/",
        json={"input": SLOW_PROMPT, "conversation": conversation_id, "stream": True},
        timeout=httpx.Timeout(TURN_TIMEOUT_S, connect=10.0),
    ) as resp:
        assert resp.status_code == 200
        for line in resp.iter_lines():
            if line.startswith("event:"):
                break

    cancel = api.post("/api/v1/responses/cancel",
                      json={"conversation_id": conversation_id})
    assert cancel.status_code == 200, cancel.text
    assert cancel.json()["cancelled"] is True

    deadline = time.monotonic() + CANCEL_VISIBLE_S
    while time.monotonic() < deadline:
        probe = api.get("/api/v1/responses/in-flight",
                        params={"conversation_id": conversation_id}).json()
        if probe["in_flight"] is False:
            break
        time.sleep(2)
    else:
        pytest.fail(
            f"turn still in flight {CANCEL_VISIBLE_S:.0f}s after cancel: {probe}")


def test_a_second_message_queues_behind_the_first(api, conversation_id):
    """The controller serialises per conversation, so two messages sent back to
    back must both finish rather than fight over one pod."""
    first = _stream_turn(api, conversation_id, QUICK_PROMPT)
    assert "response.completed" in first, first

    second = _stream_turn(api, conversation_id, QUICK_PROMPT)
    assert "response.completed" in second, second

    items = api.get(f"/api/v1/conversations/{conversation_id}/items")
    assert items.status_code == 200, items.text
    body = items.json()
    payload = body if isinstance(body, list) else body.get("items", [])
    assert len(payload) >= 4, f"expected both turns persisted, got {json.dumps(body)[:400]}"


def test_deleting_a_conversation_leaves_no_replayable_buffer(api, conversation_id):
    """turn_id is the message count, so a deleted conversation's buffer would be
    replayed as the next turn's answer if it survived."""
    assert "response.completed" in _stream_turn(api, conversation_id, QUICK_PROMPT)

    deleted = api.delete(f"/api/v1/conversations/{conversation_id}")
    assert deleted.status_code in (200, 204), deleted.text

    probe = api.get("/api/v1/responses/in-flight",
                    params={"conversation_id": conversation_id}).json()
    assert probe["has_buffer"] is False, probe
