"""Smoke tests against a deployed cowork-server.

These talk to a running deployment over HTTP, so they exercise cowork-server,
Redis, the scratchpad controller and a real scratchpad pod together.

Skipped unless COWORK_BASE_URL is set, so a normal `pytest` run ignores them.

The identity comes from auth, which provisions throwaway test users for CI.
Permanent envs (dev/staging/prod) POST to its internal endpoint with the
provisioning secret; PR envs POST to /dev/mint-test-user/, which is mounted
only where `ephemeral` is on and needs no secret. Either way the response
carries the user_id and organization_id these tests send as headers.

    COWORK_BASE_URL=https://cowork.staging.example.com \\
    TEST_USER_PROVISION_URL=https://auth.staging.example.com/v1/internal/test-users/ \\
    TEST_USER_PROVISION_SECRET=... \\
    uv run pytest tests/integration/test_post_deploy.py -v

    # PR env
    COWORK_BASE_URL=https://cowork-server-pr-123.dev.mindshub.ai \\
    TEST_USER_MINT_URL=https://auth-pr-123.dev.mindshub.ai/dev/mint-test-user/ \\
    uv run pytest tests/integration/test_post_deploy.py -v

Requests go through the ingress, which authenticates the Bearer key against
auth and injects the identity headers itself. Sending those headers from here
would achieve nothing: the ingress overwrites them from its auth subrequest.

The cross-replica test is the exception. The ingress pins a client to one pod
(cookie affinity), so it talks to two pods directly, which means no ingress and
therefore no injected identity: those two clients send the headers themselves.

    kubectl port-forward pod/cowork-server-aaa 8081:9010
    kubectl port-forward pod/cowork-server-bbb 8082:9010
    COWORK_BASE_URL_A=http://localhost:8081 COWORK_BASE_URL_B=http://localhost:8082 ...
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

# The auth hosts sit behind Cloudflare, whose bot rules 403 a default httpx or
# requests User-Agent ("error code: 1010"). The block is signature-based, so a
# browser string is enough. Same workaround as mindshub_inference/tests/env.py.
BROWSER_UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/126.0 Safari/537.36"
)


def _base_url() -> str:
    url = os.environ.get("COWORK_BASE_URL")
    if not url:
        pytest.skip("COWORK_BASE_URL not set; post-deploy tests only run against a deployment")
    return url.rstrip("/")


def _provision_identity() -> dict[str, str]:
    """A user_id, organization_id and email for a throwaway tenant.

    Three sources, in order:

    1. COWORK_TEST_API_KEY + COWORK_TEST_USER_ID + COWORK_TEST_ORG_ID, for
       running by hand against an environment where you already have a tenant.
    2. TEST_USER_MINT_URL, auth's /dev/mint-test-user/. Mounted only in
       ephemeral PR envs, needs no secret, mints a fresh user per call.
    3. TEST_USER_PROVISION_URL + TEST_USER_PROVISION_SECRET, auth's internal
       endpoint. Used for dev/staging/prod, where the dev route is not mounted.
       Provisions the fixed `cowork` suite: one @emailsink.dev tenant, reused
       across runs, with a fresh key each time.
    """
    api_key = os.environ.get("COWORK_TEST_API_KEY")
    user_id = os.environ.get("COWORK_TEST_USER_ID")
    org_id = os.environ.get("COWORK_TEST_ORG_ID")
    if api_key and user_id and org_id:
        return {
            "api_key": api_key,
            "user_id": user_id,
            "organization_id": org_id,
            "email": os.environ.get("COWORK_TEST_USER_EMAIL", "postdeploy@example.com"),
        }

    mint_url = os.environ.get("TEST_USER_MINT_URL")
    if mint_url:
        resp = httpx.post(
            mint_url, json={}, headers={"User-Agent": BROWSER_UA}, timeout=60.0)
        if resp.status_code != 201:
            pytest.fail(f"minting a PR-env test user failed: {resp.status_code} {resp.text}")
        user = resp.json()
    else:
        provision_url = os.environ.get("TEST_USER_PROVISION_URL")
        secret = os.environ.get("TEST_USER_PROVISION_SECRET")
        if not (provision_url and secret):
            pytest.skip(
                "no identity source: set COWORK_TEST_API_KEY + COWORK_TEST_USER_ID + "
                "COWORK_TEST_ORG_ID, or TEST_USER_MINT_URL, or TEST_USER_PROVISION_URL "
                "+ TEST_USER_PROVISION_SECRET"
            )
        resp = httpx.post(
            provision_url,
            json={"suite": "cowork"},
            headers={"X-Internal-Auth": secret, "User-Agent": BROWSER_UA},
            timeout=60.0,
        )
        if resp.status_code != 201:
            pytest.fail(f"provisioning the cowork test user failed: {resp.status_code} {resp.text}")
        users = resp.json()["users"]
        if not users:
            pytest.fail(f"the cowork suite provisioned no users: {resp.text}")
        user = users[0]

    if not user.get("organization_id"):
        pytest.fail(
            f"auth returned no organization_id for {user.get('email')}; "
            "the personal org is provisioned on first login and could not be resolved"
        )
    return user


@pytest.fixture(scope="session")
def identity() -> dict[str, str]:
    """Provisioned once per run: minting is a Keycloak round trip per call."""
    return _provision_identity()


def _headers(identity: dict[str, str]) -> dict[str, str]:
    """What the ingress wants: a key it can authenticate against auth.

    It then injects X-User-Id / X-Organization-Id itself from that subrequest,
    so sending them here would be overwritten anyway.
    """
    return {"Authorization": f"Bearer {identity['api_key']}"}


def _direct_headers(identity: dict[str, str]) -> dict[str, str]:
    """What a pod wants when the ingress is not in the path.

    Nothing has injected identity, so TrustedHeaderMiddleware reads these. The
    ids must be UUIDs or it rejects them.
    """
    return {
        "X-Organization-Id": identity["organization_id"],
        "X-User-Id": identity["user_id"],
        "X-User-Email": identity.get("email", "postdeploy@example.com"),
    }


@pytest.fixture
def api(identity):
    with httpx.Client(base_url=_base_url(), headers=_headers(identity), timeout=30.0) as client:
        yield client


@pytest.fixture
def conversation_id():
    """A UUID, because a non-UUID id is replaced by a canonical one and every
    later call would then be asking about a conversation that does not exist."""
    return str(uuid.uuid4())


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


def test_reconnect_works_on_the_other_replica(conversation_id, identity):
    """A turn started on one replica can be probed and tailed from another.

    Talks to two pods directly rather than through the ingress, which pins a
    client to one pod by cookie, so both halves would otherwise land on the
    same replica and prove nothing. No ingress means no injected identity, so
    these two clients send the headers themselves.
    """
    url_a = os.environ.get("COWORK_BASE_URL_A")
    url_b = os.environ.get("COWORK_BASE_URL_B")
    if not (url_a and url_b):
        pytest.skip("COWORK_BASE_URL_A and _B not set; needs two reachable pods")

    headers = _direct_headers(identity)
    with httpx.Client(base_url=url_a.rstrip("/"), headers=headers, timeout=30.0) as replica_a:
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
