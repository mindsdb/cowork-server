"""Smoke tests against a deployed cowork-server.

These talk to a running deployment over HTTP, so they exercise cowork-server,
Redis, the scratchpad controller and a real scratchpad pod together.

Skipped unless COWORK_BASE_URL is set, so a normal `pytest` run ignores them.

The identity comes from auth, which provisions throwaway test users for CI.
Permanent dev/staging POST to its internal endpoint with the provisioning
secret; PR envs POST to /dev/mint-test-user/, which is mounted only where
`ephemeral` is on and needs no secret. Prod uses a dedicated standing identity
while its fixture password remains committed. Every source
provides the user_id and organization_id these tests send as headers.

The provisioning call uses auth's Service so it works both before and after
Cloudflare Access protects /v1/internal* on the public host. CI uses cluster
DNS from a runner in the target cluster; by hand, forward the port first with
`kubectl port-forward -n staging svc/auth 8080:80`.

    COWORK_BASE_URL=https://cowork.staging.example.com \\
    TEST_USER_PROVISION_URL=http://localhost:8080/v1/internal/test-users/ \\
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

import itertools
import json
import os
import time
import uuid
from collections.abc import Iterable

import httpx
import pytest

from tests.integration.prereq import missing_prerequisite

pytestmark = pytest.mark.postdeploy

# The cancel test needs a turn that is still running one HTTP call later. A
# fast deployment answered a 40-line count before the next call landed, and the
# test skipped rather than cancelling anything. Ten times the output buys that
# margin without making a failure hang CI — the turn is cancelled or abandoned
# either way, so the extra tokens are never actually generated.
CANCEL_PROMPT = (
    "Count slowly from 1 to 400, one number per line, "
    "pausing to think between each one."
)
# The reconnect tests need a turn still running after its first answer delta.
# A 40-line count ended within the same few seconds on a PR environment, so
# they take the cancel test's length. Unlike cancel, this turn runs to the end.
RECONNECT_PROMPT = CANCEL_PROMPT
QUICK_PROMPT = "Reply with exactly the word: pong"
FIRST_TURN_PROMPT = "Reply with exactly the word: alpha"
SECOND_TURN_PROMPT = "Reply with exactly the word: bravo"

TURN_TIMEOUT_S = 180.0
CANCEL_VISIBLE_S = 45.0
CROSS_REPLICA_VISIBILITY_S = 10.0
# How long to wait for the server to report the turn as running before giving
# up on having anything to cancel. This covers the gap between the caller
# disconnecting and the registry answering, not the model's thinking time.
CANCEL_PREMISE_S = 10.0

# The auth hosts sit behind Cloudflare, whose bot rules 403 a default httpx or
# requests User-Agent ("error code: 1010"). The block is signature-based, so a
# browser string is enough. Same workaround as mindshub_inference/tests/env.py.
BROWSER_UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/126.0 Safari/537.36"
)
PROD_AUTHENTICATE_URL = "https://auth.mindshub.ai/v1/authenticate/"
PROD_COWORK_BASE_URL = "https://cowork.mindshub.ai"
CONTROLLED_TEST_EMAIL_SUFFIX = "@mindshub.ai"


class _Identity(dict[str, str]):
    """A test identity whose repr can safely appear in a pytest traceback."""

    def __repr__(self) -> str:
        redacted = dict(self)
        if "api_key" in redacted:
            redacted["api_key"] = "***"
        return repr(redacted)


def _base_url() -> str:
    url = os.environ.get("COWORK_BASE_URL")
    if not url:
        missing_prerequisite(
            "COWORK_BASE_URL not set; post-deploy tests only run against a deployment"
        )
    normalized_url = url.rstrip("/")
    if (
        os.environ.get("COWORK_TEST_IDENTITY_MODE") == "standing"
        and normalized_url != PROD_COWORK_BASE_URL
    ):
        missing_prerequisite(
            "COWORK_TEST_IDENTITY_MODE=standing requires COWORK_BASE_URL to be "
            f"exactly {PROD_COWORK_BASE_URL}; refusing to send the production key "
            "to a different origin"
        )
    return normalized_url


def _verified_prod_standing_identity() -> _Identity:
    """Resolve the dedicated prod API key through auth without mutating a user."""
    # Validate the eventual credential destination before the first network call.
    # The prod URL is committed config, but the Python guard is the last boundary
    # if workflow code or that file is changed in the same revision.
    _base_url()
    api_key = os.environ.get("COWORK_TEST_API_KEY")
    expected_email = os.environ.get("COWORK_TEST_USER_EMAIL")
    expected_org_id = os.environ.get("COWORK_TEST_ORG_ID")
    if not (api_key and expected_email and expected_org_id):
        missing_prerequisite(
            "COWORK_TEST_IDENTITY_MODE=standing requires COWORK_TEST_API_KEY and "
            "COWORK_TEST_USER_EMAIL and COWORK_TEST_ORG_ID"
        )
    if not api_key.startswith("mdb_"):
        missing_prerequisite(
            "COWORK_TEST_API_KEY must be a MindsDB API key beginning with mdb_"
        )
    if not expected_email.lower().endswith(CONTROLLED_TEST_EMAIL_SUFFIX):
        missing_prerequisite(
            "COWORK_TEST_USER_EMAIL must use a controlled, non-staff "
            "@mindshub.ai account; @emailsink.dev and the staff @mindsdb.com "
            "domain are not permitted in prod"
        )

    response = httpx.get(
        PROD_AUTHENTICATE_URL,
        headers={
            "Authorization": f"Bearer {api_key}",
            "X-MindsDB-Product": "hub",
            "User-Agent": BROWSER_UA,
        },
        timeout=30.0,
        follow_redirects=False,
    )
    if response.status_code != 200:
        pytest.fail(
            "auth rejected the dedicated prod Cowork test identity: "
            f"HTTP {response.status_code}"
        )
    try:
        payload = response.json()
    except ValueError:
        pytest.fail(
            "auth returned non-JSON for the dedicated prod Cowork test identity"
        )
    if not isinstance(payload, dict) or payload.get("valid") is not True:
        pytest.fail("auth did not validate the dedicated prod Cowork test identity")
    if (
        payload.get("auth_method") != "api_key"
        or payload.get("key_type") != "user"
        or payload.get("key_prefix") != api_key.partition(".")[0]
    ):
        pytest.fail(
            "the dedicated prod Cowork credential is not a standing user API key"
        )
    if str(payload.get("email", "")).lower() != expected_email.lower():
        pytest.fail(
            "auth returned a different email for the dedicated prod Cowork test identity"
        )
    if str(payload.get("organization_id", "")) != expected_org_id:
        pytest.fail(
            "auth returned a different organization for the dedicated prod Cowork test identity"
        )
    if not payload.get("user_id"):
        pytest.fail(
            "auth returned no user id for the dedicated prod Cowork test identity"
        )
    try:
        hub_admin = payload["entitlements"]["permissions"]["admin"]["hub"]
    except (KeyError, TypeError):
        pytest.fail(
            "auth returned no Hub admin entitlement for the dedicated prod Cowork test identity"
        )
    if hub_admin is not False:
        pytest.fail("the dedicated prod Cowork test identity has Hub admin access")
    if response.headers.get("X-Billing-Segment", "").lower() not in {"free", "paid"}:
        pytest.fail(
            "auth did not classify the dedicated prod Cowork test identity as non-employee"
        )
    return _Identity(
        {
            "api_key": api_key,
            "user_id": str(payload["user_id"]),
            "organization_id": str(payload["organization_id"]),
            "email": str(payload["email"]),
        }
    )


def _provision_identity() -> _Identity:
    """A user_id, organization_id and email for a throwaway tenant.

    Three sources, in order:

    1. COWORK_TEST_API_KEY + COWORK_TEST_USER_ID + COWORK_TEST_ORG_ID, for
       running by hand against an environment where you already have a tenant.
    2. TEST_USER_MINT_URL, auth's /dev/mint-test-user/. Mounted only in
       ephemeral PR envs, needs no secret, mints a fresh user per call.
    3. TEST_USER_PROVISION_URL + TEST_USER_PROVISION_SECRET, auth's internal
       endpoint. Used for dev/staging, where the dev route is not mounted.
       Provisions the fixed `cowork` suite: one @emailsink.dev tenant, reused
       across runs, with a fresh key each time.

    Production sets COWORK_TEST_IDENTITY_MODE=standing. In that mode a dedicated
    controlled-domain API key and its expected email are mandatory. Auth
    resolves the trusted ids live, and this function never falls back to the
    mutating internal provisioner, even if its URL is also present.
    """
    identity_mode = os.environ.get("COWORK_TEST_IDENTITY_MODE")
    if identity_mode == "standing":
        return _verified_prod_standing_identity()

    api_key = os.environ.get("COWORK_TEST_API_KEY")
    user_id = os.environ.get("COWORK_TEST_USER_ID")
    org_id = os.environ.get("COWORK_TEST_ORG_ID")
    if api_key and user_id and org_id:
        return _Identity(
            {
                "api_key": api_key,
                "user_id": user_id,
                "organization_id": org_id,
                "email": os.environ.get(
                    "COWORK_TEST_USER_EMAIL", "postdeploy@example.com"
                ),
            }
        )

    mint_url = os.environ.get("TEST_USER_MINT_URL")
    if mint_url:
        resp = httpx.post(
            mint_url,
            json={},
            headers={"User-Agent": BROWSER_UA},
            timeout=60.0,
            follow_redirects=False,
        )
        if resp.status_code != 201:
            pytest.fail(f"minting a PR-env test user failed: HTTP {resp.status_code}")
        user = resp.json()
    else:
        provision_url = os.environ.get("TEST_USER_PROVISION_URL")
        secret = os.environ.get("TEST_USER_PROVISION_SECRET")
        if not (provision_url and secret):
            missing_prerequisite(
                "no identity source: set COWORK_TEST_API_KEY + COWORK_TEST_USER_ID + "
                "COWORK_TEST_ORG_ID, or TEST_USER_MINT_URL, or TEST_USER_PROVISION_URL "
                "+ TEST_USER_PROVISION_SECRET"
            )
        resp = httpx.post(
            provision_url,
            json={"suite": "cowork"},
            headers={"X-Internal-Auth": secret, "User-Agent": BROWSER_UA},
            timeout=60.0,
            follow_redirects=False,
        )
        if resp.status_code != 201:
            pytest.fail(
                f"provisioning the cowork test user failed: HTTP {resp.status_code}"
            )
        users = resp.json()["users"]
        if not users:
            pytest.fail("the cowork suite provisioned no users")
        user = users[0]

    if not user.get("organization_id"):
        pytest.fail(
            f"auth returned no organization_id for {user.get('email')}; "
            "the personal org is provisioned on first login and could not be resolved"
        )
    return _Identity(user)


@pytest.fixture(scope="session")
def identity() -> _Identity:
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
    with httpx.Client(
        base_url=_base_url(), headers=_headers(identity), timeout=30.0
    ) as client:
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


SseFrame = tuple[str, dict]

_UNSUCCESSFUL_EVENTS = ("response.failed", "response.cancelled", "error")


def _read_sse_frames(lines: Iterable[str]) -> list[SseFrame]:
    """Group SSE lines into (event name, JSON payload) frames, in order.

    Heartbeat comments are skipped. A payload that is not a JSON object is
    kept as ``{"raw": ...}`` so a malformed frame still shows up in a failure.
    A frame cut off by the end of the stream is still returned.

    Args:
        lines: Decoded lines without their line endings, as ``iter_lines`` yields.

    Returns:
        One tuple per frame that carried an ``event:`` line.
    """
    frames: list[SseFrame] = []
    event: str | None = None
    data: list[str] = []
    # The trailing "" flushes a final frame the stream did not terminate.
    for line in itertools.chain(lines, [""]):
        if line.startswith(":"):
            continue
        if line.startswith("event:"):
            event = line.removeprefix("event:").strip()
        elif line.startswith("data:"):
            data.append(line.removeprefix("data:").strip())
        elif not line:
            if event is not None:
                raw = "\n".join(data)
                try:
                    payload = json.loads(raw)
                except ValueError:
                    payload = None
                if not isinstance(payload, dict):
                    payload = {"raw": raw[:300]}
                frames.append((event, payload))
            event, data = None, []
    return frames


def _assert_successful_turn(frames: list[SseFrame]) -> dict:
    """Fail unless the frames are one turn that finished with an answer.

    Success is exactly one ``response.completed`` whose status is
    ``completed``, no failure, error or cancel frame, and nonempty answer
    text. The text comes from the completed payload, not from joined
    deltas, because answer reset and restore frames rewrite what streamed.

    Args:
        frames: The turn's frames from ``_read_sse_frames``.

    Returns:
        The ``response.completed`` frame's payload.

    Raises:
        AssertionError: naming the first rule the stream broke, with the
            server's code and message when it reported a failure.
    """
    events = [event for event, _ in frames]
    for event, payload in frames:
        if event in _UNSUCCESSFUL_EVENTS:
            message = payload.get("error") or payload.get("message")
            raise AssertionError(
                f"turn did not complete: stream reported {event} "
                f"code={payload.get('code')!r} message={message!r} events={events}"
            )
    completed = [payload for event, payload in frames if event == "response.completed"]
    assert completed, f"turn did not complete: no response.completed frame, events={events}"
    assert len(completed) == 1, f"turn completed more than once: events={events}"

    response = completed[0].get("response") or {}
    status = response.get("status")
    assert status == "completed", (
        f"turn ended with status {status!r}: {response.get('incomplete_details')}"
    )
    assert _answer_text(completed[0]).strip(), (
        f"turn completed with no answer text: events={events}"
    )
    return completed[0]


def _answer_text(completed: dict) -> str:
    """The final answer text carried by a ``response.completed`` payload."""
    return "".join(
        part.get("text", "")
        for output in completed.get("response", {}).get("output", [])
        for part in output.get("content", [])
    )


def _stream_turn(
    client: httpx.Client,
    conversation_id: str,
    prompt: str,
    *,
    read_timeout: float = TURN_TIMEOUT_S,
    frames: list[SseFrame] | None = None,
) -> list[str]:
    """POST a turn and drain its SSE response, returning the event names.

    Pass ``frames`` to also receive every frame with its payload, which
    ``_assert_successful_turn`` needs to say why a turn did not succeed.
    """
    with client.stream(
        "POST",
        "/api/v1/responses/",
        json={"input": prompt, "conversation": conversation_id, "stream": True},
        timeout=httpx.Timeout(read_timeout, connect=10.0),
    ) as resp:
        assert resp.status_code == 200, resp.read()[:500]
        received = _read_sse_frames(resp.iter_lines())
    if frames is not None:
        frames.extend(received)
    return [event for event, _ in received]


def test_a_turn_runs_end_to_end(api, conversation_id):
    """A turn posted over HTTP reaches a pod and its answer reaches the client."""
    frames: list[SseFrame] = []
    events = _stream_turn(api, conversation_id, QUICK_PROMPT, frames=frames)

    assert "response.created" in events
    _assert_successful_turn(frames)


_ANSWER_DELTA = "response.output_text.delta"
_RECONNECT_PREMISE_SKIP = (
    "the turn ended before the connection dropped, so its replay was checked "
    "but no mid-turn reconnect happened on this run; the prompt needs to "
    "outlast the first answer delta on this deployment"
)


def _drop_after_first_answer_delta(resp: httpx.Response) -> None:
    """Read a turn's stream until the model has produced answer text, then stop.

    ``response.created`` is written before the model streams anything, so
    leaving on it would reconnect to a turn that has not started answering.
    """
    for line in resp.iter_lines():
        if line.startswith("event:") and line.removeprefix("event:").strip() == _ANSWER_DELTA:
            return


def _replica_client(base_url: str, headers: dict[str, str]) -> httpx.Client:
    """A client that talks to one pod directly, bypassing the ingress."""
    return httpx.Client(base_url=base_url.rstrip("/"), headers=headers, timeout=30.0)


def _conversation_items(client: httpx.Client, conversation_id: str) -> list[dict]:
    """The conversation's saved items, as the renderer loads them on reload."""
    items = client.get(f"/api/v1/conversations/{conversation_id}/items")
    assert items.status_code == 200, items.text[:500]
    body = items.json()
    return body if isinstance(body, list) else body.get("items", [])


def _assert_answer_saved(client: httpx.Client, conversation_id: str, completed: dict) -> None:
    """Fail unless the completed frame's answer is the saved assistant message.

    The server saves the answer before it writes the completed frame, so one
    read is enough.
    """
    message_id = completed.get("assistant_message_id")
    assert message_id, "completed frame carried no assistant_message_id"
    saved = [
        item
        for item in _conversation_items(client, conversation_id)
        if str(item.get("id")) == message_id
    ]
    assert len(saved) == 1, f"replayed answer was not saved: assistant_message_id={message_id}"
    assert saved[0].get("role") == "assistant", f"saved answer has role {saved[0].get('role')!r}"
    assert saved[0].get("content") == _answer_text(completed), (
        "replayed answer differs from the saved one: "
        f"replayed={_answer_text(completed)[:200]!r} saved={str(saved[0].get('content'))[:200]!r}"
    )


def _replay_turn(client: httpx.Client, conversation_id: str) -> dict:
    """Tail the whole turn from the start and require a successful answer.

    Returns:
        The replayed ``response.completed`` payload.
    """
    with client.stream(
        "GET",
        "/api/v1/responses/tail",
        params={"conversation_id": conversation_id, "from_seq": 0},
        timeout=httpx.Timeout(TURN_TIMEOUT_S, connect=10.0),
    ) as resp:
        assert resp.status_code == 200
        replayed = _read_sse_frames(resp.iter_lines())

    assert replayed and replayed[0][0] == "response.created", (
        f"replay did not start with response.created: {replayed[:1]}"
    )
    return _assert_successful_turn(replayed)


def test_reconnect_replays_a_turn_in_progress(api, conversation_id):
    """Close the stream mid-answer and tail it back, the page-reload path."""
    with api.stream(
        "POST",
        "/api/v1/responses/",
        json={"input": RECONNECT_PROMPT, "conversation": conversation_id, "stream": True},
        timeout=httpx.Timeout(TURN_TIMEOUT_S, connect=10.0),
    ) as resp:
        assert resp.status_code == 200
        _drop_after_first_answer_delta(resp)

    probe = api.get(
        "/api/v1/responses/in-flight", params={"conversation_id": conversation_id}
    ).json()
    assert probe["has_buffer"] is True, probe
    running = _await_running_turn(api, conversation_id)

    completed = _replay_turn(api, conversation_id)
    _assert_answer_saved(api, conversation_id, completed)

    # Only after the replay passed: a skip must never hide a broken one.
    if running is None:
        pytest.skip(_RECONNECT_PREMISE_SKIP)


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
        missing_prerequisite(
            "COWORK_BASE_URL_A and _B not set; needs two reachable pods"
        )

    headers = _direct_headers(identity)
    with _replica_client(url_a, headers) as replica_a:
        with replica_a.stream(
            "POST",
            "/api/v1/responses/",
            json={
                "input": RECONNECT_PROMPT,
                "conversation": conversation_id,
                "stream": True,
            },
            timeout=httpx.Timeout(TURN_TIMEOUT_S, connect=10.0),
        ) as resp:
            assert resp.status_code == 200
            _drop_after_first_answer_delta(resp)

    with _replica_client(url_b, headers) as replica_b:
        probe = _await_shared_buffer(replica_b, conversation_id)
        # Before the Redis backend this replica had no handle and no buffer,
        # and answered has_buffer=False.
        assert probe["has_buffer"] is True, probe
        running = _await_running_turn(replica_b, conversation_id)

        completed = _replay_turn(replica_b, conversation_id)
        _assert_answer_saved(replica_b, conversation_id, completed)

    # Only after the replay passed: a skip must never hide a broken one.
    if running is None:
        pytest.skip(_RECONNECT_PREMISE_SKIP)


def _await_shared_buffer(
    api, conversation_id, *, timeout_s=CROSS_REPLICA_VISIBILITY_S
) -> dict:
    """Wait for a peer replica to observe the turn index written by the producer.

    ``response.created`` is appended before the remote reply generator starts,
    so seeing that frame on replica A does not mean its Redis turn-index write
    has completed. The index is shared synchronously once written; this wait
    covers only that startup ordering window.
    """
    deadline = time.monotonic() + timeout_s
    while True:
        probe = api.get(
            "/api/v1/responses/in-flight",
            params={"conversation_id": conversation_id},
        ).json()
        if probe.get("has_buffer") or time.monotonic() >= deadline:
            return probe
        time.sleep(0.1)


def _await_running_turn(api, conversation_id) -> dict | None:
    """Wait until the server says this conversation has a turn running.

    Returns the probe once `in_flight` is true, or None when the turn provably
    ran to completion before it could be observed running.

    Breaking out of the stream on the first `event:` line is not enough to
    know a turn is cancellable. The formatter yields `response.created` before
    it iterates the model's stream at all, so that frame says the turn was
    registered and nothing more, while `cancel` answers `False` for a turn that
    has already finished. A cancel fired off the back of that frame is
    asserting on a race, and it lost four of the five runs on record.

    Two things answer `in_flight: false` and they are not the same. A turn that
    ran and ended has a buffer on this replica, and no amount of waiting brings
    it back. A replica that never saw this turn has no buffer, and the answer
    changes as soon as the shared store catches up. `has_buffer` is what
    separates them, and it is the only thing that does: the endpoint always
    sends `latest_seq`, as 0 when there is no buffer, so a turn that ended
    having written no records is indistinguishable by that field from a turn
    this replica never heard of. Reading it would report a routing problem as
    "the model was too fast", and a zero-record turn as no buffer at all.
    """
    deadline = time.monotonic() + CANCEL_PREMISE_S
    probe = None
    while time.monotonic() < deadline:
        probe = api.get(
            "/api/v1/responses/in-flight", params={"conversation_id": conversation_id}
        ).json()
        if probe.get("in_flight") is True:
            return probe
        # Not running, and this replica has the turn: it ended. `latest_seq`
        # deliberately does not appear here, see above.
        if probe.get("has_buffer") is True:
            return None
        time.sleep(0.5)
    # Ran out of window without ever seeing a buffer. That is not a fast model,
    # so say so rather than letting the caller blame one.
    raise AssertionError(
        f"no turn was ever visible for this conversation within {CANCEL_PREMISE_S:.0f}s, "
        f"and no buffer appeared either — the turn never reached this replica: {probe}"
    )


def test_cancel_ends_the_turn(api, conversation_id):
    """After POST /cancel, /in-flight reports the turn as finished."""
    with api.stream(
        "POST",
        "/api/v1/responses/",
        json={"input": CANCEL_PROMPT, "conversation": conversation_id, "stream": True},
        timeout=httpx.Timeout(TURN_TIMEOUT_S, connect=10.0),
    ) as resp:
        assert resp.status_code == 200
        # Wait for a frame the MODEL produced, not just the one the formatter
        # emits before it touches the model's stream. `response.created` arrives
        # whether or not anything is generating yet, so disconnecting on it is
        # what put the whole race here in the first place. A second event means
        # the turn is mid-stream at the moment this connection drops.
        seen = 0
        for line in resp.iter_lines():
            if line.startswith("event:"):
                seen += 1
                if seen >= 2:
                    break
        assert seen >= 2, "the turn produced no frame beyond response.created"

    # Establish the premise before asserting on it: there has to be a running
    # turn for "cancel stops it" to mean anything. A skip here means the model
    # finished a 400-step prompt between two HTTP calls, which is a fact about
    # the deployment and not a defect — but it also means this test covered
    # nothing on this run, so read the outcome line and not the job's colour:
    #   gh run view <id> --log | grep test_cancel_ends_the_turn
    running = _await_running_turn(api, conversation_id)
    if running is None:
        pytest.skip(
            "the turn ran to completion before it could be observed running, so "
            "there was nothing to cancel and this run proved nothing; the prompt "
            "needs to outlast two HTTP calls on this deployment"
        )

    cancel = api.post(
        "/api/v1/responses/cancel", json={"conversation_id": conversation_id}
    )
    assert cancel.status_code == 200, cancel.text
    # Now load-bearing: the turn was running one call ago, so False here means
    # the server failed to stop a turn it was told to stop.
    assert (
        cancel.json()["cancelled"] is True
    ), f"cancel reported nothing to cancel for a turn that was in flight: {running}"

    deadline = time.monotonic() + CANCEL_VISIBLE_S
    while time.monotonic() < deadline:
        probe = api.get(
            "/api/v1/responses/in-flight", params={"conversation_id": conversation_id}
        ).json()
        if probe["in_flight"] is False:
            break
        time.sleep(2)
    else:
        pytest.fail(
            f"turn still in flight {CANCEL_VISIBLE_S:.0f}s after cancel: {probe}"
        )


def test_two_successive_turns_complete_and_persist(api, conversation_id):
    """Two turns sent one after the other both finish and are saved in order.

    The second is sent only once the first has ended, which is what the UI
    queue does; a second turn sent while one runs is refused with 409, and
    tests/test_turn_already_running.py covers that.
    """
    expected: list[tuple[str, str, str]] = []
    for prompt in (FIRST_TURN_PROMPT, SECOND_TURN_PROMPT):
        frames: list[SseFrame] = []
        _stream_turn(api, conversation_id, prompt, frames=frames)
        completed = _assert_successful_turn(frames)
        created = next(payload for event, payload in frames if event == "response.created")
        assert created.get("user_message_id"), "created frame carried no user_message_id"
        assert completed.get("assistant_message_id"), (
            "completed frame carried no assistant_message_id"
        )
        expected.append((created["user_message_id"], "user", prompt))
        expected.append((completed["assistant_message_id"], "assistant", _answer_text(completed)))

    items = _conversation_items(api, conversation_id)
    ids = [str(item.get("id")) for item in items]
    positions = []
    for message_id, role, content in expected:
        assert message_id in ids, f"saved turn missing: {role} message {message_id}"
        assert ids.count(message_id) == 1, f"{role} message {message_id} saved more than once"
        item = items[ids.index(message_id)]
        assert item.get("role") == role, f"{message_id}: role {item.get('role')!r}"
        assert item.get("content") == content, f"{message_id}: content differs"
        positions.append(ids.index(message_id))
    for prompt in (FIRST_TURN_PROMPT, SECOND_TURN_PROMPT):
        questions = [
            item for item in items
            if item.get("role") == "user" and item.get("content") == prompt
        ]
        assert len(questions) == 1, f"question saved more than once: {prompt!r}"
    assert positions == sorted(positions), f"saved turns out of order: {positions}"


def test_deleting_a_conversation_leaves_no_replayable_buffer(api, conversation_id):
    """turn_id is the message count, so a deleted conversation's buffer would be
    replayed as the next turn's answer if it survived."""
    assert "response.completed" in _stream_turn(api, conversation_id, QUICK_PROMPT)

    deleted = api.delete(f"/api/v1/conversations/{conversation_id}")
    assert deleted.status_code in (200, 204), deleted.text

    probe = api.get(
        "/api/v1/responses/in-flight", params={"conversation_id": conversation_id}
    ).json()
    assert probe["has_buffer"] is False, probe


def test_deleting_a_scheduled_runs_conversation_keeps_the_run(api):
    """Deleting a conversation a scheduled run produced must release
    schedule_runs.conversation_id and schedules.last_result_conversation_id
    instead of FK-violating into a 500 (ENG-1950), and the run history must
    survive with its verdict intact. The unit suite pins this on an
    FK-enforcing SQLite engine; this is the same cascade against the
    deployment's real alembic-built Postgres."""
    created = api.post(
        "/api/v1/schedules/",
        json={
            "title": "post-deploy delete cascade",
            "prompt": QUICK_PROMPT,
            "cadence": "daily",
            "nextRunAt": "2030-01-01T00:00:00Z",
            "enabled": False,  # run-now only; the cron loop must not pick it up
        },
    )
    assert created.status_code == 201, created.text
    schedule_id = created.json()["id"]
    try:
        run_now = api.post(f"/api/v1/schedules/{schedule_id}/run-now")
        assert run_now.status_code == 202, run_now.text
        conversation_id = run_now.json()["conversation_id"]

        # Wait for the run to reach a terminal status. Which terminal status
        # is the turn's business (test_a_turn_runs_end_to_end owns that);
        # the cascade below must hold for any of them.
        deadline = time.time() + TURN_TIMEOUT_S
        run = None
        while time.time() < deadline:
            listing = api.get(f"/api/v1/schedules/{schedule_id}/runs")
            assert listing.status_code == 200, listing.text
            run = next(
                (
                    r
                    for r in listing.json()["runs"]
                    if r.get("conversationId") == conversation_id
                ),
                None,
            )
            if run is not None and run["status"] != "running":
                break
            time.sleep(2)
        assert (
            run is not None and run["status"] != "running"
        ), f"run never reached a terminal status: {run}"

        deleted = api.delete(f"/api/v1/conversations/{conversation_id}")
        assert deleted.status_code in (200, 204), deleted.text

        after = api.get(f"/api/v1/schedules/{schedule_id}/runs").json()["runs"]
        kept = next((r for r in after if r["id"] == run["id"]), None)
        assert kept is not None, "run history must outlive the chat it produced"
        assert kept["conversationId"] is None
        assert (kept["status"], kept["durationMs"]) == (
            run["status"],
            run["durationMs"],
        )

        schedule = api.get(f"/api/v1/schedules/{schedule_id}")
        assert schedule.status_code == 200, schedule.text
        assert schedule.json()["lastResultConversationId"] is None
    finally:
        api.delete(f"/api/v1/schedules/{schedule_id}")
