"""Cross-tenant isolation for the in-memory run registry endpoints.

The streaming endpoints (/responses/tail, /in-flight, /in-flight-list,
/cancel) resolve turns purely through the process-global RunRegistry, which
is keyed by conversation_id. A conversation_id is NOT an authorization token,
so in org mode these endpoints must refuse to reveal, enumerate, or cancel a
turn owned by another org. Local mode (desktop, single-user) must be
unchanged — no filtering.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from starlette.responses import JSONResponse

from anton.core.interaction.elicit import AskOption, AskRequest
from cowork.api.v1.endpoints import responses as responses_ep
from cowork.api.v1.endpoints.responses import (
    _authorized_handle,
    _require_streaming_scope,
)
from cowork.db.scoped import (
    LOCAL_SCOPE,
    MissingTenantScopeError,
    TenantScope,
    get_tenant_scope,
)
from cowork.streaming.answers import broker
from cowork.streaming.registry import TurnLifecycle, registry

ORG_A = "11111111-1111-1111-1111-111111111111"
ORG_B = "22222222-2222-2222-2222-222222222222"
CID_A = "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"
CID_B = "bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb"


def _org_scope(org_id: str | None) -> TenantScope:
    return TenantScope(org_mode=True, org_id=org_id, user_id=None)


# ── unit: the authorization decision ────────────────────────────────────────

class TestAuthorizedHandle:
    def test_none_handle_is_none(self):
        assert _authorized_handle(None, _org_scope(ORG_A)) is None

    def test_local_mode_never_filters(self):
        # A handle carrying an org is still visible in local mode (desktop):
        # local callers own everything on the instance.
        h = SimpleNamespace(org_id=ORG_A, user_id="author")
        assert _authorized_handle(h, LOCAL_SCOPE) is h

    def test_owner_match_returns_handle(self):
        h = SimpleNamespace(org_id=ORG_A, user_id="alice")
        scope = TenantScope(org_mode=True, org_id=ORG_A, user_id="alice")
        assert _authorized_handle(h, scope) is h

    def test_org_mismatch_reads_as_absent(self):
        h = SimpleNamespace(org_id=ORG_A, user_id="alice")
        scope = TenantScope(org_mode=True, org_id=ORG_B, user_id="alice")
        assert _authorized_handle(h, scope) is None

    def test_same_org_other_user_reads_as_absent(self):
        # Turns are personal: matching org is not enough, the owning user must
        # match too (staging audit P0).
        h = SimpleNamespace(org_id=ORG_A, user_id="alice")
        scope = TenantScope(org_mode=True, org_id=ORG_A, user_id="bob")
        assert _authorized_handle(h, scope) is None


class TestRequireStreamingScope:
    def test_local_mode_never_raises(self):
        _require_streaming_scope(LOCAL_SCOPE)  # no exception

    def test_org_with_id_ok(self):
        _require_streaming_scope(_org_scope(ORG_A))  # no exception

    def test_org_without_id_fails_closed(self):
        with pytest.raises(MissingTenantScopeError):
            _require_streaming_scope(_org_scope(None))


# ── endpoint-level isolation ────────────────────────────────────────────────

class _FakeBuffer:
    def __init__(self, latest_seq: int = 3) -> None:
        self._latest = latest_seq

    @property
    def latest_seq(self) -> int:
        return self._latest

    async def tail(self, from_seq: int = 0):  # closed buffer: nothing to replay
        for _ in ():
            yield _


class _FakeHandle:
    """Duck-typed RunHandle — avoids spawning a real producer task/buffer."""

    def __init__(self, conversation_id: str, org_id: str | None, *, running: bool = True) -> None:
        self.conversation_id = conversation_id
        self.org_id = org_id
        self.turn_id = 1
        self.buffer = _FakeBuffer()
        self.lifecycle = TurnLifecycle()
        self._running = running
        self.cancelled = False

    @property
    def is_running(self) -> bool:
        return self._running

    async def cancel(self) -> bool:
        self.cancelled = True
        self._running = False
        return True


@pytest.fixture(autouse=True)
def _clean_registry():
    saved = dict(registry._by_cid)
    registry._by_cid.clear()
    yield
    registry._by_cid.clear()
    registry._by_cid.update(saved)


@pytest.fixture
def client():
    app = FastAPI()
    app.include_router(responses_ep.router, prefix="/api/v1/responses")

    holder: dict[str, TenantScope] = {"scope": LOCAL_SCOPE}
    app.dependency_overrides[get_tenant_scope] = lambda: holder["scope"]

    @app.exception_handler(MissingTenantScopeError)
    async def _missing(request, exc):  # mirror create_app's handler
        return JSONResponse({"detail": "Unauthorized"}, status_code=401)

    c = TestClient(app)
    c.set_scope = lambda scope: holder.__setitem__("scope", scope)  # type: ignore[attr-defined]
    return c


def _register(conversation_id: str, org_id: str | None, *, user_id: str | None = None) -> _FakeHandle:
    handle = _FakeHandle(conversation_id, org_id)
    handle.user_id = user_id
    registry._by_cid[conversation_id] = handle
    return handle


def test_in_flight_list_scoped_to_caller_org(client):
    _register(CID_A, ORG_A)
    _register(CID_B, ORG_B)

    client.set_scope(_org_scope(ORG_A))
    ids = {r["conversation_id"] for r in client.get("/api/v1/responses/in-flight-list").json()["in_flight"]}
    assert ids == {CID_A}

    client.set_scope(_org_scope(ORG_B))
    ids = {r["conversation_id"] for r in client.get("/api/v1/responses/in-flight-list").json()["in_flight"]}
    assert ids == {CID_B}


def test_in_flight_probe_hides_other_org(client):
    _register(CID_B, ORG_B)
    client.set_scope(_org_scope(ORG_A))
    body = client.get("/api/v1/responses/in-flight", params={"conversation_id": CID_B}).json()
    assert body["in_flight"] is False and body["has_buffer"] is False

    client.set_scope(_org_scope(ORG_B))
    body = client.get("/api/v1/responses/in-flight", params={"conversation_id": CID_B}).json()
    assert body["in_flight"] is True and body["has_buffer"] is True


def test_cancel_across_org_is_404_and_noop(client):
    handle_b = _register(CID_B, ORG_B)
    client.set_scope(_org_scope(ORG_A))
    resp = client.post("/api/v1/responses/cancel", json={"conversation_id": CID_B})
    assert resp.status_code == 404  # indistinguishable from an unknown id
    assert handle_b.cancelled is False  # the victim turn keeps running

    client.set_scope(_org_scope(ORG_B))
    resp = client.post("/api/v1/responses/cancel", json={"conversation_id": CID_B})
    assert resp.status_code == 200 and resp.json()["cancelled"] is True
    assert handle_b.cancelled is True


def test_cancel_unknown_id_is_also_404(client):
    # No existence oracle: a foreign id (above) and a never-seen id look the same.
    client.set_scope(_org_scope(ORG_A))
    resp = client.post("/api/v1/responses/cancel", json={"conversation_id": CID_B})
    assert resp.status_code == 404


def test_same_org_other_user_cannot_see_or_cancel(client):
    # Staging audit P0: turns are personal. A teammate (different user) in the
    # same org must NOT tail/cancel a turn another member started — matching org
    # alone is not enough; the owning user must match too.
    handle = _register(CID_A, ORG_A, user_id="alice")
    client.set_scope(TenantScope(org_mode=True, org_id=ORG_A, user_id="bob"))
    assert client.get("/api/v1/responses/tail", params={"conversation_id": CID_A}).status_code == 404
    resp = client.post("/api/v1/responses/cancel", json={"conversation_id": CID_A})
    assert resp.status_code == 404
    assert handle.cancelled is False  # Alice's turn keeps running

    # The owner (alice) still can.
    client.set_scope(TenantScope(org_mode=True, org_id=ORG_A, user_id="alice"))
    assert client.get("/api/v1/responses/tail", params={"conversation_id": CID_A}).status_code == 200
    resp = client.post("/api/v1/responses/cancel", json={"conversation_id": CID_A})
    assert resp.status_code == 200 and resp.json()["cancelled"] is True
    assert handle.cancelled is True


def test_tail_across_org_is_404(client):
    _register(CID_B, ORG_B)
    client.set_scope(_org_scope(ORG_A))
    assert client.get("/api/v1/responses/tail", params={"conversation_id": CID_B}).status_code == 404

    client.set_scope(_org_scope(ORG_B))
    assert client.get("/api/v1/responses/tail", params={"conversation_id": CID_B}).status_code == 200


def test_org_mode_without_identity_is_401_even_when_empty(client):
    # Point 3: scope is validated before registry access, so the missing-
    # identity case fails closed regardless of whether the target exists —
    # an empty list or unknown id must not mask it behind a 200/404.
    client.set_scope(_org_scope(None))
    assert client.get("/api/v1/responses/in-flight-list").status_code == 401  # registry empty
    assert client.get(
        "/api/v1/responses/in-flight", params={"conversation_id": CID_A}
    ).status_code == 401  # unknown id
    assert client.get("/api/v1/responses/tail", params={"conversation_id": CID_A}).status_code == 401
    assert client.post(
        "/api/v1/responses/cancel", json={"conversation_id": CID_A}
    ).status_code == 401


def test_local_mode_sees_everything(client):
    # Desktop: no org filtering, even for handles that happen to carry an org.
    _register(CID_A, ORG_A)
    _register(CID_B, ORG_B)
    client.set_scope(LOCAL_SCOPE)
    ids = {r["conversation_id"] for r in client.get("/api/v1/responses/in-flight-list").json()["in_flight"]}
    assert ids == {CID_A, CID_B}
    assert client.get("/api/v1/responses/tail", params={"conversation_id": CID_A}).status_code == 200


# ── personal partitions (local tenancy with trusted identity headers) ──────

PARTITION_A = "44e92829-e701-599b-88f9-69f5bf1165eb"
PARTITION_B = "55e92829-e701-599b-88f9-69f5bf1165eb"


def _personal_scope(
    *, org_id: str | None, user_id: str | None, resources=None
) -> TenantScope:
    from cowork.db.personal_resources import ALL_PERSONAL_RESOURCES

    return TenantScope(
        personal=True,
        org_id=org_id,
        user_id=user_id,
        personal_resources=ALL_PERSONAL_RESOURCES if resources is None else resources,
    )


def test_personal_scope_hides_and_refuses_another_persons_turn(client):
    # Two people, each in their own partition: B cannot list, probe, tail,
    # cancel or answer A's running answer, and A's turn keeps running.
    handle_a = _register(CID_A, PARTITION_A, user_id="alice")
    client.set_scope(_personal_scope(org_id=PARTITION_B, user_id="bob"))

    listed = client.get("/api/v1/responses/in-flight-list").json()["in_flight"]
    assert listed == []
    probe = client.get("/api/v1/responses/in-flight", params={"conversation_id": CID_A}).json()
    assert probe["in_flight"] is False and probe["has_buffer"] is False
    assert client.get("/api/v1/responses/tail", params={"conversation_id": CID_A}).status_code == 404
    resp = client.post("/api/v1/responses/cancel", json={"conversation_id": CID_A})
    assert resp.status_code == 404
    answer = client.post(
        "/api/v1/responses/answer",
        json={"conversation_id": CID_A, "question_id": "q1", "text": "yes"},
    )
    assert answer.status_code == 404
    assert handle_a.cancelled is False

    client.set_scope(_personal_scope(org_id=PARTITION_A, user_id="alice"))
    ids = {r["conversation_id"] for r in client.get("/api/v1/responses/in-flight-list").json()["in_flight"]}
    assert ids == {CID_A}
    resp = client.post("/api/v1/responses/cancel", json={"conversation_id": CID_A})
    assert resp.status_code == 200 and resp.json()["cancelled"] is True


ASKED = AskRequest(prompt="Which database?", options=(AskOption(value="pg", label="postgres"),))


async def test_only_the_asker_answers_a_question_her_running_answer_asked(client):
    # A real pending question: a missing one also answers 404, so B's 404
    # proves the owner check only when A's own answer then lands.
    _register(CID_A, PARTITION_A, user_id="alice")
    broker.open(CID_A, "ask:1", ASKED)
    try:
        body = {"conversation_id": CID_A, "question_id": "ask:1", "values": ["pg"]}
        client.set_scope(_personal_scope(org_id=PARTITION_B, user_id="bob"))
        his = client.post("/api/v1/responses/answer", json=body)
        assert (his.status_code, his.json()) == (404, {"status": "not_found"})
        assert not broker._pending[(CID_A, "ask:1")].future.done()

        client.set_scope(_personal_scope(org_id=PARTITION_A, user_id="alice"))
        hers = client.post("/api/v1/responses/answer", json=body)
        assert (hers.status_code, hers.json()) == (200, {"accepted": True})
    finally:
        broker.reset()


async def test_with_shared_conversations_anyone_answers_a_running_answers_question(client):
    _register(CID_A, PARTITION_A, user_id="alice")
    broker.open(CID_A, "ask:1", ASKED)
    try:
        client.set_scope(_personal_scope(org_id=PARTITION_B, user_id="bob", resources=frozenset()))
        his = client.post(
            "/api/v1/responses/answer",
            json={"conversation_id": CID_A, "question_id": "ask:1", "values": ["pg"]},
        )
        assert (his.status_code, his.json()) == (200, {"accepted": True})
    finally:
        broker.reset()


def test_operator_view_sees_every_persons_turn(client):
    # The operator view of a trusted-identity deployment is LOCAL_SCOPE: it
    # lists and reaches every person's running answer.
    _register(CID_A, PARTITION_A, user_id="alice")
    _register(CID_B, PARTITION_B, user_id="bob")
    client.set_scope(LOCAL_SCOPE)
    ids = {r["conversation_id"] for r in client.get("/api/v1/responses/in-flight-list").json()["in_flight"]}
    assert ids == {CID_A, CID_B}


def test_personal_scope_with_shared_conversations_keeps_todays_behavior(client):
    # Conversations left off the personal list stay shared: every person sees
    # and can stop every running answer, as on today's shared stack.
    handle_a = _register(CID_A, PARTITION_A, user_id="alice")
    client.set_scope(_personal_scope(org_id=PARTITION_B, user_id="bob", resources=frozenset()))
    ids = {r["conversation_id"] for r in client.get("/api/v1/responses/in-flight-list").json()["in_flight"]}
    assert ids == {CID_A}
    resp = client.post("/api/v1/responses/cancel", json={"conversation_id": CID_A})
    assert resp.status_code == 200 and handle_a.cancelled is True


def test_personal_scope_without_partition_fails_closed():
    with pytest.raises(MissingTenantScopeError):
        _require_streaming_scope(_personal_scope(org_id=None, user_id="alice"))
    # Shared conversations have nothing to key, so nothing to refuse.
    _require_streaming_scope(_personal_scope(org_id=None, user_id="alice", resources=frozenset()))
