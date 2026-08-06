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
from cowork.streaming.registry import registry

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
        h = SimpleNamespace(org_id=ORG_A)
        assert _authorized_handle(h, LOCAL_SCOPE) is h

    def test_org_match_returns_handle(self):
        h = SimpleNamespace(org_id=ORG_A)
        assert _authorized_handle(h, _org_scope(ORG_A)) is h

    def test_org_mismatch_reads_as_absent(self):
        h = SimpleNamespace(org_id=ORG_A)
        assert _authorized_handle(h, _org_scope(ORG_B)) is None

    def test_user_is_not_an_authorization_gate(self):
        # Conversations are org-shared: a handle authored by one user is
        # visible to any caller in the same org.
        h = SimpleNamespace(org_id=ORG_A, user_id="author")
        assert _authorized_handle(h, _org_scope(ORG_A)) is h


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


def test_same_org_other_user_can_see_and_cancel(client):
    # Point 1: conversations are org-shared — a teammate (different user) in the
    # same org may tail and cancel a turn another member started.
    handle = _register(CID_A, ORG_A, user_id="alice")
    client.set_scope(TenantScope(org_mode=True, org_id=ORG_A, user_id="bob"))
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
