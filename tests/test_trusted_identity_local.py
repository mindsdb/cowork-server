"""COWORK_TRUST_IDENTITY_HEADERS through create_app: the startup refusals, the
middleware's place inside the bearer check, and two people kept apart over HTTP.

The two-person cases here cover what this layer decides on its own: the scope
each request gets, the rows it stamps, and the 404 for a turn posted into
another person's conversation. Pins and schedules, which no per-service gate
covers beyond the scoped session's own filter, get their two-person cases
here too: B lists, reads, changes and deletes none of A's, the operator view
lists both, and a deployment that leaves them off the list shares them. The
other per-service gates have their own tests; one matrix here sends B's
read, change and delete at each of A's other items through the real routes.
A shared project another person still works in is never deleted under them.
"""
from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from uuid import UUID, uuid4

import pytest
from fastapi import Depends
from fastapi.testclient import TestClient
from sqlalchemy import delete
from sqlmodel import Session

# Imported at module scope on purpose: cowork.server builds an app at import
# time, so importing it inside a test would raise before pytest.raises saw it.
import cowork.server as cowork_server
from cowork.common.settings.app_settings import get_app_settings
from cowork.db.personal_resources import (
    ALL_PERSONAL_RESOURCES,
    PersonalResource,
    PersonalResourceDependencyError,
)
from cowork.db.scoped import LOCAL_SCOPE, TenantScope, get_tenant_scope
from cowork.db.session import get_engine
from cowork.models.conversation import Conversation
from cowork.models.file import File
from cowork.models.pin import Pin
from cowork.models.project import Project
from cowork.models.schedule import Schedule
from cowork.models.setting import Setting
from cowork.principal import TrustedHeaderMiddleware, personal_partition_id

TOKEN = "trusted-identity-test-token"
DEPLOYMENT = "5f0c7d2e-8b1a-4c3e-9d4f-2a6b8c0e1f3a"
SUB_A = "11111111-1111-4111-8111-111111111111"
SUB_B = "22222222-2222-4222-8222-222222222222"
BEARER = {"Authorization": f"Bearer {TOKEN}"}
A = {**BEARER, "X-User-Id": SUB_A, "X-Organization-Id": DEPLOYMENT}
B = {**BEARER, "X-User-Id": SUB_B, "X-Organization-Id": DEPLOYMENT}


@pytest.fixture
def trusted(personal_deployment, monkeypatch, tmp_path):
    """create_app's environment on a trusted-identity stack. COWORK_HOME is the
    test's own folder, since create_app mirrors the bearer token into its .env.
    The two people's rows are removed afterwards, so later modules that list
    every row don't see them."""
    monkeypatch.setenv("COWORK_HOME", str(tmp_path))
    monkeypatch.setenv("COWORK_AUTH_TOKEN", TOKEN)
    monkeypatch.setenv("COWORK_IDENTITY_ENFORCE", "audit")
    get_app_settings.cache_clear()
    yield monkeypatch
    partitions = [personal_partition_id(DEPLOYMENT, sub) for sub in (SUB_A, SUB_B)]
    with Session(get_engine(get_app_settings().database.uri)) as raw:
        raw.exec(delete(Pin).where(Pin.user_id.in_([SUB_A, SUB_B])))
        raw.exec(delete(Schedule).where(Schedule.created_by.in_([SUB_A, SUB_B])))
        raw.exec(delete(Conversation).where(Conversation.org_id.in_(partitions)))
        raw.exec(delete(Project).where(Project.org_id.in_(partitions)))
        raw.exec(delete(File).where(File.org_id.in_(partitions)))
        raw.exec(delete(Setting).where(Setting.org_id.in_(partitions)))
        raw.commit()


def _client(*, scope_route: bool = False) -> TestClient:
    app = cowork_server.create_app()
    if scope_route:
        # Answers 200 on its own, so the scope it returns is the only signal.
        @app.get("/api/v1/_scope")
        def _scope(scope: TenantScope = Depends(get_tenant_scope)):
            return {
                "org_mode": scope.org_mode,
                "personal": scope.personal,
                "org_id": scope.org_id,
                "user_id": scope.user_id,
                "local": scope == LOCAL_SCOPE,
            }

    # No `with`: skipping the lifespan skips boot migrations and background workers.
    return TestClient(app)


def _loopback_client() -> TestClient:
    """A client on the loopback address, as the stack's proxy is not: some
    routes (a connection's delete among them) answer only there."""
    return TestClient(
        cowork_server.create_app(), base_url="http://127.0.0.1:26866", client=("127.0.0.1", 54321)
    )


# ── startup refusals ─────────────────────────────────────────────────────────

def test_refuses_without_require_auth(trusted):
    trusted.setenv("COWORK_REQUIRE_AUTH", "false")
    get_app_settings.cache_clear()
    with pytest.raises(RuntimeError, match="requires COWORK_REQUIRE_AUTH"):
        cowork_server.create_app()


def test_refuses_in_org_tenancy(trusted):
    trusted.setenv("COWORK_TENANCY_MODE", "org")
    trusted.setenv("COWORK_REQUIRE_AUTH", "false")
    get_app_settings.cache_clear()
    with pytest.raises(RuntimeError, match="COWORK_TRUST_IDENTITY_HEADERS is not supported in org tenancy"):
        cowork_server.create_app()


def test_refuses_a_broken_resource_dependency(trusted):
    trusted.setenv("COWORK_PERSONAL_RESOURCES", "projects,files,pins,schedules")
    get_app_settings.cache_clear()
    with pytest.raises(PersonalResourceDependencyError, match="projects without conversations"):
        cowork_server.create_app()


def test_a_broken_dependency_is_ignored_with_trust_off(trusted):
    trusted.setenv("COWORK_TRUST_IDENTITY_HEADERS", "false")
    trusted.setenv("COWORK_PERSONAL_RESOURCES", "projects")
    get_app_settings.cache_clear()
    cowork_server.create_app()


@pytest.mark.parametrize("resources", ["", "pins,schedules", "projects,conversations,files,pins,schedules"])
def test_refuses_memory_left_off_the_list(trusted, resources):
    """Memory left off is one store every person's turns learn into, so a
    deployment that names people refuses it, the empty list included."""
    trusted.setenv("COWORK_PERSONAL_RESOURCES", resources)
    get_app_settings.cache_clear()
    with pytest.raises(RuntimeError, match="needs memory in COWORK_PERSONAL_RESOURCES") as refused:
        cowork_server.create_app()
    assert "read the others' lessons" in str(refused.value)


@pytest.mark.parametrize("resources", ["memory", "projects,conversations,files,pins,schedules,memory"])
def test_boots_with_memory_and_any_resource_shared(trusted, resources):
    trusted.setenv("COWORK_PERSONAL_RESOURCES", resources)
    get_app_settings.cache_clear()
    cowork_server.create_app()


def test_memory_left_off_is_ignored_with_trust_off(trusted):
    trusted.setenv("COWORK_TRUST_IDENTITY_HEADERS", "false")
    trusted.setenv("COWORK_PERSONAL_RESOURCES", "")
    get_app_settings.cache_clear()
    cowork_server.create_app()


def test_refuses_remote_turns(trusted):
    """A remote turn keeps memory in the project's own folder, not each
    person's, so remote turns are refused where identity is trusted."""
    trusted.setenv("COWORK_TURN_BACKEND", "remote")
    get_app_settings.cache_clear()
    with pytest.raises(RuntimeError, match="COWORK_TURN_BACKEND=remote is not supported"):
        cowork_server.create_app()


def test_boots_and_says_so(trusted, caplog, owned_logger):
    trusted.setenv("COWORK_PERSONAL_RESOURCES", "memory,skills")
    get_app_settings.cache_clear()
    owned_logger("cowork.common.logger", level=logging.INFO)
    cowork_server.create_app()
    lines = [
        record.getMessage()
        for record in caplog.records
        if record.name == "cowork.common.logger" and record.levelno == logging.INFO
    ]
    assert "auth: local tenancy, trusted identity headers enabled (identity=audit, personal resources=memory,skills)" in lines


def test_boots_with_every_resource_by_default(trusted):
    assert get_app_settings().personal_resources == ALL_PERSONAL_RESOURCES
    cowork_server.create_app()


# ── the middleware sits inside the bearer check ──────────────────────────────

def test_identity_headers_are_read_only_after_the_bearer(trusted, monkeypatch):
    """The identity layer never runs for a request without the token, and
    runs once for a request with it. A 401 alone can't show the order: the
    bearer layer answers 401 to a pair without the token either way."""
    calls: list[str] = []
    dispatch = TrustedHeaderMiddleware.dispatch

    async def spy(self, request, call_next):
        calls.append(request.url.path)
        return await dispatch(self, request, call_next)

    monkeypatch.setattr(TrustedHeaderMiddleware, "dispatch", spy)
    client = _client(scope_route=True)

    res = client.get("/api/v1/_scope", headers={"X-User-Id": SUB_A, "X-Organization-Id": DEPLOYMENT})
    assert res.status_code == 401
    assert calls == []
    assert client.get("/api/v1/_scope", headers=A).status_code == 200
    assert calls == ["/api/v1/_scope"]


def test_options_naming_a_person_without_the_bearer_is_401(trusted):
    """Only a genuine preflight, which CORS answers, passes without the
    token; an OPTIONS that names a person is refused on a route that accepts
    OPTIONS."""
    client = _client()
    forged = {"X-User-Id": SUB_A, "X-Organization-Id": DEPLOYMENT}
    for headers in (forged, {**forged, "Access-Control-Request-Method": "POST"}):
        assert client.options("/api/v1/responses/", headers=headers).status_code == 401
    assert client.options("/api/v1/responses/", headers=A).status_code == 200


@pytest.mark.parametrize(
    "pair",
    [{"X-User-Id": SUB_A}, {"X-User-Id": "f:ldap:alice", "X-Organization-Id": DEPLOYMENT}],
)
def test_a_broken_pair_under_audit_is_401(trusted, pair):
    res = _client(scope_route=True).get("/api/v1/_scope", headers={**BEARER, **pair})
    assert res.status_code == 401


def test_bearer_and_pair_give_a_personal_scope(trusted):
    res = _client(scope_route=True).get("/api/v1/_scope", headers=A)
    assert res.status_code == 200, res.text
    assert res.json() == {
        "org_mode": False,
        "personal": True,
        "org_id": personal_partition_id(DEPLOYMENT, SUB_A),
        "user_id": SUB_A,
        "local": False,
    }


def test_bearer_without_a_pair_under_audit_is_the_operator_view(trusted):
    res = _client(scope_route=True).get("/api/v1/_scope", headers=BEARER)
    assert res.status_code == 200, res.text
    assert res.json()["local"] is True


def test_bearer_without_a_pair_under_enforce_is_401(trusted):
    trusted.setenv("COWORK_IDENTITY_ENFORCE", "enforce")
    get_app_settings.cache_clear()
    assert _client(scope_route=True).get("/api/v1/_scope", headers=BEARER).status_code == 401


def test_desktop_ignores_identity_headers(trusted):
    """Trust off: the same request, pair included, is LOCAL_SCOPE."""
    trusted.setenv("COWORK_TRUST_IDENTITY_HEADERS", "false")
    get_app_settings.cache_clear()
    res = _client(scope_route=True).get("/api/v1/_scope", headers=A)
    assert res.status_code == 200, res.text
    assert res.json()["local"] is True


# ── two people over HTTP ────────────────────────────────────────────────────

def _project(client: TestClient, headers: dict) -> str:
    res = client.post("/api/v1/projects/", json={"name": f"p-{uuid4().hex[:6]}"}, headers=headers)
    assert res.status_code == 201, res.text
    return res.json()["id"]


def _conversation(client: TestClient, headers: dict, *, project_id: str, topic: str) -> str:
    res = client.post("/api/v1/conversations/", json={"topic": topic, "projectId": project_id}, headers=headers)
    assert res.status_code == 201, res.text
    return res.json()["id"]


def test_new_rows_carry_the_partition_and_the_sub(trusted):
    client = _client()
    pid = _project(client, A)
    cid = _conversation(client, A, project_id=pid, topic="a-stamped")
    with Session(get_engine(get_app_settings().database.uri)) as raw:
        project = raw.get(Project, UUID(pid))
        conversation = raw.get(Conversation, UUID(cid))
    partition = personal_partition_id(DEPLOYMENT, SUB_A)
    assert (project.org_id, project.created_by) == (partition, SUB_A)
    assert (conversation.org_id, conversation.created_by) == (partition, SUB_A)


def test_two_people_and_the_operator(trusted):
    client = _client()
    a_pid = _project(client, A)
    a_cid = _conversation(client, A, project_id=a_pid, topic="a-private")
    b_pid = _project(client, B)
    b_cid = _conversation(client, B, project_id=b_pid, topic="b-private")

    b_list = client.get("/api/v1/conversations/?project=all", headers=B).json()["conversations"]
    assert a_cid not in {c["id"] for c in b_list}
    assert b_cid in {c["id"] for c in b_list}
    assert a_pid not in {p["id"] for p in client.get("/api/v1/projects/", headers=B).json()}
    assert client.get(f"/api/v1/conversations/{a_cid}", headers=B).status_code == 404
    assert client.get(f"/api/v1/conversations/{a_cid}/items", headers=B).status_code == 404

    # The shared token with no person lists both.
    operator = client.get("/api/v1/conversations/?project=all&limit=500", headers=BEARER)
    assert operator.status_code == 200, operator.text
    assert {a_cid, b_cid} <= {c["id"] for c in operator.json()["conversations"]}


@dataclass(frozen=True)
class CrossRequest:
    """One request B sends at A's item, and the status B must get."""

    method: str
    path: str
    json: dict | None = None
    status: int = 404


@dataclass(frozen=True)
class CrossCell:
    """B's requests at one of A's items, and the check that A's item is
    intact afterwards and nothing of it reached B."""

    requests: tuple[CrossRequest, ...]
    intact: Callable[[], None]


def _conversation_cell(client: TestClient) -> CrossCell:
    cid = _conversation(client, A, project_id=_project(client, A), topic="a-topic")
    b_pid = _project(client, B)

    def intact() -> None:
        mine = client.get(f"/api/v1/conversations/{cid}", headers=A)
        assert mine.status_code == 200 and mine.json()["title"] == "a-topic"
        with Session(get_engine(get_app_settings().database.uri)) as raw:
            assert str(raw.get(Conversation, UUID(cid)).project_id) != b_pid

    return CrossCell(
        requests=(
            CrossRequest("PATCH", f"/api/v1/conversations/{cid}", {"topic": "b-took-it"}),
            CrossRequest("POST", f"/api/v1/conversations/{cid}/move", {"projectId": b_pid}),
            CrossRequest("DELETE", f"/api/v1/conversations/{cid}"),
        ),
        intact=intact,
    )


def _project_cell(client: TestClient) -> CrossCell:
    name = f"a-{uuid4().hex[:6]}"
    res = client.post("/api/v1/projects/", json={"name": name}, headers=A)
    assert res.status_code == 201, res.text
    pid = res.json()["id"]

    def intact() -> None:
        mine = {p["id"]: p["name"] for p in client.get("/api/v1/projects/", headers=A).json()}
        assert mine.get(pid) == name

    return CrossCell(
        requests=(
            CrossRequest("PATCH", f"/api/v1/projects/{pid}", {"name": "b-took-it"}),
            CrossRequest("DELETE", f"/api/v1/projects/{pid}"),
            CrossRequest("GET", f"/api/v1/projects/{name}/instructions"),
            CrossRequest("GET", f"/api/v1/projects/{name}/files"),
        ),
        intact=intact,
    )


def _file_cell(client: TestClient) -> CrossCell:
    res = client.post(
        "/api/v1/files/", files={"file": ("a.txt", b"alice's", "text/plain")},
        data={"purpose": "assistants"}, headers=A,
    )
    assert res.status_code == 201, res.text
    fid = res.json()["id"]

    def intact() -> None:
        assert fid not in {f["id"] for f in client.get("/api/v1/files/", headers=B).json()["data"]}
        assert client.get(f"/api/v1/files/{fid}/content", headers=A).content == b"alice's"

    return CrossCell(
        requests=(
            CrossRequest("GET", f"/api/v1/files/{fid}"),
            CrossRequest("GET", f"/api/v1/files/{fid}/content"),
            CrossRequest("DELETE", f"/api/v1/files/{fid}"),
        ),
        intact=intact,
    )


def _skill_cell(client: TestClient) -> CrossCell:
    res = client.post("/api/v1/skills/", json={"label": "a-private-skill", "instructions": "Alice's."}, headers=A)
    assert res.status_code in (200, 201), res.text

    def intact() -> None:
        assert client.get("/api/v1/skills/a-private-skill", headers=A).status_code == 200
        assert "a-private-skill" not in {s["id"] for s in client.get("/api/v1/skills/", headers=B).json()["skills"]}

    return CrossCell(requests=(CrossRequest("GET", "/api/v1/skills/a-private-skill"),), intact=intact)


def _connection_cell(client: TestClient) -> CrossCell:
    from cowork.db.scoped import scope_from_principal
    from cowork.principal import Principal
    from cowork.services.connectors.persist import persist_connection

    alice = scope_from_principal(Principal(user_id=SUB_A, org_id=personal_partition_id(DEPLOYMENT, SUB_A)))
    persist_connection("postgres", None, "sales", {"host": "h", "password": "alicePw123"}, scope=alice)

    def intact() -> None:
        assert [c["name"] for c in client.get("/api/v1/connectors/connections/", headers=A).json()] == ["sales"]
        assert client.get("/api/v1/connectors/connections/", headers=B).json() == []

    return CrossCell(
        requests=(
            CrossRequest("GET", "/api/v1/connectors/connections/postgres/sales"),
            CrossRequest("DELETE", "/api/v1/connectors/connections/postgres/sales"),
        ),
        intact=intact,
    )


def _setting_cell(client: TestClient) -> CrossCell:
    assert client.put("/api/v1/settings/tone", json={"value": "spicy"}, headers=A).status_code == 200

    def tone(headers: dict) -> str:
        return {s["key"]: s["value"] for s in client.get("/api/v1/settings/", headers=headers).json()}["tone"]

    def intact() -> None:
        assert tone(A) == "spicy" and tone(B) != "spicy"

    return CrossCell(requests=(CrossRequest("DELETE", "/api/v1/settings/tone"),), intact=intact)


def _memory_cell(client: TestClient) -> CrossCell:
    pid = _project(client, A)
    lesson = {"scope": "project", "category": "lessons", "content": "- Pallets of 40", "project_id": pid}
    assert client.put("/api/v1/memory/", json=lesson, headers=A).status_code == 200

    def contents(headers: dict, project_id: str) -> list[str]:
        res = client.get("/api/v1/memory/", params={"project_id": project_id}, headers=headers)
        assert res.status_code == 200, res.text
        return [entry["content"] for entry in res.json()]

    def intact() -> None:
        assert any("Pallets of 40" in content for content in contents(A, pid))
        # B reads only B's own tiers at A's id: the answer for an unknown id.
        assert contents(B, pid) == contents(B, str(uuid4()))
        assert not any("Pallets of 40" in content for content in contents(B, pid))

    return CrossCell(requests=(CrossRequest("GET", f"/api/v1/memory/?project_id={pid}", status=200),), intact=intact)


@pytest.mark.parametrize(
    "cell",
    [_conversation_cell, _project_cell, _file_cell, _skill_cell, _connection_cell, _setting_cell, _memory_cell],
    ids=lambda cell: cell.__name__.strip("_").removesuffix("_cell"),
)
def test_b_reads_changes_and_deletes_none_of_as_items(trusted, cell):
    """Beside the listings above: every read, change and delete B sends at
    one of A's items is refused as if it did not exist, and A's item stays."""
    client = _loopback_client()
    made = cell(client)
    for request in made.requests:
        res = client.request(request.method, request.path, json=request.json, headers=B)
        assert res.status_code == request.status, (request, res.text)
    made.intact()


def test_turn_into_another_persons_conversation_is_404(trusted):
    client = _client()
    a_cid = _conversation(client, A, project_id=_project(client, A), topic="a-turn-target")
    b_pid = _project(client, B)

    res = client.post(
        "/api/v1/responses/", headers=B,
        json={"input": "hi", "stream": True, "conversation": a_cid, "project_id": b_pid},
    )

    assert res.status_code == 404, res.text
    assert res.json() == {"detail": "Conversation not found"}
    assert client.get(f"/api/v1/conversations/{a_cid}/items", headers=A).json() == []


def test_turn_under_another_persons_project_is_404(trusted):
    client = _client()
    a_pid = _project(client, A)

    cross = client.post("/api/v1/responses/", headers=B, json={"input": "hi", "stream": True, "project_id": a_pid})
    missing_id = str(uuid4())
    missing = client.post("/api/v1/responses/", headers=B, json={"input": "hi", "stream": True, "project_id": missing_id})

    assert cross.status_code == missing.status_code == 404, (cross.text, missing.text)
    assert cross.content.replace(a_pid.encode(), b"<ID>") == missing.content.replace(missing_id.encode(), b"<ID>")
    a_convs = client.get(f"/api/v1/conversations/?project_id={a_pid}", headers=A).json()
    assert a_convs["conversations"] == []


# ── pins and schedules, two people over HTTP ─────────────────────────────────

# Every resource personal (the default), and only pins and schedules personal
# beside memory, which a trusted deployment always keeps per person: the
# second case fails if either table is ever mapped to conversations or
# projects instead of its own resource.
PINS_AND_SCHEDULES_PERSONAL = [None, "pins,schedules,memory"]


@pytest.fixture
def made_projects(trusted):
    """Ids of the projects a test makes. Rows a shared resource keeps carry no
    partition and no author, so they are removed by project here."""
    made: list[str] = []
    yield made
    ids = [UUID(pid) for pid in made]
    with Session(get_engine(get_app_settings().database.uri)) as raw:
        raw.exec(delete(Schedule).where(Schedule.project_id.in_(ids)))
        raw.exec(delete(Conversation).where(Conversation.project_id.in_(ids)))
        raw.exec(delete(Project).where(Project.id.in_(ids)))
        raw.commit()


def _personal_resources(trusted, resources: str | None) -> None:
    if resources is not None:
        trusted.setenv("COWORK_PERSONAL_RESOURCES", resources)
        get_app_settings.cache_clear()


def _made_project(client: TestClient, headers: dict, made: list[str]) -> str:
    pid = _project(client, headers)
    made.append(pid)
    return pid


def _pin(client: TestClient, headers: dict, *, item_id: str, title: str) -> None:
    res = client.post(
        "/api/v1/pins/", json={"itemType": "conversation", "itemId": item_id, "title": title}, headers=headers
    )
    assert res.status_code == 201, res.text


def _pins(client: TestClient, headers: dict) -> list[tuple[str, str]]:
    res = client.get("/api/v1/pins/", headers=headers)
    assert res.status_code == 200, res.text
    return [(pin["item_id"], pin["title"]) for pin in res.json()["pins"]]


def _schedule(client: TestClient, headers: dict, *, project_id: str, title: str) -> str:
    res = client.post(
        "/api/v1/schedules/",
        json={
            "title": title, "prompt": "do it", "cadence": "daily",
            "nextRunAt": "2027-01-01T00:00:00Z", "projectId": project_id,
        },
        headers=headers,
    )
    assert res.status_code == 201, res.text
    return res.json()["id"]


def _schedule_ids(client: TestClient, headers: dict) -> set[str]:
    res = client.get("/api/v1/schedules/", headers=headers)
    assert res.status_code == 200, res.text
    return {schedule["id"] for schedule in res.json()["schedules"]}


@pytest.mark.parametrize("resources", PINS_AND_SCHEDULES_PERSONAL)
def test_two_people_keep_their_own_pins(trusted, made_projects, resources):
    _personal_resources(trusted, resources)
    client = _client()
    a_cid = _conversation(client, A, project_id=_made_project(client, A, made_projects), topic="a-pinned")
    _pin(client, A, item_id=a_cid, title="a-title")
    b_cid = _conversation(client, B, project_id=_made_project(client, B, made_projects), topic="b-pinned")
    _pin(client, B, item_id=b_cid, title="b-title")

    assert _pins(client, B) == [(b_cid, "b-title")]
    unpinned = client.delete(f"/api/v1/pins/{a_cid}", params={"item_type": "conversation"}, headers=B)
    assert unpinned.status_code == 404
    # Pinning the same item makes B's own pin; A's stays A's.
    _pin(client, B, item_id=a_cid, title="b-renamed")
    assert _pins(client, A) == [(a_cid, "a-title")]

    operator = _pins(client, BEARER)
    assert {(a_cid, "a-title"), (a_cid, "b-renamed"), (b_cid, "b-title")} <= set(operator)


@pytest.mark.parametrize("resources", PINS_AND_SCHEDULES_PERSONAL)
def test_two_people_keep_their_own_schedules(trusted, made_projects, resources):
    _personal_resources(trusted, resources)
    client = _client()
    a_sid = _schedule(client, A, project_id=_made_project(client, A, made_projects), title="a-schedule")
    b_sid = _schedule(client, B, project_id=_made_project(client, B, made_projects), title="b-schedule")

    assert _schedule_ids(client, B) == {b_sid}
    for method, path, body in (
        ("GET", f"/api/v1/schedules/{a_sid}", None),
        ("GET", f"/api/v1/schedules/{a_sid}/runs", None),
        ("PUT", f"/api/v1/schedules/{a_sid}", {"title": "b-took-it"}),
        ("PATCH", f"/api/v1/schedules/{a_sid}", {"enabled": False}),
        ("POST", f"/api/v1/schedules/{a_sid}/pause", None),
        ("POST", f"/api/v1/schedules/{a_sid}/resume", None),
        ("POST", f"/api/v1/schedules/{a_sid}/run-now", None),
        ("DELETE", f"/api/v1/schedules/{a_sid}", None),
    ):
        kwargs = {"json": body} if body is not None else {}
        res = client.request(method, path, headers=B, **kwargs)
        assert res.status_code == 404, (method, path, res.text)

    mine = client.get(f"/api/v1/schedules/{a_sid}", headers=A)
    assert mine.status_code == 200, mine.text
    assert (mine.json()["title"], mine.json()["enabled"]) == ("a-schedule", True)
    assert {a_sid, b_sid} <= _schedule_ids(client, BEARER)


def test_pins_and_schedules_left_off_the_list_are_shared(trusted, made_projects):
    """Left off, with every resource that does not need them still personal.
    This fails if either table is ever mapped to one of those resources."""
    _personal_resources(trusted, "files,skills,connections,memory,settings")
    client = _client()
    pid = _made_project(client, A, made_projects)
    cid = _conversation(client, A, project_id=pid, topic="team-pinned")
    _pin(client, A, item_id=cid, title="team-title")
    sid = _schedule(client, A, project_id=pid, title="team-schedule")

    assert (cid, "team-title") in _pins(client, B)
    assert sid in _schedule_ids(client, B)
    assert client.get(f"/api/v1/schedules/{sid}", headers=B).status_code == 200


# ── a shared project another person works in ────────────────────────────────

# Projects shared, and schedules (then conversations too) per person.
SHARED_PROJECTS = [
    "pins,schedules,memory",
    ",".join(sorted(ALL_PERSONAL_RESOURCES - {PersonalResource.PROJECTS})),
]


def _rows_left(*, project_id: str, schedule_ids=(), conversation_ids=()) -> bool:
    with Session(get_engine(get_app_settings().database.uri)) as raw:
        return (
            raw.get(Project, UUID(project_id)) is not None
            and all(raw.get(Schedule, UUID(sid)) is not None for sid in schedule_ids)
            and all(raw.get(Conversation, UUID(cid)) is not None for cid in conversation_ids)
        )


@pytest.mark.parametrize("resources", SHARED_PROJECTS)
def test_a_shared_project_holding_anothers_schedule_is_409(trusted, made_projects, resources):
    """A's delete can't reach B's schedule, so it is refused until B's goes;
    the creator's own delete included."""
    _personal_resources(trusted, resources)
    client = _client()
    pid = _made_project(client, A, made_projects)
    a_sid = _schedule(client, A, project_id=pid, title="a-schedule")
    b_sid = _schedule(client, B, project_id=pid, title="b-schedule")

    for headers in (A, B):
        res = client.delete(f"/api/v1/projects/{pid}", headers=headers)
        assert res.status_code == 409, res.text
        assert "Other people still have conversations or schedules" in res.json()["detail"]
    assert _rows_left(project_id=pid, schedule_ids=(a_sid, b_sid))

    assert client.delete(f"/api/v1/schedules/{b_sid}", headers=B).status_code == 204
    assert client.delete(f"/api/v1/projects/{pid}", headers=A).status_code == 204


def test_a_shared_project_holding_anothers_conversation_is_409(trusted, made_projects):
    """With conversations per person, A's scoped cascade can't see B's
    conversation, which Postgres's foreign key would refuse and SQLite would
    leave pointing at nothing."""
    _personal_resources(trusted, SHARED_PROJECTS[1])
    client = _client()
    pid = _made_project(client, A, made_projects)
    a_cid = _conversation(client, A, project_id=pid, topic="a-in-shared")
    b_cid = _conversation(client, B, project_id=pid, topic="b-in-shared")

    res = client.delete(f"/api/v1/projects/{pid}", headers=A)

    assert res.status_code == 409, res.text
    assert _rows_left(project_id=pid, conversation_ids=(a_cid, b_cid))
    assert client.get(f"/api/v1/conversations/{b_cid}", headers=B).status_code == 200


def test_a_shared_projects_conversation_workspace_stays_its_owners(trusted, made_projects):
    """With projects shared and conversations per person, `conversations/<id>/`
    inside the shared project is its owner's workspace: B reads, lists and
    deletes none of A's there, while a plain project file stays shared and the
    operator view still reads everything."""
    _personal_resources(trusted, SHARED_PROJECTS[1])
    client = _client()
    name = f"p-{uuid4().hex[:6]}"
    made = client.post("/api/v1/projects/", json={"name": name}, headers=A)
    assert made.status_code == 201, made.text
    made_projects.append(made.json()["id"])
    a_cid = _conversation(client, A, project_id=made.json()["id"], topic="a-workspace")
    secret = f"/api/v1/projects/{name}/files/conversations/{a_cid}/secret.txt"
    shared = f"/api/v1/projects/{name}/files/notes.txt"
    assert client.put(secret, json={"content": "a-only"}, headers=A).status_code == 200
    assert client.put(shared, json={"content": "everyone"}, headers=A).status_code == 200

    assert client.get(secret, headers=B).status_code == 404
    assert client.delete(secret, headers=B).status_code == 404
    assert client.put(secret, json={"content": "b-was-here"}, headers=B).status_code == 404
    listing = client.get(f"/api/v1/projects/{name}/files", headers=B)
    assert a_cid not in listing.text
    assert client.get(shared, headers=B).status_code == 200

    assert client.get(secret, headers=A).json()["content"] == "a-only"
    assert client.get(secret, headers=BEARER).status_code == 200


@pytest.mark.parametrize("resources", SHARED_PROJECTS)
def test_a_shared_project_holding_only_the_deleters_rows_is_deleted(trusted, made_projects, resources):
    _personal_resources(trusted, resources)
    client = _client()
    pid = _made_project(client, A, made_projects)
    _schedule(client, A, project_id=pid, title="a-schedule")
    _conversation(client, A, project_id=pid, topic="a-only")

    assert client.delete(f"/api/v1/projects/{pid}", headers=A).status_code == 204
    with Session(get_engine(get_app_settings().database.uri)) as raw:
        assert raw.get(Project, UUID(pid)) is None
