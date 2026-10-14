"""COWORK_TRUST_IDENTITY_HEADERS through create_app: the startup refusals, the
middleware's place inside the bearer check, and two people kept apart over HTTP.

The two-person cases here cover what this layer decides on its own: the scope
each request gets, the rows it stamps, and the 404 for a turn posted into
another person's conversation. Pins and schedules, which no per-service gate
covers beyond the scoped session's own filter, get their two-person cases
here too: B lists, reads, changes and deletes none of A's, the operator view
lists both, and a deployment that leaves them off the list shares them. The
other per-service gates have their own tests.
"""
from __future__ import annotations

import logging
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
from cowork.db.personal_resources import ALL_PERSONAL_RESOURCES, PersonalResourceDependencyError
from cowork.db.scoped import LOCAL_SCOPE, TenantScope, get_tenant_scope
from cowork.db.session import get_engine
from cowork.models.conversation import Conversation
from cowork.models.pin import Pin
from cowork.models.project import Project
from cowork.models.schedule import Schedule
from cowork.principal import personal_partition_id

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

def test_identity_headers_without_the_bearer_are_401(trusted):
    res = _client(scope_route=True).get(
        "/api/v1/_scope", headers={"X-User-Id": SUB_A, "X-Organization-Id": DEPLOYMENT}
    )
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

# Every resource personal (the default), and only pins and schedules personal:
# the second case fails if either table is ever mapped to conversations or
# projects instead of its own resource.
PINS_AND_SCHEDULES_PERSONAL = [None, "pins,schedules"]


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
