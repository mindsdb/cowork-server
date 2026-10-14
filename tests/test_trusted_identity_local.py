"""COWORK_TRUST_IDENTITY_HEADERS through create_app: the startup refusals, the
middleware's place inside the bearer check, and two people kept apart over HTTP.

The two-person cases here cover what this layer decides on its own: the scope
each request gets, the rows it stamps, and the 404 for a turn posted into
another person's conversation. The per-service gates have their own tests.
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
from cowork.models.project import Project
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
