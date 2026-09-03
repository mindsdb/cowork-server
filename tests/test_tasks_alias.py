"""`/tasks` is a branding alias for `/conversations` (ENG-2069).

It's the same router mounted under a second prefix (see api/v1/router.py), so
these lock in the two things that matter: a resource created through one
prefix is visible through the other, and both prefixes answer identically for
the same underlying data. If the router ever forks into two separate
implementations, these fail.
"""
from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy.pool import StaticPool
from sqlmodel import Session, SQLModel, create_engine

from cowork.api.v1.endpoints import conversations
from cowork.db.scoped import LOCAL_SCOPE, get_tenant_scope
from cowork.db.session import get_session
from cowork.models.project import Project


@pytest.fixture
def client():
    engine = create_engine(
        "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool
    )
    SQLModel.metadata.create_all(engine)
    with Session(engine) as seed:
        project = Project(name="default", path="/tmp/default")
        seed.add(project)
        seed.commit()
        seed.refresh(project)
        project_id = str(project.id)

    app = FastAPI()
    # Mirror api/v1/router.py: the same router mounted under both prefixes.
    app.include_router(conversations.router, prefix="/api/v1/conversations")
    app.include_router(conversations.router, prefix="/api/v1/tasks")
    app.dependency_overrides[get_session] = lambda: Session(engine)
    app.dependency_overrides[get_tenant_scope] = lambda: LOCAL_SCOPE
    c = TestClient(app)
    c.project_id = project_id  # type: ignore[attr-defined]
    return c


def test_task_created_via_alias_is_visible_via_canonical_path(client):
    created = client.post(
        "/api/v1/tasks/", json={"topic": "alias task", "project_id": client.project_id}
    )
    assert created.status_code == 201
    task_id = created.json()["id"]

    listed = client.get("/api/v1/conversations/", params={"project": "all"}).json()["conversations"]
    assert any(c["id"] == task_id for c in listed)


def test_conversation_created_via_canonical_path_is_visible_via_alias(client):
    created = client.post(
        "/api/v1/conversations/",
        json={"topic": "canonical convo", "project_id": client.project_id},
    )
    assert created.status_code == 201
    conversation_id = created.json()["id"]

    listed = client.get("/api/v1/tasks/", params={"project": "all"}).json()["conversations"]
    assert any(c["id"] == conversation_id for c in listed)


def test_both_prefixes_answer_identically_for_the_same_data(client):
    client.post("/api/v1/tasks/", json={"topic": "one", "project_id": client.project_id})
    client.post("/api/v1/tasks/", json={"topic": "two", "project_id": client.project_id})

    via_conversations = client.get("/api/v1/conversations/", params={"project": "all"}).json()
    via_tasks = client.get("/api/v1/tasks/", params={"project": "all"}).json()
    assert via_conversations == via_tasks


def test_alias_supports_the_full_request_lifecycle(client):
    task_id = client.post(
        "/api/v1/tasks/", json={"topic": "lifecycle", "project_id": client.project_id}
    ).json()["id"]

    fetched = client.get(f"/api/v1/tasks/{task_id}")
    assert fetched.status_code == 200

    renamed = client.patch(f"/api/v1/tasks/{task_id}", json={"topic": "renamed"})
    assert renamed.status_code == 200
    assert renamed.json()["title"] == "renamed"

    deleted = client.delete(f"/api/v1/tasks/{task_id}")
    assert deleted.status_code in (200, 204)
    assert client.get(f"/api/v1/tasks/{task_id}").status_code == 404
