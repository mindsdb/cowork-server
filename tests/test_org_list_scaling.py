"""Org-mode list routes do the same work whatever the org's size.

Each route is called through HTTP in MindsHub mode, the way a browser calls
it, and the SQL statements it runs are counted at the engine. A listing that
reads one row per project or per skill fails here at the second size.
"""
from __future__ import annotations

import os
from collections.abc import Iterator
from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import uuid4

import pytest
from sqlalchemy import event
from sqlmodel import Session

from cowork.common.settings.app_settings import get_app_settings
from cowork.db.session import get_engine
from cowork.models.project import Project
from cowork.models.shared_resource import SharedResourceAttribution, SharedResourceMutation
from cowork.services.shared_resources import PROJECT, project_resource_key

ALICE = "11111111-1111-4111-8111-111111111111"
BOB = "22222222-2222-4222-8222-222222222222"
ADMIN = "33333333-3333-4333-8333-333333333333"


@pytest.fixture()
def client() -> Iterator:
    saved = {k: os.environ.get(k) for k in ("COWORK_TENANCY_MODE", "COWORK_IDENTITY_ENFORCE")}
    os.environ["COWORK_TENANCY_MODE"] = "org"
    os.environ["COWORK_IDENTITY_ENFORCE"] = "enforce"
    get_app_settings.cache_clear()
    try:
        from fastapi.testclient import TestClient
        from cowork.server import create_app

        # No `with`: the lifespan's boot migrations would collide with the
        # schema conftest already built.
        test_client = TestClient(create_app())
        try:
            yield test_client
        finally:
            test_client.close()
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        get_app_settings.cache_clear()


@pytest.fixture()
def org() -> str:
    """A fresh organization per test, so sizes never mix across tests."""
    return str(uuid4())


def _headers(org: str, user: str) -> dict[str, str]:
    return {"X-User-Id": user, "X-Organization-Id": org}


def _engine():
    return get_engine(get_app_settings().database.uri)


def _counted_get(client, path: str, headers: dict[str, str]) -> tuple[list, int]:
    statements: list[str] = []

    def record(_conn, _cursor, statement, _parameters, _context, _executemany):
        statements.append(statement)

    engine = _engine()
    event.listen(engine, "before_cursor_execute", record)
    try:
        response = client.get(path, headers=headers)
    finally:
        event.remove(engine, "before_cursor_execute", record)
    assert response.status_code == 200, response.text
    return response.json(), len(statements)


def _add_projects(org: str, root: Path, start: int, count: int) -> None:
    """Projects in every attribution state a steady-state listing shows."""
    with Session(_engine()) as raw:
        for i in range(start, start + count):
            state = i % 4
            project = Project(
                id=uuid4(),
                name=f"bulk-{i:05d}",
                path=str(root / f"bulk-{i:05d}"),
                org_id=org,
                created_by=None if state == 0 else (BOB if state == 2 else ALICE),
            )
            raw.add(project)
            if state == 0:
                continue  # a project from before attribution: no row
            raw.add(
                SharedResourceAttribution(
                    org_id=org,
                    resource_kind=PROJECT,
                    resource_key=project_resource_key(project.id),
                    created_by_id=project.created_by,
                    updated_by_id=ADMIN if state == 2 else project.created_by,
                    # a live first-writer claim: pending, nothing to recover
                    pending_claim_token=str(uuid4()) if state == 3 else None,
                    pending_claim_expires_at=(
                        datetime.now(timezone.utc) + timedelta(minutes=5)
                        if state == 3
                        else None
                    ),
                )
            )
        raw.commit()


def _projects_root(client, org: str) -> Path:
    listed, _ = _counted_get(client, "/api/v1/projects/", _headers(org, ALICE))
    return Path(listed[0]["path"]).parent


def test_the_project_list_runs_the_same_statements_at_10_and_1000_projects(client, org):
    root = _projects_root(client, org)  # warm-up: adopts the org's general project
    _add_projects(org, root, 0, 9)
    small, small_statements = _counted_get(client, "/api/v1/projects/", _headers(org, BOB))
    _add_projects(org, root, 9, 990)
    large, large_statements = _counted_get(client, "/api/v1/projects/", _headers(org, BOB))

    assert (len(small), len(large)) == (10, 1000)
    assert large_statements == small_statements, (
        f"{small_statements} statements at 10 projects, {large_statements} at 1,000"
    )


def test_the_project_list_still_recovers_an_expired_claim(client, org):
    root = _projects_root(client, org)
    project_id = uuid4()
    key = project_resource_key(project_id)
    with Session(_engine()) as raw:
        raw.add(
            Project(
                id=project_id,
                name="crashed-create",
                path=str(root / "crashed-create"),
                org_id=org,
                created_by=ALICE,
            )
        )
        raw.add(
            SharedResourceAttribution(
                org_id=org,
                resource_kind=PROJECT,
                resource_key=key,
                created_by_id=ALICE,
                pending_claim_token=str(uuid4()),
                pending_claim_expires_at=datetime.now(timezone.utc) - timedelta(seconds=1),
            )
        )
        raw.commit()

    listed, _ = _counted_get(client, "/api/v1/projects/", _headers(org, ALICE))

    shown = next(project for project in listed if project["id"] == str(project_id))
    assert shown["attribution"]["createdBy"]["userId"] == ALICE
    assert shown["capabilities"]["canDelete"] is True
    with Session(_engine()) as raw:
        row = raw.query(SharedResourceAttribution).filter_by(org_id=org, resource_key=key).one()
        assert row.pending_claim_token is None
        assert row.updated_by_id == ALICE
        events = raw.query(SharedResourceMutation).filter_by(org_id=org, resource_key=key).all()
        assert [(e.action, e.actor_id) for e in events] == [("recover", ALICE)]


CAROL = "44444444-4444-4444-8444-444444444444"


def _legacy_workspace(org: str, project_path: Path, owner: str, slug: str | None = None) -> Path:
    """A conversation of ``owner`` with a workspace from before artifacts moved
    to the project root, holding the artifact ``slug`` when one is given."""
    from cowork.models.conversation import Conversation

    with Session(_engine()) as raw:
        project = raw.query(Project).filter_by(org_id=org, path=str(project_path)).one()
        conversation = Conversation(topic="before", project_id=project.id, org_id=org, created_by=owner)
        raw.add(conversation)
        raw.commit()
        conversation_id = conversation.id
    base = project_path / "conversations" / str(conversation_id) / ".anton" / "artifacts"
    base.mkdir(parents=True)
    if slug is not None:
        folder = base / slug
        folder.mkdir()
        (folder / "index.html").write_text("<html></html>")
        (folder / "metadata.json").write_text(
            '{"slug": "%s", "name": "%s", "type": "html-app"}' % (slug, slug)
        )
    return base.parent.parent


def _projects_with_alices_workspaces(client, org: str, start: int, count: int) -> list[Path]:
    root = _projects_root(client, org)
    _add_projects(org, root, start, count)
    paths = [root / f"bulk-{i:05d}" for i in range(start, start + count)]
    for path in paths:
        _legacy_workspace(org, path, ALICE)
    return paths


def test_the_artifact_list_runs_the_same_statements_at_10_and_200_projects(client, org):
    _projects_with_alices_workspaces(client, org, 0, 10)
    _counted_get(client, "/api/v1/artifacts/", _headers(org, CAROL))  # warm-up
    _, small = _counted_get(client, "/api/v1/artifacts/", _headers(org, CAROL))
    _projects_with_alices_workspaces(client, org, 10, 190)
    _, large = _counted_get(client, "/api/v1/artifacts/", _headers(org, CAROL))

    assert large == small, f"{small} statements at 10 projects, {large} at 200"


def test_the_artifact_list_never_stats_a_co_members_workspace(client, org, monkeypatch):
    from cowork.services import artifact_roots

    workspaces = _projects_with_alices_workspaces(client, org, 0, 5)
    touched: list[Path] = []
    real_is_dir = artifact_roots._is_real_directory
    real_safe = artifact_roots._storage_components_are_safe

    def is_dir(path):
        touched.append(Path(path))
        return real_is_dir(path)

    def safe(workspace, *, may_be_absent):
        touched.append(Path(workspace))
        return real_safe(workspace, may_be_absent=may_be_absent)

    monkeypatch.setattr(artifact_roots, "_is_real_directory", is_dir)
    monkeypatch.setattr(artifact_roots, "_storage_components_are_safe", safe)

    listed, _ = _counted_get(client, "/api/v1/artifacts/", _headers(org, CAROL))

    assert listed == []
    alices = [
        path for path in touched
        if any(path.is_relative_to(project / "conversations") and path != project / "conversations"
               for project in workspaces)
    ]
    assert alices == []


def test_the_callers_own_legacy_cards_stay_theirs_beside_a_co_members(client, org):
    root = _projects_root(client, org)
    _add_projects(org, root, 0, 2)
    _legacy_workspace(org, root / "bulk-00001", ALICE, slug="alices-chart")
    _legacy_workspace(org, root / "bulk-00001", BOB, slug="bobs-chart")

    listed, _ = _counted_get(client, "/api/v1/artifacts/", _headers(org, ALICE))

    cards = {card["slug"]: card for card in listed}
    assert set(cards) == {"alices-chart"}
    assert cards["alices-chart"]["capabilities"]["role"] == "owner"


def _conversations_with_messages(org: str, root: Path, start: int, count: int) -> None:
    """``count`` conversations of Alice's, each in a project of its own, with
    two messages each, so every row has a last activity and a project."""
    from cowork.models.conversation import Conversation
    from cowork.models.message import Message

    _add_projects(org, root, start, count)
    with Session(_engine()) as raw:
        projects = raw.query(Project).filter(
            Project.org_id == org, Project.name.in_([f"bulk-{i:05d}" for i in range(start, start + count)])
        ).all()
        for project in projects:
            conversation = Conversation(topic=project.name, project_id=project.id, org_id=org, created_by=ALICE)
            raw.add(conversation)
            raw.flush()
            for n, role in enumerate(("user", "assistant")):
                raw.add(Message(
                    conversation_id=conversation.id, org_id=org, role=role, content=f"{role} {n}",
                    created_at=datetime.now(timezone.utc) + timedelta(seconds=n),
                ))
        raw.commit()


def test_the_conversation_list_runs_the_same_statements_at_3_and_30_conversations(client, org):
    root = _projects_root(client, org)
    path = "/api/v1/conversations/?project=all&limit=200"
    _conversations_with_messages(org, root, 0, 3)
    small, small_statements = _counted_get(client, path, _headers(org, ALICE))
    _conversations_with_messages(org, root, 3, 27)
    large, large_statements = _counted_get(client, path, _headers(org, ALICE))

    assert (len(small["conversations"]), len(large["conversations"])) == (3, 30)
    assert all(row["project"] for row in large["conversations"])
    assert large_statements == small_statements, (
        f"{small_statements} statements at 3 conversations, {large_statements} at 30"
    )
