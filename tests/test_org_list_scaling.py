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
from fastapi.testclient import TestClient
from sqlalchemy import event
from sqlmodel import Session, select

from cowork.common.settings.app_settings import get_app_settings
from cowork.db.session import get_engine
from cowork.models.conversation import Conversation
from cowork.models.message import Message
from cowork.models.project import Project
from cowork.models.shared_resource import SharedResourceAttribution, SharedResourceMutation
from cowork.server import create_app
from cowork.services.artifact_ownership import artifact_resource_key
from cowork.services.shared_resources import ARTIFACT, PROJECT, project_resource_key

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
    # The attribution comes from the one bulk read: a finalized row shows its
    # creator and last editor, a live claim shows nobody and can't be deleted.
    for listed, (finalized, pending) in (
        (small, ("bulk-00002", "bulk-00003")),
        (large, ("bulk-00998", "bulk-00995")),
    ):
        shown = {project["name"]: project for project in listed}
        assert shown[finalized]["attribution"]["createdBy"]["userId"] == BOB
        assert shown[finalized]["attribution"]["lastModifiedBy"]["userId"] == ADMIN
        assert shown[finalized]["capabilities"]["canDelete"] is True
        assert shown[pending]["attribution"]["createdBy"] is None
        assert shown[pending]["capabilities"]["canDelete"] is False


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
        row = raw.exec(
            select(SharedResourceAttribution).where(
                SharedResourceAttribution.org_id == org,
                SharedResourceAttribution.resource_key == key,
            )
        ).one()
        assert row.pending_claim_token is None
        assert row.updated_by_id == ALICE
        events = raw.exec(
            select(SharedResourceMutation).where(
                SharedResourceMutation.org_id == org,
                SharedResourceMutation.resource_key == key,
            )
        ).all()
        assert [(e.action, e.actor_id) for e in events] == [("recover", ALICE)]


CAROL = "44444444-4444-4444-8444-444444444444"


def _write_artifact(base: Path, slug: str) -> None:
    folder = base / slug
    folder.mkdir(parents=True)
    (folder / "index.html").write_text("<html></html>")
    (folder / "metadata.json").write_text(
        '{"slug": "%s", "name": "%s", "type": "html-app"}' % (slug, slug)
    )


def _project_id(org: str, project_path: Path):
    with Session(_engine()) as raw:
        return raw.exec(
            select(Project.id).where(Project.org_id == org, Project.path == str(project_path))
        ).one()


def _legacy_workspace(
    org: str,
    project_path: Path,
    owner: str,
    slug: str | None = None,
    *,
    artifacts_folder: bool = True,
) -> Path:
    """A conversation of ``owner`` with a workspace from before artifacts moved
    to the project root, holding the artifact ``slug`` when one is given. With
    ``artifacts_folder`` false the workspace has no ``.anton/artifacts``, as
    every turn's working folder has until it writes an artifact there."""
    with Session(_engine()) as raw:
        conversation = Conversation(
            topic="before", project_id=_project_id(org, project_path), org_id=org, created_by=owner
        )
        raw.add(conversation)
        raw.commit()
        conversation_id = conversation.id
    workspace = project_path / "conversations" / str(conversation_id)
    if not artifacts_folder:
        workspace.mkdir(parents=True)
        (workspace / "notes.txt").write_text("a turn's working file")
        return workspace
    base = workspace / ".anton" / "artifacts"
    base.mkdir(parents=True)
    if slug is not None:
        _write_artifact(base, slug)
    return workspace


def _project_root_artifact(org: str, project_path: Path, owner: str, slug: str) -> None:
    """An artifact in the project root, with the attribution row its create
    records."""
    _write_artifact(project_path / ".anton" / "artifacts", slug)
    with Session(_engine()) as raw:
        raw.add(
            SharedResourceAttribution(
                org_id=org,
                resource_kind=ARTIFACT,
                resource_key=artifact_resource_key(_project_id(org, project_path), slug),
                created_by_id=owner,
                updated_by_id=owner,
            )
        )
        raw.commit()


def _bulk_paths(client, org: str, start: int, count: int) -> list[Path]:
    root = _projects_root(client, org)
    _add_projects(org, root, start, count)
    return [root / f"bulk-{i:05d}" for i in range(start, start + count)]


def _projects_with_alices_workspaces(client, org: str, start: int, count: int) -> list[Path]:
    """``count`` projects with one of Alice's legacy workspaces in each;
    returns the workspaces."""
    return [_legacy_workspace(org, path, ALICE) for path in _bulk_paths(client, org, start, count)]


@pytest.mark.parametrize("viewer", [CAROL, ALICE], ids=["owns-nothing", "owns-a-workspace-in-each"])
def test_the_artifact_list_runs_the_same_statements_at_10_and_200_projects(client, org, viewer):
    _projects_with_alices_workspaces(client, org, 0, 10)
    _counted_get(client, "/api/v1/artifacts/", _headers(org, viewer))  # warm-up
    _, small = _counted_get(client, "/api/v1/artifacts/", _headers(org, viewer))
    _projects_with_alices_workspaces(client, org, 10, 190)
    _, large = _counted_get(client, "/api/v1/artifacts/", _headers(org, viewer))

    assert large == small, f"{small} statements at 10 projects, {large} at 200"


def test_project_root_owners_come_from_one_query_at_10_and_60_projects(client, org):
    def add(start: int, count: int) -> None:
        for i, path in enumerate(_bulk_paths(client, org, start, count), start=start):
            _project_root_artifact(org, path, ALICE, f"chart-{i:05d}")

    add(0, 10)
    small, small_statements = _counted_get(client, "/api/v1/artifacts/", _headers(org, ALICE))
    add(10, 50)
    large, large_statements = _counted_get(client, "/api/v1/artifacts/", _headers(org, ALICE))

    assert (len(small), len(large)) == (10, 60)
    assert {card["capabilities"]["role"] for card in large} == {"owner"}
    assert large_statements == small_statements, (
        f"{small_statements} statements at 10 project-root cards, {large_statements} at 60"
    )


def test_the_callers_own_legacy_cards_cost_the_same_statements_at_5_and_50_projects(client, org):
    def add(start: int, count: int) -> None:
        for i, path in enumerate(_bulk_paths(client, org, start, count), start=start):
            _legacy_workspace(org, path, ALICE, slug=f"chart-{i:05d}")

    add(0, 5)
    small, small_statements = _counted_get(client, "/api/v1/artifacts/", _headers(org, ALICE))
    add(5, 45)
    large, large_statements = _counted_get(client, "/api/v1/artifacts/", _headers(org, ALICE))

    assert (len(small), len(large)) == (5, 50)
    assert {card["capabilities"]["role"] for card in large} == {"owner"}
    assert large_statements == small_statements, (
        f"{small_statements} statements at 5 legacy cards, {large_statements} at 50"
    )


def test_the_artifact_list_opens_no_empty_legacy_workspace(client, org, monkeypatch):
    from cowork.api.v1.endpoints import artifacts as artifacts_endpoint

    paths = _bulk_paths(client, org, 0, 3)
    empty = [_legacy_workspace(org, path, ALICE, artifacts_folder=False) for path in paths]
    with_card = _legacy_workspace(org, paths[0], ALICE, slug="alices-chart")
    opened: list[Path] = []
    real_list = artifacts_endpoint._list_artifacts

    def list_artifacts(sources):
        opened.extend(Path(source.base) for source in sources)
        return real_list(sources)

    monkeypatch.setattr(artifacts_endpoint, "_list_artifacts", list_artifacts)

    listed, _ = _counted_get(client, "/api/v1/artifacts/", _headers(org, ALICE))

    assert [card["slug"] for card in listed] == ["alices-chart"]
    assert with_card / ".anton" / "artifacts" in opened
    assert not [base for base in opened if any(base.is_relative_to(w) for w in empty)]


@pytest.mark.parametrize("viewer_owns_one", [False, True], ids=["owns-nothing", "owns-one"])
def test_the_artifact_list_never_stats_a_co_members_workspace(
    client, org, monkeypatch, viewer_owns_one
):
    from cowork.services import artifact_roots

    workspaces = _projects_with_alices_workspaces(client, org, 0, 5)
    if viewer_owns_one:
        # With a workspace of Carol's own the name filter runs, not the
        # early return for a caller who owns nothing.
        _legacy_workspace(org, workspaces[0].parent.parent, CAROL)
    touched: list[Path] = []
    real_is_dir = artifact_roots._is_real_directory
    real_safe = artifact_roots._storage_components_are_safe

    def is_dir(path, *, strict=False):
        touched.append(Path(path))
        return real_is_dir(path, strict=strict)

    def safe(workspace, *, may_be_absent, strict=False):
        touched.append(Path(workspace))
        return real_safe(workspace, may_be_absent=may_be_absent, strict=strict)

    monkeypatch.setattr(artifact_roots, "_is_real_directory", is_dir)
    monkeypatch.setattr(artifact_roots, "_storage_components_are_safe", safe)

    listed, _ = _counted_get(client, "/api/v1/artifacts/", _headers(org, CAROL))

    assert listed == []
    alices = [
        path for path in touched
        if any(path.is_relative_to(workspace) for workspace in workspaces)
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


def _conversations_with_messages(
    org: str, root: Path, start: int, count: int, *, tool_rows: int = 0
) -> dict[str, datetime]:
    """``count`` conversations of Alice's, each in a project of its own, so
    every row has a last activity and a project: a user message, then a last
    turn of ``tool_rows`` tool rows and the answer, all at one ``created_at``,
    as save_assistant_turn writes them in one commit. Returns each
    conversation's last activity by id."""
    _add_projects(org, root, start, count)
    last: dict[str, datetime] = {}
    with Session(_engine()) as raw:
        projects = raw.exec(
            select(Project).where(
                Project.org_id == org,
                Project.name.in_([f"bulk-{i:05d}" for i in range(start, start + count)]),
            )
        ).all()
        asked = datetime.now(timezone.utc).replace(microsecond=0)
        answered = asked + timedelta(seconds=1)
        for project in projects:
            conversation = Conversation(topic=project.name, project_id=project.id, org_id=org, created_by=ALICE)
            raw.add(conversation)
            raw.flush()
            raw.add(Message(conversation_id=conversation.id, role="user", content="ask", created_at=asked))
            for n in range(tool_rows):
                raw.add(Message(
                    conversation_id=conversation.id, role="assistant", content=f"tool row {n}",
                    created_at=answered,
                ))
            raw.add(Message(
                conversation_id=conversation.id, role="assistant", content="answer", created_at=answered,
            ))
            last[str(conversation.id)] = answered
        raw.commit()
    return last


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


def test_the_conversation_list_reads_one_row_per_conversation_whatever_its_last_turn(client, org):
    """A last turn's tool rows share its answer's ``created_at``; the list's
    last-activity read must not build a message for each of them."""
    root = _projects_root(client, org)
    last = _conversations_with_messages(org, root, 0, 20, tool_rows=40)
    loaded: list[Message] = []

    def on_load(target, _context):
        loaded.append(target)

    event.listen(Message, "load", on_load)
    try:
        listed, _ = _counted_get(
            client, "/api/v1/conversations/?project=all&limit=200", _headers(org, ALICE)
        )
    finally:
        event.remove(Message, "load", on_load)

    rows = listed["conversations"]
    assert len(rows) == 20
    for row in rows:
        shown = datetime.fromisoformat(str(row["updatedAt"]).replace("Z", "+00:00"))
        expected = last[row["id"]]
        if shown.tzinfo is None:
            expected = expected.replace(tzinfo=None)
        assert shown == expected
    assert loaded == [], f"{len(loaded)} messages built to read 20 timestamps"
