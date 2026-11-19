"""Org-mode list routes do the same work whatever the org's size.

Each route is called through HTTP in MindsHub mode, the way a browser calls
it, and the SQL statements it runs are counted at the engine. A listing that
reads one row per project or per skill fails here at the second size.
"""
from __future__ import annotations

import asyncio
import errno
import json
import os
from collections.abc import Iterator
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import NamedTuple
from uuid import UUID, uuid4

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import event
from sqlmodel import Session, select

from cowork.common.settings.app_settings import get_app_settings
from cowork.db.session import get_engine
from cowork.models.conversation import Conversation
from cowork.models.message import Message
from cowork.models.project import Project
from cowork.models.setting import Setting
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


# The held-root rows (cowork.services.held_artifact_roots): once their
# backfill has run, the list opens only the roots they name.


def _drop_sentinels() -> None:
    from cowork.services.artifact_sort_times import SENTINEL_KEY as SORT_TIMES_KEY
    from cowork.services.held_artifact_roots import SENTINEL_KEY

    with Session(_engine()) as raw:
        keys = (SENTINEL_KEY, SORT_TIMES_KEY)
        for row in raw.exec(select(Setting).where(Setting.key.in_(keys), Setting.scope.is_(None))).all():
            raw.delete(row)
        raw.commit()


@pytest.fixture()
def held_roots(org):
    """Fills ``org``'s held-root rows, and then its sort times unless
    ``times`` is false, with the startup backfills, which write the
    sentinels the list trusts them by. The sentinels are global and the
    database is shared across modules, so they go again after the test."""
    from cowork.services.artifact_sort_times import run_sort_time_backfill
    from cowork.services.held_artifact_roots import run_held_root_backfill

    def fill(*, times: bool = True) -> None:
        _drop_sentinels()
        with Session(_engine()) as raw:
            project_ids = raw.exec(select(Project.id).where(Project.org_id == org)).all()
        summary = run_held_root_backfill(project_ids=project_ids)
        assert summary is not None and summary.unread == 0
        if times:
            recorded = run_sort_time_backfill(project_ids=project_ids)
            assert recorded is not None and recorded.unread == 0

    yield fill
    _drop_sentinels()


_FILE_STORE_CALLS = ("stat", "lstat", "fstat", "open", "scandir", "listdir", "access", "readlink")


def _file_store_get(client, monkeypatch, path: str, headers: dict[str, str]) -> tuple[list, int]:
    """The answer to ``path`` and the os calls it made on the file store's
    threads (run_efs), where the list finds and walks its roots."""
    from cowork.db.units import on_efs_thread

    calls: list[str] = []

    def counted(name, real):
        def call(*args, **kwargs):
            if on_efs_thread():
                calls.append(name)
            return real(*args, **kwargs)

        return call

    with monkeypatch.context() as patched:
        for name in _FILE_STORE_CALLS:
            patched.setattr(os, name, counted(name, getattr(os, name)))
        response = client.get(path, headers=headers)
    assert response.status_code == 200, response.text
    return response.json(), len(calls)


def _alices_empty_workspaces(org: str, project_path: Path, count: int) -> None:
    """``count`` of Alice's conversations in one project, each with a working
    folder holding a file and no ``.anton/artifacts``, as most turns leave."""
    with Session(_engine()) as raw:
        project_id = _project_id(org, project_path)
        conversations = [
            Conversation(topic="turn", project_id=project_id, org_id=org, created_by=ALICE) for _ in range(count)
        ]
        raw.add_all(conversations)
        raw.commit()
        ids = [conversation.id for conversation in conversations]
    for conversation_id in ids:
        workspace = project_path / "conversations" / str(conversation_id)
        workspace.mkdir(parents=True)
        (workspace / "notes.txt").write_text("a turn's working file")


def test_the_list_makes_no_more_file_store_calls_at_1000_projects_than_at_10(client, org, held_roots, monkeypatch):
    """1,000 projects, 10 holding an artifact, against 10 that each hold one:
    a project that holds no artifact costs the list no file store call. A
    list that opened every project's root fails here."""
    for i, path in enumerate(_bulk_paths(client, org, 0, 10)):
        _project_root_artifact(org, path, ALICE, f"chart-{i:05d}")
    held_roots()
    small, small_calls = _file_store_get(client, monkeypatch, "/api/v1/artifacts/", _headers(org, BOB))
    _bulk_paths(client, org, 10, 990)
    held_roots()
    large, large_calls = _file_store_get(client, monkeypatch, "/api/v1/artifacts/", _headers(org, BOB))

    assert (len(small), len(large)) == (10, 10)
    assert large_calls <= small_calls, f"{small_calls} file store calls at 10 projects, {large_calls} at 1,000"


def test_a_projects_list_makes_no_more_file_store_calls_at_1000_workspaces_than_at_10(
    client, org, held_roots, monkeypatch
):
    """One project with 1,000 of the caller's conversation workspaces, 10
    holding an artifact, against 10 that each hold one: a workspace that
    holds no artifact costs the list no file store call. A list that stat'ed
    each of the caller's workspaces fails here."""
    [path] = _bulk_paths(client, org, 0, 1)
    for i in range(10):
        _legacy_workspace(org, path, ALICE, slug=f"chart-{i:05d}")
    listed = f"/api/v1/artifacts/?project_id={_project_id(org, path)}"
    held_roots()
    small, small_calls = _file_store_get(client, monkeypatch, listed, _headers(org, ALICE))
    _alices_empty_workspaces(org, path, 990)
    held_roots()
    large, large_calls = _file_store_get(client, monkeypatch, listed, _headers(org, ALICE))

    assert (len(small), len(large)) == (10, 10)
    assert large_calls <= small_calls, f"{small_calls} file store calls at 10 workspaces, {large_calls} at 1,000"


def _seed_every_kind_of_root(client, org: str) -> list[Path]:
    """Projects whose roots hold every kind of entry the list reads or
    refuses; returns their paths."""
    paths = _bulk_paths(client, org, 0, 8)
    shared = paths[0] / ".anton" / "artifacts"
    _project_root_artifact(org, paths[0], ALICE, "alices-chart")
    _project_root_artifact(org, paths[0], BOB, "bobs-chart")
    _write_artifact(shared, "nobodys-chart")  # no owner row: an unknown owner
    _write_artifact(shared / ".locks", "not-an-artifact")
    _write_artifact(paths[7], "outside")
    (shared / "a-link").symlink_to(paths[7] / "outside")
    _legacy_workspace(org, paths[1], ALICE, slug="alices-old-chart")
    _legacy_workspace(org, paths[1], BOB, slug="bobs-old-chart")
    _legacy_workspace(org, paths[1], ALICE)  # an empty .anton/artifacts
    _legacy_workspace(org, paths[1], ALICE, artifacts_folder=False)
    _write_artifact(paths[1] / "conversations" / "not-a-conversation" / ".anton" / "artifacts", "orphan")
    _project_root_artifact(org, paths[2], BOB, "bobs-other-chart")
    _legacy_workspace(org, paths[2], ALICE, slug="alices-other-old-chart")
    # A project whose shared root is a link: refused, as every link on the way is.
    (paths[3] / ".anton").mkdir(parents=True)
    (paths[3] / ".anton" / "artifacts").symlink_to(shared)
    return paths


@pytest.mark.parametrize("times", [False, True], ids=["held-roots", "and-sort-times"])
def test_the_list_shows_the_same_cards_through_the_held_rows(client, org, held_roots, times):
    """Every form of the list, for every viewer, answers the same before the
    rows exist and after the backfills have filled them: the held roots
    alone, and with the sort times."""
    paths = _seed_every_kind_of_root(client, org)
    lists = ["/api/v1/artifacts/"] + [f"/api/v1/artifacts/?project_id={_project_id(org, path)}" for path in paths]

    def answers() -> dict:
        return {
            (path, viewer): client.get(path, headers=_headers(org, viewer)).json()
            for path in lists
            for viewer in (ALICE, BOB, CAROL)
        }

    walked = answers()
    held_roots(times=times)
    through_rows = answers()

    assert through_rows == walked
    shown = {card["slug"] for card in walked[("/api/v1/artifacts/", ALICE)]}
    assert shown == {
        "alices-chart", "bobs-chart", "nobodys-chart", "alices-old-chart", "bobs-other-chart",
        "alices-other-old-chart",
    }


def test_a_held_row_never_shows_a_co_members_workspace_or_another_organizations_project(
    client, org, held_roots
):
    """The rows are hints: the list opens a row's root only for a project the
    caller's scope returns, and a legacy root only for a conversation the
    caller owns."""
    from cowork.models.held_artifact_root import HeldArtifactRoot

    [path] = _bulk_paths(client, org, 0, 1)
    bobs_workspace = _legacy_workspace(org, path, BOB, slug="bobs-old-chart")
    other_org = str(uuid4())
    [elsewhere] = _bulk_paths(client, other_org, 0, 1)
    _project_root_artifact(other_org, elsewhere, ALICE, "another-orgs-chart")
    held_roots()
    with Session(_engine()) as raw:
        # Rows naming the other organization's project, here and there.
        raw.add(HeldArtifactRoot(org_id=org, project_id=_project_id(other_org, elsewhere)))
        raw.add(HeldArtifactRoot(org_id=other_org, project_id=_project_id(other_org, elsewhere)))
        raw.commit()

    unfiltered = client.get("/api/v1/artifacts/", headers=_headers(org, CAROL))
    bobs_project = client.get(
        f"/api/v1/artifacts/?project_id={_project_id(org, path)}", headers=_headers(org, CAROL)
    )
    other_project = client.get(
        f"/api/v1/artifacts/?project_id={_project_id(other_org, elsewhere)}", headers=_headers(org, CAROL)
    )

    assert bobs_workspace.name in {str(c) for c in _held_conversations(org)}
    assert unfiltered.json() == []
    assert bobs_project.json() == []
    assert other_project.status_code == 404


def _artifacts_at(org: str, project_path: Path, owner: str, times: dict[str, float]) -> None:
    """One project-root artifact per name in ``times``, its metadata.json
    modified at that time."""
    for slug, at in times.items():
        _project_root_artifact(org, project_path, owner, slug)
        os.utime(project_path / ".anton" / "artifacts" / slug / "metadata.json", (at, at))


@pytest.mark.parametrize("unfiltered", [False, True], ids=["one-project", "every-project"])
def test_the_list_makes_no_more_file_store_calls_at_1000_artifacts_than_at_80(
    client, org, held_roots, monkeypatch, unfiltered
):
    """One project's root with 1,000 artifacts against one with 80, through
    the list of that project and the list of every project: with the sort
    times recorded, the list opens only the folders it shows. A list that
    read every folder's metadata.json fails here."""
    [path] = _bulk_paths(client, org, 0, 1)
    _artifacts_at(org, path, ALICE, {f"chart-{i:04d}": 1_000_000.0 + i for i in range(80)})
    held_roots()
    listed = "/api/v1/artifacts/" if unfiltered else f"/api/v1/artifacts/?project_id={_project_id(org, path)}"
    small, small_calls = _file_store_get(client, monkeypatch, listed, _headers(org, ALICE))
    _artifacts_at(org, path, ALICE, {f"chart-{i:04d}": 1_000_000.0 + i for i in range(80, 1000)})
    held_roots()
    large, large_calls = _file_store_get(client, monkeypatch, listed, _headers(org, ALICE))

    assert (len(small), len(large)) == (80, 80)
    assert [card["slug"] for card in large] == [f"chart-{i:04d}" for i in range(999, 919, -1)]
    assert large_calls <= small_calls, f"{small_calls} file store calls at 80 artifacts, {large_calls} at 1,000"


def _slugs_of(client, org: str, listed: str) -> list[str]:
    """The slugs of the cards ``listed`` shows Alice, in its order."""
    response = client.get(listed, headers=_headers(org, ALICE))
    assert response.status_code == 200, response.text
    return [card["slug"] for card in response.json()]


def _walked_slugs(client, org: str, listed: str, monkeypatch) -> list[str]:
    """``_slugs_of`` with every folder of the list's roots read, as before
    the sort times' backfill: the order a walk of the folders gives."""
    from cowork.services import artifact_sort_times

    with monkeypatch.context() as walking:
        walking.setattr(artifact_sort_times, "newest_folders", lambda *_args, **_kwargs: None)
        return _slugs_of(client, org, listed)


def _rewrite_metadata(folder: Path, *, at: float, description: str) -> None:
    """Rewrite ``folder``'s metadata.json with a new description, as anton
    rewrites it, modified at ``at``."""
    metadata = folder / "metadata.json"
    metadata.write_text(json.dumps({**json.loads(metadata.read_text()), "description": description}))
    os.utime(metadata, (at, at))


def _alices_conversation(org: str, project_path: Path):
    with Session(_engine()) as raw:
        conversation = Conversation(topic="turn", project_id=_project_id(org, project_path), org_id=org, created_by=ALICE)
        raw.add(conversation)
        raw.commit()
        return conversation.id


def _self_heal(org: str, project_path: Path):
    """The location check of the shared root of the project at
    ``project_path`` (artifact_autopublish.SelfHeal), in the organization's
    scope with no user, as the heal sweep and a list ask for it."""
    from cowork.db.scoped import TenantScope
    from cowork.services.artifact_autopublish import SelfHeal

    return SelfHeal(
        artifacts_base=project_path / ".anton" / "artifacts",
        scope=TenantScope(org_mode=True, org_id=org),
        project_id=str(_project_id(org, project_path)),
    )


def test_a_turns_edit_moves_its_card_as_a_walk_does(client, org, held_roots, monkeypatch):
    """The list gives the order a walk of every folder gives, also after a
    turn edits an artifact: a rewrite of its metadata.json alone, as anton's
    update_artifact_metadata makes, or of its content and its metadata.json.
    The turn end records the time it read (task_objects.record_new_artifacts)."""
    from cowork.services.artifact_sort_times import sort_times_below
    from cowork.services.task_objects import record_new_artifacts
    from test_artifact_ownership import scoped

    [path] = _bulk_paths(client, org, 0, 1)
    _artifacts_at(org, path, ALICE, {"first": 1_000_100.0, "second": 1_000_200.0, "third": 1_000_300.0})
    held_roots()
    listed = f"/api/v1/artifacts/?project_id={_project_id(org, path)}"
    base = path / ".anton" / "artifacts"

    def turn_end(slug: str) -> None:
        with scoped(org, ALICE) as session:
            record_new_artifacts(
                session, conversation_id=_alices_conversation(org, path), project_id=_project_id(org, path),
                slugs=[], creator=ALICE, sort_times=sort_times_below(base=base, slugs=[slug]).times,
            )

    assert _slugs_of(client, org, listed) == _walked_slugs(client, org, listed, monkeypatch) == [
        "third", "second", "first",
    ]

    _rewrite_metadata(base / "first", at=1_000_400.0, description="renamed by the agent")
    turn_end("first")
    assert _slugs_of(client, org, listed) == _walked_slugs(client, org, listed, monkeypatch) == [
        "first", "third", "second",
    ]

    (base / "second" / "index.html").write_text("<html>edited</html>")
    _rewrite_metadata(base / "second", at=1_000_500.0, description="edited by the agent")
    turn_end("second")
    assert _slugs_of(client, org, listed) == _walked_slugs(client, org, listed, monkeypatch) == [
        "second", "first", "third",
    ]


def test_a_metadata_rewrite_no_turn_names_moves_the_card_at_the_projects_next_self_heal(
    client, org, held_roots, monkeypatch
):
    """A rewrite of metadata.json that no turn names, as anton's own file
    bookkeeping makes, leaves the rows' order as it was until the project's
    next self-heal, which reads the time of every folder of its shared root
    (artifact_autopublish._record_listed_locations). The card then moves as
    a walk moves it."""
    from cowork.services.artifact_autopublish import _record_listed_locations

    [path] = _bulk_paths(client, org, 0, 1)
    _artifacts_at(org, path, ALICE, {"first": 1_000_100.0, "second": 1_000_200.0, "third": 1_000_300.0})
    held_roots()
    listed = f"/api/v1/artifacts/?project_id={_project_id(org, path)}"

    _rewrite_metadata(path / ".anton" / "artifacts" / "first", at=1_000_400.0, description="files reconciled")

    assert _slugs_of(client, org, listed) == ["third", "second", "first"]
    assert _walked_slugs(client, org, listed, monkeypatch) == ["first", "third", "second"]

    asyncio.run(_record_listed_locations(_self_heal(org, path)))

    assert _slugs_of(client, org, listed) == _walked_slugs(client, org, listed, monkeypatch) == [
        "first", "third", "second",
    ]


def test_the_servers_own_id_widening_leaves_the_order_as_it_was(client, org, held_roots, monkeypatch):
    """A rewrite of metadata.json the server makes on its own, an id
    widening (artifact_identity.ensure_full_id), puts the file's time back.
    A walk of every folder and the rows both keep the order, and so does the
    project's next self-heal, which reads every folder's time again."""
    from cowork.services.artifact_autopublish import _record_listed_locations
    from cowork.services.artifact_identity import ensure_full_id

    [path] = _bulk_paths(client, org, 0, 1)
    # metadata.json with no id, as a record from before full ids holds.
    _artifacts_at(org, path, ALICE, {"first": 1_000_100.0, "second": 1_000_200.0, "third": 1_000_300.0})
    held_roots()
    listed = f"/api/v1/artifacts/?project_id={_project_id(org, path)}"
    widened = path / ".anton" / "artifacts" / "first"
    assert "id" not in json.loads((widened / "metadata.json").read_text())

    full_id, _ = ensure_full_id(widened)

    assert json.loads((widened / "metadata.json").read_text())["id"] == full_id
    assert _slugs_of(client, org, listed) == _walked_slugs(client, org, listed, monkeypatch) == [
        "third", "second", "first",
    ]
    asyncio.run(_record_listed_locations(_self_heal(org, path)))
    assert _slugs_of(client, org, listed) == _walked_slugs(client, org, listed, monkeypatch) == [
        "third", "second", "first",
    ]


def _gone_rows(org: str, project_path: Path, *, count: int, conversation_id: UUID | None = None) -> None:
    """``count`` sort-time rows, newer than every folder the tests write, of
    folders deleted behind the server's back: no folder holds their names.
    In the project's shared root, or in ``conversation_id``'s legacy root."""
    from cowork.models.artifact_sort_time import ArtifactSortTime

    project_id = _project_id(org, project_path)
    with Session(_engine()) as raw:
        raw.add_all(
            ArtifactSortTime(
                org_id=org, project_id=project_id, conversation_id=conversation_id,
                slug=f"gone-{i:03d}", sort_time=2_000_000.0 + i,
            )
            for i in range(count)
        )
        raw.commit()


def _sort_time_slugs(org: str) -> set[str]:
    from cowork.models.artifact_sort_time import ArtifactSortTime

    with Session(_engine()) as raw:
        return set(raw.exec(select(ArtifactSortTime.slug).where(ArtifactSortTime.org_id == org)).all())


def _location_checks(monkeypatch) -> list:
    """The location checks the lists ask for
    (artifact_autopublish.request_location_check), collected here instead of
    reaching the poll."""
    from cowork.services import artifact_autopublish

    asked: list = []
    monkeypatch.setattr(artifact_autopublish, "request_location_check", asked.append)
    return asked


def test_a_list_whose_newest_rows_are_all_gone_reads_every_folder_and_asks_for_the_projects_check(
    client, org, held_roots, monkeypatch
):
    """Rows for folders deleted behind the server's back fill the query's
    limit: the list then reads every folder rather than come up short, and
    asks for the project's location check, an in-memory request. That check
    drops the rows, so the next list opens only the folders the rows name,
    and asks for nothing."""
    from cowork.services.artifact_autopublish import _record_listed_locations

    [path] = _bulk_paths(client, org, 0, 1)
    _artifacts_at(org, path, ALICE, {"kept-a": 1_000_100.0, "kept-b": 1_000_200.0})
    held_roots()
    _gone_rows(org, path, count=200)
    listed = f"/api/v1/artifacts/?project_id={_project_id(org, path)}"
    asked = _location_checks(monkeypatch)

    assert _slugs_of(client, org, listed) == ["kept-b", "kept-a"]
    [heal] = asked
    assert heal == _self_heal(org, path)

    asked.clear()
    asyncio.run(_record_listed_locations(heal))

    assert _sort_time_slugs(org) == {"kept-a", "kept-b"}
    assert _slugs_of(client, org, listed) == ["kept-b", "kept-a"]
    assert asked == []


def test_gone_rows_of_a_legacy_root_ask_for_no_location_check(client, org, held_roots, monkeypatch):
    """The self-heal lists a project's shared root alone, so rows of a
    conversation's legacy root that name gone folders ask for no location
    check, which would list the legacy root as the project's shared root.
    The list still reads every folder, as a walk does."""
    [path] = _bulk_paths(client, org, 0, 1)
    _artifacts_at(org, path, ALICE, {"kept": 1_000_100.0})
    workspace = _legacy_workspace(org, path, ALICE, slug="legacy-kept")
    legacy_metadata = workspace / ".anton" / "artifacts" / "legacy-kept" / "metadata.json"
    os.utime(legacy_metadata, (1_000_200.0, 1_000_200.0))
    held_roots()
    _gone_rows(org, path, count=200, conversation_id=UUID(workspace.name))
    listed = f"/api/v1/artifacts/?project_id={_project_id(org, path)}"
    asked = _location_checks(monkeypatch)

    assert _slugs_of(client, org, listed) == _walked_slugs(client, org, listed, monkeypatch) == [
        "legacy-kept", "kept",
    ]
    assert asked == []


# The projects whose rows the list cannot trust: those the held-root
# backfill could not read (RootsPlan.walk), and those the sort-time backfill
# could not read. The list's plan names both in NewestFolders.unread, and
# the list reads every folder of them.


def _org_project_ids(org: str) -> list:
    with Session(_engine()) as raw:
        return list(raw.exec(select(Project.id).where(Project.org_id == org)).all())


def _list_paths(org: str, *project_paths: Path) -> list[str]:
    """The unfiltered list, and the list of each project at ``project_paths``."""
    return ["/api/v1/artifacts/"] + [
        f"/api/v1/artifacts/?project_id={_project_id(org, path)}" for path in project_paths
    ]


def _failing_listing(monkeypatch, *, unreadable: Path) -> None:
    """The file server stops answering the backfills' listings of one root
    (held_artifact_roots.listable_folders). The list's own listings still
    read it."""
    from cowork.services import held_artifact_roots

    real = held_artifact_roots.dir_scandir

    def scandir(root):
        if Path(str(root.path)) == unreadable:
            raise OSError(errno.EIO, "the file server did not answer")
        return real(root)

    monkeypatch.setattr(held_artifact_roots, "dir_scandir", scandir)


class _Handed(NamedTuple):
    """What one list handed its file store half (_newest_cards): the slugs
    of the plan's newest rows, and the projects whose every folder it reads
    (NewestFolders.unread)."""

    rows: set[str]
    unread: frozenset


def _handed_to_the_file_store(monkeypatch) -> list[_Handed]:
    from cowork.api.v1.endpoints import artifacts as artifacts_endpoint

    handed: list[_Handed] = []
    real = artifacts_endpoint._newest_cards

    def newest_cards(sources, newest):
        handed.append(_Handed(rows={row.slug for row in newest.rows}, unread=newest.unread))
        return real(sources, newest)

    monkeypatch.setattr(artifacts_endpoint, "_newest_cards", newest_cards)
    return handed


def _roots_found_on_efs_threads(monkeypatch) -> list[bool]:
    """For each project whose roots a list finds on the file store
    (artifact_roots.project_roots), whether that ran on a file store thread
    (run_efs) rather than in a database unit."""
    from cowork.db.units import on_efs_thread
    from cowork.services import artifact_roots

    found: list[bool] = []
    real = artifact_roots.project_roots

    def project_roots(project, **kwargs):
        found.append(on_efs_thread())
        return real(project, **kwargs)

    monkeypatch.setattr(artifact_roots, "project_roots", project_roots)
    return found


def test_a_project_the_held_root_backfill_could_not_read_shows_its_folders_in_walk_order(
    client, org, held_roots, monkeypatch
):
    """The held-root backfill could not read one project, so its roots have
    no held rows and no sort times (RootsPlan.walk). Both list forms read
    every folder of it, and show them among the other project's row-backed
    folders in the order a walk of every folder gives. Its roots are found
    on the file store threads, never in the database unit that reads the
    newest rows."""
    from cowork.services.artifact_sort_times import run_sort_time_backfill
    from cowork.services.held_artifact_roots import run_held_root_backfill

    _drop_sentinels()
    timed, walked = _bulk_paths(client, org, 0, 2)
    _artifacts_at(org, timed, ALICE, {"timed-old": 1_000_100.0, "timed-new": 1_000_300.0})
    _artifacts_at(org, walked, ALICE, {"walked-old": 1_000_200.0, "walked-new": 1_000_400.0})
    # A workspace of Alice's, so finding the walked project's roots stats
    # its conversations folder (artifact_roots.project_roots).
    _legacy_workspace(org, walked, ALICE, artifacts_folder=False)
    lists = _list_paths(org, timed, walked)
    walk = {listed: _slugs_of(client, org, listed) for listed in lists}
    with monkeypatch.context() as failing:
        _failing_listing(failing, unreadable=walked / ".anton" / "artifacts")
        held = run_held_root_backfill(project_ids=_org_project_ids(org))
    assert held is not None and held.unread == 1
    assert run_sort_time_backfill(project_ids=_org_project_ids(org)) is not None
    handed = _handed_to_the_file_store(monkeypatch)
    found_on_efs = _roots_found_on_efs_threads(monkeypatch)

    shown = {listed: _slugs_of(client, org, listed) for listed in lists}

    assert shown == walk
    assert shown["/api/v1/artifacts/"] == ["walked-new", "timed-new", "walked-old", "timed-old"]
    walked_id = _project_id(org, walked)
    assert handed == [
        _Handed(rows={"timed-old", "timed-new"}, unread=frozenset({walked_id})),
        _Handed(rows={"timed-old", "timed-new"}, unread=frozenset()),
        _Handed(rows=set(), unread=frozenset({walked_id})),
    ]
    assert found_on_efs and all(found_on_efs)


def test_a_project_the_sort_time_backfill_could_not_read_shows_its_folders_in_walk_order(
    client, org, held_roots, monkeypatch
):
    """The sort-time backfill could not read one project's held root, so it
    has no sort times, and the sentinel lists it (NewestFolders.unread).
    Both list forms read every folder of it, and show them among the other
    project's row-backed folders in the order a walk of every folder gives."""
    from cowork.services.artifact_sort_times import run_sort_time_backfill

    _drop_sentinels()
    timed, unread = _bulk_paths(client, org, 0, 2)
    _artifacts_at(org, timed, ALICE, {"timed-old": 1_000_100.0, "timed-new": 1_000_300.0})
    _artifacts_at(org, unread, ALICE, {"unread-old": 1_000_200.0, "unread-new": 1_000_400.0})
    lists = _list_paths(org, timed, unread)
    walk = {listed: _slugs_of(client, org, listed) for listed in lists}
    held_roots(times=False)
    with monkeypatch.context() as failing:
        _failing_listing(failing, unreadable=unread / ".anton" / "artifacts")
        times = run_sort_time_backfill(project_ids=_org_project_ids(org))
    assert times is not None and times.unread == 1
    handed = _handed_to_the_file_store(monkeypatch)

    shown = {listed: _slugs_of(client, org, listed) for listed in lists}

    assert shown == walk
    assert shown["/api/v1/artifacts/"] == ["unread-new", "timed-new", "unread-old", "timed-old"]
    unread_id = _project_id(org, unread)
    assert handed == [
        _Handed(rows={"timed-old", "timed-new"}, unread=frozenset({unread_id})),
        _Handed(rows={"timed-old", "timed-new"}, unread=frozenset()),
        _Handed(rows=set(), unread=frozenset({unread_id})),
    ]


def test_rows_cut_at_the_limit_never_put_a_walked_projects_older_folders_first(
    client, org, held_roots, monkeypatch
):
    """The rows were cut at the query's limit, and every folder they name is
    gone, while a project the held-root backfill could not read holds
    LIST_CARDS older folders. A folder past the cut may be newer than those,
    so the list reads every folder, and shows the cards a walk shows: the
    timed project's folders first. Only the timed project, whose rows named
    gone folders, gets a location check."""
    from cowork.services.artifact_sort_times import run_sort_time_backfill
    from cowork.services.artifacts import LIST_CARDS
    from cowork.services.held_artifact_roots import run_held_root_backfill

    _drop_sentinels()
    timed, walked = _bulk_paths(client, org, 0, 2)
    _artifacts_at(org, timed, ALICE, {"timed-old": 1_000_100.0, "timed-new": 1_000_300.0})
    _artifacts_at(org, walked, ALICE, {f"walked-{i:03d}": 500_000.0 + i for i in range(LIST_CARDS)})
    walk = _slugs_of(client, org, "/api/v1/artifacts/")
    with monkeypatch.context() as failing:
        _failing_listing(failing, unreadable=walked / ".anton" / "artifacts")
        assert run_held_root_backfill(project_ids=_org_project_ids(org)) is not None
    assert run_sort_time_backfill(project_ids=_org_project_ids(org)) is not None
    _gone_rows(org, timed, count=200)
    asked = _location_checks(monkeypatch)

    shown = _slugs_of(client, org, "/api/v1/artifacts/")

    assert shown == walk
    assert shown[:3] == ["timed-new", "timed-old", f"walked-{LIST_CARDS - 1:03d}"]
    assert asked == [_self_heal(org, timed)]


def _in_tie_order(client, org: str, first: Path, second: Path) -> tuple[Path, Path]:
    """The projects at ``first`` and ``second`` in the order the unfiltered
    list breaks a tie between their folders: by their roots' places in the
    list (ListCandidate.source_index), which the projects' query leaves to
    the database. Each gets one folder, at one time older than any the
    tests write, named so that a tie broken by name alone would put
    ``second``'s first."""
    _artifacts_at(org, first, ALICE, {"order-b": 100.0})
    _artifacts_at(org, second, ALICE, {"order-a": 100.0})
    shown = _slugs_of(client, org, "/api/v1/artifacts/")
    return (first, second) if shown.index("order-b") < shown.index("order-a") else (second, first)


def _read_in_full(monkeypatch) -> list[set]:
    """For each collect pass a list runs (artifacts.list_candidates), the
    projects whose roots it reads every folder of."""
    from cowork.api.v1.endpoints import artifacts as artifacts_endpoint

    reads: list[set] = []
    real = artifacts_endpoint.list_candidates

    def list_candidates(sources, *, indexes):
        indexes = list(indexes)
        reads.append({UUID(sources[index].project_id) for index in indexes})
        return real(sources, indexes=indexes)

    monkeypatch.setattr(artifacts_endpoint, "list_candidates", list_candidates)
    return reads


@pytest.mark.parametrize("unread_by", ["held-roots", "sort-times"])
def test_rows_cut_at_the_limit_still_show_a_walked_projects_newer_folders(
    client, org, held_roots, monkeypatch, unread_by
):
    """The rows of one project were cut at the query's limit. One backfill
    or the other could not read another project, which holds a folder newer
    than most of the rows, one at the last row's time, and an older one.
    The list reads every folder of that project alone, and both list forms
    show the cards a walk shows, the newer folder among the rows' folders."""
    from cowork.api.v1.endpoints.artifacts import _NEWEST_CANDIDATES
    from cowork.services.artifact_sort_times import run_sort_time_backfill
    from cowork.services.held_artifact_roots import run_held_root_backfill

    _drop_sentinels()
    timed, walked = _bulk_paths(client, org, 0, 2)
    times = {f"timed-{i:03d}": 1_000_000.0 + i for i in range(_NEWEST_CANDIDATES + 10)}
    last_kept = times["timed-010"]
    _artifacts_at(org, timed, ALICE, times)
    _artifacts_at(org, walked, ALICE, {"walked-new": last_kept + 155.5, "walked-tied": last_kept, "walked-old": 1.0})
    lists = _list_paths(org, timed, walked)
    walk = {listed: _slugs_of(client, org, listed) for listed in lists}
    if unread_by == "held-roots":
        with monkeypatch.context() as failing:
            _failing_listing(failing, unreadable=walked / ".anton" / "artifacts")
            held = run_held_root_backfill(project_ids=_org_project_ids(org))
        assert held is not None and held.unread == 1
        assert run_sort_time_backfill(project_ids=_org_project_ids(org)) is not None
    else:
        held_roots(times=False)
        with monkeypatch.context() as failing:
            _failing_listing(failing, unreadable=walked / ".anton" / "artifacts")
            recorded = run_sort_time_backfill(project_ids=_org_project_ids(org))
        assert recorded is not None and recorded.unread == 1
    handed = _handed_to_the_file_store(monkeypatch)
    reads = _read_in_full(monkeypatch)

    shown = {listed: _slugs_of(client, org, listed) for listed in lists}

    assert shown == walk
    assert shown["/api/v1/artifacts/"][:6] == ["timed-169", "timed-168", "timed-167", "timed-166", "walked-new", "timed-165"]
    # The timed project's rows were cut, and each list read every folder of
    # the walked project only.
    assert [len(found.rows) for found in handed] == [_NEWEST_CANDIDATES, _NEWEST_CANDIDATES, 0]
    walked_id = _project_id(org, walked)
    assert reads == [{walked_id}, set(), {walked_id}]


@pytest.mark.parametrize("run", ["read-whole", "past-the-cap"])
def test_rows_cut_inside_a_time_two_projects_share_list_the_tie_in_walk_order(
    client, org, held_roots, monkeypatch, run
):
    """The rows were cut at the query's limit inside a run of folders that
    share one time across two projects. The query cuts that run by name,
    while a walk puts the folders of the earlier root first, so the rows
    kept at that time may not be the ones a walk shows. newest_folders
    reads the rest of the run (NewestFolders.tied), and the list sorts the
    whole run by root, then name, opening only the folders it shows. A run
    longer than artifact_sort_times._TIED_ROWS is not read: the list uses
    only the rows newer than that time, and as those fill too few cards, it
    reads every folder. Either way both list forms show the cards a walk
    shows, and as every row names a folder that is there, no list asks for
    a location check."""
    from cowork.api.v1.endpoints.artifacts import _NEWEST_CANDIDATES
    from cowork.services import artifact_sort_times
    from cowork.services.artifacts import LIST_CARDS

    _drop_sentinels()
    first, second = _in_tie_order(client, org, *_bulk_paths(client, org, 0, 2))
    tied_at = 1_000_000.0
    _artifacts_at(org, first, ALICE, {"z-tied": tied_at})
    newer = {f"newer-{i:03d}": tied_at + 1 + i for i in range(LIST_CARDS - 2)}
    tied = {f"a-tied-{i:03d}": tied_at for i in range(_NEWEST_CANDIDATES)}
    _artifacts_at(org, second, ALICE, newer | tied)
    lists = _list_paths(org, first, second)
    walk = {listed: _slugs_of(client, org, listed) for listed in lists}
    held_roots()
    if run == "past-the-cap":
        monkeypatch.setattr(artifact_sort_times, "_TIED_ROWS", _NEWEST_CANDIDATES - 1)
    asked = _location_checks(monkeypatch)
    reads = _read_in_full(monkeypatch)

    shown = {listed: _slugs_of(client, org, listed) for listed in lists}

    assert shown == walk
    assert shown["/api/v1/artifacts/"][LIST_CARDS - 2 :] == ["z-tied", "a-tied-000"]
    assert asked == []
    # One collect pass per list over the projects with no rows to trust
    # (none here), and, past the cap, one more over every folder of the
    # lists whose rows were cut: every project's, then the second's.
    if run == "read-whole":
        assert reads == [set(), set(), set()]
    else:
        first_id, second_id = _project_id(org, first), _project_id(org, second)
        assert reads == [set(), {first_id, second_id}, set(), set(), {second_id}]


@pytest.mark.parametrize("unfiltered", [False, True], ids=["one-project", "every-project"])
def test_the_list_makes_no_more_file_store_calls_at_1000_folders_of_one_time_than_at_80(
    client, org, held_roots, monkeypatch, unfiltered
):
    """One project's root with 1,000 artifacts that share one time, as an
    archive's extraction or a bulk copy leaves them, against one with 80:
    the query's limit cuts that run, newest_folders reads the rest of it
    (NewestFolders.tied), and both list forms open only the folders they
    show, the first 80 by name, as a walk shows them. A list that read
    every folder to order the run fails here."""
    [path] = _bulk_paths(client, org, 0, 1)
    tied_at = 1_000_000.0
    _artifacts_at(org, path, ALICE, {f"chart-{i:04d}": tied_at for i in range(80)})
    held_roots()
    listed = "/api/v1/artifacts/" if unfiltered else f"/api/v1/artifacts/?project_id={_project_id(org, path)}"
    small, small_calls = _file_store_get(client, monkeypatch, listed, _headers(org, ALICE))
    _artifacts_at(org, path, ALICE, {f"chart-{i:04d}": tied_at for i in range(80, 1000)})
    held_roots()
    large, large_calls = _file_store_get(client, monkeypatch, listed, _headers(org, ALICE))

    assert (len(small), len(large)) == (80, 80)
    assert [card["slug"] for card in large] == _walked_slugs(client, org, listed, monkeypatch)
    assert [card["slug"] for card in large] == [f"chart-{i:04d}" for i in range(80)]
    assert large_calls <= small_calls, f"{small_calls} file store calls at 80 tied artifacts, {large_calls} at 1,000"


def _folders_with_no_card(project_path: Path, *, unparsable: dict[str, float], emptied: dict[str, float]) -> None:
    """Folders of the project's shared root that build no card: one per name
    in ``unparsable``, whose metadata.json does not parse, and one per name
    in ``emptied``, whose metadata.json is a valid record until
    ``_empty_folders`` removes it. Each metadata.json is modified at its
    name's time."""
    base = project_path / ".anton" / "artifacts"
    for slug, at in unparsable.items():
        (base / slug).mkdir(parents=True)
        (base / slug / "metadata.json").write_text("not json")
        os.utime(base / slug / "metadata.json", (at, at))
    for slug, at in emptied.items():
        _write_artifact(base, slug)
        os.utime(base / slug / "metadata.json", (at, at))


def _empty_folders(project_path: Path, slugs) -> None:
    """Remove the metadata.json of each of ``slugs`` outside the server, as
    an agent's rm does, which leaves the folder's sort-time row."""
    for slug in slugs:
        (project_path / ".anton" / "artifacts" / slug / "metadata.json").unlink()


def test_rows_of_folders_that_build_no_card_ask_for_the_projects_check_that_drops_them(
    client, org, held_roots, monkeypatch
):
    """170 rows newer than every artifact name folders that build no card:
    their metadata.json does not parse, or is gone. They fill the query's
    limit, so both list forms read every folder rather than come up short,
    show the cards a walk shows, and ask for the project's location check.
    That check keeps only the rows of folders that hold an id, so the next
    lists open only the folders the rows name, and ask for nothing."""
    from cowork.services.artifact_autopublish import _record_listed_locations

    [path] = _bulk_paths(client, org, 0, 1)
    kept = {f"chart-{i:03d}": 1_000_000.0 + i for i in range(80)}
    _artifacts_at(org, path, ALICE, kept)
    unparsable = {f"junk-{i:03d}": 2_000_000.0 + i for i in range(85)}
    emptied = {f"emptied-{i:03d}": 2_000_100.0 + i for i in range(85)}
    _folders_with_no_card(path, unparsable=unparsable, emptied=emptied)
    held_roots()
    _empty_folders(path, emptied)
    assert _sort_time_slugs(org) == set(kept) | set(unparsable) | set(emptied)
    lists = _list_paths(org, path)
    walk = {listed: _walked_slugs(client, org, listed, monkeypatch) for listed in lists}
    asked = _location_checks(monkeypatch)
    reads = _read_in_full(monkeypatch)

    shown = {listed: _slugs_of(client, org, listed) for listed in lists}

    assert shown == walk
    assert shown["/api/v1/artifacts/"] == sorted(kept, reverse=True)
    project_id = _project_id(org, path)
    assert reads == [set(), {project_id}, set(), {project_id}]
    heal = _self_heal(org, path)
    assert asked == [heal, heal]

    asked.clear()
    reads.clear()
    asyncio.run(_record_listed_locations(heal))

    assert _sort_time_slugs(org) == set(kept)
    assert {listed: _slugs_of(client, org, listed) for listed in lists} == walk
    assert reads == [set(), set()]
    assert asked == []


def test_a_row_whose_name_a_walk_skips_opens_no_folder(client, org, held_roots, monkeypatch):
    """Rows are hints, and no writer stores a dot-named one
    (record_sort_times). One written straight to the table, newer than
    every artifact, beside a dot-named folder that would build a card,
    opens nothing: both list forms show the cards a walk shows."""
    from cowork.models.artifact_sort_time import ArtifactSortTime

    [path] = _bulk_paths(client, org, 0, 1)
    _artifacts_at(org, path, ALICE, {"visible": 1_000_100.0})
    _write_artifact(path / ".anton" / "artifacts", ".shadow")
    held_roots()
    with Session(_engine()) as raw:
        raw.add(
            ArtifactSortTime(org_id=org, project_id=_project_id(org, path), slug=".shadow", sort_time=2_000_000.0)
        )
        raw.commit()
    lists = _list_paths(org, path)

    shown = {listed: _slugs_of(client, org, listed) for listed in lists}

    assert shown == {listed: _walked_slugs(client, org, listed, monkeypatch) for listed in lists}
    assert list(shown.values()) == [["visible"], ["visible"]]


def test_a_folder_of_the_same_name_in_another_root_never_stands_for_a_gone_one(
    client, org, held_roots, monkeypatch
):
    """The rows were cut at the query's limit, and most of them name folders
    deleted behind the server's back from one project's root, while another
    project holds folders of those names and a folder newer than every row.
    The list reads every folder, as a walk does, and asks for the location
    check of the first project alone: a folder of one root never answers
    for a row of another, and the other project's rows name folders that
    are there."""
    emptied, other = _bulk_paths(client, org, 0, 2)
    _artifacts_at(org, emptied, ALICE, {"still-here": 1_000_100.0})
    namesakes = {f"gone-{i:03d}": 100.0 + i for i in range(200)}
    _artifacts_at(org, other, ALICE, {"other-new": 3_000_000.0} | namesakes)
    held_roots()
    _gone_rows(org, emptied, count=200)
    walk = _walked_slugs(client, org, "/api/v1/artifacts/", monkeypatch)
    asked = _location_checks(monkeypatch)

    shown = _slugs_of(client, org, "/api/v1/artifacts/")

    assert shown == walk
    assert shown[:3] == ["other-new", "still-here", "gone-199"]
    assert asked == [_self_heal(org, emptied)]


def _held_conversations(org: str) -> set:
    from cowork.models.held_artifact_root import HeldArtifactRoot

    with Session(_engine()) as raw:
        return set(
            raw.exec(
                select(HeldArtifactRoot.conversation_id).where(
                    HeldArtifactRoot.org_id == org, HeldArtifactRoot.conversation_id.is_not(None)
                )
            ).all()
        )


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
