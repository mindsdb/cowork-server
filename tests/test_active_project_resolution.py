"""The unified active-project model (Projects slice 2).

There used to be two overlapping notions of the "active" project: the client's
``selectedProject`` and the server's ``is_active`` flag. They could disagree,
so a task could land in the wrong project ("where did my task go?"). This slice
collapses them into a single source of truth:

  * For interactive use, the client's selection is canonical — it is sent as an
    explicit ``project`` on every ``/responses`` turn, so the server never has
    to guess.
  * The server keeps exactly one signal, ``last_selected_at``, and consults it
    ONLY as the fallback for headless / scheduled runs that omit a project.
    There is no separate ``is_active`` flag that can disagree with it.

These tests pin that the resolution is single-source and that headless /
scheduled runs resolve it correctly.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from uuid import UUID

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlmodel import Session, select

from cowork.api.v1.router import api_router
from cowork.common.settings.app_settings import get_app_settings
from cowork.db.session import get_engine
from cowork.handlers.responses import ResponsesHandler
from cowork.models.project import Project
from cowork.schemas.responses import ResponsesRequest
from cowork.services.projects import GENERAL_PROJECT_ID, ProjectService
from cowork.services.schedules import ScheduleService


@pytest.fixture()
def client() -> TestClient:
    app = FastAPI()
    app.include_router(api_router)
    return TestClient(app)


@pytest.fixture()
def session() -> Session:
    engine = get_engine(get_app_settings().database.uri)
    with Session(engine) as s:
        yield s


@pytest.fixture(autouse=True)
def clean_extra_projects():
    """Remove non-general projects and reset General's selection state so each
    test starts from a clean slate (mirrors test_project_metadata.py)."""
    engine = get_engine(get_app_settings().database.uri)

    def _cleanup() -> None:
        with Session(engine) as s:
            for project in s.exec(select(Project)).all():
                if project.id == GENERAL_PROJECT_ID:
                    project.last_selected_at = None
                    s.add(project)
                else:
                    path = Path(project.path)
                    if path.is_dir():
                        import shutil

                        shutil.rmtree(path, ignore_errors=True)
                    s.delete(project)
            s.commit()

    _cleanup()
    yield
    _cleanup()


def _handler(session: Session) -> ResponsesHandler:
    """A ResponsesHandler with just a session — enough to exercise
    ``_resolve_project_id`` without DB/harness bootstrap."""
    handler = object.__new__(ResponsesHandler)
    handler.session = session
    return handler


def _create_project(client: TestClient, name: str) -> dict:
    resp = client.post("/api/v1/projects/", json={"name": name})
    assert resp.status_code == 201, resp.text
    return resp.json()


# ── The flag is gone: a single source of truth ────────────────────

def test_project_model_has_no_is_active_flag():
    """The redundant, never-read ``is_active`` flag is removed entirely —
    there is nothing left to disagree with the client's selection."""
    assert not hasattr(Project(name="x", path="/tmp/x"), "is_active")


def test_project_service_exposes_single_resolution_path():
    """There is exactly one server-side resolver for the fallback project and
    no ``get_active_project`` reading a separate flag."""
    assert hasattr(ProjectService, "resolve_fallback_project")
    assert not hasattr(ProjectService, "get_active_project")


# ── Fallback resolution (headless / scheduled) ────────────────────

def test_fallback_defaults_to_general_when_nothing_selected(session: Session):
    assert ProjectService(session).resolve_fallback_project().id == GENERAL_PROJECT_ID


def test_fallback_follows_most_recently_selected(session: Session, client: TestClient):
    older = _create_project(client, "fallback-older")
    newer = _create_project(client, "fallback-newer")

    # Select older first, then newer — the most recent selection wins.
    ProjectService(session).touch_last_selected(UUID(older["id"]))
    ProjectService(session).touch_last_selected(UUID(newer["id"]))

    resolved = ProjectService(session).resolve_fallback_project()
    assert str(resolved.id) == newer["id"]


def test_selecting_via_patch_drives_the_fallback(session: Session, client: TestClient):
    """The client's selection PATCH (lastSelected) is the ONLY thing that moves
    the server-side fallback — proving the two notions are now one."""
    project = _create_project(client, "patch-select")

    resp = client.patch(
        f"/api/v1/projects/{project['id']}", json={"lastSelected": True}
    )
    assert resp.status_code == 200, resp.text

    resolved = ProjectService(session).resolve_fallback_project()
    assert str(resolved.id) == project["id"]


# ── Handler resolution: explicit wins, else single fallback ───────

def test_explicit_project_is_canonical_over_fallback(session: Session, client: TestClient):
    """An interactive request carries an explicit project; that always wins,
    regardless of what was most recently selected elsewhere."""
    selected = _create_project(client, "resolve-selected")
    target = _create_project(client, "resolve-target")
    # Make `selected` the most-recently-selected project.
    ProjectService(session).touch_last_selected(UUID(selected["id"]))

    # A request naming `target` resolves to target, not the selected fallback.
    req = ResponsesRequest(input="hi", project="resolve-target")
    assert str(_handler(session)._resolve_project_id(req)) == target["id"]

    # By id likewise.
    req_id = ResponsesRequest(input="hi", project_id=UUID(target["id"]))
    assert str(_handler(session)._resolve_project_id(req_id)) == target["id"]


def test_headless_request_resolves_to_most_recently_selected(session: Session, client: TestClient):
    """A headless run (no project on the request — e.g. a cron tick) resolves
    to the most-recently-selected project via the single fallback, NOT to a
    stale 'active' flag and NOT blindly to General."""
    selected = _create_project(client, "headless-selected")
    ProjectService(session).touch_last_selected(UUID(selected["id"]))

    req = ResponsesRequest(input="scheduled prompt")  # no project / project_id
    assert str(_handler(session)._resolve_project_id(req)) == selected["id"]


def test_headless_request_falls_back_to_general_when_nothing_selected(session: Session):
    req = ResponsesRequest(input="scheduled prompt")
    assert _handler(session)._resolve_project_id(req) == GENERAL_PROJECT_ID


# ── Scheduled creation uses the same single fallback ──────────────

def test_schedule_without_project_uses_fallback(session: Session, client: TestClient):
    """Creating a schedule without an explicit project pins it to the
    most-recently-selected project (the same fallback the handler uses), so a
    scheduled run and an interactive run agree on the active project."""
    selected = _create_project(client, "schedule-selected")
    ProjectService(session).touch_last_selected(UUID(selected["id"]))

    schedule = ScheduleService(session).create_schedule(
        title="nightly",
        prompt="do the thing",
        cadence="daily",
        next_run_at=datetime.now(timezone.utc),
        model="default",
    )
    assert str(schedule.project_id) == selected["id"]


def test_schedule_with_explicit_project_is_respected(session: Session, client: TestClient):
    selected = _create_project(client, "schedule-fallback")
    explicit = _create_project(client, "schedule-explicit")
    ProjectService(session).touch_last_selected(UUID(selected["id"]))

    schedule = ScheduleService(session).create_schedule(
        title="nightly",
        prompt="do the thing",
        cadence="daily",
        next_run_at=datetime.now(timezone.utc),
        model="default",
        project_id=UUID(explicit["id"]),
    )
    assert str(schedule.project_id) == explicit["id"]
