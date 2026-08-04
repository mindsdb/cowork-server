"""Chat must survive a project rename (ENG-1028).

The composer identifies the project by NAME on /responses, and the renderer
holds that name in long-lived state — so after a rename the wire can carry a
stale name. An existing conversation already pins its project via
conversation.project_id; resolving the (redundant) name for it must never
404 the turn. Creating a NEW conversation still requires a resolvable
project, so an unknown name keeps its 404 there.
"""
from __future__ import annotations

from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient
from sqlmodel import Session

from cowork.common.settings.app_settings import get_app_settings
from cowork.db.scoped import LOCAL_SCOPE, ScopedSession
from cowork.db.session import get_engine
from cowork.services.projects import ProjectService


class _StubHarness:
    """Minimal harness: one text delta, then a clean end of turn."""

    id = "stub"

    def stream_response(self, **kwargs):
        return None

    async def formatter(self, stream, model, event_sink):
        event_sink(
            "response.output_text.delta",
            {"type": "response.output_text.delta", "delta": "ok"},
        )
        if False:
            yield


@pytest.fixture()
def client():
    from cowork.server import create_app

    with patch("cowork.handlers.responses.get_harness", return_value=_StubHarness()):
        yield TestClient(create_app())


def _create_project(client: TestClient, name: str) -> dict:
    r = client.post("/api/v1/projects/", json={"name": name})
    assert r.status_code == 201, r.text
    return r.json()


def _rename_project(client: TestClient, project_id: str, name: str) -> dict:
    r = client.patch(f"/api/v1/projects/{project_id}", json={"name": name})
    assert r.status_code == 200, r.text
    return r.json()


def test_rename_then_resolve_by_new_name():
    svc = ProjectService(
        ScopedSession(Session(get_engine(get_app_settings().database.uri)), LOCAL_SCOPE)
    )
    project = svc.create_project("eng1028-svc")
    renamed = svc.update_project(project.id, name="eng1028-svc-renamed")
    assert renamed.name == "eng1028-svc-renamed"
    assert svc.get_project_by_name("eng1028-svc-renamed").id == project.id


def test_existing_conversation_survives_stale_project_name(client):
    project = _create_project(client, "eng1028-a")
    r = client.post(
        "/api/v1/conversations/", json={"topic": "t", "project": "eng1028-a"}
    )
    assert r.status_code == 201, r.text
    conv_id = r.json()["id"]

    _rename_project(client, project["id"], "eng1028-a-renamed")

    # The renderer still holds the pre-rename name for this task; the
    # conversation id alone identifies the project.
    r = client.post(
        "/api/v1/responses/",
        json={
            "input": "hello",
            "stream": False,
            "conversation": conv_id,
            "project": "eng1028-a",
        },
    )
    assert r.status_code == 200, r.text


def test_new_conversation_with_unknown_project_still_404s(client):
    r = client.post(
        "/api/v1/responses/",
        json={"input": "hello", "stream": False, "project": "eng1028-no-such"},
    )
    assert r.status_code == 404
    assert "eng1028-no-such" in r.json()["detail"]


def test_new_conversation_in_renamed_project_resolves_new_name(client):
    project = _create_project(client, "eng1028-b")
    _rename_project(client, project["id"], "eng1028-b-renamed")

    r = client.post(
        "/api/v1/responses/",
        json={"input": "hello", "stream": False, "project": "eng1028-b-renamed"},
    )
    assert r.status_code == 200, r.text
