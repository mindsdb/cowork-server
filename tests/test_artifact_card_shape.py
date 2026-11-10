"""The keys an artifact card carries, in order, on every path that sends one.

Captured from the dict cards before the card became a model; a change here is
a change to the wire shape the renderer reads."""
from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from cowork.api.v1.endpoints import artifacts as ep
from sqlmodel import Session, select

from cowork.common.settings.app_settings import get_app_settings
from cowork.db.scoped import LOCAL_SCOPE
from cowork.db.session import get_engine
from cowork.models.conversation import Conversation
from cowork.models.project import Project
from cowork.harnesses.anton_harness.stream_formatter import ArtifactCreated, format_responses_stream
from cowork.services import artifacts as a
from cowork.services.task_objects import cards_for_slugs
from test_artifacts_api_tenancy import (  # noqa: F401  (fixtures)
    ORG_A, _project_with_artifact, _published, _scoped, local_mode, org_mode,
)
from test_live_apps_sharing import _client, _duplicate, _share, people, stack  # noqa: F401


@pytest.fixture
def session(tmp_path):
    """A session that removes the projects and chats the test created under ``tmp_path``."""
    engine = get_engine(get_app_settings().database.uri)
    with Session(engine) as s:
        yield s
        s.rollback()
        for row in s.exec(select(Project)).all():
            if str(row.path).startswith(str(tmp_path)):
                for chat in s.exec(select(Conversation).where(Conversation.project_id == row.id)).all():
                    s.delete(chat)
                s.delete(row)
        s.commit()


_HEAD = [
    "id", "slug", "title", "description", "type", "kind", "ext", "updated", "mtime", "live", "bg",
    "fileCount", "folder", "path", "primary", "projectId", "projectName", "originConversationId",
    "publishedUrl", "modified",
]
DESKTOP = _HEAD + [
    "accessEmails", "orgAllowed", "ownerOnly", "allowedTeamIds", "allowedUserIds",
    "accessMode", "accessProtected", "accessPassword", "artifactKey", "serveUrl", "draftUrl",
]
ORG = _HEAD + [
    "orgAllowed", "ownerOnly", "allowedTeamIds", "allowedUserIds",
    "accessMode", "accessProtected", "artifactKey", "serveUrl", "draftUrl",
]


def _json(card):
    return card.to_json()


def _password(folder):
    _published(folder, mode="password", requires_password=True, access_password="s3cret")


async def test_a_desktop_list_card(session, tmp_path, local_mode):
    row, folder = _project_with_artifact(session, tmp_path, name="mine", org_id=None, slug="pw")
    _password(folder)

    [card] = await ep.list_artifacts(LOCAL_SCOPE, project_id=row.id, project_path=None)

    assert list(card) == DESKTOP + ["capabilities"]


async def test_an_org_list_card(session, tmp_path, org_mode):
    row, folder = _project_with_artifact(session, tmp_path, name="mine", org_id=ORG_A, slug="pw")
    _password(folder)
    scope = _scoped(session, ORG_A).scope

    [in_project] = await ep.list_artifacts(scope, project_id=row.id, project_path=None)
    # The unfiltered list spans every ORG_A project, including any another
    # test file left behind, so pick this test's card by its folder.
    [unfiltered] = [
        card
        for card in await ep.list_artifacts(scope, project_id=None, project_path=None)
        if card["folder"] == str(folder)
    ]

    assert list(in_project) == ORG + ["capabilities"]
    assert list(unfiltered) == ORG + ["capabilities"]


def test_a_desktop_inline_card(session, tmp_path, local_mode):
    _row, folder = _project_with_artifact(session, tmp_path, name="mine", org_id=None, slug="pw")
    _password(folder)
    no_primary = folder.parent / "noprimary"
    no_primary.mkdir()
    (no_primary / "metadata.json").write_text(json.dumps({"slug": "noprimary"}))

    cards = [_json(c) for c in cards_for_slugs(folder.parent, ["pw", "noprimary"], scope=LOCAL_SCOPE, project_id=None, project_name="mine")]

    assert [list(c) for c in cards] == [DESKTOP, DESKTOP[:-1]]


async def test_an_org_inline_card_and_its_stream_event(session, tmp_path, org_mode):
    row, folder = _project_with_artifact(session, tmp_path, name="mine", org_id=ORG_A, slug="pw")
    _password(folder)

    [card] = cards_for_slugs(folder.parent, ["pw"], scope=_scoped(session, ORG_A).scope, project_id=str(row.id), project_name="mine")

    assert list(_json(card)) == ORG
    assert list(await _stream_event(card)) == ORG


async def _stream_event(card) -> dict:
    """The card ``response.artifact_created`` carries for ``card``."""
    sent: list[tuple[str, dict]] = []

    async def events():
        yield ArtifactCreated(card)

    async for _chunk in format_responses_stream(events(), model="m", event_sink=lambda kind, data: sent.append((kind, data))):
        pass
    [artifact] = [data["artifact"] for kind, data in sent if kind == "response.artifact_created"]
    return artifact


def test_a_live_app_copy_card(people):
    client = _client()
    share = _share(client, people["a_project"], email="bob@example.com").json()

    card = _duplicate(client, share["shareId"], people["b_project"]).json()

    assert list(card) == DESKTOP + ["capabilities", "receivedFrom"]


def test_a_desktop_status_answers_its_own_shape(session, tmp_path, monkeypatch, local_mode):
    from cowork.server import create_app

    _row, folder = _project_with_artifact(session, tmp_path, name="mine", org_id=None, slug="pw")
    _password(folder)
    source = a.ProjectArtifacts(
        base=folder.parent, project_id=None, project_name="mine",
        trusted_anchor=folder.parent.parent.parent, root_parts=(".anton", "artifacts"),
    )
    monkeypatch.setattr(ep, "artifacts_sources_for_scan", lambda *, scope: [source])

    res = TestClient(create_app()).get("/api/v1/artifacts/status", params={"path": str(folder / "index.html")})

    assert res.status_code == 200
    assert list(res.json()) == [
        "publishedUrl", "modified", "accessMode", "accessProtected", "accessEmails",
        "orgAllowed", "ownerOnly", "artifactKey", "linkOwnerRecorded",
    ]
