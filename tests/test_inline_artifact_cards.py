"""Inline artifact cards — the guarantee that every artifact a turn produces
gets an openable card that survives reload.

Covers the shared end-of-turn path (services.task_objects.index_turn_artifacts
+ services.artifacts.card_for_folder), the serve URL a turn's card opens
through, and the persistence fix that lets an artifact-only turn (no body
text) keep its `response.artifact_created` event.
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import pytest
from sqlmodel import Session

from cowork.common.settings.app_settings import get_app_settings
from cowork.db.session import get_engine
from cowork.harnesses.anton_harness.harness import AntonHarness
from cowork.harnesses.anton_harness.stream_formatter import ArtifactCreated
from cowork.services import task_objects as t
from cowork.services.artifacts import card_for_folder
from cowork.db.scoped import LOCAL_SCOPE, ScopedSession
from cowork.services.conversations import ConversationService
from cowork.services.projects import ProjectService


@pytest.fixture
def session():
    engine = get_engine(get_app_settings().database.uri)
    with Session(engine) as s:
        yield s


def _make_artifact(base, slug, *, files: dict[str, str], meta: dict) -> None:
    folder = base / slug
    folder.mkdir(parents=True)
    for rel, body in files.items():
        path = folder / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(body)
    (folder / "metadata.json").write_text(json.dumps(meta))


# ── card builder: opens the right file ────────────────────────────────────

def test_card_for_folder_prefers_html_entry_over_alphabetical(tmp_path):
    # No explicit primary; index.html must win over an alphabetically-earlier
    # asset (data/prices.csv) so the card opens the app, not the dataset.
    _make_artifact(
        tmp_path, "dash",
        files={"index.html": "<html></html>", "data/prices.csv": "a,b\n1,2"},
        meta={"slug": "dash", "name": "Dash", "type": "html-app"},
    )
    card = card_for_folder(tmp_path / "dash")
    assert card["path"].endswith("index.html")
    assert card["ext"] == ".html"
    assert card["slug"] == "dash"
    assert card["title"] == "Dash"


def test_card_for_folder_none_on_unreadable_metadata(tmp_path):
    folder = tmp_path / "broken"
    folder.mkdir()
    (folder / "metadata.json").write_text("{ not json")
    assert card_for_folder(folder) is None


# ── serve URL: addressed by the project's name, not its label ─────────────

@pytest.fixture
def api(tmp_path, monkeypatch):
    """The real app over a projects root of this test's own.

    Yielded bare: entering the client runs the app lifespan, which re-creates
    the schema conftest already built.
    """
    from fastapi.testclient import TestClient

    from cowork.server import create_app

    monkeypatch.setenv("COWORK_PROJECTS_DIR", str(tmp_path / "projects"))
    get_app_settings.cache_clear()
    yield TestClient(create_app())
    get_app_settings.cache_clear()


def _turn_cards(monkeypatch, conversation) -> list[dict]:
    """The cards one in-process turn emits after its agent wrote `dash`."""

    class _Session:
        async def turn_stream(self, *_args, **_kwargs):
            return
            yield

    async def _fake_build(self, conversation, **_kwargs):
        return _Session(), None, None

    async def _drain():
        return [
            event
            async for event in AntonHarness().stream_response(
                conversation=conversation, input=[{"type": "text", "text": "hi"}]
            )
        ]

    monkeypatch.setattr(AntonHarness, "_build_chat_session", _fake_build)
    monkeypatch.setattr(
        t, "turn_artifact_changes",
        lambda *_a, **_k: t.ArtifactChanges(created=["dash"], touched={"dash"}),
    )
    monkeypatch.setattr(t, "record_new_artifacts", lambda *_a, **_k: None)
    events = asyncio.run(_drain())
    return [event.artifact for event in events if isinstance(event, ArtifactCreated)]


@pytest.mark.usefixtures("cleanup_tmp_projects")
@pytest.mark.parametrize("typed, chosen_folder, taken", [
    pytest.param("Sales Q3", None, None, id="space-becomes-hyphen"),
    pytest.param("Звіт продажів", None, None, id="non-latin-becomes-untitled-project"),
    # The serve route finds a project in a folder the user chose by its row,
    # not by scanning the projects root.
    pytest.param("My notes", "chosen/notes", None, id="a-folder-the-user-chose"),
    # Another project already holds the sanitized name, so this one is stored
    # as Sales-Q3-2 under the label "Sales Q3". Sanitizing the label again
    # gives the other project's name, whose folder has no such artifact.
    pytest.param("Sales Q3", None, "Sales-Q3", id="name-taken-by-another-project"),
])
def test_a_turn_card_serves_from_a_project_whose_label_is_not_its_name(
    api, monkeypatch, session, tmp_path, typed, chosen_folder, taken
):
    """`serve_artifact_file` resolves its project segment by the project's
    `name`. The label is what the user typed, and the two differ whenever
    sanitizing or de-duplicating changed the name. A card whose serve URL
    carries the label answers 404 on Download and on "open in a browser tab"."""
    projects = ProjectService(ScopedSession(session, LOCAL_SCOPE))
    if taken:
        projects.create_project(taken)
    path = None
    if chosen_folder:
        path = tmp_path / chosen_folder
        path.mkdir(parents=True)
    project = projects.create_project(typed, path=path)
    assert project.display_name != project.name
    _make_artifact(
        Path(project.path) / ".anton" / "artifacts", "dash",
        files={"index.html": "<html></html>"},
        meta={"slug": "dash", "name": "Dash", "type": "html-app"},
    )
    conversation = SimpleNamespace(id=uuid4(), project_id=project.id, project=project)

    [card] = _turn_cards(monkeypatch, conversation)

    served = api.get(card["serveUrl"])
    assert served.status_code == 200, (card["serveUrl"], served.text)
    assert card["serveUrl"] == f"/api/v1/artifacts/serve/{project.name}/dash/index.html"
    assert card["projectName"] == project.display_name


# ── persistence: the reload guarantee ──────────────────────────────────────

def test_artifact_only_turn_is_persisted(session):
    """A turn with no body text but a card event must persist, so the inline
    card replays on reload."""
    svc = ConversationService(ScopedSession(session, LOCAL_SCOPE))
    conv = svc.create_conversation(topic="t")
    event = {"type": "response.artifact_created", "sequence_number": 1,
             "artifact": {"slug": "x", "title": "X", "type": "document", "path": "/p/x/x.md", "ext": ".md"}}

    svc.save_assistant_turn(conv.id, "", [event], harness="anton")

    messages = svc.get_messages(conv.id)
    assistant = [m for m in messages if m["role"] == "assistant"]
    assert len(assistant) == 1
    assert assistant[0]["events"] == [event]


def test_empty_turn_with_no_events_is_not_persisted(session):
    svc = ConversationService(ScopedSession(session, LOCAL_SCOPE))
    conv = svc.create_conversation(topic="t")
    svc.save_assistant_turn(conv.id, "", [], harness="anton")
    assert [m for m in svc.get_messages(conv.id) if m["role"] == "assistant"] == []
