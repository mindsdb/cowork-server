"""Inline artifact cards — the guarantee that every artifact a turn produces
gets an openable card that survives reload.

Covers the shared end-of-turn path (services.task_objects.finalize_turn_artifacts
+ services.artifacts.card_for_folder) and the persistence fix that lets an
artifact-only turn (no body text) keep its `response.artifact_created` event.
"""
from __future__ import annotations

import json

import pytest
from sqlmodel import Session

from cowork.common.settings.app_settings import get_app_settings
from cowork.db.session import get_engine
from cowork.services import task_objects as t
from cowork.services.artifacts import card_for_folder
from cowork.services.conversations import ConversationService


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


# ── finalize: index + cards from one diff ──────────────────────────────────

def test_finalize_surfaces_only_new_artifacts(session, tmp_path):
    base = tmp_path / "artifacts"
    base.mkdir()
    _make_artifact(base, "old", files={"a.md": "old"}, meta={"slug": "old", "type": "document"})
    before = t.snapshot_artifact_slugs(base)

    _make_artifact(base, "new", files={"r.md": "new"}, meta={"slug": "new", "name": "New", "type": "document"})

    conv = ConversationService(session).create_conversation(topic="t")
    cards = t.finalize_turn_artifacts(conv.id, conv.project_id, base, before)
    assert [c["slug"] for c in cards] == ["new"]

    # Nothing new on a second pass.
    after = t.snapshot_artifact_slugs(base)
    assert t.finalize_turn_artifacts(conv.id, conv.project_id, base, after) == []


# ── persistence: the reload guarantee ──────────────────────────────────────

def test_artifact_only_turn_is_persisted(session):
    """A turn with no body text but a card event must persist, so the inline
    card replays on reload."""
    svc = ConversationService(session)
    conv = svc.create_conversation(topic="t")
    event = {"type": "response.artifact_created", "sequence_number": 1,
             "artifact": {"slug": "x", "title": "X", "type": "document", "path": "/p/x/x.md", "ext": ".md"}}

    svc.save_assistant_turn(conv.id, "", [event], harness="anton")

    messages = svc.get_messages(conv.id)
    assistant = [m for m in messages if m["role"] == "assistant"]
    assert len(assistant) == 1
    assert assistant[0]["events"] == [event]


def test_empty_turn_with_no_events_is_not_persisted(session):
    svc = ConversationService(session)
    conv = svc.create_conversation(topic="t")
    svc.save_assistant_turn(conv.id, "", [], harness="anton")
    assert [m for m in svc.get_messages(conv.id) if m["role"] == "assistant"] == []
