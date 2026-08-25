from pathlib import Path

import pytest
from fastapi import HTTPException

from cowork.services import local_artifact_comments as comments


def _service(tmp_path, monkeypatch):
    folder = tmp_path / "artifact"
    folder.mkdir()
    monkeypatch.setattr(comments, "artifacts_sources_for_scan", lambda: [])
    monkeypatch.setattr(
        comments,
        "resolve_artifact_folder",
        lambda _sources, _stable_id: (object(), folder, {}),
    )
    return comments.LocalArtifactComments("11111111-1111-1111-1111-111111111111")


def test_local_comment_lifecycle_is_atomic_and_revision_aware(tmp_path: Path, monkeypatch):
    service = _service(tmp_path, monkeypatch)
    created = service.create({
        "text": "Fix this title",
        "selector": '{"type":"text-quote","exact":"Old title"}',
        "revisionId": "rev-4",
        "kind": "review",
    })

    assert created["payload"]["revision_id"] == "rev-4"
    assert created["payload"]["author"]["user_id"] == "desktop-owner"
    assert service.list_payload("all")["threads"] == [created]

    replied = service.reply(created["id"], {"text": "On it"})
    assert replied["version"] == 2
    reply_id = replied["payload"]["replies"][0]["id"]

    edited = service.edit_reply(created["id"], reply_id, {"text": "Fixed locally"})
    assert edited["payload"]["replies"][0]["text"] == "Fixed locally"

    resolved = service.status(created["id"], {"status": "resolved"})
    assert resolved["status"] == "resolved"
    assert service.list_payload("resolved")["threads"][0]["id"] == created["id"]

    service.delete_thread(created["id"])
    assert service.list_payload("all")["threads"] == []


def test_local_comment_payload_has_owner_capabilities(tmp_path: Path, monkeypatch):
    payload = _service(tmp_path, monkeypatch).list_payload("all")

    assert payload["viewer"]["role"] == "owner"
    assert payload["capabilities"] == {
        "canComment": True,
        "canResolve": True,
        "canAddressWithAgent": True,
    }
    assert payload["unreadCount"] == 0


def test_local_comment_rejects_an_unbounded_selector(tmp_path: Path, monkeypatch):
    service = _service(tmp_path, monkeypatch)

    with pytest.raises(HTTPException) as exc:
        service.create({"text": "Fix this", "selector": "x" * 2_001})

    assert exc.value.status_code == 422
