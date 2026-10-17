"""A report owned by another account is a 409, never a silent copy.

A fullstack report_id is global, so a folder that carries someone else's
report_id makes the upload service answer 409 `artifact_owned_by_other_user`.
Retrying without the report_id would publish a copy under a new URL; every
publish surface says so instead.
"""
from __future__ import annotations

import copy
import json
import pickle

import pytest
from fastapi import HTTPException

import cowork.services.publish as publish_mod
from cowork.db.scoped import LOCAL_SCOPE
from cowork.services.publish import PublishOwnedByOtherUser

DETAIL = {
    "code": "artifact_owned_by_other_user",
    "message": "This artifact was published by another owner.",
}


def test_the_error_is_a_409_with_the_wire_detail():
    err = PublishOwnedByOtherUser()
    assert isinstance(err, HTTPException)
    assert err.status_code == 409
    assert err.detail == DETAIL
    # Exceptions are rebuilt from their args when pickled or copied.
    assert pickle.loads(pickle.dumps(err)).detail == DETAIL
    assert copy.copy(err).detail == DETAIL


def test_the_publisher_error_becomes_the_typed_one_and_writes_nothing(tmp_path, monkeypatch):
    from anton.publisher import ArtifactOwnedByOtherUserError

    monkeypatch.setenv("ANTON_COWORK_STATE_DIR", str(tmp_path / "state"))
    monkeypatch.setattr(publish_mod, "vault_for_scope", lambda scope: object())
    calls = []

    def reject(path, **kwargs):
        calls.append(kwargs)
        raise ArtifactOwnedByOtherUserError()

    monkeypatch.setattr("anton.publisher.publish", reject)
    page = tmp_path / "report.html"
    page.write_text("<html><body>hi</body></html>", encoding="utf-8")

    with pytest.raises(PublishOwnedByOtherUser):
        publish_mod.publish_artifact(
            page, artifacts_base=tmp_path, api_key="k", publish_url="https://view.example",
        )
    assert len(calls) == 1
    assert not (tmp_path / ".published.json").exists()


def test_update_lets_the_typed_error_through_and_keeps_the_record(tmp_path, monkeypatch):
    """The real Update path: update_artifact -> publish_artifact -> anton's 409."""
    from anton.publisher import ArtifactOwnedByOtherUserError

    monkeypatch.setenv("ANTON_COWORK_STATE_DIR", str(tmp_path / "state"))
    monkeypatch.setattr(publish_mod, "vault_for_scope", lambda scope: object())
    page = tmp_path / "report.html"
    page.write_text("<html><body>hi</body></html>", encoding="utf-8")
    record = {"report.html": {
        "report_id": "theirs", "url": "https://view.example/a/theirs", "published": True,
        "mode": "public", "requires_password": False,
    }}
    (tmp_path / ".published.json").write_text(json.dumps(record), encoding="utf-8")
    monkeypatch.setattr(
        publish_mod, "desktop_artifact_and_base", lambda raw, *, scope: publish_mod.DesktopArtifact(page, tmp_path)
    )
    monkeypatch.setattr(publish_mod, "desktop_publish_credential", lambda: ("k", "https://view.example"))
    calls = []

    def reject(path, **kwargs):
        calls.append(kwargs)
        raise ArtifactOwnedByOtherUserError()

    monkeypatch.setattr("anton.publisher.publish", reject)

    with pytest.raises(PublishOwnedByOtherUser):
        publish_mod.update_artifact(str(page), scope=LOCAL_SCOPE)
    assert [c["report_id"] for c in calls] == ["theirs"]
    assert json.loads((tmp_path / ".published.json").read_text(encoding="utf-8")) == record

