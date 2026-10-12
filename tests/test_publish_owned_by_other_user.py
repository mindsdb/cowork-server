"""A report owned by another account is a 409, never a silent copy.

A fullstack report_id is global, so a folder that carries someone else's
report_id makes the upload service answer 409 `artifact_owned_by_other_user`.
Retrying without the report_id would publish a copy under a new URL; every
publish surface says so instead.
"""
from __future__ import annotations

import copy
import pickle
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import cowork.services.publish as publish_mod
from cowork.api.v1.endpoints import publish as publish_ep
from cowork.services.publish import PublishOwnedByOtherUser

DETAIL = {
    "code": "artifact_owned_by_other_user",
    "message": "This artifact was published by another owner.",
}


def _client() -> TestClient:
    app = FastAPI()
    app.include_router(publish_ep.router, prefix="/api/v1/publish")
    return TestClient(app)


def _patch_context():
    return patch.object(
        publish_ep, "_desktop_context",
        lambda raw: (Path(raw), Path(raw).parent, "key", "https://api.mindshub.ai"),
    )


def test_the_error_carries_the_wire_detail():
    err = PublishOwnedByOtherUser()
    assert isinstance(err, RuntimeError)
    assert str(err) == DETAIL["message"]
    assert err.detail() == DETAIL
    # Exceptions are rebuilt from their args when pickled or copied.
    assert pickle.loads(pickle.dumps(err)).detail() == DETAIL
    assert str(copy.copy(err)) == DETAIL["message"]


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


def test_desktop_publish_answers_409_with_the_code():
    with _patch_context(), patch.object(publish_ep, "_publish", side_effect=PublishOwnedByOtherUser()):
        res = _client().post("/api/v1/publish/", json={"path": "/some/art"})
    assert res.status_code == 409
    assert res.json()["detail"] == DETAIL


def test_desktop_update_answers_409_with_the_code():
    with patch.object(publish_ep, "_update", side_effect=PublishOwnedByOtherUser()):
        res = _client().post("/api/v1/publish/update", json={"path": "/some/art"})
    assert res.status_code == 409
    assert res.json()["detail"] == DETAIL


def test_other_runtime_errors_still_map_to_502():
    exc = RuntimeError("Publishing failed: connection reset")
    with _patch_context(), patch.object(publish_ep, "_publish", side_effect=exc):
        res = _client().post("/api/v1/publish/", json={"path": "/some/art"})
    assert res.status_code == 502
