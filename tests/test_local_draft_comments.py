"""Draft comments on a local deployment that keeps each person's space apart.

The web client turns comments on with the comments-access POST and only then
shows the comment layer. Local tenancy has no authorization service to
provision, and needs none: the comment routes keep the threads in the
artifact's own folder (`LocalArtifactComments`), found only among the caller's
artifacts roots. So the owner's POST answers on, the owner's comments come
back, and another person's calls to the same draft answer 404.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

# Imported at module scope on purpose: cowork.server builds an app at import
# time (see tests/test_trusted_identity_local.py).
import cowork.server as cowork_server
from cowork.common.settings.app_settings import get_app_settings
from cowork.services import artifact_access, artifact_authorization_identity
from cowork.services.artifact_identity import artifact_key, ensure_full_id
from tests._live_apps import (
    BEARER_TOKEN,
    PARTITION_A,
    SUB_A,
    A,
    B,
    clear_live_app_rows,
    general_project,
    person_scope,
)


@pytest.fixture
def deployment(personal_deployment, monkeypatch, tmp_path):
    """A stack's settings: every resource each person's own, with no
    authorization service, which a local deployment never has."""
    monkeypatch.setenv("COWORK_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("COWORK_SHARED_DIR", str(tmp_path / "shared"))
    monkeypatch.setenv("COWORK_PROJECTS_DIR", str(tmp_path / "home" / "projects"))
    monkeypatch.setenv("COWORK_AUTH_TOKEN", BEARER_TOKEN)
    get_app_settings.cache_clear()

    def no_authorization_service(*_args, **_kwargs):
        raise AssertionError("local tenancy asked the authorization service")

    monkeypatch.setattr(
        artifact_authorization_identity, "ensure_authorization_key", no_authorization_service
    )
    monkeypatch.setattr(
        artifact_access, "provision_draft_review_access", no_authorization_service
    )
    yield
    clear_live_app_rows()
    get_app_settings.cache_clear()


@pytest.fixture
def client(deployment) -> TestClient:
    get_app_settings.cache_clear()
    return TestClient(cowork_server.create_app())


@pytest.fixture
def draft(deployment) -> tuple[str, str, Path]:
    """Tester A's draft artifact in A's own `general` project."""
    project_id, project_path = general_project(person_scope(PARTITION_A, SUB_A))
    folder = Path(project_path) / ".anton" / "artifacts" / "comment-test"
    folder.mkdir(parents=True)
    (folder / "index.html").write_text("<html><body><h1>Comment test</h1></body></html>")
    (folder / "metadata.json").write_text(
        json.dumps({"slug": "comment-test", "name": "comment test", "type": "html-app"})
    )
    artifact_id, _metadata = ensure_full_id(folder)
    return str(project_id), artifact_id, folder


def _comments_access(project_id: str, artifact_id: str) -> str:
    return f"/api/v1/artifacts/workspace/{project_id}/{artifact_id}/comments-access"


def _threads(artifact_id: str) -> str:
    return f"/api/v1/artifact-comments/artifact/{artifact_id}/threads"


def test_the_owner_turns_draft_comments_on(client, draft):
    project_id, artifact_id, folder = draft

    response = client.post(_comments_access(project_id, artifact_id), headers=A)

    assert response.status_code == 200
    body = response.json()
    # The keys the web client reads, and no others.
    assert set(body) == {"enabled", "artifactKey", "scope", "capabilities", "currentRevision"}
    assert body["enabled"] is True
    assert body["scope"] == "owner"
    assert body["artifactKey"] == artifact_key(artifact_id)
    assert body["capabilities"]["role"] == "owner"
    assert body["currentRevision"]["path"] == "index.html"
    # The org-mode grant marker reopens a folder to co-members; nothing here
    # may write it.
    assert not (folder / ".revisions" / "draft-review.json").exists()


def test_the_owners_comment_comes_back_on_the_next_read(client, draft):
    project_id, artifact_id, folder = draft
    client.post(_comments_access(project_id, artifact_id), headers=A)

    created = client.post(
        _threads(artifact_id),
        headers=A,
        json={"text": "Make the heading bigger", "selector": "h1"},
    )
    listed = client.get(_threads(artifact_id), headers=A)

    assert created.status_code == 200
    assert listed.status_code == 200
    [thread] = listed.json()["threads"]
    assert thread["id"] == created.json()["id"]
    assert thread["payload"]["text"] == "Make the heading bigger"
    assert (folder / ".revisions" / "comments.json").exists()


def test_another_person_gets_404_from_the_drafts_comment_routes(client, draft):
    project_id, artifact_id, folder = draft
    client.post(_threads(artifact_id), headers=A, json={"text": "Only mine"})
    journal = (folder / ".revisions" / "comments.json").read_text()

    answers = [
        client.post(_comments_access(project_id, artifact_id), headers=B),
        client.get(_threads(artifact_id), headers=B),
        client.post(_threads(artifact_id), headers=B, json={"text": "Not yours"}),
        client.get(f"/api/v1/artifact-comments/artifact/{artifact_id}/stream", headers=B),
    ]

    assert [answer.status_code for answer in answers] == [404, 404, 404, 404]
    assert (folder / ".revisions" / "comments.json").read_text() == journal


def test_opening_an_artifact_logs_no_error(client, draft, caplog):
    project_id, artifact_id, _folder = draft
    caplog.set_level(logging.DEBUG)

    review = client.get(
        f"/api/v1/artifacts/workspace/{project_id}/{artifact_id}/review", headers=A
    )
    access = client.post(_comments_access(project_id, artifact_id), headers=A)

    assert [r for r in caplog.records if r.levelno >= logging.ERROR] == []
    assert (review.status_code, access.status_code) == (200, 200)
