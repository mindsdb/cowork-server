"""Publish changes (ENG-4595): the viewer shows what a draft changed against
the live page, and puts it live with the access it already has.

The republish route is called directly, as the access route's suite does
(test_artifact_access_endpoints); the published revision is read from a real
revision journal.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

import cowork.services.artifact_autopublish  # noqa: F401  (see test_artifact_access_endpoints)
from cowork.api.v1.endpoints import artifact_workspace as aw
from cowork.db.scoped import TenantScope
from cowork.schemas.artifact_access import ArtifactAccessRequest
from cowork.services import artifact_locks, artifact_revisions
from cowork.services.artifact_identity import ensure_full_id
from cowork.services.artifact_metadata import ArtifactMetadata
from _async_stub import awaitable

ORG_SCOPE = TenantScope(
    org_mode=True, org_id="0a0a0a0a-0000-4000-8000-000000000001", user_id="0b0b0b0b-0000-4000-8000-000000000001"
)
ARTIFACT_ID = "11111111-1111-1111-1111-111111111111"


@pytest.fixture
def folder(tmp_path):
    base = tmp_path / ORG_SCOPE.org_id / "proj" / ".anton" / "artifacts"
    f = base / "rep"
    f.mkdir(parents=True)
    (f / "report.html").write_text("<html><body>v1</body></html>")
    (f / "metadata.json").write_text(json.dumps({"slug": "rep", "type": "html-app"}))
    return f


def _published(folder, **entry) -> None:
    (folder / ".published.json").write_text(json.dumps({"report.html": {
        "report_id": "r1", "url": "https://view.example/r/1", "mode": "public", "published": True, **entry,
    }}))


# ── the republish route ─────────────────────────────────────────────────


@pytest.fixture
def as_owner(monkeypatch, folder):
    monkeypatch.setattr(
        aw, "_owner_workspace",
        awaitable(lambda scope, project_ref, artifact_id: (
            SimpleNamespace(project_id="proj-1"), folder, ArtifactMetadata.model_validate({"slug": "rep"}), {},
        )),
    )
    return folder


@pytest.fixture
def publish_calls(monkeypatch):
    calls = []

    def fake_publish(artifact, *, artifacts_base, api_key, publish_url,
                     password=None, access=None, scope=None, project_id=None,
                     notify=None, stored=None, owner_save=False, notices=None, job_budget_s=None,
                     on_job_accepted=None):
        assert isinstance(access, ArtifactAccessRequest)
        calls.append({
            "folder": artifact, "access": access.publisher_access(), "scope": scope,
            "notify": notify, "stored": stored, "owner_save": owner_save, "api_key": api_key,
        })
        return {"status": "ok", "url": "https://view.example/r/1"}

    monkeypatch.setattr("cowork.services.publish.publish_artifact", fake_publish)
    return calls


@pytest.fixture
def publish_context(monkeypatch):
    monkeypatch.setattr(
        "cowork.services.artifact_autopublish._publish_url",
        lambda scope: "https://api.staging.mindshub.ai",
    )
    revoked = []

    class FakeKey:
        def __init__(self, *a, **kw):
            pass

        async def get(self):
            return "turnkey-1"

        async def revoke(self):
            revoked.append(True)

    monkeypatch.setattr("cowork.services.artifact_publish_key.PublishKey", FakeKey)
    return revoked


def _lock_is_free(folder) -> bool:
    free = artifact_locks.acquire(folder.parent, folder.name, ttl_s=60)
    if free:
        artifact_locks.release(folder.parent, folder.name)
    return free


@pytest.mark.asyncio
@pytest.mark.usefixtures("granted_product_permissions")
async def test_publish_changes_keeps_the_access_and_tells_nobody(as_owner, publish_calls, publish_context):
    _published(as_owner, mode="public")

    out = await aw.republish_artifact("proj", ARTIFACT_ID, ORG_SCOPE, principal=None)

    assert out["url"] == "https://view.example/r/1"
    [call] = publish_calls
    assert call["access"]["mode"] == "public"
    # The stored access, not an owner's new choice: nothing recorded, nobody emailed.
    assert call["owner_save"] is False
    assert call["notify"] is None
    assert call["scope"] is ORG_SCOPE
    assert call["api_key"] == "turnkey-1"
    assert publish_context == [True]
    assert _lock_is_free(as_owner)


@pytest.mark.asyncio
@pytest.mark.usefixtures("granted_product_permissions")
async def test_publish_changes_keeps_an_owner_only_page_owner_only(as_owner, publish_calls, publish_context):
    _published(as_owner, mode="restricted", emails=[], org_allowed=False, owner_only=True, access_version=2)

    await aw.republish_artifact("proj", ARTIFACT_ID, ORG_SCOPE, principal=None)

    [call] = publish_calls
    assert call["access"]["mode"] == "restricted"
    assert call["access"].get("org_allowed") is not True


@pytest.mark.asyncio
@pytest.mark.usefixtures("granted_product_permissions")
async def test_publish_changes_on_an_unshared_artifact_is_a_404_and_publishes_nothing(
    as_owner, publish_calls, publish_context,
):
    with pytest.raises(HTTPException) as caught:
        await aw.republish_artifact("proj", ARTIFACT_ID, ORG_SCOPE, principal=None)

    assert caught.value.status_code == 404
    assert publish_calls == []
    assert _lock_is_free(as_owner)


# ── which revision is live ──────────────────────────────────────────────


def _clock(monkeypatch, *stamps: str):
    """Revisions are stamped with these times, in order."""
    times = iter(stamps)
    monkeypatch.setattr(artifact_revisions, "_now", lambda: next(times))


def _capture(folder) -> dict:
    artifact_id, metadata = ensure_full_id(folder)
    return artifact_revisions.current_source(folder, metadata, artifact_id)["revision"]


def _epoch(stamp: str) -> int:
    return int(datetime.fromisoformat(stamp).replace(tzinfo=timezone.utc).timestamp())


def test_a_publish_names_the_revision_it_puts_live(folder, monkeypatch):
    _clock(monkeypatch, "2026-11-01T10:00:00+00:00")
    revision_id = artifact_revisions.published_head_revision_id(folder)

    assert revision_id
    assert revision_id == _capture(folder)["id"]


def test_the_recorded_revision_is_the_live_one_after_later_edits(folder, monkeypatch):
    _clock(monkeypatch, "2026-11-01T10:00:00+00:00", "2026-11-01T11:00:00+00:00")
    first = _capture(folder)
    (folder / "report.html").write_text("<html><body>v2</body></html>")
    second = _capture(folder)
    assert second["id"] != first["id"]

    live = artifact_revisions.published_revision(
        folder, {"report_id": "r1", "published_revision_id": first["id"]}, first["path"]
    )

    assert live["id"] == first["id"]


def test_an_older_publish_is_matched_by_its_time(folder, monkeypatch):
    # Published before ENG-4595 recorded the revision: the newest revision
    # stamped by the publish's content mtime stands in.
    _clock(monkeypatch, "2026-11-01T10:00:00+00:00", "2026-11-01T11:00:00+00:00")
    first = _capture(folder)
    (folder / "report.html").write_text("<html><body>v2</body></html>")
    _capture(folder)

    live = artifact_revisions.published_revision(
        folder, {"report_id": "r1", "published_mtime": _epoch("2026-11-01T10:00:00")}, first["path"]
    )

    assert live["id"] == first["id"]


@pytest.mark.parametrize("entry", [
    None,
    {"report_id": ""},
    {"report_id": "r1", "published": False, "published_mtime": 10**10},
    # Older than every revision: nothing to tell apart.
    {"report_id": "r1", "published_mtime": _epoch("2020-01-01T00:00:00")},
])
def test_no_live_revision_when_nothing_is_published_or_none_matches(folder, monkeypatch, entry):
    _clock(monkeypatch, "2026-11-01T10:00:00+00:00")
    first = _capture(folder)

    assert artifact_revisions.published_revision(folder, entry, first["path"]) is None


def test_the_workspace_load_says_which_revision_is_live(folder, monkeypatch):
    _clock(monkeypatch, "2026-11-01T10:00:00+00:00", "2026-11-01T11:00:00+00:00")
    first = _capture(folder)
    _published(folder, published_revision_id=first["id"])
    (folder / "report.html").write_text("<html><body>v2</body></html>")
    artifact_id, metadata = ensure_full_id(folder)

    state = aw._workspace_state(
        SimpleNamespace(project_id="proj-1"), folder, metadata, artifact_id, None, with_export=False,
    )

    assert state.result["publishedRevision"]["id"] == first["id"]
    assert state.result["revision"]["id"] != first["id"]
