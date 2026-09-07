"""Live artifact source edits keep their stable browser URL current (ENG-2380)."""
from __future__ import annotations

import json
import threading

import pytest

from cowork.api.v1.endpoints import artifact_workspace as workspace
from cowork.db.scoped import LOCAL_SCOPE, TenantScope
from cowork.services import artifact_locks
from cowork.services.artifact_revisions import current_source


ARTIFACT_ID = "11111111111111111111111111111111"
ORG_SCOPE = TenantScope(org_mode=True, org_id="org-1", user_id="user-1")


class _Session:
    def __init__(self, scope=ORG_SCOPE):
        self.scope = scope


@pytest.fixture
def artifact(tmp_path):
    folder = tmp_path / "artifacts" / "report"
    folder.mkdir(parents=True)
    (folder / "report.html").write_text("<html>before</html>", encoding="utf-8")
    metadata = {"slug": "report", "type": "html-app", "primary": "report.html"}
    (folder / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    return folder, metadata


def _publish(folder, access):
    (folder / ".published.json").write_text(
        json.dumps({
            "report.html": {
                "report_id": "report-1",
                "url": "https://view.example/report-1",
                "published": True,
                **access,
            }
        }),
        encoding="utf-8",
    )


@pytest.mark.asyncio
async def test_manual_save_synchronizes_an_existing_live_artifact(artifact, monkeypatch):
    folder, metadata = artifact
    source = current_source(folder, metadata, ARTIFACT_ID)
    synced = []

    monkeypatch.setattr(
        workspace,
        "_owner_workspace",
        lambda *_args: (object(), folder, metadata, {}),
    )

    async def fake_sync(session, live_folder):
        synced.append((session, live_folder))
        return True

    monkeypatch.setattr(workspace, "_sync_live_artifact", fake_sync)
    session = _Session()

    saved = await workspace.update_artifact_source(
        "project-1",
        ARTIFACT_ID,
        workspace._SourceUpdateBody(
            content="<html>after</html>",
            expectedRevisionId=source["revision"]["id"],
            path="report.html",
        ),
        session,
    )

    assert saved["content"] == "<html>after</html>"
    assert synced == [(session, folder)]


@pytest.mark.asyncio
async def test_noop_save_does_not_create_a_live_version(artifact, monkeypatch):
    folder, metadata = artifact
    source = current_source(folder, metadata, ARTIFACT_ID)
    monkeypatch.setattr(
        workspace,
        "_owner_workspace",
        lambda *_args: (object(), folder, metadata, {}),
    )

    async def unexpected_sync(*_args):
        raise AssertionError("unchanged source must not be re-published")

    monkeypatch.setattr(workspace, "_sync_live_artifact", unexpected_sync)

    saved = await workspace.update_artifact_source(
        "project-1",
        ARTIFACT_ID,
        workspace._SourceUpdateBody(
            content=source["content"],
            expectedRevisionId=source["revision"]["id"],
            path="report.html",
        ),
        _Session(),
    )

    assert saved["revision"]["id"] == source["revision"]["id"]


@pytest.mark.asyncio
async def test_restore_synchronizes_an_existing_live_artifact(artifact, monkeypatch):
    folder, metadata = artifact
    initial = current_source(folder, metadata, ARTIFACT_ID)
    edited = workspace.save_source(
        folder,
        metadata,
        ARTIFACT_ID,
        content="<html>after</html>",
        expected_revision_id=initial["revision"]["id"],
        rel_path="report.html",
    )
    synced = []
    monkeypatch.setattr(
        workspace,
        "_owner_workspace",
        lambda *_args: (object(), folder, metadata, {}),
    )

    async def fake_sync(session, live_folder):
        synced.append((session, live_folder))
        return True

    monkeypatch.setattr(workspace, "_sync_live_artifact", fake_sync)
    session = _Session()

    restored = await workspace.restore_artifact_revision(
        "project-1",
        ARTIFACT_ID,
        initial["revision"]["id"],
        workspace._RestoreBody(expectedRevisionId=edited["revision"]["id"]),
        session,
    )

    assert restored["content"] == "<html>before</html>"
    assert synced == [(session, folder)]


@pytest.mark.asyncio
async def test_org_sync_preserves_audience_and_reuses_live_publish(artifact, monkeypatch):
    folder, _metadata = artifact
    _publish(folder, {
        "mode": "restricted",
        "emails": ["viewer@example.com"],
        "org_allowed": False,
        "owner_only": False,
    })
    calls = []
    revoked = []

    class _Key:
        async def get(self):
            return "turn-key"

        async def revoke(self):
            revoked.append(True)

    monkeypatch.setattr(
        workspace,
        "_owner_publish_context",
        lambda _session, _folder: (folder.parent, "https://api.staging.example", _Key()),
    )

    def fake_publish(artifact_path, **kwargs):
        calls.append((artifact_path, kwargs))
        return {"status": "ok", "url": "https://view.example/report-1"}

    monkeypatch.setattr("cowork.services.publish.publish_artifact", fake_publish)

    result = await workspace._sync_live_artifact(_Session(), folder)

    assert result is True
    assert calls[0][0] == folder
    assert calls[0][1]["artifacts_base"] == folder.parent
    assert calls[0][1]["api_key"] == "turn-key"
    assert calls[0][1]["publish_url"] == "https://api.staging.example"
    assert calls[0][1]["access"] == {
        "mode": "restricted",
        "emails": ["viewer@example.com"],
        "org_allowed": False,
        "owner_only": False,
    }
    assert calls[0][1]["scope"] is ORG_SCOPE
    assert revoked == [True]


@pytest.mark.asyncio
async def test_unpublished_draft_does_not_resolve_publish_credentials(artifact, monkeypatch):
    folder, _metadata = artifact

    def unexpected_context(*_args):
        raise AssertionError("draft saves must not mint publish credentials")

    monkeypatch.setattr(workspace, "_owner_publish_context", unexpected_context)

    assert await workspace._sync_live_artifact(_Session(), folder) is None


@pytest.mark.asyncio
async def test_desktop_save_uses_the_configured_publish_environment(artifact, monkeypatch):
    folder, _metadata = artifact
    _publish(folder, {"mode": "public"})
    calls = []

    monkeypatch.setattr(
        "cowork.services.publish.desktop_publish_credential",
        lambda: ("desktop-key", "https://api.desktop.example"),
    )

    def fake_publish(artifact_path, **kwargs):
        calls.append((artifact_path, kwargs))
        return {"status": "ok"}

    monkeypatch.setattr("cowork.services.publish.publish_artifact", fake_publish)

    result = await workspace._sync_live_artifact(_Session(LOCAL_SCOPE), folder)

    assert result is True
    assert calls[0][1]["api_key"] == "desktop-key"
    assert calls[0][1]["publish_url"] == "https://api.desktop.example"
    assert calls[0][1]["access"] == {"mode": "public"}
    assert calls[0][1]["scope"] is LOCAL_SCOPE


@pytest.mark.asyncio
async def test_timed_out_publish_keeps_its_lock_and_key(artifact, monkeypatch):
    folder, _metadata = artifact
    _publish(folder, {"mode": "public"})
    publish_started = threading.Event()
    finish_publish = threading.Event()
    revoked = []

    class _Key:
        async def get(self):
            return "turn-key"

        async def revoke(self):
            revoked.append(True)

    monkeypatch.setattr(workspace, "_LIVE_PUBLISH_TIMEOUT_S", 0.01)
    monkeypatch.setattr(
        workspace,
        "_owner_publish_context",
        lambda _session, _folder: (folder.parent, "https://api.staging.example", _Key()),
    )

    def slow_publish(*_args, **_kwargs):
        publish_started.set()
        finish_publish.wait(timeout=1)
        return {"status": "ok"}

    monkeypatch.setattr("cowork.services.publish.publish_artifact", slow_publish)

    try:
        assert await workspace._sync_live_artifact(_Session(), folder) is False
        assert publish_started.is_set()
        assert revoked == []
        assert artifact_locks.acquire(folder.parent, folder.name, ttl_s=60) is False
    finally:
        finish_publish.set()


@pytest.mark.asyncio
async def test_publish_failure_does_not_undo_the_source_save(artifact, monkeypatch):
    folder, metadata = artifact
    source = current_source(folder, metadata, ARTIFACT_ID)
    monkeypatch.setattr(
        workspace,
        "_owner_workspace",
        lambda *_args: (object(), folder, metadata, {}),
    )

    async def failed_sync(*_args):
        return False

    monkeypatch.setattr(workspace, "_sync_live_artifact", failed_sync)

    saved = await workspace.update_artifact_source(
        "project-1",
        ARTIFACT_ID,
        workspace._SourceUpdateBody(
            content="<html>saved locally</html>",
            expectedRevisionId=source["revision"]["id"],
            path="report.html",
        ),
        _Session(),
    )

    assert saved["content"] == "<html>saved locally</html>"
    assert (folder / "report.html").read_text(encoding="utf-8") == "<html>saved locally</html>"
