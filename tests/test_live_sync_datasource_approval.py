"""Live sync of an app whose connections need approval stops quietly."""
from __future__ import annotations

import logging

import pytest

from cowork.api.v1.endpoints import artifact_workspace as workspace_ep
from cowork.db.scoped import LOCAL_SCOPE
from cowork.schemas.artifact_access import OWNER_ONLY_ACCESS
from cowork.services import publish


async def _true(*_a, **_k):
    return True


@pytest.mark.asyncio
async def test_live_sync_stops_without_a_traceback_when_connections_need_approval(tmp_path, monkeypatch, caplog):
    folder = tmp_path / "proj" / ".anton" / "artifacts" / "app"
    folder.mkdir(parents=True)
    monkeypatch.setattr(workspace_ep, "_acquire_live_publish_lock", _true)
    monkeypatch.setattr(publish, "published_artifact_access", lambda *_a, **_k: OWNER_ONLY_ACCESS)
    monkeypatch.setattr(publish, "desktop_publish_credential", lambda: ("k", "https://api.mindshub.ai"))

    def refuse(*_a, **_k):
        raise publish.DatasourceApprovalRequired([])

    monkeypatch.setattr(publish, "publish_artifact", refuse)
    with caplog.at_level(logging.WARNING):
        assert await workspace_ep._sync_live_artifact(LOCAL_SCOPE, folder) is False
    assert "new connections need approval" in caplog.text
    assert all(record.exc_info is None for record in caplog.records)
