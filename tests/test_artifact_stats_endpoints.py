"""Owner-only stats routes for org artifacts.

Handlers are called directly, as in tests/test_artifact_access_endpoints.py;
the permission-before-resolution boundary over HTTP lives in
tests/test_product_permissions.py.
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from types import SimpleNamespace

import anyio
import httpx
import pytest
from fastapi import HTTPException

from cowork.api.v1.endpoints import artifact_workspace as aw
from cowork.db.scoped import TenantScope
from cowork.services import artifact_locks, artifact_stats
from cowork.services.product_permissions import ProductPermissionDenied

ORG_SCOPE = TenantScope(org_mode=True, org_id="org-1", user_id="user-1")
ARTIFACT_ID = "11111111-1111-1111-1111-111111111111"
STATS_BODY = {
    "tracking": True,
    "period_days": 30,
    "total_views": 3,
    "last_viewed": "2026-10-19T10:22:31Z",
    "views_in_period": 3,
    "daily": [{"date": "2026-10-19", "views": 3}],
    "unique_viewers": 1,
    "viewers": [],
}


class _Session:
    scope = ORG_SCOPE


@pytest.fixture
def folder(tmp_path):
    base = tmp_path / "org-1" / "proj" / ".anton" / "artifacts"
    f = base / "rep"
    f.mkdir(parents=True)
    (f / "report.html").write_text("<html></html>")
    (f / "metadata.json").write_text(json.dumps({"slug": "rep", "type": "html-app"}))
    (f / ".published.json").write_text(json.dumps({
        "report.html": {"report_id": "rid-1", "url": "https://view.example/r/rid-1", "published": True},
    }))
    return f


pytestmark = [pytest.mark.asyncio, pytest.mark.usefixtures("granted_product_permissions")]


# -- publish lock timeout -----------------------------------------------------


async def test_lock_helper_gives_up_after_a_short_timeout(folder):
    assert artifact_locks.acquire(folder.parent, folder.name, ttl_s=60) is True
    loop = asyncio.get_running_loop()
    started = loop.time()
    try:
        assert await aw._acquire_live_publish_lock(folder, timeout_s=0.2) is False
    finally:
        artifact_locks.release(folder.parent, folder.name)
    assert loop.time() - started < 2.0


async def test_lock_helper_default_follows_the_publish_timeout(folder, monkeypatch):
    monkeypatch.setattr(aw, "_LIVE_PUBLISH_TIMEOUT_S", 0.05)
    assert artifact_locks.acquire(folder.parent, folder.name, ttl_s=60) is True
    loop = asyncio.get_running_loop()
    started = loop.time()
    try:
        assert await aw._acquire_live_publish_lock(folder) is False
    finally:
        artifact_locks.release(folder.parent, folder.name)
    assert loop.time() - started < 2.0


async def test_lock_helper_takes_a_free_lock(folder):
    assert await aw._acquire_live_publish_lock(folder, timeout_s=0.2) is True
    assert artifact_locks.acquire(folder.parent, folder.name, ttl_s=60) is False
    artifact_locks.release(folder.parent, folder.name)


async def test_stats_toggle_waits_five_seconds_at_most():
    assert aw._STATS_TRACKING_LOCK_TIMEOUT_S == 5.0
