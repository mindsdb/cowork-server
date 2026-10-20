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


# -- routes -------------------------------------------------------------------


@pytest.fixture
def as_owner(monkeypatch, folder):
    monkeypatch.setattr(
        aw, "_owner_workspace",
        lambda session, project_ref, artifact_id: (
            SimpleNamespace(project_id="proj-1"), folder, {"slug": "rep"}, {},
        ),
    )
    return folder


@pytest.fixture
def keys(monkeypatch):
    monkeypatch.setattr(
        "cowork.services.artifact_autopublish._publish_url", lambda scope: "https://view.example",
    )
    monkeypatch.setattr("cowork.services.artifact_autopublish._active_workspace_id", lambda scope: None)
    minted = []

    class FakeKey:
        value = "turnkey-1"

        def __init__(self, *args, **kwargs):
            self.revoked = False
            minted.append(self)

        async def get(self):
            return type(self).value

        async def revoke(self):
            # A real revoke awaits auth; the checkpoint lets a cancelled scope interrupt it.
            await asyncio.sleep(0)
            self.revoked = True

    monkeypatch.setattr("cowork.services.artifact_publish_key.PublishKey", FakeKey)
    return SimpleNamespace(minted=minted, cls=FakeKey)


@pytest.fixture
def stats_calls(monkeypatch):
    calls = []

    def fake_fetch(*, publish_url, key, report_id, client=None):
        calls.append(("fetch", publish_url, key, report_id))
        return dict(STATS_BODY)

    def fake_set(*, publish_url, key, report_id, enabled, client=None):
        calls.append(("set", publish_url, key, report_id, enabled))
        return enabled

    monkeypatch.setattr(artifact_stats, "fetch_stats", fake_fetch)
    monkeypatch.setattr(artifact_stats, "set_tracking", fake_set)
    return calls


def _upstream(monkeypatch, handler):
    monkeypatch.setattr(
        artifact_stats, "_new_client", lambda: httpx.Client(transport=httpx.MockTransport(handler)),
    )


async def _read():
    return await aw.read_artifact_stats("proj", ARTIFACT_ID, _Session())


async def _toggle(enabled: bool = False):
    return await aw.set_artifact_stats_tracking(
        "proj", ARTIFACT_ID, aw._StatsTrackingBody(enabled=enabled), _Session(),
    )


def _lock_is_free(folder) -> bool:
    if not artifact_locks.acquire(folder.parent, folder.name, ttl_s=60):
        return False
    artifact_locks.release(folder.parent, folder.name)
    return True


async def test_owner_reads_stats_with_a_key_minted_for_the_request(as_owner, keys, stats_calls):
    out = await _read()

    assert out == STATS_BODY
    assert stats_calls == [("fetch", "https://view.example", "turnkey-1", "rid-1")]
    assert len(keys.minted) == 1
    assert keys.minted[0].revoked is True


async def test_owner_toggles_tracking_under_the_publish_lock(as_owner, keys, stats_calls):
    out = await _toggle(enabled=False)

    assert out == {"tracking": False}
    assert stats_calls == [("set", "https://view.example", "turnkey-1", "rid-1", False)]
    assert keys.minted[0].revoked is True
    assert _lock_is_free(as_owner)


@pytest.mark.parametrize("call", [_read, _toggle])
async def test_a_non_owner_cannot_read_or_toggle_stats(monkeypatch, keys, stats_calls, call):
    def refuse(*_args, **_kwargs):
        raise HTTPException(status_code=403, detail="Not the owner")

    monkeypatch.setattr(aw, "_owner_workspace", refuse)

    with pytest.raises(HTTPException) as excinfo:
        await call()

    assert excinfo.value.status_code == 403
    assert stats_calls == []
    assert keys.minted == []


@pytest.mark.parametrize("call", [_read, _toggle])
async def test_missing_artifact_manage_is_refused_before_resolving_the_artifact(
    monkeypatch, keys, stats_calls, call,
):
    async def deny(scope, permission):
        assert permission == "artifact.manage"
        raise ProductPermissionDenied()

    def must_not_resolve(*_args, **_kwargs):
        raise AssertionError("artifact resolved before the permission check")

    monkeypatch.setattr(aw, "require_product_permission", deny)
    monkeypatch.setattr(aw, "_owner_workspace", must_not_resolve)

    with pytest.raises(ProductPermissionDenied) as excinfo:
        await call()

    assert excinfo.value.status_code == 403
    assert stats_calls == []
    assert keys.minted == []


@pytest.mark.parametrize("call", [_read, _toggle])
async def test_no_publish_key_is_503(as_owner, keys, stats_calls, call):
    keys.cls.value = None

    with pytest.raises(HTTPException) as excinfo:
        await call()

    assert excinfo.value.status_code == 503
    assert excinfo.value.detail == "Publishing is unavailable right now. Try again in a moment."
    assert stats_calls == []
    assert _lock_is_free(as_owner)


async def test_key_is_revoked_when_the_stats_service_fails(as_owner, keys, monkeypatch):
    _upstream(monkeypatch, lambda request: httpx.Response(503, json={"error": "DynamoDB unavailable"}))

    with pytest.raises(HTTPException) as excinfo:
        await _read()

    assert excinfo.value.status_code == 503
    assert excinfo.value.detail == {
        "code": "stats_temporarily_unavailable",
        "message": "Stats are temporarily unavailable.",
    }
    assert keys.minted[0].revoked is True


@pytest.mark.parametrize("call", [_read, _toggle])
async def test_unpublished_artifact_is_404_without_minting_a_key(as_owner, keys, stats_calls, call):
    (as_owner / ".published.json").unlink()

    with pytest.raises(HTTPException) as excinfo:
        await call()

    assert excinfo.value.status_code == 404
    assert excinfo.value.detail == "Artifact is not published"
    assert keys.minted == []
    assert _lock_is_free(as_owner)


async def test_busy_publish_lock_answers_503_quickly(as_owner, keys, stats_calls, monkeypatch):
    monkeypatch.setattr(aw, "_STATS_TRACKING_LOCK_TIMEOUT_S", 0.1)
    assert artifact_locks.acquire(as_owner.parent, as_owner.name, ttl_s=60) is True
    try:
        with pytest.raises(HTTPException) as excinfo:
            await _toggle()
    finally:
        artifact_locks.release(as_owner.parent, as_owner.name)

    assert excinfo.value.status_code == 503
    assert excinfo.value.detail == "Publishing is busy right now. Try again in a moment."
    assert stats_calls == []
    assert keys.minted == []


async def test_tracking_toggle_passes_the_short_lock_timeout(as_owner, keys, stats_calls, monkeypatch):
    seen = []

    async def spy(folder, *, timeout_s=None):
        seen.append(timeout_s)
        return True

    monkeypatch.setattr(aw, "_acquire_live_publish_lock", spy)

    await _toggle(enabled=True)

    assert seen == [5.0]


async def test_anyio_cancellation_still_revokes_the_key_and_releases_the_lock(
    as_owner, keys, monkeypatch,
):
    started = []

    def slow_set(*, publish_url, key, report_id, enabled, client=None):
        started.append(report_id)
        time.sleep(0.5)
        return enabled

    monkeypatch.setattr(artifact_stats, "set_tracking", slow_set)

    with anyio.move_on_after(0.1) as scope:
        await _toggle(enabled=True)

    # The worker thread is waited for, so the handler may still return; what
    # matters is that cleanup ran while the scope was already cancelled.
    assert scope.cancel_called
    assert started == ["rid-1"]
    assert keys.minted[0].revoked is True
    assert _lock_is_free(as_owner)


async def test_reading_stats_takes_no_publish_lock(as_owner, keys, stats_calls, monkeypatch):
    async def no_lock(*_args, **_kwargs):
        raise AssertionError("GET must not wait for the publish lock")

    monkeypatch.setattr(aw, "_acquire_live_publish_lock", no_lock)

    assert await _read() == STATS_BODY


@pytest.mark.parametrize("call", [_read, _toggle])
async def test_upstream_401_is_502_without_the_desktop_code(as_owner, keys, monkeypatch, caplog, call):
    _upstream(monkeypatch, lambda request: httpx.Response(401, json={"error": "Invalid token"}))

    with caplog.at_level(logging.WARNING, logger=aw.logger.name):
        with pytest.raises(HTTPException) as excinfo:
            await call()

    assert excinfo.value.status_code == 502
    assert excinfo.value.detail == "The publishing service rejected the stats credential."
    assert any("artifact_stats_upstream_unauthorized" in r.getMessage() for r in caplog.records)
    assert keys.minted[0].revoked is True


async def test_report_owned_by_another_account_is_409(as_owner, keys, monkeypatch):
    _upstream(monkeypatch, lambda request: httpx.Response(404, json={"code": "artifact_not_found"}))

    with pytest.raises(HTTPException) as excinfo:
        await _read()

    assert excinfo.value.status_code == 409
    assert excinfo.value.detail == {
        "code": "stats_owned_by_other_account",
        "message": "Stats are not available for this account.",
    }
