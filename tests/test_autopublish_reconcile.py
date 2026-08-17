"""End-of-turn publish reconciliation: phases, budget, ordering, guards.

Publishing happens inline after the turn's try/finally, so every guard here is
about not holding a finished turn open longer than a bounded time.
"""
from __future__ import annotations

import json

import pytest

from cowork.db.scoped import LOCAL_SCOPE, TenantScope
from cowork.services import artifact_autopublish as ap
from cowork.services import artifact_locks as locks

ORG_SCOPE = TenantScope(org_mode=True, org_id="org-1", user_id="user-1")


def _make(base, slug, *, files: dict[str, str], meta: dict):
    folder = base / slug
    folder.mkdir(parents=True, exist_ok=True)
    for rel, body in files.items():
        path = folder / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(body)
    (folder / "metadata.json").write_text(json.dumps(meta))
    return folder


@pytest.fixture
def base(tmp_path):
    root = tmp_path / "org-1" / "proj" / ".anton" / "artifacts"
    root.mkdir(parents=True)
    return root


@pytest.fixture
def enabled(monkeypatch):
    monkeypatch.setattr(ap, "_is_enabled", lambda: True)


@pytest.fixture
def publish_url(monkeypatch):
    monkeypatch.setattr(ap, "_publish_url", lambda: "https://api.staging.mindshub.ai")


@pytest.fixture
def key(monkeypatch):
    class FakeKey:
        instance_id = "inst-1"
        revoked = False

        def __init__(self, *a, **kw):
            pass

        async def get(self):
            return "turnkey-1"

        async def revoke(self):
            FakeKey.revoked = True

    FakeKey.revoked = False
    monkeypatch.setattr(ap, "PublishKey", FakeKey)
    return FakeKey


@pytest.fixture
def published(monkeypatch):
    """Record publish calls and write a plausible .published.json."""
    calls = []

    def fake_publish(artifact, *, artifacts_base, api_key, publish_url, password=None, access=None):
        calls.append({"folder": artifact, "api_key": api_key, "access": access})
        (artifact / ".published.json").write_text(json.dumps({
            "index.html": {"report_id": "rid", "url": "u", "published": True,
                           "last_md5": "m", "published_mtime": 9_999_999_999},
        }))
        return {"status": "ok", "url": "u"}

    monkeypatch.setattr(ap, "publish_artifact", fake_publish)
    return calls


pytestmark = pytest.mark.usefixtures("publish_url")


# ── guards ────────────────────────────────────────────────────────────────

async def test_disabled_setting_publishes_nothing(base, key, published, monkeypatch):
    monkeypatch.setattr(ap, "_is_enabled", lambda: False)
    _make(base, "rep", files={"report.html": "<html></html>"},
          meta={"slug": "rep", "type": "html-app"})

    out = await ap.autopublish_project_artifacts(base, ORG_SCOPE, touched={"rep"})

    assert out == set()
    assert published == []


async def test_local_mode_publishes_nothing(base, enabled, key, published):
    _make(base, "rep", files={"report.html": "<html></html>"},
          meta={"slug": "rep", "type": "html-app"})

    out = await ap.autopublish_project_artifacts(base, LOCAL_SCOPE, touched={"rep"})

    assert out == set()
    assert published == []


async def test_missing_scope_publishes_nothing(base, enabled, key, published):
    _make(base, "rep", files={"report.html": "<html></html>"},
          meta={"slug": "rep", "type": "html-app"})

    out = await ap.autopublish_project_artifacts(base, None, touched={"rep"})

    assert out == set()
    assert published == []


async def test_scope_without_user_id_publishes_nothing(base, enabled, key, published):
    partial = TenantScope(org_mode=True, org_id="org-1", user_id=None)
    _make(base, "rep", files={"report.html": "<html></html>"},
          meta={"slug": "rep", "type": "html-app"})

    out = await ap.autopublish_project_artifacts(base, partial, touched={"rep"})

    assert out == set()
    assert published == []


# ── the happy path ────────────────────────────────────────────────────────

async def test_new_artifact_is_published_restricted_to_the_org(base, enabled, key, published):
    _make(base, "rep", files={"report.html": "<html></html>"},
          meta={"slug": "rep", "type": "html-app"})

    out = await ap.autopublish_project_artifacts(base, ORG_SCOPE, touched={"rep"})

    assert out == {"rep"}
    # org_allowed is mandatory: without it resolve_access degrades `restricted`
    # with no emails to `public`, i.e. world-readable.
    assert published[0]["access"] == {"mode": "restricted", "emails": [], "org_allowed": True}
    assert published[0]["api_key"] == "turnkey-1"


async def test_key_is_revoked_after_reconciliation(base, enabled, key, published):
    _make(base, "rep", files={"report.html": "<html></html>"},
          meta={"slug": "rep", "type": "html-app"})

    await ap.autopublish_project_artifacts(base, ORG_SCOPE, touched={"rep"})

    assert key.revoked is True


async def test_nothing_to_publish_mints_no_key(base, enabled, published, monkeypatch):
    minted = []

    class FakeKey:
        instance_id = "i"

        def __init__(self, *a, **kw):
            pass

        async def get(self):
            minted.append(1)
            return "k"

        async def revoke(self):
            pass

    monkeypatch.setattr(ap, "PublishKey", FakeKey)
    _make(base, "data", files={"rows.csv": "a,b"}, meta={"slug": "data", "type": "dataset"})

    out = await ap.autopublish_project_artifacts(base, ORG_SCOPE, touched={"data"})

    assert out == set()
    assert minted == []


async def test_no_key_available_publishes_nothing(base, enabled, published, monkeypatch):
    class NoKey:
        instance_id = "i"

        def __init__(self, *a, **kw):
            pass

        async def get(self):
            return None

        async def revoke(self):
            pass

    monkeypatch.setattr(ap, "PublishKey", NoKey)
    _make(base, "rep", files={"report.html": "<html></html>"},
          meta={"slug": "rep", "type": "html-app"})

    out = await ap.autopublish_project_artifacts(base, ORG_SCOPE, touched={"rep"})

    assert out == set()
    assert published == []


# ── phases, ordering, budget ──────────────────────────────────────────────

async def test_untouched_unpublished_artifact_is_picked_up_by_phase_two(base, enabled, key, published):
    _make(base, "old", files={"a.md": "x"}, meta={"slug": "old", "type": "document"})

    out = await ap.autopublish_project_artifacts(base, ORG_SCOPE, touched=set())

    assert out == {"old"}


async def test_touched_artifact_publishes_before_the_backlog(base, enabled, key, published):
    for i in range(6):
        _make(base, f"old-{i}", files={"a.md": str(i)},
              meta={"slug": f"old-{i}", "type": "document"})
    _make(base, "hot", files={"report.html": "<html></html>"},
          meta={"slug": "hot", "type": "html-app"})

    out = await ap.autopublish_project_artifacts(base, ORG_SCOPE, touched={"hot"}, limit=1)

    assert out == {"hot"}
    assert [c["folder"].name for c in published] == ["hot"]


async def test_limit_caps_the_number_of_publishes(base, enabled, key, published):
    for i in range(4):
        _make(base, f"a-{i}", files={"a.md": str(i)},
              meta={"slug": f"a-{i}", "type": "document"})

    out = await ap.autopublish_project_artifacts(base, ORG_SCOPE, touched=set(), limit=2)

    assert len(out) == 2
    assert len(published) == 2


async def test_static_is_published_before_fullstack(base, enabled, key, published):
    _make(base, "app",
          files={"backend.py": "x", "static/index.html": "<html></html>"},
          meta={"slug": "app", "type": "fullstack-stateless-app"})
    _make(base, "doc", files={"a.md": "x"}, meta={"slug": "doc", "type": "document"})

    await ap.autopublish_project_artifacts(base, ORG_SCOPE, touched={"app", "doc"})

    assert [c["folder"].name for c in published] == ["doc", "app"]


async def test_exhausted_budget_skips_the_rest(base, enabled, key, published):
    for i in range(3):
        _make(base, f"a-{i}", files={"a.md": str(i)},
              meta={"slug": f"a-{i}", "type": "document"})

    out = await ap.autopublish_project_artifacts(
        base, ORG_SCOPE, touched=set(), budget_s=0.0, touched_budget_s=0.0,
    )

    assert out == set()
    assert published == []


async def test_locks_dir_is_not_treated_as_a_candidate(base, enabled, key, published):
    locks.acquire(base, "whatever", ttl_s=600)
    _make(base, "rep", files={"report.html": "<html></html>"},
          meta={"slug": "rep", "type": "html-app"})

    out = await ap.autopublish_project_artifacts(base, ORG_SCOPE, touched={"rep"})

    assert out == {"rep"}


# ── failure handling ──────────────────────────────────────────────────────

async def test_timeout_does_not_fail_the_call_and_keeps_the_lock(base, enabled, key, monkeypatch):
    _make(base, "rep", files={"report.html": "<html></html>"},
          meta={"slug": "rep", "type": "html-app"})

    def slow_publish(artifact, **kwargs):
        import time as _t

        _t.sleep(5)
        return {"status": "ok"}

    monkeypatch.setattr(ap, "publish_artifact", slow_publish)

    out = await ap.autopublish_project_artifacts(
        base, ORG_SCOPE, touched={"rep"}, timeout_s=0.05,
    )

    assert out == set()
    # The abandoned thread is still uploading; releasing now would let a second
    # publisher in.
    assert locks.acquire(base, "rep", ttl_s=600) is False


async def test_lock_is_released_after_a_failed_publish(base, enabled, key, monkeypatch):
    # A synchronous failure leaves no running thread, so holding the lock for its
    # whole TTL would block self-heal and break the "next turn retries" promise.
    _make(base, "rep", files={"report.html": "<html></html>"},
          meta={"slug": "rep", "type": "html-app"})

    def boom(artifact, **kwargs):
        raise RuntimeError("Publishing failed: HTTP Error 502")

    monkeypatch.setattr(ap, "publish_artifact", boom)

    out = await ap.autopublish_project_artifacts(base, ORG_SCOPE, touched={"rep"})

    assert out == set()
    assert locks.acquire(base, "rep", ttl_s=600) is True


async def test_lock_is_released_after_a_successful_publish(base, enabled, key, published):
    # Paired with the timeout test: without this one, an implementation that never
    # releases the lock would pass both.
    _make(base, "rep", files={"report.html": "<html></html>"},
          meta={"slug": "rep", "type": "html-app"})

    await ap.autopublish_project_artifacts(base, ORG_SCOPE, touched={"rep"})

    assert locks.acquire(base, "rep", ttl_s=600) is True


async def test_publish_failure_leaves_no_record_so_next_turn_retries(base, enabled, key, monkeypatch):
    _make(base, "rep", files={"report.html": "<html></html>"},
          meta={"slug": "rep", "type": "html-app"})

    def boom(artifact, **kwargs):
        raise RuntimeError("Publishing failed: HTTP Error 502")

    monkeypatch.setattr(ap, "publish_artifact", boom)

    out = await ap.autopublish_project_artifacts(base, ORG_SCOPE, touched={"rep"})

    assert out == set()
    assert not (base / "rep" / ".published.json").exists()
    assert ap.needs_publish(base / "rep", base).action == "new"


async def test_busy_lock_skips_the_slug(base, enabled, key, published):
    _make(base, "rep", files={"report.html": "<html></html>"},
          meta={"slug": "rep", "type": "html-app"})
    locks.acquire(base, "rep", ttl_s=600)

    out = await ap.autopublish_project_artifacts(base, ORG_SCOPE, touched={"rep"})

    assert out == set()
    assert published == []
