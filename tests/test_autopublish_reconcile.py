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
PROJECT_ID = "project-1"


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
    monkeypatch.setattr(ap, "_is_enabled", lambda scope: True)


@pytest.fixture
def publish_url(monkeypatch):
    monkeypatch.setattr(ap, "_publish_url", lambda scope: "https://api.staging.mindshub.ai")


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

    def fake_publish(artifact, *, artifacts_base, api_key, publish_url, password=None,
                     access=None, scope=None, project_id=None):
        calls.append({"folder": artifact, "api_key": api_key, "access": access, "scope": scope,
                      "project_id": project_id})
        (artifact / ".published.json").write_text(json.dumps({
            "index.html": {"report_id": "rid", "url": "u", "published": True,
                           "last_md5": "m", "published_mtime": 9_999_999_999},
        }))
        return {"status": "ok", "url": "u"}

    monkeypatch.setattr(ap, "publish_artifact", fake_publish)
    return calls


@pytest.fixture(autouse=True)
def owned_slugs(monkeypatch):
    """Keep testing reconciliation logic in isolation from the owner filter.

    `_owned_slugs` opens its own DB session and resolves real ownership rows;
    this module's fixtures use a non-UUID project id ("project-1") and write no
    DB rows, so the default here is "everything is owned" and individual tests
    override it to exercise the filter itself.
    """
    monkeypatch.setattr(ap, "_owned_slugs", lambda base, scope, project_id, slugs: (list(slugs), 0, 0))


pytestmark = pytest.mark.usefixtures("publish_url")


# ── guards ────────────────────────────────────────────────────────────────

async def test_disabled_setting_publishes_nothing(base, key, published, monkeypatch):
    monkeypatch.setattr(ap, "_is_enabled", lambda scope: False)
    _make(base, "rep", files={"report.html": "<html></html>"},
          meta={"slug": "rep", "type": "html-app"})

    out = await ap.autopublish_project_artifacts(base, ORG_SCOPE, project_id=PROJECT_ID, touched={"rep"})

    assert out == set()
    assert published == []


async def test_local_mode_publishes_nothing(base, enabled, key, published):
    _make(base, "rep", files={"report.html": "<html></html>"},
          meta={"slug": "rep", "type": "html-app"})

    out = await ap.autopublish_project_artifacts(base, LOCAL_SCOPE, project_id=PROJECT_ID, touched={"rep"})

    assert out == set()
    assert published == []


async def test_missing_scope_publishes_nothing(base, enabled, key, published):
    _make(base, "rep", files={"report.html": "<html></html>"},
          meta={"slug": "rep", "type": "html-app"})

    out = await ap.autopublish_project_artifacts(base, None, project_id=PROJECT_ID, touched={"rep"})

    assert out == set()
    assert published == []


async def test_scope_without_user_id_publishes_nothing(base, enabled, key, published):
    partial = TenantScope(org_mode=True, org_id="org-1", user_id=None)
    _make(base, "rep", files={"report.html": "<html></html>"},
          meta={"slug": "rep", "type": "html-app"})

    out = await ap.autopublish_project_artifacts(base, partial, project_id=PROJECT_ID, touched={"rep"})

    assert out == set()
    assert published == []


async def test_settings_are_read_with_the_passed_scope_not_the_ambient_one(
    base, key, published, monkeypatch,
):
    """Both settings reads must pass `scope` explicitly.

    The org producer that drives this (`_produce_remote`) is a detached task with
    no `use_settings_scope` binding, and `get_user_settings()` with no argument
    falls back to LOCAL_SCOPE rather than failing. The enable flag is org-scoped
    and the publish URL comes from the org's provider, so an ambient read would
    silently consult the global row and the wrong endpoint — publishing nothing,
    or publishing to the wrong place, with no error anywhere.
    """
    from cowork.common.settings import user_settings as us

    seen = []

    class _Settings:
        artifact_autopublish_enabled = True

    def fake_get_user_settings(scope=None):
        seen.append(scope)
        return _Settings()

    monkeypatch.setattr(us, "get_user_settings", fake_get_user_settings)
    monkeypatch.setattr(ap, "_publish_url", lambda scope: "https://api.staging.mindshub.ai")
    _make(base, "rep", files={"report.html": "<html></html>"},
          meta={"slug": "rep", "type": "html-app"})

    await ap.autopublish_project_artifacts(base, ORG_SCOPE, project_id=PROJECT_ID, touched={"rep"})

    assert seen and all(s is ORG_SCOPE for s in seen)


# ── the happy path ────────────────────────────────────────────────────────

async def test_new_artifact_is_published_owner_only(base, enabled, key, published):
    """A first autopublish is private to its owner (ENG-2316).

    `owner_only` is mandatory, not decoration: without it resolve_access
    degrades `restricted` with no emails to `public`, i.e. world-readable.
    """
    _make(base, "rep", files={"report.html": "<html></html>"},
          meta={"slug": "rep", "type": "html-app"})

    out = await ap.autopublish_project_artifacts(base, ORG_SCOPE, project_id=PROJECT_ID, touched={"rep"})

    assert out == {"rep"}
    assert published[0]["access"] == {"mode": "restricted", "emails": [], "owner_only": True}
    assert published[0]["api_key"] == "turnkey-1"


async def test_republish_keeps_the_access_the_owner_chose(base, enabled, key, published):
    """Re-applying the first-publish default here would silently revoke a share
    every time the agent touched the artifact — which would make the Share
    control untrustworthy rather than merely incomplete (ENG-2316)."""
    folder = _make(base, "rep", files={"index.html": "<html>v2</html>"},
                   meta={"slug": "rep", "type": "html-app"})
    (folder / ".published.json").write_text(json.dumps({
        "index.html": {
            "report_id": "rid", "url": "u", "published": True,
            "last_md5": "stale-so-the-content-counts-as-changed",
            "published_mtime": 0,
            "mode": "restricted", "emails": ["someone@example.com"],
            "org_allowed": False, "owner_only": False,
        },
    }))

    out = await ap.autopublish_project_artifacts(base, ORG_SCOPE, project_id=PROJECT_ID, touched={"rep"})

    assert out == {"rep"}
    assert published[0]["access"] == {
        "mode": "restricted", "emails": ["someone@example.com"],
        "org_allowed": False, "owner_only": False,
    }


async def test_republish_of_an_owner_only_artifact_stays_owner_only(base, enabled, key, published):
    """The degradation trap in reverse: reconstructing an owner-only entry
    without its `owner_only` key yields an empty selection, which resolve_access
    turns into `public` — silently un-privating the artifact."""
    folder = _make(base, "rep", files={"index.html": "<html>v2</html>"},
                   meta={"slug": "rep", "type": "html-app"})
    (folder / ".published.json").write_text(json.dumps({
        "index.html": {
            "report_id": "rid", "url": "u", "published": True,
            "last_md5": "stale", "published_mtime": 0,
            "mode": "restricted", "emails": [], "org_allowed": False, "owner_only": True,
        },
    }))

    await ap.autopublish_project_artifacts(base, ORG_SCOPE, project_id=PROJECT_ID, touched={"rep"})

    assert published[0]["access"]["owner_only"] is True
    assert published[0]["access"]["mode"] == "restricted"


async def test_scope_is_threaded_into_the_publisher(base, enabled, key, published):
    """The publisher resolves datasource secrets through `vault_for_scope`, which
    RAISES on an org deployment when the scope is missing rather than falling
    back to the shared namespace root. Dropping this kwarg would therefore fail
    every org publish, not silently read the wrong vault — but the reconciler
    swallows publish exceptions by design, so the failure would show up only as
    artifacts that never appear. Assert the wiring here instead."""
    _make(base, "rep", files={"report.html": "<html></html>"},
          meta={"slug": "rep", "type": "html-app"})

    await ap.autopublish_project_artifacts(base, ORG_SCOPE, project_id=PROJECT_ID, touched={"rep"})

    assert published[0]["scope"] is ORG_SCOPE


async def test_publish_key_carries_the_active_hub_workspace(base, enabled, published, monkeypatch):
    """The caller's picked MindsHub workspace (UserSettings.hub_workspace_id,
    written by the hub_workspaces selector) must reach the publish key mint,
    the same as it reaches the execution turn key."""
    captured = {}

    class FakeKey:
        instance_id = "inst-1"

        def __init__(self, *a, **kw):
            captured.update(kw)

        async def get(self):
            return "turnkey-1"

        async def revoke(self):
            pass

    monkeypatch.setattr(ap, "PublishKey", FakeKey)
    monkeypatch.setattr(ap, "_active_workspace_id", lambda scope: "ws-1")
    _make(base, "rep", files={"report.html": "<html></html>"},
          meta={"slug": "rep", "type": "html-app"})

    await ap.autopublish_project_artifacts(base, ORG_SCOPE, project_id=PROJECT_ID, touched={"rep"})

    assert captured["workspace_id"] == "ws-1"


async def test_key_is_revoked_after_reconciliation(base, enabled, key, published):
    _make(base, "rep", files={"report.html": "<html></html>"},
          meta={"slug": "rep", "type": "html-app"})

    await ap.autopublish_project_artifacts(base, ORG_SCOPE, project_id=PROJECT_ID, touched={"rep"})

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

    out = await ap.autopublish_project_artifacts(base, ORG_SCOPE, project_id=PROJECT_ID, touched={"data"})

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

    out = await ap.autopublish_project_artifacts(base, ORG_SCOPE, project_id=PROJECT_ID, touched={"rep"})

    assert out == set()
    assert published == []


@pytest.mark.parametrize("status", [403, 503])
async def test_authority_failure_skips_autopublish_and_releases_lock(
    base, enabled, published, monkeypatch, status,
):
    from cowork.services.product_permissions import ProductPermissionDenied, ProductPermissionUnavailable

    async def reject(**kwargs):
        raise (ProductPermissionDenied if status == 403 else ProductPermissionUnavailable)()

    monkeypatch.setattr("cowork.services.artifact_publish_key.mint_turn_key", reject)
    _make(base, "rep", files={"report.html": "<html></html>"},
          meta={"slug": "rep", "type": "html-app"})
    assert await ap.autopublish_project_artifacts(base, ORG_SCOPE, project_id=PROJECT_ID, touched={"rep"}) == set()
    assert published == []
    assert locks.acquire(base, "rep", ttl_s=60)
    locks.release(base, "rep")


# ── phases, ordering, budget ──────────────────────────────────────────────

async def test_untouched_unpublished_artifact_is_picked_up_by_phase_two(base, enabled, key, published):
    _make(base, "old", files={"a.md": "x"}, meta={"slug": "old", "type": "document"})

    out = await ap.autopublish_project_artifacts(base, ORG_SCOPE, project_id=PROJECT_ID, touched=set())

    assert out == {"old"}


async def test_touched_artifact_publishes_before_the_backlog(base, enabled, key, published):
    for i in range(6):
        _make(base, f"old-{i}", files={"a.md": str(i)},
              meta={"slug": f"old-{i}", "type": "document"})
    _make(base, "hot", files={"report.html": "<html></html>"},
          meta={"slug": "hot", "type": "html-app"})

    out = await ap.autopublish_project_artifacts(base, ORG_SCOPE, project_id=PROJECT_ID, touched={"hot"}, limit=1)

    assert out == {"hot"}
    assert [c["folder"].name for c in published] == ["hot"]


async def test_limit_caps_the_number_of_publishes(base, enabled, key, published):
    for i in range(4):
        _make(base, f"a-{i}", files={"a.md": str(i)},
              meta={"slug": f"a-{i}", "type": "document"})

    out = await ap.autopublish_project_artifacts(base, ORG_SCOPE, project_id=PROJECT_ID, touched=set(), limit=2)

    assert len(out) == 2
    assert len(published) == 2


async def test_static_is_published_before_fullstack(base, enabled, key, published):
    _make(base, "app",
          files={"backend.py": "x", "static/index.html": "<html></html>"},
          meta={"slug": "app", "type": "fullstack-stateless-app"})
    _make(base, "doc", files={"a.md": "x"}, meta={"slug": "doc", "type": "document"})

    await ap.autopublish_project_artifacts(base, ORG_SCOPE, project_id=PROJECT_ID, touched={"app", "doc"})

    assert [c["folder"].name for c in published] == ["doc", "app"]


async def test_exhausted_budget_skips_the_rest(base, enabled, key, published):
    for i in range(3):
        _make(base, f"a-{i}", files={"a.md": str(i)},
              meta={"slug": f"a-{i}", "type": "document"})

    out = await ap.autopublish_project_artifacts(
        base, ORG_SCOPE, project_id=PROJECT_ID, touched=set(), budget_s=0.0, touched_budget_s=0.0,
    )

    assert out == set()
    assert published == []


async def test_locks_dir_is_not_treated_as_a_candidate(base, enabled, key, published):
    locks.acquire(base, "whatever", ttl_s=600)
    _make(base, "rep", files={"report.html": "<html></html>"},
          meta={"slug": "rep", "type": "html-app"})

    out = await ap.autopublish_project_artifacts(base, ORG_SCOPE, project_id=PROJECT_ID, touched={"rep"})

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
        base, ORG_SCOPE, project_id=PROJECT_ID, touched={"rep"}, timeout_s=0.05,
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

    out = await ap.autopublish_project_artifacts(base, ORG_SCOPE, project_id=PROJECT_ID, touched={"rep"})

    assert out == set()
    assert locks.acquire(base, "rep", ttl_s=600) is True


async def test_lock_is_released_after_a_successful_publish(base, enabled, key, published):
    # Paired with the timeout test: without this one, an implementation that never
    # releases the lock would pass both.
    _make(base, "rep", files={"report.html": "<html></html>"},
          meta={"slug": "rep", "type": "html-app"})

    await ap.autopublish_project_artifacts(base, ORG_SCOPE, project_id=PROJECT_ID, touched={"rep"})

    assert locks.acquire(base, "rep", ttl_s=600) is True


async def test_publish_failure_leaves_no_record_so_next_turn_retries(base, enabled, key, monkeypatch):
    _make(base, "rep", files={"report.html": "<html></html>"},
          meta={"slug": "rep", "type": "html-app"})

    def boom(artifact, **kwargs):
        raise RuntimeError("Publishing failed: HTTP Error 502")

    monkeypatch.setattr(ap, "publish_artifact", boom)

    out = await ap.autopublish_project_artifacts(base, ORG_SCOPE, project_id=PROJECT_ID, touched={"rep"})

    assert out == set()
    assert not (base / "rep" / ".published.json").exists()
    assert ap.needs_publish(base / "rep", base).action == "new"


async def test_busy_lock_skips_the_slug(base, enabled, key, published):
    _make(base, "rep", files={"report.html": "<html></html>"},
          meta={"slug": "rep", "type": "html-app"})
    locks.acquire(base, "rep", ttl_s=600)

    out = await ap.autopublish_project_artifacts(base, ORG_SCOPE, project_id=PROJECT_ID, touched={"rep"})

    assert out == set()
    assert published == []


# ─── the metric has to survive the deployment's log level ──────────────────

async def test_metric_is_emitted_above_the_deployment_log_level(
    base, enabled, key, published, caplog,
):
    """Every deployment running this feature runs at LOG_LEVEL=WARNING
    (deployment/cowork-server/values-{staging,prod}.yaml). The reconciler
    swallows publish failures by design, so if the metric is filtered out too,
    an artifact that failed to publish is indistinguishable from one that was
    never eligible — the exact hole that made a live staging diagnosis
    impossible. Asserting the level, not just the text, is what keeps a later
    "info is the right level for a success line" cleanup from reopening it."""
    import logging

    _make(base, "rep", files={"report.html": "<html></html>"},
          meta={"slug": "rep", "type": "html-app"})

    with caplog.at_level(logging.WARNING, logger="cowork.services.artifact_autopublish"):
        await ap.autopublish_project_artifacts(base, ORG_SCOPE, project_id=PROJECT_ID, touched={"rep"})

    lines = [r for r in caplog.records if r.message.startswith("artifact_autopublish")]
    assert lines, "the metric must be visible at WARNING"
    assert any("result=published" in r.getMessage() for r in lines)


async def test_missing_project_id_skips_the_reconcile_in_org_mode(base, enabled, key, published, caplog):
    _make(base, "rep", files={"report.html": "<html></html>"},
          meta={"slug": "rep", "type": "html-app"})

    out = await ap.autopublish_project_artifacts(base, ORG_SCOPE, touched={"rep"})

    assert out == set()
    assert published == []
    assert "result=skipped reason=no_project_id" in caplog.text


async def test_project_id_reaches_publish_artifact(base, enabled, key, published):
    _make(base, "rep", files={"report.html": "<html></html>"},
          meta={"slug": "rep", "type": "html-app"})

    await ap.autopublish_project_artifacts(base, ORG_SCOPE, project_id=PROJECT_ID, touched={"rep"})

    assert published[0]["project_id"] == PROJECT_ID


# ─── owner filter (F1) ───────────────────────────────────────────────────


async def test_not_owned_slugs_are_dropped_and_logged(
    base, enabled, key, published, monkeypatch, caplog,
):
    """Only slugs this scope's user owns are planned; a drop is logged once,
    not per artifact retried and failed downstream on every turn."""
    import logging

    _make(base, "mine", files={"report.html": "<html></html>"},
          meta={"slug": "mine", "type": "html-app"})
    _make(base, "theirs", files={"other.html": "<html></html>"},
          meta={"slug": "theirs", "type": "html-app"})

    def fake_owned(base_, scope, project_id, slugs):
        return ([s for s in slugs if s == "mine"], 1, 0)

    monkeypatch.setattr(ap, "_owned_slugs", fake_owned)

    with caplog.at_level(logging.WARNING, logger="cowork.services.artifact_autopublish"):
        out = await ap.autopublish_project_artifacts(
            base, ORG_SCOPE, project_id=PROJECT_ID, touched={"mine", "theirs"},
        )

    assert out == {"mine"}
    assert [c["folder"].name for c in published] == ["mine"]
    assert "result=skipped not_owner=1" in caplog.text
