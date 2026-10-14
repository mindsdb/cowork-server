"""`_owned_slugs` against a real project root and DB rows (ENG-2961, F1).

The project root is shared by every member of a project (ENG-2056), so the
reconciler must filter `_candidate_slugs` down to what the calling scope's
user actually owns before planning a publish for any of it.
"""
from __future__ import annotations

from uuid import uuid4

import pytest
from sqlmodel import Session

from cowork.common.settings.app_settings import get_app_settings
from cowork.db.scoped import ScopedSession, TenantScope
from cowork.db.session import get_engine
from cowork.services import artifact_ownership as ownership
from cowork.services import artifact_autopublish as ap
from test_artifact_ownership import make_project, project_root, write_artifact

pytestmark = pytest.mark.usefixtures("cleanup_tmp_projects")


def _engine():
    return get_engine(get_app_settings().database.uri)


@pytest.fixture(autouse=True)
def org_deployment(monkeypatch):
    monkeypatch.setenv("COWORK_TENANCY_MODE", "org")
    get_app_settings.cache_clear()
    yield
    get_app_settings.cache_clear()


async def test_owned_slugs_splits_mine_theirs_and_unknown(tmp_path):
    org_id, mine, theirs = str(uuid4()), str(uuid4()), str(uuid4())
    project = make_project(tmp_path, org_id)
    source = project_root(project)

    write_artifact(source.base, "mine")
    write_artifact(source.base, "theirs")
    write_artifact(source.base, "orphan")

    with Session(_engine()) as raw:
        session = ScopedSession(raw, TenantScope(org_mode=True, org_id=org_id))
        ownership.record_artifact_owner(session, project.id, "mine", mine, action="create")
        ownership.record_artifact_owner(session, project.id, "theirs", theirs, action="create")

    scope = TenantScope(org_mode=True, org_id=org_id, user_id=mine)
    owners = await ap._owned_slugs(
        artifacts_base=source.base, scope=scope, project_id=str(project.id),
        slugs=["mine", "theirs", "orphan"],
    )

    assert owners == ap.OwnedSlugs(owned=["mine"], not_owner=1, owner_unknown=1)


async def test_owned_slugs_does_not_rediscover_roots_for_the_project_root(tmp_path, monkeypatch):
    """The project root is derived from the project row; listing every legacy
    conversation root on EFS per reconcile is only needed for legacy bases."""
    from cowork.services import artifact_roots

    org_id, mine = str(uuid4()), str(uuid4())
    project = make_project(tmp_path, org_id)
    source = project_root(project)
    write_artifact(source.base, "mine")
    with Session(_engine()) as raw:
        session = ScopedSession(raw, TenantScope(org_mode=True, org_id=org_id))
        ownership.record_artifact_owner(session, project.id, "mine", mine, action="create")

    def _no_discovery(*_args, **_kwargs):
        raise AssertionError("the project root must not trigger root discovery")

    monkeypatch.setattr(artifact_roots, "artifacts_sources_for_project", _no_discovery)
    scope = TenantScope(org_mode=True, org_id=org_id, user_id=mine)
    assert await ap._owned_slugs(
        artifacts_base=source.base, scope=scope, project_id=str(project.id), slugs=["mine"],
    ) == ap.OwnedSlugs(owned=["mine"], not_owner=0, owner_unknown=0)


async def test_owned_slugs_fails_closed_for_a_project_outside_the_scope(tmp_path):
    org_id, mine = str(uuid4()), str(uuid4())
    project = make_project(tmp_path, org_id)
    source = project_root(project)
    write_artifact(source.base, "mine")
    with Session(_engine()) as raw:
        session = ScopedSession(raw, TenantScope(org_mode=True, org_id=org_id))
        ownership.record_artifact_owner(session, project.id, "mine", mine, action="create")

    other_org = TenantScope(org_mode=True, org_id=str(uuid4()), user_id=mine)
    assert await ap._owned_slugs(
        artifacts_base=source.base, scope=other_org, project_id=str(project.id), slugs=["mine"],
    ) == ap.OwnedSlugs(owned=[], not_owner=0, owner_unknown=1)


@pytest.fixture
def published(monkeypatch) -> list[str]:
    """The slugs the reconciler publishes, through a stand-in key and
    publisher that record each one as published."""
    import json

    class FakeKey:
        instance_id = "inst-1"

        def __init__(self, *a, **kw):
            pass

        async def get(self):
            return "turnkey-1"

        async def revoke(self):
            pass

    slugs: list[str] = []

    def fake_publish(artifact, *, artifacts_base, api_key, publish_url, password=None,
                     access=None, scope=None, project_id=None):
        slugs.append(artifact.name)
        (artifact / ".published.json").write_text(json.dumps({
            "index.html": {"report_id": "rid", "url": "u", "published": True,
                           "last_md5": "m", "published_mtime": 9_999_999_999},
        }))
        return {"status": "ok", "url": "u"}

    monkeypatch.setattr(ap, "PublishKey", FakeKey)
    monkeypatch.setattr(ap, "publish_artifact", fake_publish)
    return slugs


async def test_reconcile_publishes_only_the_scope_users_artifact(tmp_path, monkeypatch, caplog, published):
    """End to end with the REAL owner filter: of two artifacts touched in a
    shared project root, only the one recorded for the scope's user is
    published; the other is skipped and counted once."""
    import logging

    org_id, mine, theirs = str(uuid4()), str(uuid4()), str(uuid4())
    project = make_project(tmp_path, org_id)
    source = project_root(project)
    write_artifact(source.base, "mine")
    write_artifact(source.base, "theirs")
    with Session(_engine()) as raw:
        session = ScopedSession(raw, TenantScope(org_mode=True, org_id=org_id))
        ownership.record_artifact_owner(session, project.id, "mine", mine, action="create")
        ownership.record_artifact_owner(session, project.id, "theirs", theirs, action="create")

    monkeypatch.setattr(ap, "_is_enabled", lambda scope: True)
    monkeypatch.setattr(ap, "_publish_url", lambda scope: "https://api.staging.mindshub.ai")

    scope = TenantScope(org_mode=True, org_id=org_id, user_id=mine)
    with caplog.at_level(logging.WARNING, logger=ap.__name__):
        out = await ap.autopublish_project_artifacts(
            source.base, scope, project_id=str(project.id), touched={"mine", "theirs"},
        )

    assert out == {"mine"}
    assert published == ["mine"]
    assert "result=skipped not_owner=1" in caplog.text


async def test_a_turns_reconcile_checks_out_nothing_on_the_event_loop(tmp_path, monkeypatch, published):
    """A hosted remote turn reconciles its artifacts on the event loop. The
    owner lookup runs as a database unit in a worker thread, and the enable
    flag, the publish URL and the workspace are read from the turn's settings
    snapshot, so nothing in the reconcile checks a connection out on the
    loop's thread or loads settings again."""
    from cowork.common.settings import user_settings
    from cowork.services.settings import SettingService
    from test_turn_pool_connections import _logging_checkouts

    org_id, mine = str(uuid4()), str(uuid4())
    project = make_project(tmp_path, org_id)
    source = project_root(project)
    write_artifact(source.base, "mine")
    with Session(_engine()) as raw:
        session = ScopedSession(raw, TenantScope(org_mode=True, org_id=org_id))
        ownership.record_artifact_owner(session, project.id, "mine", mine, action="create")
    loads: list[TenantScope] = []
    load = SettingService.load

    def counted_load(self):
        loads.append(self.scope)
        return load(self)

    monkeypatch.setattr(SettingService, "load", counted_load)
    scope = TenantScope(org_mode=True, org_id=org_id, user_id=mine)
    snapshot = user_settings.UserSettings(artifact_autopublish_enabled=True)

    with _logging_checkouts(_engine()) as log, user_settings.use_turn_settings(scope, snapshot):
        out = await ap.autopublish_project_artifacts(
            source.base, scope, project_id=str(project.id), touched={"mine"},
        )

    assert out == {"mine"}
    assert published == ["mine"]
    assert log.checkouts, "the owner lookup checked out no connection"
    assert log.on_the_loop == [], f"{len(log.on_the_loop)} checkout(s) on the event loop's thread"
    assert loads == []
