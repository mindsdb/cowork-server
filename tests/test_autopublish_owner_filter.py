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
    owned, not_owner, owner_unknown = ap._owned_slugs(
        source.base, scope, str(project.id), ["mine", "theirs", "orphan"]
    )

    assert owned == ["mine"]
    assert not_owner == 1
    assert owner_unknown == 1


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
    assert ap._owned_slugs(source.base, scope, str(project.id), ["mine"]) == (["mine"], 0, 0)


async def test_owned_slugs_fails_closed_for_a_project_outside_the_scope(tmp_path):
    org_id, mine = str(uuid4()), str(uuid4())
    project = make_project(tmp_path, org_id)
    source = project_root(project)
    write_artifact(source.base, "mine")
    with Session(_engine()) as raw:
        session = ScopedSession(raw, TenantScope(org_mode=True, org_id=org_id))
        ownership.record_artifact_owner(session, project.id, "mine", mine, action="create")

    other_org = TenantScope(org_mode=True, org_id=str(uuid4()), user_id=mine)
    assert ap._owned_slugs(source.base, other_org, str(project.id), ["mine"]) == ([], 0, 1)
