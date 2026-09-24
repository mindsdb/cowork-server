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
