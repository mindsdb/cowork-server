"""Resolving artifact roots — the one place that branches on tenancy mode.

Org mode resolves from the DB through a ScopedSession, so a project belonging to
another organization simply is not found. Desktop keeps the filesystem scan but
still resolves by id, which is what keeps a slug-addressed delete from acting on
whichever project happens to sort first.
"""
from __future__ import annotations

import uuid

import pytest
from sqlmodel import Session

from cowork.common.settings.app_settings import get_app_settings
from cowork.db.scoped import LOCAL_SCOPE, ScopedSession, TenantScope
from cowork.db.session import get_engine
from cowork.models.project import Project
from cowork.services.artifact_roots import (
    artifacts_source_for_project,
    artifacts_sources_for_scope,
)

ORG_A = "11111111-1111-1111-1111-111111111111"
ORG_B = "22222222-2222-2222-2222-222222222222"


@pytest.fixture
def session():
    engine = get_engine(get_app_settings().database.uri)
    with Session(engine) as s:
        yield s


def _project(session, tmp_path, *, name: str, org_id: str | None) -> Project:
    path = tmp_path / (org_id or "local") / name
    path.mkdir(parents=True, exist_ok=True)
    row = Project(id=uuid.uuid4(), name=name, path=str(path), org_id=org_id)
    session.add(row)
    session.commit()
    return row


@pytest.fixture
def org_mode(monkeypatch):
    # `_is_org_mode` is imported into artifact_roots as a module-level name, so
    # patching it there is what the resolver actually reads.
    monkeypatch.setattr("cowork.services.artifact_roots._is_org_mode", lambda: True)


def test_source_for_project_points_at_the_project_artifacts_dir(session, tmp_path, org_mode):
    row = _project(session, tmp_path, name="proj-a", org_id=ORG_A)
    scoped = ScopedSession(session, TenantScope(org_mode=True, org_id=ORG_A, user_id="u"))

    source = artifacts_source_for_project(scoped, row.id)

    assert source.base == tmp_path / ORG_A / "proj-a" / ".anton" / "artifacts"
    assert source.project_id == str(row.id)
    assert source.project_name == "proj-a"


def test_source_for_foreign_project_raises(session, tmp_path, org_mode):
    row = _project(session, tmp_path, name="proj-b", org_id=ORG_B)
    scoped = ScopedSession(session, TenantScope(org_mode=True, org_id=ORG_A, user_id="u"))

    with pytest.raises(ValueError):
        artifacts_source_for_project(scoped, row.id)


def test_sources_for_scope_covers_only_own_org(session, tmp_path, org_mode):
    mine = _project(session, tmp_path, name="mine", org_id=ORG_A)
    _project(session, tmp_path, name="theirs", org_id=ORG_B)
    scoped = ScopedSession(session, TenantScope(org_mode=True, org_id=ORG_A, user_id="u"))

    sources = artifacts_sources_for_scope(scoped)
    names = {s.project_name for s in sources}

    assert "mine" in names
    assert "theirs" not in names
    assert str(mine.id) in {s.project_id for s in sources}


def test_source_for_project_works_in_desktop_mode_too(session, tmp_path, monkeypatch):
    """Desktop resolves by id as well — that is what keeps slug-addressed delete
    from acting on whichever project sorts first."""
    monkeypatch.setattr("cowork.services.artifact_roots._is_org_mode", lambda: False)
    row = _project(session, tmp_path, name="solo", org_id=None)
    scoped = ScopedSession(session, LOCAL_SCOPE)

    source = artifacts_source_for_project(scoped, row.id)

    assert source.base == tmp_path / "local" / "solo" / ".anton" / "artifacts"
    assert source.project_id == str(row.id)


def test_sources_for_scope_falls_back_to_the_scan_in_desktop_mode(session, monkeypatch):
    """Local mode must keep listing exactly what it listed before — the scan,
    not a DB read (a desktop install has no org rows to scope by)."""
    monkeypatch.setattr("cowork.services.artifact_roots._is_org_mode", lambda: False)
    called = []
    monkeypatch.setattr(
        "cowork.services.artifact_roots.artifacts_sources_for_scan",
        lambda: called.append(1) or [],
    )

    assert artifacts_sources_for_scope(ScopedSession(session, LOCAL_SCOPE)) == []
    assert called == [1]
