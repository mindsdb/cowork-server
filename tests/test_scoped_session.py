"""Behaviour of the tenant-scoped query layer (cowork/db/scoped.py).

Uses two throwaway tables — one with an org_id column, one without — on an
in-memory SQLite engine, so the tests don't depend on app models growing
tenancy columns (that's the week-2 migration).
"""
from __future__ import annotations

import pytest
from sqlalchemy.pool import StaticPool
from sqlmodel import Field, Session, SQLModel, create_engine, select
from starlette.requests import Request

from cowork.common.settings.app_settings import get_app_settings
from cowork.db.scoped import (
    LOCAL_SCOPE,
    MissingTenantScopeError,
    ScopedSession,
    TenantMismatchError,
    TenantScope,
    get_tenant_scope,
    unsafe_unscoped_session,
)
from cowork.principal import Principal

ORG_A = "6ba7b810-9dad-11d1-80b4-00c04fd430c8"
ORG_B = "0f7f0b6a-3f0f-4c58-9e0c-6dbb3ac0f0a1"


class ScopedNote(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    org_id: str | None = Field(default=None, index=True)
    title: str = ""


class PlainNote(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    title: str = ""


@pytest.fixture()
def session():
    engine = create_engine(
        "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool
    )
    ScopedNote.__table__.create(engine)
    PlainNote.__table__.create(engine)
    with Session(engine) as s:
        s.add(ScopedNote(org_id=ORG_A, title="a1"))
        s.add(ScopedNote(org_id=ORG_A, title="a2"))
        s.add(ScopedNote(org_id=ORG_B, title="b1"))
        s.add(ScopedNote(org_id=None, title="legacy"))
        s.add(PlainNote(title="plain"))
        s.commit()
        yield s


def _org_scope(org_id: str = ORG_A) -> TenantScope:
    return TenantScope(org_mode=True, org_id=org_id, user_id="user-1")


# ── select ──────────────────────────────────────────────────────────────────

def test_org_scope_filters_select(session):
    scoped = ScopedSession(session, _org_scope())
    titles = {n.title for n in scoped.exec(scoped.select(ScopedNote)).all()}
    assert titles == {"a1", "a2"}


def test_org_scope_excludes_null_org_rows(session):
    scoped = ScopedSession(session, _org_scope())
    titles = {n.title for n in scoped.exec(scoped.select(ScopedNote)).all()}
    assert "legacy" not in titles


def test_local_scope_does_not_filter(session):
    scoped = ScopedSession(session, LOCAL_SCOPE)
    rows = scoped.exec(scoped.select(ScopedNote)).all()
    assert len(rows) == 4


def test_combined_filters_keep_org_scope(session):
    scoped = ScopedSession(session, _org_scope())
    stmt = scoped.select(ScopedNote).where(ScopedNote.title == "a1")
    assert [n.title for n in scoped.exec(stmt).all()] == ["a1"]
    # a filter matching another org's row still returns nothing
    stmt = scoped.select(ScopedNote).where(ScopedNote.title == "b1")
    assert scoped.exec(stmt).all() == []


def test_chaining_returns_executable_statement(session):
    scoped = ScopedSession(session, _org_scope())
    stmt = scoped.select(ScopedNote).order_by(ScopedNote.title).limit(1)
    assert [n.title for n in scoped.exec(stmt).all()] == ["a1"]


def test_unscoped_model_passes_through(session):
    scoped = ScopedSession(session, _org_scope())
    assert [n.title for n in scoped.exec(scoped.select(PlainNote)).all()] == ["plain"]


def test_exec_rejects_raw_statements(session):
    scoped = ScopedSession(session, _org_scope())
    with pytest.raises(TypeError):
        scoped.exec(select(ScopedNote))


# ── get ─────────────────────────────────────────────────────────────────────

def test_get_own_org_row(session):
    scoped = ScopedSession(session, _org_scope())
    row = session.exec(select(ScopedNote).where(ScopedNote.title == "a1")).one()
    assert scoped.get(ScopedNote, row.id) is not None


def test_get_other_org_row_is_none(session):
    scoped = ScopedSession(session, _org_scope())
    row = session.exec(select(ScopedNote).where(ScopedNote.title == "b1")).one()
    assert scoped.get(ScopedNote, row.id) is None


def test_get_null_org_row_is_none_in_org_mode(session):
    scoped = ScopedSession(session, _org_scope())
    row = session.exec(select(ScopedNote).where(ScopedNote.title == "legacy")).one()
    assert scoped.get(ScopedNote, row.id) is None


def test_get_in_local_scope_returns_any_row(session):
    scoped = ScopedSession(session, LOCAL_SCOPE)
    row = session.exec(select(ScopedNote).where(ScopedNote.title == "b1")).one()
    assert scoped.get(ScopedNote, row.id) is not None


# ── add / delete ────────────────────────────────────────────────────────────

def test_add_stamps_unset_org_id(session):
    scoped = ScopedSession(session, _org_scope())
    note = scoped.add(ScopedNote(title="new"))
    assert note.org_id == ORG_A


def test_add_allows_matching_org_id(session):
    scoped = ScopedSession(session, _org_scope())
    note = scoped.add(ScopedNote(org_id=ORG_A, title="new"))
    assert note.org_id == ORG_A


def test_add_rejects_conflicting_org_id(session):
    scoped = ScopedSession(session, _org_scope())
    with pytest.raises(TenantMismatchError):
        scoped.add(ScopedNote(org_id=ORG_B, title="smuggled"))


def test_add_in_local_scope_does_not_stamp(session):
    scoped = ScopedSession(session, LOCAL_SCOPE)
    note = scoped.add(ScopedNote(title="desktop"))
    assert note.org_id is None


def test_delete_rejects_other_org_row(session):
    scoped = ScopedSession(session, _org_scope())
    row = session.exec(select(ScopedNote).where(ScopedNote.title == "b1")).one()
    with pytest.raises(TenantMismatchError):
        scoped.delete(row)


def test_delete_rejects_null_org_row_in_org_mode(session):
    scoped = ScopedSession(session, _org_scope())
    row = session.exec(select(ScopedNote).where(ScopedNote.title == "legacy")).one()
    with pytest.raises(TenantMismatchError):
        scoped.delete(row)


def test_commit_rejects_row_mutated_to_another_org(session):
    scoped = ScopedSession(session, _org_scope())
    row = scoped.exec(scoped.select(ScopedNote).where(ScopedNote.title == "a1")).one()
    row.org_id = ORG_B
    with pytest.raises(TenantMismatchError):
        scoped.commit()


def test_autoflush_rejects_row_mutated_to_another_org(session):
    scoped = ScopedSession(session, _org_scope())
    row = scoped.exec(scoped.select(ScopedNote).where(ScopedNote.title == "a1")).one()
    row.org_id = ORG_B
    # any query autoflushes pending changes — must be blocked there too
    with pytest.raises(TenantMismatchError):
        scoped.exec(scoped.select(ScopedNote)).all()


def test_commit_allows_legitimate_update(session):
    scoped = ScopedSession(session, _org_scope())
    row = scoped.exec(scoped.select(ScopedNote).where(ScopedNote.title == "a1")).one()
    row.title = "a1-renamed"
    scoped.commit()
    titles = {n.title for n in scoped.exec(scoped.select(ScopedNote)).all()}
    assert "a1-renamed" in titles


def test_local_scope_commit_is_unrestricted(session):
    scoped = ScopedSession(session, LOCAL_SCOPE)
    row = session.exec(select(ScopedNote).where(ScopedNote.title == "a1")).one()
    row.org_id = ORG_B
    scoped.commit()


# ── fail-closed ─────────────────────────────────────────────────────────────

def test_org_mode_without_org_fails_closed(session):
    scoped = ScopedSession(session, TenantScope(org_mode=True, org_id=None))
    with pytest.raises(MissingTenantScopeError):
        scoped.select(ScopedNote)
    with pytest.raises(MissingTenantScopeError):
        scoped.get(ScopedNote, 1)
    with pytest.raises(MissingTenantScopeError):
        scoped.add(ScopedNote(title="x"))
    with pytest.raises(MissingTenantScopeError):
        scoped.delete(ScopedNote(title="x"))


def test_org_mode_without_org_still_allows_unscoped_models(session):
    scoped = ScopedSession(session, TenantScope(org_mode=True, org_id=None))
    assert [n.title for n in scoped.exec(scoped.select(PlainNote)).all()] == ["plain"]


# ── escape hatch ────────────────────────────────────────────────────────────

def test_unsafe_escape_hatch_is_the_raw_session(session):
    scoped = ScopedSession(session, _org_scope())
    assert unsafe_unscoped_session(scoped) is session


# ── get_tenant_scope dependency ─────────────────────────────────────────────

def _request(principal: Principal | None = None) -> Request:
    req = Request({"type": "http", "method": "GET", "path": "/", "headers": []})
    if principal is not None:
        req.state.principal = principal
    return req


@pytest.fixture()
def _settings_env(monkeypatch):
    get_app_settings.cache_clear()
    yield monkeypatch
    get_app_settings.cache_clear()


def test_scope_is_local_in_local_mode(_settings_env):
    _settings_env.delenv("COWORK_TENANCY_MODE", raising=False)
    get_app_settings.cache_clear()
    assert get_tenant_scope(_request()) == LOCAL_SCOPE


def test_scope_carries_principal_in_org_mode(_settings_env):
    _settings_env.setenv("COWORK_TENANCY_MODE", "org")
    get_app_settings.cache_clear()
    principal = Principal(user_id="u-1", org_id=ORG_A)
    scope = get_tenant_scope(_request(principal))
    assert scope == TenantScope(org_mode=True, org_id=ORG_A, user_id="u-1")


def test_scope_without_principal_in_org_mode_has_no_org(_settings_env):
    _settings_env.setenv("COWORK_TENANCY_MODE", "org")
    get_app_settings.cache_clear()
    scope = get_tenant_scope(_request())
    assert scope.org_mode is True
    assert scope.org_id is None
