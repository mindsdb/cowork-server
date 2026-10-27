"""Behaviour of the tenant-scoped query layer (cowork/db/scoped.py).

Uses two throwaway tables — one with an org_id column, one without — on an
in-memory SQLite engine, so the tests don't depend on app models growing
tenancy columns (that's the week-2 migration).

Every partitioning test runs twice, through the ``partition`` fixture: once in
hosted org mode and once in a personal scope (a trusted-identity deployment)
whose resources include the one the throwaway tables are mapped to. The
``shared_`` tests take the same personal scope with that resource left off,
which must behave exactly like LOCAL_SCOPE.
"""
from __future__ import annotations

import pytest
from sqlalchemy.pool import StaticPool
from sqlmodel import Field, Session, SQLModel, create_engine, select
from starlette.requests import Request

from cowork.common.settings.app_settings import get_app_settings
from cowork.db.personal_resources import ALL_PERSONAL_RESOURCES, MODEL_RESOURCE, PersonalResource
from cowork.db.scoped import (
    LOCAL_SCOPE,
    MissingTenantScopeError,
    ScopedSession,
    TenantMismatchError,
    TenantScope,
    get_tenant_scope,
    scope_for_identity,
    scope_for_org,
    scope_from_principal,
    scope_of_session,
    scoped_storage_root,
    scoped_user_storage_root,
    service_principal_for,
    tenant_scope_required,
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


class AuthoredNote(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    org_id: str | None = Field(default=None, index=True)
    created_by: str | None = Field(default=None)
    title: str = ""


class ChildNote(SQLModel, table=True):
    # created_by without org_id — the messages shape (tenancy via parent).
    id: int | None = Field(default=None, primary_key=True)
    created_by: str | None = Field(default=None)
    title: str = ""


class DeferredNote(SQLModel, table=True):
    # org_id present but scoping deferred — the settings shape (inert scope
    # columns until the settings split). Deferral is granted by the closed
    # allowlist in cowork/db/scoped.py, not by anything on the model; the
    # deferred-model tests patch that allowlist to include this table.
    id: int | None = Field(default=None, primary_key=True)
    org_id: str | None = Field(default=None, index=True)
    title: str = ""


@pytest.fixture()
def session():
    engine = create_engine(
        "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool
    )
    ScopedNote.__table__.create(engine)
    PlainNote.__table__.create(engine)
    AuthoredNote.__table__.create(engine)
    ChildNote.__table__.create(engine)
    DeferredNote.__table__.create(engine)
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


# The resource the throwaway tables stand for in a personal scope.
NOTES_RESOURCE = PersonalResource.PINS


def _personal_scope(
    org_id: str = ORG_A, resources: frozenset[PersonalResource] = ALL_PERSONAL_RESOURCES
) -> TenantScope:
    return TenantScope(personal=True, org_id=org_id, user_id="user-1", personal_resources=resources)


def _shared_notes_scope(org_id: str = ORG_A) -> TenantScope:
    """A person's scope that leaves the notes' resource shared."""
    return _personal_scope(org_id, ALL_PERSONAL_RESOURCES - {NOTES_RESOURCE})


@pytest.fixture()
def notes_resource(monkeypatch):
    """Map the throwaway tables to NOTES_RESOURCE, as MODEL_RESOURCE maps the
    real ones, so a personal scope can keep them personal or leave them shared."""
    for model in (ScopedNote, AuthoredNote, ChildNote):
        monkeypatch.setitem(MODEL_RESOURCE, model.__tablename__, NOTES_RESOURCE)


@pytest.fixture(params=["org", "personal"])
def partition(request, notes_resource):
    """Builds a scope that partitions the notes: hosted org mode, or a person's
    partition with NOTES_RESOURCE personal."""
    return {"org": _org_scope, "personal": _personal_scope}[request.param]


# ── select ──────────────────────────────────────────────────────────────────

def test_org_scope_filters_select(session, partition):
    scoped = ScopedSession(session, partition())
    titles = {n.title for n in scoped.exec(scoped.select(ScopedNote)).all()}
    assert titles == {"a1", "a2"}


def test_org_scope_excludes_null_org_rows(session, partition):
    scoped = ScopedSession(session, partition())
    titles = {n.title for n in scoped.exec(scoped.select(ScopedNote)).all()}
    assert "legacy" not in titles


def test_local_scope_does_not_filter(session):
    scoped = ScopedSession(session, LOCAL_SCOPE)
    rows = scoped.exec(scoped.select(ScopedNote)).all()
    assert len(rows) == 4


def test_combined_filters_keep_the_partition(session, partition):
    scoped = ScopedSession(session, partition())
    stmt = scoped.select(ScopedNote).where(ScopedNote.title == "a1")
    assert [n.title for n in scoped.exec(stmt).all()] == ["a1"]
    # a filter matching another org's row still returns nothing
    stmt = scoped.select(ScopedNote).where(ScopedNote.title == "b1")
    assert scoped.exec(stmt).all() == []


def test_chaining_returns_executable_statement(session, partition):
    scoped = ScopedSession(session, partition())
    stmt = scoped.select(ScopedNote).order_by(ScopedNote.title).limit(1)
    assert [n.title for n in scoped.exec(stmt).all()] == ["a1"]


def test_unscoped_model_passes_through(session, partition):
    scoped = ScopedSession(session, partition())
    assert [n.title for n in scoped.exec(scoped.select(PlainNote)).all()] == ["plain"]


def test_exec_rejects_raw_statements(session, partition):
    scoped = ScopedSession(session, partition())
    with pytest.raises(TypeError):
        scoped.exec(select(ScopedNote))


# ── get ─────────────────────────────────────────────────────────────────────

def test_get_own_org_row(session, partition):
    scoped = ScopedSession(session, partition())
    row = session.exec(select(ScopedNote).where(ScopedNote.title == "a1")).one()
    assert scoped.get(ScopedNote, row.id) is not None


def test_get_other_org_row_is_none(session, partition):
    scoped = ScopedSession(session, partition())
    row = session.exec(select(ScopedNote).where(ScopedNote.title == "b1")).one()
    assert scoped.get(ScopedNote, row.id) is None


def test_get_null_org_row_is_none_in_org_mode(session, partition):
    scoped = ScopedSession(session, partition())
    row = session.exec(select(ScopedNote).where(ScopedNote.title == "legacy")).one()
    assert scoped.get(ScopedNote, row.id) is None


def test_get_in_local_scope_returns_any_row(session):
    scoped = ScopedSession(session, LOCAL_SCOPE)
    row = session.exec(select(ScopedNote).where(ScopedNote.title == "b1")).one()
    assert scoped.get(ScopedNote, row.id) is not None


# ── add / delete ────────────────────────────────────────────────────────────

def test_add_stamps_unset_org_id(session, partition):
    scoped = ScopedSession(session, partition())
    note = scoped.add(ScopedNote(title="new"))
    assert note.org_id == ORG_A


def test_add_allows_matching_org_id(session, partition):
    scoped = ScopedSession(session, partition())
    note = scoped.add(ScopedNote(org_id=ORG_A, title="new"))
    assert note.org_id == ORG_A


def test_add_rejects_conflicting_org_id(session, partition):
    scoped = ScopedSession(session, partition())
    with pytest.raises(TenantMismatchError):
        scoped.add(ScopedNote(org_id=ORG_B, title="smuggled"))


def test_add_stamps_created_by(session, partition):
    scoped = ScopedSession(session, partition())
    note = scoped.add(AuthoredNote(title="mine"))
    assert note.created_by == "user-1"
    assert note.org_id == ORG_A


def test_add_stamps_created_by_on_child_models(session, partition):
    # messages shape: created_by but no org_id — still attributed.
    scoped = ScopedSession(session, partition())
    note = scoped.add(ChildNote(title="reply"))
    assert note.created_by == "user-1"


def test_add_does_not_overwrite_created_by(session, partition):
    scoped = ScopedSession(session, partition())
    note = scoped.add(AuthoredNote(title="theirs", created_by="someone-else"))
    assert note.created_by == "someone-else"


def test_add_in_local_scope_does_not_stamp_created_by(session):
    scoped = ScopedSession(session, LOCAL_SCOPE)
    note = scoped.add(AuthoredNote(title="desktop"))
    assert note.created_by is None


def test_add_in_local_scope_does_not_stamp(session):
    scoped = ScopedSession(session, LOCAL_SCOPE)
    note = scoped.add(ScopedNote(title="desktop"))
    assert note.org_id is None


def test_delete_rejects_other_org_row(session, partition):
    scoped = ScopedSession(session, partition())
    row = session.exec(select(ScopedNote).where(ScopedNote.title == "b1")).one()
    with pytest.raises(TenantMismatchError):
        scoped.delete(row)


def test_delete_rejects_null_org_row_in_org_mode(session, partition):
    scoped = ScopedSession(session, partition())
    row = session.exec(select(ScopedNote).where(ScopedNote.title == "legacy")).one()
    with pytest.raises(TenantMismatchError):
        scoped.delete(row)


def test_commit_rejects_row_mutated_to_another_org(session, partition):
    scoped = ScopedSession(session, partition())
    row = scoped.exec(scoped.select(ScopedNote).where(ScopedNote.title == "a1")).one()
    row.org_id = ORG_B
    with pytest.raises(TenantMismatchError):
        scoped.commit()


def test_autoflush_rejects_row_mutated_to_another_org(session, partition):
    scoped = ScopedSession(session, partition())
    row = scoped.exec(scoped.select(ScopedNote).where(ScopedNote.title == "a1")).one()
    row.org_id = ORG_B
    # any query autoflushes pending changes — must be blocked there too
    with pytest.raises(TenantMismatchError):
        scoped.exec(scoped.select(ScopedNote)).all()


def test_commit_allows_legitimate_update(session, partition):
    scoped = ScopedSession(session, partition())
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

def test_flush_stamps_unswept_writes(session, partition):
    # A service not yet swept writes through the raw request session; the
    # scope's flush hook adopts the row instead of crashing the request.
    ScopedSession(session, partition())
    session.add(ScopedNote(title="from-unswept-service"))
    session.add(ChildNote(title="authored"))
    session.commit()
    row = session.exec(select(ScopedNote).where(ScopedNote.title == "from-unswept-service")).one()
    assert row.org_id == ORG_A
    child = session.exec(select(ChildNote).where(ChildNote.title == "authored")).one()
    assert child.created_by == "user-1"


def test_flush_allows_deleting_null_org_rows(session, partition):
    # Pre-sweep rows cleaned up by their own tenant (cascade case).
    ScopedSession(session, partition())
    row = session.exec(select(ScopedNote).where(ScopedNote.title == "legacy")).one()
    session.delete(row)
    session.commit()


def test_flush_still_rejects_cross_org_deletes(session, partition):
    ScopedSession(session, partition())
    row = session.exec(select(ScopedNote).where(ScopedNote.title == "b1")).one()
    session.delete(row)
    with pytest.raises(TenantMismatchError):
        session.commit()


@pytest.mark.parametrize(
    "scope",
    [
        TenantScope(org_mode=True, org_id=None),
        TenantScope(personal=True, org_id=None, personal_resources=ALL_PERSONAL_RESOURCES),
    ],
    ids=["org", "personal"],
)
def test_partitioned_scope_without_org_fails_closed(session, notes_resource, scope):
    scoped = ScopedSession(session, scope)
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


# ── deferred models (the settings shape) ────────────────────────────────────

@pytest.fixture()
def _defer_note(monkeypatch):
    """Grant DeferredNote a slot on the closed allowlist for one test."""
    import cowork.db.scoped as scoped_mod

    monkeypatch.setattr(
        scoped_mod,
        "_TENANCY_DEFERRED_TABLES",
        scoped_mod._TENANCY_DEFERRED_TABLES | {DeferredNote.__tablename__},
    )


def test_deferral_is_allowlist_gated_not_model_gated(session, partition):
    # Without an allowlist entry, an org_id-bearing model is scoped no matter
    # what the model itself declares — there is no model-side opt-out.
    scoped = ScopedSession(session, partition())
    note = scoped.add(DeferredNote(title="scoped-by-default"))
    assert note.org_id == ORG_A


def test_deferred_model_select_is_unfiltered_in_org_mode(session, _defer_note):
    # Rows written before the settings split (NULL org, e.g. env-seeded)
    # must stay visible to org-mode reads.
    session.add(DeferredNote(org_id=None, title="env-seeded"))
    session.add(DeferredNote(org_id=ORG_B, title="other-org"))
    session.commit()
    scoped = ScopedSession(session, _org_scope())
    titles = {n.title for n in scoped.exec(scoped.select(DeferredNote)).all()}
    assert titles == {"env-seeded", "other-org"}


def test_deferred_model_add_is_not_stamped(session, _defer_note):
    scoped = ScopedSession(session, _org_scope())
    note = scoped.add(DeferredNote(title="unstamped"))
    scoped.commit()
    assert note.org_id is None


def test_deferred_model_flush_is_not_stamped(session, _defer_note):
    # The unswept-service flush hook must leave deferred rows alone too.
    ScopedSession(session, _org_scope())
    session.add(DeferredNote(title="from-raw-session"))
    session.commit()
    row = session.exec(select(DeferredNote).where(DeferredNote.title == "from-raw-session")).one()
    assert row.org_id is None


def test_deferred_model_ignores_missing_org(session, _defer_note):
    # Fail-closed does not apply: deferred models behave like unscoped ones.
    scoped = ScopedSession(session, TenantScope(org_mode=True, org_id=None))
    scoped.add(DeferredNote(title="no-org-in-scope"))
    scoped.commit()
    assert scoped.exec(scoped.select(DeferredNote)).all()


# ── escape hatch ────────────────────────────────────────────────────────────

def test_unsafe_escape_hatch_is_the_raw_session(session, partition):
    scoped = ScopedSession(session, partition())
    assert unsafe_unscoped_session(scoped) is session


def test_wrapping_a_scoped_session_is_rejected(session, partition):
    # A wrapper-of-wrapper breaks exec() silently later — fail at construction.
    scoped = ScopedSession(session, partition())
    with pytest.raises(TypeError, match="cannot wrap"):
        ScopedSession(scoped, partition())


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


# ── service_principal_for: identity for a background job from its own row ──

def test_service_principal_is_none_in_local_mode(_settings_env):
    _settings_env.delenv("COWORK_TENANCY_MODE", raising=False)
    get_app_settings.cache_clear()
    assert service_principal_for(ORG_A, "u-1") is None


def test_service_principal_built_from_org_and_user_in_org_mode(_settings_env):
    _settings_env.setenv("COWORK_TENANCY_MODE", "org")
    get_app_settings.cache_clear()
    principal = service_principal_for(ORG_A, "u-1")
    assert principal == Principal(user_id="u-1", org_id=ORG_A)


def test_service_principal_fails_closed_without_org_id(_settings_env):
    _settings_env.setenv("COWORK_TENANCY_MODE", "org")
    get_app_settings.cache_clear()
    with pytest.raises(MissingTenantScopeError):
        service_principal_for(None, "u-1")


def test_service_principal_fails_closed_without_user_id(_settings_env):
    _settings_env.setenv("COWORK_TENANCY_MODE", "org")
    get_app_settings.cache_clear()
    with pytest.raises(MissingTenantScopeError):
        service_principal_for(ORG_A, None)


# ── a personal scope that leaves the notes' resource shared ─────────────────
# Each of these is a LOCAL_SCOPE test above, run with a person in scope.

def test_shared_resource_is_not_filtered(session, notes_resource):
    scoped = ScopedSession(session, _shared_notes_scope())
    assert len(scoped.exec(scoped.select(ScopedNote)).all()) == 4


def test_shared_resource_get_returns_any_row(session, notes_resource):
    scoped = ScopedSession(session, _shared_notes_scope())
    row = session.exec(select(ScopedNote).where(ScopedNote.title == "b1")).one()
    assert scoped.get(ScopedNote, row.id) is not None


def test_shared_resource_add_stamps_nothing(session, notes_resource):
    scoped = ScopedSession(session, _shared_notes_scope())
    note = scoped.add(AuthoredNote(title="shared"))
    child = scoped.add(ChildNote(title="shared-child"))
    assert note.org_id is None
    assert note.created_by is None
    assert child.created_by is None


def test_shared_resource_flush_stamps_nothing(session, notes_resource):
    ScopedSession(session, _shared_notes_scope())
    session.add(ScopedNote(title="raw-shared"))
    session.add(ChildNote(title="raw-shared-child"))
    session.commit()
    assert session.exec(select(ScopedNote).where(ScopedNote.title == "raw-shared")).one().org_id is None
    assert session.exec(select(ChildNote).where(ChildNote.title == "raw-shared-child")).one().created_by is None


def test_shared_resource_allows_any_delete(session, notes_resource):
    scoped = ScopedSession(session, _shared_notes_scope())
    row = session.exec(select(ScopedNote).where(ScopedNote.title == "b1")).one()
    scoped.delete(row)
    scoped.commit()


def test_shared_resource_needs_no_org(session, notes_resource):
    scope = TenantScope(personal=True, org_id=None, personal_resources=frozenset())
    scoped = ScopedSession(session, scope)
    assert len(scoped.exec(scoped.select(ScopedNote)).all()) == 4


def test_shared_and_personal_tables_in_one_scope(session, monkeypatch):
    """One scope, two resources: the personal table is filtered, the shared one
    is not."""
    monkeypatch.setitem(MODEL_RESOURCE, ScopedNote.__tablename__, PersonalResource.PINS)
    monkeypatch.setitem(MODEL_RESOURCE, AuthoredNote.__tablename__, PersonalResource.SKILLS)
    session.add(AuthoredNote(org_id=ORG_B, title="someone-elses"))
    session.commit()
    scoped = ScopedSession(session, _personal_scope(resources=frozenset({PersonalResource.PINS})))
    assert {n.title for n in scoped.exec(scoped.select(ScopedNote)).all()} == {"a1", "a2"}
    assert [n.title for n in scoped.exec(scoped.select(AuthoredNote)).all()] == ["someone-elses"]


def test_unmapped_table_is_partitioned_in_a_personal_scope(session):
    """A table nobody mapped yet is never shared by omission."""
    scoped = ScopedSession(session, _personal_scope(resources=frozenset()))
    assert {n.title for n in scoped.exec(scoped.select(ScopedNote)).all()} == {"a1", "a2"}


# ── session.info ────────────────────────────────────────────────────────────

def test_partitioned_scope_is_recorded_on_the_session(session, partition):
    scope = partition()
    ScopedSession(session, scope)
    assert scope_of_session(session) == scope


def test_local_scope_is_not_recorded_on_the_session(session):
    ScopedSession(session, LOCAL_SCOPE)
    assert scope_of_session(session) is None


def test_two_people_on_one_session_is_refused(session):
    ScopedSession(session, _personal_scope(ORG_A))
    with pytest.raises(RuntimeError, match="different org"):
        ScopedSession(session, _personal_scope(ORG_B))


# ── TenantScope ─────────────────────────────────────────────────────────────

def test_local_scope_partitions_nothing():
    assert LOCAL_SCOPE.partitioned is False
    assert not any(LOCAL_SCOPE.partitioned_for(resource) for resource in PersonalResource)
    assert LOCAL_SCOPE.for_resource(PersonalResource.MEMORY) is LOCAL_SCOPE


def test_org_scope_partitions_every_resource():
    scope = _org_scope()
    assert scope.partitioned is True
    assert all(scope.partitioned_for(resource) for resource in PersonalResource)
    assert scope.for_resource(PersonalResource.MEMORY) is scope


def test_personal_scope_partitions_only_its_resources():
    scope = _personal_scope(resources=frozenset({PersonalResource.MEMORY}))
    assert scope.partitioned is True
    assert scope.partitioned_for(PersonalResource.MEMORY) is True
    assert scope.partitioned_for(PersonalResource.SKILLS) is False
    assert scope.for_resource(PersonalResource.MEMORY) is scope
    assert scope.for_resource(PersonalResource.SKILLS) is LOCAL_SCOPE


def test_personal_scope_with_no_resources_is_still_partitioned():
    # A person exists for logs and traces; every resource is shared.
    scope = _personal_scope(resources=frozenset())
    assert scope.partitioned is True
    assert not any(scope.partitioned_for(resource) for resource in PersonalResource)


# ── scope builders ──────────────────────────────────────────────────────────

PARTITION = "44e92829-e701-599b-88f9-69f5bf1165eb"
SUB = "9a8b7c6d-5e4f-4a3b-8c2d-1e0f9a8b7c6d"


def test_desktop_never_builds_a_partition(_settings_env):
    """Trust off: a principal, an org_id or both still give LOCAL_SCOPE."""
    _settings_env.delenv("COWORK_TENANCY_MODE", raising=False)
    _settings_env.delenv("COWORK_TRUST_IDENTITY_HEADERS", raising=False)
    get_app_settings.cache_clear()
    assert scope_from_principal(Principal(user_id=SUB, org_id=PARTITION)) == LOCAL_SCOPE
    assert scope_for_identity(org_id=PARTITION, user_id=SUB) == LOCAL_SCOPE
    assert scope_for_org(PARTITION) == LOCAL_SCOPE
    assert tenant_scope_required() is False


def test_trusted_identity_builds_a_personal_scope(_settings_env, personal_deployment):
    scope = scope_from_principal(Principal(user_id=SUB, org_id=PARTITION))
    assert scope == TenantScope(
        personal=True, org_id=PARTITION, user_id=SUB, personal_resources=ALL_PERSONAL_RESOURCES
    )
    assert scope.org_mode is False
    assert tenant_scope_required() is True


def test_trusted_identity_scope_carries_the_configured_resources(_settings_env, personal_deployment):
    _settings_env.setenv("COWORK_PERSONAL_RESOURCES", "memory,skills")
    get_app_settings.cache_clear()
    scope = scope_from_principal(Principal(user_id=SUB, org_id=PARTITION))
    assert scope.personal_resources == {PersonalResource.MEMORY, PersonalResource.SKILLS}


def test_trusted_identity_without_a_person_is_the_operator_view(_settings_env, personal_deployment):
    assert scope_from_principal(None) == LOCAL_SCOPE
    assert get_tenant_scope(_request()) == LOCAL_SCOPE


def test_trusted_identity_org_id_alone_is_a_personless_partition(_settings_env, personal_deployment):
    scope = scope_for_org(PARTITION)
    assert scope.personal is True and scope.org_mode is False
    assert scope.org_id == PARTITION and scope.user_id is None
    assert scope_for_org(None) == LOCAL_SCOPE
    assert scope_for_identity(org_id=None, user_id=SUB) == LOCAL_SCOPE


def test_org_tenancy_builders_are_unchanged(_settings_env, org_deployment):
    assert scope_for_identity(org_id=ORG_A, user_id="u-1") == TenantScope(org_mode=True, org_id=ORG_A, user_id="u-1")
    assert scope_for_org(ORG_A) == TenantScope(org_mode=True, org_id=ORG_A)
    assert scope_for_org(None) == LOCAL_SCOPE
    assert scope_from_principal(None) == TenantScope(org_mode=True)
    assert tenant_scope_required() is True


# ── service_principal_for with trusted identity ─────────────────────────────

def test_personal_schedules_run_as_their_creator(_settings_env, personal_deployment):
    # The stored org_id is already the partition id: used as it is, never hashed again.
    assert service_principal_for(PARTITION, SUB) == Principal(user_id=SUB, org_id=PARTITION)


@pytest.mark.parametrize("org_id, user_id", [(None, SUB), (PARTITION, None), (None, None)])
def test_personal_schedule_without_an_owner_fails_closed(_settings_env, personal_deployment, org_id, user_id):
    with pytest.raises(MissingTenantScopeError):
        service_principal_for(org_id, user_id)


def test_shared_schedules_run_unscoped(_settings_env, personal_deployment):
    _settings_env.setenv("COWORK_PERSONAL_RESOURCES", "memory,skills")
    get_app_settings.cache_clear()
    assert service_principal_for(PARTITION, SUB) is None
    assert service_principal_for(None, None) is None


# ── storage roots ───────────────────────────────────────────────────────────

@pytest.fixture()
def _roots(_settings_env, tmp_path):
    _settings_env.setenv("COWORK_SHARED_DIR", str(tmp_path / "shared"))
    get_app_settings.cache_clear()
    return tmp_path


def test_personal_store_lives_in_the_partition(_roots, personal_deployment):
    shared = _roots / "shared"
    scope = _personal_scope(PARTITION, ALL_PERSONAL_RESOURCES)
    assert scoped_storage_root(_roots / "base-skills", scope, store="skills") == shared / PARTITION / "skills"
    assert scoped_user_storage_root(_roots / "base-memory", scope, store="memory") == (
        shared / PARTITION / "memory" / "users" / "user-1"
    )


def test_user_store_segments_match_scoped_user_storage_root(_roots, personal_deployment):
    """The per-person tier walk (remove_project_memory_tiers) derives its path
    from user_store_segments, so it must land exactly where
    scoped_user_storage_root puts the tier."""
    from cowork.db.scoped import user_store_segments

    shared = _roots / "shared"
    scope = _personal_scope(PARTITION, ALL_PERSONAL_RESOURCES)
    assert shared.joinpath(*user_store_segments(PARTITION, "user-1", store="memory")) == (
        scoped_user_storage_root(_roots / "base-memory", scope, store="memory")
    )


def test_shared_store_stays_at_the_base(_roots, personal_deployment):
    scope = _personal_scope(PARTITION, ALL_PERSONAL_RESOURCES - {PersonalResource.SKILLS, PersonalResource.MEMORY})
    assert scoped_storage_root(_roots / "base-skills", scope, store="skills") == _roots / "base-skills"
    assert scoped_user_storage_root(_roots / "base-memory", scope, store="memory") == _roots / "base-memory"
    # files are still personal in the same scope
    assert scoped_storage_root(_roots / "base-files", scope, store="files") == _roots / "shared" / PARTITION / "files"


def test_unmapped_store_is_partitioned_in_a_personal_scope(_roots, personal_deployment):
    scope = _personal_scope(PARTITION, frozenset())
    assert scoped_storage_root(_roots / "x", scope, store="not-yet-mapped") == _roots / "shared" / PARTITION / "not-yet-mapped"


def test_personal_memory_without_a_person_fails_closed(_roots, personal_deployment):
    scope = TenantScope(personal=True, org_id=PARTITION, personal_resources=ALL_PERSONAL_RESOURCES)
    with pytest.raises(MissingTenantScopeError):
        scoped_user_storage_root(_roots / "base-memory", scope, store="memory")


def test_no_scope_fails_closed_with_trusted_identity(_roots, personal_deployment):
    with pytest.raises(MissingTenantScopeError):
        scoped_storage_root(_roots / "base-skills", None, store="skills")
    with pytest.raises(MissingTenantScopeError):
        scoped_user_storage_root(_roots / "base-memory", None, store="memory")


def test_no_scope_is_the_base_on_desktop(_roots, _settings_env):
    _settings_env.delenv("COWORK_TRUST_IDENTITY_HEADERS", raising=False)
    _settings_env.delenv("COWORK_TENANCY_MODE", raising=False)
    get_app_settings.cache_clear()
    assert scoped_storage_root(_roots / "base-skills", None, store="skills") == _roots / "base-skills"
    assert scoped_user_storage_root(_roots / "base-memory", None, store="memory") == _roots / "base-memory"


def test_local_scope_is_the_base_with_trusted_identity(_roots, personal_deployment):
    # The operator view reads the base stores.
    assert scoped_storage_root(_roots / "base-skills", LOCAL_SCOPE, store="skills") == _roots / "base-skills"
