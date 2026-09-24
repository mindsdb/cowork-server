"""One-time owner backfill for project-root artifacts (ENG-2961, D5)."""
from __future__ import annotations

import json
import os
from contextlib import contextmanager
from uuid import uuid4

import pytest
import sqlalchemy as sa
from sqlmodel import Session, select

from cowork.common.settings.app_settings import get_app_settings
from cowork.db.scoped import ScopedSession, TenantScope
from cowork.db.session import get_engine
from cowork.models.setting import Setting
from cowork.models.task_object import TaskObject
from cowork.services import artifact_owner_backfill as backfill
from cowork.services import artifact_ownership as ownership
from cowork.services.task_objects import KIND_ARTIFACT
from test_artifact_ownership import (
    legacy_root,
    make_conversation,
    make_project,
    project_root,
    write_artifact,
)


def _engine():
    return get_engine(get_app_settings().database.uri)


def _drop_sentinel():
    with Session(_engine()) as raw:
        for row in raw.exec(select(Setting).where(Setting.key == backfill.SENTINEL_KEY)).all():
            raw.delete(row)
        raw.commit()


@pytest.fixture(autouse=True)
def org_deployment(monkeypatch):
    monkeypatch.setenv("COWORK_TENANCY_MODE", "org")
    get_app_settings.cache_clear()
    _drop_sentinel()
    yield
    _drop_sentinel()
    get_app_settings.cache_clear()


@pytest.fixture(autouse=True)
def cleanup_test_projects(tmp_path):
    """Clean up projects (and their dependent rows) created by this test to
    avoid database pollution: the session-scoped test DB is shared across
    modules, and a leaked Project row (especially one with a real
    ``.anton/artifacts`` dir, as this module's tests create) can break other
    modules' tests that scan or query all projects."""
    from sqlmodel import select
    from cowork.models.conversation import Conversation
    from cowork.models.project import Project
    from cowork.models.shared_resource import SharedResourceAttribution

    yield

    with Session(_engine()) as session:
        projects = session.exec(select(Project)).all()
        leaked = [p for p in projects if tmp_path.as_posix() in p.path]
        leaked_ids = {p.id for p in leaked}
        if leaked_ids:
            for row in session.exec(select(TaskObject)).all():
                if row.project_id in leaked_ids:
                    session.delete(row)
            for row in session.exec(select(Conversation)).all():
                if row.project_id in leaked_ids:
                    session.delete(row)
            for row in session.exec(select(SharedResourceAttribution)).all():
                key = row.resource_key or ""
                if any(str(pid) in key for pid in leaked_ids):
                    session.delete(row)
        for project in leaked:
            session.delete(project)
        session.commit()


def _owner(org_id, source, slug):
    with Session(_engine()) as raw:
        session = ScopedSession(raw, TenantScope(org_mode=True, org_id=org_id))
        return ownership.resolve_artifact_owner(session, source, slug)


def _index(conversation_id, project_id, slug):
    with Session(_engine()) as raw:
        raw.add(TaskObject(conversation_id=conversation_id, project_id=project_id, kind=KIND_ARTIFACT, ref=slug))
        raw.commit()


def _run(*projects, **kwargs):
    # Scoped to this test's own projects: the suite's database is shared, and an
    # unfiltered pass would write owner rows under other modules' projects.
    return backfill.run_artifact_owner_backfill(
        project_ids={project.id for project in projects}, **kwargs
    )


@pytest.fixture
def org(tmp_path):
    org_id, owner = str(uuid4()), str(uuid4())
    project = make_project(tmp_path, org_id)
    return org_id, owner, project, project_root(project)


def test_provenance_in_the_same_project_records_the_owner(org):
    org_id, owner, project, source = org
    write_artifact(source.base, "a", make_conversation(project, owner))

    summary = _run(project)

    assert summary is not None
    assert _owner(org_id, source, "a") == ownership.OwnerResolution(owner, "recorded")
    assert (str(project.id), "a") not in summary.unknown


def test_foreign_or_missing_provenance_falls_back_to_task_objects(org, tmp_path):
    org_id, owner, project, source = org
    other_project = make_project(tmp_path, org_id)
    conversation_id = make_conversation(project, owner)
    write_artifact(source.base, "foreign", make_conversation(other_project, str(uuid4())))
    write_artifact(source.base, "missing", uuid4())
    for slug in ("foreign", "missing"):
        _index(conversation_id, project.id, slug)

    _run(project, other_project)

    assert _owner(org_id, source, "foreign").owner_user_id == owner
    assert _owner(org_id, source, "missing").owner_user_id == owner


def test_ambiguous_or_absent_task_objects_stay_unknown(org):
    org_id, owner, project, source = org
    write_artifact(source.base, "none")
    write_artifact(source.base, "two")
    _index(make_conversation(project, owner), project.id, "two")
    _index(make_conversation(project, str(uuid4())), project.id, "two")

    summary = _run(project)

    assert _owner(org_id, source, "none").unknown
    assert _owner(org_id, source, "two").unknown
    assert {(str(project.id), "none"), (str(project.id), "two")} <= set(summary.unknown)


def test_symlinked_metadata_falls_back_to_task_objects(org, tmp_path):
    org_id, owner, project, source = org
    decoy = write_artifact(tmp_path / "decoy", "decoy", make_conversation(project, str(uuid4())))
    folder = source.base / "linked"
    folder.mkdir()
    os.symlink(decoy / "metadata.json", folder / "metadata.json")
    _index(make_conversation(project, owner), project.id, "linked")

    _run(project)

    assert _owner(org_id, source, "linked").owner_user_id == owner


def test_existing_rows_and_legacy_roots_are_left_alone(org):
    org_id, owner, project, source = org
    first = str(uuid4())
    with Session(_engine()) as raw:
        session = ScopedSession(raw, TenantScope(org_mode=True, org_id=org_id))
        ownership.record_artifact_owner(session, project.id, "kept", first, action="create")
    write_artifact(source.base, "kept", make_conversation(project, owner))
    legacy = legacy_root(project, str(make_conversation(project, owner)))
    write_artifact(legacy.base, "old", uuid4())

    summary = _run(project)

    assert _owner(org_id, source, "kept").owner_user_id == first
    assert (str(project.id), "old") not in summary.unknown


def test_rows_are_org_scoped(org):
    org_id, owner, project, source = org
    write_artifact(source.base, "a", make_conversation(project, owner))

    _run(project)

    with Session(_engine()) as raw:
        session = ScopedSession(raw, TenantScope(org_mode=True, org_id=str(uuid4())))
        assert ownership.resolve_artifact_owner(session, source, "a").unknown


def test_projects_without_an_org_are_skipped_and_counted(tmp_path):
    orphan = make_project(tmp_path, None)
    summary = _run(orphan)
    assert summary.skipped_projects == 1


def test_sentinel_makes_it_run_once(org):
    org_id, owner, project, source = org
    assert _run(project) is not None
    write_artifact(source.base, "later", make_conversation(project, owner))
    assert _run(project) is None
    assert _owner(org_id, source, "later").unknown


def test_a_replica_without_the_lock_skips(org):
    org_id, owner, project, source = org
    write_artifact(source.base, "a", make_conversation(project, owner))

    @contextmanager
    def held_elsewhere(_engine):
        yield False

    assert _run(project, try_lock=held_elsewhere) is None
    assert _owner(org_id, source, "a").unknown


def test_an_aborted_pass_writes_no_sentinel(org, monkeypatch):
    def boom(*_args, **_kwargs):
        raise RuntimeError("database went away")

    monkeypatch.setattr(backfill, "_backfill_project", boom)
    with pytest.raises(RuntimeError):
        _run(org[2])
    with Session(_engine()) as raw:
        assert raw.exec(select(Setting).where(Setting.key == backfill.SENTINEL_KEY)).first() is None


def test_a_transient_db_error_aborts_the_pass_and_writes_no_sentinel(org, monkeypatch):
    """A short DB outage mid-walk must not mark the rest of the walk `unknown`
    forever: the pass has to abort so the sentinel is not written and the next
    start retries the whole project (ENG-2961, F2)."""
    write_artifact(org[3].base, "no-provenance")

    def boom(*_args, **_kwargs):
        raise sa.exc.OperationalError("stmt", {}, Exception("db down"))

    monkeypatch.setattr(backfill, "_owner_from_task_objects", boom)

    with pytest.raises(sa.exc.OperationalError):
        _run(org[2])

    with Session(_engine()) as raw:
        assert raw.exec(select(Setting).where(Setting.key == backfill.SENTINEL_KEY)).first() is None


def test_local_mode_never_runs(monkeypatch):
    monkeypatch.setenv("COWORK_TENANCY_MODE", "local")
    get_app_settings.cache_clear()
    assert backfill.run_artifact_owner_backfill() is None


@pytest.mark.asyncio
async def test_lifespan_task_swallows_failures(monkeypatch):
    from cowork import server

    def boom(*_args, **_kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(backfill, "run_artifact_owner_backfill", boom)
    await server._run_artifact_owner_backfill()
