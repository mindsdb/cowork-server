"""One-time owner backfill for project-root artifacts (ENG-2961, D5)."""
from __future__ import annotations

import json
import logging
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


PRIVATE_SQL = "SELECT 'private_backfill_sql_marker'"
PRIVATE_BIND = "private_backfill_bind_marker"
PRIVATE_DETAIL = "private_backfill_driver_detail_marker"


def _assert_safe_database_log(caplog, logged, error_type, level, *, project_id, slug):
    records = [record for record in caplog.records
               if record.name == backfill.__name__
               and record.getMessage().startswith("Database operation failed:")]
    assert len(records) == 1
    record = records[0]
    assert record.levelno == level
    site = f"artifact_owner_backfill._backfill_project:{record.lineno}"
    assert record.getMessage() == (
        f"Database operation failed: error_type={error_type} sqlstate=unknown site={site}"
    )
    assert record.args == (error_type, "unknown", site)
    # The project and slug the message used to carry travel as attributes.
    assert (record.project_id, record.artifact_slug) == (str(project_id), slug)
    assert record.exc_info is None and record.exc_text is None
    for captured in caplog.records:
        assert all(value not in repr(captured.__dict__)
                   for value in (PRIVATE_SQL, PRIVATE_BIND, PRIVATE_DETAIL))
    [line] = [line for line in logged.output().splitlines() if "Database operation failed" in line]
    assert f"[Project:{project_id}][Artifact:{slug!r}]" in line
    assert all(value not in logged.output() for value in (PRIVATE_SQL, PRIVATE_BIND, PRIVATE_DETAIL))


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


pytestmark = pytest.mark.usefixtures("cleanup_tmp_projects")


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


def test_a_data_error_on_one_artifact_marks_only_it_unknown(org, monkeypatch, caplog, owned_logger):
    """Only connection-level errors abort the pass. A `DataError` (like any
    statement error) is about one artifact: that one stays unknown and the rest
    of the walk, and the sentinel, still happen."""
    org_id, owner, project, source = org
    write_artifact(source.base, "bad")
    write_artifact(source.base, "good", make_conversation(project, owner))
    real = backfill._owner_from_task_objects

    def data_error_for_bad(session, project_id, slug):
        if slug == "bad":
            raise sa.exc.DataError(PRIVATE_SQL, {"value": PRIVATE_BIND}, Exception(PRIVATE_DETAIL))
        return real(session, project_id, slug)

    monkeypatch.setattr(backfill, "_owner_from_task_objects", data_error_for_bad)
    logged = owned_logger(backfill.__name__)

    with caplog.at_level("WARNING", logger=backfill.__name__):
        summary = _run(project)

    assert summary is not None
    assert (str(project.id), "bad") in summary.unknown
    assert _owner(org_id, source, "good").owner_user_id == owner
    _assert_safe_database_log(caplog, logged, "DataError", logging.WARNING, project_id=project.id, slug="bad")
    with Session(_engine()) as raw:
        assert raw.exec(select(Setting).where(Setting.key == backfill.SENTINEL_KEY)).first() is not None


def test_an_interface_error_aborts_the_pass_and_logs_safe_metadata(org, monkeypatch, caplog, owned_logger):
    write_artifact(org[3].base, "no-provenance")
    failure = sa.exc.InterfaceError(PRIVATE_SQL, {"value": PRIVATE_BIND}, Exception(PRIVATE_DETAIL))

    def boom(*_args, **_kwargs):
        raise failure

    monkeypatch.setattr(backfill, "_owner_from_task_objects", boom)
    logged = owned_logger(backfill.__name__)

    with caplog.at_level("WARNING", logger=backfill.__name__):
        with pytest.raises(sa.exc.InterfaceError) as caught:
            _run(org[2])

    assert caught.value is failure
    assert caught.value.statement == PRIVATE_SQL
    assert caught.value.params == {"value": PRIVATE_BIND}
    assert caught.value.orig.args == (PRIVATE_DETAIL,)
    _assert_safe_database_log(
        caplog, logged, "InterfaceError", logging.ERROR, project_id=org[2].id, slug="no-provenance",
    )
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
