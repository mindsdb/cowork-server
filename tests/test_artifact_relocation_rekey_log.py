"""A failed owner rekey names the moved folder at its new address.

The destination already has a folder with the same name, so the moved folder
takes a prefixed slug there. The record's project and slug name that folder,
not the destination's unrelated original.
"""
from __future__ import annotations

import logging
from uuid import uuid4

import pytest
from sqlalchemy import event
from sqlalchemy.exc import OperationalError
from sqlmodel import Session

from cowork.db.scoped import ScopedSession, TenantScope
from cowork.models.shared_resource import SharedResourceAttribution
from cowork.services import artifact_ownership as ownership
from cowork.services import task_objects
from test_artifact_ownership import make_conversation, make_project, project_root, scoped, write_artifact
from test_artifact_relocation_ownership import _engine, _index, _relocate

pytestmark = pytest.mark.usefixtures("cleanup_tmp_projects")

DRIVER_TEXT = "server closed the connection unexpectedly"
OWNER_UPDATE = f"UPDATE {SharedResourceAttribution.__tablename__} "


def _task_with_a_colliding_artifact(*, tmp_path, slugs: tuple[str, ...]):
    """An org-mode task whose owned artifacts are indexed in ``slugs`` order.

    The destination already holds its own ``report`` folder.
    """
    org_id, owner = str(uuid4()), str(uuid4())
    source, dest = make_project(tmp_path, org_id), make_project(tmp_path, org_id)
    conversation_id = make_conversation(source, owner)
    for slug in slugs:
        write_artifact(project_root(source).base, slug, conversation_id)
        _index(conversation_id, source.id, slug)
        with scoped(org_id) as session:
            ownership.record_artifact_owner(session, source.id, slug, owner, action="create")
    write_artifact(project_root(dest).base, "report")
    return org_id, owner, source, dest, conversation_id


def test_a_database_failure_on_a_rekey_logs_the_moved_folder_and_the_next_rekey_runs(
    tmp_path, caplog, owned_logger
):
    org_id, owner, source, dest, conversation_id = _task_with_a_colliding_artifact(
        tmp_path=tmp_path, slugs=("report", "notes")
    )
    moved_slug = f"{str(conversation_id)[:8]}-report"
    moved_key = ownership.artifact_resource_key(dest.id, moved_slug)
    owner_updates: list[str] = []

    def fail_the_first_owner_update(conn, cursor, statement, parameters, context, executemany):
        if statement.startswith(OWNER_UPDATE):
            owner_updates.append(str(parameters))
            if len(owner_updates) == 1:
                raise OperationalError(statement, parameters, Exception(DRIVER_TEXT))

    logged = owned_logger(task_objects.logger.name, level=logging.WARNING)
    event.listen(_engine(), "before_cursor_execute", fail_the_first_owner_update)
    try:
        moved = _relocate(org_id, owner, conversation_id, source.id, dest.id)
    finally:
        event.remove(_engine(), "before_cursor_execute", fail_the_first_owner_update)

    assert moved == 2
    # The colliding rekey failed first, and the one after it still ran.
    assert len(owner_updates) == 2
    assert moved_key in owner_updates[0]
    with Session(_engine()) as raw:
        session = ScopedSession(raw, TenantScope(org_mode=True, org_id=org_id, user_id=owner))
        assert ownership.resolve_artifact_owner(session, project_root(dest), "notes").owner_user_id == owner
        assert ownership.resolve_artifact_owner(session, project_root(source), "report").owner_user_id == owner
    [record] = [
        record for record in caplog.records
        if record.name == task_objects.logger.name and record.levelno == logging.WARNING
    ]
    assert (record.project_id, record.conversation_id, record.artifact_slug) == (
        str(dest.id), str(conversation_id), moved_slug,
    )
    assert record.getMessage().startswith("Database operation failed: error_type=OperationalError ")
    output = logged.output()
    assert f"[Project:{dest.id}][Conversation:{conversation_id}][Artifact:{moved_slug!r}]: " in output
    assert DRIVER_TEXT not in output


def test_a_failed_rekey_outside_the_database_names_both_slugs(tmp_path, monkeypatch, caplog, owned_logger):
    org_id, owner, source, dest, conversation_id = _task_with_a_colliding_artifact(tmp_path=tmp_path, slugs=("report",))

    def fail_the_rekey(*_a, **_k):
        raise RuntimeError("rekey failed")

    monkeypatch.setattr(ownership, "rekey_artifact_owner", fail_the_rekey)
    logged = owned_logger(task_objects.logger.name, level=logging.WARNING)

    assert _relocate(org_id, owner, conversation_id, source.id, dest.id) == 1

    moved_slug = f"{str(conversation_id)[:8]}-report"
    assert (project_root(dest).base / moved_slug).is_dir()
    [record] = [
        record for record in caplog.records
        if record.name == task_objects.logger.name and record.levelno == logging.WARNING
    ]
    assert (record.project_id, record.conversation_id, record.artifact_slug) == (
        str(dest.id), str(conversation_id), moved_slug,
    )
    assert record.getMessage() == f"Could not move the owner of artifact 'report' to project {dest.name!r}"
    assert f"[Project:{dest.id}][Conversation:{conversation_id}][Artifact:{moved_slug!r}]: " in logged.output()
