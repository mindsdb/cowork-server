"""record_new_artifacts writes each slug's index row and owner row on its own:
a failed slug is rolled back and logged as an error naming it, and the slugs
after it are still recorded."""
from __future__ import annotations

import logging
from types import SimpleNamespace
from uuid import uuid4

import pytest
from sqlalchemy import event
from sqlalchemy.exc import OperationalError

from cowork.db import session as db_session
from cowork.db.scoped import LOCAL_SCOPE
from cowork.db.units import DatabaseBusy
from cowork.models.shared_resource import SharedResourceAttribution
from cowork.models.task_object import TaskObject
from cowork.services import artifact_ownership as ownership
from cowork.services import task_objects
from cowork.services.task_objects import record_new_artifacts

from test_artifact_ownership import make_conversation, make_project, project_root, scoped

pytestmark = pytest.mark.usefixtures("cleanup_tmp_projects")

FAILING = "a-report"
DRIVER_TEXT = "server closed the connection unexpectedly"


def _connection_lost() -> OperationalError:
    return OperationalError("INSERT", {}, Exception(DRIVER_TEXT))


def _fail_the_index_row(mapper, connection, target: TaskObject) -> None:
    """Fails the failing slug's insert inside its flush, as a dropped
    connection does, so the session needs a rollback before its next write."""
    if target.ref == FAILING:
        raise _connection_lost()


def _fail_the_owner_row(mapper, connection, target: SharedResourceAttribution) -> None:
    if target.resource_key.endswith(f"/{FAILING}"):
        raise _connection_lost()


def test_a_failed_slug_is_logged_as_an_error_and_the_next_slug_is_recorded(tmp_path, caplog, owned_logger):
    org_id, owner = "org-record", "user-record"
    project = make_project(tmp_path, org_id)
    conversation_id = make_conversation(project, owner)
    logged = owned_logger(task_objects.logger.name)

    event.listen(TaskObject, "before_insert", _fail_the_index_row)
    event.listen(SharedResourceAttribution, "before_insert", _fail_the_owner_row)
    try:
        with caplog.at_level(logging.WARNING, logger=task_objects.logger.name):
            with scoped(org_id) as session:
                record_new_artifacts(
                    session, conversation_id=conversation_id, project_id=project.id,
                    slugs=[FAILING, "b-report"], creator=owner,
                )
    finally:
        event.remove(TaskObject, "before_insert", _fail_the_index_row)
        event.remove(SharedResourceAttribution, "before_insert", _fail_the_owner_row)

    errors = [
        record for record in caplog.records
        if record.name == task_objects.logger.name and record.levelno == logging.ERROR
    ]
    assert len(errors) == 2, [record.getMessage() for record in errors]
    # The database filter replaces each message with its type and call site;
    # the ids travel as record attributes, which the console line renders.
    for record in errors:
        assert record.getMessage() == (
            f"Database operation failed: error_type=OperationalError sqlstate=unknown "
            f"site=task_objects.record_new_artifacts:{record.lineno}"
        )
        assert (record.project_id, record.conversation_id, record.artifact_slug) == (
            str(project.id), str(conversation_id), FAILING,
        )
        assert record.exc_info is None
        assert DRIVER_TEXT not in repr(record.__dict__)
    # The index write's line comes first in record_new_artifacts, then the owner's.
    assert errors[0].lineno < errors[1].lineno
    lines = [line for line in logged.output().splitlines() if "Database operation failed" in line]
    assert len(lines) == 2
    for line in lines:
        assert f"[Project:{project.id}][Conversation:{conversation_id}][Artifact:{FAILING!r}]" in line
        assert "b-report" not in line
    assert DRIVER_TEXT not in logged.output()

    with scoped(org_id, owner) as session:
        indexed = session.exec(
            session.select(TaskObject.ref).where(TaskObject.conversation_id == conversation_id)
        ).all()
        owners = ownership.resolve_artifact_owners(
            session, project_root(project), [FAILING, "b-report"],
        )
    assert indexed == ["b-report"]
    assert owners["b-report"].owner_user_id == owner
    assert owners[FAILING].unknown


def test_a_refused_index_session_names_the_conversation_project_and_slugs(monkeypatch, caplog, owned_logger):
    """The remote turn indexes on a session of its own. When the pool refuses
    it, the database filter keeps only the call site, so the ids ride the
    record."""
    conversation_id, project_id = str(uuid4()), str(uuid4())

    def refuse(uri):
        raise DatabaseBusy("no connection freed within POOL_TIMEOUT")

    monkeypatch.setattr(db_session, "get_open_session", refuse)
    logged = owned_logger(task_objects.logger.name, level=logging.WARNING)

    task_objects._index_new_slugs(
        SimpleNamespace(created_by="user-1"), conversation_id, project_id, ["sales-report", "chart"], LOCAL_SCOPE,
    )

    [record] = [
        record for record in caplog.records
        if record.name == task_objects.logger.name and record.levelno == logging.WARNING
    ]
    assert record.getMessage() == (
        f"Database operation failed: error_type=DatabaseBusy sqlstate=unknown "
        f"site=task_objects._index_new_slugs:{record.lineno}"
    )
    assert (record.conversation_id, record.project_id, record.artifact_slugs) == (
        conversation_id, project_id, ("sales-report", "chart"),
    )
    assert (
        f"[Project:{project_id}][Conversation:{conversation_id}][Artifacts:'sales-report', 'chart']: "
        "Database operation failed"
    ) in logged.output()
