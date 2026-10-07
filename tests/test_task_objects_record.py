"""record_new_artifacts writes each slug's index row and owner row on its own:
a failed slug is rolled back and logged as an error naming it, and the slugs
after it are still recorded."""
from __future__ import annotations

import logging

import pytest
from sqlalchemy import event
from sqlalchemy.exc import OperationalError

from cowork.models.shared_resource import SharedResourceAttribution
from cowork.models.task_object import TaskObject
from cowork.services import artifact_ownership as ownership
from cowork.services import task_objects
from cowork.services.task_objects import record_new_artifacts

from test_artifact_ownership import make_conversation, make_project, project_root, scoped

pytestmark = pytest.mark.usefixtures("cleanup_tmp_projects")

FAILING = "a-report"


def _connection_lost() -> OperationalError:
    return OperationalError("INSERT", {}, Exception("server closed the connection unexpectedly"))


def _fail_the_index_row(mapper, connection, target: TaskObject) -> None:
    """Fails the failing slug's insert inside its flush, as a dropped
    connection does, so the session needs a rollback before its next write."""
    if target.ref == FAILING:
        raise _connection_lost()


def _fail_the_owner_row(mapper, connection, target: SharedResourceAttribution) -> None:
    if target.resource_key.endswith(f"/{FAILING}"):
        raise _connection_lost()


def test_a_failed_slug_is_logged_as_an_error_and_the_next_slug_is_recorded(tmp_path, caplog):
    org_id, owner = "org-record", "user-record"
    project = make_project(tmp_path, org_id)
    conversation_id = make_conversation(project, owner)

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
    for record in errors:
        message = record.getMessage()
        assert repr(FAILING) in message and "b-report" not in message
        assert str(conversation_id) in message and str(project.id) in message
        assert record.exc_info is not None
    assert "index" in errors[0].getMessage() and "owner" in errors[1].getMessage()

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
