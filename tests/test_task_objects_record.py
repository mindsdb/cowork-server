"""record_new_artifacts writes each slug's index row and owner row on its own:
a failed slug is rolled back and logged as an error naming it, and the slugs
after it are still recorded."""
from __future__ import annotations
from tests._logging_assertions import assert_log_fields

import logging

import pytest
from sqlalchemy import event
from sqlalchemy.exc import OperationalError

from cowork.models.shared_resource import SharedResourceAttribution
from cowork.models.task_object import TaskObject
from cowork.services import artifact_ownership as ownership
from cowork.services import task_objects
from cowork.services.artifact_identity import ListedIds
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
        assert_log_fields(line, project_id=str(project.id), conversation_id=str(conversation_id), artifact_slug=FAILING)
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


def _missing_owner_warnings(caplog) -> list[logging.LogRecord]:
    return [
        record for record in caplog.records
        if record.name == task_objects.logger.name and record.levelno == logging.WARNING
        and record.getMessage().startswith("artifact owner not recorded")
    ]


def test_a_conversation_with_no_creator_warns_only_when_a_created_folder_needs_an_owner(
    tmp_path, caplog, owned_logger,
):
    """A channel conversation has no creator. Its turn end checks the folders
    the diff listed even when it created none; that call leaves no artifact
    without an owner, so it logs nothing. A folder it created is left
    without one, so that call logs the missing owner once."""
    org_id = "org-no-creator"
    project = make_project(tmp_path, org_id)
    conversation_id = make_conversation(project, None)
    owned_logger(task_objects.logger.name, level=logging.WARNING)

    with scoped(org_id) as session:
        record_new_artifacts(
            session, conversation_id=conversation_id, project_id=project.id,
            slugs=[], creator=None, listed_ids=ListedIds(ids={"brief": "id-of-brief"}, unreadable=frozenset()),
        )
    assert _missing_owner_warnings(caplog) == []

    with scoped(org_id) as session:
        record_new_artifacts(
            session, conversation_id=conversation_id, project_id=project.id,
            slugs=["made"], creator=None,
        )
    assert [record.getMessage() for record in _missing_owner_warnings(caplog)] == [
        f"artifact owner not recorded: conversation {conversation_id} has no creator",
    ]


def test_a_failed_check_of_the_listed_folders_raises_once_the_created_rows_are_written(tmp_path, monkeypatch):
    """The turn end listed every folder, and the check of their locations
    fails after the created folder's index and owner rows are written.
    record_new_artifacts raises ListedLocationsUnchecked, so its caller asks
    for the project's self-heal, and the rows written before the check stay
    committed though the unit then rolls back."""
    from cowork.db.units import unit_session
    from cowork.services import artifact_locations

    org_id, owner = "org-listed-check", "user-listed-check"
    project = make_project(tmp_path, org_id)
    conversation_id = make_conversation(project, owner)

    def check(session, **kwargs):
        raise _connection_lost()

    monkeypatch.setattr(artifact_locations, "record_listed_folders", check)
    with scoped(org_id) as session:
        scope = session.scope
    with pytest.raises(task_objects.ListedLocationsUnchecked) as raised:
        with unit_session(scope=scope) as session:
            record_new_artifacts(
                session, conversation_id=conversation_id, project_id=project.id,
                slugs=["made"], creator=owner,
                listed_ids=ListedIds(ids={"made": "id-of-made"}, unreadable=frozenset()),
            )

    assert isinstance(raised.value.__cause__, OperationalError)
    with scoped(org_id, owner) as session:
        indexed = session.exec(
            session.select(TaskObject.ref).where(TaskObject.conversation_id == conversation_id)
        ).all()
        owners = ownership.resolve_artifact_owners(session, project_root(project), ["made"])
    assert indexed == ["made"]
    assert owners["made"].owner_user_id == owner
