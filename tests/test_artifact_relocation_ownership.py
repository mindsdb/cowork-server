"""Moving a task moves only the artifacts its creator owns (ENG-2961, D8)."""
from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import pytest
from sqlmodel import Session, select

from cowork.common.settings.app_settings import get_app_settings
from cowork.db.scoped import ScopedSession, TenantScope
from cowork.db.session import get_engine
from cowork.models.conversation import Conversation
from cowork.models.project import Project
from cowork.models.task_object import TaskObject
from cowork.services import artifact_ownership as ownership
from cowork.services.task_objects import KIND_ARTIFACT, TaskObjectService
from test_artifact_ownership import (
    make_conversation,
    make_project,
    project_root,
    write_artifact,
)


def _engine():
    return get_engine(get_app_settings().database.uri)


def _index(conversation_id, project_id, slug):
    with Session(_engine()) as raw:
        raw.add(TaskObject(conversation_id=conversation_id, project_id=project_id, kind=KIND_ARTIFACT, ref=slug))
        raw.commit()


def _rows(conversation_id):
    with Session(_engine()) as raw:
        return {
            row.ref: row.project_id
            for row in raw.exec(select(TaskObject).where(TaskObject.conversation_id == conversation_id)).all()
        }


def _relocate(org_id, user_id, conversation_id, source_id, dest_id) -> int:
    with Session(_engine()) as raw:
        session = ScopedSession(raw, TenantScope(org_mode=True, org_id=org_id, user_id=user_id))
        conversation = session.get(Conversation, conversation_id)
        return TaskObjectService(session).relocate_to_project(
            conversation, session.get(Project, source_id), session.get(Project, dest_id)
        )


@pytest.fixture(autouse=True)
def cleanup_test_projects(tmp_path):
    """Delete Project (and their TaskObject) rows created by this module's
    tests under this test's tmp_path. Autouse fixtures in
    test_artifact_ownership.py do not carry over through import, and this
    module creates its own Project/Conversation/TaskObject rows against the
    shared session-scoped test DB, so it needs equivalent teardown to avoid
    leaking rows into other test modules (see test_artifact_roots.py)."""
    yield

    with Session(_engine()) as session:
        projects = session.exec(select(Project)).all()
        stale = [p for p in projects if tmp_path.as_posix() in p.path]
        stale_ids = {p.id for p in stale}
        if stale_ids:
            for row in session.exec(
                select(TaskObject).where(TaskObject.project_id.in_(stale_ids))
            ).all():
                session.delete(row)
        for project in stale:
            session.delete(project)
        session.commit()


def test_relocation_moves_owned_artifacts_and_rekeys_their_owner(tmp_path):
    org_id, owner, other = str(uuid4()), str(uuid4()), str(uuid4())
    source, dest = make_project(tmp_path, org_id), make_project(tmp_path, org_id)
    conversation_id = make_conversation(source, owner)
    root = project_root(source)
    for slug in ("mine", "foreign", "orphan"):
        write_artifact(root.base, slug, conversation_id)
        _index(conversation_id, source.id, slug)
    with Session(_engine()) as raw:
        session = ScopedSession(raw, TenantScope(org_mode=True, org_id=org_id))
        ownership.record_artifact_owner(session, source.id, "mine", owner, action="create")
        ownership.record_artifact_owner(session, source.id, "foreign", other, action="create")

    moved = _relocate(org_id, owner, conversation_id, source.id, dest.id)

    assert moved == 1
    dest_root = project_root(dest)
    assert (dest_root.base / "mine").is_dir()
    assert (root.base / "foreign").is_dir()
    assert (root.base / "orphan").is_dir()
    rows = _rows(conversation_id)
    assert rows["mine"] == dest.id
    assert rows["foreign"] == source.id
    assert rows["orphan"] == source.id
    with Session(_engine()) as raw:
        session = ScopedSession(raw, TenantScope(org_mode=True, org_id=org_id, user_id=owner))
        assert ownership.resolve_artifact_owner(session, dest_root, "mine").owner_user_id == owner
        assert ownership.resolve_artifact_owner(session, root, "mine").unknown


def test_org_mode_reconcile_does_not_index_from_rewritten_provenance(tmp_path):
    org_id, owner = str(uuid4()), str(uuid4())
    source, dest = make_project(tmp_path, org_id), make_project(tmp_path, org_id)
    conversation_id = make_conversation(source, owner)
    root = project_root(source)
    # A co-member's folder whose provenance was rewritten to name this task.
    write_artifact(root.base, "planted", conversation_id)

    moved = _relocate(org_id, owner, conversation_id, source.id, dest.id)

    assert moved == 0
    assert (root.base / "planted").is_dir()
    assert _rows(conversation_id) == {}


def test_stale_owner_row_at_the_destination_is_replaced(tmp_path):
    org_id, owner, stale = str(uuid4()), str(uuid4()), str(uuid4())
    source, dest = make_project(tmp_path, org_id), make_project(tmp_path, org_id)
    conversation_id = make_conversation(source, owner)
    root = project_root(source)
    write_artifact(root.base, "mine", conversation_id)
    _index(conversation_id, source.id, "mine")
    with Session(_engine()) as raw:
        session = ScopedSession(raw, TenantScope(org_mode=True, org_id=org_id))
        ownership.record_artifact_owner(session, source.id, "mine", owner, action="create")
        # Left behind by a delete whose attribution cleanup failed.
        ownership.record_artifact_owner(session, dest.id, "mine", stale, action="create")

    assert _relocate(org_id, owner, conversation_id, source.id, dest.id) == 1

    with Session(_engine()) as raw:
        session = ScopedSession(raw, TenantScope(org_mode=True, org_id=org_id, user_id=owner))
        assert ownership.resolve_artifact_owner(
            session, project_root(dest), "mine"
        ).owner_user_id == owner
