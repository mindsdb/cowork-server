"""The held_artifact_roots rows: the backfill finds every root that holds a
folder, the list trusts the rows only once it has finished, and each place
the server sees folders in a root records it."""
from __future__ import annotations

import errno
import logging
import threading
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from uuid import UUID, uuid4

import pytest
from sqlmodel import Session, select

from cowork.common.settings.app_settings import get_app_settings
from cowork.db.session import get_engine
from cowork.models.held_artifact_root import HeldArtifactRoot
from cowork.models.setting import Setting
from cowork.services import held_artifact_roots
from cowork.services.artifact_autopublish import _record_listing
from cowork.services.artifact_identity import ListedIds
from cowork.services.held_artifact_roots import (
    SENTINEL_KEY,
    HeldRoot,
    held_roots,
    hold_root,
    run_held_root_backfill,
)
from cowork.services.task_objects import record_new_artifacts
from test_artifact_ownership import make_conversation, make_project, project_root, scoped, write_artifact
from test_artifact_relocation_ownership import _index, _relocate

pytestmark = pytest.mark.usefixtures("cleanup_tmp_projects")

OWNER = "11111111-1111-4111-8111-111111111111"


def _engine():
    return get_engine(get_app_settings().database.uri)


def _no_lock(_engine, _key):
    return nullcontext(True)


def _sentinel_present() -> bool:
    with Session(_engine()) as raw:
        return raw.exec(select(Setting).where(Setting.key == SENTINEL_KEY, Setting.scope.is_(None))).first() is not None


@pytest.fixture
def backfill_pending():
    """No pass has written the sentinel; one a test writes goes after it, as
    the database is shared across modules."""
    yield
    with Session(_engine()) as raw:
        for row in raw.exec(select(Setting).where(Setting.key == SENTINEL_KEY)).all():
            raw.delete(row)
        raw.commit()


def _rows(org_id: str) -> set[HeldRoot]:
    with Session(_engine()) as raw:
        rows = raw.exec(select(HeldArtifactRoot).where(HeldArtifactRoot.org_id == org_id)).all()
        return {HeldRoot(row.project_id, row.conversation_id) for row in rows}


def _workspace(project, conversation_id) -> Path:
    return Path(project.path) / "conversations" / str(conversation_id)


def test_the_backfill_records_each_root_that_holds_a_folder(tmp_path, org_deployment, backfill_pending):
    org_id = str(uuid4())
    held, empty, refused = (make_project(tmp_path, org_id) for _ in range(3))
    write_artifact(project_root(held).base, "report")
    owned = make_conversation(held, OWNER)
    write_artifact(_workspace(held, owned) / ".anton" / "artifacts", "old-report")
    # Workspaces the list reads nothing from: an empty artifacts folder, none
    # at all, and one whose name is no conversation id.
    (_workspace(held, make_conversation(held, OWNER)) / ".anton" / "artifacts").mkdir(parents=True)
    (_workspace(held, make_conversation(held, OWNER)) / "notes.txt").parent.mkdir(parents=True)
    write_artifact(Path(held.path) / "conversations" / "not-a-conversation" / ".anton" / "artifacts", "orphan")
    # Shared roots holding only what the list skips: a dot folder and a link.
    project_root(empty).base.joinpath(".locks").mkdir()
    (project_root(empty).base / "linked").symlink_to(project_root(held).base / "report")
    # A shared root that is itself a link.
    (Path(refused.path) / ".anton").mkdir()
    (Path(refused.path) / ".anton" / "artifacts").symlink_to(project_root(held).base)

    summary = run_held_root_backfill(try_lock=_no_lock, project_ids=[held.id, empty.id, refused.id])

    assert summary is not None and (summary.held, summary.unread) == (2, 0)
    assert _rows(org_id) == {HeldRoot(held.id, None), HeldRoot(held.id, owned)}
    assert _sentinel_present()
    # Once per environment.
    assert run_held_root_backfill(try_lock=_no_lock, project_ids=[held.id]) is None


@pytest.mark.parametrize("failure", [errno.EIO, errno.ESTALE])
def test_a_root_the_backfill_cannot_read_leaves_no_sentinel(
    tmp_path, org_deployment, backfill_pending, monkeypatch, caplog, failure
):
    """A read error is not an empty root: the list would then miss its
    cards for good. The pass goes on, and the next start walks again."""
    org_id = str(uuid4())
    unreadable, readable = make_project(tmp_path, org_id), make_project(tmp_path, org_id)
    write_artifact(project_root(unreadable).base, "report")
    write_artifact(project_root(readable).base, "report")
    real_scandir = held_artifact_roots.dir_scandir

    def scandir(root):
        if Path(str(root.path)) == project_root(unreadable).base:
            raise OSError(failure, "the file server did not answer")
        return real_scandir(root)

    with monkeypatch.context() as failing, caplog.at_level(logging.WARNING, logger=held_artifact_roots.logger.name):
        failing.setattr(held_artifact_roots, "dir_scandir", scandir)
        summary = run_held_root_backfill(try_lock=_no_lock, project_ids=[unreadable.id, readable.id])

    assert summary is not None and summary.unread == 1
    assert _rows(org_id) == {HeldRoot(readable.id, None)}
    assert not _sentinel_present()
    [unread] = [
        record for record in caplog.records
        if record.name == held_artifact_roots.logger.name
        and getattr(record, "event", None) == "held_artifact_roots_backfill_unread"
    ]
    assert unread.project_id == str(unreadable.id)

    run_held_root_backfill(try_lock=_no_lock, project_ids=[unreadable.id, readable.id])
    assert _rows(org_id) == {HeldRoot(unreadable.id, None), HeldRoot(readable.id, None)}
    assert _sentinel_present()


def test_a_root_the_server_may_not_read_holds_nothing_and_the_pass_finishes(
    tmp_path, org_deployment, backfill_pending, monkeypatch
):
    """The list could show nothing below it either."""
    org_id = str(uuid4())
    project = make_project(tmp_path, org_id)
    write_artifact(project_root(project).base, "report")

    def refused(_root):
        raise PermissionError(errno.EACCES, "permission denied")

    monkeypatch.setattr(held_artifact_roots, "dir_scandir", refused)
    summary = run_held_root_backfill(try_lock=_no_lock, project_ids=[project.id])

    assert summary is not None and (summary.held, summary.unread) == (0, 0)
    assert _rows(org_id) == set()
    assert _sentinel_present()


def test_a_stopped_backfill_writes_no_sentinel(tmp_path, org_deployment, backfill_pending):
    project = make_project(tmp_path, str(uuid4()))
    stop = threading.Event()
    stop.set()

    assert run_held_root_backfill(try_lock=_no_lock, project_ids=[project.id], stop=stop) is None
    assert not _sentinel_present()


def test_the_desktop_runs_no_backfill(tmp_path, backfill_pending):
    project = make_project(tmp_path, None)
    write_artifact(project_root(project).base, "report")

    assert run_held_root_backfill(try_lock=_no_lock, project_ids=[project.id]) is None
    assert not _sentinel_present()


def test_the_list_trusts_the_rows_only_once_the_backfill_has_finished(tmp_path, org_deployment, backfill_pending):
    org_id, other_org = str(uuid4()), str(uuid4())
    project, elsewhere = make_project(tmp_path, org_id), make_project(tmp_path, other_org)
    with scoped(org_id) as session:
        hold_root(session, project_id=project.id)
    with scoped(other_org) as session:
        hold_root(session, project_id=elsewhere.id)

    with scoped(org_id) as session:
        assert held_roots(session, project_ids=[project.id, elsewhere.id]) is None
    run_held_root_backfill(try_lock=_no_lock, project_ids=[])
    with scoped(org_id) as session:
        # Another organization's row is not this scope's, whatever it names.
        assert held_roots(session, project_ids=[project.id, elsewhere.id]) == {HeldRoot(project.id, None)}
    with scoped(None) as session:
        assert held_roots(session, project_ids=[project.id]) is None


def test_a_root_is_recorded_once_and_its_legacy_roots_beside_it(tmp_path):
    org_id = str(uuid4())
    project = make_project(tmp_path, org_id)
    conversation_id = make_conversation(project, OWNER)

    with scoped(org_id) as session:
        for _ in range(2):
            hold_root(session, project_id=project.id)
            hold_root(session, project_id=project.id, conversation_id=conversation_id)

    assert _rows(org_id) == {HeldRoot(project.id, None), HeldRoot(project.id, conversation_id)}


@pytest.mark.parametrize("conversation", [False, True], ids=["shared", "legacy"])
def test_a_root_another_writer_recorded_first_is_left_as_it_is(tmp_path, conversation):
    """Two turn ends in one project can both find no row and both insert:
    the unique index refuses the second, and its writer goes on."""
    org_id = str(uuid4())
    project = make_project(tmp_path, org_id)
    conversation_id = make_conversation(project, OWNER) if conversation else None
    with scoped(org_id) as session:
        hold_root(session, project_id=project.id, conversation_id=conversation_id)

    with scoped(org_id) as session:
        # The other writer committed after this one's check.
        session.exec = lambda _query: SimpleNamespace(first=lambda: None)
        hold_root(session, project_id=project.id, conversation_id=conversation_id)

    assert _rows(org_id) == {HeldRoot(project.id, conversation_id)}


@pytest.mark.parametrize(
    ("slugs", "listed", "held"),
    [
        (["report"], None, True),
        ([], ListedIds(ids={"report": "a" * 32}, unreadable=frozenset()), True),
        ([], ListedIds(ids={}, unreadable=frozenset({"report"})), True),
        ([], ListedIds(ids={}, unreadable=frozenset()), False),
    ],
    ids=["created", "listed", "listed-unreadable", "nothing"],
)
def test_a_turn_end_records_the_shared_root_it_found_folders_in(tmp_path, slugs, listed, held):
    org_id = str(uuid4())
    project = make_project(tmp_path, org_id)
    conversation_id = make_conversation(project, OWNER)

    with scoped(org_id) as session:
        record_new_artifacts(
            session, conversation_id=conversation_id, project_id=project.id, slugs=slugs, creator=OWNER,
            listed_ids=listed,
        )

    assert _rows(org_id) == ({HeldRoot(project.id, None)} if held else set())


def test_a_desktop_turn_end_records_no_root(tmp_path):
    project = make_project(tmp_path, None)
    conversation_id = make_conversation(project, None)

    with scoped(None) as session:
        record_new_artifacts(
            session, conversation_id=conversation_id, project_id=project.id, slugs=["report"], creator=None,
        )

    with Session(_engine()) as raw:
        assert raw.exec(select(HeldArtifactRoot).where(HeldArtifactRoot.project_id == project.id)).first() is None


@pytest.mark.parametrize(
    ("listing", "held"),
    [
        (ListedIds(ids={"report": "a" * 32}, unreadable=frozenset()), True),
        (ListedIds(ids={}, unreadable=frozenset()), False),
    ],
    ids=["a-folder", "empty"],
)
def test_the_self_heals_listing_records_the_root_when_it_holds_a_folder(tmp_path, listing, held):
    org_id = str(uuid4())
    project = make_project(tmp_path, org_id)

    with scoped(org_id) as session:
        _record_listing(session, project_id=UUID(str(project.id)), listing=listing)

    assert _rows(org_id) == ({HeldRoot(project.id, None)} if held else set())


def test_a_task_move_records_the_destinations_shared_root(tmp_path):
    org_id, owner = str(uuid4()), str(uuid4())
    source, dest = make_project(tmp_path, org_id), make_project(tmp_path, org_id)
    conversation_id = make_conversation(source, owner)
    write_artifact(project_root(source).base, "report", conversation_id)
    _index(conversation_id, source.id, "report")
    with scoped(org_id) as session:
        from cowork.services import artifact_ownership as ownership

        ownership.record_artifact_owner(session, source.id, "report", owner, action="create")
        hold_root(session, project_id=source.id)

    assert _relocate(org_id, owner, conversation_id, source.id, dest.id) == 1

    assert _rows(org_id) == {HeldRoot(source.id, None), HeldRoot(dest.id, None)}
