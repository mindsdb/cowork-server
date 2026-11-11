"""The held_artifact_roots rows: the backfill finds every root that holds a
folder, the list trusts the rows only once it has walked every project, and
each place the server sees folders in a root records it."""
from __future__ import annotations

import asyncio
import errno
import logging
import os
import shutil
import threading
from contextlib import contextmanager, nullcontext
from pathlib import Path
from types import SimpleNamespace
from uuid import UUID, uuid4

import pytest
from sqlalchemy import delete
from sqlmodel import Session, select

from cowork import server
from cowork.common.settings.app_settings import get_app_settings
from cowork.db import units
from cowork.db.scoped import SYSTEM_SCOPE, ScopedSession
from cowork.db.session import get_engine
from cowork.models.held_artifact_root import HeldArtifactRoot
from cowork.models.project import Project
from cowork.models.setting import Setting
from cowork.services import artifact_location_backfill, artifact_owner_backfill, held_artifact_roots
from cowork.services.artifact_autopublish import _record_listing
from cowork.services.artifact_identity import ListedIds
from cowork.services.artifact_owner_backfill import BackfillProgress, read_progress, write_sentinel
from cowork.services.artifact_roots import plan_scope_roots, readable_sources, with_held_roots
from cowork.services.held_artifact_roots import (
    PROGRESS_KEY,
    SENTINEL_KEY,
    HeldRoot,
    HeldRootSentinel,
    held_roots,
    hold_root,
    read_sentinel,
    run_held_root_backfill,
)
from cowork.services.task_objects import record_new_artifacts
from test_artifact_ownership import make_conversation, make_project, project_root, scoped, write_artifact
from test_artifact_relocation_ownership import _index, _relocate
from test_org_list_scaling import _headers, client  # noqa: F401  (client is a fixture)
from test_router_probe_lifespan import boot  # noqa: F401  (a fixture)

pytestmark = pytest.mark.usefixtures("cleanup_tmp_projects")

OWNER = "11111111-1111-4111-8111-111111111111"
CO_MEMBER = "22222222-2222-4222-8222-222222222222"


def _engine():
    return get_engine(get_app_settings().database.uri)


def _no_lock(_engine, _key):
    return nullcontext(True)


def _sentinel() -> HeldRootSentinel | None:
    with Session(_engine()) as raw:
        return read_sentinel(ScopedSession(raw, SYSTEM_SCOPE))


def _sentinel_present() -> bool:
    return _sentinel() is not None


def _progress_present() -> bool:
    with Session(_engine()) as raw:
        return raw.exec(select(Setting).where(Setting.key == PROGRESS_KEY, Setting.scope.is_(None))).first() is not None


@pytest.fixture
def backfill_pending():
    """No pass has written the sentinel or kept progress; what a test writes
    goes after it, as the database is shared across modules, so one test's
    progress never skips the next test's projects."""
    yield
    with Session(_engine()) as raw:
        for row in raw.exec(select(Setting).where(Setting.key.in_((SENTINEL_KEY, PROGRESS_KEY)))).all():
            raw.delete(row)
        raw.commit()


def _walked(monkeypatch) -> list[UUID]:
    """The projects each later pass walks, in order."""
    walked: list[UUID] = []
    real = held_artifact_roots._held_in

    def held_in(project, *, stop):
        walked.append(project.id)
        return real(project, stop=stop)

    monkeypatch.setattr(held_artifact_roots, "_held_in", held_in)
    return walked


def _failing_scandir(monkeypatch, *, unreadable: Path, failure: int = errno.EIO) -> None:
    """The file server stops answering for one root."""
    real_scandir = held_artifact_roots.dir_scandir

    def scandir(root):
        if Path(str(root.path)) == unreadable:
            raise OSError(failure, "the file server did not answer")
        return real_scandir(root)

    monkeypatch.setattr(held_artifact_roots, "dir_scandir", scandir)


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
def test_a_project_the_backfill_cannot_read_is_listed_in_the_sentinel_and_walked_again(
    tmp_path, org_deployment, backfill_pending, monkeypatch, caplog, failure
):
    """A read error is not an empty root: the list would then miss its
    cards for good. The pass goes on and still writes the sentinel, which
    lists the project, so the fast path holds for every other one. A later
    start walks only the listed project, and empties the list once it
    reads."""
    org_id = str(uuid4())
    unreadable, readable = make_project(tmp_path, org_id), make_project(tmp_path, org_id)
    write_artifact(project_root(unreadable).base, "report")
    write_artifact(project_root(readable).base, "report")
    ids = [unreadable.id, readable.id]

    with monkeypatch.context() as failing, caplog.at_level(logging.WARNING, logger=held_artifact_roots.logger.name):
        _failing_scandir(failing, unreadable=project_root(unreadable).base, failure=failure)
        summary = run_held_root_backfill(try_lock=_no_lock, project_ids=ids)

    assert summary is not None and summary.unread == 1
    assert _rows(org_id) == {HeldRoot(readable.id, None)}
    assert _sentinel() == HeldRootSentinel(unread=[unreadable.id])
    assert not _progress_present()
    [unread] = [
        record for record in caplog.records
        if record.name == held_artifact_roots.logger.name
        and getattr(record, "event", None) == "held_artifact_roots_backfill_unread"
    ]
    assert unread.project_id == str(unreadable.id)

    walked = _walked(monkeypatch)
    with monkeypatch.context() as failing:
        _failing_scandir(failing, unreadable=project_root(unreadable).base, failure=failure)
        run_held_root_backfill(try_lock=_no_lock, project_ids=ids)
    assert walked == [unreadable.id]
    # A pass over the sentinel's list keeps no progress row.
    assert _sentinel() == HeldRootSentinel(unread=[unreadable.id]) and not _progress_present()

    walked.clear()
    summary = run_held_root_backfill(try_lock=_no_lock, project_ids=ids)
    assert walked == [unreadable.id]
    assert summary is not None and (summary.held, summary.unread) == (1, 0)
    assert _rows(org_id) == {HeldRoot(unreadable.id, None), HeldRoot(readable.id, None)}
    assert _sentinel() == HeldRootSentinel(unread=[]) and not _progress_present()

    walked.clear()
    assert run_held_root_backfill(try_lock=_no_lock, project_ids=ids) is None
    assert walked == []


def test_a_stopped_backfill_resumes_after_the_last_project_it_finished(
    tmp_path, org_deployment, backfill_pending, monkeypatch
):
    """A stop at shutdown costs only the project the walk was in: the
    progress row keeps the ones it finished, and the next start goes on
    from there."""
    org_id = str(uuid4())
    projects = [make_project(tmp_path, org_id) for _ in range(3)]
    for project in projects:
        write_artifact(project_root(project).base, "report")
    ids = sorted(project.id for project in projects)
    stop = threading.Event()
    walked = _walked(monkeypatch)
    real_save = held_artifact_roots.save_progress

    def save_progress(engine, key, *, progress):
        # Shutdown lands once the first project is finished.
        real_save(engine, key, progress=progress)
        stop.set()

    with monkeypatch.context() as stopping:
        stopping.setattr(held_artifact_roots, "save_progress", save_progress)
        assert run_held_root_backfill(try_lock=_no_lock, project_ids=ids, stop=stop) is None

    assert walked == ids[:1]
    assert _sentinel() is None and _progress_present()
    assert _rows(org_id) == {HeldRoot(ids[0], None)}

    walked.clear()
    summary = run_held_root_backfill(try_lock=_no_lock, project_ids=ids)

    assert walked == ids[1:]
    assert summary is not None and summary.held == 2
    assert _rows(org_id) == {HeldRoot(project_id, None) for project_id in ids}
    assert _sentinel() == HeldRootSentinel() and not _progress_present()


def test_a_resumed_backfill_drops_an_unread_project_deleted_since(
    tmp_path, org_deployment, backfill_pending, monkeypatch
):
    """A project the stopped pass could not read is deleted before the next
    start. The query never returns it again, so the resumed pass drops it
    from its progress, and the sentinel it writes does not list it."""
    org_id = str(uuid4())
    projects = [make_project(tmp_path, org_id) for _ in range(2)]
    for project in projects:
        write_artifact(project_root(project).base, "report")
    # The first in id order, so the kept progress is already past it.
    unreadable, readable = sorted(projects, key=lambda project: project.id)
    ids = [unreadable.id, readable.id]
    stop = threading.Event()
    real_save = held_artifact_roots.save_progress

    def save_progress(engine, key, *, progress):
        # Shutdown lands once the unreadable project is finished.
        real_save(engine, key, progress=progress)
        stop.set()

    with monkeypatch.context() as stopping:
        _failing_scandir(stopping, unreadable=project_root(unreadable).base)
        stopping.setattr(held_artifact_roots, "save_progress", save_progress)
        assert run_held_root_backfill(try_lock=_no_lock, project_ids=ids, stop=stop) is None
    assert read_progress(_engine(), PROGRESS_KEY) == BackfillProgress(after=unreadable.id, unread=[unreadable.id])
    with Session(_engine()) as raw:
        raw.delete(raw.get(Project, unreadable.id))
        raw.commit()

    summary = run_held_root_backfill(try_lock=_no_lock, project_ids=ids)

    assert summary is not None and (summary.held, summary.unread) == (1, 0)
    assert _rows(org_id) == {HeldRoot(readable.id, None)}
    assert _sentinel() == HeldRootSentinel(unread=[]) and not _progress_present()


def test_a_resumed_backfill_walks_again_an_unread_project_it_had_passed(
    tmp_path, org_deployment, backfill_pending, monkeypatch
):
    """The stopped pass could not read the first project in id order, and its
    progress is already past it. The resumed pass still walks it, so the
    sentinel lists it while it stays unreadable, and a later start holds its
    root once it reads."""
    org_id = str(uuid4())
    projects = [make_project(tmp_path, org_id) for _ in range(2)]
    for project in projects:
        write_artifact(project_root(project).base, "report")
    unreadable, readable = sorted(projects, key=lambda project: project.id)
    ids = [unreadable.id, readable.id]
    stop = threading.Event()
    real_save = held_artifact_roots.save_progress

    def save_progress(engine, key, *, progress):
        # Shutdown lands once the unreadable project is finished.
        real_save(engine, key, progress=progress)
        stop.set()

    with monkeypatch.context() as stopping:
        _failing_scandir(stopping, unreadable=project_root(unreadable).base)
        stopping.setattr(held_artifact_roots, "save_progress", save_progress)
        assert run_held_root_backfill(try_lock=_no_lock, project_ids=ids, stop=stop) is None
    assert read_progress(_engine(), PROGRESS_KEY) == BackfillProgress(after=unreadable.id, unread=[unreadable.id])

    with monkeypatch.context() as still_failing:
        _failing_scandir(still_failing, unreadable=project_root(unreadable).base)
        summary = run_held_root_backfill(try_lock=_no_lock, project_ids=ids)

    assert summary is not None and (summary.held, summary.unread) == (1, 1)
    assert _rows(org_id) == {HeldRoot(readable.id, None)}
    assert _sentinel() == HeldRootSentinel(unread=[unreadable.id]) and not _progress_present()

    summary = run_held_root_backfill(try_lock=_no_lock, project_ids=ids)

    assert summary is not None and (summary.held, summary.unread) == (1, 0)
    assert _rows(org_id) == {HeldRoot(unreadable.id, None), HeldRoot(readable.id, None)}
    assert _sentinel() == HeldRootSentinel(unread=[])


def test_each_pass_logs_its_start_or_why_it_walks_nothing(
    tmp_path, org_deployment, backfill_pending, monkeypatch, caplog
):
    """An operator tells a walk that runs from one that never started, and a
    first walk from one that resumes it or walks the sentinel's list, at the
    WARNING level every hosted deployment logs at. The count is the projects
    the pass walks."""
    org_id = str(uuid4())
    projects = [make_project(tmp_path, org_id) for _ in range(3)]
    for project in projects:
        write_artifact(project_root(project).base, "report")
    last = max(projects, key=lambda project: project.id)
    ids = [project.id for project in projects]
    stop = threading.Event()
    real_save = held_artifact_roots.save_progress

    def save_progress(engine, key, *, progress):
        # Shutdown lands once the first project is finished.
        real_save(engine, key, progress=progress)
        stop.set()

    @contextmanager
    def taken(_engine, _key):
        yield False

    with caplog.at_level(logging.WARNING, logger=held_artifact_roots.logger.name):
        assert run_held_root_backfill(try_lock=taken, project_ids=ids) is None
        with monkeypatch.context() as stopping:
            stopping.setattr(held_artifact_roots, "save_progress", save_progress)
            assert run_held_root_backfill(try_lock=_no_lock, project_ids=ids, stop=stop) is None
        with monkeypatch.context() as failing:
            # The resumed pass cannot read the last project, so the sentinel
            # lists it, and the next start walks it alone.
            _failing_scandir(failing, unreadable=project_root(last).base)
            run_held_root_backfill(try_lock=_no_lock, project_ids=ids)
        assert _sentinel() == HeldRootSentinel(unread=[last.id])
        run_held_root_backfill(try_lock=_no_lock, project_ids=ids)
        assert run_held_root_backfill(try_lock=_no_lock, project_ids=ids) is None

    lines = [
        (record.event, record.reason, getattr(record, "count", None))
        for record in caplog.records
        if record.name == held_artifact_roots.logger.name
        and record.levelno == logging.WARNING
        and getattr(record, "event", None)
        in ("held_artifact_roots_backfill_started", "held_artifact_roots_backfill_skipped")
    ]
    assert lines == [
        ("held_artifact_roots_backfill_skipped", "locked", None),
        ("held_artifact_roots_backfill_started", "first", 3),
        ("held_artifact_roots_backfill_started", "resumed", 2),
        ("held_artifact_roots_backfill_started", "unread", 1),
        ("held_artifact_roots_backfill_skipped", "finished", None),
    ]


def test_a_pass_that_takes_the_lock_after_another_replica_finished_walks_nothing(
    tmp_path, org_deployment, backfill_pending, monkeypatch, caplog
):
    """The pass read no sentinel, then another replica finished the walk and
    let go of the lock before this one took it. The pass reads the sentinel
    again under the lock, so it walks no project a second time."""
    project = make_project(tmp_path, str(uuid4()))
    write_artifact(project_root(project).base, "report")
    walked = _walked(monkeypatch)

    @contextmanager
    def finished_meanwhile(engine, _key):
        write_sentinel(engine, SENTINEL_KEY, value=HeldRootSentinel().model_dump_json())
        yield True

    with caplog.at_level(logging.WARNING, logger=held_artifact_roots.logger.name):
        assert run_held_root_backfill(try_lock=finished_meanwhile, project_ids=[project.id]) is None

    assert walked == []
    assert [
        (record.event, record.reason) for record in caplog.records if record.name == held_artifact_roots.logger.name
    ] == [("held_artifact_roots_backfill_skipped", "finished")]


def test_a_pass_that_takes_the_lock_after_another_replica_listed_an_unread_project_walks_only_that_one(
    tmp_path, org_deployment, backfill_pending, monkeypatch, caplog
):
    """The pass read no sentinel, then another replica finished the walk with
    one project it could not read and let go of the lock. The pass reads the
    sentinel again under the lock and walks only the project it lists, not
    every project as a first pass."""
    org_id = str(uuid4())
    listed, other = (make_project(tmp_path, org_id) for _ in range(2))
    for project in (listed, other):
        write_artifact(project_root(project).base, "report")
    walked = _walked(monkeypatch)

    @contextmanager
    def listed_meanwhile(engine, _key):
        write_sentinel(engine, SENTINEL_KEY, value=HeldRootSentinel(unread=[listed.id]).model_dump_json())
        yield True

    with caplog.at_level(logging.WARNING, logger=held_artifact_roots.logger.name):
        summary = run_held_root_backfill(try_lock=listed_meanwhile, project_ids=[listed.id, other.id])

    assert summary is not None and (summary.held, summary.unread) == (1, 0)
    assert walked == [listed.id]
    assert _rows(org_id) == {HeldRoot(listed.id, None)}
    assert _sentinel() == HeldRootSentinel(unread=[])
    assert not _progress_present()
    assert [
        (record.event, record.reason, record.count)
        for record in caplog.records
        if record.name == held_artifact_roots.logger.name
        and getattr(record, "event", None) == "held_artifact_roots_backfill_started"
    ] == [("held_artifact_roots_backfill_started", "unread", 1)]


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


def test_the_desktop_runs_no_backfill(tmp_path, backfill_pending, caplog):
    project = make_project(tmp_path, None)
    write_artifact(project_root(project).base, "report")

    with caplog.at_level(logging.WARNING, logger=held_artifact_roots.logger.name):
        assert run_held_root_backfill(try_lock=_no_lock, project_ids=[project.id]) is None
    assert not _sentinel_present()
    [skipped] = [record for record in caplog.records if record.name == held_artifact_roots.logger.name]
    assert (skipped.event, skipped.reason) == ("held_artifact_roots_backfill_skipped", "not_org")


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


def test_a_global_value_another_writer_added_first_is_left_as_it_is(monkeypatch):
    """Two writers of one global settings row, a backfill's sentinel or its
    progress, can both find no row and both insert: the unique index refuses
    the second, and its writer goes on."""
    engine = _engine()
    key = f"_held_artifact_roots_test_{uuid4().hex}"
    real = artifact_owner_backfill._global_row

    def global_row_then_the_other_writer(raw, row_key):
        row = real(raw, row_key)
        # The other writer commits after this one's read.
        with Session(engine) as other:
            other.add(Setting(key=row_key, value="first"))
            other.commit()
        return row

    try:
        with monkeypatch.context() as racing:
            racing.setattr(artifact_owner_backfill, "_global_row", global_row_then_the_other_writer)
            artifact_owner_backfill._write_global_value(engine, key, "second")
        with Session(engine) as raw:
            assert [row.value for row in raw.exec(select(Setting).where(Setting.key == key)).all()] == ["first"]
    finally:
        with Session(engine) as raw:
            raw.exec(delete(Setting).where(Setting.key == key))
            raw.commit()


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


def test_a_delete_by_old_folder_name_finds_what_the_list_shows(tmp_path, org_deployment, backfill_pending):
    """Once the rows are complete, a delete by an old folder name looks only
    in the roots they name, as the list does: a folder in a root no writer
    recorded is not found, never one outside the caller's roots."""
    from cowork.api.v1.endpoints.artifacts import _legacy_artifact_in_plan, _plan_scoped_project

    org_id = str(uuid4())
    project = make_project(tmp_path, org_id)
    write_artifact(project_root(project).base, "listed")
    run_held_root_backfill(try_lock=_no_lock, project_ids=[project.id])
    owned = make_conversation(project, OWNER)
    write_artifact(_workspace(project, owned) / ".anton" / "artifacts", "never-recorded")

    with scoped(org_id, OWNER) as session:
        plan = _plan_scoped_project(session, project_ref=str(project.id))

    assert plan.held == {HeldRoot(project.id, None)}
    source, folder = _legacy_artifact_in_plan(plan, "listed")
    assert folder == project_root(project).base / "listed"
    assert _legacy_artifact_in_plan(plan, "never-recorded") == (None, None)


def test_held_roots_never_returns_a_co_members_legacy_row(tmp_path, org_deployment, backfill_pending):
    """The query reads every shared root's row, and a legacy root's only for
    a conversation the caller owns, by the plan's owner rule: a scope with no
    user, such as a channel turn's, owns the conversations no member
    created."""
    org_id = str(uuid4())
    project = make_project(tmp_path, org_id)
    owners, co_members, nobodys = (make_conversation(project, user) for user in (OWNER, CO_MEMBER, None))
    with scoped(org_id) as session:
        hold_root(session, project_id=project.id)
        for conversation_id in (owners, co_members, nobodys):
            hold_root(session, project_id=project.id, conversation_id=conversation_id)
    run_held_root_backfill(try_lock=_no_lock, project_ids=[])

    def held(user_id):
        with scoped(org_id, user_id) as session:
            return held_roots(session, project_ids=[project.id])

    shared = HeldRoot(project.id, None)
    assert held(OWNER) == {shared, HeldRoot(project.id, owners)}
    assert held(CO_MEMBER) == {shared, HeldRoot(project.id, co_members)}
    assert held(str(uuid4())) == {shared}
    assert held(None) == {shared, HeldRoot(project.id, nobodys)}


def _slugs(http, path: str, *, org_id: str, user_id: str = OWNER) -> set[str]:
    """The slugs of the cards `path` lists for `user_id`."""
    response = http.get(path, headers=_headers(org_id, user_id))
    assert response.status_code == 200, response.text
    return {card["slug"] for card in response.json()}


def test_the_list_trusts_the_rows_except_for_the_projects_the_backfill_could_not_read(
    tmp_path, org_deployment, backfill_pending, client, monkeypatch  # noqa: F811
):
    """A project the sentinel lists has its roots found on the file store,
    the owner filter included; every other project's come from the rows, so
    a folder in a root no row names stays out."""
    org_id = str(uuid4())
    read, unread = make_project(tmp_path, org_id), make_project(tmp_path, org_id)
    write_artifact(project_root(read).base, "from-a-row")
    write_artifact(project_root(unread).base, "walked")
    with monkeypatch.context() as failing:
        _failing_scandir(failing, unreadable=project_root(unread).base)
        run_held_root_backfill(try_lock=_no_lock, project_ids=[read.id, unread.id])
    assert _sentinel() == HeldRootSentinel(unread=[unread.id])
    # Written after the backfill, where no writer records a row.
    write_artifact(_workspace(read, make_conversation(read, OWNER)) / ".anton" / "artifacts", "never-recorded")
    write_artifact(_workspace(unread, make_conversation(unread, OWNER)) / ".anton" / "artifacts", "walked-old")
    write_artifact(_workspace(unread, make_conversation(unread, CO_MEMBER)) / ".anton" / "artifacts", "co-members")

    assert _slugs(client, "/api/v1/artifacts/", org_id=org_id) == {"from-a-row", "walked", "walked-old"}
    assert _slugs(client, f"/api/v1/artifacts/?project_id={read.id}", org_id=org_id) == {"from-a-row"}
    assert _slugs(client, f"/api/v1/artifacts/?project_id={unread.id}", org_id=org_id) == {"walked", "walked-old"}
    assert _slugs(
        client, f"/api/v1/artifacts/?project_id={unread.id}", org_id=org_id, user_id=CO_MEMBER
    ) == {"walked", "co-members"}


def test_a_sentinel_value_it_cannot_read_reads_as_no_sentinel(tmp_path, org_deployment, backfill_pending, caplog):
    """A sentinel row whose value is not a HeldRootSentinel, as one written by
    hand, never fails the list: the list finds its roots on the file store as
    before the backfill, and the next pass rewrites the value."""
    org_id = str(uuid4())
    project = make_project(tmp_path, org_id)
    write_artifact(project_root(project).base, "report")
    write_sentinel(_engine(), SENTINEL_KEY, value="1")

    with caplog.at_level(logging.WARNING, logger=held_artifact_roots.logger.name):
        with scoped(org_id, OWNER) as session:
            plan = with_held_roots(session, plan_scope_roots(session))

    assert plan.held is None
    assert project_root(project).base in [source.base for source in readable_sources(plan)]
    assert [getattr(record, "event", None) for record in caplog.records if record.levelno == logging.WARNING] == [
        "held_artifact_roots_sentinel_unreadable"
    ]

    summary = run_held_root_backfill(try_lock=_no_lock, project_ids=[project.id])

    assert summary is not None and (summary.held, summary.unread) == (1, 0)
    assert _sentinel() == HeldRootSentinel(unread=[])


def test_the_list_walks_a_project_whose_unread_pass_lands_while_it_reads(
    tmp_path, org_deployment, backfill_pending, monkeypatch
):
    """Another replica's pass over the unread projects commits a project's
    rows, then the sentinel without it, while with_held_roots reads. The
    plan read the sentinel before the rows, so the project is still walked,
    never left both unheld and unwalked."""
    org_id = str(uuid4())
    project = make_project(tmp_path, org_id)
    write_artifact(project_root(project).base, "report")
    with monkeypatch.context() as failing:
        _failing_scandir(failing, unreadable=project_root(project).base)
        run_held_root_backfill(try_lock=_no_lock, project_ids=[project.id])
    assert _sentinel() == HeldRootSentinel(unread=[project.id])
    real = held_artifact_roots.held_roots

    def held_roots_then_the_unread_pass(session, *, project_ids):
        held = real(session, project_ids=project_ids)
        run_held_root_backfill(try_lock=_no_lock, project_ids=[project.id])
        return held

    monkeypatch.setattr(held_artifact_roots, "held_roots", held_roots_then_the_unread_pass)
    with scoped(org_id, OWNER) as session:
        plan = with_held_roots(session, plan_scope_roots(session))

    # The pass landed: the project now has its row and the list is empty.
    assert _rows(org_id) == {HeldRoot(project.id, None)}
    assert _sentinel() == HeldRootSentinel(unread=[])
    assert plan.walk == {project.id}
    assert project_root(project).base in [source.base for source in readable_sources(plan)]


def test_a_held_row_never_shows_a_project_in_another_partition(
    tmp_path, org_deployment, backfill_pending, client, monkeypatch  # noqa: F811
):
    """A project of the caller's organization whose folder sits in another
    partition of the shared root stays out of the list, even when a held row
    names it. The backfill never opens that folder and records no row for
    it; the test writes one itself to check the list."""
    org_id = str(uuid4())
    shared_root = Path(os.path.realpath(get_app_settings().storage.shared_root))
    partition = shared_root / str(uuid4())
    path = partition / "projects" / f"proj-{uuid4().hex[:8]}"
    path.mkdir(parents=True)
    elsewhere = Project(id=uuid4(), name=path.name, path=str(path), org_id=org_id)
    with Session(_engine()) as raw:
        raw.add(elsewhere)
        raw.commit()
        raw.refresh(elsewhere)
        raw.expunge(elsewhere)
    try:
        write_artifact(project_root(elsewhere).base, "in-another-partition")
        home = make_project(tmp_path, org_id)
        write_artifact(project_root(home).base, "ours")
        walked = _walked(monkeypatch)
        run_held_root_backfill(try_lock=_no_lock, project_ids=[elsewhere.id, home.id])
        assert walked == [home.id]
        assert _rows(org_id) == {HeldRoot(home.id, None)}
        with scoped(org_id) as session:
            hold_root(session, project_id=elsewhere.id)
        assert HeldRoot(elsewhere.id, None) in _rows(org_id)

        assert _slugs(client, "/api/v1/artifacts/", org_id=org_id) == {"ours"}
        assert _slugs(client, f"/api/v1/artifacts/?project_id={elsewhere.id}", org_id=org_id) == set()
    finally:
        with Session(_engine()) as raw:
            raw.exec(delete(HeldArtifactRoot).where(HeldArtifactRoot.project_id == elsewhere.id))
            raw.exec(delete(Project).where(Project.id == elsewhere.id))
            raw.commit()
        shutil.rmtree(partition, ignore_errors=True)


def _startup_passes(monkeypatch, project_ids) -> list[str]:
    """Run the startup task's held-root pass over `project_ids` alone, as
    every backfill test does on the shared database, and name each pass as
    it runs. The location backfill has its own tests and walks nothing
    here."""
    passes: list[str] = []
    real = held_artifact_roots.run_held_root_backfill

    def location(*, stop):
        passes.append("location")

    def held(*, stop):
        passes.append("held")
        return real(project_ids=project_ids, stop=stop)

    monkeypatch.setattr(artifact_location_backfill, "run_artifact_location_backfill", location)
    monkeypatch.setattr(held_artifact_roots, "run_held_root_backfill", held)
    return passes


def _org_tenancy_for_the_pass(monkeypatch) -> None:
    """Org tenancy for the held-root pass alone, as boot gives it to the
    lifespan alone: the rest of the lifespan's startup is not under test
    here."""
    settings = held_artifact_roots.get_app_settings()
    monkeypatch.setattr(
        held_artifact_roots, "get_app_settings", lambda: settings.model_copy(update={"tenancy_mode": "org"})
    )


async def _run_the_startup_task(boot) -> None:  # noqa: F811
    """Start the lifespan, which starts the backfills' task, and wait for
    that task to end."""
    app, _order, _probe = boot(tenancy="org")
    async with server.lifespan(app):
        await asyncio.wait_for(asyncio.shield(app.state.artifact_location_backfill), 20)


async def test_the_startup_task_runs_the_held_root_backfill(
    tmp_path, backfill_pending, boot, monkeypatch  # noqa: F811
):
    """The task the lifespan starts runs the held-root pass after the
    location backfill, on the file store's threads, and ends once the
    sentinel is written."""
    org_id = str(uuid4())
    project = make_project(tmp_path, org_id)
    write_artifact(project_root(project).base, "report")
    passes = _startup_passes(monkeypatch, [project.id])
    _org_tenancy_for_the_pass(monkeypatch)

    await _run_the_startup_task(boot)

    assert passes == ["location", "held"]
    assert _rows(org_id) == {HeldRoot(project.id, None)}
    assert _sentinel() == HeldRootSentinel()


def _retry_waits(monkeypatch) -> list[float]:
    """The seconds of each wait the startup task asks for between its tries.
    Each wait ends at once, so the test goes on."""
    waits: list[float] = []
    real_sleep = asyncio.sleep

    async def sleep(seconds):
        waits.append(seconds)
        await real_sleep(0)

    monkeypatch.setattr(server.asyncio, "sleep", sleep)
    return waits


async def test_the_startup_task_waits_and_runs_the_passes_again_after_a_lock_miss(
    tmp_path, org_deployment, backfill_pending, monkeypatch
):
    """A replica that found the walk's lock taken waits
    ARTIFACT_BACKFILL_RETRY_SECONDS and tries again while the sentinel is
    missing, so a walker that died never leaves it unwritten until the next
    deploy, and the replica never spins on the lock."""
    org_id = str(uuid4())
    project = make_project(tmp_path, org_id)
    write_artifact(project_root(project).base, "report")
    passes = _startup_passes(monkeypatch, [project.id])
    waits = _retry_waits(monkeypatch)
    tries = iter([False])

    @contextmanager
    def lock(_engine, _key):
        # Another replica holds the walk at the first try.
        yield next(tries, True)

    monkeypatch.setattr(held_artifact_roots, "advisory_try_lock", lock)

    await asyncio.wait_for(server._run_artifact_backfills(threading.Event()), 20)

    assert passes == ["location", "held", "location", "held"]
    assert waits == [server.ARTIFACT_BACKFILL_RETRY_SECONDS]
    assert _rows(org_id) == {HeldRoot(project.id, None)}
    assert _sentinel() == HeldRootSentinel()


async def test_the_startup_task_tries_no_more_once_shutdown_sets_the_stop(
    tmp_path, org_deployment, backfill_pending, monkeypatch
):
    project = make_project(tmp_path, str(uuid4()))
    passes = _startup_passes(monkeypatch, [project.id])
    stop = threading.Event()

    @contextmanager
    def taken(_engine, _key):
        # Shutdown lands while another replica holds the walk.
        stop.set()
        yield False

    monkeypatch.setattr(held_artifact_roots, "advisory_try_lock", taken)
    waits = _retry_waits(monkeypatch)

    await asyncio.wait_for(server._run_artifact_backfills(stop), 20)

    assert passes == ["location", "held"]
    assert waits == []
    assert not _sentinel_present()


async def test_the_startup_task_waits_and_tries_again_when_it_cannot_read_the_sentinel(
    tmp_path, org_deployment, backfill_pending, monkeypatch, caplog
):
    """A failed read of the sentinel after the passes does not end the task:
    it logs the failure, waits ARTIFACT_BACKFILL_RETRY_SECONDS, and runs the
    passes again, which then find the sentinel."""
    project = make_project(tmp_path, str(uuid4()))
    passes = _startup_passes(monkeypatch, [project.id])
    waits = _retry_waits(monkeypatch)
    real_run_db = units.run_db
    failures = iter([units.DatabaseBusy("no database connection freed within POOL_TIMEOUT")])

    async def run_db(fn, *, scope):
        # The first read of the sentinel fails; the database answers after.
        failure = next(failures, None)
        if failure is not None:
            raise failure
        return await real_run_db(fn, scope=scope)

    monkeypatch.setattr(units, "run_db", run_db)

    with caplog.at_level(logging.WARNING, logger=server.logger.name):
        await asyncio.wait_for(server._run_artifact_backfills(threading.Event()), 20)

    assert passes == ["location", "held", "location", "held"]
    assert server.ARTIFACT_BACKFILL_RETRY_SECONDS == 300
    assert waits == [server.ARTIFACT_BACKFILL_RETRY_SECONDS]
    assert [
        record.operation for record in caplog.records
        if record.name == server.logger.name and getattr(record, "event", None) == "artifact_backfill_failed"
    ] == ["held_artifact_roots_sentinel"]
    assert _sentinel() == HeldRootSentinel()


async def test_the_startup_task_ends_once_the_sentinel_lists_an_unread_project(
    tmp_path, org_deployment, backfill_pending, monkeypatch
):
    """A sentinel that lists a project the pass could not read still ends
    the task: the next start walks that project again."""
    project = make_project(tmp_path, str(uuid4()))
    write_artifact(project_root(project).base, "report")
    passes = _startup_passes(monkeypatch, [project.id])
    waits = _retry_waits(monkeypatch)
    _failing_scandir(monkeypatch, unreadable=project_root(project).base)

    await asyncio.wait_for(server._run_artifact_backfills(threading.Event()), 20)

    assert passes == ["location", "held"]
    assert waits == []
    assert _sentinel() == HeldRootSentinel(unread=[project.id])
