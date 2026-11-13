"""The artifact_sort_times rows: the backfill reads every held root's folders,
each writer keeps the newest time a folder had, a self-heal drops only the
rows of folders gone before its listing, and the newest rows come in one
query."""
from __future__ import annotations

import errno
import logging
import os
import shutil
import threading
from contextlib import contextmanager, nullcontext
from pathlib import Path
from uuid import UUID, uuid4

import pytest
from sqlalchemy import delete, event
from sqlmodel import Session, select

from cowork.common.settings.app_settings import get_app_settings
from cowork.db.scoped import ScopedSession, TenantScope
from cowork.db.session import get_engine
from cowork.db.units import unit_session
from cowork.models.artifact_sort_time import ArtifactSortTime
from cowork.models.held_artifact_root import HeldArtifactRoot
from cowork.models.project import Project
from cowork.models.setting import Setting
from cowork.services import artifact_sort_times, held_artifact_roots
from cowork.services.artifact_identity import is_listable_folder_name
from cowork.services.artifact_locations import Location
from cowork.services.artifact_owner_backfill import BackfillProgress, read_progress, write_sentinel
from cowork.services.artifact_roots import plan_scope_roots, with_held_roots
from cowork.services.artifact_identity import ListedIds
from cowork.services.artifact_sort_times import (
    PROGRESS_KEY,
    SENTINEL_KEY,
    SortTime,
    SortTimeRow,
    SortTimeSentinel,
    SortTimesRead,
    id_holder_times,
    move_sort_times,
    newest_folders,
    reconcile_sort_times,
    record_sort_times,
    run_sort_time_backfill,
    sort_time_rows,
    sort_times_below,
)
from cowork.services.held_artifact_roots import PROGRESS_KEY as HELD_PROGRESS_KEY
from cowork.services.held_artifact_roots import SENTINEL_KEY as HELD_KEY
from cowork.services.held_artifact_roots import HeldRoot, HeldRootSentinel, hold_root, run_held_root_backfill
from cowork.services.task_objects import record_new_artifacts, snapshot_artifact_state, turn_artifact_changes
from test_artifact_ownership import make_conversation, make_project, project_root, scoped, write_artifact
from test_artifact_relocation_ownership import _index, _relocate
from test_held_artifact_roots import _commit_rename, _refused_project_folder, _rename
from tests._artifact_rows import sort_sentinel, sort_times_of, timed_artifact
from tests._logging_assertions import assert_log_fields

pytestmark = pytest.mark.usefixtures("cleanup_tmp_projects")

OWNER = "11111111-1111-4111-8111-111111111111"


def _engine():
    return get_engine(get_app_settings().database.uri)


def _no_lock(_engine, _key):
    return nullcontext(True)


def _progress_present() -> bool:
    with Session(_engine()) as raw:
        return raw.exec(select(Setting).where(Setting.key == PROGRESS_KEY, Setting.scope.is_(None))).first() is not None


@pytest.fixture
def backfills_pending():
    """No pass has written either sentinel or kept progress; those a test
    writes go after it, as the database is shared across modules."""
    yield
    keys = (SENTINEL_KEY, PROGRESS_KEY, HELD_KEY, HELD_PROGRESS_KEY)
    with Session(_engine()) as raw:
        for row in raw.exec(select(Setting).where(Setting.key.in_(keys))).all():
            raw.delete(row)
        raw.commit()


def _workspace_root(project, conversation_id) -> Path:
    return Path(project.path) / "conversations" / str(conversation_id) / ".anton" / "artifacts"


def _walked(monkeypatch) -> list[UUID]:
    """The projects each later sort-time pass walks, in order."""
    walked: list[UUID] = []
    real = artifact_sort_times._time_project

    def time_project(engine, *, held, **kwargs):
        walked.append(held.project.id)
        return real(engine, held=held, **kwargs)

    monkeypatch.setattr(artifact_sort_times, "_time_project", time_project)
    return walked


def _failing_scandir(monkeypatch, *, unreadable: Path) -> None:
    """The file server stops answering for one root, for every listing of
    a root the list would open (held_artifact_roots.listable_folders)."""
    real_scandir = held_artifact_roots.dir_scandir

    def scandir(root):
        if Path(str(root.path)) == unreadable:
            raise OSError(errno.EIO, "the file server did not answer")
        return real_scandir(root)

    monkeypatch.setattr(held_artifact_roots, "dir_scandir", scandir)


def _pass_lines(caplog) -> list[tuple]:
    """Each sort-time pass's start or skip line: its event, reason and count."""
    return [
        (record.event, record.reason, getattr(record, "count", None))
        for record in caplog.records
        if record.name == artifact_sort_times.logger.name
        and record.levelno == logging.WARNING
        and getattr(record, "event", None)
        in ("artifact_sort_times_backfill_started", "artifact_sort_times_backfill_skipped")
    ]


# The backfill.


def test_the_backfill_records_the_time_of_every_folder_in_the_held_roots(
    tmp_path, org_deployment, backfills_pending
):
    org_id = str(uuid4())
    project = make_project(tmp_path, org_id)
    base = project_root(project).base
    timed_artifact(base, "report", 1_000_100.0)
    # Folders the list never opens: a dot-named copy, a folder with no
    # metadata.json, one whose metadata.json is no plain file, and a link.
    timed_artifact(base, ".report.bak", 1_000_900.0)
    (base / "no-metadata").mkdir()
    (base / "fifo-metadata").mkdir()
    os.mkfifo(base / "fifo-metadata" / "metadata.json")
    (base / "linked").symlink_to(base / "report")
    owned = make_conversation(project, OWNER)
    timed_artifact(_workspace_root(project, owned), "old-report", 1_000_200.0)

    # Waits for the held rows, which name the roots it reads.
    assert run_sort_time_backfill(try_lock=_no_lock, project_ids=[project.id]) is None
    run_held_root_backfill(try_lock=_no_lock, project_ids=[project.id])
    with scoped(org_id) as session:
        # A turn end recorded this folder's time already: the pass reads it
        # and writes nothing for it.
        record_sort_times(session, project_id=project.id, times={"report": 1_000_100.0})
    summary = run_sort_time_backfill(try_lock=_no_lock, project_ids=[project.id])

    assert summary is not None and (summary.recorded, summary.unread) == (1, 0)
    assert sort_times_of(org_id) == {
        (project.id, None, "report"): 1_000_100.0,
        (project.id, owned, "old-report"): 1_000_200.0,
    }
    assert sort_sentinel() == SortTimeSentinel(unread=[])
    assert not _progress_present()
    assert run_sort_time_backfill(try_lock=_no_lock, project_ids=[project.id]) is None


def test_a_project_the_backfill_cannot_read_is_listed_in_the_sentinel_and_walked_again(
    tmp_path, org_deployment, backfills_pending, monkeypatch
):
    """The pass writes its sentinel even so, listing the project; the next
    start walks only that project and empties the list."""
    org_id = str(uuid4())
    readable, unreadable = sorted((make_project(tmp_path, org_id) for _ in range(2)), key=lambda p: p.id)
    for project in (readable, unreadable):
        timed_artifact(project_root(project).base, "report", 1_000_100.0)
    ids = [readable.id, unreadable.id]
    run_held_root_backfill(try_lock=_no_lock, project_ids=ids)
    walked = _walked(monkeypatch)

    with monkeypatch.context() as failing:
        _failing_scandir(failing, unreadable=project_root(unreadable).base)
        summary = run_sort_time_backfill(try_lock=_no_lock, project_ids=ids)

    assert summary is not None and (summary.recorded, summary.unread) == (1, 1)
    assert sort_sentinel() == SortTimeSentinel(unread=[unreadable.id])
    assert not _progress_present()
    assert sort_times_of(org_id) == {(readable.id, None, "report"): 1_000_100.0}

    summary = run_sort_time_backfill(try_lock=_no_lock, project_ids=ids)

    assert summary is not None and (summary.recorded, summary.unread) == (1, 0)
    assert walked == [readable.id, unreadable.id, unreadable.id]
    assert sort_times_of(org_id) == {
        (readable.id, None, "report"): 1_000_100.0,
        (unreadable.id, None, "report"): 1_000_100.0,
    }
    assert sort_sentinel() == SortTimeSentinel(unread=[])
    assert not _progress_present()


def test_a_project_the_held_backfill_could_not_read_stays_listed_until_it_has_held_rows(
    tmp_path, org_deployment, backfills_pending, monkeypatch
):
    """A project the held-root sentinel lists has no held rows from its
    backfill, so the sort-time pass reads none of its roots, even one a turn
    end held since, and lists it. Once a later held pass reads it, the next
    sort-time start times its roots and drops it from the list."""
    org_id = str(uuid4())
    read, unread = sorted((make_project(tmp_path, org_id) for _ in range(2)), key=lambda p: p.id)
    for project in (read, unread):
        timed_artifact(project_root(project).base, "report", 1_000_100.0)
    ids = [read.id, unread.id]
    with monkeypatch.context() as failing:
        _failing_scandir(failing, unreadable=project_root(unread).base)
        run_held_root_backfill(try_lock=_no_lock, project_ids=ids)
    assert held_artifact_roots._read_sentinel(_engine()) == HeldRootSentinel(unread=[unread.id])
    with scoped(org_id) as session:
        # A turn end in the unread project holds its root meanwhile.
        hold_root(session, project_id=unread.id)
    walked = _walked(monkeypatch)

    summary = run_sort_time_backfill(try_lock=_no_lock, project_ids=ids)
    assert summary is not None and (summary.recorded, summary.unread, summary.held_pending) == (1, 0, 1)
    assert sort_sentinel() == SortTimeSentinel(unread=[unread.id])
    # A later start while the held-root sentinel still lists it.
    run_sort_time_backfill(try_lock=_no_lock, project_ids=ids)
    assert sort_sentinel() == SortTimeSentinel(unread=[unread.id])
    assert walked == [read.id]
    assert sort_times_of(org_id) == {(read.id, None, "report"): 1_000_100.0}

    run_held_root_backfill(try_lock=_no_lock, project_ids=ids)
    assert held_artifact_roots._read_sentinel(_engine()) == HeldRootSentinel(unread=[])
    summary = run_sort_time_backfill(try_lock=_no_lock, project_ids=ids)

    assert summary is not None and (summary.recorded, summary.unread) == (1, 0)
    assert walked == [read.id, unread.id]
    assert sort_times_of(org_id) == {(read.id, None, "report"): 1_000_100.0, (unread.id, None, "report"): 1_000_100.0}
    assert sort_sentinel() == SortTimeSentinel(unread=[])
    assert not _progress_present()


def test_a_stopped_backfill_resumes_after_the_last_project_it_finished(
    tmp_path, org_deployment, backfills_pending, monkeypatch
):
    org_id = str(uuid4())
    first, second = sorted((make_project(tmp_path, org_id) for _ in range(2)), key=lambda p: p.id)
    for project in (first, second):
        timed_artifact(project_root(project).base, "report", 1_000_100.0)
    ids = [first.id, second.id]
    run_held_root_backfill(try_lock=_no_lock, project_ids=ids)
    stop = threading.Event()
    real_save = artifact_sort_times.save_progress

    def save_progress(engine, key, *, progress):
        # Shutdown lands once the first project is finished.
        real_save(engine, key, progress=progress)
        stop.set()

    walked = _walked(monkeypatch)
    with monkeypatch.context() as stopping:
        stopping.setattr(artifact_sort_times, "save_progress", save_progress)
        assert run_sort_time_backfill(try_lock=_no_lock, project_ids=ids, stop=stop) is None

    assert sort_sentinel() is None
    assert read_progress(_engine(), PROGRESS_KEY) == BackfillProgress(after=first.id, unread=[])

    summary = run_sort_time_backfill(try_lock=_no_lock, project_ids=ids)

    assert summary is not None and summary.recorded == 1
    assert walked == [first.id, second.id]
    assert set(sort_times_of(org_id)) == {(first.id, None, "report"), (second.id, None, "report")}
    assert sort_sentinel() == SortTimeSentinel(unread=[]) and not _progress_present()


def _stop_once_a_project_is_finished(monkeypatch, stop: threading.Event) -> None:
    """Shutdown lands once the first pass has saved its progress for one
    project."""
    real_save = artifact_sort_times.save_progress

    def save_progress(engine, key, *, progress):
        real_save(engine, key, progress=progress)
        stop.set()

    monkeypatch.setattr(artifact_sort_times, "save_progress", save_progress)


@pytest.mark.parametrize("held_by_a_turn_end", [False, True], ids=["no-held-row", "turn-end-held-row"])
def test_a_resumed_first_pass_walks_a_project_the_held_backfill_has_read_since(
    tmp_path, org_deployment, backfills_pending, monkeypatch, held_by_a_turn_end
):
    """The first pass leaves out the first project in id order, which the
    held-root sentinel lists, and is stopped once it has finished the next,
    so its progress is past both. The next start's held-root pass reads the
    first project and holds its root. The resumed pass walks it, since its
    progress kept it, and the sentinel lists no project."""
    org_id = str(uuid4())
    pending, finished, last = sorted((make_project(tmp_path, org_id) for _ in range(3)), key=lambda p: p.id)
    for project in (pending, finished, last):
        timed_artifact(project_root(project).base, "report", 1_000_100.0)
    ids = [pending.id, finished.id, last.id]
    with monkeypatch.context() as failing:
        _failing_scandir(failing, unreadable=project_root(pending).base)
        run_held_root_backfill(try_lock=_no_lock, project_ids=ids)
    assert held_artifact_roots._read_sentinel(_engine()) == HeldRootSentinel(unread=[pending.id])
    if held_by_a_turn_end:
        with scoped(org_id) as session:
            hold_root(session, project_id=pending.id)
    stop = threading.Event()
    with monkeypatch.context() as stopping:
        _stop_once_a_project_is_finished(stopping, stop)
        assert run_sort_time_backfill(try_lock=_no_lock, project_ids=ids, stop=stop) is None
    assert read_progress(_engine(), PROGRESS_KEY) == BackfillProgress(after=finished.id, unread=[pending.id])

    run_held_root_backfill(try_lock=_no_lock, project_ids=ids)
    assert held_artifact_roots._read_sentinel(_engine()) == HeldRootSentinel(unread=[])
    walked = _walked(monkeypatch)
    summary = run_sort_time_backfill(try_lock=_no_lock, project_ids=ids)

    assert summary is not None and (summary.recorded, summary.unread, summary.held_pending) == (2, 0, 0)
    assert sorted(walked, key=str) == sorted([pending.id, last.id], key=str)
    assert sort_times_of(org_id) == {(project.id, None, "report"): 1_000_100.0 for project in (pending, finished, last)}
    assert sort_sentinel() == SortTimeSentinel(unread=[])
    assert not _progress_present()


def test_a_resumed_backfill_walks_again_an_unread_project_it_had_passed(
    tmp_path, org_deployment, backfills_pending, monkeypatch
):
    """The stopped pass could not read the first project in id order, and its
    progress is already past it. The resumed pass still walks it, so the
    sentinel lists it while it stays unreadable, and a later start times its
    root once it reads."""
    org_id = str(uuid4())
    unreadable, readable = sorted((make_project(tmp_path, org_id) for _ in range(2)), key=lambda p: p.id)
    for project in (unreadable, readable):
        timed_artifact(project_root(project).base, "report", 1_000_100.0)
    ids = [unreadable.id, readable.id]
    run_held_root_backfill(try_lock=_no_lock, project_ids=ids)
    stop = threading.Event()
    with monkeypatch.context() as stopping:
        _failing_scandir(stopping, unreadable=project_root(unreadable).base)
        _stop_once_a_project_is_finished(stopping, stop)
        assert run_sort_time_backfill(try_lock=_no_lock, project_ids=ids, stop=stop) is None
    assert read_progress(_engine(), PROGRESS_KEY) == BackfillProgress(after=unreadable.id, unread=[unreadable.id])
    walked = _walked(monkeypatch)

    with monkeypatch.context() as still_failing:
        _failing_scandir(still_failing, unreadable=project_root(unreadable).base)
        summary = run_sort_time_backfill(try_lock=_no_lock, project_ids=ids)

    assert summary is not None and (summary.recorded, summary.unread) == (1, 1)
    assert walked == [unreadable.id, readable.id]
    assert sort_times_of(org_id) == {(readable.id, None, "report"): 1_000_100.0}
    assert sort_sentinel() == SortTimeSentinel(unread=[unreadable.id]) and not _progress_present()

    summary = run_sort_time_backfill(try_lock=_no_lock, project_ids=ids)

    assert summary is not None and (summary.recorded, summary.unread) == (1, 0)
    assert set(sort_times_of(org_id)) == {(unreadable.id, None, "report"), (readable.id, None, "report")}
    assert sort_sentinel() == SortTimeSentinel(unread=[])


def test_a_resumed_backfill_drops_an_unread_project_deleted_since(
    tmp_path, org_deployment, backfills_pending, monkeypatch
):
    """A project the stopped pass could not read is deleted before the next
    start. The query never returns it again, so the resumed pass drops it
    from its progress, and the sentinel it writes does not list it."""
    org_id = str(uuid4())
    unreadable, readable = sorted((make_project(tmp_path, org_id) for _ in range(2)), key=lambda p: p.id)
    for project in (unreadable, readable):
        timed_artifact(project_root(project).base, "report", 1_000_100.0)
    ids = [unreadable.id, readable.id]
    run_held_root_backfill(try_lock=_no_lock, project_ids=ids)
    stop = threading.Event()
    with monkeypatch.context() as stopping:
        _failing_scandir(stopping, unreadable=project_root(unreadable).base)
        _stop_once_a_project_is_finished(stopping, stop)
        assert run_sort_time_backfill(try_lock=_no_lock, project_ids=ids, stop=stop) is None
    assert read_progress(_engine(), PROGRESS_KEY) == BackfillProgress(after=unreadable.id, unread=[unreadable.id])
    with Session(_engine()) as raw:
        raw.exec(delete(HeldArtifactRoot).where(HeldArtifactRoot.project_id == unreadable.id))
        raw.delete(raw.get(Project, unreadable.id))
        raw.commit()

    summary = run_sort_time_backfill(try_lock=_no_lock, project_ids=ids)

    assert summary is not None and (summary.recorded, summary.unread) == (1, 0)
    assert sort_times_of(org_id) == {(readable.id, None, "report"): 1_000_100.0}
    assert sort_sentinel() == SortTimeSentinel(unread=[]) and not _progress_present()


@pytest.mark.parametrize("old_path", ["gone", "taken"])
def test_a_project_renamed_during_its_walk_is_listed_and_timed_again_at_its_new_path(
    tmp_path, org_deployment, backfills_pending, monkeypatch, owned_logger, old_path
):
    """A rename commits between the pass's query and the project's walk, so
    the walk reads the old path: the folder is gone there, or another
    project took the old name and its folders are there. The pass reads the
    project's row before it records, finds another path there, and records
    nothing: it lists the project, so the list reads every folder of it,
    and the next pass times the folders at the new path. Taking the gone
    folder for an empty root would leave the project with no row and
    unlisted, so the list would show none of its cards; taking the other
    project's folders would sort them as this project's."""
    org_id = str(uuid4())
    project = make_project(tmp_path, org_id)
    timed_artifact(project_root(project).base, "report", 1_000_100.0)
    run_held_root_backfill(try_lock=_no_lock, project_ids=[project.id])
    walked_path = Path(project.path)
    renamed = walked_path.with_name(f"{walked_path.name}-renamed")
    real = artifact_sort_times._time_project

    def renamed_before_the_walk(engine, *, held, **kwargs):
        _rename(held.project.id, to=renamed)
        if old_path == "taken":
            taker = Project(id=uuid4(), name=walked_path.name, path=str(walked_path), org_id=org_id)
            with Session(_engine()) as raw:
                raw.add(taker)
                raw.commit()
            timed_artifact(walked_path / ".anton" / "artifacts", "theirs", 1_000_900.0)
        return real(engine, held=held, **kwargs)

    logged = owned_logger(artifact_sort_times.logger.name, level=logging.WARNING)
    with monkeypatch.context() as renaming:
        renaming.setattr(artifact_sort_times, "_time_project", renamed_before_the_walk)
        summary = run_sort_time_backfill(try_lock=_no_lock, project_ids=[project.id])

    assert summary is not None and (summary.recorded, summary.unread) == (0, 1)
    assert sort_times_of(org_id) == {}
    assert sort_sentinel() == SortTimeSentinel(unread=[project.id])
    assert_log_fields(
        logged.output(), log_level="WARNING", event="artifact_sort_times_backfill_moved", project_id=str(project.id)
    )

    summary = run_sort_time_backfill(try_lock=_no_lock, project_ids=[project.id])

    assert summary is not None and (summary.recorded, summary.unread) == (1, 0)
    assert sort_times_of(org_id) == {(project.id, None, "report"): 1_000_100.0}
    assert sort_sentinel() == SortTimeSentinel(unread=[])


def test_a_project_whose_rename_moved_its_folder_but_has_not_committed_is_listed(
    tmp_path, org_deployment, backfills_pending, monkeypatch, owned_logger
):
    """A rename moves the project's folder before it commits the new path
    (ProjectService.stage_project_update), and the walk runs in between: it
    finds the folder gone at the path the project's row still holds. The
    pass finds the folder gone after its walk, so it lists the project
    rather than take it for one whose roots hold no folder, which would
    leave it with no row and unlisted, so the list would show none of its
    cards. The next pass, after the rename commits, times the folders at
    the new path."""
    org_id = str(uuid4())
    project = make_project(tmp_path, org_id)
    timed_artifact(project_root(project).base, "report", 1_000_100.0)
    run_held_root_backfill(try_lock=_no_lock, project_ids=[project.id])
    renamed = Path(project.path).with_name(f"{Path(project.path).name}-renamed")
    real = artifact_sort_times._time_project

    def moved_before_the_walk(engine, *, held, **kwargs):
        Path(held.project.path).rename(renamed)
        return real(engine, held=held, **kwargs)

    logged = owned_logger(artifact_sort_times.logger.name, level=logging.WARNING)
    with monkeypatch.context() as renaming:
        renaming.setattr(artifact_sort_times, "_time_project", moved_before_the_walk)
        summary = run_sort_time_backfill(try_lock=_no_lock, project_ids=[project.id])

    assert summary is not None and (summary.recorded, summary.unread) == (0, 1)
    assert sort_times_of(org_id) == {}
    assert sort_sentinel() == SortTimeSentinel(unread=[project.id])
    assert_log_fields(
        logged.output(), log_level="WARNING", event="artifact_sort_times_backfill_moved", project_id=str(project.id)
    )

    _commit_rename(project.id, to=renamed)
    summary = run_sort_time_backfill(try_lock=_no_lock, project_ids=[project.id])

    assert summary is not None and (summary.recorded, summary.unread) == (1, 0)
    assert sort_times_of(org_id) == {(project.id, None, "report"): 1_000_100.0}
    assert sort_sentinel() == SortTimeSentinel(unread=[])


def test_a_project_whose_folder_is_gone_for_good_is_listed_once_then_read_with_no_rows(
    tmp_path, org_deployment, backfills_pending
):
    """A project with a held root loses its folder, as a wiped volume leaves
    it until the project is opened again (ProjectService.ensure_dir_exists).
    The first pass cannot tell that from a rename that moved the folder and
    has not committed, so it lists the project. The pass over its list finds
    the folder still missing at the path the project's row holds, and counts
    the project as read, with no rows: the sentinel lists none, rather than
    list the project again on every pass."""
    org_id = str(uuid4())
    project = make_project(tmp_path, org_id)
    timed_artifact(project_root(project).base, "report", 1_000_100.0)
    run_held_root_backfill(try_lock=_no_lock, project_ids=[project.id])
    shutil.rmtree(project.path)

    summary = run_sort_time_backfill(try_lock=_no_lock, project_ids=[project.id])

    assert summary is not None and (summary.recorded, summary.unread) == (0, 1)
    assert sort_sentinel() == SortTimeSentinel(unread=[project.id])

    summary = run_sort_time_backfill(try_lock=_no_lock, project_ids=[project.id])

    assert summary is not None and (summary.recorded, summary.unread) == (0, 0)
    assert sort_times_of(org_id) == {}
    assert sort_sentinel() == SortTimeSentinel(unread=[])


@pytest.mark.parametrize("error", [errno.EACCES, errno.EPERM, errno.ELOOP], ids=["EACCES", "EPERM", "ELOOP"])
def test_a_project_folder_the_server_may_not_reach_is_listed_once_then_read_with_no_rows(
    tmp_path, org_deployment, backfills_pending, monkeypatch, error
):
    """After the held-root pass, the stat of a held project's folder fails
    with an error the walk reads as a root holding nothing (EACCES, EPERM,
    ELOOP), and so does the walk's listing of its root. The pass counts
    the folder as missing (walked_folder_is_gone), as the walk does: the
    first pass lists the project, as it lists one whose folder is gone, and
    the pass over its list counts it as read with no rows, rather than
    list it again on every pass, which would keep the startup task
    running."""
    org_id = str(uuid4())
    project = make_project(tmp_path, org_id)
    timed_artifact(project_root(project).base, "report", 1_000_100.0)
    run_held_root_backfill(try_lock=_no_lock, project_ids=[project.id])
    _refused_project_folder(monkeypatch, project, error=error)

    summary = run_sort_time_backfill(try_lock=_no_lock, project_ids=[project.id])

    assert summary is not None and (summary.recorded, summary.unread) == (0, 1)
    assert sort_sentinel() == SortTimeSentinel(unread=[project.id])

    summary = run_sort_time_backfill(try_lock=_no_lock, project_ids=[project.id])

    assert summary is not None and (summary.recorded, summary.unread) == (0, 0)
    assert sort_times_of(org_id) == {}
    assert sort_sentinel() == SortTimeSentinel(unread=[])


def test_a_pass_over_the_sentinels_list_lists_again_a_project_whose_folder_moves_during_its_walk(
    tmp_path, org_deployment, backfills_pending, monkeypatch, owned_logger
):
    """The sentinel lists the project, and a rename moves its folder once the
    pass has found it there and before the pass reads its folders' times.
    The pass finds the folder gone after its walk, so it lists the project
    again rather than count it as read with no rows, which would leave the
    list showing none of its cards. The next pass, after the rename
    commits, times the folders at the new path."""
    org_id = str(uuid4())
    project = make_project(tmp_path, org_id)
    timed_artifact(project_root(project).base, "report", 1_000_100.0)
    run_held_root_backfill(try_lock=_no_lock, project_ids=[project.id])
    write_sentinel(_engine(), SENTINEL_KEY, value=SortTimeSentinel(unread=[project.id]).model_dump_json())
    renamed = Path(project.path).with_name(f"{Path(project.path).name}-renamed")
    real = artifact_sort_times._times_in

    def moved_during_the_walk(source):
        Path(project.path).rename(renamed)
        return real(source)

    logged = owned_logger(artifact_sort_times.logger.name, level=logging.WARNING)
    with monkeypatch.context() as renaming:
        renaming.setattr(artifact_sort_times, "_times_in", moved_during_the_walk)
        summary = run_sort_time_backfill(try_lock=_no_lock, project_ids=[project.id])

    assert summary is not None and (summary.recorded, summary.unread) == (0, 1)
    assert sort_times_of(org_id) == {}
    assert sort_sentinel() == SortTimeSentinel(unread=[project.id])
    assert_log_fields(
        logged.output(), log_level="WARNING", event="artifact_sort_times_backfill_moved", project_id=str(project.id)
    )

    _commit_rename(project.id, to=renamed)
    summary = run_sort_time_backfill(try_lock=_no_lock, project_ids=[project.id])

    assert summary is not None and (summary.recorded, summary.unread) == (1, 0)
    assert sort_times_of(org_id) == {(project.id, None, "report"): 1_000_100.0}
    assert sort_sentinel() == SortTimeSentinel(unread=[])


def test_a_stop_between_two_roots_of_a_project_records_neither(
    tmp_path, org_deployment, backfills_pending, monkeypatch
):
    """Shutdown lands once the pass has read one of a project's two held
    roots. The walk ends before it opens the other, records neither, and
    writes no sentinel, so the next start walks the project again."""
    org_id = str(uuid4())
    project = make_project(tmp_path, org_id)
    timed_artifact(project_root(project).base, "report", 1_000_100.0)
    owned = make_conversation(project, OWNER)
    timed_artifact(_workspace_root(project, owned), "old-report", 1_000_200.0)
    run_held_root_backfill(try_lock=_no_lock, project_ids=[project.id])
    stop = threading.Event()
    opened: list[Path] = []
    real_times_in = artifact_sort_times._times_in

    def times_in_then_stop(source):
        opened.append(Path(source.base))
        found = real_times_in(source)
        stop.set()
        return found

    with monkeypatch.context() as stopping:
        stopping.setattr(artifact_sort_times, "_times_in", times_in_then_stop)
        assert run_sort_time_backfill(try_lock=_no_lock, project_ids=[project.id], stop=stop) is None

    assert len(opened) == 1
    assert sort_times_of(org_id) == {}
    assert sort_sentinel() is None
    assert read_progress(_engine(), PROGRESS_KEY) == BackfillProgress()


def test_a_stopped_backfill_writes_no_sentinel(tmp_path, org_deployment, backfills_pending):
    project = make_project(tmp_path, str(uuid4()))
    timed_artifact(project_root(project).base, "report", 1_000_100.0)
    run_held_root_backfill(try_lock=_no_lock, project_ids=[project.id])
    stop = threading.Event()
    stop.set()

    assert run_sort_time_backfill(try_lock=_no_lock, project_ids=[project.id], stop=stop) is None
    assert sort_sentinel() is None


def test_a_pass_over_the_sentinels_list_keeps_no_progress(tmp_path, org_deployment, backfills_pending, monkeypatch):
    """Once the sentinel exists, a pass that walks its list saves no
    progress, even when it is stopped: a progress row would make a rerun
    after the sentinel's deletion skip projects."""
    org_id = str(uuid4())
    project = make_project(tmp_path, org_id)
    timed_artifact(project_root(project).base, "report", 1_000_100.0)
    run_held_root_backfill(try_lock=_no_lock, project_ids=[project.id])
    write_sentinel(_engine(), SENTINEL_KEY, value=SortTimeSentinel(unread=[project.id]).model_dump_json())
    stop = threading.Event()
    real = artifact_sort_times._time_project

    def time_then_stop(engine, **kwargs):
        read = real(engine, **kwargs)
        stop.set()
        return read

    with monkeypatch.context() as stopping:
        stopping.setattr(artifact_sort_times, "_time_project", time_then_stop)
        run_sort_time_backfill(try_lock=_no_lock, project_ids=[project.id], stop=stop)
    assert not _progress_present()

    run_sort_time_backfill(try_lock=_no_lock, project_ids=[project.id])

    assert not _progress_present()
    assert sort_sentinel() == SortTimeSentinel(unread=[])


def test_each_pass_logs_its_start_or_why_it_walks_nothing(
    tmp_path, org_deployment, backfills_pending, monkeypatch, caplog, owned_logger
):
    """An operator tells a walk that runs from one that never started, at
    the WARNING level every hosted deployment logs at, with every field the
    console keeps (structured_logging.EVENT_FIELDS). The count is the
    projects the pass walks, not their held roots."""
    org_id = str(uuid4())
    projects = sorted((make_project(tmp_path, org_id) for _ in range(3)), key=lambda p: p.id)
    for project in projects:
        timed_artifact(project_root(project).base, "report", 1_000_100.0)
    # Four held roots in three projects: the count is the projects.
    timed_artifact(_workspace_root(projects[0], make_conversation(projects[0], OWNER)), "old-report", 1_000_200.0)
    last = projects[-1]
    ids = [project.id for project in projects]
    stop = threading.Event()
    real_save = artifact_sort_times.save_progress

    def save_progress(engine, key, *, progress):
        real_save(engine, key, progress=progress)
        stop.set()

    @contextmanager
    def taken(_engine, _key):
        yield False

    logged = owned_logger(artifact_sort_times.logger.name, level=logging.WARNING)
    run_sort_time_backfill(try_lock=_no_lock, project_ids=ids)
    run_held_root_backfill(try_lock=_no_lock, project_ids=ids)
    run_sort_time_backfill(try_lock=taken, project_ids=ids)
    with monkeypatch.context() as stopping:
        stopping.setattr(artifact_sort_times, "save_progress", save_progress)
        run_sort_time_backfill(try_lock=_no_lock, project_ids=ids, stop=stop)
    with monkeypatch.context() as failing:
        _failing_scandir(failing, unreadable=project_root(last).base)
        run_sort_time_backfill(try_lock=_no_lock, project_ids=ids)
    run_sort_time_backfill(try_lock=_no_lock, project_ids=ids)
    run_sort_time_backfill(try_lock=_no_lock, project_ids=ids)

    assert _pass_lines(caplog) == [
        ("artifact_sort_times_backfill_skipped", "held_pending", None),
        ("artifact_sort_times_backfill_skipped", "locked", None),
        ("artifact_sort_times_backfill_started", "first", 3),
        ("artifact_sort_times_backfill_started", "resumed", 2),
        ("artifact_sort_times_backfill_started", "unread", 1),
        ("artifact_sort_times_backfill_skipped", "finished", None),
    ]
    output = logged.output()
    assert_log_fields(
        output, log_level="WARNING", event="artifact_sort_times_backfill_skipped", reason="held_pending",
        message="artifact_sort_times backfill skipped its walk",
    )
    assert_log_fields(
        output, log_level="WARNING", event="artifact_sort_times_backfill_started", reason="resumed", count=2
    )
    assert_log_fields(output, log_level="WARNING", event="artifact_sort_times_backfill_stopped")
    assert_log_fields(
        output, log_level="WARNING", event="artifact_sort_times_backfill_unread",
        project_id=str(last.id), error_type="OSError",
    )
    assert_log_fields(
        output, log_level="WARNING", event="artifact_sort_times_backfill_summary", outcome="recorded", count=1
    )
    assert_log_fields(
        output, log_level="WARNING", event="artifact_sort_times_backfill_summary", outcome="unread", count=1
    )
    assert_log_fields(
        output, log_level="WARNING", event="artifact_sort_times_backfill_summary", outcome="held_pending", count=0
    )
    assert sort_sentinel() == SortTimeSentinel(unread=[])


def test_the_desktop_runs_no_backfill(tmp_path, backfills_pending, caplog):
    with caplog.at_level(logging.WARNING, logger=artifact_sort_times.logger.name):
        assert run_sort_time_backfill(try_lock=_no_lock, project_ids=[]) is None
    assert _pass_lines(caplog) == [("artifact_sort_times_backfill_skipped", "not_org", None)]
    assert sort_sentinel() is None


def test_a_pass_that_takes_the_lock_after_another_replica_finished_walks_nothing(
    tmp_path, org_deployment, backfills_pending, monkeypatch, caplog
):
    """The pass read no sentinel, then another replica finished the walk and
    let go of the lock before this one took it. The pass reads the sentinel
    again under the lock, so it walks no project a second time."""
    project = make_project(tmp_path, str(uuid4()))
    timed_artifact(project_root(project).base, "report", 1_000_100.0)
    run_held_root_backfill(try_lock=_no_lock, project_ids=[project.id])
    walked = _walked(monkeypatch)

    @contextmanager
    def finished_meanwhile(engine, _key):
        write_sentinel(engine, SENTINEL_KEY, value=SortTimeSentinel().model_dump_json())
        yield True

    with caplog.at_level(logging.WARNING, logger=artifact_sort_times.logger.name):
        assert run_sort_time_backfill(try_lock=finished_meanwhile, project_ids=[project.id]) is None

    assert walked == []
    assert _pass_lines(caplog) == [("artifact_sort_times_backfill_skipped", "finished", None)]


def test_a_pass_that_takes_the_lock_after_another_replica_listed_a_project_walks_only_that_one(
    tmp_path, org_deployment, backfills_pending, monkeypatch, caplog
):
    org_id = str(uuid4())
    listed, other = (make_project(tmp_path, org_id) for _ in range(2))
    for project in (listed, other):
        timed_artifact(project_root(project).base, "report", 1_000_100.0)
    run_held_root_backfill(try_lock=_no_lock, project_ids=[listed.id, other.id])
    walked = _walked(monkeypatch)

    @contextmanager
    def listed_meanwhile(engine, _key):
        write_sentinel(engine, SENTINEL_KEY, value=SortTimeSentinel(unread=[listed.id]).model_dump_json())
        yield True

    with caplog.at_level(logging.WARNING, logger=artifact_sort_times.logger.name):
        summary = run_sort_time_backfill(try_lock=listed_meanwhile, project_ids=[listed.id, other.id])

    assert summary is not None and (summary.recorded, summary.unread) == (1, 0)
    assert walked == [listed.id]
    assert sort_times_of(org_id) == {(listed.id, None, "report"): 1_000_100.0}
    assert sort_sentinel() == SortTimeSentinel(unread=[])
    assert not _progress_present()
    assert _pass_lines(caplog) == [("artifact_sort_times_backfill_started", "unread", 1)]


def test_a_sentinel_value_it_cannot_read_reads_as_no_sentinel(tmp_path, org_deployment, backfills_pending, caplog):
    """A sentinel row written by hand never fails a reader: it reads as no
    sentinel, logged, and the next pass walks as a first pass and rewrites
    the value."""
    project = make_project(tmp_path, str(uuid4()))
    timed_artifact(project_root(project).base, "report", 1_000_100.0)
    run_held_root_backfill(try_lock=_no_lock, project_ids=[project.id])
    write_sentinel(_engine(), SENTINEL_KEY, value="1")

    with caplog.at_level(logging.WARNING, logger=artifact_sort_times.logger.name):
        assert sort_sentinel() is None
        run_sort_time_backfill(try_lock=_no_lock, project_ids=[project.id])

    assert [
        getattr(record, "event", None) for record in caplog.records if record.name == artifact_sort_times.logger.name
    ][:2] == ["artifact_sort_times_sentinel_unreadable", "artifact_sort_times_sentinel_unreadable"]
    assert ("artifact_sort_times_backfill_started", "first", 1) in _pass_lines(caplog)
    assert sort_sentinel() == SortTimeSentinel(unread=[])


def test_the_backfill_never_opens_a_project_in_another_partition(
    tmp_path, org_deployment, backfills_pending, monkeypatch
):
    """A held row may name a project of the organization whose folder sits
    in another partition of the shared root; the pass leaves it out, as the
    held-root pass does, and still writes its sentinel."""
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
        timed_artifact(project_root(elsewhere).base, "in-another-partition", 1_000_100.0)
        home = make_project(tmp_path, org_id)
        timed_artifact(project_root(home).base, "ours", 1_000_200.0)
        run_held_root_backfill(try_lock=_no_lock, project_ids=[elsewhere.id, home.id])
        with scoped(org_id) as session:
            hold_root(session, project_id=elsewhere.id)
        opened: list[Path] = []
        real_times_in = artifact_sort_times._times_in

        def times_in(source):
            opened.append(Path(source.base))
            return real_times_in(source)

        monkeypatch.setattr(artifact_sort_times, "_times_in", times_in)

        summary = run_sort_time_backfill(try_lock=_no_lock, project_ids=[elsewhere.id, home.id])

        assert summary is not None and (summary.recorded, summary.unread) == (1, 0)
        assert opened == [project_root(home).base]
        assert sort_times_of(org_id) == {(home.id, None, "ours"): 1_000_200.0}
        assert sort_sentinel() == SortTimeSentinel(unread=[])
    finally:
        with Session(_engine()) as raw:
            raw.exec(delete(ArtifactSortTime).where(ArtifactSortTime.project_id == elsewhere.id))
            raw.exec(delete(HeldArtifactRoot).where(HeldArtifactRoot.project_id == elsewhere.id))
            raw.exec(delete(Project).where(Project.id == elsewhere.id))
            raw.commit()
        shutil.rmtree(partition, ignore_errors=True)


# The writers.


def test_a_write_keeps_the_newest_time_and_skips_an_unchanged_one(tmp_path):
    org_id = str(uuid4())
    project = make_project(tmp_path, org_id)
    updates: list[str] = []

    def count_updates(_conn, _cursor, statement, *_args):
        if statement.lstrip().upper().startswith("UPDATE ARTIFACT_SORT_TIMES"):
            updates.append(statement)

    with scoped(org_id) as session:
        record_sort_times(session, project_id=project.id, times={"report": 2_000.0, "chart": 1_000.0})
    event.listen(_engine(), "before_cursor_execute", count_updates)
    try:
        with scoped(org_id) as session:
            record_sort_times(session, project_id=project.id, times={"report": 1_500.0, "chart": 1_000.0})
        assert updates == []
        with scoped(org_id) as session:
            record_sort_times(session, project_id=project.id, times={"chart": 3_000.0})
        assert len(updates) == 1
    finally:
        event.remove(_engine(), "before_cursor_execute", count_updates)

    assert sort_times_of(org_id) == {(project.id, None, "report"): 2_000.0, (project.id, None, "chart"): 3_000.0}


def _locked_statements(statements: list[tuple[str, tuple]]):
    """A statement listener that notes, for each statement on
    artifact_sort_times or a row it points at, its kind (`lock`, a select
    FOR UPDATE or FOR KEY SHARE, which SQLite does not run but the statement
    carries; `select`, `insert`, `update` or `delete`), its table, and the
    slugs it names."""

    def listen(_conn, _cursor, statement, _parameters, context, _executemany):
        compiled = getattr(context, "compiled", None)
        if compiled is None:
            return
        verb = statement.lstrip().split(None, 1)[0].lower()
        table = next(
            (name for name in ("artifact_sort_times", "projects", "conversations") if f" {name}" in statement), None
        )
        if table is None:
            return
        if verb == "select" and getattr(compiled.statement, "_for_update_arg", None) is not None:
            verb = "lock"
        slugs: list[str] = []
        for parameters in context.compiled_parameters:
            for key, value in parameters.items():
                if key.startswith("slug"):
                    slugs.extend(value if isinstance(value, (list, tuple)) else [value])
        statements.append((verb, table, tuple(slugs)))

    return listen


def test_a_write_that_changes_nothing_locks_no_row(tmp_path):
    """A heal, the sweep or a turn end with no pod list records every folder
    of a root, most of them unchanged. An unlocked read leaves those out,
    so recording an unchanged root locks no row and writes none; a folder
    whose time rises is the only row locked, after the project's row."""
    org_id = str(uuid4())
    project = make_project(tmp_path, org_id)
    times = {f"folder-{index:03d}": 1_000.0 + index for index in range(50)}
    with scoped(org_id) as session:
        record_sort_times(session, project_id=project.id, times=times)
    statements: list[tuple[str, str, tuple]] = []
    listen = _locked_statements(statements)
    event.listen(_engine(), "before_cursor_execute", listen)
    try:
        with scoped(org_id) as session:
            unchanged = record_sort_times(session, project_id=project.id, times=times)
        quiet = list(statements)
        statements.clear()
        with scoped(org_id) as session:
            risen = record_sort_times(session, project_id=project.id, times={**times, "folder-007": 9_000.0})
    finally:
        event.remove(_engine(), "before_cursor_execute", listen)

    assert unchanged == 0
    assert [verb for verb, _table, _slugs in quiet] == ["select"]
    assert risen == 1
    assert [(verb, table, slugs) for verb, table, slugs in statements if verb != "select"] == [
        ("lock", "projects", ()),
        ("lock", "artifact_sort_times", ("folder-007",)),
        ("update", "artifact_sort_times", ()),
    ]


def _another_writer_adds(org_id: str, project_id: UUID, slug: str, sort_time: float) -> None:
    """Another writer's row, committed on its own connection."""
    with Session(_engine()) as raw:
        raw.add(ArtifactSortTime(org_id=org_id, project_id=project_id, slug=slug, sort_time=sort_time))
        raw.commit()


def _rows_then(monkeypatch, *others) -> list[int]:
    """After each of the first locked reads (_rows), another writer commits
    the row `others` names for that read, so this write's insert of it fails
    on the unique index. Returns the number of folders each _write was
    handed, in order."""
    real_rows, real_write = artifact_sort_times._rows, artifact_sort_times._write
    pending = list(others)
    batches: list[int] = []

    def rows(session, **kwargs):
        found = real_rows(session, **kwargs)
        if pending:
            _another_writer_adds(*pending.pop(0))
        return found

    def write(session, *, times, **kwargs):
        batches.append(len(times))
        return real_write(session, times=times, **kwargs)

    monkeypatch.setattr(artifact_sort_times, "_rows", rows)
    monkeypatch.setattr(artifact_sort_times, "_write", write)
    return batches


def test_a_write_that_finds_another_writers_newer_row_keeps_it(tmp_path, monkeypatch):
    """Another writer adds the folder's row, with a newer time, between this
    write's locked read and its insert: a lock holds only rows that exist.
    The insert fails on the unique index, and the retry reads that row and
    leaves its time."""
    org_id = str(uuid4())
    project = make_project(tmp_path, org_id)
    batches = _rows_then(monkeypatch, (org_id, project.id, "report", 2_000.0))
    with scoped(org_id) as session:
        written = record_sort_times(session, project_id=project.id, times={"report": 1_000.0})

    assert (written, batches) == (0, [1, 1])
    assert sort_times_of(org_id) == {(project.id, None, "report"): 2_000.0}


def test_a_write_retried_after_another_writers_row_writes_every_folder_of_the_batch(tmp_path, monkeypatch):
    """Another writer adds one folder's row, with an older time, between this
    write's locked read and its insert, so the batch's commit fails on the
    unique index. The batch is written again once, whole: the newer time
    over the other writer's row, and the batch's other new folder."""
    org_id = str(uuid4())
    project = make_project(tmp_path, org_id)
    batches = _rows_then(monkeypatch, (org_id, project.id, "report", 500.0))
    with scoped(org_id) as session:
        written = record_sort_times(session, project_id=project.id, times={"report": 1_000.0, "chart": 2_000.0})

    assert (written, batches) == (2, [2, 2])
    assert sort_times_of(org_id) == {(project.id, None, "report"): 1_000.0, (project.id, None, "chart"): 2_000.0}


def test_a_batch_that_fails_twice_is_written_folder_by_folder(tmp_path, monkeypatch):
    """Other writers add a row of the batch before each of its first two
    commits. The write then goes folder by folder, so every folder is still
    written: each newer time over the other writer's older row."""
    org_id = str(uuid4())
    project = make_project(tmp_path, org_id)
    batches = _rows_then(
        monkeypatch, (org_id, project.id, "report", 500.0), (org_id, project.id, "chart", 100.0)
    )
    with scoped(org_id) as session:
        written = record_sort_times(session, project_id=project.id, times={"report": 1_000.0, "chart": 2_000.0})

    assert (written, batches) == (2, [2, 2, 1, 1])
    assert sort_times_of(org_id) == {(project.id, None, "report"): 1_000.0, (project.id, None, "chart"): 2_000.0}


def _another_writer_sets(project_id: UUID, slug: str, sort_time: float) -> None:
    """Another writer moves a folder's row to `sort_time`, committed on its
    own connection."""
    with Session(_engine()) as raw:
        row = raw.exec(
            select(ArtifactSortTime).where(ArtifactSortTime.project_id == project_id, ArtifactSortTime.slug == slug)
        ).one()
        row.sort_time = sort_time
        raw.add(row)
        raw.commit()


def _rows_held_by(session: ScopedSession, project_id: UUID) -> list[ArtifactSortTime]:
    """The rows of `project_id`, loaded into `session`. The caller keeps the
    list, so the session holds the rows as a unit's session holds one it
    still refers to across its commits (cowork.db.units.unit_session,
    expire_on_commit=False), such as the row reconcile_sort_times moved
    back before it records the listing."""
    return list(session.exec(session.select(ArtifactSortTime).where(ArtifactSortTime.project_id == project_id)).all())


def test_a_write_reads_the_rows_it_locks_from_the_database_not_the_session(tmp_path, monkeypatch):
    """This write's unlocked read finds the folder's row at 1000, and its
    session holds that row. A turn end then commits 3000 before this write
    locks the row. The locked read (_rows) takes the committed time, not
    the one the session holds, so the write leaves 3000 rather than put its
    older 1500 over it."""
    org_id = str(uuid4())
    project = make_project(tmp_path, org_id)
    with scoped(org_id) as session:
        record_sort_times(session, project_id=project.id, times={"report": 1_000.0})
    real_times_held = artifact_sort_times._times_held
    held_rows: list[ArtifactSortTime] = []

    def times_held_then_a_turn_end(session, **kwargs):
        held = real_times_held(session, **kwargs)
        held_rows.extend(_rows_held_by(session, project.id))
        _another_writer_sets(project.id, "report", 3_000.0)
        return held

    monkeypatch.setattr(artifact_sort_times, "_times_held", times_held_then_a_turn_end)
    with unit_session(scope=TenantScope(org_mode=True, org_id=org_id)) as session:
        written = record_sort_times(session, project_id=project.id, times={"report": 1_500.0})

    assert [row.sort_time for row in held_rows] == [3_000.0]
    assert written == 0
    assert sort_times_of(org_id) == {(project.id, None, "report"): 3_000.0}


def test_a_write_reads_the_times_it_compares_from_the_database_not_the_session(tmp_path):
    """The session holds the folder's row at 3000. A self-heal then moves
    the row back to 1000, the time of an older copy now in its place, and
    the folder is written again at 2000. The unlocked read (_times_held)
    takes the committed 1000, so 2000 is written; the 3000 the session
    holds would leave the folder out, and the row at 1000."""
    org_id = str(uuid4())
    project = make_project(tmp_path, org_id)
    with scoped(org_id) as session:
        record_sort_times(session, project_id=project.id, times={"report": 3_000.0})

    with unit_session(scope=TenantScope(org_mode=True, org_id=org_id)) as session:
        held_rows = _rows_held_by(session, project.id)
        _another_writer_sets(project.id, "report", 1_000.0)
        written = record_sort_times(session, project_id=project.id, times={"report": 2_000.0})

    assert len(held_rows) == 1
    assert written == 1
    assert sort_times_of(org_id) == {(project.id, None, "report"): 2_000.0}


def test_a_name_the_list_would_not_open_gets_no_row(tmp_path):
    """A dot-named copy, which the walk never shows, and a name that is not
    valid UTF-8, which no row can store, are left out without failing the
    write of the others."""
    org_id = str(uuid4())
    project = make_project(tmp_path, org_id)
    with scoped(org_id) as session:
        record_sort_times(
            session, project_id=project.id, times={"report": 1.0, ".report.bak": 2.0, "bad\udcff": 3.0, "../up": 4.0},
        )

    assert sort_times_of(org_id) == {(project.id, None, "report"): 1.0}


def test_the_listable_folder_name_rule():
    assert is_listable_folder_name("report")
    for name in (".report.bak", ".locks", "bad\udcff", "a/b", "..", "", "nul\0"):
        assert not is_listable_folder_name(name), name


def test_the_desktop_records_no_time(tmp_path):
    project = make_project(tmp_path, None)
    conversation_id = make_conversation(project, None)

    with scoped(None) as session:
        record_new_artifacts(
            session, conversation_id=conversation_id, project_id=project.id, slugs=[], creator=None,
            sort_times={"report": 1.0},
        )
        record_sort_times(session, project_id=project.id, times={"chart": 1.0})

    with Session(_engine()) as raw:
        assert raw.exec(select(ArtifactSortTime).where(ArtifactSortTime.project_id == project.id)).first() is None


def test_a_turn_end_records_its_folders_times_and_replaces_older_ones(tmp_path):
    org_id = str(uuid4())
    project = make_project(tmp_path, org_id)
    conversation_id = make_conversation(project, OWNER)

    for times in ({"report": 1.0, "chart": 2.0}, {"report": 5.0}):
        with scoped(org_id) as session:
            record_new_artifacts(
                session, conversation_id=conversation_id, project_id=project.id, slugs=[], creator=OWNER,
                sort_times=times,
            )

    assert sort_times_of(org_id) == {(project.id, None, "report"): 5.0, (project.id, None, "chart"): 2.0}


def _row_ids(org_id: str) -> dict[tuple, UUID]:
    with Session(_engine()) as raw:
        rows = raw.exec(select(ArtifactSortTime).where(ArtifactSortTime.org_id == org_id)).all()
        return {(row.project_id, row.conversation_id, row.slug): row.id for row in rows}


def test_the_rows_a_heal_reads_before_its_listing_are_the_shared_roots_rows(tmp_path):
    org_id = str(uuid4())
    project, other = make_project(tmp_path, org_id), make_project(tmp_path, org_id)
    conversation_id = make_conversation(project, OWNER)
    with scoped(org_id) as session:
        record_sort_times(session, project_id=project.id, times={"report": 1.0, "chart": 2.0})
        record_sort_times(session, project_id=project.id, conversation_id=conversation_id, times={"legacy": 3.0})
        record_sort_times(session, project_id=other.id, times={"elsewhere": 4.0})
        rows = sort_time_rows(session, project_id=project.id)
    ids = _row_ids(org_id)

    assert rows == {
        "report": SortTimeRow(ids[(project.id, None, "report")], 1.0),
        "chart": SortTimeRow(ids[(project.id, None, "chart")], 2.0),
    }
    with scoped(None) as session:
        assert sort_time_rows(session, project_id=project.id) == {}


@pytest.mark.parametrize("complete", [True, False], ids=["root-read", "root-unread"])
def test_a_heal_writes_every_time_it_read_and_drops_only_rows_gone_before_its_listing(tmp_path, complete):
    """A listed time replaces the row's, an older one too while the row is
    still the one read before the listing; a folder the listing found
    holding an id keeps its row even
    when its time didn't read; a gone folder's row goes only when the
    listing read the root itself, and only when it existed before the
    listing began, so a row a turn end wrote meanwhile stays."""
    org_id = str(uuid4())
    project = make_project(tmp_path, org_id)
    with scoped(org_id) as session:
        record_sort_times(
            session, project_id=project.id,
            times={"kept": 1.0, "moved": 1.0, "no-time": 2.0, "unread": 3.0, "gone": 4.0},
        )
        droppable = sort_time_rows(session, project_id=project.id)
        # A turn end records a folder the listing, made before it, missed.
        record_sort_times(session, project_id=project.id, times={"fresh": 5.0})
        reconcile_sort_times(
            session,
            project_id=project.id,
            listed={"kept": 0.5, "moved": 9.0, "copied": 7.0},
            kept={"kept", "moved", "no-time", "copied", "unread"},
            droppable=droppable,
            complete=complete,
        )

    expected = {"kept": 0.5, "moved": 9.0, "no-time": 2.0, "unread": 3.0, "copied": 7.0, "fresh": 5.0}
    if not complete:
        expected["gone"] = 4.0
    assert sort_times_of(org_id) == {(project.id, None, slug): at for slug, at in expected.items()}


def test_reconcile_drops_a_folder_with_no_metadata(tmp_path):
    org_id = str(uuid4())
    project = make_project(tmp_path, org_id)
    with scoped(org_id) as session:
        record_sort_times(session, project_id=project.id, times={"broken": 1.0})
        droppable = sort_time_rows(session, project_id=project.id)
        reconcile_sort_times(
            session, project_id=project.id, listed={}, kept=set(), droppable=droppable, complete=True
        )

    assert sort_times_of(org_id) == {}


def test_a_heal_keeps_a_row_a_task_move_brought_under_a_gone_folders_slug(tmp_path):
    """The destination has a row for `report`, a folder deleted outside a
    turn. A heal reads the rows, then lists the root and finds no `report`.
    A task move then brings a `report` in: it drops the gone folder's row
    and moves the source's row under that slug, with the source row's id.
    Both rows hold the same time, as a copy made with `cp -p` and its
    original do. The heal drops only the row it read, matched by id as well
    as time, so the moved folder keeps its row."""
    org_id = str(uuid4())
    source, dest = make_project(tmp_path, org_id), make_project(tmp_path, org_id)
    with scoped(org_id) as session:
        record_sort_times(session, project_id=source.id, times={"report": 1_000.0})
        record_sort_times(session, project_id=dest.id, times={"report": 1_000.0, "keep": 3_000.0})
    with scoped(org_id) as session:
        droppable = sort_time_rows(session, project_id=dest.id)
    with scoped(org_id) as session:
        move_sort_times(session, project_id=source.id, moved={"report": "report"}, to_project_id=dest.id)
    with scoped(org_id) as session:
        reconcile_sort_times(
            session, project_id=dest.id, listed={"keep": 3_000.0}, kept={"keep"}, droppable=droppable, complete=True
        )

    assert sort_times_of(org_id) == {(dest.id, None, "report"): 1_000.0, (dest.id, None, "keep"): 3_000.0}


def test_a_heal_keeps_a_row_a_turn_end_moved_to_a_newer_time(tmp_path):
    """A heal reads the rows, then lists the root while `report` is deleted
    outside a turn. A turn then makes `report` again, and its turn end
    writes the newer time. The heal's listing found no `report`, but the
    row is no longer the one it read, so it stays."""
    org_id = str(uuid4())
    project = make_project(tmp_path, org_id)
    with scoped(org_id) as session:
        record_sort_times(session, project_id=project.id, times={"report": 1_000.0})
        droppable = sort_time_rows(session, project_id=project.id)
        record_sort_times(session, project_id=project.id, times={"report": 2_000.0})
        reconcile_sort_times(session, project_id=project.id, listed={}, kept=set(), droppable=droppable, complete=True)

    assert sort_times_of(org_id) == {(project.id, None, "report"): 2_000.0}


@pytest.mark.parametrize("complete", [True, False])
def test_a_heal_moves_a_row_back_to_the_older_time_its_folder_now_holds(tmp_path, complete):
    """`report` is replaced by an older copy under the same name, as
    `rm -rf report && mv draft report` does. A walk sorts it by the older
    `metadata.json`, and so does the heal's listing, which writes that
    time over the row it read before the listing."""
    org_id = str(uuid4())
    project = make_project(tmp_path, org_id)
    with scoped(org_id) as session:
        record_sort_times(session, project_id=project.id, times={"report": 2_000.0})
        droppable = sort_time_rows(session, project_id=project.id)
        reconcile_sort_times(
            session, project_id=project.id, listed={"report": 1_000.0}, kept={"report"}, droppable=droppable,
            complete=complete,
        )

    assert sort_times_of(org_id) == {(project.id, None, "report"): 1_000.0}


def test_a_heal_never_moves_back_a_row_a_turn_end_moved_while_it_listed(tmp_path):
    """The heal's listing read `report` at 1000. A turn end then edits it
    and writes 3000. The row is no longer the one the heal read before its
    listing, so the heal's older time never replaces it."""
    org_id = str(uuid4())
    project = make_project(tmp_path, org_id)
    with scoped(org_id) as session:
        record_sort_times(session, project_id=project.id, times={"report": 2_000.0})
        droppable = sort_time_rows(session, project_id=project.id)
        record_sort_times(session, project_id=project.id, times={"report": 3_000.0})
        reconcile_sort_times(
            session, project_id=project.id, listed={"report": 1_000.0}, kept={"report"}, droppable=droppable,
            complete=True,
        )

    assert sort_times_of(org_id) == {(project.id, None, "report"): 3_000.0}


def test_a_task_move_carries_each_folders_time(tmp_path):
    org_id, owner = str(uuid4()), str(uuid4())
    source, dest = make_project(tmp_path, org_id), make_project(tmp_path, org_id)
    conversation_id = make_conversation(source, owner)
    write_artifact(project_root(source).base, "report", conversation_id)
    _index(conversation_id, source.id, "report")
    with scoped(org_id) as session:
        from cowork.services import artifact_ownership as ownership

        ownership.record_artifact_owner(session, source.id, "report", owner, action="create")
        hold_root(session, project_id=source.id)
        record_sort_times(session, project_id=source.id, times={"report": 1_000_100.0})

    assert _relocate(org_id, owner, conversation_id, source.id, dest.id) == 1

    assert sort_times_of(org_id) == {(dest.id, None, "report"): 1_000_100.0}


def test_a_move_onto_a_gone_folders_row_replaces_it(tmp_path):
    """The destination still has a row under a moved folder's new slug, for
    a folder deleted outside a turn. The move clears it first, so every
    moved row lands in one commit and the source keeps none."""
    org_id = str(uuid4())
    source, dest = make_project(tmp_path, org_id), make_project(tmp_path, org_id)
    with scoped(org_id) as session:
        record_sort_times(session, project_id=source.id, times={"report": 1_000.0, "chart": 2_000.0})
        record_sort_times(session, project_id=dest.id, times={"report": 9_000.0, "keep": 3_000.0})
        move_sort_times(session, project_id=source.id, moved={"report": "report", "chart": "chart"}, to_project_id=dest.id)

    assert sort_times_of(org_id) == {
        (dest.id, None, "report"): 1_000.0,
        (dest.id, None, "chart"): 2_000.0,
        (dest.id, None, "keep"): 3_000.0,
    }


def test_a_move_gives_its_rows_their_new_slugs_in_slug_order(tmp_path):
    """A writer that adds rows under the destination's new slugs, such as
    the destination's self-heal listing the folders just moved there, adds
    them in slug order (_write). The move takes the unique index's entries
    for the new slugs in that order too, one row at a time, so the two
    queue rather than deadlock on PostgreSQL. The rows' ids sort the other
    way round, as a single flush would send them."""
    org_id = str(uuid4())
    source, dest = make_project(tmp_path, org_id), make_project(tmp_path, org_id)
    moved = {"a-old": "z-new", "b-old": "y-new", "c-old": "x-new"}
    with Session(_engine()) as raw:
        for row_id, slug in zip(sorted(uuid4() for _ in moved), moved):
            raw.add(ArtifactSortTime(id=row_id, org_id=org_id, project_id=source.id, slug=slug, sort_time=1.0))
        raw.commit()
    statements: list[tuple[str, str, tuple]] = []
    listen = _locked_statements(statements)
    event.listen(_engine(), "before_cursor_execute", listen)
    try:
        with scoped(org_id) as session:
            move_sort_times(session, project_id=source.id, moved=moved, to_project_id=dest.id)
    finally:
        event.remove(_engine(), "before_cursor_execute", listen)

    updated = [slug for verb, table, slugs in statements if verb == "update" for slug in slugs]
    assert updated == ["x-new", "y-new", "z-new"]
    assert sort_times_of(org_id) == {(dest.id, None, slug): 1.0 for slug in moved.values()}


def test_a_deleted_folder_takes_its_time_with_it(tmp_path):
    from cowork.api.v1.endpoints.artifacts import _forget_deleted_folder

    org_id = str(uuid4())
    project = make_project(tmp_path, org_id)
    with scoped(org_id) as session:
        record_sort_times(session, project_id=project.id, times={"report": 1.0, "chart": 2.0})
        _forget_deleted_folder(session, artifact_id="a" * 32, location=Location(project.id, None, "report"))

    assert sort_times_of(org_id) == {(project.id, None, "chart"): 2.0}


def test_a_delete_whose_location_drop_fails_still_drops_the_sort_time(tmp_path, monkeypatch):
    """The two rows of a deleted folder go in their own tries, so a failed
    drop of the location row never keeps the sort time, which would sort a
    folder that is gone among the newest."""
    from cowork.api.v1.endpoints import artifacts as endpoint

    def refuse(session, **_):
        raise RuntimeError("location drop refused")

    monkeypatch.setattr(endpoint, "forget_location", refuse)
    org_id = str(uuid4())
    project = make_project(tmp_path, org_id)
    with scoped(org_id) as session:
        record_sort_times(session, project_id=project.id, times={"report": 1.0, "chart": 2.0})
        endpoint._forget_deleted_folder(session, artifact_id="a" * 32, location=Location(project.id, None, "report"))

    assert sort_times_of(org_id) == {(project.id, None, "chart"): 2.0}


# The reads.


def test_sort_times_below_reads_only_folders_the_list_would_open_without_following_a_link(tmp_path):
    project = make_project(tmp_path, str(uuid4()))
    base = project_root(project).base
    timed_artifact(base, "report", 1_000_100.0)
    timed_artifact(base, ".report.bak", 1_000_200.0)
    outside = timed_artifact(tmp_path / "outside", "planted", 1_000_900.0)
    (base / "linked").symlink_to(outside)
    (base / "no-metadata").mkdir()

    assert sort_times_below(
        base=base, slugs=["report", ".report.bak", "linked", "no-metadata", "missing", "../outside", "bad\udcff"]
    ) == SortTimesRead(times={"report": 1_000_100.0}, unreadable=frozenset())
    assert sort_times_below(base=base, slugs=[]) == SortTimesRead(times={}, unreadable=frozenset())
    assert sort_times_below(base=tmp_path / "gone", slugs=["report"]) == SortTimesRead(
        times={}, unreadable=frozenset()
    )
    # A base the trusted-root open refuses, here one with no name to open:
    # no card can come from it, so it holds no time and nothing unread.
    assert sort_times_below(base=Path("/"), slugs=["report"]) == SortTimesRead(times={}, unreadable=frozenset())


def test_sort_times_below_gives_no_time_to_a_folder_whose_metadata_json_is_a_link(tmp_path):
    """The id reader refuses a linked metadata.json, so the list builds no
    card for its folder: it gets no time, even when the link names a regular
    file."""
    project = make_project(tmp_path, str(uuid4()))
    base = project_root(project).base
    timed_artifact(base, "report", 1_000_100.0)
    outside = timed_artifact(tmp_path / "outside", "planted", 1_000_900.0)
    (base / "relinked").mkdir()
    (base / "relinked" / "metadata.json").symlink_to(outside / "metadata.json")

    assert sort_times_below(base=base, slugs=["report", "relinked"]) == SortTimesRead(
        times={"report": 1_000_100.0}, unreadable=frozenset()
    )


def test_id_holder_times_times_only_the_folders_that_hold_an_id(tmp_path):
    """A folder whose metadata.json does not parse, or is no JSON object,
    holds no id and builds no card, so it gets no time, as the self-heal
    gives it none. One with no id field holds the id the reader derives, so
    it is timed. Ids the caller
    read already are used as they are: a folder whose id it could not read
    gets no time and comes back unreadable."""
    project = make_project(tmp_path, str(uuid4()))
    base = project_root(project).base
    timed_artifact(base, "report", 1_000_100.0)
    timed_artifact(base, "stalled", 1_000_200.0)
    for slug, text in (("derived-id", '{"title": "scratch"}'), ("not-json", "{"), ("not-object", "[]")):
        (base / slug).mkdir()
        (base / slug / "metadata.json").write_text(text)
        os.utime(base / slug / "metadata.json", (1_000_300.0, 1_000_300.0))

    assert id_holder_times(
        base=base, slugs=["report", "derived-id", "not-json", "not-object", ".hidden", "missing"]
    ) == SortTimesRead(times={"report": 1_000_100.0, "derived-id": 1_000_300.0}, unreadable=frozenset())
    read = ListedIds(ids={"report": str(uuid4())}, unreadable=frozenset({"stalled", "elsewhere"}))
    assert id_holder_times(base=base, slugs=["report", "stalled"], ids=read) == SortTimesRead(
        times={"report": 1_000_100.0}, unreadable=frozenset({"stalled"})
    )
    assert id_holder_times(base=base, slugs=[]) == SortTimesRead(times={}, unreadable=frozenset())


def test_sort_times_below_returns_the_folders_it_could_not_read(tmp_path, monkeypatch):
    """An EIO is no answer: the folder is returned as unreadable, never
    taken for one with no time, and every folder is when the root itself
    fails that way. A folder the server may not read has no time, since no
    card can come from it."""
    project = make_project(tmp_path, str(uuid4()))
    base = project_root(project).base
    for slug in ("report", "stalled", "refused"):
        timed_artifact(base, slug, 1_000_100.0)
    real_stat = artifact_sort_times.dir_child_stat

    def child_stat(root, name, child, **kwargs):
        if name == "stalled":
            raise OSError(errno.EIO, "the file server did not answer")
        if name == "refused":
            raise PermissionError(errno.EACCES, "permission denied")
        return real_stat(root, name, child, **kwargs)

    monkeypatch.setattr(artifact_sort_times, "dir_child_stat", child_stat)

    assert sort_times_below(base=base, slugs=["report", "stalled", "refused"]) == SortTimesRead(
        times={"report": 1_000_100.0}, unreadable=frozenset({"stalled"})
    )

    def stalled_root(*_args):
        raise OSError(errno.EIO, "the file server did not answer")

    monkeypatch.setattr(artifact_sort_times, "_opened_from_trusted_root", stalled_root)

    assert sort_times_below(base=base, slugs=["report", "refused"]) == SortTimesRead(
        times={}, unreadable=frozenset({"report", "refused"})
    )


@pytest.mark.parametrize("keeps_rows", [True, False], ids=["org", "person"])
def test_a_turn_end_reads_the_time_of_every_folder_it_created_or_changed(tmp_path, org_deployment, keeps_rows):
    """The pod sent no list, so the turn may have edited any folder: in an
    organization every folder's time is read, the unchanged one's too, and
    the record writes only the times that moved (record_sort_times). A
    person's partition keeps no rows, so its turn end reads no time."""
    org_id = str(uuid4())
    project = make_project(tmp_path, org_id)
    base = project_root(project).base
    conversation_id = make_conversation(project, OWNER)
    timed_artifact(base, "kept", 1_000_100.0)
    timed_artifact(base, "edited", 1_000_200.0)
    before, before_mtimes = snapshot_artifact_state(base)
    timed_artifact(base, "made", 1_000_300.0)
    (base / "edited" / "index.html").write_text("<html>changed</html>")
    os.utime(base / "edited" / "metadata.json", (1_000_400.0, 1_000_400.0))

    changes = turn_artifact_changes(
        conversation_id=conversation_id, artifacts_base=base, before=before, before_mtimes=before_mtimes,
        scope=TenantScope(org_mode=keeps_rows, org_id=org_id, user_id=OWNER),
    )

    assert changes.sort_times == (
        {"kept": 1_000_100.0, "made": 1_000_300.0, "edited": 1_000_400.0} if keeps_rows else {}
    )
    assert changes.untimed == frozenset()


@pytest.mark.parametrize("tie_order", [("tie-a", "tie-b", "tie-c"), ("tie-c", "tie-b", "tie-a")])
def test_the_newest_rows_of_the_roots_asked_for(tmp_path, backfills_pending, tie_order):
    """Newest first across the shared and the legacy roots asked for, cut at
    the limit; another member's legacy root, another organization's rows and
    a legacy row of the conversation in another project are not among the
    roots asked for. Tied times come in slug order, whatever order their
    rows were written in."""
    org_id, other_org = str(uuid4()), str(uuid4())
    project, elsewhere, moved_to = (make_project(tmp_path, org) for org in (org_id, other_org, org_id))
    owned, others = make_conversation(project, OWNER), make_conversation(project, OWNER)
    with scoped(org_id) as session:
        for slug in tie_order:
            record_sort_times(session, project_id=project.id, times={slug: 2.0})
        record_sort_times(session, project_id=project.id, times={"b": 3.0, "a": 1.0})
        record_sort_times(session, project_id=project.id, conversation_id=owned, times={"old": 4.0})
        record_sort_times(session, project_id=project.id, conversation_id=others, times={"theirs": 5.0})
        record_sort_times(session, project_id=moved_to.id, conversation_id=owned, times={"other-project": 6.0})
    with scoped(other_org) as session:
        record_sort_times(session, project_id=elsewhere.id, times={"elsewhere": 9.0})
    write_sentinel(_engine(), SENTINEL_KEY, value=SortTimeSentinel().model_dump_json())
    roots = [HeldRoot(project.id, None), HeldRoot(project.id, owned), HeldRoot(elsewhere.id, None)]

    with scoped(org_id) as session:
        everything = newest_folders(session, roots=roots, limit=10)
        cut = newest_folders(session, roots=roots, limit=2)
        nothing = newest_folders(session, roots=[], limit=10)

    assert everything.rows == (
        SortTime(project.id, owned, "old", 4.0),
        SortTime(project.id, None, "b", 3.0),
        SortTime(project.id, None, "tie-a", 2.0),
        SortTime(project.id, None, "tie-b", 2.0),
        SortTime(project.id, None, "tie-c", 2.0),
        SortTime(project.id, None, "a", 1.0),
    )
    assert everything.complete and everything.unread == frozenset()
    assert cut.rows == everything.rows[:2] and not cut.complete
    assert nothing.rows == () and nothing.complete
    with scoped(None) as session:
        assert newest_folders(session, roots=roots, limit=10) is None


@pytest.mark.parametrize("sentinel", [None, "1"], ids=["no-sentinel", "unreadable-sentinel"])
def test_the_newest_rows_wait_for_the_backfills_sentinel(tmp_path, backfills_pending, sentinel):
    """Before the backfill has written a sentinel it can read, a project's
    rows may be missing: newest_folders answers None, and the list reads
    every folder of its roots."""
    org_id = str(uuid4())
    project = make_project(tmp_path, org_id)
    with scoped(org_id) as session:
        record_sort_times(session, project_id=project.id, times={"report": 1.0})
    if sentinel is not None:
        write_sentinel(_engine(), SENTINEL_KEY, value=sentinel)

    with scoped(org_id) as session:
        assert newest_folders(session, roots=[HeldRoot(project.id, None)], limit=10) is None


def test_the_newest_rows_leave_out_the_projects_the_sentinel_lists(tmp_path, backfills_pending):
    """A project the backfill could not time may have rows for only some of
    its folders, so its roots are left out of the rows, rows a turn end
    wrote included, and named as unread: the list reads every folder of
    them. One read of the sentinel."""
    org_id = str(uuid4())
    timed, unread = make_project(tmp_path, org_id), make_project(tmp_path, org_id)
    owned = make_conversation(unread, OWNER)
    with scoped(org_id) as session:
        record_sort_times(session, project_id=timed.id, times={"report": 1.0})
        record_sort_times(session, project_id=unread.id, times={"newer": 2.0})
        record_sort_times(session, project_id=unread.id, conversation_id=owned, times={"newest": 3.0})
    write_sentinel(_engine(), SENTINEL_KEY, value=SortTimeSentinel(unread=[unread.id, uuid4()]).model_dump_json())
    roots = [HeldRoot(timed.id, None), HeldRoot(unread.id, None), HeldRoot(unread.id, owned)]
    reads: list[str] = []

    def count_settings_reads(_conn, _cursor, statement, *_args):
        if "FROM settings" in statement:
            reads.append(statement)

    event.listen(_engine(), "before_cursor_execute", count_settings_reads)
    try:
        with scoped(org_id) as session:
            newest = newest_folders(session, roots=roots, limit=10)
            only_unread = newest_folders(session, roots=roots[1:], limit=10)
    finally:
        event.remove(_engine(), "before_cursor_execute", count_settings_reads)

    assert newest.rows == (SortTime(timed.id, None, "report", 1.0),)
    assert newest.complete and newest.unread == frozenset({unread.id})
    assert only_unread.rows == () and only_unread.unread == frozenset({unread.id})
    assert len(reads) == 2


def test_with_held_roots_reads_the_held_sentinel_once(tmp_path, org_deployment, backfills_pending):
    """A list's plan reads the held-root sentinel once: held_root_rows makes
    no check of its own."""
    org_id = str(uuid4())
    project = make_project(tmp_path, org_id)
    write_artifact(project_root(project).base, "report")
    run_held_root_backfill(try_lock=_no_lock, project_ids=[project.id])
    reads: list[str] = []

    def count_settings_reads(_conn, _cursor, statement, *_args):
        if "FROM settings" in statement:
            reads.append(statement)

    event.listen(_engine(), "before_cursor_execute", count_settings_reads)
    try:
        with scoped(org_id, OWNER) as session:
            plan = with_held_roots(session, plan_scope_roots(session))
    finally:
        event.remove(_engine(), "before_cursor_execute", count_settings_reads)

    assert HeldRoot(project.id, None) in plan.held
    assert len(reads) == 1
