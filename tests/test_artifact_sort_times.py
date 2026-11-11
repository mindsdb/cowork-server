"""The artifact_sort_times rows: the backfill reads every held root's folders,
each writer keeps a folder's time, and the list reads the newest rows only
once the backfill has finished."""
from __future__ import annotations

import errno
import os
import threading
from contextlib import nullcontext
from pathlib import Path
from uuid import UUID, uuid4

import pytest
from sqlmodel import Session, select

from cowork.common.settings.app_settings import get_app_settings
from cowork.db.session import get_engine
from cowork.models.artifact_sort_time import ArtifactSortTime
from cowork.models.setting import Setting
from cowork.services import artifact_sort_times
from cowork.services.artifact_autopublish import _record_listing
from cowork.services.artifact_identity import ListedIds
from cowork.services.artifact_locations import Location
from cowork.services.artifact_sort_times import (
    SENTINEL_KEY,
    SortTime,
    newest_folders,
    reconcile_sort_times,
    run_sort_time_backfill,
    sort_times_below,
)
from cowork.services.held_artifact_roots import SENTINEL_KEY as HELD_KEY
from cowork.services.held_artifact_roots import HeldRoot, hold_root, run_held_root_backfill
from cowork.services.task_objects import record_new_artifacts, snapshot_artifact_state, turn_artifact_changes
from test_artifact_ownership import make_conversation, make_project, project_root, scoped, write_artifact
from test_artifact_relocation_ownership import _index, _relocate

pytestmark = pytest.mark.usefixtures("cleanup_tmp_projects")

OWNER = "11111111-1111-4111-8111-111111111111"


def _engine():
    return get_engine(get_app_settings().database.uri)


def _no_lock(_engine, _key):
    return nullcontext(True)


def _sentinel(key: str) -> bool:
    with Session(_engine()) as raw:
        return raw.exec(select(Setting).where(Setting.key == key, Setting.scope.is_(None))).first() is not None


@pytest.fixture
def backfills_pending():
    """No pass has written either sentinel; those a test writes go after it,
    as the database is shared across modules."""
    yield
    with Session(_engine()) as raw:
        for row in raw.exec(select(Setting).where(Setting.key.in_((SENTINEL_KEY, HELD_KEY)))).all():
            raw.delete(row)
        raw.commit()


def _times(org_id: str) -> dict[tuple, float]:
    with Session(_engine()) as raw:
        rows = raw.exec(select(ArtifactSortTime).where(ArtifactSortTime.org_id == org_id)).all()
        return {(row.project_id, row.conversation_id, row.slug): row.sort_time for row in rows}


def _artifact(base: Path, slug: str, at: float) -> Path:
    folder = write_artifact(base, slug)
    os.utime(folder / "metadata.json", (at, at))
    return folder


def _workspace_root(project, conversation_id) -> Path:
    return Path(project.path) / "conversations" / str(conversation_id) / ".anton" / "artifacts"


def test_the_backfill_records_the_time_of_every_folder_in_the_held_roots(
    tmp_path, org_deployment, backfills_pending
):
    org_id = str(uuid4())
    project = make_project(tmp_path, org_id)
    base = project_root(project).base
    _artifact(base, "report", 1_000_100.0)
    (base / "no-metadata").mkdir()
    (base / "fifo-metadata").mkdir()
    os.mkfifo(base / "fifo-metadata" / "metadata.json")
    (base / "linked").symlink_to(base / "report")
    owned = make_conversation(project, OWNER)
    _artifact(_workspace_root(project, owned), "old-report", 1_000_200.0)

    # Waits for the held rows, which name the roots it reads.
    assert run_sort_time_backfill(try_lock=_no_lock, project_ids=[project.id]) is None
    run_held_root_backfill(try_lock=_no_lock, project_ids=[project.id])
    summary = run_sort_time_backfill(try_lock=_no_lock, project_ids=[project.id])

    assert summary is not None and (summary.recorded, summary.unread) == (2, 0)
    assert _times(org_id) == {
        (project.id, None, "report"): 1_000_100.0,
        (project.id, owned, "old-report"): 1_000_200.0,
    }
    assert _sentinel(SENTINEL_KEY)
    assert run_sort_time_backfill(try_lock=_no_lock, project_ids=[project.id]) is None


def test_a_root_the_backfill_cannot_read_leaves_no_sentinel(tmp_path, org_deployment, backfills_pending, monkeypatch):
    org_id = str(uuid4())
    project = make_project(tmp_path, org_id)
    _artifact(project_root(project).base, "report", 1_000_100.0)
    run_held_root_backfill(try_lock=_no_lock, project_ids=[project.id])

    def unreadable(_root):
        raise OSError(errno.EIO, "the file server did not answer")

    with monkeypatch.context() as failing:
        failing.setattr(artifact_sort_times, "dir_scandir", unreadable)
        summary = run_sort_time_backfill(try_lock=_no_lock, project_ids=[project.id])

    assert summary is not None and summary.unread == 1
    assert not _sentinel(SENTINEL_KEY)
    run_sort_time_backfill(try_lock=_no_lock, project_ids=[project.id])
    assert _times(org_id) == {(project.id, None, "report"): 1_000_100.0}
    assert _sentinel(SENTINEL_KEY)


def test_a_stopped_backfill_writes_no_sentinel(tmp_path, org_deployment, backfills_pending):
    project = make_project(tmp_path, str(uuid4()))
    _artifact(project_root(project).base, "report", 1_000_100.0)
    run_held_root_backfill(try_lock=_no_lock, project_ids=[project.id])
    stop = threading.Event()
    stop.set()

    assert run_sort_time_backfill(try_lock=_no_lock, project_ids=[project.id], stop=stop) is None
    assert not _sentinel(SENTINEL_KEY)


def test_the_newest_rows_come_only_once_the_backfill_has_finished(tmp_path, org_deployment, backfills_pending):
    org_id, other_org = str(uuid4()), str(uuid4())
    project, elsewhere = make_project(tmp_path, org_id), make_project(tmp_path, other_org)
    owned, others = make_conversation(project, OWNER), make_conversation(project, OWNER)
    with scoped(org_id) as session:
        artifact_sort_times.record_sort_times(session, project_id=project.id, times={"a": 1.0, "b": 3.0, "c": 2.0})
        artifact_sort_times.record_sort_times(session, project_id=project.id, conversation_id=owned, times={"old": 4.0})
        artifact_sort_times.record_sort_times(session, project_id=project.id, conversation_id=others, times={"theirs": 5.0})
    with scoped(other_org) as session:
        artifact_sort_times.record_sort_times(session, project_id=elsewhere.id, times={"elsewhere": 9.0})
    roots = [HeldRoot(project.id, None), HeldRoot(project.id, owned), HeldRoot(elsewhere.id, None)]

    with scoped(org_id) as session:
        assert newest_folders(session, roots=roots, limit=10) is None
    run_held_root_backfill(try_lock=_no_lock, project_ids=[])
    run_sort_time_backfill(try_lock=_no_lock, project_ids=[])
    with scoped(org_id) as session:
        everything = newest_folders(session, roots=roots, limit=10)
        cut = newest_folders(session, roots=roots, limit=2)

    # Another member's legacy root and another organization's rows are not
    # among the roots asked for, or not this scope's.
    assert everything.rows == (
        SortTime(project.id, owned, "old", 4.0),
        SortTime(project.id, None, "b", 3.0),
        SortTime(project.id, None, "c", 2.0),
        SortTime(project.id, None, "a", 1.0),
    )
    assert everything.complete
    assert cut.rows == everything.rows[:2] and not cut.complete
    with scoped(None) as session:
        assert newest_folders(session, roots=roots, limit=10) is None


def test_sort_times_below_reads_only_plain_folders_without_following_a_link(tmp_path):
    project = make_project(tmp_path, str(uuid4()))
    base = project_root(project).base
    _artifact(base, "report", 1_000_100.0)
    outside = _artifact(tmp_path / "outside", "planted", 1_000_900.0)
    (base / "linked").symlink_to(outside)
    (base / "no-metadata").mkdir()

    assert sort_times_below(base=base, slugs=["report", "linked", "no-metadata", "missing", "../outside"]) == {
        "report": 1_000_100.0
    }
    assert sort_times_below(base=base, slugs=[]) == {}


def test_a_turn_end_reads_the_time_of_every_folder_it_created_or_changed(tmp_path, org_deployment):
    project = make_project(tmp_path, str(uuid4()))
    base = project_root(project).base
    conversation_id = make_conversation(project, OWNER)
    _artifact(base, "kept", 1_000_100.0)
    _artifact(base, "edited", 1_000_200.0)
    before, before_mtimes = snapshot_artifact_state(base)
    _artifact(base, "made", 1_000_300.0)
    (base / "edited" / "index.html").write_text("<html>changed</html>")
    os.utime(base / "edited" / "metadata.json", (1_000_400.0, 1_000_400.0))

    changes = turn_artifact_changes(
        conversation_id=conversation_id, artifacts_base=base, before=before, before_mtimes=before_mtimes,
    )

    assert changes.sort_times == {"made": 1_000_300.0, "edited": 1_000_400.0}


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

    assert _times(org_id) == {(project.id, None, "report"): 5.0, (project.id, None, "chart"): 2.0}


def test_the_desktop_records_no_time(tmp_path):
    project = make_project(tmp_path, None)
    conversation_id = make_conversation(project, None)

    with scoped(None) as session:
        record_new_artifacts(
            session, conversation_id=conversation_id, project_id=project.id, slugs=[], creator=None,
            sort_times={"report": 1.0},
        )

    with Session(_engine()) as raw:
        assert raw.exec(select(ArtifactSortTime).where(ArtifactSortTime.project_id == project.id)).first() is None


@pytest.mark.parametrize("complete", [True, False], ids=["root-read", "root-unread"])
def test_the_self_heals_listing_adds_what_no_row_names_and_drops_what_is_gone(tmp_path, complete):
    """A row a turn end wrote keeps its time; a folder the listing found
    holding an id keeps its row even when its time didn't read; a gone
    folder's row goes only when the listing read the root itself."""
    org_id = str(uuid4())
    project = make_project(tmp_path, org_id)
    with scoped(org_id) as session:
        artifact_sort_times.record_sort_times(
            session, project_id=project.id, times={"kept": 1.0, "no-time": 2.0, "unread": 3.0, "gone": 4.0},
        )
    listing = ListedIds(
        ids={"kept": "a" * 32, "no-time": "b" * 32, "copied": "c" * 32},
        unreadable=frozenset({"unread"}),
        base_unreadable=not complete,
    )

    with scoped(org_id) as session:
        _record_listing(
            session, project_id=UUID(str(project.id)), listing=listing, times={"kept": 9.0, "copied": 7.0},
        )

    expected = {"kept": 1.0, "no-time": 2.0, "unread": 3.0, "copied": 7.0}
    if not complete:
        expected["gone"] = 4.0
    assert _times(org_id) == {(project.id, None, slug): at for slug, at in expected.items()}


def test_reconcile_drops_a_folder_with_no_metadata(tmp_path):
    org_id = str(uuid4())
    project = make_project(tmp_path, org_id)
    with scoped(org_id) as session:
        artifact_sort_times.record_sort_times(session, project_id=project.id, times={"broken": 1.0})
        reconcile_sort_times(session, project_id=project.id, listed={}, kept=set(), complete=True)

    assert _times(org_id) == {}


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
        artifact_sort_times.record_sort_times(session, project_id=source.id, times={"report": 1_000_100.0})

    assert _relocate(org_id, owner, conversation_id, source.id, dest.id) == 1

    assert _times(org_id) == {(dest.id, None, "report"): 1_000_100.0}


def test_a_deleted_folder_takes_its_time_with_it(tmp_path):
    from cowork.api.v1.endpoints.artifacts import _forget_deleted_folder

    org_id = str(uuid4())
    project = make_project(tmp_path, org_id)
    with scoped(org_id) as session:
        artifact_sort_times.record_sort_times(session, project_id=project.id, times={"report": 1.0, "chart": 2.0})
        _forget_deleted_folder(session, artifact_id="a" * 32, location=Location(project.id, None, "report"))

    assert _times(org_id) == {(project.id, None, "chart"): 2.0}
