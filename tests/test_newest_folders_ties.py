"""newest_folders when its limit cuts a run of rows that share one time: it
reads the rest of the run (NewestFolders.tied), up to
artifact_sort_times._TIED_ROWS rows of it, so the list can order the whole
run as a walk of every folder does."""
from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from uuid import uuid4

import pytest
from sqlalchemy import event
from sqlmodel import Session, select

from cowork.db.scoped import ScopedSession
from cowork.models.artifact_sort_time import ArtifactSortTime
from cowork.services import artifact_sort_times
from cowork.services.artifact_owner_backfill import write_sentinel
from cowork.services.artifact_sort_times import (
    SENTINEL_KEY,
    SortTime,
    SortTimeSentinel,
    newest_folders,
    record_sort_times,
)
from cowork.services.held_artifact_roots import HeldRoot
from test_artifact_ownership import make_conversation, make_project, scoped
from test_artifact_sort_times import OWNER, _engine, backfills_pending  # noqa: F401  (backfills_pending is a fixture)

pytestmark = pytest.mark.usefixtures("cleanup_tmp_projects", "backfills_pending")


def _every_row(org_id: str) -> tuple[SortTime, ...]:
    """Every row of ``org_id``, in the order newest_folders' query reads
    them."""
    with Session(_engine()) as raw:
        rows = raw.exec(
            select(ArtifactSortTime)
            .where(ArtifactSortTime.org_id == org_id)
            .order_by(ArtifactSortTime.sort_time.desc(), ArtifactSortTime.slug, ArtifactSortTime.id)
        ).all()
        return tuple(SortTime(row.project_id, row.conversation_id, row.slug, row.sort_time) for row in rows)


@contextmanager
def _queries_of_the_rows() -> Iterator[list[str]]:
    """The statements that read artifact_sort_times while the context is
    open."""
    statements: list[str] = []

    def record(_conn, _cursor, statement, *_args):
        if "FROM artifact_sort_times" in statement:
            statements.append(statement)

    event.listen(_engine(), "before_cursor_execute", record)
    try:
        yield statements
    finally:
        event.remove(_engine(), "before_cursor_execute", record)


def _roots_with_a_tied_run(tmp_path):
    """Three roots, two projects' shared roots and a legacy root, whose rows
    share the time 2.0, one name in two roots among them, beside a newer row
    and an older one. Returns the organization and the roots."""
    org_id = str(uuid4())
    project, other = make_project(tmp_path, org_id), make_project(tmp_path, org_id)
    owned = make_conversation(project, OWNER)
    with scoped(org_id) as session:
        record_sort_times(session, project_id=project.id, times={"newest": 3.0, "report": 2.0, "z-last": 2.0})
        record_sort_times(session, project_id=project.id, conversation_id=owned, times={"b-legacy": 2.0})
        record_sort_times(session, project_id=other.id, times={"a-first": 2.0, "report": 2.0, "oldest": 1.0})
    write_sentinel(_engine(), SENTINEL_KEY, value=SortTimeSentinel().model_dump_json())
    return org_id, [HeldRoot(project.id, None), HeldRoot(project.id, owned), HeldRoot(other.id, None)]


@pytest.mark.parametrize("limit", [2, 3, 4, 5])
def test_a_run_the_limit_cuts_comes_whole_in_two_queries(tmp_path, limit):
    """Wherever the limit cuts the run at 2.0, the rows kept and the rest of
    the run (tied) are every row at least as new as 2.0, in the query's
    order, each once, the two rows named "report" included."""
    org_id, roots = _roots_with_a_tied_run(tmp_path)
    every = _every_row(org_id)

    with scoped(org_id) as session, _queries_of_the_rows() as statements:
        newest = newest_folders(session, roots=roots, limit=limit)

    assert newest.rows == every[:limit]
    assert newest.rows + newest.tied == every[:-1]
    assert [row.sort_time for row in newest.tied] == [2.0] * (len(every) - 1 - limit)
    assert not newest.complete and newest.ties_read
    assert len(statements) == 2


@pytest.mark.parametrize("limit", [1, 6])
def test_a_limit_that_cuts_between_two_times_reads_no_run(tmp_path, limit):
    """When the row past the limit is older than the last row kept, the
    rows kept hold every row at that row's time: one query, and no tied
    rows."""
    org_id, roots = _roots_with_a_tied_run(tmp_path)
    every = _every_row(org_id)

    with scoped(org_id) as session, _queries_of_the_rows() as statements:
        newest = newest_folders(session, roots=roots, limit=limit)

    assert newest.rows == every[:limit]
    assert newest.tied == () and newest.ties_read and not newest.complete
    assert len(statements) == 1


@pytest.mark.parametrize("cap", [4, 5])
def test_a_run_longer_than_the_cap_is_not_read(tmp_path, monkeypatch, cap):
    """A run of five rows at 2.0: with room for four, the rows kept are as
    the first query cut them, no tied rows come back, and ties_read says the
    run was not read whole; with room for five, the run comes whole."""
    org_id, roots = _roots_with_a_tied_run(tmp_path)
    every = _every_row(org_id)
    monkeypatch.setattr(artifact_sort_times, "_TIED_ROWS", cap)

    with scoped(org_id) as session:
        newest = newest_folders(session, roots=roots, limit=3)

    assert newest.rows == every[:3] and not newest.complete
    if cap == 4:
        assert newest.tied == () and not newest.ties_read
    else:
        assert newest.rows + newest.tied == every[:-1] and newest.ties_read


def test_a_folder_of_one_name_in_a_shared_root_and_a_legacy_root_comes_from_both(tmp_path):
    """One project's shared root and the owner's legacy root each hold a
    folder named "report" at 2.0, and the limit keeps one of them. tied
    holds the other: a folder is its root and its name, so the folder of
    one root never stands for the folder of that name in the other."""
    org_id = str(uuid4())
    project = make_project(tmp_path, org_id)
    owned = make_conversation(project, OWNER)
    with scoped(org_id) as session:
        record_sort_times(session, project_id=project.id, times={"newest": 3.0, "report": 2.0})
        record_sort_times(session, project_id=project.id, conversation_id=owned, times={"report": 2.0})
    write_sentinel(_engine(), SENTINEL_KEY, value=SortTimeSentinel().model_dump_json())
    every = _every_row(org_id)

    with scoped(org_id) as session:
        newest = newest_folders(session, roots=[HeldRoot(project.id, None), HeldRoot(project.id, owned)], limit=2)

    assert newest.rows == every[:2]
    assert newest.rows + newest.tied == every
    assert {row.conversation_id for row in every if row.slug == "report"} == {None, owned}
    assert newest.ties_read and not newest.complete


def _added_again(org_id: str, folder: SortTime) -> None:
    """Drop ``folder``'s row and add the folder's row again at the same
    time, under a new id, as a writer that deletes the row and a writer
    that records the folder again would leave it."""
    with Session(_engine()) as raw:
        row = raw.exec(
            select(ArtifactSortTime).where(
                ArtifactSortTime.org_id == org_id,
                ArtifactSortTime.project_id == folder.project_id,
                ArtifactSortTime.slug == folder.slug,
            )
        ).one()
        raw.delete(row)
        raw.commit()
        raw.add(
            ArtifactSortTime(
                org_id=org_id,
                project_id=folder.project_id,
                conversation_id=folder.conversation_id,
                slug=folder.slug,
                sort_time=folder.sort_time,
            )
        )
        raw.commit()


def test_a_folder_whose_row_is_added_again_between_the_two_queries_comes_once(tmp_path, monkeypatch):
    """A kept row at the cut's time is dropped and its folder's row added
    again, at that time and under a new id, after the first query and
    before the read of the run, as two queries at READ COMMITTED may see
    it. The run then holds that folder under another id. tied leaves it
    out, since the rows kept hold its folder, so the list never opens it
    twice."""
    org_id, roots = _roots_with_a_tied_run(tmp_path)
    every = _every_row(org_id)
    limit = 3
    readded = every[limit - 1]
    assert readded.sort_time == 2.0
    real_exec = ScopedSession.exec
    reads: list[str] = []

    def exec_after_a_writer(self, stmt):
        if "FROM artifact_sort_times" in str(stmt._stmt):
            if len(reads) == 1:
                _added_again(org_id, readded)
            reads.append(str(stmt._stmt))
        return real_exec(self, stmt)

    monkeypatch.setattr(ScopedSession, "exec", exec_after_a_writer)
    with scoped(org_id) as session:
        newest = newest_folders(session, roots=roots, limit=limit)

    assert len(reads) == 2
    assert newest.rows == every[:limit]
    assert newest.rows + newest.tied == every[:-1]
    assert readded not in newest.tied
