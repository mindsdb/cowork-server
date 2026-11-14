"""The artifact_sort_times rows on PostgreSQL.

SQLite ignores FOR UPDATE and serializes every writer, so the SQLite suite
(tests/test_artifact_sort_times.py) cannot show two writers at once. Here a
writer that read a folder's row while another one held it waits for that
writer's commit and reads its time, so the newer time stays; two writers
that add the same new folders queue rather than deadlock, whatever order
each was handed them in; a writer never deadlocks a project or conversation
delete that cascades into its rows, and neither does a held root's;
a run of tied times is cut in the code point order of slug, whatever the
database's collation; and the revision's downgrade takes the sentinel and
the progress row with the table. These run when COWORK_TEST_POSTGRES_URI
points at a disposable database, on the schema the migrations build, as
CI's service tests do (postgres_service). Each writer's session is built as
a database unit builds its own (cowork.db.units.unit_session), keeping its
rows across commits (expire_on_commit=False).
"""
from __future__ import annotations

import os
import random
import threading
import time
from uuid import UUID, uuid4

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.pool import NullPool
from sqlmodel import Session, select

from cowork.common.settings.app_settings import get_app_settings
from cowork.db import session as db_session
from cowork.db.scoped import ScopedSession, TenantScope
from cowork.models.artifact_sort_time import ArtifactSortTime
from cowork.models.conversation import Conversation
from cowork.models.held_artifact_root import HeldArtifactRoot
from cowork.models.project import Project
from cowork.models.setting import Setting
from cowork.services import artifact_sort_times
from cowork.services.artifact_owner_backfill import write_sentinel
from cowork.services.artifact_sort_times import (
    PROGRESS_KEY,
    SENTINEL_KEY,
    SortTime,
    SortTimeSentinel,
    newest_folders,
    record_sort_times,
)
from cowork.services.conversations import ConversationService
from cowork.services.held_artifact_roots import SENTINEL_KEY as HELD_KEY
from cowork.services.held_artifact_roots import HeldRoot, hold_root
from cowork.services.projects import ProjectService

POSTGRES_URI = os.environ.get("COWORK_TEST_POSTGRES_URI", "")

pytestmark = [
    pytest.mark.postgres_service,
    pytest.mark.skipif(not POSTGRES_URI, reason="set COWORK_TEST_POSTGRES_URI to a disposable database"),
]

ORG_ID = "44444444-4444-4444-8444-444444444444"
OWNER = "55555555-5555-4555-8555-555555555555"


def _empty_public_schema() -> None:
    """Drop everything the migrations built, alembic_version included, on a
    connection of its own. The lock timeout turns a session a failed test
    left open into an error here rather than a teardown that waits
    forever."""
    engine = create_engine(POSTGRES_URI, poolclass=NullPool)
    try:
        with engine.begin() as connection:
            connection.execute(text("SET LOCAL lock_timeout = '15s'"))
            connection.execute(text("DROP SCHEMA public CASCADE"))
            connection.execute(text("CREATE SCHEMA public"))
    finally:
        engine.dispose()


@pytest.fixture
def engine(monkeypatch):
    """A database the migrations built from an empty schema, emptied again
    after the test."""
    monkeypatch.setenv("DATABASE_URI", POSTGRES_URI)
    get_app_settings.cache_clear()
    _empty_public_schema()
    command.upgrade(Config("alembic.ini"), "head")
    engine = create_engine(POSTGRES_URI)
    try:
        yield engine
    finally:
        engine.dispose()
        _empty_public_schema()
        get_app_settings.cache_clear()


def _session(engine) -> ScopedSession:
    """A writer's session, built as a database unit builds its own
    (cowork.db.units.unit_session): its own connection, at the server's
    default isolation (READ COMMITTED), keeping its rows across commits."""
    return ScopedSession(
        db_session.get_session_factory(engine)(), TenantScope(org_mode=True, org_id=ORG_ID, user_id=OWNER)
    )


def _project(engine) -> UUID:
    project_id = uuid4()
    with Session(engine) as raw:
        raw.add(Project(id=project_id, name="sort-times", path=f"/projects/{project_id}", org_id=ORG_ID))
        raw.commit()
    return project_id


def _committed_time(engine, project_id: UUID, slug: str) -> float:
    with Session(engine) as raw:
        row = raw.exec(
            select(ArtifactSortTime).where(ArtifactSortTime.project_id == project_id, ArtifactSortTime.slug == slug)
        ).one()
        return row.sort_time


def _wait_for_a_lock_wait(engine) -> None:
    """Return once a backend of this database waits on a lock."""
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        with engine.connect() as connection:
            waiting = connection.execute(
                text(
                    "SELECT count(*) FROM pg_stat_activity "
                    "WHERE datname = current_database() AND wait_event_type = 'Lock'"
                )
            ).scalar_one()
        if waiting:
            return
        time.sleep(0.05)
    raise AssertionError("no writer waited on the row's lock")


def test_a_writer_that_waited_on_another_writers_row_keeps_its_newer_time(engine):
    """One writer holds the folder's row, having raised its time to 2000 but
    not committed. Another reads the row to write 1500: it waits on the lock
    (_rows reads FOR UPDATE), then reads the 2000 the first committed, and
    leaves it. A read without the lock would take the old 1000, and its
    write would put 1500 over the newer time."""
    project_id = _project(engine)
    first = _session(engine)
    record_sort_times(first, project_id=project_id, times={"report": 1_000.0})
    rows = artifact_sort_times._rows(first, project_id=project_id, conversation_id=None, slugs=["report"])
    rows["report"].sort_time = 2_000.0
    first.add(rows["report"])
    first.flush()
    errors: list[BaseException] = []

    def later_writer() -> None:
        session = _session(engine)
        try:
            record_sort_times(session, project_id=project_id, times={"report": 1_500.0})
        except BaseException as exc:  # noqa: BLE001 - reported below
            errors.append(exc)
        finally:
            session.close()

    writer = threading.Thread(target=later_writer)
    writer.start()
    try:
        _wait_for_a_lock_wait(engine)
        first.commit()
    finally:
        first.close()
        writer.join(30)

    assert not writer.is_alive()
    assert errors == []
    assert _committed_time(engine, project_id, "report") == 2_000.0


def _sentinel_written(engine) -> None:
    """The backfill has written its sentinel, so newest_folders answers."""
    write_sentinel(engine, SENTINEL_KEY, value=SortTimeSentinel().model_dump_json())


def test_two_writers_adding_the_same_new_folders_in_opposite_orders_queue_rather_than_deadlock(engine):
    """Each writer is handed the same new folders, one in ascending and one
    in descending order. Both add their rows in slug order (_write), so the
    second waits on the first's first row, then finds the rows on its retry
    (record_sort_times); in the order each was handed, they would deadlock
    and one would fail with an error the retry does not catch."""
    project_id = _project(engine)
    slugs = [f"folder-{index:04d}" for index in range(3000)]
    orders = {"ascending": slugs, "descending": list(reversed(slugs))}
    ready = threading.Barrier(len(orders))
    errors: dict[str, str] = {}

    def writer(name: str) -> None:
        session = _session(engine)
        try:
            ready.wait()
            record_sort_times(session, project_id=project_id, times={slug: 1_000.0 for slug in orders[name]})
        except BaseException as exc:  # noqa: BLE001 - reported below
            errors[name] = f"{type(exc).__name__}: {exc}"
        finally:
            session.close()

    writers = [threading.Thread(target=writer, args=(name,)) for name in orders]
    for thread in writers:
        thread.start()
    for thread in writers:
        thread.join(60)

    assert not any(thread.is_alive() for thread in writers)
    assert errors == {}
    with Session(engine) as raw:
        rows = raw.exec(select(ArtifactSortTime).where(ArtifactSortTime.project_id == project_id)).all()
    assert sorted(row.slug for row in rows) == slugs


def _conversation(engine, project_id: UUID) -> UUID:
    conversation_id = uuid4()
    with Session(engine) as raw:
        raw.add(Conversation(id=conversation_id, topic="task", project_id=project_id, org_id=ORG_ID, created_by=OWNER))
        raw.commit()
    return conversation_id


def _delete_project(engine, project_id: UUID, _conversation_id: UUID | None) -> None:
    """ProjectService.delete_project: the project's conversations, then the
    project, in one commit."""
    session = _session(engine)
    try:
        assert ProjectService(session).delete_project(project_id)
    finally:
        session.close()


def _delete_conversation(engine, _project_id: UUID, conversation_id: UUID | None) -> None:
    """ConversationService.delete_conversation: the conversation alone."""
    session = _session(engine)
    try:
        assert ConversationService(session).delete_conversation(conversation_id)
    finally:
        session.close()


def _write_beside_a_delete(engine, monkeypatch, *, project_id: UUID, conversation_id: UUID | None, delete) -> dict:
    """A writer records a folder whose row exists and a new one, and pauses
    once it holds its locks (_rows). `delete` then runs on a session of its
    own and waits on one of those locks, the writer goes on, and both end.
    Returns the error each side raised, by side."""
    real_rows = artifact_sort_times._rows
    locked, resume = threading.Event(), threading.Event()
    errors: dict[str, str] = {}

    def rows_then_pause(session, **kwargs):
        found = real_rows(session, **kwargs)
        if threading.current_thread().name == "writer":
            locked.set()
            resume.wait(30)
        return found

    def write() -> None:
        session = _session(engine)
        try:
            record_sort_times(
                session, project_id=project_id, conversation_id=conversation_id, times={"a-existing": 5.0, "b-new": 5.0}
            )
        finally:
            session.close()

    def run(side: str, work) -> None:
        try:
            work()
        except BaseException as exc:  # noqa: BLE001 - reported below
            errors[side] = f"{type(exc).__name__}: {exc}"

    seed = _session(engine)
    try:
        record_sort_times(seed, project_id=project_id, conversation_id=conversation_id, times={"a-existing": 1.0})
    finally:
        seed.close()
    monkeypatch.setattr(artifact_sort_times, "_rows", rows_then_pause)
    writer = threading.Thread(target=run, args=("writer", write), name="writer")
    deleter = threading.Thread(target=run, args=("delete", lambda: delete(engine, project_id, conversation_id)))
    writer.start()
    try:
        assert locked.wait(15), "the writer never took its locks"
        deleter.start()
        _wait_for_a_lock_wait(engine)
    finally:
        resume.set()
        writer.join(30)
        if deleter.ident is not None:
            deleter.join(30)
    assert not writer.is_alive() and not deleter.is_alive()
    return errors


def _rows_of(engine, project_id: UUID) -> list[tuple]:
    with Session(engine) as raw:
        rows = raw.exec(select(ArtifactSortTime).where(ArtifactSortTime.project_id == project_id)).all()
        return sorted((row.conversation_id, row.slug, row.sort_time) for row in rows)


def test_a_writer_never_deadlocks_a_project_delete(engine, monkeypatch):
    """The delete takes the project's row, then cascades into its folders'
    rows. A writer that locked a folder's row first and then took the
    project's row for its new folder's foreign key would wait on the delete
    while the delete waits on it, and one would fail. The writer takes the
    project's row first (lock_root_parents), so the delete waits behind it,
    then removes every row."""
    project_id = _project(engine)

    errors = _write_beside_a_delete(
        engine, monkeypatch, project_id=project_id, conversation_id=None, delete=_delete_project
    )

    assert errors == {}
    assert _rows_of(engine, project_id) == []


@pytest.mark.parametrize("delete", [_delete_project, _delete_conversation], ids=["project", "conversation"])
def test_a_legacy_roots_writer_never_deadlocks_a_delete_of_its_conversation(engine, monkeypatch, delete):
    """Both deletes take the conversation's row first, a project delete then
    the project's, and cascade into the legacy root's rows. The writer takes
    the conversation's row, then the project's, before its folders' rows
    (lock_root_parents), so each delete waits behind it, and no side
    fails."""
    project_id = _project(engine)
    conversation_id = _conversation(engine, project_id)

    errors = _write_beside_a_delete(
        engine, monkeypatch, project_id=project_id, conversation_id=conversation_id, delete=delete
    )

    assert errors == {}
    assert _rows_of(engine, project_id) == []


def test_a_held_legacy_root_never_deadlocks_a_project_delete(engine):
    """ProjectService.delete_project deletes the project's conversations,
    then the project, in one transaction, each cascading into the held
    rows. Here the delete has taken the conversation's row when a writer
    holds its legacy root. The insert's foreign key checks run after the
    row is written, in either order; where the project's runs first, a
    writer that locked nothing first would hold the project's row while it
    waits on the conversation's, and the delete would then wait on the
    project's. hold_root takes the conversation's row, then the project's, first
    (lock_root_parents): it waits on the conversation alone, the delete
    finishes, and the writer finds its parents gone and records nothing."""
    project_id = _project(engine)
    conversation_id = _conversation(engine, project_id)
    errors: dict[str, str] = {}

    def hold() -> None:
        session = _session(engine)
        try:
            hold_root(session, project_id=project_id, conversation_id=conversation_id)
        except BaseException as exc:  # noqa: BLE001 - reported below
            errors["writer"] = f"{type(exc).__name__}: {exc}"
        finally:
            session.close()

    writer = threading.Thread(target=hold)
    with engine.connect() as connection:
        try:
            with connection.begin():
                connection.execute(text("DELETE FROM conversations WHERE id = :id"), {"id": conversation_id})
                writer.start()
                _wait_for_a_lock_wait(engine)
                connection.execute(text("DELETE FROM projects WHERE id = :id"), {"id": project_id})
        except BaseException as exc:  # noqa: BLE001 - reported below
            errors["delete"] = f"{type(exc).__name__}: {exc}"
        finally:
            writer.join(30)

    assert not writer.is_alive()
    assert errors == {}
    with Session(engine) as raw:
        assert raw.exec(select(HeldArtifactRoot).where(HeldArtifactRoot.project_id == project_id)).all() == []


def test_the_newest_rows_match_each_legacy_root_by_its_project_and_conversation(engine):
    """The legacy clause matches (project, conversation) pairs, so a row of
    the conversation in another project takes no slot, and tied times come
    in slug order."""
    project_id, other_id = _project(engine), _project(engine)
    conversation_id = uuid4()
    with Session(engine) as raw:
        raw.add(Conversation(id=conversation_id, topic="task", project_id=project_id, org_id=ORG_ID))
        raw.commit()
    _sentinel_written(engine)
    session = _session(engine)
    try:
        record_sort_times(session, project_id=project_id, times={"tie-b": 2.0, "tie-a": 2.0})
        record_sort_times(session, project_id=project_id, conversation_id=conversation_id, times={"old": 3.0})
        record_sort_times(session, project_id=other_id, conversation_id=conversation_id, times={"elsewhere": 4.0})
        newest = newest_folders(
            session, roots=[HeldRoot(project_id, None), HeldRoot(project_id, conversation_id)], limit=10
        )
    finally:
        session.close()

    assert newest.rows == (
        SortTime(project_id, conversation_id, "old", 3.0),
        SortTime(project_id, None, "tie-a", 2.0),
        SortTime(project_id, None, "tie-b", 2.0),
    )


def test_a_run_of_tied_times_is_cut_in_slug_order(engine):
    """PostgreSQL's top-N sort keeps no order among ties, so the query cuts
    a run of folders with one time in one root by slug: the same rows on
    every read, and the ones a walk of the root (sorted by name) puts first.
    The names mix case and hyphens, which a locale collation such as
    en_US.utf8 orders differently ("a-report" before "B-report", and
    "salesforce" before "sales-q"); the slug column compares by code point,
    as Python does."""
    project_id = _project(engine)
    slugs = sorted(
        [f"B-report-{index:02d}" for index in range(30)]
        + [f"a-report-{index:02d}" for index in range(30)]
        + [f"sales-q{index:02d}" for index in range(40)]
        + [f"salesforce{index:02d}" for index in range(40)]
    )
    written = list(slugs)
    random.Random(4082).shuffle(written)
    _sentinel_written(engine)
    session = _session(engine)
    try:
        record_sort_times(session, project_id=project_id, times={slug: 1_000.0 for slug in written})
        newest = newest_folders(session, roots=[HeldRoot(project_id, None)], limit=80)
    finally:
        session.close()

    assert [row.slug for row in newest.rows] == slugs[:80]
    assert newest.rows[-1].slug == "sales-q19"
    assert not newest.complete


def test_the_slug_column_compares_by_code_point_whatever_the_databases_collation(engine):
    """The revision gives slug the "C" collation. CI's Alpine image already
    compares text by byte, so there the test above passes without it; a
    database on a glibc locale (Debian's postgres image, RDS) does not."""
    with engine.connect() as connection:
        collation = connection.execute(
            text(
                "SELECT collation_name FROM information_schema.columns "
                "WHERE table_name = 'artifact_sort_times' AND column_name = 'slug'"
            )
        ).scalar_one()
    assert collation == "C"


def test_the_downgrade_takes_the_sentinel_and_the_progress_with_the_table(engine):
    write_sentinel(engine, SENTINEL_KEY, value='{"unread": []}')
    write_sentinel(engine, PROGRESS_KEY, value='{"after": null, "unread": []}')
    write_sentinel(engine, HELD_KEY, value='{"unread": []}')

    command.downgrade(Config("alembic.ini"), "7a18a01d56a0")

    with Session(engine) as raw:
        keys = {row.key for row in raw.exec(select(Setting)).all()}
    assert "artifact_sort_times" not in inspect(engine).get_table_names()
    assert SENTINEL_KEY not in keys and PROGRESS_KEY not in keys
    assert HELD_KEY in keys
