"""The skill_access backfill on PostgreSQL (ENG-3787).

cowork.services.skill_access.backfill_legacy_access runs in the middle of a
request, commits in a session of its own, one insert per transaction, and
relies on the unique keys of ``skill_access`` and ``settings`` when two
replicas backfill one org at once. The SQLite suite
(tests/test_org_skill_sharing.py) shares one connection between every
session, so there the backfill's commit also commits the request's work, and
nothing runs at once: it can show none of this. These run when
COWORK_TEST_POSTGRES_URI points at a disposable database, on the schema the
migrations build.
"""
from __future__ import annotations

import os
import threading
from collections.abc import Iterator
from contextlib import contextmanager

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine, text
from sqlalchemy.pool import NullPool
from sqlmodel import Session

import cowork.services.skill_access as skill_access
from cowork.common.settings.app_settings import get_app_settings
from cowork.db.scoped import ScopedSession, TenantScope
from cowork.models.setting import Setting
from cowork.models.shared_resource import SharedResourceAttribution
from cowork.models.skill_access import SkillAccess
from cowork.services.shared_resources import SKILL

POSTGRES_URI = os.environ.get("COWORK_TEST_POSTGRES_URI", "")

pytestmark = pytest.mark.skipif(not POSTGRES_URI, reason="set COWORK_TEST_POSTGRES_URI to a disposable database")

ORG_ID = "33333333-3333-4333-8333-333333333333"
ALICE = "a1111111-1111-4111-8111-111111111111"
BOB = "b2222222-2222-4222-8222-222222222222"
SLUGS = ["alice-guide", "house-style"]
# What the backfill writes for SLUGS: each org-wide, owned by the creator
# its attribution names, or by nobody.
BACKFILLED = [("alice-guide", True, ALICE), ("house-style", True, None)]


def _empty_public_schema() -> None:
    """Drop everything the migrations built, alembic_version included, on a
    connection of its own. A downgrade to base can't stand in: on PostgreSQL
    the downgrade of f7d2b9e4a1c6 compares a uuid column to text and fails
    partway, leaving rows that make the next test's inserts collide. The
    lock timeout turns a session a failed test left open into an error here
    rather than a teardown that waits forever."""
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


def _session(engine, user: str = ALICE) -> ScopedSession:
    """A request's session: its own connection, at the server's default
    isolation (READ COMMITTED)."""
    return ScopedSession(Session(engine), TenantScope(org_mode=True, org_id=ORG_ID, user_id=user))


@contextmanager
def _another_connection(engine) -> Iterator[ScopedSession]:
    session = _session(engine, BOB)
    try:
        yield session
    finally:
        session.close()


def _attributed(engine, slug: str, creator: str) -> None:
    with _another_connection(engine) as session:
        session.add(SharedResourceAttribution(
            resource_kind=SKILL, resource_key=slug, created_by_id=creator, updated_by_id=creator,
        ))
        session.commit()


def _committed_rows(engine) -> list[tuple[str, bool, str | None]]:
    """The org's rows as a connection of its own sees them: committed only."""
    with _another_connection(engine) as session:
        return sorted(
            (row.skill_slug, row.org_allowed, row.owner_user_id)
            for row in session.exec(session.select(SkillAccess)).all()
        )


def _committed_settings(engine, key: str) -> int:
    with _another_connection(engine) as session:
        return len(session.exec(
            session.select(Setting).where(Setting.key == key, Setting.org_id == ORG_ID)
        ).all())


def test_the_backfill_never_commits_the_requests_unfinished_work(engine):
    """The request has written something it hasn't committed when its first
    read of the org's rows runs the backfill. The backfill's rows commit;
    the request's work stays the request's to commit or roll back."""
    _attributed(engine, "alice-guide", ALICE)
    request = _session(engine)
    request.add(Setting(key="request-work", value="1", scope="user", org_id=ORG_ID, user_id=ALICE))
    request.flush()

    skill_access.backfill_legacy_access(request, lambda: SLUGS)

    assert _committed_rows(engine) == BACKFILLED
    assert _committed_settings(engine, skill_access.BACKFILL_SETTING_KEY) == 1
    assert _committed_settings(engine, "request-work") == 0
    request.rollback()
    request.close()
    assert _committed_settings(engine, "request-work") == 0
    assert _committed_rows(engine) == BACKFILLED


def test_two_replicas_backfilling_one_org_at_once_write_each_row_once(engine, monkeypatch):
    """Both find no settings row and no skill rows before either writes.
    The unique keys decide: one row per skill, one settings row for the org,
    and neither request fails."""
    _attributed(engine, "alice-guide", ALICE)
    both_read = threading.Barrier(2, timeout=15)
    read_rows = skill_access.access_rows

    def rows_then_wait(session):
        rows = read_rows(session)
        both_read.wait()
        return rows

    monkeypatch.setattr(skill_access, "access_rows", rows_then_wait)
    errors: list[BaseException] = []

    def backfill(user: str) -> None:
        session = _session(engine, user)
        try:
            skill_access.backfill_legacy_access(session, lambda: SLUGS)
        except BaseException as exc:  # noqa: BLE001 - reported below
            errors.append(exc)
        finally:
            session.close()

    replicas = [threading.Thread(target=backfill, args=(user,)) for user in (ALICE, BOB)]
    for replica in replicas:
        replica.start()
    for replica in replicas:
        replica.join(30)

    assert not any(replica.is_alive() for replica in replicas)
    assert errors == []
    assert _committed_rows(engine) == BACKFILLED
    assert _committed_settings(engine, skill_access.BACKFILL_SETTING_KEY) == 1


def test_the_request_reads_the_rows_the_backfill_committed_in_its_open_transaction(engine):
    """The request's transaction is open (it has read the org's settings)
    when the backfill commits in another session. At READ COMMITTED its next
    read sees the rows, so the first read of the org already has them."""
    _attributed(engine, "alice-guide", ALICE)
    request = _session(engine)
    assert not skill_access._backfilled(request)

    skill_access.backfill_legacy_access(request, lambda: SLUGS)

    rows = skill_access.access_rows(request)
    assert sorted((slug, row.org_allowed, row.owner_user_id) for slug, row in rows.items()) == BACKFILLED
    assert skill_access._backfilled(request)
    request.close()
