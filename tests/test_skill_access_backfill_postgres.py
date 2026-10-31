"""The skill_access backfill on PostgreSQL.

cowork.services.skill_access.backfill_legacy_access runs in the middle of a
request, commits in a session of its own, one insert per transaction, and
relies on the unique keys of ``skill_access`` and ``settings`` when two
replicas backfill one org at once. The SQLite suite
(tests/test_org_skill_sharing.py) shares one connection between every
session, so there the backfill's commit also commits the request's work, and
nothing runs at once: it can show none of this. The last three check the
skill sharing and artifact owner choices revisions' downgrades, the owner
choice store, and two saves racing to record an artifact's first choice, on
PostgreSQL. These run when COWORK_TEST_POSTGRES_URI
points at a disposable database, on the schema the migrations build.
"""
from __future__ import annotations

import os
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from uuid import UUID

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
# What the backfill writes for SLUGS: each shared with everyone, owned by the
# creator its attribution names, or by nobody.
BACKFILLED = [("alice-guide", True, UUID(ALICE)), ("house-style", True, None)]


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


def _committed_rows(engine) -> list[tuple[str, bool, UUID | None]]:
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


def test_a_downgrade_takes_the_marker_with_the_rows_and_the_owner_choices_with_their_table(engine):
    """On PostgreSQL too: downgrading below skill sharing deletes the
    backfill marker beside dropping the rows it vouches for, so a
    re-upgrade's first read backfills again, and the artifact owner choices'
    revision drops its table on the way down and builds it again on the way
    up."""
    from sqlalchemy import inspect

    _attributed(engine, "alice-guide", ALICE)
    request = _session(engine)
    skill_access.backfill_legacy_access(request, lambda: SLUGS)
    request.close()
    assert _committed_settings(engine, skill_access.BACKFILL_SETTING_KEY) == 1
    with engine.begin() as connection:
        connection.execute(text(
            "INSERT INTO artifact_owner_choices (id, org_id, artifact_key, access_mode, published, owner_user_id) "
            "VALUES (gen_random_uuid(), :org, 'artifact/1', 'password', true, :owner)"
        ), {"org": ORG_ID, "owner": ALICE})
    engine.dispose()

    command.downgrade(Config("alembic.ini"), "a3734c9e5210")
    with engine.connect() as connection:
        tables = set(inspect(connection).get_table_names())
        marker = connection.execute(
            text("SELECT count(*) FROM settings WHERE key = :key"), {"key": skill_access.BACKFILL_SETTING_KEY}
        ).scalar_one()
    assert not {"skill_access", "skill_user_setting", "artifact_owner_choices"} & tables
    assert marker == 0
    engine.dispose()

    command.upgrade(Config("alembic.ini"), "head")
    request = _session(engine)
    skill_access.backfill_legacy_access(request, lambda: SLUGS)
    request.close()
    assert _committed_rows(engine) == BACKFILLED
    with engine.connect() as connection:
        assert connection.execute(text("SELECT count(*) FROM artifact_owner_choices")).scalar_one() == 0


def test_the_artifact_owner_choice_is_recorded_and_replaced_on_postgresql(engine):
    """The owner's choice is written in a session of its own, as the publish
    that records it runs in a worker thread, and a later save replaces it."""
    from cowork.models.artifact_owner_choice import AccessMode
    from cowork.services import artifact_owner_choice as choices

    scope = TenantScope(org_mode=True, org_id=ORG_ID, user_id=ALICE)
    choices.record_owner_choice(key="artifact/1", choice=choices.OwnerChoice(AccessMode.PASSWORD, True), scope=scope)
    choices.record_owner_choice(key="artifact/1", choice=choices.OwnerChoice(AccessMode.PUBLIC, True), scope=scope)
    assert choices.read_owner_choice(keys=["artifact/1"], scope=scope) == choices.OwnerChoice(AccessMode.PUBLIC, True)
    with engine.connect() as connection:
        assert connection.execute(text("SELECT count(*) FROM artifact_owner_choices")).scalar_one() == 1


def test_two_saves_racing_to_record_an_artifacts_first_choice_both_succeed_on_postgresql(engine, monkeypatch):
    """The second of two saves that each found no row hits the unique key on
    commit, rolls back, and records its choice over the row the first
    committed: one row, the later save's choice, and no failed save."""
    from cowork.models.artifact_owner_choice import AccessMode
    from cowork.services import artifact_owner_choice as choices

    scope = TenantScope(org_mode=True, org_id=ORG_ID, user_id=ALICE)
    key = "artifact/raced"
    stage = choices._stage_choice
    raced: list = []

    def stage_then_race(session, *, key, choice):
        stage(session, key=key, choice=choice)
        if not raced:
            # The other save found no row either, and commits its own first.
            raced.append(choice)
            with choices.authorization_session(scope) as other:
                stage(other, key=key, choice=choices.OwnerChoice(AccessMode.PUBLIC, True))
                other.commit()

    monkeypatch.setattr(choices, "_stage_choice", stage_then_race)
    choices.record_owner_choice(key=key, choice=choices.OwnerChoice(AccessMode.RESTRICTED, True), scope=scope)

    assert raced, "the race never ran"
    assert choices.read_owner_choice(keys=[key], scope=scope) == choices.OwnerChoice(AccessMode.RESTRICTED, True)
    with engine.connect() as connection:
        assert connection.execute(text("SELECT count(*) FROM artifact_owner_choices")).scalar_one() == 1
