"""Both storage engines enforce one ending per run after upgrading old inboxes."""
import importlib
from datetime import datetime, timedelta, timezone
from uuid import uuid4

import pytest
import sqlalchemy as sa
from alembic.migration import MigrationContext
from alembic.operations import Operations
from sqlmodel import Session, create_engine, select

from cowork.models.notice import Notice


@pytest.fixture
def notice_postgres(monkeypatch):
    import os
    from alembic import command
    from cowork.common.settings.app_settings import get_app_settings
    from cowork.db.migrations import _alembic_config
    from tests.integration.prereq import missing_prerequisite

    uri = os.environ.get("COWORK_TEST_POSTGRES_URI")
    if not uri:
        missing_prerequisite("set COWORK_TEST_POSTGRES_URI to a disposable database",
                             required_by="COWORK_REQUIRE_SERVICES")
    monkeypatch.setenv("DATABASE_URI", uri)
    get_app_settings.cache_clear()
    # Use the application's programmatic config so migrations leave pytest's
    # logging configuration intact for the tests that follow.
    config = _alembic_config(uri)
    engine = create_engine(uri)
    try:
        command.upgrade(config, "head")
        yield engine
    finally:
        engine.dispose()
        command.downgrade(config, "base")
        get_app_settings.cache_clear()


def _upgrade_old_inbox(engine, monkeypatch):
    migration = importlib.import_module("cowork.db.alembic.versions.7e93b26a4c18_unique_scheduled_run_notice")
    source = str(uuid4())
    now = datetime.now(timezone.utc)
    with engine.begin() as connection:
        monkeypatch.setattr(migration, "op", Operations(MigrationContext.configure(connection)))
        migration.downgrade()
    with Session(engine) as session:
        first = Notice(kind="scheduled_run.failed", source_id=source, subject="first", created_at=now, read_at=now)
        duplicate = Notice(kind="scheduled_run.stopped", source_id=source, subject="second", created_at=now + timedelta(seconds=1))
        other_kind = Notice(kind="other", source_id=source, subject="other")
        session.add_all([first, duplicate, other_kind])
        session.commit()
        first_id, other_id = first.id, other_kind.id
    with engine.begin() as connection:
        monkeypatch.setattr(migration, "op", Operations(MigrationContext.configure(connection)))
        migration.upgrade()
        migration.upgrade()  # Model-created databases and repeated upgrades are safe.
    with Session(engine) as session:
        rows = session.exec(select(Notice).where(Notice.source_id == source)).all()
        assert {row.id for row in rows} == {first_id, other_id}
        assert session.get(Notice, first_id).read_at is not None
        session.add(Notice(kind="scheduled_run.succeeded", source_id=source, subject="third"))
        with pytest.raises(sa.exc.IntegrityError):
            session.commit()
        session.rollback()
        # Unrelated notices without a source are not deduplicated.
        session.add_all([Notice(kind="scheduled_run.failed"), Notice(kind="scheduled_run.failed")])
        session.commit()
    with engine.begin() as connection:
        monkeypatch.setattr(migration, "op", Operations(MigrationContext.configure(connection)))
        migration.downgrade()
        migration.upgrade()


def test_notice_dedup_migration_sqlite(tmp_path, monkeypatch):
    engine = create_engine(f"sqlite:///{tmp_path / 'notices.db'}")
    try:
        Notice.__table__.create(engine)
        _upgrade_old_inbox(engine, monkeypatch)
    finally:
        engine.dispose()


@pytest.mark.postgres_service
def test_notice_dedup_migration_postgres(notice_postgres, monkeypatch):
    _upgrade_old_inbox(notice_postgres, monkeypatch)
