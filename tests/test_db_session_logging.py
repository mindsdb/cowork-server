from __future__ import annotations

import logging
from pathlib import Path
from unittest import mock

import pytest
import sqlalchemy as sa
from alembic import command
from fastapi import HTTPException
from sqlalchemy.exc import IntegrityError

from cowork.common.settings import app_settings
from cowork.db import session as db_session
from cowork.db.migrations import INITIAL_REVISION, _alembic_config

BIND_SECRET = "bound_credential_marker"
SQL_SECRET = "private_statement_marker"
DETAIL_SECRET = "private_error_detail_marker"
SECRETS = (BIND_SECRET, SQL_SECRET, DETAIL_SECRET)


def _records(caplog: pytest.LogCaptureFixture, level: int) -> list[logging.LogRecord]:
    return [
        record for record in caplog.records
        if record.name == db_session.logger.name and record.levelno == level
    ]


def _assert_safe(records: list[logging.LogRecord]) -> None:
    assert records
    for record in records:
        assert record.exc_info is None
        assert record.exc_text is None
        assert record.stack_info is None
        assert not any(secret in repr(record.__dict__) for secret in SECRETS)


@pytest.fixture
def database_uri(tmp_path: Path):
    uri = f"sqlite:///{tmp_path / 'session.db'}"
    yield uri
    engine = db_session._engines.pop(uri, None)
    if engine is not None:
        db_session._session_factories.pop(id(engine), None)
        engine.dispose()


@pytest.mark.parametrize("uri", ["sqlite://", "postgresql+psycopg://user:password@localhost/test"])
def test_both_application_engine_paths_hide_bound_parameters(uri: str) -> None:
    # PostgreSQL engine construction loads the real dialect without opening a
    # connection. SQLite below exercises what this option actually logs.
    engine = db_session._create_engine(uri)
    try:
        assert engine.hide_parameters is True
    finally:
        engine.dispose()


def test_sqlalchemy_logs_hide_bound_values_at_the_real_engine_call_site(
    caplog: pytest.LogCaptureFixture,
) -> None:
    engine = db_session._create_engine("sqlite://")
    try:
        with caplog.at_level(logging.INFO, logger="sqlalchemy.engine.Engine"):
            with engine.connect() as connection:
                assert connection.execute(sa.text("SELECT :value"), {"value": BIND_SECRET}).scalar_one() == BIND_SECRET
        records = [record for record in caplog.records if record.name == "sqlalchemy.engine.Engine"]
        assert records
        assert any("SQL parameters hidden" in record.getMessage() for record in records)
        assert all(BIND_SECRET not in repr(record.__dict__) for record in records)
    finally:
        engine.dispose()


@pytest.mark.parametrize(
    "uri",
    [
        f"sqlite://{SQL_SECRET}:{BIND_SECRET}@localhost/{DETAIL_SECRET}",
        f"postgresql+{DETAIL_SECRET}://{SQL_SECRET}:{BIND_SECRET}@localhost/test",
    ],
)
def test_engine_creation_failure_logs_only_its_type_and_preserves_the_error(
    uri: str,
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.ERROR, logger=db_session.logger.name):
        with pytest.raises(RuntimeError) as raised:
            db_session.get_engine(uri)
    cause = raised.value.__cause__
    assert cause is not None
    assert str(raised.value) == f"Engine creation failed: {str(cause).lower()}"
    assert uri not in db_session._engines
    records = _records(caplog, logging.ERROR)
    assert len(records) == 1
    assert records[0].getMessage() == f"Failed to create engine: error_type={type(cause).__name__}"
    _assert_safe(records)


def test_real_failed_sql_rolls_back_and_reraises_without_logging_sql_or_details(
    database_uri: str,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    engine = db_session.get_engine(database_uri)
    with engine.begin() as connection:
        connection.exec_driver_sql(f"CREATE TABLE {SQL_SECRET} ({DETAIL_SECRET} TEXT UNIQUE)")
    generator = db_session.get_session(database_uri)
    db = next(generator)
    rollback = mock.Mock(wraps=db.rollback)
    close = mock.Mock(wraps=db.close)
    monkeypatch.setattr(db, "rollback", rollback)
    monkeypatch.setattr(db, "close", close)
    statement = sa.text(f"INSERT INTO {SQL_SECRET} ({DETAIL_SECRET}) VALUES (:value)")
    db.execute(statement, {"value": BIND_SECRET})
    with pytest.raises(IntegrityError) as failed:
        db.execute(statement, {"value": BIND_SECRET})
    # The exception still has the SQL and the driver's error detail. Hiding
    # parameters alone cannot protect a logger that prints this exception.
    assert SQL_SECRET in str(failed.value)
    assert DETAIL_SECRET in str(failed.value.orig)
    with caplog.at_level(logging.ERROR, logger=db_session.logger.name):
        with pytest.raises(IntegrityError) as reraised:
            generator.throw(failed.value)
    assert reraised.value is failed.value
    rollback.assert_called_once_with()
    close.assert_called_once_with()
    with engine.connect() as connection:
        assert connection.exec_driver_sql(f"SELECT count(*) FROM {SQL_SECRET}").scalar_one() == 0
    records = _records(caplog, logging.ERROR)
    assert len(records) == 1
    assert records[0].getMessage() == "Session error: error_type=IntegrityError"
    _assert_safe(records)


@pytest.mark.parametrize("status", [404, 409, 500])
def test_http_error_details_are_not_logged_and_the_original_answer_is_preserved(
    status: int,
    database_uri: str,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    generator = db_session.get_session(database_uri)
    db = next(generator)
    rollback = mock.Mock(wraps=db.rollback)
    close = mock.Mock(wraps=db.close)
    monkeypatch.setattr(db, "rollback", rollback)
    monkeypatch.setattr(db, "close", close)
    exc = HTTPException(status_code=status, detail={"private": SECRETS})
    with caplog.at_level(logging.DEBUG, logger=db_session.logger.name):
        with pytest.raises(HTTPException) as raised:
            generator.throw(exc)
    assert raised.value is exc
    assert raised.value.status_code == status
    assert raised.value.detail == {"private": SECRETS}
    rollback.assert_called_once_with()
    close.assert_called_once_with()
    if status < 500:
        records = [record for record in _records(caplog, logging.DEBUG) if "client error" in record.getMessage()]
        assert len(records) == 1
        assert records[0].getMessage() == f"Session rolled back after a client error: error_type=HTTPException status={status}"
        assert not _records(caplog, logging.ERROR)
    else:
        records = _records(caplog, logging.ERROR)
        assert len(records) == 1
        assert records[0].getMessage() == "Session error: error_type=HTTPException"
    _assert_safe(records)


def test_an_unexpected_failure_logs_its_type_without_exception_text(
    database_uri: str,
    caplog: pytest.LogCaptureFixture,
) -> None:
    generator = db_session.get_session(database_uri)
    next(generator)
    exc = RuntimeError(" ".join(SECRETS))
    with caplog.at_level(logging.ERROR, logger=db_session.logger.name):
        with pytest.raises(RuntimeError) as raised:
            generator.throw(exc)
    assert raised.value is exc
    records = _records(caplog, logging.ERROR)
    assert len(records) == 1
    assert records[0].getMessage() == "Session error: error_type=RuntimeError"
    _assert_safe(records)


def test_alembic_online_engine_hides_parameters_without_a_supplied_connection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    uri = f"sqlite:///{tmp_path / 'migration.db'}"
    settings = app_settings.get_app_settings().model_copy(deep=True)
    settings.database.uri = uri
    monkeypatch.setattr(app_settings, "get_app_settings", lambda: settings)
    create_engine = sa.create_engine
    engines = []

    def capture_engine(*args, **kwargs):
        engine = create_engine(*args, **kwargs)
        engines.append(engine)
        return engine

    monkeypatch.setattr(sa, "create_engine", capture_engine)
    try:
        # command.upgrade drives env.py's online fallback; the normal startup
        # path supplies its own connection and would bypass this engine.
        command.upgrade(_alembic_config(uri), INITIAL_REVISION)
        assert len(engines) == 1
        assert engines[0].hide_parameters is True
        with caplog.at_level(logging.INFO, logger="sqlalchemy.engine.Engine"):
            with engines[0].connect() as connection:
                assert connection.execute(sa.text("SELECT :value"), {"value": BIND_SECRET}).scalar_one() == BIND_SECRET
        records = [record for record in caplog.records if record.name == "sqlalchemy.engine.Engine"]
        assert records
        assert any("SQL parameters hidden" in record.getMessage() for record in records)
        assert all(BIND_SECRET not in repr(record.__dict__) for record in records)
    finally:
        for engine in engines:
            engine.dispose()
