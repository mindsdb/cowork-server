from __future__ import annotations

import logging
from pathlib import Path
from unittest import mock

import pytest
import sqlalchemy as sa
from alembic import command
from fastapi import HTTPException
from sqlalchemy.exc import IntegrityError
from sqlalchemy.exc import TimeoutError as PoolTimeoutError

from cowork.common.settings import app_settings
from cowork.db import session as db_session
from cowork.db.migrations import INITIAL_REVISION, _alembic_config
from cowork.db.units import DatabaseBusy

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
def database_uri(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    # get_session takes no parameters and opens settings.database.uri.
    uri = f"sqlite:///{tmp_path / 'session.db'}"
    monkeypatch.setattr(db_session.settings.database, "uri", uri)
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
    generator = db_session.get_session()
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
    generator = db_session.get_session()
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
        records = [record for record in _records(caplog, logging.DEBUG) if "expected refusal" in record.getMessage()]
        assert len(records) == 1
        assert records[0].getMessage() == (
            f"Session rolled back after an expected refusal: error_type=HTTPException status={status}"
        )
        assert not _records(caplog, logging.ERROR)
    else:
        records = _records(caplog, logging.ERROR)
        assert len(records) == 1
        assert records[0].getMessage() == "Session error: error_type=HTTPException status=500 cause=none"
    _assert_safe(records)


def _raised_http_error(*, chain: str) -> HTTPException:
    """A 500 whose cause is an ordinary failure, linked the way routes link it."""
    try:
        try:
            raise OSError(28, "No space left on device")
        except OSError as failure:
            if chain == "cause":
                raise HTTPException(status_code=500, detail={"private": SECRETS}) from failure
            if chain == "context":
                raise HTTPException(status_code=500, detail={"private": SECRETS})
            raise HTTPException(status_code=500, detail={"private": SECRETS}) from None
    except HTTPException as raised:
        return raised


@pytest.mark.parametrize("chain", ["cause", "context", "suppressed"])
def test_a_5xx_http_exception_logs_its_status_and_cause_but_not_its_detail(
    chain: str,
    database_uri: str,
    caplog: pytest.LogCaptureFixture,
    owned_logger,
) -> None:
    # Starlette answers an HTTPException without logging it, so this line is
    # the only record of why the request failed.
    logged = owned_logger(db_session.logger.name)
    generator = db_session.get_session()
    next(generator)
    exc = _raised_http_error(chain=chain)
    with pytest.raises(HTTPException) as raised:
        generator.throw(exc)
    assert raised.value is exc
    [record] = _records(caplog, logging.ERROR)
    if chain == "suppressed":
        assert record.getMessage() == "Session error: error_type=HTTPException status=500 cause=none"
        assert record.exc_info is None
    else:
        assert record.getMessage() == "Session error: error_type=HTTPException status=500 cause=OSError"
        assert isinstance(record.exc_info[1], OSError) and record.exc_info[1] is exc.__context__
        assert "OSError: [Errno 28] No space left on device" in logged.output()
    assert not any(secret in logged.output() for secret in SECRETS)
    assert not any(secret in repr(record.__dict__) for secret in SECRETS)


# A 4xx on an error that is not an HTTPException, such as a provider SDK's
# BadRequestError, is not the endpoint's answer, so it is no expected refusal.
@pytest.mark.parametrize("status", [400, 404, 502])
def test_an_error_with_a_status_that_is_not_an_http_exception_logs_only_its_type(
    status: int,
    database_uri: str,
    caplog: pytest.LogCaptureFixture,
    owned_logger,
) -> None:
    # Starlette re-raises it to Uvicorn, which logs the whole chain through
    # the filters, so the session line names only the type.
    class UpstreamFailure(Exception):
        status_code = status

    logged = owned_logger(db_session.logger.name, level=logging.DEBUG)
    generator = db_session.get_session()
    next(generator)
    exc = UpstreamFailure(" ".join(SECRETS))
    exc.__cause__ = OSError(28, "No space left on device")
    with pytest.raises(UpstreamFailure) as raised:
        generator.throw(exc)
    assert raised.value is exc
    assert db_session._is_expected_refusal(exc) is False
    assert not [record for record in _records(caplog, logging.DEBUG) if "expected refusal" in record.getMessage()]
    [record] = _records(caplog, logging.ERROR)
    assert record.getMessage() == "Session error: error_type=UpstreamFailure"
    _assert_safe([record])
    assert "No space left on device" not in logged.output()


def test_a_5xx_http_exception_from_a_database_error_logs_only_safe_metadata(
    database_uri: str,
    caplog: pytest.LogCaptureFixture,
    owned_logger,
) -> None:
    logged = owned_logger(db_session.logger.name)
    engine = db_session.get_engine(database_uri)
    with engine.begin() as connection:
        connection.exec_driver_sql(f"CREATE TABLE {SQL_SECRET} ({DETAIL_SECRET} TEXT UNIQUE)")
    generator = db_session.get_session()
    db = next(generator)
    statement = sa.text(f"INSERT INTO {SQL_SECRET} ({DETAIL_SECRET}) VALUES (:value)")
    db.execute(statement, {"value": BIND_SECRET})
    try:
        try:
            db.execute(statement, {"value": BIND_SECRET})
        except IntegrityError as failure:
            raise HTTPException(status_code=500, detail=str(failure)) from failure
    except HTTPException as exc:
        with pytest.raises(HTTPException):
            generator.throw(exc)
    [record] = _records(caplog, logging.ERROR)
    assert record.getMessage().startswith("Database operation failed: error_type=IntegrityError sqlstate=unknown ")
    _assert_safe([record])
    assert not any(secret in logged.output() for secret in SECRETS)


@pytest.mark.parametrize(
    "exc",
    [
        pytest.param(PoolTimeoutError(f"QueuePool limit reached {' '.join(SECRETS)}"), id="pool"),
        pytest.param(DatabaseBusy(f"no database unit slot freed {' '.join(SECRETS)}"), id="unit-slot"),
    ],
)
def test_a_full_pool_rolls_back_and_logs_one_debug_line_for_the_503(
    exc: PoolTimeoutError,
    database_uri: str,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    # The app answers both with a 503 that names the wait (cowork.server). A
    # plain pool timeout has no status_code, so the line names 503 itself; an
    # error-level line per refused request would bury the server's own line.
    generator = db_session.get_session()
    db = next(generator)
    rollback = mock.Mock(wraps=db.rollback)
    close = mock.Mock(wraps=db.close)
    monkeypatch.setattr(db, "rollback", rollback)
    monkeypatch.setattr(db, "close", close)
    with caplog.at_level(logging.DEBUG, logger=db_session.logger.name):
        with pytest.raises(PoolTimeoutError) as raised:
            generator.throw(exc)
    assert raised.value is exc
    rollback.assert_called_once_with()
    close.assert_called_once_with()
    records = [record for record in _records(caplog, logging.DEBUG) if "expected refusal" in record.getMessage()]
    assert len(records) == 1
    assert records[0].getMessage() == (
        f"Session rolled back after an expected refusal: error_type={type(exc).__name__} status=503"
    )
    assert not _records(caplog, logging.ERROR)
    _assert_safe(records)


def test_an_unexpected_failure_logs_its_type_without_exception_text(
    database_uri: str,
    caplog: pytest.LogCaptureFixture,
) -> None:
    generator = db_session.get_session()
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
