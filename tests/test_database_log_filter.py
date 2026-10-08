from __future__ import annotations

import io
import json
import logging
import os
import re
import subprocess
import sys
import textwrap
from pathlib import Path

import httpx
import openai
import psycopg.errors
import psycopg2.errors
import pytest
import sqlalchemy as sa
from sqlalchemy.exc import IntegrityError, OperationalError

from cowork.common import logger as app_logger
from cowork.db import session as db_session
from tests._uvicorn_harness import Launch, run_uvicorn_app
from tests.test_db_session_logging import BIND_SECRET, DETAIL_SECRET, SECRETS, SQL_SECRET


def _sql_error() -> IntegrityError:
    engine = db_session._create_engine("sqlite://")
    try:
        with engine.begin() as connection:
            connection.exec_driver_sql(f"CREATE TABLE {SQL_SECRET} ({DETAIL_SECRET} TEXT UNIQUE)")
            statement = sa.text(f"INSERT INTO {SQL_SECRET} ({DETAIL_SECRET}) VALUES (:value)")
            connection.execute(statement, {"value": BIND_SECRET})
            try:
                connection.execute(statement, {"value": BIND_SECRET})
            except IntegrityError as exc:
                return exc
        raise AssertionError("the duplicate insert must fail")
    finally:
        engine.dispose()


@pytest.mark.parametrize("handler_kind", ["console", "all_file", "error_file"])
@pytest.mark.parametrize("chain_kind", ["cause", "context", "suppressed_context", "group"])
@pytest.mark.parametrize("error_kind", ["orm", "driver"])
def test_each_owned_handler_sanitizes_a_wrapped_real_database_error(
    handler_kind: str,
    chain_kind: str,
    error_kind: str,
    caplog: pytest.LogCaptureFixture,
    owned_logger,
) -> None:
    logged = owned_logger(f"cowork.test.database_filter.{handler_kind}", kind=handler_kind)
    database_error = _sql_error()
    logged_error = database_error if error_kind == "orm" else database_error.orig
    try:
        if chain_kind == "cause":
            raise RuntimeError("wrapper " + " ".join(SECRETS)) from logged_error
        if chain_kind in ("context", "suppressed_context"):
            try:
                raise logged_error
            except Exception:
                if chain_kind == "context":
                    raise RuntimeError("wrapper " + " ".join(SECRETS))
                # `from None` hides the context from the traceback, but the
                # wrapper's own message still repeats the database text.
                raise RuntimeError("wrapper " + " ".join(SECRETS)) from None
        raise ExceptionGroup("group " + " ".join(SECRETS), [logged_error])
    except Exception as wrapper:
        # The whole message must be replaced: callers such as scheduler
        # interpolate an error before adding exc_info. The caller's ids
        # travel as record attributes and survive the replacement.
        logged.logger.exception(
            f"Raw database error: {wrapper}",
            extra=app_logger.log_context(
                request_id="request-123",
                project_id="project-1",
                schedule_id="schedule-1",
                conversation_id="conversation-1",
                artifact_slug="q3-report",
            ),
            stack_info=True,
        )
    records = [record for record in caplog.records if record.name == logged.logger.name and record.levelno == logging.ERROR]
    assert len(records) == 1
    record = records[0]
    assert record.getMessage() == (
        "Database operation failed: error_type=IntegrityError sqlstate=unknown "
        f"site=test_database_log_filter.test_each_owned_handler_sanitizes_a_wrapped_real_database_error:{record.lineno}"
    )
    assert (record.request_id, record.project_id, record.schedule_id, record.conversation_id, record.artifact_slug) == (
        "request-123", "project-1", "schedule-1", "conversation-1", "q3-report",
    )
    assert record.exc_info is None
    assert record.exc_text is None
    assert record.stack_info is None
    assert not any(secret in repr(record.__dict__) for secret in SECRETS)
    emitted = logged.output()
    assert "error_type=IntegrityError" in emitted
    assert (
        "[Req:request-123][Project:project-1][Schedule:schedule-1][Conversation:conversation-1][Artifact:'q3-report']"
        in emitted
    )
    assert not any(secret in emitted for secret in SECRETS)
    assert database_error.statement and SQL_SECRET in database_error.statement
    assert BIND_SECRET in repr(database_error.params)
    assert DETAIL_SECRET in str(database_error.orig)


@pytest.mark.parametrize("sqlstate", ["23505", DETAIL_SECRET, None])
def test_database_exception_arguments_and_sqlstate_are_sanitized(
    sqlstate: str | None,
    caplog: pytest.LogCaptureFixture,
    owned_logger,
) -> None:
    class DriverFailure(Exception):
        pass

    DriverFailure.sqlstate = sqlstate
    error = OperationalError(" ".join(SECRETS), {"secret": BIND_SECRET}, DriverFailure(DETAIL_SECRET))
    logged = owned_logger("cowork.test.database_filter.argument")
    # Coding-service synchronization logs its exception as an argument
    # without exc_info, so the filter must inspect arguments too.
    logged.logger.error("Database failed: %s", error)
    records = [record for record in caplog.records if record.name == logged.logger.name and record.levelno == logging.ERROR]
    assert len(records) == 1
    expected_state = "23505" if sqlstate == "23505" else "unknown"
    assert records[0].getMessage() == (
        f"Database operation failed: error_type=OperationalError sqlstate={expected_state} "
        f"site=test_database_log_filter.test_database_exception_arguments_and_sqlstate_are_sanitized:"
        f"{records[0].lineno}"
    )
    assert not any(secret in repr(records[0].__dict__) for secret in SECRETS)
    assert not any(secret in logged.output() for secret in SECRETS)
    assert error.orig.sqlstate == sqlstate


def test_other_exception_logs_keep_their_message_and_traceback(
    caplog: pytest.LogCaptureFixture,
    owned_logger,
) -> None:
    logged = owned_logger("cowork.test.database_filter.other")
    try:
        raise RuntimeError("ordinary application failure")
    except RuntimeError:
        logged.logger.exception("Other failure")
    records = [record for record in caplog.records if record.name == logged.logger.name and record.levelno == logging.ERROR]
    assert len(records) == 1
    assert records[0].getMessage() == "Other failure"
    assert records[0].exc_info is not None
    assert str(records[0].exc_info[1]) == "ordinary application failure"
    assert "RuntimeError: ordinary application failure" in logged.output()


# A route's database error escapes get_session, and Starlette re-raises it to
# Uvicorn, which logs "Exception in ASGI application" with the whole chain.
_DATABASE_FAILURE_APP = f"""
    import sqlalchemy as sa
    from fastapi import Depends, FastAPI
    from cowork.common.logger import setup_logging
    from cowork.db.session import get_session

    # cowork.server runs this at import, under every launcher.
    setup_logging()
    app = FastAPI()

    @app.get("/fail")
    def fail(session=Depends(get_session)):
        session.execute(sa.text("CREATE TABLE {SQL_SECRET} ({DETAIL_SECRET} TEXT UNIQUE)"))
        for _ in range(2):
            session.execute(sa.text("INSERT INTO {SQL_SECRET} VALUES ('{BIND_SECRET}')"))
"""


@pytest.mark.parametrize("launch", ["module", "run"])
def test_uvicorns_default_log_config_prints_only_sanitized_database_errors(tmp_path: Path, launch: Launch) -> None:
    run = run_uvicorn_app(tmp_path=tmp_path, app_source=_DATABASE_FAILURE_APP, paths=("/fail",), launch=launch)
    assert run.statuses == (500,)
    assert re.search(
        r"Database operation failed: error_type=IntegrityError sqlstate=unknown site=(h11|httptools)_impl\.run_asgi:\d+",
        run.stderr,
    ), run.stderr
    # uvicorn.error keeps its INFO level, so the shutdown line still prints.
    assert "Finished server process" in run.stderr
    assert not any(secret in run.stdout + run.stderr for secret in SECRETS)


@pytest.mark.parametrize("order", ["setup_logging_first", "dict_config_first"])
def test_setup_logging_adds_each_uvicorn_filter_once_in_either_configuration_order(tmp_path: Path, order: str) -> None:
    # setup_logging runs at import in several modules. The CLI order runs it
    # before Uvicorn's dictConfig; `python -m uvicorn` runs dictConfig first.
    # Both close process-wide handlers, so run them apart.
    program = textwrap.dedent(f"""
        import io, json, logging, logging.config, sys
        import sqlalchemy as sa
        from uvicorn.config import LOGGING_CONFIG
        from cowork.common.logger import setup_logging
        dict_config_first = sys.argv[1] == "dict_config_first"
        if dict_config_first:
            logging.config.dictConfig(LOGGING_CONFIG)
        for _ in range(3):
            setup_logging()
        if not dict_config_first:
            logging.config.dictConfig(LOGGING_CONFIG)
        names = ('uvicorn', 'uvicorn.error', 'uvicorn.access', 'uvicorn.asgi')
        stream = io.StringIO()
        for handler in logging.getLogger('uvicorn').handlers:
            handler.setStream(stream)
        engine = sa.create_engine('sqlite://')
        with engine.connect() as connection:
            connection.exec_driver_sql("CREATE TABLE {SQL_SECRET} (value TEXT UNIQUE)")
            connection.exec_driver_sql("INSERT INTO {SQL_SECRET} VALUES ('{BIND_SECRET}')")
            try:
                connection.exec_driver_sql("INSERT INTO {SQL_SECRET} VALUES ('{BIND_SECRET}')")
            except sa.exc.IntegrityError:
                # A child logger's record reaches the parent's handler, which
                # only a filter on that handler reads.
                logging.getLogger('uvicorn.error.child').error('Exception in ASGI application', exc_info=True)
        print(json.dumps({{
            'logger_filters': {{name: [type(f).__name__ for f in logging.getLogger(name).filters] for name in names}},
            'handler_filters': {{
                name: [[type(f).__name__ for f in handler.filters] for handler in logging.getLogger(name).handlers]
                for name in ('uvicorn', 'uvicorn.access')
            }},
            'error_level': logging.getLogger('uvicorn.error').level,
            'child_output': stream.getvalue(),
        }}))
    """)
    result = subprocess.run(
        [sys.executable, "-c", program, order], capture_output=True, text=True,
        env={**os.environ, "COWORK_HOME": str(tmp_path), "DATABASE_URI": "sqlite://"}, timeout=60,
    )
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout.splitlines()[-1])
    expected = ["DatabaseErrorFilter", "ProviderErrorFilter"]
    assert report["logger_filters"] == {
        "uvicorn": expected, "uvicorn.error": expected, "uvicorn.access": expected, "uvicorn.asgi": expected,
    }
    assert report["error_level"] == logging.INFO
    if order == "dict_config_first":
        # The handlers Uvicorn installed before setup_logging ran carry the
        # filters too, so a record from a child logger is replaced.
        assert report["handler_filters"] == {"uvicorn": [expected], "uvicorn.access": [expected]}
        assert "Database operation failed: error_type=IntegrityError" in report["child_output"]
        assert not any(secret in report["child_output"] for secret in SECRETS)
    else:
        # dictConfig replaces the handlers; Uvicorn logs only on the loggers
        # that carry the filters themselves.
        assert report["handler_filters"] == {"uvicorn": [[]], "uvicorn.access": [[]]}


async def _start_and_stop(*, app) -> bool:
    """Drive the app's lifespan as Uvicorn does; True when startup failed."""
    import uvicorn
    from uvicorn.lifespan.on import LifespanOn

    # log_config=None: Uvicorn's dictConfig would close pytest's handlers.
    config = uvicorn.Config(app, lifespan="on", log_config=None)
    config.load()
    lifespan = LifespanOn(config)
    await lifespan.startup()
    if lifespan.should_exit:
        return True
    await lifespan.shutdown()
    return False


@pytest.fixture
def uvicorn_error_lines(monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture):
    """The ERROR lines Uvicorn logs, captured on its own logger.

    Building the app re-runs setup_logging, which replaces root's handlers, so
    caplog's root handler would miss them.
    """
    uvicorn_logger = logging.getLogger("uvicorn.error")
    monkeypatch.setattr(uvicorn_logger, "handlers", [caplog.handler])
    monkeypatch.setattr(uvicorn_logger, "propagate", False)
    monkeypatch.setattr(uvicorn_logger, "level", logging.INFO)
    return lambda: [record.getMessage() for record in caplog.records
                    if record.name == "uvicorn.error" and record.levelno == logging.ERROR]


async def test_a_database_error_at_startup_leaves_no_database_text_in_uvicorns_log(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, owned_logger, uvicorn_error_lines,
) -> None:
    from cowork import server

    database_error = _sql_error()

    def failing_setup() -> None:
        raise database_error

    monkeypatch.setattr(server, "run_dev_setup", failing_setup)
    logged = owned_logger(server.logger.name)
    assert await _start_and_stop(app=server.create_app()) is True
    # Starlette hands Uvicorn the failure as traceback text, which no filter reads.
    lines = uvicorn_error_lines()
    for captured in caplog.records:
        assert not any(secret in captured.getMessage() for secret in SECRETS), captured.getMessage()
    assert not any(secret in logged.output() for secret in SECRETS)
    assert any(line.rstrip().endswith("RuntimeError: Database operation failed during server startup") for line in lines), lines
    assert "Application startup failed. Exiting." in lines
    [record] = [record for record in caplog.records
                if record.name == server.logger.name and record.levelno == logging.ERROR]
    assert record.getMessage().startswith(
        "Database operation failed: error_type=IntegrityError sqlstate=unknown site=server._lifespan_without_database_text:"
    )


async def test_a_database_error_at_shutdown_leaves_no_database_text_in_uvicorns_log(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, uvicorn_error_lines,
) -> None:
    from contextlib import asynccontextmanager

    from cowork import server

    database_error = _sql_error()

    @asynccontextmanager
    async def failing_shutdown(app):
        yield
        raise database_error

    monkeypatch.setattr(server, "lifespan", failing_shutdown)
    assert await _start_and_stop(app=server.create_app()) is False
    lines = uvicorn_error_lines()
    for captured in caplog.records:
        assert not any(secret in captured.getMessage() for secret in SECRETS), captured.getMessage()
    assert any(line.rstrip().endswith("RuntimeError: Database operation failed during server shutdown") for line in lines), lines


async def test_a_startup_error_without_a_database_cause_keeps_its_traceback(
    monkeypatch: pytest.MonkeyPatch, uvicorn_error_lines,
) -> None:
    from cowork import server

    def failing_setup() -> None:
        raise PermissionError("ordinary startup failure")

    monkeypatch.setattr(server, "run_dev_setup", failing_setup)
    assert await _start_and_stop(app=server.create_app()) is True
    lines = uvicorn_error_lines()
    assert any(line.rstrip().endswith("PermissionError: ordinary startup failure") for line in lines), lines


def _provider_error() -> openai.BadRequestError:
    return openai.BadRequestError(
        SQL_SECRET, response=httpx.Response(400, request=httpx.Request("POST", "https://example.com")), body=None,
    )


@pytest.mark.parametrize("error_kind", ["database", "provider"])
def test_a_record_another_handler_formatted_first_loses_its_cached_traceback(
    error_kind: str, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, owned_logger,
) -> None:
    logged = owned_logger("cowork.test.database_filter.formatted_first")
    plain_stream = io.StringIO()
    plain = logging.StreamHandler(plain_stream)
    # The plain handler formats first and caches the traceback in exc_text.
    monkeypatch.setattr(logged.logger, "handlers", [plain, *logged.logger.handlers])
    if error_kind == "database":
        error = _sql_error()
    else:
        error = _provider_error()
    try:
        try:
            raise RuntimeError("wrapper " + " ".join(SECRETS)) from error
        except RuntimeError:
            logged.logger.exception("Failure: %s", " ".join(SECRETS))
        assert SQL_SECRET in plain_stream.getvalue()
        [record] = [record for record in caplog.records if record.name == logged.logger.name]
        assert record.exc_text is None
        assert "Traceback" not in logged.output()
        assert not any(secret in logged.output() for secret in SECRETS)
    finally:
        plain.close()


@pytest.mark.parametrize("filter_type", [app_logger.DatabaseErrorFilter, app_logger.ProviderErrorFilter])
def test_the_filters_reset_the_message_a_formatter_cached(filter_type) -> None:
    if filter_type is app_logger.DatabaseErrorFilter:
        error = _sql_error()
    else:
        error = _provider_error()
    record = logging.LogRecord("cowork.test", logging.ERROR, __file__, 1, "Failure: %s", (error,), None)
    logging.Formatter().format(record)
    assert SQL_SECRET in record.message
    assert filter_type().filter(record)
    assert record.message == record.getMessage()
    assert SQL_SECRET not in record.message


class Psycopg2UniqueViolation(psycopg2.errors.UniqueViolation):
    # psycopg2 fills pgcode only for an error it raises from a server reply.
    pgcode = "23505"


# What Postgres puts in a unique violation's text: DETAIL echoes the value,
# even when the engine hides bound parameters.
_POSTGRES_DETAIL = (
    'duplicate key value violates unique constraint "users_email_key"\n'
    f"DETAIL:  Key (email)=({DETAIL_SECRET}) already exists."
)


@pytest.mark.parametrize("wrapped", [False, True], ids=["raw", "wrapped"])
@pytest.mark.parametrize("driver", ["psycopg", "psycopg2"])
def test_real_postgres_driver_errors_log_only_their_type_and_sqlstate(
    driver: str, wrapped: bool, caplog: pytest.LogCaptureFixture, owned_logger,
) -> None:
    driver_error = (psycopg.errors.UniqueViolation if driver == "psycopg" else Psycopg2UniqueViolation)(_POSTGRES_DETAIL)
    error = (
        IntegrityError(f"INSERT INTO users (email) VALUES ('{SQL_SECRET}')", {"email": BIND_SECRET}, driver_error,
                       hide_parameters=True)
        if wrapped else driver_error
    )
    logged = owned_logger("cowork.test.database_filter.postgres")
    try:
        raise error
    except Exception:
        logged.logger.exception("Insert failed: %s", error)
    [record] = [record for record in caplog.records if record.name == logged.logger.name]
    assert record.getMessage() == (
        f"Database operation failed: error_type={type(error).__name__} sqlstate=23505 "
        f"site=test_database_log_filter.test_real_postgres_driver_errors_log_only_their_type_and_sqlstate:"
        f"{record.lineno}"
    )
    assert record.exc_info is None and record.exc_text is None
    assert not any(secret in logged.output() for secret in SECRETS)
    assert DETAIL_SECRET in str(driver_error)


def test_app_logging_disables_sqlalchemy_query_and_connection_diagnostics(tmp_path: Path) -> None:
    # SQLAlchemy defaults its parent namespace to WARNING. An explicit debug
    # configuration can change that; application setup must still keep query
    # literals, result rows and pool connection representations out of logs.
    program = textwrap.dedent("""
        import json, logging
        from cowork.common.logger import setup_logging
        from cowork.db.session import get_engine
        logging.getLogger('sqlalchemy').setLevel(logging.DEBUG)
        setup_logging()
        records = []
        class Capture(logging.Handler):
            def emit(self, record):
                records.append(record)
        logging.getLogger().addHandler(Capture())
        engine = get_engine('sqlite://')
        with engine.connect() as connection:
            result = connection.exec_driver_sql("SELECT 'private_statement_marker'").scalar_one()
            assert result == 'private_statement_marker'
        engine.dispose()
        diagnostics = [record for record in records
                       if record.name.startswith(('sqlalchemy.engine', 'sqlalchemy.pool'))]
        print(json.dumps({
            'engine_level': logging.getLogger('sqlalchemy.engine').level,
            'pool_level': logging.getLogger('sqlalchemy.pool').level,
            'diagnostics': [record.getMessage() for record in diagnostics],
        }))
    """)
    result = subprocess.run(
        [sys.executable, "-c", program], capture_output=True, text=True,
        env={**os.environ, "COWORK_HOME": str(tmp_path), "DATABASE_URI": "sqlite://", "LOG_LEVEL": "DEBUG"},
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout.splitlines()[-1])
    assert report == {"engine_level": logging.WARNING, "pool_level": logging.WARNING, "diagnostics": []}
    assert not any(secret in result.stdout + result.stderr for secret in SECRETS)
