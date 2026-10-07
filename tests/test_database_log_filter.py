from __future__ import annotations

import io
import json
import logging
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
import sqlalchemy as sa
from sqlalchemy.exc import IntegrityError, OperationalError

from cowork.common import logger as app_logger
from cowork.db import session as db_session
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
@pytest.mark.parametrize("chain_kind", ["cause", "context", "group"])
@pytest.mark.parametrize("error_kind", ["orm", "driver"])
def test_each_owned_handler_sanitizes_a_wrapped_real_database_error(
    handler_kind: str,
    chain_kind: str,
    error_kind: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    stream = io.StringIO()
    if handler_kind == "console":
        monkeypatch.setenv("RICH_LOGGING", "false")
        handler = app_logger.setup_console_handler()
        handler.setStream(stream)
        unused_handlers = []
    else:
        handlers = app_logger.setup_file_logging(str(tmp_path))
        handler = handlers[0 if handler_kind == "all_file" else 1]
        unused_handlers = [other for other in handlers if other is not handler]
    logger = logging.getLogger(f"cowork.test.database_filter.{handler_kind}")
    # No propagation: this must prove the selected production handler owns
    # the filter, rather than borrow the root console's protection.
    monkeypatch.setattr(logger, "handlers", [handler, caplog.handler])
    monkeypatch.setattr(logger, "propagate", False)
    monkeypatch.setattr(logger, "level", logging.ERROR)
    database_error = _sql_error()
    logged_error = database_error if error_kind == "orm" else database_error.orig
    try:
        try:
            if chain_kind == "cause":
                raise RuntimeError("wrapper " + " ".join(SECRETS)) from logged_error
            if chain_kind == "context":
                try:
                    raise logged_error
                except Exception:
                    raise RuntimeError("wrapper " + " ".join(SECRETS))
            raise ExceptionGroup("group " + " ".join(SECRETS), [logged_error])
        except Exception as wrapper:
            # The whole message must be replaced: callers such as scheduler
            # interpolate an error before adding exc_info.
            logger.exception(f"Raw database error: {wrapper}", extra={"request_id": "request-123"}, stack_info=True)
        records = [record for record in caplog.records if record.name == logger.name and record.levelno == logging.ERROR]
        assert len(records) == 1
        record = records[0]
        assert record.getMessage() == "Database operation failed: error_type=IntegrityError sqlstate=unknown"
        assert record.request_id == "request-123"
        assert record.exc_info is None
        assert record.exc_text is None
        assert record.stack_info is None
        assert not any(secret in repr(record.__dict__) for secret in SECRETS)
        handler.flush()
        emitted = stream.getvalue() if handler_kind == "console" else Path(handler.baseFilename).read_text()
        assert "error_type=IntegrityError" in emitted
        assert not any(secret in emitted for secret in SECRETS)
        assert database_error.statement and SQL_SECRET in database_error.statement
        assert BIND_SECRET in repr(database_error.params)
        assert DETAIL_SECRET in str(database_error.orig)
    finally:
        handler.close()
        for other in unused_handlers:
            other.close()


@pytest.mark.parametrize("sqlstate", ["23505", DETAIL_SECRET, None])
def test_database_exception_arguments_and_sqlstate_are_sanitized(
    sqlstate: str | None,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    class DriverFailure(Exception):
        pass

    DriverFailure.sqlstate = sqlstate
    error = OperationalError(" ".join(SECRETS), {"secret": BIND_SECRET}, DriverFailure(DETAIL_SECRET))
    handler = app_logger.setup_console_handler()
    stream = io.StringIO()
    handler.setStream(stream)
    logger = logging.getLogger("cowork.test.database_filter.argument")
    monkeypatch.setattr(logger, "handlers", [handler, caplog.handler])
    monkeypatch.setattr(logger, "propagate", False)
    monkeypatch.setattr(logger, "level", logging.ERROR)
    try:
        # Coding-service synchronization logs its exception as an argument
        # without exc_info, so the filter must inspect arguments too.
        logger.error("Database failed: %s", error)
        records = [record for record in caplog.records if record.name == logger.name and record.levelno == logging.ERROR]
        assert len(records) == 1
        expected_state = "23505" if sqlstate == "23505" else "unknown"
        assert records[0].getMessage() == f"Database operation failed: error_type=OperationalError sqlstate={expected_state}"
        assert not any(secret in repr(records[0].__dict__) for secret in SECRETS)
        assert not any(secret in stream.getvalue() for secret in SECRETS)
        assert error.orig.sqlstate == sqlstate
    finally:
        handler.close()


def test_other_exception_logs_keep_their_message_and_traceback(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    handler = app_logger.setup_console_handler()
    handler.setStream(io.StringIO())
    logger = logging.getLogger("cowork.test.database_filter.other")
    monkeypatch.setattr(logger, "handlers", [handler, caplog.handler])
    monkeypatch.setattr(logger, "propagate", False)
    monkeypatch.setattr(logger, "level", logging.ERROR)
    try:
        try:
            raise RuntimeError("ordinary application failure")
        except RuntimeError:
            logger.exception("Other failure")
        records = [record for record in caplog.records if record.name == logger.name and record.levelno == logging.ERROR]
        assert len(records) == 1
        assert records[0].getMessage() == "Other failure"
        assert records[0].exc_info is not None
        assert str(records[0].exc_info[1]) == "ordinary application failure"
    finally:
        handler.close()


def test_cli_uvicorn_handlers_sanitize_the_real_asgi_error_path(tmp_path: Path) -> None:
    # dictConfig changes and closes process-wide handlers. Run Uvicorn in a
    # subprocess so proving its real configuration never disables pytest's
    # capture handler for later tests. No socket or server process is started.
    program = textwrap.dedent("""
        import asyncio, h11, json, logging
        from unittest import mock
        from types import SimpleNamespace
        import uvicorn
        from uvicorn.protocols.http.h11_impl import RequestResponseCycle
        from cowork import cli
        from cowork.db import session

        records = []
        class Capture(logging.Handler):
            def emit(self, record):
                records.append(record)

        async def app(scope, receive, send):
            session.get_engine('postgresql+private_error_detail_marker://private_statement_marker:bound_credential_marker@localhost/test')

        async def run():
            with mock.patch.object(cli.uvicorn, 'run') as start:
                cli.main()
            config = uvicorn.Config(app, **{
                key: value for key, value in start.call_args.kwargs.items()
                if key not in ('host', 'port', 'reload', 'timeout_graceful_shutdown')
            })
            logger = logging.getLogger('uvicorn.error')
            logger.addHandler(Capture())
            conn = h11.Connection(h11.SERVER)
            conn.receive_data(b'GET /fail HTTP/1.1\\r\\nHost: localhost\\r\\n\\r\\n')
            conn.next_event()
            transport = mock.Mock()
            cycle = RequestResponseCycle(
                scope={'type': 'http', 'headers': [], 'http_version': '1.1', 'method': 'GET',
                       'path': '/fail', 'query_string': b'', 'client': ('127.0.0.1', 1234)},
                conn=conn, transport=transport,
                flow=SimpleNamespace(write_paused=False), logger=logger,
                access_logger=logging.getLogger('uvicorn.access'), access_log=False,
                default_headers=[], message_event=asyncio.Event(), on_response=lambda: None,
            )
            await cycle.run_asgi(app)
            response = b''.join(call.args[0] for call in transport.write.call_args_list)
            selected = [record for record in records if record.name == 'uvicorn.error' and record.levelno == logging.ERROR]
            print(json.dumps({
                'status_500': b'500' in response.split(b'\\r\\n', 1)[0],
                'records': [{'message': record.getMessage(), 'exc_info': record.exc_info is not None,
                             'exc_text': record.exc_text, 'args': record.args} for record in selected],
            }))
        asyncio.run(run())
    """)
    result = subprocess.run(
        [sys.executable, "-c", program], capture_output=True, text=True,
        env={**os.environ, "COWORK_HOME": str(tmp_path), "DATABASE_URI": "sqlite://"},
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout.splitlines()[-1])
    assert report["status_500"] is True
    assert report["records"] == [{
        "message": "Database operation failed: error_type=NoSuchModuleError sqlstate=unknown",
        "exc_info": False, "exc_text": None, "args": ["NoSuchModuleError", "unknown"],
    }]
    assert not any(secret in result.stdout + result.stderr for secret in SECRETS)


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
