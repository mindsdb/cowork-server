"""PostgreSQL session timeouts and the slow-statement line on the app's engine.

Both timeouts are off unless their settings are set, and SQLite engines never
carry either, so the desktop is unchanged. The slow-statement line names a
statement by its verb, first table and a fingerprint of its SQL text, never by
the SQL or its parameters.
"""
from __future__ import annotations

import logging
import os

import pytest
import sqlalchemy as sa

from cowork.common.settings.app_settings import DatabaseSettings
from cowork.db import session as session_mod
from cowork.db.session import (
    log_slow_statements,
    postgres_session_options,
    statement_fingerprint,
    statement_operation,
)
from tests.integration.prereq import missing_prerequisite

STATEMENT_LOGGER = "cowork.db.statements"


@pytest.fixture
def slow_lines(monkeypatch, caplog):
    # The migration tests' alembic env calls fileConfig, which disables every
    # logger that existed before it for the rest of the session.
    monkeypatch.setattr(logging.getLogger(STATEMENT_LOGGER), "disabled", False)
    caplog.set_level(logging.WARNING, logger=STATEMENT_LOGGER)

    def lines() -> list[logging.LogRecord]:
        return [
            record for record in caplog.records
            if record.name == STATEMENT_LOGGER and getattr(record, "event", None) == "db_statement_slow"
        ]

    return lines


def test_both_timeouts_are_off_by_default():
    assert postgres_session_options(DatabaseSettings()) == ""


@pytest.mark.parametrize(
    ("settings", "expected"),
    [
        ({"statement_timeout_ms": 30000}, "-c statement_timeout=30000"),
        ({"idle_in_transaction_timeout_ms": 60000}, "-c idle_in_transaction_session_timeout=60000"),
        (
            {"statement_timeout_ms": 30000, "idle_in_transaction_timeout_ms": 60000},
            "-c statement_timeout=30000 -c idle_in_transaction_session_timeout=60000",
        ),
    ],
)
def test_each_timeout_that_is_set_becomes_a_session_option(settings, expected):
    assert postgres_session_options(DatabaseSettings(**settings)) == expected


@pytest.mark.parametrize(
    ("statement", "operation"),
    [
        ("SELECT schedule_runs.id FROM schedule_runs WHERE schedule_runs.id = %(id)s", "SELECT schedule_runs"),
        ('UPDATE "files" SET purpose=%(purpose)s WHERE files.id = %(id)s', "UPDATE files"),
        ("INSERT INTO messages (id, body) VALUES (%(id)s, %(body)s)", "INSERT messages"),
        ("BEGIN", "BEGIN"),
    ],
)
def test_a_statement_is_named_by_its_verb_and_first_table(statement, operation):
    assert statement_operation(statement) == operation


def test_the_fingerprint_names_one_sql_text_whatever_its_whitespace():
    assert statement_fingerprint("SELECT 1\n   FROM t") == statement_fingerprint("SELECT 1 FROM t")
    assert statement_fingerprint("SELECT 1 FROM t") != statement_fingerprint("SELECT 2 FROM t")


def test_a_statement_at_the_threshold_is_logged_without_its_sql(slow_lines):
    engine = sa.create_engine("sqlite://")
    log_slow_statements(engine, threshold_ms=0)

    with engine.connect() as connection:
        connection.execute(sa.text("SELECT 'a secret value' AS v"))

    (line,) = slow_lines()
    assert line.levelno == logging.WARNING
    assert line.operation == "SELECT"
    assert line.statement_fingerprint == statement_fingerprint("SELECT 'a secret value' AS v")
    assert isinstance(line.duration_ms, int)
    assert "secret" not in line.getMessage()
    assert "secret" not in str(vars(line))


def test_a_statement_under_the_threshold_is_not_logged(slow_lines):
    engine = sa.create_engine("sqlite://")
    log_slow_statements(engine, threshold_ms=60_000)

    with engine.connect() as connection:
        connection.execute(sa.text("SELECT 1"))

    assert slow_lines() == []


def test_a_sqlite_engine_takes_neither_timeout_nor_the_slow_line(monkeypatch, slow_lines):
    """The desktop runs SQLite, so the app's own engine there is as before."""
    database = session_mod.settings.database
    monkeypatch.setattr(database, "statement_timeout_ms", 50)
    monkeypatch.setattr(database, "slow_statement_log_ms", 1)
    monkeypatch.setattr(session_mod, "create_engine", _recording_create_engine(captured := []))

    engine = session_mod._create_engine("sqlite://")
    with engine.connect() as connection:
        connection.execute(sa.text("SELECT 1"))

    assert captured == [{"check_same_thread": False}]
    assert slow_lines() == []


def _recording_create_engine(captured: list):
    def create(uri, **kwargs):
        captured.append(kwargs.get("connect_args"))
        return sa.create_engine(uri, **kwargs)

    return create


@pytest.mark.postgres_service
def test_the_app_engine_on_postgres_times_out_and_logs_slow_statements(monkeypatch, slow_lines):
    uri = os.environ.get("COWORK_TEST_POSTGRES_URI")
    if not uri:
        missing_prerequisite(
            "set COWORK_TEST_POSTGRES_URI to a disposable database", required_by="COWORK_REQUIRE_SERVICES",
        )
    database = session_mod.settings.database
    monkeypatch.setattr(database, "statement_timeout_ms", 500)
    monkeypatch.setattr(database, "idle_in_transaction_timeout_ms", 700)
    monkeypatch.setattr(database, "slow_statement_log_ms", 100)
    engine = session_mod._create_engine(uri)
    try:
        with engine.connect() as connection:
            assert connection.execute(sa.text("SHOW statement_timeout")).scalar() == "500ms"
            assert connection.execute(sa.text("SHOW idle_in_transaction_session_timeout")).scalar() == "700ms"
            connection.execute(sa.text("SELECT pg_sleep(0.2)"))
            with pytest.raises(sa.exc.OperationalError) as cut:
                connection.execute(sa.text("SELECT pg_sleep(2)"))
        assert cut.value.orig.sqlstate == "57014"
        (line,) = slow_lines()
        assert line.operation == "SELECT"
        assert line.duration_ms >= 200
    finally:
        engine.dispose()
