"""Database connection and session management using SQLAlchemy"""

import hashlib
import re
import time

from sqlalchemy import create_engine, event
from sqlalchemy.exc import TimeoutError as PoolTimeoutError
from sqlalchemy.orm import sessionmaker
from sqlmodel import Session as SQLModelSession
from starlette.exceptions import HTTPException

from cowork.common.logger import get_logger, setup_logging
from cowork.common.settings.app_settings import DatabaseSettings, get_app_settings

logger = setup_logging()
statement_logger = get_logger("cowork.db.statements")
settings = get_app_settings()

# Global cache for engines and session factories
_engines = {}
_session_factories = {}



def _is_expected_refusal(exc: BaseException) -> bool:
    """True for an answer the request was meant to get: a 4xx HTTPException, or
    the 503 for a pool that freed no connection in time (cowork.server answers
    it). Another error's status_code, such as a provider SDK's, is not one."""
    if isinstance(exc, PoolTimeoutError):
        return True
    if not isinstance(exc, HTTPException):
        return False
    status = exc.status_code
    return type(status) is int and 400 <= status < 500

def postgres_session_options(database: DatabaseSettings) -> str:
    """The libpq ``options`` that set each app connection's session timeouts,
    or "" when both are off."""
    options = []
    if database.statement_timeout_ms:
        options.append(f"-c statement_timeout={database.statement_timeout_ms}")
    if database.idle_in_transaction_timeout_ms:
        options.append(f"-c idle_in_transaction_session_timeout={database.idle_in_transaction_timeout_ms}")
    return " ".join(options)


_OPERATION_TABLE = re.compile(r"\b(?:FROM|INTO|UPDATE|JOIN)\s+\"?([A-Za-z_][A-Za-z0-9_]*)", re.IGNORECASE)


def statement_operation(statement: str) -> str:
    """The statement's verb and first table, such as ``SELECT schedule_runs``.
    Built from the SQL text alone, so no bound value reaches the log."""
    words = statement.split(None, 1)
    verb = words[0].upper() if words else "UNKNOWN"
    table = _OPERATION_TABLE.search(statement)
    return f"{verb} {table.group(1)}" if table else verb


def statement_fingerprint(statement: str) -> str:
    """A short, stable name for one statement's SQL text, so its slow runs
    can be counted together."""
    return hashlib.sha256(" ".join(statement.split()).encode()).hexdigest()[:12]


def log_slow_statements(engine, *, threshold_ms: int) -> None:
    """Log each statement on ``engine`` that takes at least ``threshold_ms``.

    The line carries the statement's verb and first table and a fingerprint
    of its SQL text, never the SQL or its parameters.
    """

    @event.listens_for(engine, "before_cursor_execute")
    def _started(conn, cursor, statement, parameters, context, executemany):
        conn.info["statement_started"] = time.perf_counter()

    @event.listens_for(engine, "after_cursor_execute")
    def _finished(conn, cursor, statement, parameters, context, executemany):
        started = conn.info.pop("statement_started", None)
        if started is None:
            return
        duration_ms = (time.perf_counter() - started) * 1000
        if duration_ms >= threshold_ms:
            statement_logger.warning(
                "Database statement was slow",
                extra={
                    "event": "db_statement_slow",
                    "duration_ms": round(duration_ms),
                    "operation": statement_operation(statement),
                    "statement_fingerprint": statement_fingerprint(statement),
                },
            )


def _create_engine(db_uri: str):
    is_sqlite = db_uri.startswith("sqlite")
    try:
        if is_sqlite:
            engine = create_engine(
                db_uri,
                connect_args={"check_same_thread": False},
                hide_parameters=True,
            )
        else:
            options = postgres_session_options(settings.database)
            engine = create_engine(
                db_uri,
                connect_args={"options": options} if options else {},
                hide_parameters=True,
                pool_size=settings.database.pool_size,
                max_overflow=settings.database.max_overflow,
                pool_timeout=settings.database.pool_timeout,
                pool_recycle=settings.database.pool_recycle,
                pool_pre_ping=settings.database.pool_pre_ping,
            )
            if settings.database.slow_statement_log_ms:
                log_slow_statements(engine, threshold_ms=settings.database.slow_statement_log_ms)
        return engine
    except Exception as e:
        error_msg = str(e).lower()
        logger.error("Failed to create engine: error_type=%s", type(e).__name__)
        raise RuntimeError(f"Engine creation failed: {error_msg}") from e


def get_engine(db_uri: str):
    """Get or create a database engine for the specified connection URI.

    Args:
        db_uri: Database connection URI

    Returns:
        SQLAlchemy engine instance
    """
    if db_uri not in _engines:
        _engines[db_uri] = _create_engine(db_uri=db_uri)
    return _engines[db_uri]


def get_session_factory(engine):
    """Get or create a session factory for the specified engine.

    Args:
        engine: SQLAlchemy engine instance

    Returns:
        SQLAlchemy session maker instance
    """
    engine_id = id(engine)
    if engine_id not in _session_factories:
        _session_factories[engine_id] = sessionmaker(
            class_=SQLModelSession,  # <- SQLModel-compatible session
            autocommit=False,
            autoflush=False,
            bind=engine,
            expire_on_commit=False,
        )
    return _session_factories[engine_id]


def get_session():
    """
    FastAPI dependency that provides a database session with automatic cleanup.

    This is a generator-based dependency that ensures sessions are always closed
    after the request completes, preventing database connection leaks.

    It takes no parameters: FastAPI reads a dependency's parameters from the
    request, so the database URI comes from settings only.

    Usage:
        @router.get("/example")
        async def endpoint(session: Session = Depends(get_session)):
            # Session will be automatically closed after this function completes
            pass

    Yields:
        SQLModelSession: Database session that will be automatically closed
    """

    engine = get_engine(db_uri=settings.database.uri)
    session_factory = get_session_factory(engine)

    db = session_factory()
    try:
        logger.debug("🔗 Created database session")
        yield db
    except Exception as e:
        # A 4xx HTTPException is the endpoint's intended answer (a 409 for a
        # steer during an approval, a 404 for a missing task), and so is the
        # 503 for a full pool. Each still rolls the transaction back, but none
        # is an error worth an error-level log. A plain pool timeout carries no
        # status_code; cowork.server answers it with a 503.
        # Exception text and tracebacks can expose SQL, credentials or database
        # error details, even when create_engine hides bound parameters.
        if _is_expected_refusal(e):
            logger.debug(
                "Session rolled back after an expected refusal: error_type=%s status=%s",
                type(e).__name__,
                getattr(e, "status_code", 503),
            )
        elif isinstance(e, HTTPException):
            # Starlette answers an HTTPException (FastAPI's subclasses it)
            # itself and logs nothing, so this line is the only record of what
            # caused a 5xx. The cause's traceback goes through the handler
            # filters, which replace a database or provider chain; the
            # exception's own detail is the client's answer and stays out.
            cause = e.__cause__ or (None if e.__suppress_context__ else e.__context__)
            logger.error(
                "Session error: error_type=%s status=%d cause=%s",
                type(e).__name__,
                e.status_code,
                type(cause).__name__ if cause is not None else "none",
                exc_info=cause,
            )
        else:
            # Any other exception goes on to Uvicorn, which logs its traceback
            # through the same filters, or to an app exception handler that
            # answers it. That includes an error that carries a status_code
            # without being an HTTPException, such as a provider SDK's.
            logger.error("Session error: error_type=%s", type(e).__name__)
        db.rollback()
        raise
    finally:
        logger.debug("🔒 Closing database session")
        db.close()


def get_open_session(db_uri: str = settings.database.uri):
    """
    Get an open SQLModel session.

    Args:
        db_uri: Database connection URI (defaults to settings value)

    Returns:
        SQLModelSession: Open SQLModel session
    """
    engine = get_engine(db_uri=db_uri)
    session_factory = get_session_factory(engine)
    return session_factory()
