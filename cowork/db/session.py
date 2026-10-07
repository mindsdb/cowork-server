"""Database connection and session management using SQLAlchemy"""

from sqlalchemy import create_engine
from sqlalchemy.exc import TimeoutError as PoolTimeoutError
from sqlalchemy.orm import sessionmaker
from sqlmodel import Session as SQLModelSession

from cowork.common.logger import setup_logging
from cowork.common.settings.app_settings import get_app_settings

logger = setup_logging()
settings = get_app_settings()

# Global cache for engines and session factories
_engines = {}
_session_factories = {}



def _is_expected_refusal(exc: BaseException) -> bool:
    """True for an answer the request was meant to get: a 4xx, or the 503 for a
    pool that freed no connection in time (cowork.server answers it)."""
    if isinstance(exc, PoolTimeoutError):
        return True
    status = getattr(exc, "status_code", None)
    return isinstance(status, int) and 400 <= status < 500

def _create_engine(db_uri: str):
    is_sqlite = db_uri.startswith("sqlite")
    try:
        if is_sqlite:
            engine = create_engine(
                db_uri,
                connect_args={"check_same_thread": False},
            )
        else:
            engine = create_engine(
                db_uri,
                pool_size=settings.database.pool_size,
                max_overflow=settings.database.max_overflow,
                pool_timeout=settings.database.pool_timeout,
                pool_recycle=settings.database.pool_recycle,
                pool_pre_ping=settings.database.pool_pre_ping,
            )
        return engine
    except Exception as e:
        error_msg = str(e).lower()
        logger.error(f"Failed to create engine: {error_msg}")
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
        # is an error worth a traceback.
        if _is_expected_refusal(e):
            logger.debug(f"Session rolled back after an expected refusal: {e}")
        else:
            logger.exception(f"❌ Session error: {str(e)}")
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
