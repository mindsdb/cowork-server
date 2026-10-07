import logging

# Add logging import for rotating file handler
import logging.handlers
import os
import re
import sys
from collections.abc import Iterator
from copy import deepcopy
from pathlib import Path
from sqlite3 import Error as SQLiteError

from anthropic import APIError as AnthropicAPIError
from anton.core.llm.provider import CURATED_PROVIDER_ERRORS
from openai import APIError as OpenAIAPIError
from psycopg import Error as PsycopgError
from psycopg2 import Error as Psycopg2Error
from sqlalchemy.exc import SQLAlchemyError, StatementError

try:
    import colorlog

    HAS_COLORLOG = True
except ImportError:
    HAS_COLORLOG = False

try:
    from rich.console import Console
    from rich.logging import RichHandler

    HAS_RICH = True
except ImportError:
    HAS_RICH = False


def _database_error(exc: BaseException) -> SQLAlchemyError | SQLiteError | PsycopgError | Psycopg2Error | None:
    """Find a database error even when an application exception wraps it."""
    pending = [exc]
    seen: set[int] = set()
    while pending:
        current = pending.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        if isinstance(current, (SQLAlchemyError, SQLiteError, PsycopgError, Psycopg2Error)):
            return current
        if current.__cause__ is not None:
            pending.append(current.__cause__)
        if current.__context__ is not None:
            pending.append(current.__context__)
        if isinstance(current, BaseExceptionGroup):
            pending.extend(current.exceptions)
    return None


class DatabaseErrorFilter(logging.Filter):
    """Replace database exception records before a handler formats them.

    hide_parameters on an engine protects bound values, but not SQL literals,
    driver error details or a caller that already interpolated the exception
    into its message. Replace the entire message and clear the traceback;
    the propagated exception and any response built from it stay unchanged.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        candidates: list[BaseException] = []
        if record.exc_info is not None and record.exc_info[1] is not None:
            candidates.append(record.exc_info[1])
        if isinstance(record.msg, BaseException):
            candidates.append(record.msg)
        args = record.args.values() if isinstance(record.args, dict) else (record.args or ())
        candidates.extend(arg for arg in args if isinstance(arg, BaseException))
        for exc in candidates:
            error = _database_error(exc)
            if error is None:
                continue
            driver_error = error.orig if isinstance(error, StatementError) else error
            sqlstate = getattr(driver_error, "sqlstate", None) or getattr(driver_error, "pgcode", None)
            # Driver SQLSTATE is a five-character code, never its DETAIL or text.
            if not isinstance(sqlstate, str) or re.fullmatch(r"[A-Z0-9]{5}", sqlstate) is None:
                sqlstate = "unknown"
            record.msg = "Database operation failed: error_type=%s sqlstate=%s"
            record.args = (type(error).__name__, sqlstate)
            record.message = record.getMessage()
            record.exc_info = None
            record.exc_text = None
            record.stack_info = None
            break
        return True


def _exception_chain(exc: BaseException) -> Iterator[BaseException]:
    """Walk causes, implicit contexts and groups without following cycles."""
    pending = [exc]
    seen: set[int] = set()
    while pending:
        current = pending.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        yield current
        if current.__context__ is not None:
            pending.append(current.__context__)
        if current.__cause__ is not None:
            pending.append(current.__cause__)
        if isinstance(current, BaseExceptionGroup):
            pending.extend(current.exceptions)


def exception_http_status(exc: BaseException) -> int | None:
    """Return an actual HTTP status, never a provider body field coerced to text."""
    for error in _exception_chain(exc):
        status = getattr(error, "status_code", None)
        if type(status) is int and 100 <= status <= 599:
            return status
    return None


_PROVIDER_ERROR_TYPES = (OpenAIAPIError, AnthropicAPIError, *CURATED_PROVIDER_ERRORS)


class ProviderErrorFilter(logging.Filter):
    """Keep SDK bodies and typed provider error text out of owned log handlers.

    A provider may echo prompts or credentials in its message, body or request.
    Replace the entire record, including already interpolated wrapper messages,
    while leaving the original exception and client response unchanged.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        candidates: list[BaseException] = []
        if record.exc_info is not None and record.exc_info[1] is not None:
            candidates.append(record.exc_info[1])
        if isinstance(record.msg, BaseException):
            candidates.append(record.msg)
        args = record.args.values() if isinstance(record.args, dict) else (record.args or ())
        candidates.extend(arg for arg in args if isinstance(arg, BaseException))
        for exc in candidates:
            error = next((
                error for error in _exception_chain(exc)
                if isinstance(error, _PROVIDER_ERROR_TYPES)
            ), None)
            if error is None:
                continue
            status = exception_http_status(exc)
            record.msg = "Provider operation failed: error_type=%s status=%s"
            record.args = (type(error).__name__, status if status is not None else "unknown")
            record.message = record.getMessage()
            record.exc_info = None
            record.exc_text = None
            record.stack_info = None
            break
        return True


def uvicorn_logging_config() -> dict[str, object]:
    """Keep Uvicorn's handlers from bypassing the exception privacy filters."""
    from uvicorn.config import LOGGING_CONFIG

    config = deepcopy(LOGGING_CONFIG)
    config["filters"] = {
        "database_errors": {"()": DatabaseErrorFilter},
        "provider_errors": {"()": ProviderErrorFilter},
    }
    for handler in config["handlers"].values():
        handler["filters"] = ["database_errors", "provider_errors"]
    return config


class EventLoopClosedFilter(logging.Filter):
    """Filter out 'Event loop is closed' errors that occur during cleanup"""

    def filter(self, record) -> bool:
        # Filter out the specific RuntimeError about event loop being closed
        return not (
            record.name == "asyncio"
            and record.levelno == logging.ERROR
            and "Event loop is closed" in record.getMessage()
        )


class CustomFormatter(logging.Formatter):
    """Formatter that renders the optional ``user_id`` / ``request_id`` a call
    site may attach via ``extra=``.

    Both render as an empty string when absent, which is what lets a format
    string reference ``%(request_context)s`` unconditionally — the vast
    majority of records carry neither. An explicitly ``None`` value counts as
    absent.
    """

    def format(self, record):
        # None is absent, not a value: a producer with no id to offer still
        # passes the key, and ``[Req:None]`` is worse than no prefix at all.
        user_id = getattr(record, "user_id", None)
        record.user_context = f"[User:{user_id}]" if user_id is not None else ""

        request_id = getattr(record, "request_id", None)
        record.request_context = f"[Req:{request_id}]" if request_id is not None else ""

        return super().format(record)


def get_colored_formatter():
    """The console handler's formatter, colored when colorlog is installed.

    Both branches derive from CustomFormatter because both reference
    ``%(request_context)s``, and only a CustomFormatter defines it — a plain
    Formatter raises per record, which logging swallows into a dropped line.
    The console is the stream the desktop captures into the log tail it offers
    to copy, so an id rendered only on a file handler reaches nobody: file
    logging is off unless ENABLE_FILE_LOGGING is exported, which nothing in
    the stack does.
    """
    if not HAS_COLORLOG:
        return CustomFormatter(
            "%(asctime)s [%(levelname)0s] %(name)s%(request_context)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

    class _ColoredCustomFormatter(CustomFormatter, colorlog.ColoredFormatter):
        """Inherits the context injection so the two branches cannot drift."""

    return _ColoredCustomFormatter(
        "%(log_color)s%(asctime)s [%(levelname)0s] %(name)s%(request_context)s: "
        "%(message)s%(reset)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        log_colors={
            "DEBUG": "cyan",
            "INFO": "green",
            "WARNING": "yellow",
            "ERROR": "red",
            "CRITICAL": "red,bg_white",
        },
        secondary_log_colors={
            "message": {
                "DEBUG": "white",
                "INFO": "white",
                "WARNING": "yellow",
                "ERROR": "red",
                "CRITICAL": "red",
            }
        },
    )


def setup_file_logging(log_dir: str = "logs", max_bytes: int = 10485760, backup_count: int = 5):
    """Setup file logging with rotation"""
    log_path = Path(log_dir)
    log_path.mkdir(exist_ok=True)

    # Create handlers for different log levels
    handlers = []

    # All logs file
    all_logs_handler = logging.handlers.RotatingFileHandler(
        log_path / "minds.log", maxBytes=max_bytes, backupCount=backup_count
    )
    # CustomFormatter, not logging.Formatter: it is what defines
    # %(request_context)s, and it renders an empty string for the records that
    # carry no request_id — which is most of them. A plain Formatter would
    # raise ValueError on every such line.
    all_logs_handler.setFormatter(
        CustomFormatter(
            "%(asctime)s [%(levelname)8s] %(name)s%(request_context)s "
            "[%(filename)s:%(lineno)d] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    all_logs_handler.addFilter(DatabaseErrorFilter())
    all_logs_handler.addFilter(ProviderErrorFilter())
    handlers.append(all_logs_handler)

    # Error logs file
    error_handler = logging.handlers.RotatingFileHandler(
        log_path / "errors.log", maxBytes=max_bytes, backupCount=backup_count
    )
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(
        CustomFormatter(
            "%(asctime)s [%(levelname)8s] %(name)s%(request_context)s "
            "[%(filename)s:%(lineno)d] %(message)s\n%(stack_info)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    error_handler.addFilter(DatabaseErrorFilter())
    error_handler.addFilter(ProviderErrorFilter())
    handlers.append(error_handler)

    return handlers


def setup_console_handler():
    """Setup console handler with colors and rich formatting if available"""
    if HAS_RICH and os.getenv("RICH_LOGGING", "false").lower() == "true":
        console = Console(stderr=True)
        handler = RichHandler(
            console=console,
            show_path=False,
            show_time=True,
            rich_tracebacks=True,
            tracebacks_show_locals=True,
        )
        # CustomFormatter for the same reason the other branch uses one: this
        # is a console stream too, and the desktop tails it.
        handler.setFormatter(CustomFormatter(
            "[%(name)s]%(request_context)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S",
        ))
    else:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(get_colored_formatter())

    handler.addFilter(DatabaseErrorFilter())
    handler.addFilter(ProviderErrorFilter())
    return handler


def _resolved_log_level_name() -> str:
    """The configured log level, read through AppSettings like every other
    setting, falling back to `os.getenv` only if settings cannot be built.

    Reading it from the environment alone was the bug: the desktop keeps its
    config in `<COWORK_HOME>/.env`, which no environment variable carries, so a
    customer who set LOG_LEVEL there still got the WARNING default and a log
    holding nothing but uvicorn access lines. AppSettings already reads that
    file through its env-file chain, and pydantic-settings still ranks real
    environment variables above it, so a deployment's LOG_LEVEL keeps winning.

    Only CONSTRUCTION is guarded, and deliberately broadly: `cowork_home()` runs
    inside the env-file chain, so the failure modes are not only pydantic's, and
    this runs at import, before anything could report the error. Reading the
    field is outside the guard, so a missing or renamed `log_level` raises here
    instead of silently degrading every deployment to WARNING.
    """
    from cowork.common.settings.app_settings import get_app_settings

    try:
        settings = get_app_settings()
    except Exception:
        # `get_app_settings` is lru_cached and does not cache exceptions, so the
        # app's own call still raises later with the real message.
        return os.getenv("LOG_LEVEL", "WARNING")
    return settings.log_level


def setup_logging():
    """Setup comprehensive logging configuration"""
    log_level_str = _resolved_log_level_name()
    enable_file_logging = os.getenv("ENABLE_FILE_LOGGING", "false").lower() == "true"
    log_dir = os.getenv("LOG_DIR", "logs")

    # Map string log level to logging constants
    log_level_map = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL,
    }

    # Default to INFO if an invalid level is provided
    log_level = log_level_map.get(log_level_str.upper(), logging.WARNING)

    # Clear any existing handlers
    root_logger = logging.getLogger()
    root_logger.handlers.clear()

    # Setup handlers
    handlers = []

    # Console handler
    console_handler = setup_console_handler()
    console_handler.setLevel(log_level)
    handlers.append(console_handler)

    # File handlers (if enabled)
    if enable_file_logging:
        file_handlers = setup_file_logging(log_dir)
        for handler in file_handlers:
            handler.setLevel(log_level)
            handlers.append(handler)

    # Configure root logger
    logging.basicConfig(level=log_level, handlers=handlers, force=True)

    # Suppress verbose logging from third-party libraries
    third_party_loggers = [
        "httpcore.http11",
        "openai._base_client",
        "anthropic._base_client",
        "httpcore.connection",
        "httpx",
        "httpx2",
        "urllib3",
        "faiss",
        "asyncio",
        "requests",
        "boto3",
        "botocore",
        "s3transfer",
        "transformers",
        "torch",
        "tensorflow",
        "urllib3.connectionpool",
    ]

    for logger_name in third_party_loggers:
        logging.getLogger(logger_name).setLevel(logging.ERROR)

    # SQLAlchemy diagnostic logs can contain SQL literals, result rows and
    # connection representations. Keep them off even when another logger
    # config enables diagnostics on the sqlalchemy parent namespace.
    for logger_name in ("sqlalchemy.engine", "sqlalchemy.pool"):
        logging.getLogger(logger_name).setLevel(logging.WARNING)

    # Add filter to suppress "Event loop is closed" errors during cleanup
    event_loop_filter = EventLoopClosedFilter()
    logging.getLogger("asyncio").addFilter(event_loop_filter)

    # Create application logger
    logger = logging.getLogger(__name__)

    return logger


def get_logger(name: str) -> logging.Logger:
    """Get a logger with the given name"""
    return logging.getLogger(name)
