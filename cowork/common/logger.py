import logging

# Add logging import for rotating file handler
import logging.handlers
import os
import re
import sys
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from sqlite3 import Error as SQLiteError
from typing import Any
from uuid import UUID

# cowork-server does not declare anthropic or openai itself: both arrive as
# anton-agent dependencies, so an anton-agent release that drops either one
# breaks this import.
from anthropic import AnthropicError
from anton.core.llm.provider import CURATED_PROVIDER_ERRORS
from openai import OpenAIError
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


_DATABASE_ERROR_TYPES = (SQLAlchemyError, SQLiteError, PsycopgError, Psycopg2Error)

# The SDK base classes, not only their APIError: an OpenAIError such as
# LengthFinishReasonError carries provider text without being an APIError.
_PROVIDER_ERROR_TYPES = (OpenAIError, AnthropicError, *CURATED_PROVIDER_ERRORS)


def _exception_chain(*, exc: BaseException) -> Iterator[BaseException]:
    """Yield ``exc``, then its cause, its context and its group members, depth first.

    Each exception is yielded once, so a cycle ends the walk. ``__context__`` is
    followed even when ``__suppress_context__`` is set: a wrapper raised
    ``from None`` can still repeat the original error's text in its own message.
    """
    pending = [exc]
    seen: set[int] = set()
    while pending:
        current = pending.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        yield current
        # Pushed in reverse, so the walk pops the cause first and the members last.
        if isinstance(current, BaseExceptionGroup):
            pending.extend(reversed(current.exceptions))
        if current.__context__ is not None:
            pending.append(current.__context__)
        if current.__cause__ is not None:
            pending.append(current.__cause__)


def find_database_error(*, exc: BaseException) -> SQLAlchemyError | SQLiteError | PsycopgError | Psycopg2Error | None:
    """The first database error in ``exc``'s chain, even when an application exception wraps it."""
    return next((error for error in _exception_chain(exc=exc) if isinstance(error, _DATABASE_ERROR_TYPES)), None)


def _exception_candidates(*, record: logging.LogRecord) -> list[BaseException]:
    """The exceptions a record carries: its exc_info, then its message, then its arguments."""
    candidates: list[BaseException] = []
    if record.exc_info is not None and record.exc_info[1] is not None:
        candidates.append(record.exc_info[1])
    if isinstance(record.msg, BaseException):
        candidates.append(record.msg)
    args = record.args.values() if isinstance(record.args, dict) else (record.args or ())
    candidates.extend(arg for arg in args if isinstance(arg, BaseException))
    return candidates


def _replace_record(*, record: logging.LogRecord, msg: str, args: tuple[object, ...]) -> None:
    """Swap a record's message for safe metadata and drop everything that repeats the error.

    A handler without these filters may have formatted the record already,
    caching its text in ``message`` and its traceback in ``exc_text``; a later
    formatter would print that cached traceback, so both are reset too.
    """
    record.msg = msg
    record.args = args
    record.message = record.getMessage()
    record.exc_info = None
    record.exc_text = None
    record.stack_info = None


class DatabaseErrorFilter(logging.Filter):
    """Replace database exception records before a handler formats them.

    hide_parameters on an engine protects bound values, but not SQL literals,
    driver error details or a caller that already interpolated the exception
    into its message. Replace the entire message and clear the traceback;
    the propagated exception and any response built from it stay unchanged.
    The replacement names the call site from the record's own code location.
    A caller's ids travel as record attributes (see CustomFormatter), which
    this filter leaves alone.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        for exc in _exception_candidates(record=record):
            error = find_database_error(exc=exc)
            if error is None:
                continue
            driver_error = error.orig if isinstance(error, StatementError) else error
            sqlstate = getattr(driver_error, "sqlstate", None) or getattr(driver_error, "pgcode", None)
            # Driver SQLSTATE is a five-character code, never its DETAIL or text.
            if not isinstance(sqlstate, str) or re.fullmatch(r"[A-Z0-9]{5}", sqlstate) is None:
                sqlstate = "unknown"
            _replace_record(
                record=record,
                msg="Database operation failed: error_type=%s sqlstate=%s site=%s.%s:%s",
                args=(type(error).__name__, sqlstate, record.module, record.funcName, record.lineno),
            )
            break
        return True


def exception_http_status(exc: BaseException) -> int | None:
    """Return an actual HTTP status, never a provider body field coerced to text."""
    for error in _exception_chain(exc=exc):
        status = getattr(error, "status_code", None)
        if type(status) is int and 100 <= status <= 599:
            return status
    return None


def _provider_status(*, error: BaseException) -> int | str:
    """The provider error's own HTTP status, or ``unknown`` when it has no real one."""
    status = getattr(error, "status_code", None)
    return status if type(status) is int and 100 <= status <= 599 else "unknown"


class ProviderErrorFilter(logging.Filter):
    """Keep SDK bodies and typed provider error text out of owned log handlers.

    A provider may echo prompts or credentials in its message, body or request.
    Replace the entire record, including already interpolated wrapper messages,
    while leaving the original exception and client response unchanged.

    The replacement names the record's own exception class, the first provider
    error in its chain and that provider error's own HTTP status, never a
    wrapper's. Mirrors anton.core.llm.provider.ProviderErrorFilter; replace it
    with an import once that release is pinned.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        for exc in _exception_candidates(record=record):
            provider_error = next(
                (error for error in _exception_chain(exc=exc) if isinstance(error, _PROVIDER_ERROR_TYPES)), None,
            )
            if provider_error is None:
                continue
            _replace_record(
                record=record,
                msg="Provider operation failed: error_type=%s provider_error=%s status=%s",
                args=(type(exc).__name__, type(provider_error).__name__, _provider_status(error=provider_error)),
            )
            break
        return True


def _add_privacy_filters(*, target: logging.Filterer) -> None:
    """Attach the database and provider filters to a handler or logger, once each."""
    for filter_type in (DatabaseErrorFilter, ProviderErrorFilter):
        if not any(isinstance(existing, filter_type) for existing in target.filters):
            target.addFilter(filter_type())


class EventLoopClosedFilter(logging.Filter):
    """Filter out 'Event loop is closed' errors that occur during cleanup"""

    def filter(self, record) -> bool:
        # Filter out the specific RuntimeError about event loop being closed
        return not (
            record.name == "asyncio"
            and record.levelno == logging.ERROR
            and "Event loop is closed" in record.getMessage()
        )


@dataclass(frozen=True)
class _ContextField:
    """One id a call site may attach via ``extra=``, and how it renders."""

    attribute: str
    label: str
    render: Callable[[Any], str] = str


def _quoted_slugs(slugs: Sequence[str]) -> str:
    return ", ".join(repr(slug) for slug in slugs)


# Rendered in this order into ``%(request_context)s``. The database and
# provider filters replace a record's message but never these attributes, so
# a sanitized line still names the request, project, schedule, conversation
# and artifacts it came from. Call sites build them with log_context.
_REQUEST_CONTEXT_FIELDS = (
    _ContextField("request_id", "Req"),
    _ContextField("project_id", "Project"),
    _ContextField("schedule_id", "Schedule"),
    _ContextField("conversation_id", "Conversation"),
    # Folder names the agent chose: quoted, as the messages quoted them with
    # %r, so a newline or bracket in one cannot pass for another line or field.
    _ContextField("artifact_slug", "Artifact", repr),
    _ContextField("artifact_slugs", "Artifacts", _quoted_slugs),
)


def log_context(
    *,
    request_id: str | None = None,
    project_id: UUID | str | None = None,
    schedule_id: UUID | str | None = None,
    conversation_id: UUID | str | None = None,
    artifact_slug: str | None = None,
    artifact_slugs: Sequence[str] | None = None,
) -> dict[str, str | tuple[str, ...]]:
    """The ``extra=`` mapping for the ids that locate a failure.

    Each keyword is one of the record attributes in _REQUEST_CONTEXT_FIELDS,
    so a misspelled name fails at the call instead of rendering nothing. Ids
    become strings. ``artifact_slug`` names one folder and ``artifact_slugs``
    several, as a tuple. A None value is left out.
    """
    ids = {
        "request_id": request_id,
        "project_id": project_id,
        "schedule_id": schedule_id,
        "conversation_id": conversation_id,
        "artifact_slug": artifact_slug,
    }
    context: dict[str, str | tuple[str, ...]] = {name: str(value) for name, value in ids.items() if value is not None}
    if artifact_slugs is not None:
        context["artifact_slugs"] = tuple(artifact_slugs)
    return context


class CustomFormatter(logging.Formatter):
    """Formatter that renders the optional ``user_id`` and the
    _REQUEST_CONTEXT_FIELDS ids a call site may attach via ``extra=``.

    Each renders as an empty string when absent, which is what lets a format
    string reference ``%(request_context)s`` unconditionally — the vast
    majority of records carry none. An explicitly ``None`` value counts as
    absent.
    """

    def format(self, record):
        # None is absent, not a value: a producer with no id to offer still
        # passes the key, and ``[Req:None]`` is worse than no prefix at all.
        user_id = getattr(record, "user_id", None)
        record.user_context = f"[User:{user_id}]" if user_id is not None else ""

        record.request_context = "".join(
            f"[{field.label}:{field.render(value)}]"
            for field in _REQUEST_CONTEXT_FIELDS
            if (value := getattr(record, field.attribute, None)) is not None
        )

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
    _add_privacy_filters(target=all_logs_handler)
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
    _add_privacy_filters(target=error_handler)
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

    _add_privacy_filters(target=handler)
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

    # Uvicorn's handlers come from whichever log config the launcher chose: the
    # cowork-server CLI, `python -m uvicorn` in this repo's image, or `uvicorn
    # spa_wrapper:app` in the web image. Uvicorn's default config gives the
    # `uvicorn` logger its own unfiltered handler and no propagation to root.
    # A filter on each logger Uvicorn writes to runs before any of those
    # handlers, and a later dictConfig keeps it. The levels stay as the
    # launcher set them, so lifecycle lines such as "Finished server process"
    # still print. A logger filter misses a record that a child logger
    # propagates, so when the launcher configured Uvicorn first, each handler
    # it already installed gets the filters too.
    for logger_name in ("uvicorn", "uvicorn.error", "uvicorn.access", "uvicorn.asgi"):
        uvicorn_logger = logging.getLogger(logger_name)
        _add_privacy_filters(target=uvicorn_logger)
        for handler in uvicorn_logger.handlers:
            _add_privacy_filters(target=handler)

    # Suppress verbose logging from third-party libraries
    third_party_loggers = [
        # The whole transport namespaces: at DEBUG, httpcore and httpcore2
        # (httpx2's transport, which both provider SDKs use) log every
        # response's headers, set-cookie included.
        "httpcore",
        "httpcore2",
        "openai._base_client",
        "anthropic._base_client",
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
