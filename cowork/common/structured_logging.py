"""Stdlib logging with a shared JSON envelope and explicit diagnostic fields."""

import logging
import os
import re
import sys
import traceback
from datetime import UTC, datetime
from http import HTTPMethod
from math import isfinite
from pathlib import Path
from uuid import UUID

import structlog

EVENT_FIELDS = frozenset(
    [
        "event",
        "reason",
        "outcome",
        "operation",
        "provider",
        "error_type",
        "error_class",
        "status_code",
        "request_id",
        "request_id_source",
        "turn_request_id",
        "artifact_slug",
        "artifact_slugs",
        "sqlstate",
        "provider_error",
        "conversation_id",
        "project_id",
        "schedule_id",
        "turn_id",
        "job_id",
        "trace_id",
        "method",
        "route",
        "duration_ms",
        "user_id",
        "organization_id",
        "workspace_id",
        "attempt",
        "attempts",
        "count",
        "slot",
        "model_alias",
        "managed_connection_file",
        "retryable",
    ]
)
HEALTH_PATHS = frozenset(
    ["/health", "/api/v1/health", "/api/v1/health/", "/api/v1/health/live"]
)


def safe_exception_traceback(exc_info: tuple) -> str | None:
    """Keep types and stack locations, excluding values, source, notes and locals."""
    if not exc_info or exc_info[1] is None:
        return None
    lines, seen = [], set()

    def visit(error, tb):
        if error is None or id(error) in seen:
            return
        seen.add(id(error))
        for frame in traceback.extract_tb(tb):
            lines.append(
                f'  File "{Path(frame.filename).name}", line {frame.lineno}, in {frame.name}'
            )
        lines.append(f"{type(error).__name__}: [error details omitted]")
        if isinstance(error, BaseExceptionGroup):
            for child in error.exceptions:
                visit(child, child.__traceback__)
        cause = error.__cause__ or (
            None if error.__suppress_context__ else error.__context__
        )
        if cause is not None:
            visit(cause, cause.__traceback__)

    visit(exc_info[1], exc_info[2])
    return "\n".join(lines)


def _event_schema(logger, method_name, event_dict):
    record = event_dict["_record"]
    if event_dict.get("exc_info"):
        # Exception messages may repeat a request or provider payload.
        event_dict["message"] = "Operation failed"
        event_dict.setdefault("event", "operation_failed")
        event_dict.setdefault("operation", record.funcName)
    args = event_dict.pop("positional_args", ())
    # Keep static message templates; formatting positional values can expose
    # provider replies, generated code, private paths or credential strings.
    if (
        args
        and isinstance(record.msg, str)
        and not getattr(record, "_diagnostic_safe_template", False)
    ):
        event_dict["message"] = record.msg
    if not record.name.startswith(("cowork", "anton", "uvicorn", "__main__")):
        event_dict["message"] = "Dependency diagnostic"
        event_dict.setdefault("event", "dependency_diagnostic")
    if record.name == "asyncio" and isinstance(record.msg, str):
        # asyncio appends task/callback reprs after its static first line.
        # Keep the actionable condition without copying those private values.
        conditions = {
            "Task was destroyed but it is pending!": "asyncio_task_destroyed",
            "Task exception was never retrieved": "asyncio_unretrieved_exception",
            "Future exception was never retrieved": "asyncio_unretrieved_exception",
            "Unhandled exception in event loop": "asyncio_unhandled_exception",
        }
        first_line = record.msg.splitlines()[0] if record.msg else ""
        if first_line in conditions:
            event_dict.update(message=first_line, event=conditions[first_line])
    errors = [
        arg
        for arg in (args.values() if isinstance(args, dict) else args)
        if isinstance(arg, BaseException)
    ]
    if errors:
        event_dict["message"] = "Operation failed"
        event_dict["error_type"] = type(errors[0]).__name__
        event_dict.setdefault("event", "operation_failed")
        event_dict.setdefault("operation", record.funcName)
    # ASGI lifespan.failed messages reach Uvicorn as already formatted text,
    # without exc_info. Keep only source coordinates and exception classes.
    lifecycle_trace = None
    if (
        record.name == "uvicorn.error"
        and Path(record.pathname).parent.name == "lifespan"
        and record.funcName == "send"
        and isinstance(record.msg, str)
    ):
        safe_lines = []
        for line in record.msg.splitlines():
            frame = re.fullmatch(
                r'\s*File "([^"\n]+)", line ([0-9]+), in ([A-Za-z0-9_<>.]+)', line
            )
            error = re.match(r"^([A-Za-z_][A-Za-z0-9_.]*(?:Error|Exception)):", line)
            if frame:
                safe_lines.append(f"{Path(frame[1]).name}:{frame[2]} in {frame[3]}")
            elif error:
                safe_lines.append(error[1])
        lifecycle_trace = "\n".join(safe_lines)
        event_dict.update(
            message="Application lifecycle failed", event="lifecycle_failed"
        )
    context = structlog.contextvars.get_contextvars()
    if not isinstance(record.msg, str):
        event_dict["message"] = "Operation diagnostic"
    event_dict.setdefault(
        "event",
        "operation_failed"
        if record.levelno >= logging.WARNING
        else "operation_progress",
    )
    event_dict.setdefault("operation", record.funcName)
    payload = {
        "logged_at": datetime.fromtimestamp(record.created, tz=UTC).isoformat(),
        "logger": record.name,
        "log_level": record.levelname,
        "message": event_dict["message"],
    }
    for field in EVENT_FIELDS:
        value = event_dict.get(field)
        if (
            field == "artifact_slugs"
            and isinstance(value, (list, tuple))
            and all(type(slug) is str for slug in value)
        ):
            payload[field] = list(value)
            continue
        if isinstance(value, UUID):
            value = str(value)
        if type(value) in (str, bool, int) or (
            type(value) is float and isfinite(value)
        ):
            payload[field] = value
    # Turns already have a server-generated request ID in the response protocol.
    # Keep it separately when an HTTP request also supplies a correlation ID.
    correlation = context.get("request_id")
    if correlation:
        if payload.get("request_id") and payload["request_id"] != correlation:
            payload["turn_request_id"] = payload["request_id"]
        payload["request_id"] = correlation
    if lifecycle_trace:
        payload["exception"] = lifecycle_trace
    exc_info = event_dict.get("exc_info")
    if exc_info:
        payload["exc_info"] = exc_info
        if isinstance(exc_info, tuple) and exc_info[1] is not None:
            payload["error_type"] = type(exc_info[1]).__name__
    return payload


def make_formatter():
    log_format = os.environ.get("LOG_FORMAT", "json")
    if log_format not in {"json", "text"}:
        raise ValueError("LOG_FORMAT must be json or text")
    renderer = (
        structlog.processors.JSONRenderer(
            separators=(",", ":"), ensure_ascii=False, allow_nan=False
        )
        if log_format == "json"
        else structlog.processors.KeyValueRenderer(
            key_order=["logged_at", "log_level", "logger", "message"]
        )
    )
    return structlog.stdlib.ProcessorFormatter(
        pass_foreign_args=True,
        foreign_pre_chain=[
            structlog.processors.EventRenamer("message"),
            structlog.contextvars.merge_contextvars,
            structlog.stdlib.ExtraAdder(allow=EVENT_FIELDS),
        ],
        processors=[
            _event_schema,
            structlog.processors.ExceptionRenderer(safe_exception_traceback),
            renderer,
        ],
    )


class AccessLogFilter(logging.Filter):
    """Filter exact probe routes before removing raw paths and query strings."""

    def __init__(self):
        super().__init__()
        self.mode = os.environ.get("ACCESS_LOG_MODE", "errors")
        if self.mode not in {"errors", "all"}:
            raise ValueError("ACCESS_LOG_MODE must be errors or all")

    def filter(self, record):
        if (
            record.name != "uvicorn.access"
            or not isinstance(record.args, tuple)
            or len(record.args) != 5
        ):
            return True
        _, method, target, _, status = record.args
        path = target.split("?", 1)[0]
        if method in {"GET", "HEAD"} and path in HEALTH_PATHS and 200 <= status < 300:
            return False
        if status < 400 and self.mode == "errors":
            return False
        record.msg, record.args = "HTTP request completed", ()
        record.event = "http_access"
        record.method = method if method in HTTPMethod else "OTHER"
        record.status_code = status
        if status >= 400:
            record.levelno = logging.ERROR if status >= 500 else logging.WARNING
            record.levelname = logging.getLevelName(record.levelno)
        return True


class ServerDiagnosticFilter(logging.Filter):
    def filter(self, record):
        # RequestLoggingMiddleware already emitted this escaping exception,
        # with its correlation. Uvicorn re-logs the same object after unwinding.
        return not (
            record.name.startswith("uvicorn.")
            and record.exc_info
            and getattr(record.exc_info[1], "_cowork_diagnostic_logged", False)
        )


def console_handler(access: bool = False):
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(make_formatter())
    from cowork.common.logger import _add_privacy_filters

    _add_privacy_filters(target=handler)
    if access:
        handler.addFilter(AccessLogFilter())
    else:
        handler.setLevel(os.environ.get("LOG_LEVEL", "WARNING").upper())
    return handler
