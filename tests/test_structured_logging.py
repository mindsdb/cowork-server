"""Runtime diagnostics must remain useful without disclosing exception values."""

import io
import json
import logging

import pytest

from cowork.common.logger import setup_console_handler


def test_console_is_json_and_omits_exception_values(monkeypatch):
    monkeypatch.delenv("LOG_FORMAT", raising=False)
    output = io.StringIO()
    handler = setup_console_handler()
    handler.setStream(output)
    try:
        raise RuntimeError("private-provider-body")
    except RuntimeError:
        record = logging.LogRecord(
            "cowork.test",
            logging.ERROR,
            __file__,
            20,
            "Operation failed",
            (),
            __import__("sys").exc_info(),
        )
    record.conversation_id = "conversation-1"
    record.credentials = {"token": "private-extra"}
    handler.handle(record)
    event = json.loads(output.getvalue())
    assert event["log_level"] == "ERROR"
    assert event["conversation_id"] == "conversation-1"
    assert event["error_type"] == "RuntimeError"
    assert "private-" not in output.getvalue()
    assert "test_console_is_json" in event["exception"]


@pytest.mark.parametrize("mode", ["errors", "all"])
def test_successful_probe_is_quiet_but_failed_probe_survives(monkeypatch, mode):
    from cowork.common.structured_logging import AccessLogFilter, make_formatter

    monkeypatch.setenv("ACCESS_LOG_MODE", mode)
    policy = AccessLogFilter()

    def record(status):
        return logging.LogRecord(
            "uvicorn.access",
            logging.INFO,
            __file__,
            1,
            '%s - "%s %s HTTP/%s" %d',
            ("secret-client", "GET", "/api/v1/health/live?token=secret", "1.1", status),
            None,
        )

    assert not policy.filter(record(200))
    failure = record(503)
    assert policy.filter(failure)
    assert failure.levelno == logging.ERROR
    event = json.loads(make_formatter().format(failure))
    assert event["status_code"] == 503
    assert "secret" not in json.dumps(event)


def test_ordinary_success_obeys_access_control(monkeypatch):
    from cowork.common.structured_logging import AccessLogFilter

    record = logging.LogRecord(
        "uvicorn.access",
        logging.INFO,
        __file__,
        1,
        '%s - "%s %s HTTP/%s" %d',
        ("client", "GET", "/api/v1/projects", "1.1", 200),
        None,
    )
    monkeypatch.setenv("ACCESS_LOG_MODE", "errors")
    assert not AccessLogFilter().filter(record)
    monkeypatch.setenv("ACCESS_LOG_MODE", "all")
    assert AccessLogFilter().filter(record)


def test_preformatted_lifespan_failure_does_not_disclose_values():
    from cowork.common.structured_logging import make_formatter

    record = logging.LogRecord(
        "uvicorn.error",
        logging.ERROR,
        "/uvicorn/lifespan/on.py",
        121,
        'Traceback (most recent call last):\n  File "/app/service.py", line 12, in start\n    raise RuntimeError("private-source")\nRuntimeError: private-provider-body\n',
        (),
        None,
        func="send",
    )
    event = json.loads(make_formatter().format(record))
    assert "private-" not in str(event)
    assert event["event"] == "lifecycle_failed"
    assert "service.py" in event["exception"]
    assert "RuntimeError" in event["exception"]
    record = logging.LogRecord(
        "asyncio",
        logging.ERROR,
        __file__,
        1,
        "Task was destroyed but it is pending!\ntask: private-task-value",
        (),
        None,
    )
    event = json.loads(make_formatter().format(record))
    assert event["event"] == "asyncio_task_destroyed"
    assert "private-" not in str(event)


@pytest.mark.asyncio
async def test_request_boundary_correlates_concurrent_refusals_and_stream_failure(
    monkeypatch,
):
    import asyncio

    import httpx
    from starlette.applications import Starlette
    from starlette.responses import Response, StreamingResponse
    from starlette.routing import Route
    from structlog.contextvars import get_contextvars

    from cowork.common.request_logging import RequestLoggingMiddleware

    output = io.StringIO()
    handler = setup_console_handler()
    handler.setStream(output)
    logger = logging.getLogger("cowork.common.request_logging")
    monkeypatch.setattr(logger, "handlers", [handler])
    monkeypatch.setattr(logger, "propagate", False)
    monkeypatch.setattr(logger, "level", logging.WARNING)

    async def endpoint(request):
        await asyncio.sleep(0)
        assert get_contextvars()["request_id"] == request.headers["x-request-id"]
        # A rejected candidate is followed by a successful alternative: only
        # the final response, not intermediate permission attempts, is logged.
        status = int(request.path_params["status"])
        return Response("private-response", status_code=status)

    async def stream(request):
        async def body():
            yield b"private-output"
            await asyncio.sleep(0)
            raise RuntimeError("private-stream-error")

        return StreamingResponse(body())

    app = RequestLoggingMiddleware(
        Starlette(routes=[Route("/{status:int}", endpoint), Route("/stream", stream)])
    )
    ids = [f"00000000-0000-0000-0000-{i:012d}" for i in range(4)]
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        responses = await asyncio.gather(
            *(
                client.get(
                    f"/{status}?token=private-query", headers={"x-request-id": ids[i]}
                )
                for i, status in enumerate((200, 401, 403))
            )
        )
        assert [response.status_code for response in responses] == [200, 401, 403]
        assert [response.headers["x-request-id"] for response in responses] == ids[:3]
        with pytest.raises(RuntimeError, match="private-stream-error"):
            await client.get("/stream", headers={"x-request-id": ids[3]})
    records = [json.loads(line) for line in output.getvalue().splitlines()]
    assert sorted((record["request_id"], record["event"]) for record in records) == [
        (ids[1], "request_refused"),
        (ids[2], "request_refused"),
        (ids[3], "request_failed"),
    ]
    assert "private-" not in output.getvalue()
    assert "request_id" not in get_contextvars()


def test_http_correlation_preserves_the_turn_protocol_id():
    from structlog.contextvars import bind_contextvars, reset_contextvars

    from cowork.common.structured_logging import make_formatter

    token = bind_contextvars(request_id="00000000-0000-0000-0000-000000000001")
    try:
        record = logging.LogRecord(
            "cowork.test", logging.WARNING, __file__, 1, "Turn failed", (), None
        )
        record.request_id = "server-generated-turn-id"
        event = json.loads(make_formatter().format(record))
        assert event["request_id"] == "00000000-0000-0000-0000-000000000001"
        assert event["turn_request_id"] == "server-generated-turn-id"
        assert record.request_id == "server-generated-turn-id"
    finally:
        reset_contextvars(**token)
