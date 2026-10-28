"""Runtime diagnostics must remain useful without disclosing exception values."""

import io
import json
import logging

import pytest

from cowork.common.logger import setup_console_handler


@pytest.mark.parametrize(
    ("expected_org", "status", "reason"),
    [
        (None, 426, "expected_organization_missing"),
        ("private-invalid-org", 409, "expected_organization_invalid"),
        ("00000000-0000-0000-0000-000000000001", 409, "organization_mismatch"),
    ],
)
def test_registered_organization_refusal_logs_one_safe_reason(
    monkeypatch, org_deployment, expected_org, status, reason
):
    from fastapi.testclient import TestClient

    from cowork.server import create_app
    from tests.test_principal import _browser_headers

    app = create_app()
    output = io.StringIO()
    handler = setup_console_handler()
    handler.setStream(output)
    for name in ("cowork.principal", "cowork.common.request_logging"):
        logger = logging.getLogger(name)
        monkeypatch.setattr(logger, "handlers", [handler])
        monkeypatch.setattr(logger, "propagate", False)
        monkeypatch.setattr(logger, "level", logging.WARNING)
    response = TestClient(app).get(
        "/api/v1/conversations?token=private-query",
        headers=_browser_headers(expected_org),
    )
    assert response.status_code == status
    assert response.json()["code"] == "organization_reload_required"
    [diagnostic] = [json.loads(line) for line in output.getvalue().splitlines()]
    assert diagnostic["event"] == "request_refused"
    assert diagnostic["reason"] == reason
    assert diagnostic["request_id"] == response.headers["x-request-id"]
    assert "private-" not in output.getvalue()


@pytest.mark.parametrize("launch", ["module", "run"])
@pytest.mark.parametrize("failure", ["provider", "database"])
def test_uvicorn_does_not_repeat_sanitized_request_exception(tmp_path, launch, failure):
    from tests._uvicorn_harness import run_uvicorn_app

    run = run_uvicorn_app(
        tmp_path=tmp_path,
        launch=launch,
        paths=("/logging-regression-crash",),
        app_source=f"""
            from contextlib import asynccontextmanager
            import httpx, openai, sqlite3
            from cowork.server import create_app

            app = create_app()
            @asynccontextmanager
            async def no_backing_services(app):
                yield
            app.router.lifespan_context = no_backing_services

            @app.get("/logging-regression-crash")
            async def crash():
                if {failure!r} == "database":
                    raise sqlite3.OperationalError("private-database-query")
                raise openai.BadRequestError(
                    "private-provider-response",
                    response=httpx.Response(400, request=httpx.Request("POST", "https://example.test")),
                    body={{"message": "private-provider-response"}},
                )
        """,
    )
    assert run.statuses == (500,)
    records = [json.loads(line) for line in run.stderr.splitlines()]
    failures = [
        r
        for r in records
        if r["logger"] in ("cowork.common.request_logging", "uvicorn.error")
    ]
    assert len(failures) == 1
    assert failures[0]["event"] == "request_failed"
    assert failures[0]["status_code"] == 500
    access = next(r for r in records if r.get("event") == "http_access")
    assert access["request_id"] == failures[0]["request_id"]
    assert "private-" not in run.stdout + run.stderr


@pytest.mark.parametrize("supplied", [None, "private-request-id", "0" * 32])
def test_registered_application_correlates_unhandled_500(monkeypatch, supplied):
    from uuid import UUID

    from fastapi.testclient import TestClient
    from structlog.contextvars import get_contextvars

    from cowork.server import create_app

    app = create_app()

    @app.get("/logging-regression-crash")
    async def crash():
        raise RuntimeError("private-exception")

    output = io.StringIO()
    handler = setup_console_handler()
    handler.setStream(output)
    logger = logging.getLogger("cowork.common.request_logging")
    monkeypatch.setattr(logger, "handlers", [handler])
    monkeypatch.setattr(logger, "propagate", False)
    monkeypatch.setattr(logger, "level", logging.WARNING)
    response = TestClient(app, raise_server_exceptions=False).get(
        "/logging-regression-crash",
        headers={} if supplied is None else {"x-request-id": supplied},
    )
    assert response.status_code == 500
    assert response.text == "Internal Server Error"
    [diagnostic] = [json.loads(line) for line in output.getvalue().splitlines()]
    assert diagnostic["event"] == "request_failed"
    assert response.headers["x-request-id"] == diagnostic["request_id"]
    assert UUID(diagnostic["request_id"])
    assert diagnostic["request_id_source"] == (
        "header" if supplied == "0" * 32 else "generated"
    )
    assert "private-" not in output.getvalue()
    assert "request_id" not in get_contextvars()


@pytest.mark.parametrize("handled", [True, False])
def test_registered_application_preserves_custom_500_response(monkeypatch, handled):
    from fastapi.responses import JSONResponse
    from fastapi.testclient import TestClient
    from structlog.contextvars import get_contextvars

    from cowork.server import create_app

    app = create_app()

    @app.exception_handler(500)
    async def custom_error(request, exc):
        return JSONResponse(
            {"error": "custom"}, status_code=500, headers={"x-error-kind": "custom"}
        )

    @app.get("/logging-regression-500")
    async def error():
        if handled:
            return await custom_error(None, None)
        raise RuntimeError("private-exception")

    output = io.StringIO()
    handler = setup_console_handler()
    handler.setStream(output)
    logger = logging.getLogger("cowork.common.request_logging")
    monkeypatch.setattr(logger, "handlers", [handler])
    monkeypatch.setattr(logger, "propagate", False)
    monkeypatch.setattr(logger, "level", logging.WARNING)
    response = TestClient(app, raise_server_exceptions=False).get(
        "/logging-regression-500"
    )
    assert response.status_code == 500
    assert response.json() == {"error": "custom"}
    assert response.headers["x-error-kind"] == "custom"
    [diagnostic] = [json.loads(line) for line in output.getvalue().splitlines()]
    assert diagnostic["event"] == "request_failed"
    assert diagnostic["request_id"] == response.headers["x-request-id"]
    assert diagnostic["reason"] == (
        "internal_server_error" if handled else "unhandled_exception"
    )
    assert "private-" not in output.getvalue()
    assert "request_id" not in get_contextvars()


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
    from starlette.responses import Response, StreamingResponse
    from starlette.routing import Route
    from structlog.contextvars import get_contextvars

    from cowork.server import create_app

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

    app = create_app()
    app.router.routes.extend(
        [Route("/{status:int}", endpoint), Route("/stream", stream)]
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
