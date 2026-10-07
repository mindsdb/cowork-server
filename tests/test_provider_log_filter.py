"""Owned handlers must not archive provider bodies, requests or wrapped messages."""
from __future__ import annotations

import io
import json
import logging
import os
import subprocess
import sys
import textwrap
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock
from uuid import uuid4

import anthropic
import httpx
import httpx2
import openai
import pytest
from fastapi import HTTPException

from cowork.common import logger as app_logger

PRIVATE = "provider_private_request_marker"


def _sdk_error(provider: str = "openai"):
    # Drive the SDK HTTP path: these are the error objects real providers emit.
    body = {"error": {"message": PRIVATE, "type": PRIVATE + "-type",
                      "code": PRIVATE + "-code", "param": PRIVATE + "-param"}}
    transport_module = httpx if provider == "openai" else httpx2
    transport = transport_module.MockTransport(lambda request: transport_module.Response(400, json=body))
    client_type = openai.OpenAI if provider == "openai" else anthropic.Anthropic
    with client_type(api_key="private-api-key", max_retries=0,
                     http_client=transport_module.Client(transport=transport)) as client:
        try:
            if provider == "openai":
                client.responses.create(model="test-model", input=PRIVATE)
            else:
                client.messages.create(model="test-model", max_tokens=1,
                                       messages=[{"role": "user", "content": PRIVATE}])
        except (openai.APIError, anthropic.APIError) as exc:
            return exc
    raise AssertionError("the endpoint must reject the request")


@pytest.fixture
def owned_logger(monkeypatch, caplog, tmp_path):
    handlers = []

    def install(name, kind="console", level=logging.ERROR):
        stream = io.StringIO()
        if kind == "console":
            monkeypatch.setenv("RICH_LOGGING", "false")
            handler = app_logger.setup_console_handler()
            handler.setStream(stream)
            handlers.append(handler)
        else:
            files = app_logger.setup_file_logging(str(tmp_path))
            handlers.extend(files)
            handler = files[0 if kind == "all_file" else 1]
        logger = logging.getLogger(name)
        monkeypatch.setattr(logger, "handlers", [handler, caplog.handler])
        monkeypatch.setattr(logger, "propagate", False)
        monkeypatch.setattr(logger, "level", level)
        monkeypatch.setattr(logger, "disabled", False)
        return logger, handler, stream

    yield install
    for handler in handlers:
        handler.close()


def _assert_safe(caplog, name, *, error_type="BadRequestError", status=400, level=logging.ERROR):
    records = [r for r in caplog.records if r.name == name and r.levelno == level]
    assert len(records) == 1
    record = records[0]
    assert record.getMessage() == f"Provider operation failed: error_type={error_type} status={status}"
    assert record.args == (error_type, status)
    assert record.exc_info is None
    assert record.exc_text is None
    assert record.stack_info is None
    assert PRIVATE not in repr(record.__dict__)
    return record


@pytest.mark.parametrize("kind", ["console", "all_file", "error_file"])
@pytest.mark.parametrize("chain", ["cause", "context", "group"])
@pytest.mark.parametrize("provider", ["openai", "anthropic", "anton"])
def test_owned_handlers_sanitize_real_provider_errors(kind, chain, provider, owned_logger, caplog):
    from anton.core.llm.provider import EndpointConfigurationError

    error = EndpointConfigurationError(PRIVATE) if provider == "anton" else _sdk_error(provider)
    logger, handler, stream = owned_logger("cowork.test.provider_filter", kind)
    try:
        if chain == "cause":
            raise RuntimeError("wrapper " + PRIVATE) from error
        if chain == "context":
            try:
                raise error
            except Exception:
                raise RuntimeError("wrapper " + PRIVATE)
        raise ExceptionGroup("group " + PRIVATE, [error])
    except Exception as wrapper:
        logger.exception(f"Provider failure: {wrapper}", extra={"request_id": "request-123"}, stack_info=True)
    status = "unknown" if provider == "anton" else 400
    record = _assert_safe(caplog, logger.name, error_type=type(error).__name__, status=status)
    assert record.request_id == "request-123"
    handler.flush()
    emitted = stream.getvalue() if kind == "console" else Path(handler.baseFilename).read_text()
    assert PRIVATE not in emitted
    assert PRIVATE in str(error)
    if provider != "anton":
        assert PRIVATE in repr(error.body)
        assert error.response.status_code == 400


@pytest.mark.parametrize("shape", ["argument", "mapping", "message"])
def test_provider_exception_without_traceback_is_filtered(shape, owned_logger, caplog):
    error = _sdk_error()
    logger, _, stream = owned_logger("cowork.test.provider_argument", level=logging.INFO)
    if shape == "argument":
        logger.info("User-facing provider error: %s", error)
    elif shape == "mapping":
        logger.info("User-facing provider error: %(error)s", {"error": error})
    else:
        logger.info(error)
    _assert_safe(caplog, logger.name, level=logging.INFO)
    assert PRIVATE not in stream.getvalue()
    assert PRIVATE in str(error)


@pytest.mark.parametrize("status", [True, 99, 600, "400", PRIVATE, None])
def test_provider_status_must_be_an_integer_http_status(status, owned_logger, caplog):
    error = _sdk_error()
    error.status_code = status
    logger, _, _ = owned_logger("cowork.test.provider_status")
    logger.error("Provider error: %s", error)
    _assert_safe(caplog, logger.name, status="unknown")
    assert error.status_code == status


def test_bare_stream_api_error_and_hostile_status_are_never_coerced(owned_logger, caplog):
    class HostileStatus:
        def __str__(self):
            raise AssertionError("provider status must not be coerced")

    error = openai.APIError(PRIVATE, httpx.Request("POST", "https://example.com/" + PRIVATE),
                            body={"message": PRIVATE})
    error.status_code = HostileStatus()
    error.__cause__ = error  # Malformed chains must terminate.
    logger, _, _ = owned_logger("cowork.test.provider_stream")
    logger.error("Provider error: %s", error)
    _assert_safe(caplog, logger.name, error_type="APIError", status="unknown")


def test_nonprovider_connection_errors_keep_their_traceback(owned_logger, caplog):
    logger, _, stream = owned_logger("cowork.test.ordinary_connection")
    try:
        raise ConnectionError("ordinary socket failure")
    except ConnectionError:
        logger.exception("Reconnect failed")
    (record,) = [r for r in caplog.records if r.name == logger.name and r.levelno == logging.ERROR]
    assert record.getMessage() == "Reconnect failed"
    assert record.exc_info is not None
    assert str(record.exc_info[1]) == "ordinary socket failure"
    assert "ordinary socket failure" in stream.getvalue()


@pytest.mark.asyncio
@pytest.mark.parametrize("stage", ["build", "turn"])
@pytest.mark.parametrize("provider", ["openai", "anthropic"])
async def test_real_credential_probe_keeps_verdict_but_sanitizes_log(
    stage, provider, monkeypatch, tmp_path, owned_logger, caplog
):
    from cowork.services.connectors import probe as probe_module

    error = _sdk_error(provider)
    closed = []

    class Session:
        async def turn_stream(self, *args, **kwargs):
            raise error
            yield

    session = Session()

    def build(config):
        if stage == "build":
            raise error
        return session

    monkeypatch.setattr(probe_module, "_probe_tmp_dir", lambda: tmp_path)
    monkeypatch.setattr(probe_module, "build_chat_session", build)
    monkeypatch.setattr(probe_module, "close_session_scratchpads", lambda s, **kwargs: closed.append(s))
    logger, _, stream = owned_logger(probe_module.__name__)
    probe = probe_module.CredentialProbe(engine="postgres", credentials={"password": PRIVATE},
                                         llm_client=None, workspace=None)
    events = [event async for event in probe.run()]
    assert events[-1][0] == "verdict"
    outcome = events[-1][1]
    assert outcome.status == "failure"
    prefix = "Could not start probe: " if stage == "build" else "Probe crashed: "
    assert outcome.error == prefix + str(error)
    assert closed == ([] if stage == "build" else [session])
    assert not list(tmp_path.glob("probe-*.env"))
    _assert_safe(caplog, logger.name)
    assert PRIVATE not in stream.getvalue()
    assert PRIVATE in repr(error.body)


@pytest.mark.asyncio
async def test_real_probe_handler_keeps_sse_outcome_but_sanitizes_iteration_error(
    monkeypatch, owned_logger, caplog
):
    from cowork.handlers import probe as handler_module

    error = _sdk_error()

    async def failed_probe():
        raise error
        yield

    monkeypatch.setattr(handler_module.store, "get", lambda _: {"values": {"password": PRIVATE}})
    monkeypatch.setattr(handler_module.registry, "get_connector", lambda _: SimpleNamespace(
        form=SimpleNamespace(form_id="probe-form", model_dump=lambda: {"form_id": "probe-form"})))
    monkeypatch.setattr("cowork.common.settings.user_settings.get_user_settings",
                        lambda: SimpleNamespace(resolved_planning_provider=None))
    monkeypatch.setattr("cowork.services.providers.web_tool_kwargs_for", lambda _: {})
    monkeypatch.setattr(handler_module.ProbeHandler, "_build_llm_client", lambda *a, **kw: object())
    monkeypatch.setattr(handler_module, "CredentialProbe", lambda **kw: SimpleNamespace(run=failed_probe))
    logger, _, stream = owned_logger(handler_module.__name__)
    handler = handler_module.ProbeHandler(session=MagicMock())
    events = [event async for event in handler.run("staged", "postgres", None, "test", None)]
    assert PRIVATE in "".join(events)  # Client outcome is deliberately unchanged.
    completed = json.loads(events[-1].split("data: ", 1)[1])
    assert completed["response"]["status"] == "retry"
    _assert_safe(caplog, logger.name)
    assert PRIVATE not in stream.getvalue()


@pytest.mark.asyncio
@pytest.mark.parametrize("curated", [False, True])
async def test_real_response_collection_keeps_http_outcome_but_sanitizes_provider_error(
    curated, monkeypatch, owned_logger, caplog
):
    from anton.core.llm.provider import ProviderAuthError
    from cowork.handlers import responses

    error = ProviderAuthError(PRIVATE) if curated else _sdk_error()

    async def formatter(*args):
        raise error
        yield

    monkeypatch.setattr(responses, "get_user_settings", lambda _: SimpleNamespace(harness="anton"))
    handler = responses.ResponsesHandler(MagicMock())
    handler.harness = SimpleNamespace(formatter=formatter)
    logger, _, stream = owned_logger(responses.__name__, level=logging.INFO)
    with pytest.raises(HTTPException) as caught:
        await handler._collect(None, uuid4(), "test-model", "input")
    assert caught.value.status_code == (400 if curated else 500)
    assert caught.value.detail["code"] == ("provider_auth" if curated else "anton_error")
    record = _assert_safe(caplog, logger.name, error_type=type(error).__name__,
                          status="unknown" if curated else 400,
                          level=logging.INFO if curated else logging.ERROR)
    assert record.request_id == caught.value.detail["request_id"]
    assert PRIVATE not in stream.getvalue()
    assert PRIVATE in str(error)


@pytest.mark.asyncio
async def test_real_streamed_turn_preserves_failure_frame_and_sanitizes_sdk_traceback(
    monkeypatch, tmp_path, owned_logger, caplog
):
    from cowork.handlers import responses
    from cowork.harnesses.anton_harness.stream_formatter import format_responses_stream
    from cowork.streaming.buffer import FileStreamBuffer, read_records
    from tests.test_model_wait_ticker import _handler

    error = _sdk_error()

    class Harness:
        formatter = staticmethod(format_responses_stream)

        async def stream_response(self, **kwargs):
            raise error
            yield

    saved = {}
    handler = _handler(monkeypatch, saved, Harness())
    buffer = FileStreamBuffer(tmp_path / "turn.log")
    logger, _, stream = owned_logger(responses.__name__)
    await handler._run_turn(conv_id=uuid4(), harness_input=[], original_content="input",
                            model="anton", disabled=None, harness_name="anton", harness_id="anton", buffer=buffer)
    records = list(read_records(buffer.path))
    assert records[-1].type == "Error"
    failed = saved["events"][-1]
    assert failed["type"] == "response.failed"
    assert failed["code"] == "anton_error"
    assert failed["error"] == "An unexpected error occurred."
    record = _assert_safe(caplog, logger.name)
    assert record.request_id == failed["request_id"]
    assert PRIVATE not in stream.getvalue()
    assert PRIVATE in str(error)


def test_cli_uvicorn_handlers_sanitize_real_provider_failure(tmp_path):
    # Isolate dictConfig, which closes process-wide handlers. No socket needed.
    program = textwrap.dedent("""
        import asyncio, h11, httpx, json, logging, openai
        from unittest import mock
        from types import SimpleNamespace
        import uvicorn
        from uvicorn.protocols.http.h11_impl import RequestResponseCycle
        from cowork import cli
        records = []
        class Capture(logging.Handler):
            def emit(self, record): records.append(record)
        error = openai.BadRequestError('provider_private_request_marker',
            response=httpx.Response(400, request=httpx.Request('POST', 'https://example.com/provider_private_request_marker')),
            body={'message': 'provider_private_request_marker'})
        async def app(scope, receive, send):
            raise RuntimeError('wrapper provider_private_request_marker') from error
        async def run():
            with mock.patch.object(cli.uvicorn, 'run') as start: cli.main()
            config = uvicorn.Config(app, **{key: value for key, value in start.call_args.kwargs.items()
                if key not in ('host', 'port', 'reload', 'timeout_graceful_shutdown')})
            logger = logging.getLogger('uvicorn.error')
            logger.addHandler(Capture())
            conn = h11.Connection(h11.SERVER)
            conn.receive_data(b'GET /fail HTTP/1.1\\r\\nHost: localhost\\r\\n\\r\\n')
            conn.next_event()
            transport = mock.Mock()
            cycle = RequestResponseCycle(
                scope={'type': 'http', 'headers': [], 'http_version': '1.1', 'method': 'GET',
                       'path': '/fail', 'query_string': b'', 'client': ('127.0.0.1', 1234)},
                conn=conn, transport=transport, flow=SimpleNamespace(write_paused=False), logger=logger,
                access_logger=logging.getLogger('uvicorn.access'), access_log=False,
                default_headers=[], message_event=asyncio.Event(), on_response=lambda: None)
            await cycle.run_asgi(app)
            response = b''.join(call.args[0] for call in transport.write.call_args_list)
            selected = [r for r in records if r.name == 'uvicorn.error' and r.levelno == logging.ERROR]
            print(json.dumps({'status_500': b'500' in response.split(b'\\r\\n', 1)[0],
                'records': [{'message': r.getMessage(), 'args': r.args, 'traceback': r.exc_info is not None,
                             'exc_text': r.exc_text} for r in selected],
                'original_message': str(error)}))
        asyncio.run(run())
    """)
    result = subprocess.run([sys.executable, "-c", program], capture_output=True, text=True,
                            env={**os.environ, "COWORK_HOME": str(tmp_path), "DATABASE_URI": "sqlite://"}, timeout=30)
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout.splitlines()[-1])
    assert report["status_500"] is True
    assert report["records"] == [{"message": "Provider operation failed: error_type=BadRequestError status=400",
                                  "args": ["BadRequestError", 400], "traceback": False, "exc_text": None}]
    assert report["original_message"] == PRIVATE
    assert PRIVATE not in result.stderr
    assert PRIVATE not in "\n".join(result.stdout.splitlines()[:-1])


def test_debug_app_logging_suppresses_actual_anthropic_request_diagnostics(tmp_path):
    program = textwrap.dedent("""
        import json, logging
        from cowork.common.logger import setup_logging
        from tests.test_provider_log_filter import _sdk_error
        for name in ('anthropic', 'anthropic._base_client', 'httpx2'):
            logging.getLogger(name).setLevel(logging.DEBUG)
        setup_logging()
        records = []
        class Capture(logging.Handler):
            def emit(self, record): records.append(record)
        logging.getLogger().addHandler(Capture())
        error = _sdk_error('anthropic')
        logging.getLogger('cowork.test.provider_debug').error('Provider failure: %s', error)
        diagnostics = [r for r in records if r.name in ('anthropic._base_client', 'httpx2')]
        print(json.dumps({'diagnostics': [r.getMessage() for r in diagnostics],
            'levels': {name: logging.getLogger(name).level for name in ('anthropic._base_client', 'httpx2')},
            'errors': [r.getMessage() for r in records if r.name == 'cowork.test.provider_debug']}))
    """)
    result = subprocess.run([sys.executable, "-c", program], capture_output=True, text=True,
                            env={**os.environ, "COWORK_HOME": str(tmp_path), "DATABASE_URI": "sqlite://",
                                 "LOG_LEVEL": "DEBUG"}, timeout=30)
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout.splitlines()[-1])
    assert report == {"diagnostics": [],
                      "levels": {"anthropic._base_client": logging.ERROR, "httpx2": logging.ERROR},
                      "errors": ["Provider operation failed: error_type=BadRequestError status=400"]}
    assert PRIVATE not in result.stdout + result.stderr
