"""Owned handlers must not archive provider bodies, requests or wrapped messages."""
from __future__ import annotations

import asyncio
import json
import logging
import os
import subprocess
import sys
import textwrap
from types import SimpleNamespace
from uuid import uuid4

import anthropic
import httpx
import httpx2
import openai
import pytest
from fastapi import HTTPException
from openai.types.chat import ChatCompletion

from cowork.common import logger as app_logger
from cowork.db.scoped import LOCAL_SCOPE
from tests._uvicorn_harness import Launch, run_uvicorn_app

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


def _assert_safe(caplog, name, *, error_type="BadRequestError", provider_error=None, status=400,
                 level=logging.ERROR):
    provider_error = provider_error or error_type
    records = [r for r in caplog.records if r.name == name and r.levelno == level]
    assert len(records) == 1
    record = records[0]
    assert record.getMessage() == (
        f"Provider operation failed: error_type={error_type} provider_error={provider_error} status={status}"
    )
    assert record.args == (error_type, provider_error, status)
    assert record.exc_info is None
    assert record.exc_text is None
    assert record.stack_info is None
    assert PRIVATE not in repr(record.__dict__)
    return record


@pytest.mark.parametrize("kind", ["console", "all_file", "error_file"])
@pytest.mark.parametrize("chain", ["cause", "context", "suppressed_context", "group"])
@pytest.mark.parametrize("provider", ["openai", "anthropic", "anton"])
def test_owned_handlers_sanitize_real_provider_errors(kind, chain, provider, owned_logger, caplog):
    from anton.core.llm.provider import EndpointConfigurationError

    error = EndpointConfigurationError(PRIVATE) if provider == "anton" else _sdk_error(provider)
    logged = owned_logger("cowork.test.provider_filter", kind=kind)
    try:
        if chain == "cause":
            raise RuntimeError("wrapper " + PRIVATE) from error
        if chain in ("context", "suppressed_context"):
            try:
                raise error
            except Exception:
                if chain == "context":
                    raise RuntimeError("wrapper " + PRIVATE)
                # `from None` hides the context from the traceback, but the
                # wrapper's own message still repeats the provider's text.
                raise RuntimeError("wrapper " + PRIVATE) from None
        raise ExceptionGroup("group " + PRIVATE, [error])
    except Exception as wrapper:
        logged.logger.exception(f"Provider failure: {wrapper}", extra={"request_id": "request-123"}, stack_info=True)
    status = "unknown" if provider == "anton" else 400
    # The record's own exception is the wrapper; the provider error is in its chain.
    wrapper_type = "ExceptionGroup" if chain == "group" else "RuntimeError"
    record = _assert_safe(caplog, logged.logger.name, error_type=wrapper_type, provider_error=type(error).__name__,
                          status=status)
    assert record.request_id == "request-123"
    emitted = logged.output()
    assert PRIVATE not in emitted
    assert PRIVATE in str(error)
    if provider != "anton":
        assert PRIVATE in repr(error.body)
        assert error.response.status_code == 400


@pytest.mark.parametrize("shape", ["argument", "mapping", "message"])
def test_provider_exception_without_traceback_is_filtered(shape, owned_logger, caplog):
    error = _sdk_error()
    logged = owned_logger("cowork.test.provider_argument", level=logging.INFO)
    if shape == "argument":
        logged.logger.info("User-facing provider error: %s", error)
    elif shape == "mapping":
        logged.logger.info("User-facing provider error: %(error)s", {"error": error})
    else:
        logged.logger.info(error)
    _assert_safe(caplog, logged.logger.name, level=logging.INFO)
    assert PRIVATE not in logged.output()
    assert PRIVATE in str(error)


class _TextStatus(int):
    """An in-range int whose rendering carries private text."""

    def __str__(self) -> str:
        return PRIVATE

    __repr__ = __str__


# An int subclass can render anything, so only an exact int counts as a status.
@pytest.mark.parametrize("status", [True, 99, 600, "400", PRIVATE, None, _TextStatus(429)])
def test_provider_status_must_be_an_integer_http_status(status, owned_logger, caplog):
    error = _sdk_error()
    error.status_code = status
    logged = owned_logger("cowork.test.provider_status")
    logged.logger.error("Provider error: %s", error)
    _assert_safe(caplog, logged.logger.name, status="unknown")
    assert error.status_code == status


def test_bare_stream_api_error_and_hostile_status_are_never_coerced(owned_logger, caplog):
    class HostileStatus:
        def __str__(self):
            raise AssertionError("provider status must not be coerced")

    error = openai.APIError(PRIVATE, httpx.Request("POST", "https://example.com/" + PRIVATE),
                            body={"message": PRIVATE})
    error.status_code = HostileStatus()
    error.__cause__ = error  # Malformed chains must terminate.
    logged = owned_logger("cowork.test.provider_stream")
    logged.logger.error("Provider error: %s", error)
    _assert_safe(caplog, logged.logger.name, error_type="APIError", status="unknown")


def test_nonprovider_connection_errors_keep_their_traceback(owned_logger, caplog):
    logged = owned_logger("cowork.test.ordinary_connection")
    try:
        raise ConnectionError("ordinary socket failure")
    except ConnectionError:
        logged.logger.exception("Reconnect failed")
    (record,) = [r for r in caplog.records if r.name == logged.logger.name and r.levelno == logging.ERROR]
    assert record.getMessage() == "Reconnect failed"
    assert record.exc_info is not None
    assert str(record.exc_info[1]) == "ordinary socket failure"
    assert "ordinary socket failure" in logged.output()


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
    logged = owned_logger(probe_module.__name__)
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
    _assert_safe(caplog, logged.logger.name)
    assert PRIVATE not in logged.output()
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
    monkeypatch.setattr(handler_module, "_read_probe_settings",
                        lambda _session: SimpleNamespace(resolved_planning_provider=None))
    monkeypatch.setattr("cowork.services.providers.web_tool_kwargs_for", lambda _: {})
    monkeypatch.setattr(handler_module.ProbeHandler, "_build_llm_client", lambda *a, **kw: object())
    monkeypatch.setattr(handler_module, "CredentialProbe", lambda **kw: SimpleNamespace(run=failed_probe))
    logged = owned_logger(handler_module.__name__)
    handler = handler_module.ProbeHandler(scope=LOCAL_SCOPE)
    events = [event async for event in handler.run("staged", "postgres", None, "test", None)]
    assert PRIVATE in "".join(events)  # Client outcome is deliberately unchanged.
    completed = json.loads(events[-1].split("data: ", 1)[1])
    assert completed["response"]["status"] == "retry"
    _assert_safe(caplog, logged.logger.name)
    assert PRIVATE not in logged.output()


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

    handler = responses.ResponsesHandler()
    handler.harness = SimpleNamespace(formatter=formatter)
    logged = owned_logger(responses.__name__, level=logging.INFO)
    with pytest.raises(HTTPException) as caught:
        await handler._collect(None, uuid4(), "test-model", "input")
    assert caught.value.status_code == (400 if curated else 500)
    assert caught.value.detail["code"] == ("provider_auth" if curated else "anton_error")
    record = _assert_safe(caplog, logged.logger.name, error_type=type(error).__name__,
                          status="unknown" if curated else 400,
                          level=logging.INFO if curated else logging.ERROR)
    assert record.request_id == caught.value.detail["request_id"]
    assert PRIVATE not in logged.output()
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
    logged = owned_logger(responses.__name__)
    await handler._run_turn(conv_id=uuid4(), harness_input=[], original_content="input",
                            model="anton", disabled=None, harness_name="anton", harness_id="anton", buffer=buffer)
    records = list(read_records(buffer.path))
    assert records[-1].type == "Error"
    failed = saved["events"][-1]
    assert failed["type"] == "response.failed"
    assert failed["code"] == "anton_error"
    assert failed["error"] == "An unexpected error occurred."
    record = _assert_safe(caplog, logged.logger.name)
    assert record.request_id == failed["request_id"]
    assert PRIVATE not in logged.output()
    assert PRIVATE in str(error)


# A route's provider error escapes to Uvicorn, which logs "Exception in ASGI
# application" with the whole chain, provider message and body included.
_PROVIDER_FAILURE_APP = f"""
    import httpx, openai
    from fastapi import FastAPI
    from cowork.common.logger import setup_logging

    # cowork.server runs this at import, under every launcher.
    setup_logging()
    app = FastAPI()

    @app.get("/fail")
    async def fail():
        error = openai.BadRequestError(
            "{PRIVATE}",
            response=httpx.Response(400, request=httpx.Request("POST", "https://example.com/{PRIVATE}")),
            body={{"message": "{PRIVATE}"}},
        )
        raise RuntimeError("wrapper {PRIVATE}") from error
"""


@pytest.mark.parametrize("launch", ["module", "run"])
def test_uvicorns_default_log_config_prints_only_sanitized_provider_errors(tmp_path, launch: Launch):
    run = run_uvicorn_app(tmp_path=tmp_path, app_source=_PROVIDER_FAILURE_APP, paths=("/fail",), launch=launch)
    assert run.statuses == (500,)
    assert ("Provider operation failed: error_type=RuntimeError provider_error=BadRequestError status=400"
            in run.stderr), run.stderr
    assert "Finished server process" in run.stderr
    assert PRIVATE not in run.stdout + run.stderr


def test_debug_app_logging_keeps_real_sdk_transport_traces_out(tmp_path):
    # A real localhost provider, so each SDK sends through its default HTTP
    # client and transport, as it does against OpenAI or Anthropic.
    program = textwrap.dedent(f"""
        import http.server, json, logging, threading
        import anthropic, openai
        from cowork.common.logger import setup_logging
        setup_logging()
        records = []
        class Capture(logging.Handler):
            def emit(self, record): records.append(record)
        logging.getLogger().addHandler(Capture())

        class Provider(http.server.BaseHTTPRequestHandler):
            def do_POST(self):
                self.rfile.read(int(self.headers["Content-Length"]))
                body = json.dumps({{"error": {{"message": "{PRIVATE}", "type": "invalid_request_error"}}}}).encode()
                self.send_response(400)
                self.send_header("set-cookie", "session={PRIVATE}-cookie")
                self.send_header("x-request-id", "{PRIVATE}-request-id")
                self.send_header("content-type", "application/json")
                self.send_header("content-length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            def log_message(self, *args):
                pass

        server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Provider)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        base = "http://127.0.0.1:%d" % server.server_address[1]
        calls = [
            lambda: openai.OpenAI(api_key="test-key", base_url=base + "/v1", max_retries=0)
                .responses.create(model="test-model", input="hello"),
            lambda: anthropic.Anthropic(api_key="test-key", base_url=base, max_retries=0)
                .messages.create(model="test-model", max_tokens=1, messages=[{{"role": "user", "content": "hello"}}]),
        ]
        for call in calls:
            try:
                call()
            except (openai.APIError, anthropic.APIError) as error:
                logging.getLogger("cowork.test.provider_debug").error("Provider failure: %s", error)
        server.shutdown()
        transport = ("httpcore", "httpcore2", "httpx", "httpx2", "openai._base_client", "anthropic._base_client")
        print(json.dumps({{
            "diagnostics": sorted({{r.name for r in records if r.name.split(".")[0] in transport[:4]
                                    or r.name in transport[4:]}}),
            "levels": {{name: logging.getLogger(name).level for name in transport}},
            "errors": [r.getMessage() for r in records if r.name == "cowork.test.provider_debug"],
        }}))
    """)
    result = subprocess.run([sys.executable, "-c", program], capture_output=True, text=True,
                            env={**os.environ, "COWORK_HOME": str(tmp_path), "DATABASE_URI": "sqlite://",
                                 "LOG_LEVEL": "DEBUG"}, timeout=60)
    assert result.returncode == 0, result.stderr
    assert PRIVATE not in result.stdout + result.stderr
    report = json.loads(result.stdout.splitlines()[-1])
    assert report == {
        "diagnostics": [],
        "levels": dict.fromkeys(
            ("httpcore", "httpcore2", "httpx", "httpx2", "openai._base_client", "anthropic._base_client"),
            logging.ERROR,
        ),
        "errors": ["Provider operation failed: error_type=BadRequestError provider_error=BadRequestError status=400",
                   "Provider operation failed: error_type=BadRequestError provider_error=BadRequestError status=400"],
    }


def _request() -> httpx.Request:
    return httpx.Request("POST", "https://provider.example/v1/chat/completions")


def _rate_limit_error() -> openai.RateLimitError:
    return openai.RateLimitError(
        PRIVATE, response=httpx.Response(429, request=_request()), body={"message": PRIVATE},
    )


def _raised(*, error: BaseException) -> BaseException:
    try:
        raise error
    except BaseException as caught:
        return caught


def _wrapped_with_its_own_status() -> BaseException:
    try:
        raise _rate_limit_error()
    except openai.RateLimitError as provider_error:
        wrapper = RuntimeError(PRIVATE)
        wrapper.status_code = 502  # A gateway status on the wrapper is not the provider's.
        try:
            raise wrapper from provider_error
        except RuntimeError as caught:
            return caught


def _raised_while_handling() -> BaseException:
    try:
        try:
            raise _rate_limit_error()
        except openai.RateLimitError:
            raise KeyError(PRIVATE)
    except KeyError as caught:
        return caught


def _cause_subtree_before_context() -> BaseException:
    # The provider error sits one level down the cause, and another sits on the
    # wrapper's own context. A depth-first walk reaches the cause's context first.
    cause = RuntimeError(PRIVATE)
    cause.__context__ = _rate_limit_error()
    wrapper = RuntimeError(PRIVATE)
    wrapper.__cause__ = cause
    wrapper.__context__ = openai.APIConnectionError(message=PRIVATE, request=_request())
    return wrapper


# Anton's ProviderErrorFilter runs these same cases and expects the same lines,
# so the two copies stay aligned.
@pytest.mark.parametrize("build,expected", [
    (lambda: _raised(error=_rate_limit_error()),
     "error_type=RateLimitError provider_error=RateLimitError status=429"),
    (_wrapped_with_its_own_status,
     "error_type=RuntimeError provider_error=RateLimitError status=429"),
    (lambda: _raised(error=openai.APIConnectionError(message=PRIVATE, request=_request())),
     "error_type=APIConnectionError provider_error=APIConnectionError status=unknown"),
    (_raised_while_handling,
     "error_type=KeyError provider_error=RateLimitError status=429"),
    (lambda: _raised(error=openai.LengthFinishReasonError(completion=ChatCompletion.model_construct(usage=None))),
     "error_type=LengthFinishReasonError provider_error=LengthFinishReasonError status=unknown"),
    (_cause_subtree_before_context,
     "error_type=RuntimeError provider_error=RateLimitError status=429"),
], ids=["provider_429", "wrapper_status", "no_status", "raised_while_handling", "not_an_api_error",
        "cause_subtree_before_context"])
def test_provider_filter_parity_cases(build, expected: str) -> None:
    error = build()
    record = logging.LogRecord("cowork.test.provider_parity", logging.ERROR, __file__, 1,
                               f"Failed: {PRIVATE}", (), (type(error), error, error.__traceback__))
    assert app_logger.ProviderErrorFilter().filter(record)
    assert record.getMessage() == f"Provider operation failed: {expected}"
    assert record.exc_info is None and record.exc_text is None and record.stack_info is None
    assert PRIVATE not in logging.Formatter().format(record)


@pytest.mark.parametrize("path", ["streamed", "collected"])
def test_a_repaired_content_validation_error_keeps_its_count_in_the_log(path, monkeypatch, owned_logger, caplog):
    from anton.core.llm.provider import ContentValidationError
    from cowork.handlers import responses
    from tests.test_inprocess_request_id import _RecBuffer, _failed_payload, _failing_handler, _run

    saved: dict = {}
    handler = _failing_handler(monkeypatch, saved, ContentValidationError(PRIVATE))
    logged = owned_logger(responses.__name__, level=logging.WARNING)
    conversation_id = uuid4()
    if path == "streamed":
        _run(handler, _RecBuffer(), conv_id=conversation_id)
        request_id = _failed_payload(saved)["request_id"]
    else:
        handler.scope = None
        handler.harness = responses.get_harness("anton")
        with pytest.raises(HTTPException) as caught:
            asyncio.run(handler._collect(None, conversation_id, "test-model", "input"))
        request_id = caught.value.detail["request_id"]
    assert saved.get("repaired")
    [record] = [r for r in caplog.records if r.name == logged.logger.name and "repaired" in r.getMessage()]
    assert record.levelno == logging.WARNING
    # The ids travel as record attributes, not in the message.
    assert record.getMessage() == (
        "[responses] content validation error; "
        "repaired 1 message(s) with image content: error_type=ContentValidationError"
    )
    assert (record.request_id, record.conversation_id) == (request_id, str(conversation_id))
    emitted = logged.output()
    assert f"[Req:{request_id}][Conversation:{conversation_id}]" in emitted
    assert PRIVATE not in emitted
