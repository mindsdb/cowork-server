"""The configured router through Cowork, Anton and the OpenAI SDK's HTTP layer."""

import asyncio
import json
import logging
from unittest.mock import AsyncMock

import httpx
import openai
import pytest
from pydantic import SecretStr

from cowork.common.settings.user_settings import Provider, UserSettings
from cowork.handlers import response_routing as routing
from cowork.handlers.first_take import FirstTake
from cowork.services.providers import build_llm_client


MODEL = "gpt-6-astra"
HISTORY = [{"role": "user", "content": "Hello"}]


def _response(status="completed", output=(), **extra):
    # OpenAI Responses wire shape, parsed by the real SDK before Anton sees it.
    return {
        "id": "resp_router", "object": "response", "created_at": 0,
        "model": MODEL, "status": status, "output": list(output),
        "usage": {"input_tokens": 10, "output_tokens": 3, "total_tokens": 13},
        "parallel_tool_calls": True, "tool_choice": "auto", "tools": [], **extra,
    }


def _text(text):
    return {
        "type": "response.output_text.delta", "item_id": "msg_router",
        "output_index": 0, "content_index": 0, "delta": text, "logprobs": [],
    }


def _done():
    return {"type": "response.completed", "response": _response()}


class _Stream(httpx.AsyncByteStream):
    def __init__(self, *events, hold_open=False):
        self.events = events
        self.hold_open = hold_open
        self.waiting = asyncio.Event()
        self.closed = False

    async def __aiter__(self):
        for number, event in enumerate(self.events):
            payload = {"sequence_number": number, **event}
            yield f"event: {event['type']}\ndata: {json.dumps(payload)}\n\n".encode()
        if self.hold_open:
            self.waiting.set()
            await asyncio.Event().wait()

    async def aclose(self):
        self.closed = True


class _Endpoint:
    def __init__(self, monkeypatch):
        self.monkeypatch = monkeypatch
        self.requests: list[httpx.Request] = []
        self.clients: list[httpx.AsyncClient] = []
        self.model = MODEL
        self.status = 200
        self.json_body = None
        self.stream = _Stream(_text("Hello!"), _done())
        real_client = openai.AsyncOpenAI

        def client(**kwargs):
            http_client = httpx.AsyncClient(transport=httpx.MockTransport(self.handle))
            self.clients.append(http_client)
            return real_client(http_client=http_client, max_retries=0, **kwargs)

        monkeypatch.setattr(openai, "AsyncOpenAI", client)

    def handle(self, request):
        self.requests.append(request)
        if self.json_body is not None:
            return httpx.Response(self.status, json=self.json_body)
        return httpx.Response(
            self.status, headers={"content-type": "text/event-stream"}, stream=self.stream,
        )

    def configure(self, effort="low", model=MODEL):
        self.model = model
        settings = UserSettings(
            _env_file=None,
            planning_provider=Provider.OPENAI_COMPATIBLE,
            coding_provider=Provider.OPENAI_COMPATIBLE,
            router_provider=Provider.OPENAI_COMPATIBLE,
            planning_model="gpt-6.1-sol", coding_model="gpt-6.1-sol", router_model=model,
            router_reasoning_effort=effort,
            openai_compatible_api_key=SecretStr("test-router-key"),
            openai_base_url="https://router.example.com/v1",
        )
        self.monkeypatch.setattr(routing, "get_user_settings", lambda: settings)
        self.monkeypatch.setattr(
            "cowork.common.settings.user_settings.get_user_settings", lambda: settings,
        )

    def binding(self, effort="low", model=MODEL):
        self.configure(effort, model)
        binding = routing._settings_binding()
        assert binding is not None
        return binding

    def request_body(self, effort="low", *, tools=True, streaming=True):
        (request,) = self.requests
        assert str(request.url) == "https://router.example.com/v1/responses"
        assert request.headers["authorization"] == "Bearer test-router-key"
        body = json.loads(request.content)
        assert body["model"] == self.model
        assert body.get("stream", False) is streaming
        assert "reasoning_effort" not in body
        if effort:
            assert body["reasoning"]["effort"] == effort
        else:
            assert "reasoning" not in body
        if tools:
            assert len(body["tools"]) == 1
            assert body["tools"][0]["type"] == "function"
            assert body["tools"][0]["name"] == "delegate"
        else:
            assert "tools" not in body
        return body


@pytest.fixture
async def endpoint(monkeypatch, openai_compatible_api):
    openai_compatible_api("responses")
    # Wire-contract tests must not measure cold SDK imports on a busy runner.
    # The timeout test below supplies its own short deadline.
    monkeypatch.setattr(routing, "_GATE_FIRST_EVENT_SECONDS", 10.0)
    monkeypatch.setattr(routing, "_GATE_TOTAL_SECONDS", 15.0)
    endpoint = _Endpoint(monkeypatch)
    yield endpoint
    await asyncio.gather(*(client.aclose() for client in endpoint.clients))


async def _route(binding):
    return await routing.decide_route(
        binding=binding, history=HISTORY, has_non_text_input=False,
        has_attachments=False, has_disabled_connections=False,
    )


@pytest.mark.parametrize("effort", ["low", "none", None, ""])
@pytest.mark.parametrize("model", ["gpt-6-astra", "gpt-6.1-sol"])
async def test_configured_router_answers_over_responses(endpoint, effort, model):
    """The wire keeps the selected effort; the fake accepts every value here."""
    decision = await _route(endpoint.binding(effort, model))

    endpoint.request_body(effort)
    assert decision == routing.RouteDecision(
        route=routing.DIRECT_CONTEXT, reason="router_direct_response",
        provider="openai_compatible", model=model, text="Hello!",
    )
    assert endpoint.stream.closed


async def test_router_responses_preserves_prompt_history_and_tool_contract(endpoint):
    await _route(endpoint.binding())

    body = endpoint.request_body()
    assert body["instructions"] == routing.with_deployment_identity(routing._SYSTEM_PROMPT)
    assert body["input"] == [{"type": "message", "role": "user", "content": "Hello"}]
    assert body["max_output_tokens"] == 1024
    assert body["tools"] == [{
        "type": "function",
        "name": "delegate",
        "description": "Delegate when a request requires the full agent.",
        "parameters": {
            "type": "object",
            "properties": {"reason": {"type": "string"}},
            "required": ["reason"],
        },
        "strict": False,
    }]


async def test_responses_startup_event_does_not_extend_router_deadline(endpoint, monkeypatch):
    monkeypatch.setattr(routing, "_GATE_FIRST_EVENT_SECONDS", 0.05)
    endpoint.stream = _Stream({
        "type": "response.created", "response": _response("in_progress", usage=None),
    }, hold_open=True)

    decision = await asyncio.wait_for(_route(endpoint.binding()), timeout=0.5)

    endpoint.request_body()
    assert endpoint.stream.waiting.is_set()
    assert endpoint.stream.closed
    assert decision.reason == "router_timeout"
    assert decision.fallback is True
    assert decision.text == ""


async def test_router_delegation_closes_the_responses_stream(endpoint):
    endpoint.stream = _Stream({
        "type": "response.output_item.added", "output_index": 0,
        "item": {
            "type": "function_call", "id": "fc_router", "call_id": "call_router",
            "name": "delegate", "arguments": "", "status": "in_progress",
        },
    }, hold_open=True)

    decision = await _route(endpoint.binding())

    endpoint.request_body()
    assert decision.route == routing.DELEGATED_AGENTIC
    assert decision.reason == "router_declined_direct_response"
    assert decision.fallback is False
    assert decision.text == ""
    assert endpoint.stream.closed
    assert not endpoint.stream.waiting.is_set()


async def test_refused_none_is_not_retried_with_another_effort(endpoint, caplog):
    private = "private-provider-request-sentinel"
    endpoint.status = 400
    endpoint.json_body = {"error": {
        "message": f"Unsupported value 'none'. {private}",
        "type": "invalid_request_error", "param": "reasoning.effort", "code": "unsupported_value",
    }}
    with caplog.at_level(logging.WARNING, logger=routing.__name__):
        decision = await _route(endpoint.binding("none"))

    endpoint.request_body("none")
    assert decision.reason == "router_unavailable"
    assert decision.fallback is True
    assert decision.text == ""
    (record,) = [
        record for record in caplog.records
        if record.name == routing.__name__ and record.levelno == logging.WARNING
    ]
    assert (record.event, record.status_code) == ("router_call_failed", 400)
    assert record.exc_info is None
    assert private not in repr(record.__dict__)


@pytest.mark.parametrize("kind", ["response.failed", "error"])
async def test_in_band_responses_failure_discards_partial_gate_text(endpoint, kind):
    error = {"code": "server_error", "message": "The provider failed"}
    event = (
        {"type": kind, "response": _response("failed", error=error)}
        if kind == "response.failed"
        else {"type": kind, **error, "param": None}
    )
    endpoint.stream = _Stream(_text("Unfinished answer"), event)

    decision = await _route(endpoint.binding())

    endpoint.request_body()
    assert decision.reason == "router_unavailable"
    assert decision.fallback is True
    assert decision.text == ""
    assert endpoint.stream.closed


async def test_cancelling_the_gate_closes_the_responses_stream(endpoint):
    endpoint.stream = _Stream({
        "type": "response.created", "response": _response("in_progress", usage=None),
    }, hold_open=True)
    task = asyncio.create_task(_route(endpoint.binding()))
    try:
        await asyncio.wait_for(endpoint.stream.waiting.wait(), timeout=1)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    finally:
        if not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    endpoint.request_body()
    assert endpoint.stream.closed


async def test_first_take_uses_the_router_responses_transport_without_tools(endpoint):
    take = FirstTake(endpoint.binding(), HISTORY).start()
    await take._writer

    endpoint.request_body(tools=False)
    assert take._first_word_at is not None
    assert endpoint.stream.closed


async def test_history_summary_uses_the_router_responses_transport_without_tools(endpoint):
    endpoint.configure()
    endpoint.json_body = _response(output=[{
        "type": "message", "id": "msg_summary", "role": "assistant", "status": "completed",
        "content": [{"type": "output_text", "text": "Earlier discussion.", "annotations": []}],
    }])
    client = build_llm_client()

    response = await client.summarize(system="Summarize the conversation.", messages=HISTORY)

    endpoint.request_body(tools=False, streaming=False)
    assert response.content == "Earlier discussion."


class _ChatStream(_Stream):
    async def __aiter__(self):
        for event in self.events:
            yield f"data: {json.dumps(event)}\n\n".encode()
        yield b"data: [DONE]\n\n"


def _probe_output(api, *, delegate=False):
    if api == "chat_completions":
        delta = {"content": "Hello!"}
        if delegate:
            delta = {"tool_calls": [{
                "index": 0, "id": "call_probe", "type": "function",
                "function": {"name": "delegate", "arguments": '{"reason":"Needs tools"}'},
            }]}
        return _ChatStream({
            "id": "chat_probe", "object": "chat.completion.chunk", "created": 0,
            "model": MODEL, "choices": [{
                "index": 0, "delta": delta, "finish_reason": "tool_calls" if delegate else "stop",
            }],
        })
    if not delegate:
        return _Stream(_text("Hello!"), _done())
    return _Stream(
        {
            "type": "response.output_item.added", "output_index": 0,
            "item": {"type": "function_call", "id": "fc_probe", "call_id": "call_probe",
                     "name": "delegate", "arguments": "", "status": "in_progress"},
        },
        {
            "type": "response.function_call_arguments.done", "output_index": 0,
            "item_id": "fc_probe", "arguments": '{"reason":"Needs tools"}',
        },
        _done(),
    )


def _configure_startup_probe(endpoint, monkeypatch, *, effort):
    from cowork.services import router_probe

    endpoint.configure(effort)
    monkeypatch.setattr(router_probe, "run_db", AsyncMock(return_value=routing.get_user_settings()))
    return router_probe


@pytest.mark.parametrize("api,effort,delegate", [
    ("responses", "low", False), ("responses", None, True),
    ("chat_completions", "none", False), ("chat_completions", "low", True),
])
async def test_startup_probe_uses_real_configured_router(
    endpoint, monkeypatch, openai_compatible_api, caplog, api, effort, delegate,
):
    probe = _configure_startup_probe(endpoint, monkeypatch, effort=effort)
    openai_compatible_api(api)
    endpoint.stream = _probe_output(api, delegate=delegate)
    with caplog.at_level(logging.INFO, logger=probe.__name__):
        await probe.run_router_startup_probe()

    (request,) = endpoint.requests
    body = json.loads(request.content)
    assert body["model"] == MODEL
    assert body["stream"] is True
    assert "tool_choice" not in body
    assert "store" not in body  # The probe retains the adapter's existing retention policy.
    if api == "responses":
        endpoint.request_body(effort)
        assert body["instructions"] == routing.with_deployment_identity(routing._SYSTEM_PROMPT)
        assert body["input"] == [{"role": "user", "content": "Hello", "type": "message"}]
        assert body["max_output_tokens"] == routing._DIRECT_MAX_TOKENS
    else:
        assert request.url.path == "/v1/chat/completions"
        assert body.get("reasoning_effort") == effort
        assert body["messages"] == [
            {"role": "system", "content": routing.with_deployment_identity(routing._SYSTEM_PROMPT)},
            {"role": "user", "content": "Hello"},
        ]
        assert body["tools"][0]["function"]["name"] == "delegate"
        assert body["max_completion_tokens"] == routing._DIRECT_MAX_TOKENS
    assert any(
        r.name == probe.__name__ and r.levelno == logging.INFO
        and getattr(r, "event", None) == "router_startup_probe_passed"
        for r in caplog.records
    )
    assert endpoint.stream.closed
    assert all(client.is_closed for client in endpoint.clients)


@pytest.mark.parametrize("api", ["responses", "chat_completions"])
async def test_startup_refused_effort_does_not_substitute(endpoint, monkeypatch, openai_compatible_api, caplog, api):
    probe = _configure_startup_probe(endpoint, monkeypatch, effort="none")
    openai_compatible_api(api)
    private = "private-provider-refusal"
    endpoint.status = 400
    endpoint.json_body = {"error": {
        "message": private, "type": "invalid_request_error",
        "param": "reasoning.effort" if api == "responses" else "reasoning_effort",
        "code": "unsupported_value",
    }}
    await probe.run_router_startup_probe()

    (request,) = endpoint.requests
    body = json.loads(request.content)
    assert (body["reasoning"]["effort"] if api == "responses" else body["reasoning_effort"]) == "none"
    records = [r for r in caplog.records if r.name == probe.__name__]
    assert len(records) == 1
    assert records[0].levelno == logging.WARNING
    assert (records[0].event, records[0].reason) == ("router_startup_probe_unverified", "provider_refused")
    assert records[0].status_code == 400
    assert private not in repr([r.__dict__ for r in caplog.records])
    assert all(client.is_closed for client in endpoint.clients)


@pytest.mark.parametrize("kind", ["incomplete", "failed", "missing_terminal", "empty", "malformed_tool"])
async def test_startup_responses_does_not_pass_an_unfinished_or_invalid_response(endpoint, monkeypatch, caplog, kind):
    probe = _configure_startup_probe(endpoint, monkeypatch, effort="low")
    if kind == "incomplete":
        endpoint.stream = _Stream(_text("Partial"), {
            "type": "response.incomplete",
            "response": _response("incomplete", incomplete_details={"reason": "max_output_tokens"}),
        })
    elif kind == "failed":
        endpoint.stream = _Stream(_text("Partial"), {
            "type": "response.failed",
            "response": _response("failed", error={"code": "server_error", "message": "private-failure"}),
        })
    elif kind == "missing_terminal":
        endpoint.stream = _Stream(_text("Partial"))
    elif kind == "empty":
        endpoint.stream = _Stream(_done())
    else:
        endpoint.stream = _probe_output("responses", delegate=True)
        endpoint.stream.events[1]["arguments"] = '{"reason":"cut'
    await probe.run_router_startup_probe()

    records = [r for r in caplog.records if r.name == probe.__name__]
    assert len(records) == 1
    assert records[0].event == "router_startup_probe_unverified"
    assert records[0].levelno == logging.WARNING
    assert "private-failure" not in repr([r.__dict__ for r in caplog.records])
    assert endpoint.stream.closed
    assert all(client.is_closed for client in endpoint.clients)


async def test_startup_does_not_probe_coding_when_router_construction_fails(endpoint, monkeypatch, caplog):
    from cowork.services import router_probe

    settings = UserSettings(
        _env_file=None,
        planning_provider=Provider.OPENAI, coding_provider=Provider.OPENAI,
        router_provider=Provider.OPENAI_COMPATIBLE,
        planning_model=MODEL, coding_model=MODEL, router_model=MODEL,
        openai_api_key=SecretStr("test-direct-key"),
        openai_compatible_api_key=SecretStr("test-router-key"),
        openai_base_url="",
    )
    assert settings.resolved_router_provider is Provider.OPENAI_COMPATIBLE
    monkeypatch.setattr(router_probe, "run_db", AsyncMock(return_value=settings))
    with caplog.at_level(logging.INFO, logger=router_probe.__name__):
        await router_probe.run_router_startup_probe()

    assert endpoint.requests == []
    assert endpoint.clients  # Planning and coding clients still need cleanup.
    assert all(client.is_closed for client in endpoint.clients)
    records = [r for r in caplog.records if r.name == router_probe.__name__]
    assert len(records) == 1
    assert records[0].levelno == logging.WARNING
    assert (records[0].event, records[0].reason) == (
        "router_startup_probe_unverified", "configuration_unavailable"
    )
