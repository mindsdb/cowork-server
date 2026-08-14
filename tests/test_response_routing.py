from types import SimpleNamespace

import pytest

from cowork.handlers.response_routing import (
    DELEGATED_AGENTIC,
    DIRECT_CONTEXT,
    decide_route,
)
from cowork.handlers.responses import ResponsesHandler


class _Provider:
    value = "minds_cloud"


class _Client:
    def __init__(self, response):
        self.response = response
        self.calls = []

    async def gate(self, **kwargs):
        self.calls.append(kwargs)
        return self.response


def _response(*, content="", tool_calls=None, stop_reason="end_turn"):
    return SimpleNamespace(
        content=content,
        tool_calls=tool_calls or [],
        stop_reason=stop_reason,
    )


def test_handler_defers_anton_harness_initialization(monkeypatch):
    import cowork.handlers.responses as responses

    monkeypatch.setattr(
        responses,
        "get_user_settings",
        lambda scope: SimpleNamespace(harness="anton"),
    )
    monkeypatch.setattr(
        responses,
        "get_harness",
        lambda name: (_ for _ in ()).throw(AssertionError("Anton must be lazy")),
    )

    handler = ResponsesHandler(session=object())

    assert handler.harness is None


@pytest.mark.asyncio
async def test_text_context_routes_direct_with_resolved_router(monkeypatch):
    import cowork.handlers.response_routing as routing

    client = _Client(_response(content="The result was 42."))

    async def fake_gate(llm, *, history):
        assert llm is client
        return "The result was 42."

    monkeypatch.setattr(
        routing,
        "get_user_settings",
        lambda: SimpleNamespace(resolved_router_provider=_Provider(), resolved_router_model="router-model"),
    )
    monkeypatch.setattr(routing, "build_llm_client", lambda: client)
    monkeypatch.setattr(routing, "_gate", fake_gate)

    decision = await decide_route(
        history=[
            {"role": "user", "content": "What was the result?"},
            {"role": "assistant", "content": "The result was 42."},
            {"role": "user", "content": "Repeat it."},
        ],
        has_non_text_input=False,
        has_attachments=False,
        has_disabled_connections=False,
    )

    assert decision.route == DIRECT_CONTEXT
    assert decision.text == "The result was 42."
    assert decision.provider == "minds_cloud"
    assert decision.model == "router-model"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("has_non_text_input", "has_attachments", "has_disabled_connections", "reason"),
    [
        (True, False, False, "non_text_input"),
        (False, True, False, "attachments_present"),
        (False, False, True, "connection_context_present"),
    ],
)
async def test_ineligible_request_shapes_delegate_without_router_call(
    monkeypatch, has_non_text_input, has_attachments, has_disabled_connections, reason
):
    import cowork.handlers.response_routing as routing

    monkeypatch.setattr(routing, "build_llm_client", lambda: (_ for _ in ()).throw(AssertionError()))

    decision = await decide_route(
        history=[{"role": "user", "content": "Hello"}],
        has_non_text_input=has_non_text_input,
        has_attachments=has_attachments,
        has_disabled_connections=has_disabled_connections,
    )

    assert decision.route == DELEGATED_AGENTIC
    assert decision.reason == reason


@pytest.mark.asyncio
async def test_router_decline_delegates(monkeypatch):
    import cowork.handlers.response_routing as routing

    client = _Client(_response(content="DELEGATE"))
    monkeypatch.setattr(
        routing,
        "get_user_settings",
        lambda: SimpleNamespace(resolved_router_provider=_Provider(), resolved_router_model="router-model"),
    )
    monkeypatch.setattr(routing, "build_llm_client", lambda: client)
    monkeypatch.setattr(
        routing,
        "_gate",
        lambda *args, **kwargs: __import__("asyncio").sleep(0, result=None),
    )

    decision = await decide_route(
        history=[{"role": "user", "content": "Search the web for today's news."}],
        has_non_text_input=False,
        has_attachments=False,
        has_disabled_connections=False,
    )

    assert decision.route == DELEGATED_AGENTIC
    assert decision.reason == "router_declined_direct_response"


@pytest.mark.asyncio
async def test_router_error_fails_open_to_anton(monkeypatch):
    import cowork.handlers.response_routing as routing

    monkeypatch.setattr(
        routing,
        "get_user_settings",
        lambda: SimpleNamespace(resolved_router_provider=_Provider(), resolved_router_model="router-model"),
    )
    monkeypatch.setattr(routing, "build_llm_client", lambda: (_ for _ in ()).throw(RuntimeError("down")))

    decision = await decide_route(
        history=[{"role": "user", "content": "Hello"}],
        has_non_text_input=False,
        has_attachments=False,
        has_disabled_connections=False,
    )

    assert decision.route == DELEGATED_AGENTIC
    assert decision.reason == "router_unavailable"
    assert decision.fallback is True


@pytest.mark.asyncio
async def test_slow_gate_times_out_and_fails_open(monkeypatch):
    import asyncio

    import cowork.handlers.response_routing as routing

    monkeypatch.setattr(
        routing,
        "get_user_settings",
        lambda: SimpleNamespace(resolved_router_provider=_Provider(), resolved_router_model="router-model"),
    )
    monkeypatch.setattr(routing, "build_llm_client", lambda: object())

    async def hung_gate(llm, *, history):
        await asyncio.sleep(30)

    monkeypatch.setattr(routing, "_gate", hung_gate)
    monkeypatch.setattr(routing, "_GATE_TIMEOUT_SECONDS", 0.01)

    decision = await decide_route(
        history=[{"role": "user", "content": "Hello"}],
        has_non_text_input=False,
        has_attachments=False,
        has_disabled_connections=False,
    )

    assert decision.route == DELEGATED_AGENTIC
    assert decision.reason == "router_timeout"
    assert decision.fallback is True


def test_direct_sse_frames_use_real_newlines():
    frame = ResponsesHandler._direct_sse("response.created", {"type": "response.created"})

    assert frame.startswith("event: response.created\n")
    assert "\ndata: " in frame
    assert frame.endswith("\n\n")
    assert "\\n" not in frame  # regression: literal backslash-n broke SSE parsing


def _routing_handler(monkeypatch):
    import cowork.handlers.responses as responses

    monkeypatch.setattr(
        responses,
        "get_user_settings",
        lambda scope: SimpleNamespace(harness="anton"),
    )
    return ResponsesHandler(session=object())


@pytest.mark.asyncio
async def test_route_request_runs_gate_under_org_scope(monkeypatch):
    import cowork.handlers.responses as responses
    from cowork.common.settings.user_settings import _current_scope
    from cowork.handlers.response_routing import RouteDecision

    handler = _routing_handler(monkeypatch)
    sentinel_scope = object()
    handler.scope = sentinel_scope

    monkeypatch.setattr(
        responses,
        "ConversationService",
        lambda scoped: SimpleNamespace(get_ordered_messages=lambda _cid: []),
    )
    seen = {}

    async def fake_decide_route(**kwargs):
        seen["scope"] = _current_scope.get()
        return RouteDecision(route=DELEGATED_AGENTIC, reason="test")

    monkeypatch.setattr(responses, "decide_route", fake_decide_route)

    await handler._route_request(
        conversation_id=None,
        harness_input=[{"type": "text", "text": "Hello"}],
        has_attachments=False,
        has_disabled_connections=False,
    )

    assert seen["scope"] is sentinel_scope


@pytest.mark.asyncio
async def test_ineligible_route_skips_the_history_query(monkeypatch):
    import cowork.handlers.responses as responses

    handler = _routing_handler(monkeypatch)
    monkeypatch.setattr(
        responses,
        "ConversationService",
        lambda scoped: (_ for _ in ()).throw(AssertionError("must not touch the DB")),
    )

    decision = await handler._route_request(
        conversation_id=None,
        harness_input=[{"type": "text", "text": "Hello"}],
        has_attachments=True,
        has_disabled_connections=False,
    )

    assert decision.route == DELEGATED_AGENTIC
    assert decision.reason == "attachments_present"
